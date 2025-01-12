# --------------------------------------------------------
# InternVL
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import warnings
from typing import Any, List, Optional, Tuple, Union
import re

import torch.utils.checkpoint
import torch.distributed as dist
# from internvl.model.internlm2.modeling_internlm2 import InternLM2ForCausalLM
# from internvl.model.phi3.modeling_phi3 import Phi3ForCausalLM
from ..internlm2.modeling_internlm2 import InternLM2ForCausalLM
from ..phi3.modeling_phi3 import Phi3ForCausalLM
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,
                          LlamaTokenizer, Qwen2ForCausalLM)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging

from .configuration_internvl_chat import InternVLChatConfig
from .modeling_intern_vit import InternVisionModel

logger = logging.get_logger(__name__)


class InternVLChatModel(PreTrainedModel):
    config_class = InternVLChatConfig
    main_input_name = 'pixel_values'
    _no_split_modules = ['InternVision', 'LlamaDecoderLayer', 'InternLM2DecoderLayer',
                         'Phi3DecoderLayer', 'Qwen2DecoderLayer']

    def __init__(self, config: InternVLChatConfig, vision_model=None, language_model=None):
        super().__init__(config)

        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version

        logger.info(f'num_image_token: {self.num_image_token}')
        logger.info(f'ps_version: {self.ps_version}')
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = InternVisionModel(config.vision_config)
        if language_model is not None:
            self.language_model = language_model
        else:
            if config.llm_config.architectures[0] == 'LlamaForCausalLM':
                self.language_model = LlamaForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'InternLM2ForCausalLM':
                self.language_model = InternLM2ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Phi3ForCausalLM':
                self.language_model = Phi3ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Qwen2ForCausalLM':
                self.language_model = Qwen2ForCausalLM(config.llm_config)
            else:
                raise NotImplementedError(f'{config.llm_config.architectures[0]} is not implemented.')

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )

        # if config.force_image_size != config.vision_config.image_size:
        #     self.vision_model.resize_pos_embeddings(
        #         old_size=config.vision_config.image_size,
        #         new_size=config.force_image_size,
        #         patch_size=config.vision_config.patch_size
        #     )

        self.img_context_token_id = None
        self.neftune_alpha = None

        if config.use_backbone_lora:
            self.wrap_backbone_lora(r=config.use_backbone_lora, lora_alpha=2 * config.use_backbone_lora)

        if config.use_llm_lora:
            self.wrap_llm_lora(r=config.use_llm_lora, lora_alpha=2 * config.use_llm_lora)

    def wrap_backbone_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        lora_config = LoraConfig(
            r=r,
            target_modules=['attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.vision_model = get_peft_model(self.vision_model, lora_config)
        self.vision_model.print_trainable_parameters()

    def wrap_llm_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        lora_config = LoraConfig(
            r=r,
            target_modules=['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                            'mlp.gate_proj', 'mlp.down_proj', 'mlp.up_proj'],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            task_type='CAUSAL_LM'
        )
        self.language_model = get_peft_model(self.language_model, lora_config)
        self.language_model.enable_input_require_grads()
        self.language_model.print_trainable_parameters()

    def forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cu_seqlens: Optional[torch.IntTensor] = None,
            loss_weight = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

        vit_embeds = self.extract_feature(pixel_values)
        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
            ignore_flag = False
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                  f'vit_embeds.shape={vit_embeds.shape}')
            n_token = min(selected.sum(), vit_embeds.shape[0])
            input_embeds[:n_token] = input_embeds[:n_token] * 0.0 + vit_embeds[:n_token]
            ignore_flag = True

        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask if cu_seqlens is None else cu_seqlens,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            if cu_seqlens is None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(shift_logits, shift_labels)
                if ignore_flag:
                    loss = loss * 0.0
            else:
                assert labels.shape[0] == 1 and loss_weight is not None
                loss_fct = CrossEntropyLoss(reduction='none')
                loss = loss_fct(shift_logits, shift_labels)
                shift_weights = torch.FloatTensor(loss_weight[0][1:]).contiguous().to(shift_logits.device)
                shift_weights_sum = shift_weights.sum()
                dist.all_reduce(shift_weights_sum, op=dist.ReduceOp.AVG)
                # import pdb 
                # pdb.set_trace()
                loss = loss * shift_weights
                if shift_weights_sum < 1:
                    print(f'warning: shift_weights_sum too small: {shift_weights_sum}')
                loss = loss.sum() / (shift_weights_sum + 1e-6)  # for very extramly case that shift_weights_sum=0, which should never happen
                if ignore_flag:
                    loss = loss * 0.0

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def noised_embed(self, vit_embeds, noise_alpha=5):
        dims = torch.tensor(vit_embeds.size(1) * vit_embeds.size(2))
        mag_norm = noise_alpha / torch.sqrt(dims)
        noise = torch.zeros_like(vit_embeds).uniform_(-mag_norm, mag_norm)
        return vit_embeds + noise

    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        if self.training and self.neftune_alpha is not None:
            vit_embeds = self.noised_embed(vit_embeds, self.neftune_alpha)

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)#.to(pixel_values.device)
        return vit_embeds

    def batch_chat(self, tokenizer, pixel_values, image_counts, questions, generation_config, history=None,
                         return_history=False, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>',
                         IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'):
        if history is not None or return_history:
            print('Now multi-turn chat is not supported in batch_chat.')
            raise NotImplementedError
        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        from .conversation import get_conv_template

        queries = []
        image_bs = pixel_values.shape[0]
        # print(f'dynamic ViT batch size: {image_bs}, image_counts: {image_counts}')
        for idx, image_count in enumerate(image_counts):
            image_token = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * image_count + IMG_END_TOKEN
            question = image_token + '\n' + questions[idx]
            template = get_conv_template(self.template)
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()
            queries.append(query)
        tokenizer.padding_side = 'left'
        model_inputs = tokenizer(queries, return_tensors='pt', padding=True)
        input_ids = model_inputs['input_ids'].cuda()
        attention_mask = model_inputs['attention_mask'].cuda()
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)
        generation_config['eos_token_id'] = eos_token_id

        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        responses = [response.split(template.sep)[0].strip() for response in responses]
        return responses

    def chat(self, tokenizer, pixel_values, question, generation_config, history=None, return_history=False,
             IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'):

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        from internvl.conversation import get_conv_template

        template = get_conv_template(self.template)
        # also add end-of-assistant token in eos token id to avoid unnecessary generation
        eos_token_id = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids(["<|im_end|>"])[0]]

        image_bs = pixel_values.shape[0]
        print(f'dynamic ViT batch size: {image_bs}')
        if history is None:
            history = []
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * image_bs + IMG_END_TOKEN
            question = image_tokens + '\n' + question
        else:
            for (old_question, old_answer) in history:
                template.append_message(template.roles[0], old_question)
                template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()
        if self.template in ['internlm2-chat']:
            model_inputs = tokenizer(query, return_tensors='pt')
        else:
            model_inputs = tokenizer(query, return_tensors='pt', add_special_tokens=False)
        input_ids = model_inputs['input_ids'].cuda()
        attention_mask = model_inputs['attention_mask'].cuda()
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split("<|im_end|>")[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(image_tokens, '<image>')
            print(query_to_print, response)
            return response
        return response
    
    def chat_with_conv(self, tokenizer, pixel_values, dynamic_nums, conv, generation_config, history=None, return_history=False,
             IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'):

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        # also add end-of-assistant token in eos token id to avoid unnecessary generation
        eos_token_id = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids(["<|im_end|>"])[0]]

        query = conv.get_prompt()

        if pixel_values is not None:
            print(f'image num: {len(dynamic_nums)}')
            matches = re.findall(r"<image>", query)
            assert len(matches) == len(dynamic_nums), '<image> error' + query
            image_tokens_list = list()
            total_dynamic_num = 0
            for index in range(len(dynamic_nums)):
                dynamic_num = dynamic_nums[index]
                total_dynamic_num += dynamic_num
                print(f'dynamic ViT batch size: {dynamic_num}')
                image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * dynamic_num + IMG_END_TOKEN
                image_tokens_list.append(image_tokens)
            assert total_dynamic_num == pixel_values.shape[0], f'dynamic num not equal, {total_dynamic_num}, {pixel_values.shape[0]}'

            image_tokens_iter = iter(image_tokens_list)

            query = re.sub(r"<image>", lambda match:next(image_tokens_iter), query)

        if self.template in ['internlm2-chat']:
            model_inputs = tokenizer(query, return_tensors='pt')
        else:
            model_inputs = tokenizer(query, return_tensors='pt', add_special_tokens=False)
        input_ids = model_inputs['input_ids'].cuda()
        attention_mask = model_inputs['attention_mask'].cuda()
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split("<|im_end|>")[0].strip()
        if pixel_values is not None:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
        else:
            query_to_print = query
        print(query_to_print, response)
        return response

    def multi_image_chat(self, tokenizer, pixel_values, image_counts, question, generation_config, history=None,
                         return_history=False, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'):

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        from internvl.conversation import get_conv_template

        template = get_conv_template(self.template)
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)

        if history is None:
            history = []
            image_tokens = ''
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}, image_counts: {image_counts}')
            for idx, image_count in enumerate(image_counts):
                image_tokens += f'<image {idx+1}> (图{idx+1}):' + IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * image_count + IMG_END_TOKEN
            question = image_tokens + '\n' + question
        else:
            for (old_question, old_answer) in history:
                template.append_message(template.roles[0], old_question)
                template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()
        model_inputs = tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].cuda()
        attention_mask = model_inputs['attention_mask'].cuda()
        generation_config['eos_token_id'] = eos_token_id

        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep)[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(image_tokens, '<image>')
            print(query_to_print, response)
            return response
        return response

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.extract_feature(pixel_values)

            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs
