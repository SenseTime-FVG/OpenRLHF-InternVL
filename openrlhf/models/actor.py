from typing import Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, set_seed
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from .ring_attn_utils import convert_ring_attn_params
from .utils import log_probs_from_logits, reset_position_ids
import sys
# sys.path.append("/mnt/afs/wangjiahao/workspace/OpenRLHF/openrlhf/cli/internvl")
# sys.path.append("/mnt/afs/wangjiahao/workspace/OpenRLHF/openrlhf/cli")
sys.path.append("../")
sys.path.append("../../")
from ..internvl.dist_utils import init_dist
from ..internvl.model import *
from ..internvl.patch import (concat_pad_data_collator, packed_pad_data_collator,
                            replace_llama_rmsnorm_with_fused_rmsnorm,
                            replace_train_sampler, replace_internlm2_attention_class,
                            replace_qwen2_attention_class, dpo_concat_pad_data_collator,
                            kto_concat_pad_data_collator)
from ..internvl.train.constants import (BOX_END_TOKEN, BOX_START_TOKEN,
                                      IMG_CONTEXT_TOKEN, IMG_END_TOKEN,
                                      IMG_START_TOKEN, QUAD_END_TOKEN,
                                      QUAD_START_TOKEN, REF_END_TOKEN,
                                      REF_START_TOKEN)
from ..internvl.train.dataset import (TCSLoader, build_datasets)
from ..internvl.train.trainer_monkey_patch import replace_create_optimizer, add_conine_with_min_lr_scheduler

class Actor(nn.Module):
    """
    Base class for Actor models in reinforcement learning.

    This class serves as a foundation for implementing various actor models, which are responsible for selecting actions based on the policy learned from the environment.

    Args:
        pretrain_or_model (nn.Module): A pretrained model or a new model instance to be used as the actor.
        use_flash_attention_2 (bool, optional): Whether to utilize Flash Attention 2.0 for improved performance. Defaults to False.
        bf16 (bool, optional): Enable bfloat16 precision for model computations. Defaults to True.
        load_in_4bit (bool, optional): Load the model in 4-bit precision. Defaults to False.
        lora_rank (int, optional): Rank for LoRA adaptation. Defaults to 0.
        lora_alpha (int, optional): Alpha parameter for LoRA. Defaults to 16.
        lora_dropout (float, optional): Dropout rate for LoRA layers. Defaults to 0.
        target_modules (list, optional): List of target modules for applying LoRA. Defaults to None.
        ds_config (dict, optional): Configuration for DeepSpeed, enabling model partitioning across multiple GPUs. Defaults to None.
        device_map (dict, optional): Device mapping for loading the model onto specific devices. Defaults to None.
        packing_samples (bool, optional): Whether to pack samples during training. Defaults to False.
    """

    def __init__(
        self,
        pretrain_or_model,
        use_flash_attention_2=False,
        bf16=True,
        load_in_4bit=False,
        lora_rank=0,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=None,
        ds_config=None,
        device_map=None,
        packing_samples=False,
        internvl=False,
        args=None,
        **kwargs
    ) -> None:
        super().__init__()

        if isinstance(pretrain_or_model, str):
            attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"

            # Note: dschf is defined in function scope to avoid global effects
            # https://huggingface.co/docs/transformers/deepspeed#non-trainer-deepspeed-integration
            if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
                dschf = HfDeepSpeedConfig(ds_config)
            else:
                dschf = None

            if load_in_4bit:
                assert bf16, "we only support bnb_4bit_compute_dtype = bf16"
                nf4_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            else:
                nf4_config = None
            if internvl:
                #logger.info('Loading InternVLChatModel...')
                print('Loading InternVLChatModel...')
                # Set seed for torch dataloaders.
                set_seed(42)
                config = InternVLChatConfig.from_pretrained(pretrain_or_model)
                config.vision_config.drop_path_rate = 0
                if config.llm_config.model_type == 'internlm2':
                    config.llm_config.attn_implementation = 'flash_attention_2'  # for InternLM
                    #logger.info('Using flash_attention_2 for InternLM')
                    print('Using flash_attention_2 for InternLM')
                else:
                    config.llm_config._attn_implementation = 'flash_attention_2'  # for LLaMA
                    #logger.info('Using flash_attention_2 for LLaMA')
                    print('Using flash_attention_2 for LLaMA')
                #要改成传进来的参数
                # config.template = data_args.conv_style
                # config.select_layer = model_args.vision_select_layer
                # config.dynamic_image_size = data_args.dynamic_image_size
                # config.use_thumbnail = data_args.use_thumbnail
                # config.ps_version = model_args.ps_version
                # config.min_dynamic_patch = data_args.min_dynamic_patch
                # config.max_dynamic_patch = data_args.max_dynamic_patch
                config.template = "internlm2-chat-v3"
                config.select_layer = args.vision_select_layer
                config.dynamic_image_size = not(args.not_dynamic_image_size)
                config.use_thumbnail = not(args.not_use_thumbnail)
                config.ps_version = args.ps_version
                config.min_dynamic_patch = args.min_dynamic_patch
                config.max_dynamic_patch = args.max_dynamic_patch
                self.model = InternVLChatModel.from_pretrained(
                    pretrain_or_model, torch_dtype=torch.bfloat16, config=config)
                self.model.img_context_token_id = args.img_context_token_id
                self.model.neftune_alpha = args.neftune_alpha
                self.model.config.force_image_size = args.force_image_size
                patch_size = self.model.config.vision_config.patch_size
                self.model.num_image_token = int((args.force_image_size // patch_size) ** 2 * (args.down_sample_ratio ** 2))
                self.model.language_model.config.use_cache = False
                self.model.vision_model.gradient_checkpointing = True
                self.model.vision_model.encoder.gradient_checkpointing = True
                self.model.language_model._set_gradient_checkpointing()
                #self.model.vision_model.gradient_checkpointing = True
                #self.model.vision_model.encoder.gradient_checkpointing = True
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    pretrain_or_model,
                    trust_remote_code=True,
                    attn_implementation=attn_implementation,
                    quantization_config=nf4_config,
                    torch_dtype=torch.bfloat16 if bf16 else "auto",
                    device_map=device_map,
                )

            # LoRA
            if lora_rank > 0:
                # https://github.com/huggingface/peft/issues/137
                self.model.enable_input_require_grads()
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    target_modules=target_modules,
                    lora_dropout=lora_dropout,
                    bias="none",
                )
                self.model = get_peft_model(self.model, lora_config)

                if load_in_4bit:
                    for name, module in self.model.named_modules():
                        if isinstance(module, LoraLayer):
                            module = module.to(torch.bfloat16)
                        if "norm" in name:
                            module = module.to(torch.float32)
                        if "lm_head" in name or "embed_tokens" in name:
                            if hasattr(module, "weight"):
                                module = module.to(torch.bfloat16)

            # MoE - balancing loss
            model_config = self.model.config.to_dict()
            if "output_router_logits" in model_config:
                print("[MoE] set output_router_logits as True")
                self.model.config.output_router_logits = True

            # https://github.com/huggingface/transformers/issues/26877
            # Use `model.generate(use_cache=True)` instead.`
            self.model.config.use_cache = False

            # packing samples using Flash Attention 2
            self.packing_samples = packing_samples
        else:
            self.model = pretrain_or_model

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, **kwargs) -> Union[
        Tuple[torch.LongTensor, torch.LongTensor],
        Tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor],
    ]:
        generate_args = {
            "input_ids": input_ids,
            "top_k": kwargs.get("top_k", None),
            "top_p": kwargs.get("top_p", None),
            "do_sample": kwargs.get("do_sample", True),
            "early_stopping": True,
            "temperature": kwargs.get("temperature", 1),
            "use_cache": True,
            "num_beams": kwargs.get("num_beams", 1),
            "attention_mask": kwargs.get("attention_mask"),
            "eos_token_id": kwargs.get("eos_token_id"),
            "pad_token_id": kwargs.get("pad_token_id"),
            "min_new_tokens": kwargs.get("min_new_tokens", 1),
        }

        if kwargs.get("max_new_tokens", None):
            generate_args["max_new_tokens"] = kwargs.get("max_new_tokens")
        if kwargs.get("max_length", None):
            generate_args["max_length"] = kwargs.get("max_length")

        # Call generate
        sequences = self.model.generate(**generate_args)

        # Prepare mask tensor
        eos_token_id = generate_args["eos_token_id"]
        pad_token_id = generate_args["pad_token_id"]

        return self.process_sequences(sequences, input_ids.size(1), eos_token_id, pad_token_id)

    def process_sequences(self, sequences: torch.Tensor, input_len, eos_token_id, pad_token_id):
        attention_mask = (sequences.ne(eos_token_id) & sequences.ne(pad_token_id)).to(dtype=torch.long)
        seq_length = attention_mask.size(1)

        # The following code is equivalent to:
        #
        # for i in range(attention_mask.size(0)):
        #     for t in reversed(range(seq_length)):
        #         if attention_mask[i][t] > 0.5:
        #             attention_mask[i][min(t + 1, seq_length - 1)] = True
        #             sequences[i][min(t + 1, seq_length - 1)] = eos_token_id
        #             break
        #
        eos_indices = seq_length - attention_mask.long().fliplr().argmax(dim=1, keepdim=True).clamp(min=1)
        sequences.scatter_(dim=1, index=eos_indices, value=eos_token_id)

        # For Llama3 and Qwen2 models, there are some eos_tokens in the middle of the prompt.
        first_token_indices = attention_mask.long().argmax(dim=1, keepdim=True)
        mask = torch.arange(seq_length).unsqueeze(0).expand(sequences.size(0), -1).to(device=sequences.device)
        attention_mask = (mask >= first_token_indices) & (mask <= eos_indices).to(dtype=torch.long)

        # in RL, state_i (current token) + action_i (next token) -> state_i+1 (next token)
        state_seq = sequences[:, input_len - 1 : -1]
        action_mask = state_seq.ne(eos_token_id) & state_seq.ne(pad_token_id)
        action_mask[:, 0] = 1

        return sequences, attention_mask, action_mask

    def forward(
        self,
        sequences: torch.LongTensor,
        num_actions: Optional[Union[int, list[int]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_output=False,
        ring_attn_group: Optional[dist.ProcessGroup] = None,
        packed_seq_lens: Optional[list[int]] = None,
    ) -> torch.Tensor:
        """Returns action log probs"""
        if not self.packing_samples:
            # https://github.com/OpenRLHF/OpenRLHF/issues/217
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
        else:
            # convert attention_mask to position_ids
            if ring_attn_group is not None:
                sequences, attention_mask, position_ids = convert_ring_attn_params(
                    sequences, attention_mask, packed_seq_lens, ring_attn_group
                )
            else:
                position_ids = reset_position_ids(attention_mask)
            # explicitly ignore attention_mask for packing_samples
            attention_mask = None

        output = self.model(sequences, attention_mask=attention_mask, position_ids=position_ids)

        if num_actions is None:
            assert return_output
            return output

        log_probs = log_probs_from_logits(output["logits"][:, :-1, :], sequences[:, 1:])

        if not self.packing_samples:
            action_log_probs = log_probs[:, -num_actions:]
        else:
            assert isinstance(num_actions, list) and len(num_actions) == len(packed_seq_lens)
            action_log_probs = []
            offset = 0
            for num_action, seq_len in zip(num_actions, packed_seq_lens):
                start, end = max(0, offset + seq_len - num_action - 1), offset + seq_len - 1
                action_log_probs.append(log_probs[:, start:end])
                offset += seq_len
            action_log_probs = torch.cat(action_log_probs, dim=1)

        if return_output:
            return (action_log_probs, output)
        else:
            return action_log_probs

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs={"use_reentrant": False}):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()

    def print_trainable_parameters(self):
        self.model.print_trainable_parameters()
