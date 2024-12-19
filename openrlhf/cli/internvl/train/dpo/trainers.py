from torch import nn
from torch.utils.data.sampler import Sampler, RandomSampler, SequentialSampler
import os
import math
import torch
# import wandb
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from trl import DPOTrainer
from trl.trainer.dpo_trainer import MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES
from transformers import Trainer
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Literal
from torch import Tensor
from torch.nn import Module
# from utils.utils import is_main_process
from transformers import PreTrainedModel
# from internvl.train.dpo.muffin_inference_logp import get_batch_logps, get_batch_logps_minicpm
from huggingface_hub.utils._deprecation import _deprecate_arguments
from contextlib import contextmanager, nullcontext
import warnings
from trl.trainer.utils import pad_to_length
from torch.utils.data import ConcatDataset
from tqdm import tqdm
import deepspeed
from trl.models import PreTrainedModelWrapper
from copy import deepcopy

from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from deepspeed.runtime.zero.partitioned_param_coordinator import PartitionedParameterCoordinator
from deepspeed.runtime.zero import partitioned_param_coordinator
from deepspeed.accelerator import get_accelerator
# def _map(self, *args, **kwargs):
#     return self

# def add_column(self, item):
#     self.reference_chosen_logps = item[0]
#     self.reference_rejected_logps = item[1]

# ConcatDataset.add_column = add_column
# ConcatDataset.map = _map


class MultimodalDPOTrainer(DPOTrainer):


    # def compute_reference_log_probs(self, padded_batch: Dict) -> Dict:
    #     """Computes log probabilities of the reference model for a single padded batch of a DPO specific dataset."""
    #     compte_ref_context_manager = amp.autocast("cuda") if self._peft_has_been_casted_to_bf16 else nullcontext()
    #     # import pdb; pdb.set_trace()
    #     # compute reference logps
    #     # with torch.no_grad(), compte_ref_context_manager:
    #     with torch.no_grad(), amp.autocast("cuda"):
    #         if self.ref_model is None:
    #             with self.null_ref_context():
    #                 reference_chosen_logps, reference_rejected_logps = self.concatenated_forward(
    #                     self.model, padded_batch
    #                 )[:2]
    #         else:
    #             reference_chosen_logps, reference_rejected_logps = self.concatenated_forward(
    #                 self.ref_model, padded_batch
    #             )[:2]

    #     return reference_chosen_logps, reference_rejected_logps

    def precompute_ref(self):
        
        dataloader_params = {
            "batch_size": self.args.per_device_train_batch_size,
            "collate_fn": self.data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "shuffle": False,
        }

        # prepare dataloader
        data_loader = self.accelerator.prepare(DataLoader(self.train_dataset, **dataloader_params))

        reference_chosen_logps = []
        reference_rejected_logps = []
        for padded_batch in tqdm(iterable=data_loader, desc="Train dataset reference log probs"):
            padded_batch['pixel_values'] = padded_batch['pixel_values'].to(torch.bfloat16)
            reference_chosen_logp, reference_rejected_logp = self.compute_reference_log_probs(padded_batch)
            reference_chosen_logp, reference_rejected_logp = self.accelerator.gather_for_metrics(
                (reference_chosen_logp, reference_rejected_logp)
            )
            reference_chosen_logps.append(reference_chosen_logp.cpu())
            reference_rejected_logps.append(reference_rejected_logp.cpu())

            # Unnecessary cache clearing to avoid OOM
            torch.cuda.empty_cache()
            self.accelerator.free_memory()
            
        # del self.ref_model
        all_reference_chosen_logps = torch.cat(reference_chosen_logps).float().numpy()
        all_reference_rejected_logps = torch.cat(reference_rejected_logps).float().numpy()
        # self.train_dataset.add_column(
        #     {
        #         'reference_chosen_logps': all_reference_chosen_logps,
        #         'reference_rejected_logps': all_reference_rejected_logps
        #     }
        # )
        # import pdb; pdb.set_trace()
        # for item, chosen, rejected in tqdm(zip(self.train_dataset, all_reference_chosen_logps, all_reference_rejected_logps)):
        #     item['reference_chosen_logps'] = chosen
        #     item['reference_rejected_logps'] = rejected
        
        self.train_dataset.add_column(
            name="reference_chosen_logps", column=all_reference_chosen_logps
        )
        self.train_dataset.add_column(
            name="reference_rejected_logps", column=all_reference_rejected_logps
        )
        del self.ref_model

    # def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
    #     if self.train_dataset is None or not has_length(self.train_dataset):
    #         return None

    #     # Build the sampler.
    #     # if self.args.group_by_length:
    #     #     if is_datasets_available() and isinstance(self.train_dataset, datasets.Dataset):
    #     #         lengths = (
    #     #             self.train_dataset[self.args.length_column_name]
    #     #             if self.args.length_column_name in self.train_dataset.column_names
    #     #             else None
    #     #         )
    #     #     else:
    #     #         lengths = None
    #     #     model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
    #     #     return LengthGroupedSampler(
    #     #         self.args.train_batch_size * self.args.gradient_accumulation_steps,
    #     #         dataset=self.train_dataset,
    #     #         lengths=lengths,
    #     #         model_input_name=model_input_name,
    #     #     )

    #     # else:
    #     return RandomSampler(self.train_dataset)
    
    @staticmethod
    def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool = False,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            label_pad_token_id: The label pad token id.
            is_encoder_decoder: Whether the model is an encoder-decoder model.

        Returns:
            A Tuple of two tensor of shape ((batch_size,), (batch_size,)) containing the sum of log probabilities of the given labels under the given logits in the first tensor and the number of non-masked tokens in the second tensor.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        return (per_token_logps * loss_mask).sum(-1), loss_mask.sum(-1)
        

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Subclass of transformers.src.transformers.trainer.get_train_dataloader to precompute `ref_log_probs`.
        """
        if self.precompute_ref_log_probs and not self._precomputed_train_ref_log_probs:
            self.precompute_ref()
            self._precomputed_train_ref_log_probs = True

        return super().get_train_dataloader()

    
    @staticmethod
    def concatenated_inputs(
        batch: Dict[str, Union[List, torch.LongTensor]],
        is_encoder_decoder: bool = False,
        is_vision_model: bool = False,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.LongTensor]:
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
            is_encoder_decoder: Whether the model is an encoder-decoder model.
            label_pad_token_id: The label pad token id.
            padding_value: The padding value to use for the concatenated inputs_ids.
            device: The device for the concatenated inputs.

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """
        concatenated_batch = {}

        if is_encoder_decoder:
            max_length = max(batch["chosen_labels"].shape[1], batch["rejected_labels"].shape[1])
        else:
            max_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])

        for k in batch:
            if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
        for k in batch:
            if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("rejected", "concatenated")
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                ).to(device=device)

        if is_encoder_decoder:
            concatenated_batch["concatenated_input_ids"] = batch["prompt_input_ids"].repeat(2, 1).to(device=device)
            concatenated_batch["concatenated_attention_mask"] = (
                batch["prompt_attention_mask"].repeat(2, 1).to(device=device)
            )

        if 'pixel_values' in batch:
            concatenated_batch['pixel_values'] = batch['pixel_values'].repeat(2, 1, 1, 1)
            concatenated_batch['image_flags'] = batch['image_flags'].repeat(2)

        return concatenated_batch

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.
    
        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            is_vision_model=self.is_vision_model,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )
        len_chosen = batch["chosen_labels"].shape[0]
        
        model_kwargs = {}

        if self.is_encoder_decoder:
            model_kwargs["labels"] = concatenated_batch["concatenated_labels"]
            model_kwargs["decoder_input_ids"] = concatenated_batch.pop("concatenated_decoder_input_ids", None)

        if self.is_vision_model:
            model_kwargs["pixel_values"] = concatenated_batch["pixel_values"]
            model_kwargs["pixel_attention_mask"] = concatenated_batch["pixel_attention_mask"]

        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True

        outputs = model(
            input_ids=concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            pixel_values=concatenated_batch['pixel_values'],
            image_flags=concatenated_batch['image_flags'],
            use_cache=False,
            **model_kwargs,
        )
        all_logits = outputs.logits

        all_logps, size_completion = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            # average_log_prob=self.loss_type == "ipo",
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        def cross_entropy_loss(logits, labels):
            if not self.is_encoder_decoder:
                # Shift so that tokens < n predict n
                logits = logits[..., :-1, :].contiguous()
                labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            logits = logits.view(-1, logits.shape[-1])
            labels = labels.view(-1)
            # Enable model parallelism
            labels = labels.to(logits.device)
            loss = loss_fct(logits, labels)
            return loss

        labels = concatenated_batch["concatenated_labels"].clone()
        nll_loss = cross_entropy_loss(all_logits[:len_chosen], labels[:len_chosen])

        if self.loss_type == "ipo":
            all_logps = all_logps / size_completion

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        if self.aux_loss_enabled:
            return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, nll_loss, outputs.aux_loss)

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, nll_loss)

    
    def _prepare_deepspeed(self, model: Optional[Union[PreTrainedModel, nn.Module, str]]):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

        if config_kwargs["zero_optimization"]["stage"] != 3 and isinstance(model, PreTrainedModel):
            for param in model.parameters():
                param.requires_grad = False
            print('using single gpu for reference model inference')
            model.eval()
            model = model.to(self.accelerator.device)
        elif config_kwargs["zero_optimization"]["stage"] == 3 and isinstance(model, PreTrainedModel):
            del model
            model = 'precompute_ref_logps'
        return model

        # if model is not None:
        #     if hasattr(model, "config"):
        #         hidden_size = (
        #             max(model.config.hidden_sizes)
        #             if getattr(model.config, "hidden_sizes", None)
        #             else getattr(model.config, "hidden_size", None)
        #         )
        #         if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
        #             # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
        #             # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
        #             config_kwargs.update(
        #                 {
        #                     "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
        #                     "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
        #                     "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
        #                 }
        #             )

        # # If ZeRO-3 is used, we shard both the active and reference model.
        # # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        # if config_kwargs["zero_optimization"]["stage"] != 3:
        #     config_kwargs["zero_optimization"]["stage"] = 0
        
        # config_kwargs.pop('')
        # config_kwargs.update(
        #     {
        #         "zero_optimization.offload_param": {
        #             "device": "cpu",  # Offloading parameters to CPU (optional)
        #             "pin_memory": True,
        #         }
        #     }
        # )
        # for param in model.parameters():
        #     param.requires_grad = False
        # for name, param in model.named_parameters():
        #     print(name, param.shape)
        # print(config_kwargs)
        # qwen language_model.model.embed_tokens.weight
        # deepspeed.zero.register_external_parameter(model, model.language_model.model.tok_embeddings.weight)
        # model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        # model.eval()
        
        # 假设你有一个已经初始化的模型和配置
        # prefetch_bucket_size = 1024 * 1024 * 2  # 2MB, 可以根据模型调整
        # max_reuse_distance_in_numel = 1024 * 1024 * 4  # 4M elements (调整为适合你模型的值)
        # max_available_parameters_in_numel = 1024 * 1024 * 8  # 8M elements (根据你的GPU内存设置)

        # 使用默认的CUDA stream
        # allgather_stream = get_accelerator().Stream

        # # 创建 `InflightParamRegistry`
        # inflight_param_registry = partitioned_param_coordinator.InflightParamRegistry()
        # for name, param in model.named_modules():
        #     if hasattr(param, "ds_summary"):
        #         # 打印 DeepSpeed 管理下的参数摘要
        #         print(f"Parameter {name} summary: {param.ds_summary()}")

        #         # 如果参数状态是 NOT_AVAILABLE，则尝试注册到 DeepSpeed
        #         if param.ds_status != ZeroParamStatus.AVAILABLE:
        #             print(f"Parameter {name} is not available, attempting to gather...")
        #             deepspeed.zero.register_external_parameter(model, param)

        #             # 尝试使用 DeepSpeed 的机制强制 gather 参数
        #             param_coordinator = PartitionedParameterCoordinator(
        #                 prefetch_bucket_sz=config_kwargs['zero_optimization']['stage3_prefetch_bucket_size'],
        #                 max_reuse_distance_in_numel=config_kwargs['zero_optimization']['stage3_max_reuse_distance'],
        #                 max_available_parameters_in_numel=config_kwargs['zero_optimization']['stage3_max_live_parameters'],
        #                 allgather_stream=allgather_stream,
        #                 inflight_param_registry=inflight_param_registry
        #             )
        #             param_coordinator.fetch_sub_module(param, forward=True)

    
        # # 手动加载所有参数（确保它们处于可用状态）
        # param_coordinator = PartitionedParameterCoordinator(
        #     prefetch_bucket_sz=config_kwargs['zero_optimization']['stage3_prefetch_bucket_size'],
        #     max_reuse_distance_in_numel=config_kwargs['zero_optimization']['stage3_max_reuse_distance'],
        #     max_available_parameters_in_numel='',
        #     allgather_stream='',
        #     inflight_param_registry=''
        # )

        # # 强制加载所有未加载的参数
        # for name, param in model.named_parameters():
        #     if hasattr(param, 'ds_status') and param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
        #         param_coordinator.fetch_sub_module(param, forward=True)
                
        # return model


    # def _prepare_deepspeed(self, model):
    #     deepspeed_plugin = self.accelerator.state.deepspeed_plugin
    #     config_kwargs = deepspeed_plugin.deepspeed_config
    #     if config_kwargs["zero_optimization"]["stage"] == 3:
    #         print('Enable DPOTrainer._prepare_deepspeed')
    #         return super()._prepare_deepspeed(model)

    #     print('Disable DPOTrainer._prepare_deepspeed')
    #     for param in model.parameters():
    #         param.requires_grad = False

    #     model.eval()
    #     model = model.to(self.accelerator.device)
    #     return model

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        forward_output = self.concatenated_forward(model, batch)
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_nll_loss,
        ) = forward_output[:5]
        if self.aux_loss_enabled:
            aux_loss = forward_output[5]

        # if reference_chosen_logps and reference_rejected_logps in batch use them, otherwise use the reference model
        # import pdb;pdb.set_trace()
        if (
            "reference_chosen_logps" in batch
            and "reference_rejected_logps" in batch
            # and (self.precompute_ref_log_probs or self.args.rpo_alpha is not None)
        ):
            # print('----------------use precompute_ref_log_probs--------------------')
            reference_chosen_logps = batch["reference_chosen_logps"]
            reference_rejected_logps = batch["reference_rejected_logps"]
        else:
            # self.ref_model.eval()
            # for name, param in model.named_parameters():
            #     print(name, param.shape)
            with torch.no_grad():
                if self.ref_model is None:
                    with self.null_ref_context():
                        (
                            reference_chosen_logps,
                            reference_rejected_logps,
                            _,
                            _,
                            _,
                        ) = self.concatenated_forward(self.model, batch)
                else:
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                        _,
                        _,
                    ) = self.concatenated_forward(self.ref_model, batch)

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        if self.args.rpo_alpha is not None:
            # losses = losses * self.args.rpo_alpha + policy_nll_loss
            losses = losses + policy_nll_loss * self.args.rpo_alpha

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().cpu()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().mean().cpu()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().mean().cpu()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().mean().cpu()
        if self.args.rpo_alpha is not None:
            metrics[f"{prefix}nll_loss"] = policy_nll_loss.detach().mean().cpu()

        if self.aux_loss_enabled:
            return losses.mean() + getattr(model.config, "router_aux_loss_coef", 0.0) * aux_loss, metrics

        return losses.mean(), metrics




class DPOTrainerSelf(DPOTrainer):
    
    # MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES['internvl_chat'] = 'internvl_chat'
    
    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not self.use_dpo_data_collator:
            warnings.warn(
                "compute_loss is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )
        self.is_vision_model = True
        compute_loss_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext

        with compute_loss_context_manager():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")

        # Make sure to move the loss to the device the original accumulating loss is at back in the `Trainer` class:
        loss = loss.to(self.args.device)
        # force log the metrics
        self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        # self.is_vision_model = True
        # is_vision_model = True
        # self.set_is_vision_model()
        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=False,
            is_vision_model=self.is_vision_model,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )
        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs = {}

        if self.is_encoder_decoder:
            model_kwargs["labels"] = concatenated_batch["concatenated_labels"]
            model_kwargs["decoder_input_ids"] = concatenated_batch.pop("concatenated_decoder_input_ids", None)

        # if self.is_vision_model:
        #     model_kwargs["pixel_values"] = concatenated_batch["pixel_values"]
            # model_kwargs["pixel_attention_mask"] = concatenated_batch["pixel_attention_mask"]

        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True

        outputs = model(
            concatenated_batch["pixel_values"],
            image_flags=concatenated_batch['image_flags'],
            input_ids=concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
            **model_kwargs,
        )
        all_logits = outputs.logits

        all_logps, size_completion = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            # average_log_prob=self.loss_type == "ipo",
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        def cross_entropy_loss(logits, labels):
            if not self.is_encoder_decoder:
                # Shift so that tokens < n predict n
                logits = logits[..., :-1, :].contiguous()
                labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            logits = logits.view(-1, logits.shape[-1])
            labels = labels.view(-1)
            # Enable model parallelism
            labels = labels.to(logits.device)
            loss = loss_fct(logits, labels)
            return loss

        labels = concatenated_batch["concatenated_labels"].clone()
        nll_loss = cross_entropy_loss(all_logits[:len_chosen], labels[:len_chosen])

        if self.loss_type == "ipo":
            all_logps = all_logps / size_completion

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        if self.aux_loss_enabled:
            return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, nll_loss, outputs.aux_loss)

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, nll_loss)
    
    @staticmethod
    def concatenated_inputs(
        batch: Dict[str, Union[List, torch.LongTensor]],
        is_encoder_decoder: bool = False,
        is_vision_model: bool = True,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.LongTensor]:
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
            is_encoder_decoder: Whether the model is an encoder-decoder model.
            label_pad_token_id: The label pad token id.
            padding_value: The padding value to use for the concatenated inputs_ids.
            device: The device for the concatenated inputs.

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """
        concatenated_batch = {}

        if is_encoder_decoder:
            max_length = max(batch["chosen_labels"].shape[1], batch["rejected_labels"].shape[1])
        else:
            max_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])

        for k in batch:
            if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
        for k in batch:
            if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("rejected", "concatenated")
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                ).to(device=device)

        if is_encoder_decoder:
            concatenated_batch["concatenated_input_ids"] = batch["prompt_input_ids"].repeat(2, 1).to(device=device)
            concatenated_batch["concatenated_attention_mask"] = (
                batch["prompt_attention_mask"].repeat(2, 1).to(device=device)
            )

        if is_vision_model:
            concatenated_batch["pixel_values"] = batch["prompt_pixel_values"].repeat(2, 1, 1, 1, 1).flatten(start_dim=0, end_dim=1).to(device=device)
            # import pdb;pdb.set_trace()
            # print(type(batch['image_flags']))
            # print()
            # image_flags = [torch.tensor(flags).to(device=device) for flags in batch['prompt_image_flags']]
            concatenated_batch['image_flags'] = batch['prompt_image_flags'][0].repeat(2).flatten().to(device=device)
            # concatenated_batch['image_flags'] = batch['image_flags'][0].to(device=device)
            # concatenated_batch["pixel_attention_mask"] = (
            #     batch["prompt_pixel_attention_mask"].repeat(2, 1, 1, 1).to(device=device)
            # )
        return concatenated_batch


    # def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
    #     if self.train_dataset is None:
    #         return None

    #     # Build the sampler.
    #     # return RandomSampler(self.train_dataset)
    #     return SequentialSampler(self.train_dataset)

# if __name__ == "__main__":
    # print(DPOTrainerSelf.is_vision_model)
    # trainer = DPOTrainerSelf(
    #     model,
    #     ref_model=None, # not needed when using peft
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     tokenizer=tokenizer,
    #     dataset_num_proc=32,
    #     # peft_config=LoraConfig(target_modules="all-linear"),
    # )