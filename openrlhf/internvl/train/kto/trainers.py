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
from trl import DPOTrainer, KTOTrainer
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

def _map(self, *args, **kwargs):
    return self

def _shuffle(self, seed):
    return self
    
ConcatDataset.map = _map
ConcatDataset.shuffle = _shuffle


class MultiModalKTOTrainer(KTOTrainer):
    
    

    
    def forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        if self.calculate_KL:
            KL_logps = None
            KL_model_kwargs = (
                {
                    "input_ids": batch["KL_prompt_input_ids"],
                    "attention_mask": batch["KL_prompt_attention_mask"],
                    "labels": batch["KL_completion_labels"],
                    "decoder_input_ids": batch.get("KL_completion_decoder_input_ids"),
                }
                if self.is_encoder_decoder
                else {
                    "input_ids": batch["KL_completion_input_ids"],
                    "attention_mask": batch["KL_completion_attention_mask"],
                }
            )
            with torch.no_grad():
                KL_logits = model(
                    **KL_model_kwargs,
                ).logits

            KL_logps = self.get_batch_logps(
                KL_logits,
                batch["KL_completion_labels"],
                average_log_prob=False,
                is_encoder_decoder=self.is_encoder_decoder,
                label_pad_token_id=self.label_pad_token_id,
            )
        else:
            KL_logps = None

        model_kwargs = (
            {
                "labels": batch["completion_labels"],
                "decoder_input_ids": batch.get("completion_decoder_input_ids"),
            }
            if self.is_encoder_decoder
            else {}
        )
        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True

        outputs = model(
            batch["completion_input_ids"],
            attention_mask=batch["completion_attention_mask"],
            pixel_values=batch['pixel_values'],
            image_flags=batch['image_flags'],
            **model_kwargs,
        )
        completion_logits = outputs.logits

        completion_logps = self.get_batch_logps(
            completion_logits,
            batch["completion_labels"],
            average_log_prob=False,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        if completion_logps.shape[0] != len(batch["label"]):
            raise ValueError(
                "There is a mismatch between the number of examples in this batch and the number of "
                "examples for which an output sequence was predicted."
            )

        chosen_idx = [i for i in range(completion_logps.shape[0]) if batch["label"][i] is True]
        rejected_idx = [i for i in range(completion_logps.shape[0]) if batch["label"][i] is False]

        chosen_logps = completion_logps[chosen_idx, ...]
        rejected_logps = completion_logps[rejected_idx, ...]

        chosen_logits = completion_logits[chosen_idx, ...]
        rejected_logits = completion_logits[rejected_idx, ...]

        if self.aux_loss_enabled:
            return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, KL_logps, outputs.aux_loss)
        else:
            return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, KL_logps)
        
    def _prepare_deepspeed(self, model):
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepspeed_plugin.deepspeed_config
        if config_kwargs["zero_optimization"]["stage"] == 3:
            print('Enable DPOTrainer._prepare_deepspeed')
            return super()._prepare_deepspeed(model)

        print('Disable DPOTrainer._prepare_deepspeed')
        for param in model.parameters():
            param.requires_grad = False

        model.eval()
        model = model.to(self.accelerator.device)
        return model



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