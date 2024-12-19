# --------------------------------------------------------
# InternVL
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
from typing import Optional, Tuple, Union

import torch.utils.checkpoint
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging

from .configuration_internvl_distillation import InternVLDistillationConfig
from .modeling_intern_vit import InternVisionModel

logger = logging.get_logger(__name__)


class InternVLDistillation(PreTrainedModel):
    config_class = InternVLDistillationConfig
    main_input_name = 'pixel_values'
    _no_split_modules = ['InternVisionEncoderLayer']

    def __init__(self, config: InternVLDistillationConfig, teacher_model=None, student_model=None):
        super().__init__(config)

        # image_size = config.force_image_size or config.teacher_config.image_size
        patch_size = config.teacher_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.loss_fn = nn.CosineSimilarity(dim=-1)

        if student_model is not None:
            self.student_model = student_model
        else:
            self.student_model = InternVisionModel(config.student_config)
        if teacher_model is not None:
            self.teacher_model = teacher_model
        else:
            self.teacher_model = InternVisionModel(config.teacher_config)

        student_hidden_size = config.student_config.hidden_size
        teacher_hidden_size = config.teacher_config.hidden_size
        num_layers = abs(self.select_layer)
        self.linear_layers = nn.ModuleList()
        for i in range(num_layers):
            self.linear_layers.append(nn.Linear(student_hidden_size, teacher_hidden_size))

        if config.force_image_size != config.student_config.image_size:
            self.student_model.resize_pos_embeddings(
                old_size=config.student_config.image_size,
                new_size=config.force_image_size,
                patch_size=config.student_config.patch_size
            )
        if config.force_image_size != config.teacher_config.image_size:
            self.teacher_model.resize_pos_embeddings(
                old_size=config.teacher_config.image_size,
                new_size=config.force_image_size,
                patch_size=config.teacher_config.patch_size
            )

    def forward(
            self,
            pixel_values: torch.FloatTensor,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        with torch.no_grad():
            teacher_embeds = self.teacher_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer:]

        student_embeds = self.student_model(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True).hidden_states[self.select_layer:]

        loss_dict = {}
        for i in range(len(teacher_embeds)):
            student_embed = student_embeds[i]
            student_embed = self.linear_layers[i](student_embed)
            current_loss = -self.loss_fn(teacher_embeds[i], student_embed).mean()
            loss_dict[f'loss_{i}'] = current_loss
        if torch.distributed.get_rank() == 0:
            print(loss_dict)
        loss = sum(loss_dict.values()) / len(loss_dict.values())

        return CausalLMOutputWithPast(
            loss=loss,
        )
