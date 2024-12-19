# --------------------------------------------------------
# InternVL
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import copy

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from .configuration_intern_vit import InternVisionConfig

logger = logging.get_logger(__name__)


class InternVLDistillationConfig(PretrainedConfig):
    model_type = 'internvl_distillation'
    is_composition = True

    def __init__(
            self,
            teacher_config=None,
            student_config=None,
            select_layer=-1,
            force_image_size=None,
            **kwargs):
        super().__init__(**kwargs)

        if teacher_config is None:
            teacher_config = {}
            logger.info('teacher_config is None. Initializing the InternVisionConfig with default values.')

        if student_config is None:
            student_config = {}
            logger.info('student_config is None. Initializing the InternVisionConfig with default values.')

        self.teacher_config = InternVisionConfig(**teacher_config)
        self.student_config = InternVisionConfig(**student_config)
        self.select_layer = select_layer
        self.force_image_size = force_image_size
        logger.info(f'vision_select_layer: {self.select_layer}')

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output['teacher_config'] = self.teacher_config.to_dict()
        output['student_config'] = self.student_config.to_dict()
        output['model_type'] = self.__class__.model_type
        output['select_layer'] = self.select_layer
        output['force_image_size'] = self.force_image_size

        return output
