# --------------------------------------------------------
# InternVL
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from .configuration_internlm2 import InternLM2Config
from .modeling_internlm2 import InternLM2Model, InternLM2ForCausalLM
from .tokenization_internlm2 import InternLM2Tokenizer
from .tokenization_internlm2_fast import InternLM2TokenizerFast

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModel

AutoConfig.register("internlm2", InternLM2Config)
AutoTokenizer.register(InternLM2Config, InternLM2Tokenizer, InternLM2TokenizerFast)
AutoModel.register(InternLM2Config, InternLM2Model)
AutoModelForCausalLM.register(InternLM2Config, InternLM2ForCausalLM)

__all__ = ['InternLM2Config', 'InternLM2ForCausalLM',
           'InternLM2Tokenizer', 'InternLM2TokenizerFast']
