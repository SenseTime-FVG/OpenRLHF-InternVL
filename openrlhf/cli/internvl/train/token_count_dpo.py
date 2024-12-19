import gc
import pdb
import logging
import math
import os
import random
import sys
import traceback
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, Optional
import torch.multiprocessing as mp
import orjson as json
import torch
import torch.distributed as dist
import transformers
from tqdm import tqdm
from internvl.dist_utils import init_dist
from internvl.model import *
from internvl.patch import (concat_pad_data_collator, packed_pad_data_collator,
                            replace_llama_rmsnorm_with_fused_rmsnorm,
                            replace_train_sampler, replace_internlm2_attention_class,
                            replace_qwen2_attention_class, dpo_concat_pad_data_collator,
                            kto_concat_pad_data_collator)
from internvl.train.constants import (BOX_END_TOKEN, BOX_START_TOKEN,
                                      IMG_CONTEXT_TOKEN, IMG_END_TOKEN,
                                      IMG_START_TOKEN, QUAD_END_TOKEN,
                                      QUAD_START_TOKEN, REF_END_TOKEN,
                                      REF_START_TOKEN)
from internvl.train.dataset import (TCSLoader, build_datasets)
from internvl.train.trainer_monkey_patch import replace_create_optimizer, add_conine_with_min_lr_scheduler
from PIL import Image, ImageFile, PngImagePlugin
from torch.utils.data import Dataset
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          HfArgumentParser, Trainer, TrainingArguments,AutoProcessor,
                          set_seed)
from trl import DPOConfig, KTOConfig
from dpo.trainers import DPOTrainerSelf, MultimodalDPOTrainer
# from kto.trainers import MultiModalKTOTrainer
from kto.kto_trainer import MultiModalKTOTrainer
# from dpo.dpo_dataset import build_dpo_dataset
from dpo.dpo_dataset_new import build_dpo_datasets, DPO_Dataset
from kto.kto_dataset import build_kto_datasets
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.logging import (enable_default_handler,
                                        enable_explicit_format, set_verbosity)
from typing import Dict, Literal, Optional
from enum import Enum
import multiprocessing
import numpy as np

# Upgrade transformers to v4.37.2, we don't need it anymore
# replace_llama2_attn_with_flash_attn()
replace_llama_rmsnorm_with_fused_rmsnorm()
# replace_train_sampler()
try:
    from aoss_client.client import Client
except ImportError as E:
    print('please install aoss_client')
    exit(-1)

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

os.environ['TOKENIZERS_PARALLELISM'] = 'true'


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    vision_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    llm_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    mlp_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    freeze_llm: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the LLM decoder.'},
    )
    freeze_backbone: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the vision backbone of the model.'},
    )
    freeze_mlp: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the MLP layers of the model.'},
    )
    unfreeze_vit_layers: int = field(
        default=0,
        metadata={'help': 'Specify the number of ViT layers to unfreeze. Default is 0.'},
    )
    vision_select_layer: int = field(
        default=-1,
        metadata={'help': 'Specify the layer of ViT feature map to use. Default is last layer.'},
    )
    use_backbone_lora: int = field(
        default=0,
        metadata={'help': 'Set the LoRA adapter rank for the backbone model. Default is 0.'}
    )
    use_llm_lora: int = field(
        default=0,
        metadata={'help': 'Set the LoRA adapter rank for the LLM. Default is 0.'}
    )
    unfreeze_lm_head: bool = field(
        default=False,
        metadata={'help': "Set to True to unfreeze the language model's head."},
    )
    use_custom_trainer: bool = field(
        default=False,
        metadata={'help': 'Set to True to enable the use of a custom trainer.'},
    )
    grad_checkpoint: Optional[bool] = field(
        default=False,
        metadata={'help': 'Set to True to use gradient checkpointing.'},
    )
    drop_path_rate: float = field(
        default=0.0,
        metadata={'help': 'Set the drop path rate for the ViT model. Default is 0.'},
    )
    ps_version: str = field(
        default='v1',
        metadata={'help': 'Specify the version of pixel shuffle implementation. Default is `v1`.'
                          'Please use `v2` to fix the bug of transposed image.'}
    )
    add_special_token: Optional[bool] = field(
        default=True,
        metadata={'help': 'add special token'},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    max_seq_length: Optional[int] = field(
        default=2048,
        metadata={
            'help': (
                'The maximum total input sequence length after tokenization. Sequences longer '
                'than this will be truncated, sequences shorter will be padded.'
            )
        },
    )
    force_image_size: Optional[int] = field(
        default=224,
        metadata={'help': 'Set the desired size for the image. Default is 224.'},
    )
    down_sample_ratio: Optional[float] = field(
        default=1.0,
        metadata={'help': 'Set the desired down-sampling ratio for the image. Default is 1.0.'},
    )
    pad2square: Optional[bool] = field(
        default=False,
        metadata={'help': 'Pad the image to a square shape if set to True.'},
    )
    conv_style: Optional[str] = field(
        default='internvl_zh', metadata={'help': 'Prompt style for a conversation.'}
    )
    meta_path: Optional[str] = field(
        default=None,
        metadata={'help': 'The path of the meta file of datasets.'},
    )
    use_data_resampling: Optional[bool] = field(
        default=False,
        metadata={'help': 'Set to True to use data resampling.'},
    )
    dynamic_image_size: Optional[bool] = field(
        default=False,
        metadata={'help': 'Set to True to use dynamic image size.'},
    )
    use_thumbnail: Optional[bool] = field(
        default=False,
        metadata={'help': 'Set to True to add a thumbnail image.'},
    )
    min_dynamic_patch: Optional[int] = field(
        default=1,
        metadata={'help': 'The minimum number of dynamic patches. Default is 1.'},
    )
    max_dynamic_patch: Optional[int] = field(
        default=12,
        metadata={'help': 'The maximum number of dynamic patches. Default is 6.'},
    )
    neftune_alpha: Optional[float] = field(
        default=None,
        metadata={'help': 'The noise_alpha value for NEFTune. Default is None.'},
    )
    normalize_type: Optional[str] = field(
        default='imagenet',
        metadata={'help': 'The normalize type for the image. Default is imagenet.'},
    )
    force_image_aug: Optional[bool] = field(
        default=True,
        metadata={'help': 'use image aug for all data'},
    )
    scale_threshold: Optional[str] = field(
        default='old',
        metadata={'help': 'dynamic resolution strategy type and scale threshold(1.0, 0.9, 0.75)'},
    )
    pack_dataset: Optional[bool] = field(
        default=False,
        metadata={'help': 'using pack dataset'},
    )
    vit_packed_length: Optional[int] = field(
        default=14,
        metadata={'help': 'max pack vit patch'},
    )
    llm_packed_length: Optional[int] = field(
        default=6144,
        metadata={'help': 'max pack seq'},
    )
    loss_reduction: Optional[str] = field(
        default='sqrt',
        metadata={'help': 'Loss reduction method. Default is `sqrt`'},
    )
    iter_time: Optional[int] = field(
        default=1,
        metadata={'help': 'iter_time for packed '},
    )
    token_out_dir: Optional[str] = field(
        default=None,
        metadata={'help': 'The directory to save token numbers.'},
    )
    split_number: Optional[int] = field(
        default=1,
        metadata={'help': 'The total number of splits'},
    )
    split_index: Optional[int] = field(
        default=0,
        metadata={'help': 'The index of splits'},
    )
    
    # data_dir: str = '/mnt/afs/user/qinrui/project/hf_home/datasets/openbmb___rlaif-v-dataset'
    # data_dir: str = '/mnt/afs/liangjinwei/.cache/huggingface/datasets/openbmb___rlaif-v-dataset'
    


@dataclass
class CustomTrainingArguments(DPOConfig):
    """
    Arguments to train model.
    """
    vit_lr_decay: Optional[float] = field(
        default=1.0,
        metadata={'help': 'vit learning rate decay'},
    )
    vit_lr_scale: Optional[float] = field(
        default=1.0,
        metadata={'help': 'vit learning rate scale'},
    )
    llm_lr_scale: Optional[float] = field(
        default=1.0,
        metadata={'help': 'llm learning rate scale'},
    )
    mlp_lr_scale: Optional[float] = field(
        default=1.0,
        metadata={'help': 'mlp learning rate scale'},
    )
    min_lr_rate: Optional[float] = field(
        default=0.0,
        metadata={'help': 'min_lr_rate for cosine'},
    )
    task: str = field(
        default='DPO',
        metadata={
            'help': 'LM for language modeling. DPO for direct preference optimization'
        }
    )
    loss_type: str = field(
        default='sigmoid',
        metadata={
            'help': 'kto, apo_zero_unpaired can be used'
        }
    )
    precompute_ref_log_probs: bool = field(
        default=False,
        metadata={
            'help': 'whether to precompute ref log probs'
        }
    )
    beta: float = field(
        default=0.1,
        metadata={
            'help': 'beta for KL divergence'
        }
    )


def counting_tokens(params):
    """
    计算给定索引列表中每个样本的 token 数量。
    """
    inds, ds_name, args, kwargs = params
    token_lengths = []
    dataset = DPO_Dataset(*args, **kwargs)

    if inds[0] == 0:
        print(ds_name)
        inds = tqdm(inds)
    for idx in inds:
        item = dataset.__getitem__(idx)
        if item is not None:
            chosen_tokens = len(item['chosen_input_ids'])
            rejected_tokens = len(item['rejected_input_ids'])
        else:
            chosen_tokens, rejected_tokens = -1, -1
        token_lengths.append({'chosen': chosen_tokens, 'rejected': rejected_tokens})
    return token_lengths

def main():
    # Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # If use DeepSpeed zero3, init_dist must before HfArgumentParser
    launcher = os.environ.get('LAUNCHER', 'slurm')
    init_dist(launcher=launcher, backend='nccl')
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.pack_dataset = data_args.pack_dataset
    training_args.remove_unused_columns = False
    # training_args.gradient_checkpointing = model_args.grad_checkpoint
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry('InternV-Chat', model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    set_verbosity(log_level)
    enable_default_handler()
    enable_explicit_format()

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model, tokenizer, and image processor
    tokenizer_path = model_args.model_name_or_path or model_args.llm_path
    logger.info(f'Loading Tokenizer: {tokenizer_path}')
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, add_eos_token=False, trust_remote_code=False, use_fast=True)
    tokenizer.tokenizer_path = tokenizer_path
    tokenizer.model_max_length = data_args.max_seq_length

    if model_args.add_special_token:
        token_list = [IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN,
                    QUAD_START_TOKEN, QUAD_END_TOKEN, REF_START_TOKEN,
                    REF_END_TOKEN, BOX_START_TOKEN, BOX_END_TOKEN]
        num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=True)
        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        logger.info(f"num_new_tokens: {num_new_tokens} img_context_token_id: {img_context_token_id}")
    else:
        token_list = [IMG_CONTEXT_TOKEN, ]
        num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=True)
        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        logger.info(f"num_new_tokens: {num_new_tokens} img_context_token_id: {img_context_token_id}")
    tcs_loader = TCSLoader('~/aoss.conf')

    set_seed(training_args.seed)

    try:
        model_config = json.load(f"{tokenizer_path}/config.json")
        patch_size = model_config['vision_config']['patch_size']
    except:
        patch_size = 14
    
    datasets = {}
    ds_collections = json.loads(open(data_args.meta_path).read())

    # filter
    keys_lengths = []
    for key in ds_collections:
        save_path = os.path.join(
            str(data_args.token_out_dir), 
            os.path.basename(str(ds_collections[key]['annotation'])).replace('.jsonl', '_token_count.jsonl'))
        if os.path.exists(save_path):
            with open(save_path) as f:
                if len(f.readlines()) ==  ds_collections[key]['length']:
                    print(f"token number file {save_path} exists")
                    continue
        else:
            keys_lengths.append((key, ds_collections[key]['length']))

    # split
    split_number = max(min(data_args.split_number, len(ds_collections)), 1)
    split_index = max(min(data_args.split_index, split_number-1), 0)

    keys = [[] for _ in range(split_number)]
    partitions_lengths = [0] * split_number
    keys_lengths.sort(key=lambda x: (x[1], x[0]), reverse=True)
    for key, length in keys_lengths:
        min_index = partitions_lengths.index(min(partitions_lengths))
        keys[min_index].append(key)
        partitions_lengths[min_index] += length
    keys = keys[split_index]
    print(f"dataset names: {keys}")


    for ds_name in keys:
        # pdb.set_trace()

        save_path = os.path.join(
            str(data_args.token_out_dir), 
            os.path.basename(str(ds_collections[ds_name]['annotation'])).replace('.jsonl', '_token_count.jsonl'))
        if os.path.exists(save_path):
            with open(save_path) as f:
                if len(f.readlines()) ==  ds_collections[ds_name]['length']:
                    print(f"token number file {save_path} exists")
                    continue

        repeat_time = 1
        ds_collections[ds_name].pop('ref_logps', None)

        # setting max_dynamic_patch
        if 'max_dynamic_patch' in ds_collections[ds_name]:
            max_num = ds_collections[ds_name]['max_dynamic_patch']
            logger.info(f'max_dynamic_patch is set to {max_num} according to the meta file')
        else:
            max_num = data_args.max_dynamic_patch

        # other parameters
        is_train = data_args.force_image_aug
        max_num_image = ds_collections[ds_name]['max_num_image'] if 'max_num_image' in ds_collections[ds_name] else 32
        min_num_image = ds_collections[ds_name]['min_num_image'] if 'min_num_image' in ds_collections[ds_name] else 4
        is_chinese = ds_collections[ds_name]['is_chinese'] if 'is_chinese' in ds_collections[ds_name] else False
        sampling_method = ds_collections[ds_name]['sampling_method'] if 'sampling_method' in ds_collections[ds_name] else "rand"
        fix_seed = ds_collections[ds_name]['fix_seed'] if 'fix_seed' in ds_collections[ds_name] else False
        num_image_token = int((data_args.force_image_size // patch_size) ** 2 * (data_args.down_sample_ratio ** 2))

        args = (data_args.conv_style, ds_collections[ds_name], tokenizer, tcs_loader)
        kwargs = dict(
            num_image_token=num_image_token,
            image_size=data_args.force_image_size,
            is_train=is_train,
            pad2square=data_args.pad2square,
            group_by_length=training_args.group_by_length,
            dynamic_image_size=data_args.dynamic_image_size,
            use_thumbnail=data_args.use_thumbnail,
            min_dynamic_patch=data_args.min_dynamic_patch,
            max_dynamic_patch=max_num,
            repeat_time=repeat_time,
            normalize_type=data_args.normalize_type,
            scale_threshold=data_args.scale_threshold,
            ds_name=ds_name,
            max_num_image=max_num_image,
            min_num_image=min_num_image,
            sampling_method=sampling_method,
            is_chinese=is_chinese,
            fix_seed=fix_seed
        )

        # # 确定使用的进程数
        # dataset_length = int(ds_collections[ds_name]['length'])
        # if dataset_length < 100:
        #     PROCESSES = 1
        # elif dataset_length < 10 * 1000:
        #     PROCESSES = 8
        # else:
        #     PROCESSES = 12

        # PROCESSES = 2


        # # 分割索引
        # index_splits = np.array_split(range(dataset_length), PROCESSES)

        # # 创建 Pool 并并行计算
        # with multiprocessing.Pool(PROCESSES) as pool:
        #     token_lengths_all = pool.map(
        #         counting_tokens,
        #         [(inds, ds_name, args, kwargs) for inds in index_splits]
        #     )

        # # 合并所有结果
        # all_token_lengths = [item for sublist in token_lengths_all for item in sublist]

        # # 将结果写入文件
        # with open(save_path, 'wb') as f:
        #     for token_length in all_token_lengths:
        #         f.write(json.dumps(token_length).strip() + b'\n')

        # print(f"save token length file {save_path}")


        dataset = DPO_Dataset(*args, **kwargs)
        def counting_tokens(inds):
            token_lengths = []

            if inds[0] == 0:
                print(ds_name)
                inds = tqdm(inds)
            for idx in inds:
                item = dataset.__getitem__(idx)
                if item is not None:
                    chosen_tokens = len(item['chosen_input_ids'])
                    rejected_tokens = len(item['rejected_input_ids'])
                else:
                    chosen_tokens, rejected_tokens = -1, -1
                token_lengths.append({'chosen': chosen_tokens, 'rejected': rejected_tokens})
            return token_lengths

        l_token_lengths = counting_tokens(range(len(dataset)))
        # pdb.set_trace()

        with open(save_path, 'wb') as f:
            for token_length in l_token_lengths:
                f.write(json.dumps(token_length) + b'\n')

        print(f"save token length file {save_path}")


if __name__ == '__main__':
    mp.set_start_method('fork', force=True)
    main()
