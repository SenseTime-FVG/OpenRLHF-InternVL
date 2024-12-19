import sys
import logging
import os
import io
import re
import json
import random
import copy
import math
import gc
import mmap
from copy import deepcopy
import traceback
from typing import Dict, List, Union, Optional
import numpy as np
import torch
import torchvision.transforms as T
import transformers
from internvl.conversation import get_conv_template
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset
from torchvision.transforms.functional import InterpolationMode
from internvl.train.constants import (CLIP_MEAN, CLIP_STD, IMAGENET_MEAN, IMAGENET_STD,
                        IMG_CONTEXT_TOKEN, IMG_END_TOKEN, IMG_START_TOKEN,
                        SIGLIP_MEAN, SIGLIP_STD, IGNORE_TOKEN_ID)
from internvl.train.dataset import preprocess_internlm_v2, preprocess_internlm_v3, preprocess_internlm, dynamic_preprocess_v2, dynamic_preprocess_v3, find_closest_aspect_ratio_v2, find_closest_aspect_ratio_v3, build_transform, dynamic_preprocess_old, dynamic_preprocess_v1
from trl.trainer.utils import pad_to_length
# from train_utils import conversation_lib
import cv2
from datasets import features, load_dataset
import datasets
from packaging import version
import tokenizers
from collections import defaultdict

try:
    from aoss_client.client import Client
except ImportError as E:
    print('please install aoss_client')
    exit(-1)
logger = logging.getLogger(__name__)


class KTO_Dataset(Dataset):
    default_seed = 42
    """Dataset for supervised fine-tuning."""

    def __init__(self, template_name, meta, tokenizer, tcs_loader, num_image_token, ds_name,
                 image_size=448, is_train=True, pad2square=False, group_by_length=False,
                 dynamic_image_size=False, use_thumbnail=False, min_dynamic_patch=1,
                 max_dynamic_patch=6, repeat_time=1, normalize_type='imagenet', scale_threshold="old",
                 is_token_counting=False, read_image=True, max_num_image=32, min_num_image=4, sampling_method="rand", is_chinese=False, fix_seed=False,
                 max_dynamic_images = 6, max_multi_image_dynamic_patch = 6, max_patches = 42, task='kto'):
        super(KTO_Dataset, self).__init__()
        assert not pad2square  # should nenver be used
        assert min_dynamic_patch==1
        assert use_thumbnail
        self.task = task
        self.max_dynamic_images = max_dynamic_images
        self.max_multi_image_dynamic_patch = max_multi_image_dynamic_patch
        self.max_patches = max_patches

        self.tokenizer = tokenizer
        self.template_name = template_name
        self.num_image_token = num_image_token
        self.ds_name = ds_name
        self.img_context_token_id =  tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        # if not is_token_counting:
        #     logger.info(f'[Dataset] num_image_token: {num_image_token}')
        #     logger.info(f'[Dataset] dynamic_image_size: {dynamic_image_size}')
        #     logger.info(f'[Dataset] use_thumbnail: {use_thumbnail}')
        #     logger.info(f'[Dataset] use_aug: {is_train}')
        #     logger.info(f'[Dataset] min_dynamic_patch: {min_dynamic_patch}, max_dynamic_patch: {max_dynamic_patch}')
        #     logger.info(f'[Dataset] repeat_time: {repeat_time}')
        self.image_size = image_size
        self.is_train = is_train
        self.pad2square = pad2square
        self.is_token_counting = is_token_counting
        self.read_image = read_image
        self.scale_threshold = scale_threshold
        # self.raw_data = load_dataset(data_dir)['train'].cast_column("image", datasets.Image(decode=False))
        
        assert meta['annotation'].endswith('jsonl'), f'annotation must be jsonl, but got {meta["annotation"]}'
        self.file = open(meta['annotation'], 'rb')
        self.mmap = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        self.raw_data = self.calculate_offsets()
        ori_length = len(self.raw_data)
        if 'meta_file' in meta:
            self.meta_data = json.load(open(meta['meta_file'], "r"))
            assert len(self.meta_data) == len(self.raw_data)
            if isinstance(self.meta_data[0], list):
                self.meta_data = [{
                    'vit_num': x[0],
                    'token_num': x[1]
                } for x in self.meta_data]
            valid_inds = [i for i in range(len(self.meta_data)) if
                          0 < self.meta_data[i]['token_num'] <= tokenizer.model_max_length and self.meta_data[i]['vit_num'] <= self.max_patches]
            self.raw_data = [self.raw_data[i] for i in valid_inds]
            self.meta_data = [self.meta_data[i] for i in valid_inds]
            if len(self.raw_data) != ori_length and not is_token_counting:
                logger.info(f'{self.ds_name} Filter From {ori_length} to {len(self.raw_data)}')
        else:
            self.meta_data = None

        # process the repeat time here
        new_raw_data = []
        new_meta_data = []
        repeat_time_integer = int(repeat_time // 1)
        repeat_time_decimal = repeat_time % 1
        for _ in range(repeat_time_integer):
            new_raw_data.extend(self.raw_data)
            if self.meta_data is not None:
                new_meta_data.extend(self.meta_data)
        if repeat_time_decimal != 0:
            num_sample = max(int(len(self.raw_data) * repeat_time_decimal),1)  # at least one
            _state = random.getstate()
            random.seed(self.default_seed)
            choice_inds = random.sample(range(len(self.raw_data)), num_sample)
            new_raw_data.extend([self.raw_data[i] for i in choice_inds])
            if self.meta_data is not None:
                new_meta_data.extend([self.meta_data[i] for i in choice_inds])
            random.setstate(_state)
        if self.meta_data is not None:
            assert len(new_raw_data) == len(new_meta_data)
        self.raw_data = np.array(new_raw_data)
        self.meta_data = new_meta_data if self.meta_data is not None else None

        self.root = meta['root']
        self.cached_data_dict = {}
        self.tcs_loader = tcs_loader
        self.group_by_length = group_by_length
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.normalize_type = normalize_type
        # if self.group_by_length:
        #     if self.meta_data is not None:
        #         self.length = [x['token_num'] for x in self.meta_data]
        #     else:
        #         self.conv2length = {}  # using dict to speedup the calculation of token length
        #         self.length = []
        #         for data_item in self.raw_data:
        #             data_item = json.loads(data_item)
        #             if 'length' in data_item:
        #                 token_length = data_item['length']  # use precomputed length if exists
        #             else:
        #                 # compute token length using tokenizer
        #                 conversations = '\n'.join([temp['value'] for temp in data_item['conversations']])
        #                 str_length = len(conversations)
        #                 if str_length not in self.conv2length:
        #                     token_length = tokenizer(
        #                         conversations, return_tensors='pt', padding=False, truncation=False,
        #                     ).input_ids.size(1)
        #                     self.conv2length[str_length] = token_length + num_image_token * (
        #                                 max_dynamic_patch + use_thumbnail)
        #                 else:
        #                     token_length = self.conv2length[str_length]
        #             self.length.append(token_length)
        self.max_num_image = max_num_image
        self.min_num_image = min_num_image 
        self.sampling_method = sampling_method
        self.is_chinese = is_chinese
        self.fix_seed = fix_seed

        if self.fix_seed:
            random.seed(self.default_seed)
            np.random.seed(self.default_seed)

        self.random_i = list(range(len(self.raw_data)))
        random.shuffle(self.random_i)

        gc.collect()

    def __len__(self):
        return len(self.raw_data)
    
    def calculate_offsets(self):
        offsets = []
        offset = 0
        while offset < self.mmap.size():
            offsets.append(offset)
            offset = self.mmap.find(b'\n', offset) + 1
            if offset == 0:  # find returns -1 if '\n' is not found
                break
        return offsets

    
    def pure_text_get_item(self, data_item):
        
        image = Image.new('RGB', (224, 224), (255, 255, 255))
        if self.scale_threshold == "old":
            images = dynamic_preprocess_old(image, min_num=self.min_dynamic_patch, max_num=self.max_dynamic_patch,
                                        image_size=self.image_size, use_thumbnail=self.use_thumbnail)
        elif self.scale_threshold == "v1":
            images = dynamic_preprocess_v1(image, min_num=self.min_dynamic_patch, max_num=self.max_dynamic_patch,
                                        image_size=self.image_size, use_thumbnail=self.use_thumbnail)
        elif self.scale_threshold == "v2":
            images = dynamic_preprocess_v2(image, min_num=self.min_dynamic_patch, max_num=self.max_dynamic_patch,
                                        image_size=self.image_size, use_thumbnail=self.use_thumbnail)
        elif self.scale_threshold == "v3":
            images = dynamic_preprocess_v3(image, min_num=self.min_dynamic_patch, max_num=self.max_dynamic_patch,
                                        image_size=self.image_size, use_thumbnail=self.use_thumbnail)
        else:
            raise NotImplementedError
        transform = build_transform(is_train=self.is_train, input_size=self.image_size,
                                    pad2square=self.pad2square, normalize_type=self.normalize_type)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)
        assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'
        if self.template_name == 'internlm2-chat':
            preprocess_function = preprocess_internlm
        elif self.template_name == 'internlm2-chat-v2':
            preprocess_function = preprocess_internlm_v2
        elif self.template_name == 'internlm2-chat-v3':
            preprocess_function = preprocess_internlm_v3
        else:
            raise NotImplementedError
        
        # chosen_conv = [
        #     data_item['prompt'][0],
        #     data_item['chosen'][0]
        # ]
        # chosen_ret = preprocess_function(self.template_name, [deepcopy(chosen_conv)],
        #             self.tokenizer, [], self.num_image_token, ds_name=self.ds_name,
        #             truncation=True)

        # rejected_conv = [
        #     data_item['prompt'][0],
        #     data_item['rejected'][0]
        # ]
        # rejected_ret = preprocess_function(self.template_name, [deepcopy(rejected_conv)],
        #         self.tokenizer, [], self.num_image_token, ds_name=self.ds_name,
        #         truncation=True)

        # ret = dict(
        #     chosen_input_ids=chosen_ret['input_ids'][0],
        #     chosen_labels=chosen_ret['labels'][0],
        #     chosen_attention_mask=chosen_ret['attention_mask'][0],
        #     rejected_input_ids=rejected_ret['input_ids'][0],
        #     rejected_labels=rejected_ret['labels'][0],
        #     rejected_attention_mask=rejected_ret['attention_mask'][0],
        #     pixel_values=pixel_values,
        #     image_flags=torch.tensor([0] * num_patches, dtype=torch.long),
        # )
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                            self.tokenizer, [], self.num_image_token, ds_name=self.ds_name,
                            truncation=True)
        
        # if self.task == 'kto':
        ret = dict(
            completion_input_ids=ret['input_ids'][0],
            completion_labels=ret['labels'][0],
            completion_attention_mask=ret['attention_mask'][0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([0] * num_patches, dtype=torch.long),
            label=torch.tensor([data_item["label"]])
        )

        # if self.is_token_counting:
        #     ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
        #                     self.tokenizer, [], self.num_image_token, ds_name=self.ds_name,
        #                     truncation=False)
        #     ret = dict(
        #         input_ids=ret['input_ids'][0],
        #         labels=ret['labels'][0],
        #         attention_mask=ret['attention_mask'][0],
        #         num_patches=num_patches,
        #         image_flags=torch.tensor([0] * num_patches, dtype=torch.long)
        #     )
        # else:
        #     ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
        #                     self.tokenizer, [], self.num_image_token, ds_name=self.ds_name,
        #                     truncation=True)
        #     ret = dict(
        #         input_ids=ret['input_ids'][0],
        #         labels=ret['labels'][0],
        #         attention_mask=ret['attention_mask'][0],
        #         pixel_values=pixel_values,
        #         image_flags=torch.tensor([0] * num_patches, dtype=torch.long)
        #     )
        return ret
    
    
    def multi_modal_get_item(self, data_item):
        for i, conv in enumerate(data_item['conversations']):
            if conv['from'] == 'human':
                if '<image>' not in conv['value']:
                    data_item['conversations'][i]['value'] = '<image>\n' + data_item['conversations'][i]['value']
                break
        # assert len(data_item['prompt']) == len(data_item['chosen']) == len(data_item['rejected']), f'multi round chat not match'
        # single image 
        image_files = [1, ]
        if type(data_item['image']) is str:
            image_files = [data_item['image'], ]
        
        else:
            assert type(data_item['image']) is list
            image_files = data_item['image']
        image_list = []
        if self.read_image:
            for image_file in image_files:
                image_file = os.path.join(self.root, image_file) if not image_file.startswith('s3://') else self.root + image_file
                image = self.tcs_loader(image_file)
                image_list.append(image)
        else:
            if type(data_item['image']) is str:
                heights = [data_item['height'], ]
                widths = [data_item['width'], ]
            else:
                assert type(data_item['image']) is list
                heights = data_item['height']
                widths = data_item['width']
            image_list = [Image.new('RGB', (widths[i], heights[i]), (255, 255, 255)) for i in range(len(widths))]
        # image_list = [data_item['image']]
        transform = build_transform(is_train=self.is_train, input_size=self.image_size,
                                    pad2square=self.pad2square, normalize_type=self.normalize_type)
        raw_images = []
        num_patches_per_image = []
        pixel_values = []

        if self.read_image:
            for image in image_list:
                raw_images.append(image)
                if self.dynamic_image_size and len(image_files) <= self.max_dynamic_images:
                    if self.scale_threshold == "v2":
                        patches = dynamic_preprocess_v2(image, min_num=self.min_dynamic_patch, max_num=self.max_dynamic_patch if len(image_files) == 1 else self.max_multi_image_dynamic_patch,
                                                    image_size=self.image_size, use_thumbnail=self.use_thumbnail)
                    elif self.scale_threshold == "v3":
                        patches = dynamic_preprocess_v3(image, min_num=self.min_dynamic_patch, max_num=self.max_dynamic_patch if len(image_list) == 1 else self.max_multi_image_dynamic_patch,
                                                        image_size=self.image_size, use_thumbnail=self.use_thumbnail)
                    else:
                        raise NotImplementedError
                else:
                    patches = [image]
                num_patches_per_image.append(len(patches))
                pixel_values.extend([transform(patch) for patch in patches])
            pixel_values = torch.stack(pixel_values)
            # print('-----------', pixel_values.shape, len(patches))
            # import pdb;pdb.set_trace()
            num_patches = pixel_values.size(0)
            
        else:
            for image in image_list:
                raw_images.append(image)
                if self.dynamic_image_size and len(image_list) <= self.max_dynamic_images:
                    if self.scale_threshold == "v2":
                        target_aspect_ratio = find_closest_aspect_ratio_v2(image, min_num=self.min_dynamic_patch, max_num=self.max_dynamic_patch if len(image_list) == 1 else self.max_multi_image_dynamic_patch,
                                                        image_size=self.image_size, use_thumbnail=self.use_thumbnail)
                        num_patches_image = target_aspect_ratio[0] * target_aspect_ratio[1] 
                        num_patches_per_image.append(num_patches_image+1 if self.use_thumbnail and num_patches_image > 1 else num_patches_image)
                    elif self.scale_threshold == "v3":
                        target_aspect_ratio = find_closest_aspect_ratio_v3(image, min_num=self.min_dynamic_patch, max_num=self.max_dynamic_patch if len(image_list) == 1 else self.max_multi_image_dynamic_patch,
                                                        image_size=self.image_size, use_thumbnail=self.use_thumbnail)
                        num_patches_image = target_aspect_ratio[0] * target_aspect_ratio[1] 
                        num_patches_per_image.append(num_patches_image+1 if self.use_thumbnail and num_patches_image > 1 else num_patches_image)
                    else:
                        raise NotImplementedError
                else:
                    num_patches_per_image.append(1)
            num_patches = sum(num_patches_per_image)
        
        if not self.dynamic_image_size:
            assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'
        if self.template_name == 'internlm2-chat':
            preprocess_function = preprocess_internlm
        elif self.template_name == 'internlm2-chat-v2':
            preprocess_function = preprocess_internlm_v2
        elif self.template_name == 'internlm2-chat-v3':
            preprocess_function = preprocess_internlm_v3
        else:
            raise NotImplementedError

        

        # chosen_conv = [
        #     data_item['prompt'][0],
        #     data_item['chosen'][0]
        # ]
        # chosen_ret = preprocess_function(self.template_name, [deepcopy(chosen_conv)],
        #             self.tokenizer, num_patches_per_image, self.num_image_token, ds_name=self.ds_name,
        #             truncation=True)

        # rejected_conv = [
        #     data_item['prompt'][0],
        #     data_item['rejected'][0]
        # ]
        # rejected_ret = preprocess_function(self.template_name, [deepcopy(rejected_conv)],
        #         self.tokenizer, num_patches_per_image, self.num_image_token, ds_name=self.ds_name,
        #         truncation=True)

        # ret = dict(
        #     chosen_input_ids=chosen_ret['input_ids'][0],
        #     chosen_labels=chosen_ret['labels'][0],
        #     chosen_attention_mask=chosen_ret['attention_mask'][0],
        #     rejected_input_ids=rejected_ret['input_ids'][0],
        #     rejected_labels=rejected_ret['labels'][0],
        #     rejected_attention_mask=rejected_ret['attention_mask'][0],
        #     pixel_values=pixel_values,
        #     image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
        # )
        # loss_type == apo_zero_unpaired
        
            
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                            self.tokenizer, num_patches_per_image, self.num_image_token, ds_name=self.ds_name,
                            truncation=True)
        
        
        # if self.task == 'kto':


        ret = dict(
            completion_input_ids=ret['input_ids'][0],
            completion_labels=ret['labels'][0],
            completion_attention_mask=ret['attention_mask'][0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
            label=torch.tensor([data_item["label"]])
        )
        # if self.is_token_counting:
        #     ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
        #                     self.tokenizer, num_patches_per_image, self.num_image_token, ds_name=self.ds_name,
        #                     truncation=False)
        #     ret = dict(
        #         input_ids=ret['input_ids'][0],
        #         labels=ret['labels'][0],
        #         attention_mask=ret['attention_mask'][0],
        #         num_patches=num_patches,
        #         image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
        #         height=raw_images[0].height if len(raw_images) == 1 else [image.height for image in raw_images],
        #         width=raw_images[0].width if len(raw_images) == 1 else [image.width for image in raw_images],
        #     )
        # else:
            
        #     ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
        #                     self.tokenizer, num_patches_per_image, self.num_image_token, ds_name=self.ds_name,
        #                     truncation=True)
        #     ret = dict(
        #         input_ids=ret['input_ids'][0],
        #         labels=ret['labels'][0],
        #         attention_mask=ret['attention_mask'][0],
        #         pixel_values=pixel_values,
        #         image_flags=torch.tensor([1] * num_patches, dtype=torch.long)
        #     )
        #    # 'chosen_input_ids', 'chosen_attention_mask', 'chosen_labels', 'rejected_input_ids', 'rejected_attention_mask', 'rejected_labels', 
        return ret

    # def _get_kl_dataset(batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    #     """Creates mismatched pairs of prompts and completions for the KL dataset by adding a +1 offset to the order of completions."""
    #     batch["answer_input_ids"] = [batch["answer_input_ids"][-1]] + batch["answer_input_ids"][:-1]
    #     batch["answer_attention_mask"] = [batch["answer_attention_mask"][-1]] + batch["answer_attention_mask"][:-1]
    #     return batch


    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        assert i < len(self.raw_data)
        i = self.random_i[i]
        load_try = 1
        i = i % len(self.raw_data)
        while load_try:
            line_start = self.raw_data[i]
            line_end = self.mmap.find(b'\n', line_start)
            if line_end == -1:
                line_end = self.mmap.size()
            data = self.mmap[line_start:line_end].decode('utf-8')
            data_item = json.loads(data)
            try:
                if 'image' in data_item and data_item['image'] is not None and data_item['image'] != "":
                    ret = self.multi_modal_get_item(data_item)
                # elif 'video' in data_item and data_item['video'] is not None and data_item['video'] != "":
                #     ret = self.video_get_item(data_item, i)
                else:
                    ret = self.pure_text_get_item(data_item)
                break
            except Exception as e:
                print(e, self.ds_name)
                traceback.print_exc()
                if 'image' in data_item:
                    if type(data_item['image']) == list:
                        images = [self.root + item for item in data_item['image']]
                        print(f'Failed to load image: {images}, the dataset is: {self.ds_name}')
                    else:
                        if data_item['image'].startswith('s3://'):
                            data_path = self.root + data_item['image']
                        else:
                            data_path = os.path.join(self.root, data_item['image'])
                        print(f'Failed to load image: {data_path}, the dataset is: {self.ds_name}')
                elif 'video' in data_item:
                        data_path = os.path.join(self.root, data_item['video'])
                        print(f'Failed to load video: {data_path}, the dataset is: {self.ds_name}')
                if self.is_token_counting:
                    return None
                i = (i+100) % len(self.raw_data)
            load_try = load_try - 1
        # for s in ['chosen', 'rejected']:
        ret['ds_name'] = self.ds_name  # for debug
        if ret['image_flags'].sum() * self.num_image_token != (ret['completion_input_ids'] == self.img_context_token_id).sum():
            print(f"Image_flags mismatch {self.ds_name} {i} {data_item}")
            sys.stdout.flush()
        # print(ret.keys())
        return ret

    
    # def map(self, func, **kwargs):
    #     return self


def build_kto_datasets(data_args, tokenizer, tcs_loader, model, group_by_length=False,
                   dynamic_image_size=False, use_thumbnail=False, min_dynamic_patch=1,
                   max_dynamic_patch=6, normalize_type='imagenet'):
    datasets = []
    ds_collections = json.loads(open(data_args.meta_path).read())
    for ds_name in ds_collections.keys():
        repeat_time = ds_collections[ds_name]['repeat_time']
        if 'max_dynamic_patch' in ds_collections[ds_name]:
            max_num = ds_collections[ds_name]['max_dynamic_patch']
            logger.info(f'max_dynamic_patch is set to {max_num} according to the meta file')
        else:
            max_num = max_dynamic_patch
        is_train = data_args.force_image_aug
        max_num_image = ds_collections[ds_name]['max_num_image'] if 'max_num_image' in ds_collections[ds_name] else 32
        min_num_image = ds_collections[ds_name]['min_num_image'] if 'min_num_image' in ds_collections[ds_name] else 4
        is_chinese = ds_collections[ds_name]['is_chinese'] if 'is_chinese' in ds_collections[ds_name] else False
        sampling_method = ds_collections[ds_name]['sampling_method'] if 'sampling_method' in ds_collections[ds_name] else "rand"
        fix_seed = ds_collections[ds_name]['fix_seed'] if 'fix_seed' in ds_collections[ds_name] else False

        try:
            dataset = KTO_Dataset(
                data_args.conv_style, ds_collections[ds_name],
                tokenizer,
                tcs_loader,
                num_image_token=model.num_image_token,
                image_size=data_args.force_image_size,
                is_train=is_train,
                pad2square=data_args.pad2square,
                group_by_length=group_by_length,
                dynamic_image_size=dynamic_image_size,
                use_thumbnail=use_thumbnail,
                min_dynamic_patch=min_dynamic_patch,
                max_dynamic_patch=max_num,
                repeat_time=repeat_time,
                normalize_type=normalize_type,
                scale_threshold=data_args.scale_threshold,
                ds_name=ds_name,
                max_num_image=max_num_image,
                min_num_image=min_num_image,
                sampling_method=sampling_method,
                is_chinese=is_chinese,
                fix_seed=fix_seed
            )
        except Exception:
            logger.info(f'Error in loading dataset: {ds_name}')
            exit()
        datasets.append(dataset)
    assert not data_args.use_data_resampling
    train_dataset = ConcatDataset(datasets)
    
    # if data_args.pack_dataset:
    #     train_dataset = InternPackedDataset(
    #         train_dataset,
    #         num_image_token=model.num_image_token,
    #         vit_packed_length=data_args.vit_packed_length,
    #         llm_packed_length=data_args.llm_packed_length,
    #         loss_reduction=data_args.loss_reduction,
    #         tokenizer=tokenizer,
    #         iter_time=data_args.iter_time,
    #     )
    return train_dataset




# if __name__ == '__main__':
#     dataset = DPO_Dataset(data_dir='/mnt/afs/user/qinrui/project/hf_home/datasets/openbmb___rlaif-v-dataset')



