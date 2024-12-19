
import io
import gc
import os
import json
import random
import numpy
import torch
import base64
import pandas as pd
import os.path as op
import torch.utils.data as torch_data

from PIL import Image
from typing import List, Iterator
# from muffin.data.tsv_file import TSVFile
from torch.utils.data.sampler import Sampler
# from muffin.data.data_processors import register_data_processor
# from internvl.train.dpo.muffin_inference_logp import inference_logp
import datasets as hf_datasets
# from internvl.train.dpo.train_muffin import DataCollatorForDPODataset
# from internvl.train.dpo.train_utils import preprocess_v1, expand_image_token, preprocess
import transformers
from functools import partial
import copy

def bytes_to_PIL_image(img_buffer):
    img_io = io.BytesIO(img_buffer)
    img_io.seek(0)
    image = Image.open(img_io).convert('RGB')
    return image

class RLAIFVDataset(torch_data.Dataset):
    def __init__(self, data_dir: str, reference_model=None,
                 tokenizer=None, image_token_len=None, img_processor=None, use_im_start_end=True, is_llava15=False):
        super().__init__()

        if not op.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)

        data_path = [file for file in os.listdir(data_dir) if file.endswith('.parquet') and 'logp' in file]
        self.data_path = data_dir

        if len(data_path) == 0:
            assert reference_model is not None, "`reference_model` is mandatory when logps do not exist."

            if not op.exists('./RLAIF-V-Dataset'):
                os.mkdir('./RLAIF-V-Dataset')
            hf_data = hf_datasets.load_dataset('openbmb/RLAIF-V-Dataset', cache_dir='./RLAIF-V-Dataset')['train'].cast_column("image", hf_datasets.Image(decode=False))

            inference_logp(reference_model, tokenizer, hf_data, self.data_path,
                            image_token_len, img_processor, use_im_start_end, is_llava15=is_llava15)

            torch.distributed.barrier()

            self.data = hf_datasets.load_dataset(data_dir)['train'].cast_column("image", hf_datasets.Image(decode=False))
        else:
            self.data = hf_datasets.load_dataset(data_dir)['train'].cast_column("image", hf_datasets.Image(decode=False))

        self.line_idx = list(range(len(self.data)))
        random.shuffle(self.line_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[self.line_idx[index]]
        question = {'from': 'human', 'value': f"<image>\n{sample['question']}"}
        chosen = {'from': 'gpt', 'value': sample['chosen']}
        rejected = {'from': 'gpt', 'value': sample['rejected']}

        image = bytes_to_PIL_image(sample['image']['bytes'])

        metainfo = {
            "origin_dataset": sample['origin_dataset'],
            "origin_split": sample['origin_split'],
            "origin_idx": sample['idx'],
            "image_id": sample['image_path'],
        }

        data_dict = {
            'image': image,
            "question": question,
            "chosen": chosen,
            "rejected": rejected,
            "idx": sample['idx'],
            "metainfo": metainfo
        }
        logps=json.loads(sample['logps'])

        if type(logps) == type([]):
            (data_dict['ref_win_logp'], data_dict['ref_win_avg_logp'], data_dict['ref_win_per_token_logp'],
            data_dict['ref_rej_logp'], data_dict['ref_rej_avg_logp'], data_dict['ref_rej_per_token_logp']) = logps
        else:
            (data_dict['ref_win_logp'], data_dict['ref_win_avg_logp'], data_dict['ref_win_per_token_logp'],
            data_dict['ref_rej_logp'], data_dict['ref_rej_avg_logp'], data_dict['ref_rej_per_token_logp']) = logps['logps']

        return data_dict



class DPODataset(torch_data.Dataset):
    def __init__(self,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_dir: str,
                 multimodal_cfg: dict,
                 reference_model = None):
        super(DPODataset, self).__init__()

        self.tokenizer = tokenizer
        self.list_data_dict = RLAIFVDataset(data_dir, reference_model, tokenizer,multimodal_cfg['image_token_len'], multimodal_cfg['image_processor'], multimodal_cfg['use_im_start_end'], is_llava15=True)
        self.multimodal_cfg = multimodal_cfg
        self.multimodal_cfg['keep_image_tag'] = True


    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        source: dict = self.list_data_dict[i]
        preprocess_func = partial(preprocess_v1, has_image=True)
        rej_data_dict, win_data_dict = encode_multimodal_preference_sample(
            source, self.tokenizer, self.multimodal_cfg, preprocess_func=preprocess_func)
        return rej_data_dict, win_data_dict

    
def encode_multimodal_preference_sample(source, tokenizer, multimodal_cfg, preprocess_func=None):
    if isinstance(source['chosen'], list):
        win_conv = source['chosen']
        rej_conv = source['rejected']
    elif isinstance(source['chosen'], dict):
        win_conv = copy.deepcopy([source['question'], source["chosen"]])
        rej_conv = copy.deepcopy([source['question'], source["rejected"]])

    if 'image' in source:
        image = source['image']
        image = multimodal_cfg['image_processor'](image)
        win_conv = expand_image_token(win_conv, multimodal_cfg)
        rej_conv = expand_image_token(rej_conv, multimodal_cfg)

    if preprocess_func is None:
        rej_data_dict = preprocess([rej_conv], tokenizer)
        rej_data_dict = dict(input_ids=rej_data_dict["input_ids"][0],
                             labels=rej_data_dict["labels"][0])

        win_data_dict = preprocess([win_conv], tokenizer)
        win_data_dict = dict(input_ids=win_data_dict["input_ids"][0],
                             labels=win_data_dict["labels"][0])
    else:
        rej_data_dict = preprocess_func([rej_conv], tokenizer)
        win_data_dict = preprocess_func([win_conv], tokenizer)

        if 'context_ids' in rej_data_dict:
            rej_data_dict = dict(input_ids=rej_data_dict["input_ids"][0],
                                labels=rej_data_dict["labels"][0],
                                image_bounds=rej_data_dict['image_bounds'][0],
                                context_ids=rej_data_dict['context_ids'][0],
                                position_ids=rej_data_dict['position_ids'][0]
                                )
            win_data_dict = dict(input_ids=win_data_dict["input_ids"][0],
                                labels=win_data_dict["labels"][0],
                                image_bounds=win_data_dict['image_bounds'][0],
                                context_ids=win_data_dict['context_ids'][0],
                                position_ids=win_data_dict['position_ids'][0]
                                )
        else:
            rej_data_dict = dict(input_ids=rej_data_dict["input_ids"][0],
                                labels=rej_data_dict["labels"][0])
            win_data_dict = dict(input_ids=win_data_dict["input_ids"][0],
                                labels=win_data_dict["labels"][0])

    # print(f'rej dict: {rej_data_dict}', flush=True)
    # print('inputs:', tokenizer.decode([(x if x != -200 else 0) for x in rej_data_dict['input_ids'].tolist()]), flush=True)
    # print('labels:', tokenizer.decode([(x if x != -100 else 0) for x in rej_data_dict['labels'].tolist()]), flush=True)

    # image exist in the data
    if 'image' in source:
        rej_data_dict['image'] = win_data_dict['image'] = image
    elif multimodal_cfg['is_multimodal']:
        # image does not exist in the data, but the model is multimodal
        crop_size = multimodal_cfg['image_processor'].crop_size
        rej_data_dict['image'] = win_data_dict['image'] = torch.zeros(
            3, crop_size['height'], crop_size['width'])

    if 'ref_win_logp' in source:
        rej_data_dict['ref_rej_logp'] = source['ref_rej_logp']
        win_data_dict['ref_win_logp'] = source['ref_win_logp']
        rej_data_dict['ref_rej_avg_logp'] = source['ref_rej_avg_logp']
        win_data_dict['ref_win_avg_logp'] = source['ref_win_avg_logp']
        rej_data_dict['ref_rej_per_token_logp'] = source['ref_rej_per_token_logp']
        win_data_dict['ref_win_per_token_logp'] = source['ref_win_per_token_logp']
    return rej_data_dict, win_data_dict



def build_dpo_dataset(tokenizer, data_args,reference_model):
    train_dataset = DPODataset(tokenizer=tokenizer,
                               data_dir=data_args.data_dir,
                               multimodal_cfg=dict(
                                   is_multimodal=data_args.is_multimodal,
                                   image_token_len=data_args.image_token_len,
                                   image_folder=data_args.image_folder,
                                   image_aspect_ratio=data_args.image_aspect_ratio,
                                   use_im_start_end=getattr(
                                       data_args, 'mm_use_im_start_end', False),
                                   image_processor=getattr(
                                       data_args, 'image_processor', None),
                                   data_source_names=getattr(
                                       data_args, 'data_source_names'),
                                   data_source_weights=getattr(data_args, 'data_source_weights'),
                                   shuffle_data=data_args.shuffle_data
                                   ),
                               reference_model=reference_model)
    print(f'Train data size is {len(train_dataset)}', flush=True)
    data_collator = DataCollatorForDPODataset(
        tokenizer=tokenizer, beta=data_args.dpo_beta, mod_token_weight=data_args.dpo_token_weight)

    if data_args.eval_data_source_names is not None:
        eval_datasets = {}
        for name in data_args.eval_data_source_names:
            eval_dataset = DPODataset(tokenizer=tokenizer,
                                      data_dir=data_args.data_dir,
                                      multimodal_cfg=dict(
                                          is_multimodal=data_args.is_multimodal,
                                          image_token_len=data_args.image_token_len,
                                          image_folder=data_args.image_folder,
                                          image_aspect_ratio=data_args.image_aspect_ratio,
                                          use_im_start_end=getattr(
                                              data_args, 'mm_use_im_start_end', False),
                                          image_processor=getattr(
                                              data_args, 'image_processor', None),
                                          data_source_names=[name],
                                          data_source_weights=[1],
                                           shuffle_data=False
                                          ),
                                      reference_model=reference_model)
            eval_datasets[name] = eval_dataset
    else:
        eval_datasets = None

    return dict(train_dataset=train_dataset,
                eval_dataset=eval_datasets,
                data_collator=data_collator)


