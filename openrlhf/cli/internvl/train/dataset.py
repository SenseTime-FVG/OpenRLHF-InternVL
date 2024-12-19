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
from typing import Dict, List
import numpy as np
import torch
import torchvision.transforms as T
import transformers
from internvl.conversation import get_conv_template
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset
from torchvision.transforms.functional import InterpolationMode
from .constants import (CLIP_MEAN, CLIP_STD, IMAGENET_MEAN, IMAGENET_STD,
                        IMG_CONTEXT_TOKEN, IMG_END_TOKEN, IMG_START_TOKEN,
                        SIGLIP_MEAN, SIGLIP_STD, IGNORE_TOKEN_ID)

import cv2
import decord
from decord import VideoReader
import imageio
import os

try:
    from aoss_client.client import Client
except ImportError as E:
    print('please install aoss_client')
    exit(-1)
logger = logging.getLogger(__name__)


def get_frame_indices(num_frames, vlen, sample='rand', fix_start=None, input_fps=1, max_num_frames=-1):
    if sample in ["rand", "middle"]: # uniform sampling
        acc_samples = min(num_frames, vlen)
        # split the video into `acc_samples` intervals, and sample from each interval.
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == 'rand':
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
            except:
                frame_indices = np.random.permutation(vlen)[:acc_samples]
                frame_indices.sort()
                frame_indices = list(frame_indices)
        elif fix_start is not None:
            frame_indices = [x[0] + fix_start for x in ranges]
        elif sample == 'middle':
            frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError

        if len(frame_indices) < num_frames:  # padded with last frame
            padded_frame_indices = [frame_indices[-1]] * num_frames
            padded_frame_indices[:len(frame_indices)] = frame_indices
            frame_indices = padded_frame_indices
    elif "fps" in sample:  # fps0.5, sequentially sample frames at 0.5 fps
        output_fps = float(sample[3:])
        duration = float(vlen) / input_fps
        delta = 1 / output_fps  # gap between frames, this is also the clip length each frame represents
        frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
        frame_indices = np.around(frame_seconds * input_fps).astype(int)
        frame_indices = [e for e in frame_indices if e < vlen]
        if max_num_frames > 0 and len(frame_indices) > max_num_frames:
            frame_indices = frame_indices[:max_num_frames]
            # frame_indices = np.linspace(0 + delta / 2, duration + delta / 2, endpoint=False, num=max_num_frames)
    else:
        raise ValueError
    return frame_indices


def read_frames_gif(
        video_path, num_frames, sample='rand', fix_start=None, 
        client=None, min_num_frames=4
    ):
    if video_path.startswith('s3'):
        video_bytes = client.get(video_path)
        gif = imageio.get_reader(io.BytesIO(video_bytes))
    else:
        gif = imageio.get_reader(video_path)
    vlen = len(gif)
    # with open("/mnt/afs/user/liuqinying/Husky-13b-llama2/vlen.txt", 'a') as f:
    #     f.write(str(vlen) + '\n')
    
    t_num_frames = np.random.randint(min_num_frames, num_frames + 1)
    frame_indices = get_frame_indices(
        t_num_frames, vlen, sample=sample, fix_start=fix_start
    )
    frames = []
    for index, frame in enumerate(gif):
        if index in frame_indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB).astype(np.uint8)
            frame = Image.fromarray(frame)
            frames.append(frame)
    return frames


def read_frames_decord(
        video_path, num_frames, sample='rand', fix_start=None, 
        client=None, clip=None, min_num_frames=4
    ):
    if video_path.startswith('s3'):
        video_bytes = client.get(video_path)
        video_reader = VideoReader(io.BytesIO(video_bytes), num_threads=1)
    else:
        video_reader = VideoReader(video_path, num_threads=1)
    vlen = len(video_reader)
    # with open("/mnt/afs/user/liuqinying/Husky-13b-llama2/vlen.txt", 'a') as f:
    #     f.write(str(vlen) + '\n')
    fps = video_reader.get_avg_fps()
    duration = vlen / float(fps)
    if clip:
        start, end = clip
        duration = end - start
        vlen = int(duration * fps)
        start_index = int(start * fps)
        
    # t_num_frames = min(max(int(duration * sample_fps), min_num_frames), num_frames)
    t_num_frames = np.random.randint(min_num_frames, num_frames + 1)
        
    frame_indices = get_frame_indices(
        t_num_frames, vlen, sample=sample, fix_start=fix_start,
        input_fps=fps
    )
    if clip:
        frame_indices = [f + start_index for f in frame_indices]
    frames = video_reader.get_batch(frame_indices).asnumpy()  # (T, H, W, C), np.uint8
    frames = [Image.fromarray(frames[i]) for i in range(frames.shape[0])]
    return frames


def read_frames_folder(
        video_path, num_frames, sample='rand', fix_start=None, 
        client=None, clip=None, min_num_frames=4
    ):
    def extract_number(file_name):
        match = re.search(r'(\d+)\.[a-zA-Z]+$', file_name)
        if match:
            return int(match.group(1))
        print("Invalid File name")
        return float('inf')  # if not found any number, return inf to ensure the file is at last
    
    if video_path.startswith('s3://'):
        image_list = list(client.list(video_path))
    else:
        image_list = sorted(list(os.listdir(video_path)))
    frames = sorted(image_list, key=extract_number)
    frames = [os.path.join(video_path, f) for f in frames]
    vlen = len(frames)
    # with open("/mnt/afs/user/liuqinying/Husky-13b-llama2/vlen.txt", 'a') as f:
    #     f.write(str(vlen) + '\n')
    
    t_num_frames = np.random.randint(min_num_frames, num_frames + 1)
        
    if vlen > t_num_frames:
        frame_indices = get_frame_indices(
            t_num_frames, vlen, sample=sample, fix_start=fix_start
            )
        frames = [frames[i] for i in frame_indices]
    # read frames
    frames = [Image.open(io.BytesIO(client.get(f))) for f in frames ]
    return frames


def pil_loader(img_str):
    buff = io.BytesIO(img_str)
    img = Image.open(buff)
    return img.convert('RGB')


class TCSLoader(object):

    def __init__(self, conf_path, sc_config_key='sensecore'):
        print(f'[TCSLoader] config_path: {conf_path}')
        print('--> before Client(conf_path)')
        self.client = Client(conf_path)
        self.sc_config_key = sc_config_key
        print('--> after Client(conf_path)')

    def __call__(self, fn, image_type='image', max_num_frames=-1, min_num_frames=4, sample='rand', clip=None):
        if image_type == "image":
            if fn.startswith("s3://"):
                img_value_str = self.client.get(fn)
                img = pil_loader(img_value_str)
            else:
                img = Image.open(fn).convert('RGB')
            return img
        
        elif image_type == "video":
            if fn.endswith('/'):
                frames = read_frames_folder(fn, num_frames=max_num_frames, min_num_frames=min_num_frames, client=self.client, sample=sample)
            elif fn.endswith('.gif'):
                frames = read_frames_gif(fn, num_frames=max_num_frames, min_num_frames=min_num_frames, client=self.client, sample=sample)
            else:
                frames = read_frames_decord(fn, num_frames=max_num_frames, min_num_frames=min_num_frames, client=self.client, sample=sample, clip=clip)
            return frames
                
        
    

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def simulate_jpeg_degradation(quality):
    def jpeg_degrade(img):
        with io.BytesIO() as output:
            img.convert('RGB').save(output, format='JPEG', quality=quality)
            output.seek(0)  # Move the reading cursor to the start of the stream
            img_jpeg = Image.open(output).copy()  # Use .copy() to make sure the image is loaded in memory
        return img_jpeg
    return jpeg_degrade


# Define the JPEG compression quality range, pre-create all JPEG compression functions
qualities = list(range(75, 101))
jpeg_degrade_functions = {quality: simulate_jpeg_degradation(quality) for quality in qualities}


def build_transform(is_train, input_size, pad2square=False, normalize_type='imagenet'):
    if normalize_type == 'imagenet':
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    elif normalize_type == 'clip':
        MEAN, STD = CLIP_MEAN, CLIP_STD
    elif normalize_type == 'siglip':
        MEAN, STD = SIGLIP_MEAN, SIGLIP_STD
    else:
        raise NotImplementedError
    if is_train:  # use data augumentation
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.RandomChoice([T.Lambda(jpeg_degrade_functions[quality]) for quality in qualities]),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
    else:
        if pad2square is False:  # now we use this transform function by default
            transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD)
            ])
        else:
            transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Lambda(lambda img: expand2square(img, tuple(int(x * 255) for x in MEAN))),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD)
            ])

    return transform


def preprocess(
        template_name,
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        num_image_token: int,
        text_only: bool = False,
        padding: bool = False,
        truncation: bool = True,
        ds_name: str = None,
        num_image: int = 1
) -> Dict:
    conv = get_conv_template(template_name)
    roles = {'human': conv.roles[0], 'gpt': conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]['from']] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence['from']]
            assert role == conv.roles[j % 2], f'{i}'
            conv.append_message(role, sentence['value'])
        conversations.append(conv.get_prompt())

    image_tokens = f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token}{IMG_END_TOKEN}'
    if not text_only:
        new_conversations = []
        for conversation in conversations:
            conversation = conversation.replace('<image>', image_tokens, num_image)
            new_conversations.append(conversation)
        conversations = new_conversations

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors='pt',
        padding=padding,
        max_length=tokenizer.model_max_length,
        truncation=truncation,
    ).input_ids
    targets = input_ids.clone()

    # assert conv.sep_style == SeparatorStyle.ADD_COLON_TWO

    # Mask targets. Only compute loss on the assistant outputs.
    sep = conv.sep + conv.roles[1] + ': '
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        turns = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID
        for i, turn in enumerate(turns):
            if turn == '':
                break
            turn_len = len(tokenizer(turn).input_ids)

            parts = turn.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            # "-2" is hardcoded for the Llama tokenizer to make the offset correct.
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy:
                # The legacy and non-legacy modes handle special tokens differently
                instruction_len -= 1

            # Ignore the user instructions
            target[cur_len: cur_len + instruction_len] = IGNORE_TOKEN_ID
            cur_len += turn_len

            if i != 0 and not tokenizer.legacy:
                # The legacy and non-legacy modes handle special tokens differently
                cur_len -= 1

        target[cur_len:] = IGNORE_TOKEN_ID

        if False:  # Inspect and check the correctness of masking
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            logger.info(tokenizer.decode(z))
            exit()

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                print(
                    f'WARNING: tokenization mismatch: {cur_len} vs. {total_len}.'
                    f' #turn = {len(turns) - 1}. (ignored). This dataset is {ds_name}.'
                )
                sys.stdout.flush()

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


def preprocess_mpt(
        template_name,
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        num_image_token: int,
        text_only: bool = False,
        padding: bool = False,
        truncation: bool = True,
        ds_name: str = None,
        num_image: int = 1
) -> Dict:
    conv = get_conv_template(template_name)
    roles = {'human': conv.roles[0], 'gpt': conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]['from']] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence['from']]
            assert role == conv.roles[j % 2], f'{i}'
            conv.append_message(role, sentence['value'])
        conversations.append(conv.get_prompt())

    image_tokens = f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token}{IMG_END_TOKEN}'
    if not text_only:
        new_conversations = []
        for conversation in conversations:
            conversation = conversation.replace('<image>', image_tokens, num_image)
            new_conversations.append(conversation)
        conversations = new_conversations

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors='pt',
        padding=padding,
        max_length=tokenizer.model_max_length,
        truncation=truncation,
    ).input_ids
    targets = input_ids.clone()

    # Mask targets. Only compute loss on the assistant outputs.
    sep = conv.sep + conv.roles[1]  # <|im_end|><|im_start|>assistant\n
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        turns = conversation.split(conv.sep)
        re_turns = [conv.sep.join(turns[:3])]  # system + user + gpt
        for conv_idx in range(3, len(turns), 2):
            re_turns.append(conv.sep.join(turns[conv_idx:conv_idx + 2]))  # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_TOKEN_ID
        for i, turn in enumerate(re_turns):
            if turn == '':
                break
            turn_len = len(tokenizer(turn).input_ids) + 1

            parts = turn.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            instruction_len = len(tokenizer(parts[0]).input_ids)

            # Ignore the user instructions
            target[cur_len: cur_len + instruction_len] = IGNORE_TOKEN_ID
            # print(f'[question {i}]', tokenizer.decode(input_ids[:, cur_len: cur_len + instruction_len][0]))
            # print(f'[answer {i}]', tokenizer.decode(input_ids[:, cur_len + instruction_len: cur_len + turn_len][0]))
            # print(f'[label {i}]', target[cur_len + instruction_len: cur_len + turn_len])
            cur_len += turn_len

        target[cur_len:] = IGNORE_TOKEN_ID

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                print(
                    f'WARNING: tokenization mismatch: {cur_len} vs. {total_len}.'
                    f' #turn = {len(turns) - 1}. (ignored). This dataset is {ds_name}.'
                )
                sys.stdout.flush()

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


def preprocess_phi3(
        template_name,
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        num_image_token: int,
        text_only: bool = False,
        padding: bool = False,
        truncation: bool = True,
        ds_name: str = None,
        num_image: int = 1
) -> Dict:
    conv = get_conv_template(template_name)
    roles = {'human': conv.roles[0], 'gpt': conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]['from']] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence['from']]
            assert role == conv.roles[j % 2], f'{i}'
            conv.append_message(role, sentence['value'])
        conversations.append(conv.get_prompt())

    image_tokens = f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token}{IMG_END_TOKEN}'
    if not text_only:
        new_conversations = []
        for conversation in conversations:
            conversation = conversation.replace('<image>', image_tokens, num_image)
            new_conversations.append(conversation)
        conversations = new_conversations

    # Tokenize conversations
    tokenizer.padding_side = 'right'
    input_ids = tokenizer(
        conversations,
        return_tensors='pt',
        padding=padding,
        max_length=tokenizer.model_max_length,
        truncation=truncation,
    ).input_ids
    targets = input_ids.clone()

    # Mask targets. Only compute loss on the assistant outputs.
    sep = conv.sep + conv.roles[1]  # <|end|>\n<|assistant|>
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(int(tokenizer.pad_token_id)).sum())

        turns = conversation.split(conv.sep)
        re_turns = [conv.sep.join(turns[:3])]  # system + user + gpt
        for conv_idx in range(3, len(turns), 2):
            re_turns.append(conv.sep.join(turns[conv_idx:conv_idx + 2]))  # user + gpt
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID
        endoftext_id = tokenizer.convert_tokens_to_ids('<|endoftext|>')
        target[target == endoftext_id] = IGNORE_TOKEN_ID

        for i, turn in enumerate(re_turns):
            if turn == '':
                break
            if i == 0:
                turn_len = len(tokenizer(turn).input_ids)
            else:
                turn_len = len(tokenizer(turn).input_ids) - 1
            parts = turn.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if i == 0:
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1
            else:
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            # Ignore the user instructions
            target[cur_len: cur_len + instruction_len] = IGNORE_TOKEN_ID
            # print(f'[question {i}]', tokenizer.decode(input_ids[:, cur_len: cur_len + instruction_len][0]))
            # print(f'[answer {i}]', tokenizer.decode(input_ids[:, cur_len + instruction_len: cur_len + turn_len][0]))
            # print(f'[label {i}]', target[cur_len + instruction_len: cur_len + turn_len])
            cur_len += turn_len

        target[cur_len:] = IGNORE_TOKEN_ID

        if False:  # Inspect and check the correctness of masking
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            print(repr(tokenizer.decode(z)))

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                print(
                    f'WARNING: tokenization mismatch: {cur_len} vs. {total_len}.'
                    f' #turn = {len(turns) - 1}. (ignored). This dataset is {ds_name}.'
                )
                sys.stdout.flush()

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


def preprocess_llama3(
        template_name,
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        num_image_token: int,
        text_only: bool = False,
        padding: bool = False,
        truncation: bool = True,
        ds_name: str = None,
        num_image: int = 1
) -> Dict:
    conv = get_conv_template(template_name)
    roles = {'human': conv.roles[0], 'gpt': conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]['from']] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence['from']]
            assert role == conv.roles[j % 2], f'{i}'
            conv.append_message(role, sentence['value'])
        conversations.append(conv.get_prompt())

    image_tokens = f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token}{IMG_END_TOKEN}'
    if not text_only:
        new_conversations = []
        for conversation in conversations:
            conversation = conversation.replace('<image>', image_tokens, num_image)
            new_conversations.append(conversation)
        conversations = new_conversations

    # Tokenize conversations
    tokenizer.padding_side = 'right'
    input_ids = tokenizer(
        conversations,
        return_tensors='pt',
        padding=padding,
        max_length=tokenizer.model_max_length,
        truncation=truncation,
    ).input_ids
    targets = input_ids.clone()

    # Mask targets. Only compute loss on the assistant outputs.
    sep = conv.sep + conv.roles[1]  # <|end|>\n<|assistant|>
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        turns = conversation.split(conv.sep)
        re_turns = [conv.sep.join(turns[:3])]  # system + user + gpt
        for conv_idx in range(3, len(turns), 2):
            re_turns.append(conv.sep.join(turns[conv_idx:conv_idx + 2]))  # user + gpt
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID
        for i, turn in enumerate(re_turns):
            if turn == '':
                break
            if i == 0:
                turn_len = len(tokenizer(turn).input_ids)
            else:
                turn_len = len(tokenizer(turn).input_ids) + 1
            parts = turn.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if i == 0:
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1
            else:
                instruction_len = len(tokenizer(parts[0]).input_ids)

            # Ignore the user instructions
            target[cur_len: cur_len + instruction_len] = IGNORE_TOKEN_ID
            # print(f'[question {i}]', tokenizer.decode(input_ids[:, cur_len: cur_len + instruction_len][0]))
            # print(f'[answer {i}]', tokenizer.decode(input_ids[:, cur_len + instruction_len: cur_len + turn_len][0]))
            # print(f'[label {i}]', target[cur_len + instruction_len: cur_len + turn_len])
            cur_len += turn_len

        target[cur_len:] = IGNORE_TOKEN_ID

        if False:  # Inspect and check the correctness of masking
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, int(tokenizer.pad_token_id), z)
            print(repr(tokenizer.decode(z)))

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                print(
                    f'WARNING: tokenization mismatch: {cur_len} vs. {total_len}.'
                    f' #turn = {len(turns) - 1}. (ignored). This dataset is {ds_name}.'
                )
                sys.stdout.flush()

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


def preprocess_internlm(
        template_name,
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        num_patches_per_image: List[int],
        num_image_token: int,
        text_only: bool = False,
        padding: bool = False,
        truncation: bool = True,
        ds_name: str = None,
) -> Dict:
    conv = get_conv_template(template_name)
    roles = {'human': conv.roles[0], 'gpt': conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]['from']] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence['from']]
            assert role == conv.roles[j % 2], f'{i}'
            sentence['value'] = sentence['value'].strip()
            if sentence['value'][0] == '\n':
                sentence['value'] = sentence['value'][1:]
            conv.append_message(role, sentence['value'])
        conversations.append(conv.get_prompt())

    num_image = len(num_patches_per_image)
    if not text_only:
        new_conversations = []
        cur_image_idx = 0
        for conversation in conversations:
            while "<image>" in conversation and cur_image_idx < num_image:
                image_tokens = f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token * num_patches_per_image[cur_image_idx]}{IMG_END_TOKEN}'
                conversation = conversation.replace('<image>', image_tokens, 1)
                cur_image_idx += 1
            new_conversations.append(conversation)
        conversations = new_conversations
    
    if cur_image_idx < num_image:
        print(f'WARNING: tokenization mismatch: {cur_image_idx} vs. {num_image}. This dataset is {ds_name}.')
        sys.stdout.flush()
        # raise ValueError

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors='pt',
        padding=padding,
        max_length=tokenizer.model_max_length,
        truncation=truncation,
    ).input_ids
    targets = input_ids.clone()

    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())  # 浦语里面 pad_token_id = eos_token_id
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID  # <s>
        parts = conversation.split(conv.roles[1])  # [UNUSED_TOKEN_146]assistant\n
        info = parts[0] + conv.roles[1]
        temp_len = len(tokenizer(info).input_ids) - 1  # 去除tokenizer的<s>
        target[cur_len: cur_len + temp_len] = IGNORE_TOKEN_ID
        cur_len = cur_len + temp_len

        for index in range(1, len(parts) - 1):
            info = parts[index]
            part1, part2 = info.split(conv.roles[0])
            temp_len = len(tokenizer(part1).input_ids) - 1
            cur_len = cur_len + temp_len
            part = conv.roles[0] + part2 + conv.roles[1]
            temp_len = len(tokenizer(part).input_ids) - 1
            target[cur_len: cur_len + temp_len] = IGNORE_TOKEN_ID
            cur_len = cur_len + temp_len
        last_info = parts[-1]
        temp_len = len(tokenizer(last_info).input_ids) - 1
        cur_len = cur_len + temp_len

        target[cur_len:] = IGNORE_TOKEN_ID
        if False:  # Inspect and check the correctness of masking
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            print(repr(tokenizer.decode(z)))

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                print(f'WARNING: tokenization mismatch: {cur_len} vs. {total_len}. This dataset is {ds_name}.')
                sys.stdout.flush()

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


def preprocess_internlm_v2(
        template_name,
        conversations,
        tokenizer: transformers.PreTrainedTokenizer,
        num_patches_per_image: List[int],
        num_image_token: int,
        text_only: bool = False,
        padding: bool = False,
        truncation: bool = True,
        ds_name: str = None,
) -> Dict:
    assert not padding
    assert template_name == 'internlm2-chat-v2'
    conversations = conversations[0]

    default_system_system_message = '你是商汤科技开发的日日新多模态大模型，英文名叫SenseChat, 是一个有用无害的人工智能助手。'
    roles = {'human': 'user', 'gpt': 'assistant', 'system': 'system', 'knowledge': 'knowledge'}
    conversation_start = "<|im_start|>"
    conversation_end = "<|im_end|>\n"
    system_template = conversation_start + 'system\n{system_message}' + conversation_end
    num_image = len(num_patches_per_image)
    if not text_only:
        new_conversations = []
        cur_image_idx = 0
        for conv in conversations:
            while "<image>" in conv['value'] and cur_image_idx < num_image:
                image_tokens = f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token * num_patches_per_image[cur_image_idx]}{IMG_END_TOKEN}'
                conv['value'] = conv['value'].replace('<image>', image_tokens, 1)
                cur_image_idx += 1
            new_conversations.append(conv)
        conversations = new_conversations

        if cur_image_idx < num_image:
            print(f'WARNING: image flag mismatch: cur_image_idx {cur_image_idx} vs. num_image {num_image}. This dataset is {ds_name}.')
            sys.stdout.flush()

    # Tokenize conversations
    tokens = []
    labels = []
    # check the first is system,
    if conversations[0]['from'] != 'system':  # default
        tokenized_system = tokenizer(system_template.format(system_message=default_system_system_message),
                                     return_attention_mask=False, add_special_tokens=False)['input_ids']
    else:
        tokenized_system = tokenizer(system_template.format(system_message=conversations[0]['value']),
                                     return_attention_mask=False, add_special_tokens=False)['input_ids']
        conversations = conversations[1:]
    tokens.extend(tokenized_system)
    labels.extend([IGNORE_TOKEN_ID] * len(tokenized_system))
    for i, conv in enumerate(conversations):
        if conv['from'] not in roles:
            print(f"Unknown role, skip. {conv}")
            continue
        role = roles[conv['from']]
        content = conv['value']
        if role == 'user' or role == 'knowledge' or (role == 'assistant' and conv.get('is_input', False)):
            user_info = f"{conversation_start}{role}\n{content}{conversation_end}"
            tokenized_user = tokenizer(user_info, return_attention_mask=False, add_special_tokens=False)['input_ids']
            tokens.extend(tokenized_user)
            labels.extend([IGNORE_TOKEN_ID] * len(tokenized_user))
        elif role == 'assistant':
            assis_start = f"{conversation_start}{role}\n"
            tokens_assistant_start = tokenizer(assis_start, return_attention_mask=False, add_special_tokens=False)[
                'input_ids']
            tokens.extend(tokens_assistant_start)
            labels.extend([IGNORE_TOKEN_ID] * len(tokens_assistant_start))
            assis_info = f"{content}{conversation_end}"
            tokenized_assistant = tokenizer(assis_info, return_attention_mask=False, add_special_tokens=False)[
                'input_ids']
            tokens.extend(tokenized_assistant)
            labels.extend(copy.deepcopy(tokenized_assistant))
        else:
            print(f"Not processed role, skip. {conv}")
    if truncation and len(tokens) > tokenizer.model_max_length:
        tokens = tokens[:tokenizer.model_max_length]
        labels = labels[:tokenizer.model_max_length]
    input_ids = torch.LongTensor([tokens])  # [1,N] to match the before
    targets = torch.LongTensor([labels])  # [1,N] to match the before
    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


def preprocess_internlm_v3(
        template_name,
        conversations,
        tokenizer: transformers.PreTrainedTokenizer,
        num_patches_per_image: List[int],
        num_image_token: int,
        text_only: bool = False,
        padding: bool = False,
        truncation: bool = True,
        ds_name: str = None,
) -> Dict:
    assert not padding
    assert template_name == 'internlm2-chat-v3'
    conversations = conversations[0]

    default_system_system_message = ''
    roles = {'human': 'user', 'gpt': 'assistant', 'system': 'system', 'knowledge': 'knowledge'}
    conversation_start = "<|im_start|>"
    conversation_end = "<|im_end|>\n"
    system_template = conversation_start + 'system\n{system_message}' + conversation_end
    num_image = len(num_patches_per_image)
    if not text_only:
        new_conversations = []
        cur_image_idx = 0
        for conv in conversations:
            while "<image>" in conv['value'] and cur_image_idx < num_image:
                image_tokens = f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token * num_patches_per_image[cur_image_idx]}{IMG_END_TOKEN}'
                conv['value'] = conv['value'].replace('<image>', image_tokens, 1)
                cur_image_idx += 1
            new_conversations.append(conv)
        conversations = new_conversations

        if cur_image_idx < num_image:
            print(f'WARNING: image flag mismatch: cur_image_idx {cur_image_idx} vs. num_image {num_image}. This dataset is {ds_name}.')
            sys.stdout.flush()

    # Tokenize conversations
    tokens = []
    labels = []
    # check the first is system,
    if conversations[0]['from'] != 'system':  # default
        pass
    else:
        tokenized_system = tokenizer(system_template.format(system_message=conversations[0]['value']),
                                     return_attention_mask=False, add_special_tokens=False)['input_ids']
        conversations = conversations[1:]
        tokens.extend(tokenized_system)
        labels.extend([IGNORE_TOKEN_ID] * len(tokenized_system))
    for i, conv in enumerate(conversations):
        if conv['from'] not in roles:
            print(f"Unknown role, skip. {conv}")
            continue
        role = roles[conv['from']]
        content = conv['value']
        if role == 'user' or role == 'knowledge' or (role == 'assistant' and conv.get('is_input', False)):
            user_info = f"{conversation_start}{role}\n{content}{conversation_end}"
            tokenized_user = tokenizer(user_info, return_attention_mask=False, add_special_tokens=False)['input_ids']
            tokens.extend(tokenized_user)
            labels.extend([IGNORE_TOKEN_ID] * len(tokenized_user))
        elif role == 'assistant':
            assis_start = f"{conversation_start}{role}\n"
            tokens_assistant_start = tokenizer(assis_start, return_attention_mask=False, add_special_tokens=False)[
                'input_ids']
            tokens.extend(tokens_assistant_start)
            labels.extend([IGNORE_TOKEN_ID] * len(tokens_assistant_start))
            assis_info = f"{content}{conversation_end}"
            tokenized_assistant = tokenizer(assis_info, return_attention_mask=False, add_special_tokens=False)[
                'input_ids']
            tokens.extend(tokenized_assistant)
            labels.extend(copy.deepcopy(tokenized_assistant))
        else:
            print(f"Not processed role, skip. {conv}")
    if truncation and len(tokens) > tokenizer.model_max_length:
        tokens = tokens[:tokenizer.model_max_length]
        labels = labels[:tokenizer.model_max_length]
    input_ids = torch.LongTensor([tokens])  # [1,N] to match the before
    targets = torch.LongTensor([labels])  # [1,N] to match the before
    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    # print(f'width: {width}, height: {height}, best_ratio: {best_ratio}')
    return best_ratio


def find_minimal_patch_aspect_ratio(target_ratios, orig_width, orig_height, image_size, scale_threshold=1.0):
    max_gain = float('-inf')
    for ratio in target_ratios:
        scale_factor = min(
            ratio[0] * image_size / orig_width,
            ratio[1] * image_size / orig_height,
        )
        gain = min(scale_factor, scale_threshold)
        if gain > max_gain:
            max_gain = gain
            best_scale_factor = scale_factor
            best_ratio = ratio

    return best_ratio, best_scale_factor


def change_coordinate(input_str, new_width, new_height, target_width, target_height):

    def modify_bbox(match):

        def validate_and_modify(match):
            bbox = eval(match.group(0))
            assert len(bbox) == 4
            if not all(0<=num<=1000 for num in bbox):
                print('WARNING: box larger than 1000 or smaller then 0')
                bbox  = [max(x, 0) for x in bbox]
                bbox  = [min(x, 999) for x in bbox]
            if new_width == target_width:
                bbox[1] = (bbox[1]-500)*new_height/target_height + 500
                bbox[3] = (bbox[3]-500)*new_height/target_height + 500
            elif new_height == target_height:
                bbox[0] = (bbox[0]-500)*new_width/target_width + 500
                bbox[2] = (bbox[2]-500)*new_width/target_width + 500
            return f'[{", ".join(str(round(num)) for num in bbox)}]'

        bbox_list_str = match.group(1)
        return "<box>["+re.sub(r'\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]', validate_and_modify, bbox_list_str)+"]</box>"

    output_str = re.sub(r"<box>\[(.*?)\]</box>", modify_bbox, input_str)

    return output_str



def dynamic_preprocess(image, data_item, min_num=1, max_num=6, image_size=448, use_thumbnail=False, normalize_type='imagenet', scale_threshold=1.0):
    orig_width, orig_height = image.size

    # calculate the existing image aspect ratio
    target_ratios = list(
        (i, j) for i in range(1, max_num + 1) for j in range(1, max_num + 1) if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio, target_scale_factor = find_minimal_patch_aspect_ratio(
        target_ratios, orig_width, orig_height, image_size, scale_threshold=scale_threshold)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize and pad the image
    new_width, new_height = round(orig_width*target_scale_factor), round(orig_height*target_scale_factor)
    resized_img = image.resize((new_width, new_height))

    if normalize_type == 'imagenet':
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    elif normalize_type == 'clip':
        MEAN, STD = CLIP_MEAN, CLIP_STD
    elif normalize_type == 'siglip':
        MEAN, STD = SIGLIP_MEAN, SIGLIP_STD
    else:
        raise NotImplementedError

    background_color = tuple(int(x*255) for x in MEAN)
    padded_resized_img = Image.new(resized_img.mode, (target_width, target_height), background_color)
    if new_width == target_width:
        padded_resized_img.paste(resized_img, (0, (target_height-new_height)//2))
    elif new_height == target_height:
        padded_resized_img.paste(resized_img, ((target_width-new_width)//2, 0))
    else:
        print(
            f"resize image error: target_width: {target_width}, target_height: {target_height},"
            f"new_width: {new_width}, new_height: {new_height}, target_aspect_ratio: {target_aspect_ratio}"
        )
        raise Exception("Resize Image Error")

    # change bbox coordinates according to the image transform
    if data_item:
        for conversation in data_item['conversations']:
            conversation['value'] = change_coordinate(conversation['value'], new_width, new_height, target_width, target_height)

    assert padded_resized_img.size == (target_width, target_height)

    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = padded_resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def dynamic_preprocess_old(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def dynamic_preprocess_v1(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    max_patch = np.ceil(orig_width * orig_height * 2 * 2 / image_size / image_size)

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num and i * j <= max_patch)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def find_closest_aspect_ratio_v2(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    max_patch = np.ceil(orig_width * orig_height * 2 / image_size / image_size)

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num and i * j <= max_patch)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    return target_aspect_ratio

def dynamic_preprocess_v2(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    target_aspect_ratio = find_closest_aspect_ratio_v2(image, min_num, max_num, image_size, use_thumbnail)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def find_closest_aspect_ratio_v3(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    # llava like, from https://huggingface.co/openbmb/MiniCPM-V-2_6/blob/3f7a8da1b7a8b928b5ee229fae33cf43fd64cf31/image_processing_minicpmv.py#L257 with modification 
    assert min_num == 1
    original_width, original_height = image.size
    log_ratio = math.log(original_width / original_height)
    ratio = original_width * original_height / (image_size * image_size)
    multiple = min(math.ceil(ratio), max_num)
    if multiple <= 1:
        return [1, 1]
    candidate_split_grids_nums = []
    for i in [multiple - 1, multiple, multiple + 1]:
        if i > max_num:
            continue
        candidate_split_grids_nums.append(i)
    
    candidate_grids = []
    for split_grids_nums in candidate_split_grids_nums:
        m = 1
        while m <= split_grids_nums:
            if split_grids_nums % m == 0:
                candidate_grids.append([m, split_grids_nums // m])
            m += 1
    best_grid = [1, 1]
    min_error = float("inf")
    for grid in candidate_grids:
        error = abs(log_ratio - math.log(grid[0] / grid[1]))
        if error < min_error:
            best_grid = grid
            min_error = error

    return best_grid

def dynamic_preprocess_v3(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    target_aspect_ratio = find_closest_aspect_ratio_v3(image, min_num, max_num, image_size, use_thumbnail)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

class LazySupervisedDataset(Dataset):
    default_seed = 42
    """Dataset for supervised fine-tuning."""

    def __init__(self, template_name, meta, tokenizer, tcs_loader, num_image_token, ds_name,
                 image_size=224, is_train=True, pad2square=False, group_by_length=False,
                 dynamic_image_size=False, use_thumbnail=False, min_dynamic_patch=1,
                 max_dynamic_patch=6, repeat_time=1, normalize_type='imagenet', scale_threshold="old",
                 is_token_counting=False, read_image=True, max_num_image=32, min_num_image=4, sampling_method="rand", is_chinese=False, fix_seed=False,
                 max_dynamic_images = 6, max_multi_image_dynamic_patch = 6, max_patches = 42):
        super(LazySupervisedDataset, self).__init__()
        assert not pad2square  # should nenver be used
        assert min_dynamic_patch==1
        assert use_thumbnail

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
        if self.group_by_length:
            if self.meta_data is not None:
                self.length = [x['token_num'] for x in self.meta_data]
            else:
                self.conv2length = {}  # using dict to speedup the calculation of token length
                self.length = []
                for data_item in self.raw_data:
                    data_item = json.loads(data_item)
                    if 'length' in data_item:
                        token_length = data_item['length']  # use precomputed length if exists
                    else:
                        # compute token length using tokenizer
                        conversations = '\n'.join([temp['value'] for temp in data_item['conversations']])
                        str_length = len(conversations)
                        if str_length not in self.conv2length:
                            token_length = tokenizer(
                                conversations, return_tensors='pt', padding=False, truncation=False,
                            ).input_ids.size(1)
                            self.conv2length[str_length] = token_length + num_image_token * (
                                        max_dynamic_patch + use_thumbnail)
                        else:
                            token_length = self.conv2length[str_length]
                    self.length.append(token_length)
        self.max_num_image = max_num_image
        self.min_num_image = min_num_image 
        self.sampling_method = sampling_method
        self.is_chinese = is_chinese
        self.fix_seed = fix_seed

        if self.fix_seed:
            random.seed(self.default_seed)
            np.random.seed(self.default_seed)
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

    def video_get_item(self, data_item, i):
        for i, conv in enumerate(data_item['conversations']):
            if conv['from'] == 'human':
                if '<video>' not in conv['value']:
                    data_item['conversations'][i]['value'] = '<video>\n' + data_item['conversations'][i]['value']
                break
        video_file = data_item['video']
        video_path = os.path.join(self.root, video_file)
        transform = build_transform(is_train=self.is_train, input_size=self.image_size,
                                    pad2square=self.pad2square, normalize_type=self.normalize_type)
    

        random.seed(self.default_seed+len(data_item['conversations'][0]['value']))
        np.random.seed(self.default_seed+len(data_item['conversations'][0]['value']))
        clip = data_item.get("clip", None)
        if self.read_image:
            image_list = self.tcs_loader(video_path, 
                                        image_type='video', 
                                        max_num_frames=self.max_num_image, 
                                        min_num_frames=self.min_num_image, 
                                        sample=self.sampling_method, 
                                        clip=clip)
        else:
            image_list = [Image.new('RGB', (data_item['width'][i], data_item['height'][i]), (255, 255, 255)) for i in range(len(data_item['height']))]  # w,h
        # shuffle random add frame id 

        special_tokens = "\n".join(["Frame{}:<image>".format(i+1) for i in range(len(image_list))])
        
        data_item['conversations'][0]['value'] = data_item['conversations'][0]['value'].replace("<video>\n", special_tokens)
        raw_images = []
        num_patches_per_image = []
        pixel_values = []

        if self.read_image:
            for image in image_list:
                raw_images.append(image)
                if self.dynamic_image_size and len(image_list) <= self.max_dynamic_images:
                    if self.scale_threshold == "v2":
                        patches = dynamic_preprocess_v2(image, min_num=self.min_dynamic_patch, max_num=self.max_dynamic_patch if len(image_list) == 1 else self.max_multi_image_dynamic_patch,
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
        if self.template_name == 'internlm2-chat':
            preprocess_function = preprocess_internlm
        elif self.template_name == 'internlm2-chat-v2':
            preprocess_function = preprocess_internlm_v2
        elif self.template_name == 'internlm2-chat-v3':
            preprocess_function = preprocess_internlm_v3
        else:
            raise NotImplementedError

        if self.is_token_counting:
            ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                            self.tokenizer, num_patches_per_image, self.num_image_token, ds_name=self.ds_name,
                            truncation=False)
            ret = dict(
                input_ids=ret['input_ids'][0],
                labels=ret['labels'][0],
                attention_mask=ret['attention_mask'][0],
                num_patches=num_patches,
                image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
                height=raw_images[0].height if len(raw_images) == 1 else [image.height for image in raw_images],
                width=raw_images[0].width if len(raw_images) == 1 else [image.width for image in raw_images],
            )
        else:
            ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                            self.tokenizer, num_patches_per_image, self.num_image_token, ds_name=self.ds_name,
                            truncation=True)
            ret = dict(
                input_ids=ret['input_ids'][0],
                labels=ret['labels'][0],
                attention_mask=ret['attention_mask'][0],
                pixel_values=pixel_values,
                image_flags=torch.tensor([1] * num_patches, dtype=torch.long)
            )
        return ret
    
    
    def multi_modal_get_item(self, data_item):
        for i, conv in enumerate(data_item['conversations']):
            if conv['from'] == 'human':
                if '<image>' not in conv['value']:
                    data_item['conversations'][i]['value'] = '<image>\n' + data_item['conversations'][i]['value']
                break

        # single image 
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
        if self.is_token_counting:
            ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                            self.tokenizer, num_patches_per_image, self.num_image_token, ds_name=self.ds_name,
                            truncation=False)
            ret = dict(
                input_ids=ret['input_ids'][0],
                labels=ret['labels'][0],
                attention_mask=ret['attention_mask'][0],
                num_patches=num_patches,
                image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
                height=raw_images[0].height if len(raw_images) == 1 else [image.height for image in raw_images],
                width=raw_images[0].width if len(raw_images) == 1 else [image.width for image in raw_images],
            )
        else:
            ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                            self.tokenizer, num_patches_per_image, self.num_image_token, ds_name=self.ds_name,
                            truncation=True)
            ret = dict(
                input_ids=ret['input_ids'][0],
                labels=ret['labels'][0],
                attention_mask=ret['attention_mask'][0],
                pixel_values=pixel_values,
                image_flags=torch.tensor([1] * num_patches, dtype=torch.long)
            )
        return ret

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

        if self.is_token_counting:
            ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                            self.tokenizer, [], self.num_image_token, ds_name=self.ds_name,
                            truncation=False)
            ret = dict(
                input_ids=ret['input_ids'][0],
                labels=ret['labels'][0],
                attention_mask=ret['attention_mask'][0],
                num_patches=num_patches,
                image_flags=torch.tensor([0] * num_patches, dtype=torch.long)
            )
        else:
            ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                            self.tokenizer, [], self.num_image_token, ds_name=self.ds_name,
                            truncation=True)
            ret = dict(
                input_ids=ret['input_ids'][0],
                labels=ret['labels'][0],
                attention_mask=ret['attention_mask'][0],
                pixel_values=pixel_values,
                image_flags=torch.tensor([0] * num_patches, dtype=torch.long)
            )
        return ret

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        assert i < len(self.raw_data)

        # i = i % len(self.raw_data)
        while True:
            line_start = self.raw_data[i]
            line_end = self.mmap.find(b'\n', line_start)
            if line_end == -1:
                line_end = self.mmap.size()
            data = self.mmap[line_start:line_end].decode('utf-8')
            data_item = json.loads(data)
            try:
                if 'image' in data_item and data_item['image'] is not None and data_item['image'] != "":
                    ret = self.multi_modal_get_item(data_item)
                elif 'video' in data_item and data_item['video'] is not None and data_item['video'] != "":
                    ret = self.video_get_item(data_item, i)
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
        ret['ds_name'] = self.ds_name  # for debug
        if ret['image_flags'].sum() * self.num_image_token != (ret['input_ids'] == self.img_context_token_id).sum():
            print(f"Image_flags mismatch {self.ds_name} {i} {data_item}")
            sys.stdout.flush()
        return ret


def get_token_sum(g):
    sum = 0
    for i in g:
        sum += i[2]
    return sum


def get_vit_num(g):
    vit_num = 0
    for _ in g:
        vit_num += _[1]
    return vit_num


class InternPackedDataset(Dataset):
    def __init__(self,
                 dataset,
                 vit_packed_length,
                 llm_packed_length,
                 loss_reduction,
                 tokenizer,
                 num_image_token=256,
                 iter_time=1):
        self.dataset = dataset
        self.vit_packed_length = vit_packed_length
        self.llm_packed_length = llm_packed_length
        self.loss_reduction = loss_reduction
        self.num_image_token = num_image_token
        self.img_context_token_id =  tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.llm_thresh = llm_packed_length - 128

        self.vit_lengths, self.llm_lengths = [], []
        self.iter_time = iter_time
        self.seed = 233
        logger.info("Begin preprocess dataset")
        self.preprocess()
        logger.info("Preprocess dataset successed")
        self.pack_groups = self.get_packed_groups()
        del self.dict_num_tokens
        for ds in self.dataset.datasets:
            del ds.meta_data
        gc.collect()

    def preprocess(self):
        dict_num_tokens = {}
        num_datasets = len(self.dataset.datasets)
        for idx in range(num_datasets):
            sub_dataset = self.dataset.datasets[idx]
            assert len(sub_dataset.meta_data) > 0, f"Dataset {idx} meta info does not exist."
            dict_num_tokens[idx] = {
                "lengths": len(sub_dataset),
                "token_lengths": sub_dataset.meta_data
            }
        self.dict_num_tokens = dict_num_tokens

    def _random_groups(self, token_lengths, seed=None):
        """
        tokens_length: [(idx, vit_img_num, llm_token_len)]
        """
        rng = np.random.RandomState(seed)
        index = list(range(len(token_lengths)))
        rng.shuffle(index)

        pack_groups = []
        vit_token_length_sum, llm_token_length_sum = 0, 0
        each_group = []
        for idx, sample_id in enumerate(index):
            vit_sample_length, llm_sample_length = token_lengths[sample_id][1], token_lengths[sample_id][2]
            if vit_sample_length > self.vit_packed_length or llm_sample_length > self.llm_packed_length:
                continue
            vit_token_length_sum += vit_sample_length
            llm_token_length_sum += llm_sample_length
            if vit_token_length_sum > self.vit_packed_length or llm_token_length_sum > self.llm_packed_length:
                pack_groups.append(each_group)
                vit_token_length_sum = vit_sample_length
                llm_token_length_sum = llm_sample_length
                each_group = [token_lengths[sample_id]]
            else:
                each_group.append(token_lengths[sample_id])
            if idx == len(token_lengths) - 1:
                if len(each_group) > 0:
                    pack_groups.append(each_group)
        return pack_groups

    def process_random_groups_input(self, groups, accu_length=0):
        new_groups = []
        for idx, item in enumerate(groups):
            if item["vit_num"] == -1:
                logger.info(f"item {idx} was filted.")
                continue
            new_groups.append((idx + accu_length, item['vit_num'], item['token_num']))
        return new_groups

    def iter_random_groups(self, groups, llm_thresh=None, seed=None, iter_time=300):
        if llm_thresh is None:
            llm_thresh = self.llm_packed_length
        if seed is None:
            seed = self.seed
        groups = self._random_groups(groups, seed=seed)
        if iter_time == 1:
            return groups
        output = []
        for i in range(iter_time - 1):
            logger.info(f"iter_random_groups {i} / {iter_time - 1}")
            need_process_groups = []
            for g in groups:
                vit_num = get_vit_num(g)
                llm_num = get_token_sum(g)
                if vit_num == self.vit_packed_length or llm_num >= llm_thresh:
                    output.append(g)
                else:
                    need_process_groups.extend(g)
            if len(need_process_groups) >= 0:
                groups = self._random_groups(need_process_groups, seed + i)
            else:
                break
        if len(need_process_groups) > 0:
            output.extend(self._random_groups(need_process_groups, seed + i))
        return output

    def collect_packed_info(self, packed_groups):
        info_dict = {}
        info_dict['vit_num_info'] = {}
        vit_num_min = 10000000
        vit_num_max = 0
        llm_num_min = 10000000
        llm_num_max = 0
        vit_ave_num = 0
        llm_ave_num = 0
        sample_num = 0
        for group in packed_groups:
            vit_num = get_vit_num(group)
            llm_num = get_token_sum(group)
            if vit_num not in info_dict['vit_num_info']:
                info_dict['vit_num_info'][vit_num] = 0
            info_dict['vit_num_info'][vit_num] += 1
            vit_num_min = min(vit_num_min, vit_num)
            vit_num_max = max(vit_num_max, vit_num)
            llm_num_min = min(llm_num_min, llm_num)
            llm_num_max = max(llm_num_max, llm_num)
            vit_ave_num += vit_num
            llm_ave_num += llm_num
            sample_num += len(group)
        info_dict['vit_num_min'] = vit_num_min
        info_dict['vit_num_max'] = vit_num_max
        info_dict['vit_ave_num'] = vit_ave_num / float(len(packed_groups))
        info_dict['llm_ave_num'] = llm_ave_num / float(len(packed_groups))
        info_dict['sample_num'] = sample_num
        info_dict['packed_group_num'] = len(packed_groups)
        return info_dict

    def get_packed_groups(self):
        num_datasets = len(self.dataset.datasets)
        accu_length = 0
        input_groups = []
        for d_idx in range(num_datasets):
            dict_item = self.dict_num_tokens[d_idx]
            token_lengths = dict_item["token_lengths"]
            groups = self.process_random_groups_input(token_lengths, accu_length)
            logger.info(f"get_packed_groups {d_idx}.")
            input_groups.extend(groups)
            accu_length += len(token_lengths)

        groups = self.iter_random_groups(input_groups, llm_thresh=self.llm_thresh,
                                         seed=self.seed, iter_time=self.iter_time)

        print(self.collect_packed_info(groups), flush=True)
        logger.info("get_packed_groups done!")
        return groups

    def __getitem__(self, item: int):
        item = item % len(self.pack_groups)
        # print("InternPackedDataset Start", item, self.pack_groups[item])
        # sys.stdout.flush()
        while True:
            try:
                groups = self.pack_groups[item]
                input_ids, pixel_values = [], []
                labels, position_ids, image_flags = [], [], []
                cu_seqlens = [0]
                loss_weight = []
                ds_name = []
                for g in groups:
                    idx, num_patches, llm_length = g
                    meta = self.dataset.__getitem__(idx)
                    ds_name.append(meta['ds_name'])
                    # if len(meta["input_ids"]) != llm_length:
                    #     print(f"Length mismatch {item} {groups} {g} {meta['ds_name']} {len(meta['input_ids'])}, {llm_length}")
                    #     sys.stdout.flush()
                    # if meta["pixel_values"].size(0) != num_patches:
                    #     print(f"Patch mismatch {item} {groups} {g} {meta['ds_name']} {meta['pixel_values'].size(0)}, {num_patches}")
                    #     sys.stdout.flush()
                    input_ids.append(meta['input_ids'])
                    pixel_values.append(meta['pixel_values'])
                    labels.append(meta['labels'])
                    cu_seqlens.append(len(meta['input_ids']))
                    position_ids.extend(list(range(len(meta['input_ids']))))
                    image_flags.append(meta.get('image_flags', torch.tensor([0], dtype=torch.long)))

                    num_loss_token = (meta['labels']!= IGNORE_TOKEN_ID).sum()
                    if self.loss_reduction=='sqrt':
                        loss_weight.extend([1./math.sqrt(num_loss_token) if x!=IGNORE_TOKEN_ID else 0 for x in meta['labels']])
                    elif self.loss_reduction=='sample':
                        loss_weight.extend([1./num_loss_token if x!=IGNORE_TOKEN_ID else 0 for x in meta['labels']])
                    elif self.loss_reduction=='token':
                        loss_weight.extend([1. if x!=IGNORE_TOKEN_ID else 0 for x in meta['labels']])
                    else:
                        raise NotImplementedError
                    
                cu_seqlens = np.cumsum(np.array(cu_seqlens)).tolist()
                input_ids = torch.cat(input_ids)
                pixel_values = torch.cat(pixel_values)

                labels = torch.cat(labels)
                cu_seqlens = torch.IntTensor(cu_seqlens)
                position_ids = torch.LongTensor(position_ids)
                # loss_weight = torch.FloatTensor(loss_weight)
                image_flags = torch.cat(image_flags)

                assert len(loss_weight) == len(input_ids)
                assert cu_seqlens[-1] == input_ids.size(0), f"Cu_seqlens mismatch, {item} {groups} cu_seqlens: {cu_seqlens}, input_ids {input_ids.size(0)}"
                assert cu_seqlens[-1] <= self.llm_packed_length + 128 and pixel_values.size(0) <= self.vit_packed_length, f"Too Long sample, {item} {groups} Length {input_ids.size(0)} Patch {image_flags.shape[0]}, Cu_seqlens {cu_seqlens}"
                assert image_flags.shape[0] == pixel_values.shape[0], f"Flags mismatch, {item} {groups} Length {len(meta['input_ids'])}, {llm_length} Patch {meta['pixel_values'].size(0)}, {num_patches} Flags {image_flags.shape[0]} Patch {pixel_values.shape[0]}"
                ret = {
                    "input_ids": input_ids,
                    "labels": labels,
                    "cu_seqlens": cu_seqlens,
                    "position_ids": position_ids,
                    "pixel_values": pixel_values,
                    "image_flags": image_flags,
                    'loss_weight': loss_weight,
                }
                if (image_flags==1).sum()*self.num_image_token != (input_ids==self.img_context_token_id).sum():
                    print(f"Image_flags mismatch {item} {groups} {ds_name} {meta['pixel_values'].size(0)}, {num_patches}")
                    sys.stdout.flush()
                # logger.info(f"Sample patch: {pixel_values.size(0)} length: {cu_seqlens[-1]}")
                # print("InternPackedDataset Finish", item, groups)
                # sys.stdout.flush()
                break
            except Exception as e:
                print(f"{e} Fail to load data, {item}, {groups}")
                # i = random.randint(0, len(self.raw_data) - 1)
                item = (item + 100) % len(self.pack_groups)
        return ret

    def __len__(self):
        n_packs = len(self.pack_groups)
        return n_packs


def build_datasets(data_args, tokenizer, tcs_loader, model, group_by_length=False,
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
            dataset = LazySupervisedDataset(
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

    if data_args.pack_dataset:
        train_dataset = InternPackedDataset(
            train_dataset,
            num_image_token=model.num_image_token,
            vit_packed_length=data_args.vit_packed_length,
            llm_packed_length=data_args.llm_packed_length,
            loss_reduction=data_args.loss_reduction,
            tokenizer=tokenizer,
            iter_time=data_args.iter_time,
        )
    return train_dataset
