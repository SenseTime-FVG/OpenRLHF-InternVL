import json
import logging
import os
import random
import sys
import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.distributed as dist
import torchvision.transforms as T
import transformers
from internvl.dist_utils import init_dist
from internvl.model.internvl_distillation import (InternVisionModel,
                                                  InternVLDistillation,
                                                  InternVLDistillationConfig)
from internvl.train.constants import IMAGENET_MEAN, IMAGENET_STD
from internvl.train.dataset import ConcatDataset, TCSLoader, build_transform
from internvl.train.trainer_monkey_patch import replace_create_optimizer
from PIL import Image, ImageFile, PngImagePlugin
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode
from transformers import (HfArgumentParser, Trainer, TrainingArguments,
                          default_data_collator, set_seed)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.logging import (enable_default_handler,
                                        enable_explicit_format, set_verbosity)

try:
    from petrel_client.client import Client
    from petrel_client.common.config import Config
    has_tcs_loader = True
except ImportError as E:
    print('petrel_client is not installed. Using PIL to load images.')
    has_tcs_loader = False

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    student_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    teacher_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    vision_select_layer: int = field(
        default=-1,
        metadata={'help': 'Specify the layer of ViT feature map to use. Default is last layer.'},
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


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    force_image_size: Optional[int] = field(
        default=448,
        metadata={'help': 'Set the desired size for the image. Default is 448.'},
    )
    meta_path: Optional[str] = field(
        default=None,
        metadata={'help': 'The path of the meta file of datasets.'},
    )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, meta, tcs_loader, image_size=224, is_train=False):
        super(LazySupervisedDataset, self).__init__()
        logger.info(f'[Dataset] image_size: {image_size}')

        self.image_size = image_size
        self.is_train = is_train
        logger.info('Formatting inputs...Skip in lazy mode')
        total_ranks = torch.distributed.get_world_size()
        current_rank = torch.distributed.get_rank()
        basename = os.path.basename(meta['annotation']).replace('.jsonl', '')
        data_dir = os.path.join(os.path.dirname(meta['annotation']), f'{basename}_temp')
        if not os.path.exists(data_dir):
            try:
                os.makedirs(data_dir)
            except:
                pass
        temp_path = os.path.join(data_dir, f'{basename}_{current_rank}_of_{total_ranks}.jsonl')
        if os.path.exists(temp_path):
            with open(temp_path, 'r') as f:
                self.raw_data = f.readlines()
        else:
            with open(meta['annotation'], 'r') as f:
                self.raw_data = f.readlines()
            total_lines = len(self.raw_data)
            logger.info(f'total_ranks: {total_ranks}, current_rank: {current_rank}, total_lines: {total_lines}')
            lines_per_rank = total_lines // total_ranks  # 每个rank分得的行数
            start_line = lines_per_rank * current_rank  # 当前rank开始的行数
            end_line = start_line + lines_per_rank  # 当前rank结束的行数
            self.raw_data = self.raw_data[start_line:end_line]  # 读取当前rank对应的行
            writer = open(temp_path, 'w')
            writer.writelines(self.raw_data)
            writer.close()
        random.shuffle(self.raw_data)

        self.root = meta['root']
        self.tcs_loader = tcs_loader

    def __len__(self):
        return len(self.raw_data) * torch.distributed.get_world_size()

    def multi_modal_get_item(self, data_item):
        if data_item['image'].startswith('s3://'):
            image_path = self.root + data_item['image']
        else:
            image_path = os.path.join(self.root, data_item['image'])
        if self.tcs_loader is not None:
            image = self.tcs_loader(image_path)
        else:
            image = Image.open(image_path).convert('RGB')

        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.RandomResizedCrop(self.image_size, scale=(0.8, 1.0), ratio=(3. / 4., 4. / 3.),
                                interpolation=InterpolationMode.BICUBIC),
            # T.Resize((self.input_size, self.input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        pixel_values = transform(image)

        return dict(
            pixel_values=pixel_values,
        )

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        i = i % len(self.raw_data)
        while True:
            try:
                data_item = json.loads(self.raw_data[i])
                ret = self.multi_modal_get_item(data_item)
                break
            except Exception as e:
                print(e)
                data_item = json.loads(self.raw_data[i])
                if 'image' in data_item:
                    if data_item['image'].startswith('s3://'):
                        data_path = self.root + data_item['image']
                    else:
                        data_path = os.path.join(self.root, data_item['image'])
                    print(f'Failed to load image: {data_path}, the dataset is: {self.ds_name}')
                i = random.randint(0, len(self.raw_data) - 1)
        return ret


def build_datasets(data_args, tcs_loader):
    datasets = []
    ds_collections = json.loads(open(data_args.meta_path).read())
    for ds_name in ds_collections.keys():
        repeat_time = ds_collections[ds_name]['repeat_time']
        # try:
        dataset = LazySupervisedDataset(
            ds_collections[ds_name],
            tcs_loader,
            image_size=data_args.force_image_size,
            is_train=ds_collections[ds_name]['data_augment'],
        )
        # except Exception:
        #     logger.info(f'Error in loading dataset: {ds_name}')
        #     exit()
        dataset.ds_name = ds_name
        for i in range(repeat_time):
            logger.info(f'Add dataset:{ds_name}_{i} with length: {len(dataset)}')
            datasets.append(dataset)
        train_dataset = ConcatDataset(datasets)
    return train_dataset


def main():
    # Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # If use DeepSpeed zero3, init_dist must before HfArgumentParser
    launcher = os.environ.get('LAUNCHER', 'slurm')
    init_dist(launcher=launcher, backend='nccl')
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry('InternV-Chat', model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    set_verbosity(log_level)
    enable_default_handler()
    enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f'Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}'
        + f'distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}'
    )
    logger.info(f'Training/evaluation parameters {training_args}')

    # Detecting last checkpoint and eventually continue from last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f'Output directory ({training_args.output_dir}) already exists and is not empty. '
                'Use --overwrite_output_dir to overcome.'
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f'Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change '
                'the `--output_dir` or add `--overwrite_output_dir` to train from scratch.'
            )
    # Set seed before initializing model.
    set_seed(training_args.seed)

    tcs_loader = TCSLoader('~/petreloss.conf') if has_tcs_loader else None

    if model_args.model_name_or_path is not None:
        logger.info('Loading InternVLDistillation...')
        config = InternVLDistillationConfig.from_pretrained(model_args.model_name_or_path)
        config.student_config.drop_path_rate = model_args.drop_path_rate
        config.select_layer = model_args.vision_select_layer
        model = InternVLDistillation.from_pretrained(
            model_args.model_name_or_path, torch_dtype=torch.bfloat16, config=config)
    else:
        logger.info('Loading Teacher...')
        teacher_model = InternVisionModel.from_pretrained(
            model_args.teacher_path, torch_dtype=torch.bfloat16)
        logger.info('Loading Student...')
        student_model = InternVisionModel.from_pretrained(
            model_args.student_path, torch_dtype=torch.bfloat16)
        logger.info('Building InternVLDistillationConfig...')
        internvl_distillation_config = InternVLDistillationConfig(
            teacher_model.config.to_dict(),
            student_model.config.to_dict(),
            select_layer=model_args.vision_select_layer,
            force_image_size=data_args.force_image_size)
        internvl_distillation_config.force_image_size = data_args.force_image_size
        logger.info('Building InternVLDistillation...')
        model = InternVLDistillation(internvl_distillation_config, teacher_model, student_model)

    # patch_size = model.config.vision_config.patch_size
    # if model.config.force_image_size != data_args.force_image_size and \
    #         model.config.vision_config.image_size != data_args.force_image_size:
    #     model.vision_model.resize_pos_embeddings(old_size=model.config.vision_config.image_size,
    #                                              new_size=data_args.force_image_size,
    #                                              patch_size=patch_size)
    #     model.config.vision_config.image_size = data_args.force_image_size
    # model.config.force_image_size = data_args.force_image_size

    if model_args.grad_checkpoint:
        model.student_model.gradient_checkpointing = True
        model.student_model.encoder.gradient_checkpointing = True

    train_dataset = build_datasets(data_args, tcs_loader)

    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False

    model.teacher_model = model.teacher_model.eval()
    _freeze_params(model.teacher_model)

    # print trainable parameters
    if dist.get_rank() == 0:
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(name)

    # set seed for torch dataloaders
    set_seed(training_args.seed)

    # Initialize our Trainer
    if model_args.use_custom_trainer:
        replace_create_optimizer()

    # do we need default_data_collator?
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=None,
        tokenizer=None,
        data_collator=default_data_collator
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        metrics['train_samples'] = len(train_dataset)

        trainer.log_metrics('train', metrics)
        trainer.save_metrics('train', metrics)
        trainer.save_state()


if __name__ == '__main__':
    main()
