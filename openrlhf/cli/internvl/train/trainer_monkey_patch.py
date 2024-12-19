import json
import os

import torch
import torch.nn as nn
import transformers
from transformers import Trainer, logging
from transformers.trainer import is_sagemaker_mp_enabled
from transformers.optimization import get_scheduler

from functools import partial
from torch.optim.lr_scheduler import LambdaLR
import math 

logger = logging.get_logger(__name__)


def get_num_layer_for_vit(name):
    # assert name.startswith("vision_model.")

    if "embeddings." in name:
        return 0
    if "layers." in name:
        var_name = name.split('layers.')[-1]
        layer_id = int(var_name.split('.')[0])
        return layer_id + 1
    else:
        raise ValueError


def param_classification(name):
    if name.startswith("vision_model."):
        return "vit"
    elif name.startswith("language_model."):
        return "llm"
    elif name.startswith("mlp1."):
        return "mlp"
    else:
        print(f"Treat {name} as mlp")
        return "mlp"


def create_optimizer(self):
    """
    Setup the optimizer.

    We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
    Trainer's init through `optimizers`, or subclass and override this method in a subclass.
    """
    
    opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
    parameter_groups = {}
    vit_num_layers = len(opt_model.vision_model.encoder.layers) + 1
    print(vit_num_layers)
    vit_lr_decay = self.args.vit_lr_decay
    vit_lr_scale = self.args.vit_lr_scale
    llm_lr_scale = self.args.llm_lr_scale
    mlp_lr_scale = self.args.mlp_lr_scale
    
    for name, param in opt_model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith('.bias'):
            group_name = 'no_decay'
            this_weight_decay = 0.
        else:
            group_name = 'decay'
            this_weight_decay = self.args.weight_decay
        
        cls = param_classification(name)
        if cls == "vit":
            layer_id = get_num_layer_for_vit(name)
            group_name = '%s_layer_%d_%s' % (cls, layer_id, group_name)
        else:
            group_name = '%s_%s' % (cls, group_name)
        if group_name not in parameter_groups:
            if cls == "vit":
                scale = vit_lr_decay ** (vit_num_layers - layer_id - 1) * vit_lr_scale
                # scale = vit_lr_scale
            elif cls == "llm":
                scale = llm_lr_scale
            elif cls == "mlp":
                scale = mlp_lr_scale
            scale = min(1.0, scale)
            parameter_groups[group_name] = {
                'weight_decay': this_weight_decay,
                'params': [],
                'param_names': [],
                'lr_scale': scale,
                'group_name': group_name,
                'lr': scale * self.args.learning_rate,
            }
        parameter_groups[group_name]['params'].append(param)
        parameter_groups[group_name]['param_names'].append(name)
    
    rank = torch.distributed.get_rank()
    if rank == 0:
        to_display = {}
        for key in parameter_groups:
            to_display[key] = {
                'param_names': parameter_groups[key]['param_names'],
                'lr_scale': parameter_groups[key]['lr_scale'],
                'lr': parameter_groups[key]['lr'],
                'weight_decay': parameter_groups[key]['weight_decay'],
            }
        print('Param groups = %s' % json.dumps(to_display, indent=2))
    optimizer_grouped_parameters = list(parameter_groups.values())
    optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
    
    self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    if optimizer_cls.__name__ == "Adam8bit":
        import bitsandbytes
        
        manager = bitsandbytes.optim.GlobalOptimManager.get_instance()
        
        skipped = 0
        for module in opt_model.modules():
            if isinstance(module, nn.Embedding):
                skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                logger.info(f"skipped {module}: {skipped / 2 ** 20}M params")
                manager.register_module_override(module, "weight", {"optim_bits": 32})
                logger.debug(f"bitsandbytes: will optimize {module} in fp32")
        logger.info(f"skipped: {skipped / 2 ** 20}M params")
    
    if is_sagemaker_mp_enabled():
        import smdistributed.modelparallel.torch as smp
        self.optimizer = smp.DistributedOptimizer(self.optimizer)
    return self.optimizer


def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float, min_lr_rate: float = 0.0
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    factor = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
    factor = factor * (1 - min_lr_rate) + min_lr_rate
    return max(0, factor)


def get_cosine_with_min_lr_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
    min_lr: float = None,
    min_lr_rate: float = None,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to min_lr, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
        min_lr (`float`, *optional*):
            The minimum learning rate to reach after the cosine schedule.
        min_lr_rate (`float`, *optional*):
            The minimum learning rate as a ratio of the initial learning rate. If set, `min_lr` should not be set.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    if min_lr is not None and min_lr_rate is not None:
        raise ValueError("Only one of min_lr or min_lr_rate should be set")
    elif min_lr is not None:
        min_lr_rate = min_lr / optimizer.defaults["lr"]
    elif min_lr_rate is None:
        raise ValueError("One of min_lr or min_lr_rate should be set through the `lr_scheduler_kwargs`")

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        min_lr_rate=min_lr_rate,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
    """
    Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
    passed as an argument.
    Args:
        num_training_steps (int): The number of training steps to do.
    """
    if self.lr_scheduler is None:
        if self.args.lr_scheduler_type == "cosine" and (self.args.min_lr_rate > 0):
            self.lr_scheduler = get_cosine_with_min_lr_schedule_with_warmup(optimizer, 
                                                                            num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                                                                            num_training_steps=num_training_steps,
                                                                            min_lr_rate= self.args.min_lr_rate)
        else:
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
                scheduler_specific_kwargs=self.args.lr_scheduler_kwargs,
            )
        self._created_lr_scheduler = True
    return self.lr_scheduler


def replace_create_optimizer():
    print('Replace original create_optimizer with custom create_optimizer')
    transformers.Trainer.create_optimizer = create_optimizer

def add_conine_with_min_lr_scheduler():
    print('support cosine scheduler with min_lr/min_lr_rate')
    transformers.Trainer.create_scheduler = create_scheduler