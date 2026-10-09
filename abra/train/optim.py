import torch
import transformers
from torch.optim import lr_scheduler

# Schedulers whose step() is called once per optimizer step; the others once per epoch.
PER_ITERATION = {"cosine", "sequential"}


def build_optimizer(args, param_groups):
    if args.optim == "adamw":
        return torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    if args.optim == "sgd":
        return torch.optim.SGD(param_groups, lr=args.lr, weight_decay=args.weight_decay,
                               momentum=args.momentum, nesterov=args.nesterov)
    raise ValueError(args.optim)


def build_scheduler(args, optimizer, steps_per_epoch):
    """``steps_per_epoch`` counts optimizer steps (batches / accumulation_steps)."""
    if args.scheduler is None:
        return None
    if args.scheduler == "step":
        return lr_scheduler.StepLR(optimizer, args.step_size)
    if args.scheduler == "multi":
        return lr_scheduler.MultiStepLR(optimizer, milestones=args.step_milestones, gamma=0.1)
    total = steps_per_epoch * args.epochs
    if args.scheduler == "cosine":
        return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total * args.warmup_ratio),
                                                            num_training_steps=total)
    if args.scheduler == "sequential":
        warmup, base_lr, min_lr = 1000, 2e-4, 1e-6
        return lr_scheduler.SequentialLR(optimizer, milestones=[warmup], schedulers=[
            lr_scheduler.LinearLR(optimizer, start_factor=min_lr / base_lr, total_iters=warmup),
            lr_scheduler.CosineAnnealingLR(optimizer, T_max=total - warmup, eta_min=min_lr)])
    raise ValueError(args.scheduler)
