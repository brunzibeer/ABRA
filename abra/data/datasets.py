"""Datasets and loaders: ODVG jsonl for training, COCO json for evaluation."""
from pathlib import Path

import torch
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler

import groundingdino.util.misc as misc
from groundingdino.datasets.coco import CocoDetection
from groundingdino.datasets.odvg import ODVGDataset
from abra.data.transforms import make_coco_transforms
from abra.utils.run import worker_init_fn


def image_folder(root, split, add_split):
    """``root/split`` when ``add_split`` (default: only for cityscapes/AODRaw/bdd roots), else ``root``."""
    if add_split is None:
        add_split = any(n in str(root) for n in ("cityscapes", "AODRaw", "bdd"))
    return Path(root) / split if add_split else Path(root)


def train_dataset(args, root=None, annotations=None, transforms="train"):
    root = root or args.root
    transforms = make_coco_transforms("train", fix_size=args.fix_size, args=args) if transforms == "train" else transforms
    return ODVGDataset(root=image_folder(root, "train", args.add_split), anno=annotations or args.train_annotations,
                       label_map_anno=args.label_map, max_labels=args.max_labels, transforms=transforms)


def test_dataset(args, root=None, annotations=None):
    root = root or args.root
    transforms = make_coco_transforms("val", fix_size=args.fix_size, args=args)
    return CocoDetection(img_folder=image_folder(root, "val", args.add_split), ann_file=annotations or args.test_annotations,
                         transforms=transforms, return_masks=args.masks, aux_target_hacks=None)


def train_loader(dataset, args):
    sampler = BatchSampler(RandomSampler(dataset), batch_size=args.batch_size, drop_last=True)
    return DataLoader(dataset, batch_sampler=sampler, collate_fn=misc.collate_fn, num_workers=args.num_workers,
                      worker_init_fn=worker_init_fn)


def test_loader(dataset, args):
    return DataLoader(dataset, sampler=SequentialSampler(dataset), drop_last=False, collate_fn=misc.collate_fn,
                      num_workers=args.num_workers, worker_init_fn=worker_init_fn)


def coco_api(dataset):
    while isinstance(dataset, torch.utils.data.Subset):
        dataset = dataset.dataset
    return dataset.coco
