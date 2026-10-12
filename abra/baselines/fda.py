"""FDA baseline (paper Table 2): Grounding DINO trained on labelled source images restyled with the
low-frequency amplitude of unlabelled target images, evaluated on the target test split.

Ported from ``utils/fda_engine.py`` (tag ``official``) with the same behaviour:
- images are resized to 1024x512 and cropped (no other augmentation), FDA is applied in [0, 255];
- by default only the supervised detection loss on FDA-restyled source images is used, as in the original code
  (its entropy options never reached the loss); ``--fda_ent > 0`` adds FDA's entropy minimisation on the
  unlabelled target images (new, see ``target_entropy``);
- the target split is evaluated before training and after every ``--test_period`` epochs; ``best`` is
  saved when AP improves, except after the first epoch (as in the original code).
``--fda_lb`` (band size L) is now honoured; the original code always used L = 0.01, the default.
"""
from pathlib import Path

import torch
import torchvision.transforms as T
from tqdm import tqdm

from abra.baselines.fda_transform import FDA_source_to_target
from abra.data import datasets as data
from abra.data.transforms import make_coco_transforms
from abra.eval.evaluate import evaluate
from abra.model import build
from abra.model.layers import freeze_all, parameter_groups, selected_modules, trainable_keys, unfreeze_modules
from abra.train.optim import build_optimizer
from abra.utils.checkpoint import load_checked
from abra.weights.io import load_domain_expert, save_domain_expert
from groundingdino.datasets.coco import CocoDetection
from groundingdino.datasets.odvg import ODVGDataset
from groundingdino.models.GroundingDINO.groundingdino import create_positive_map
import math
import numpy as np
import os
from PIL import Image
import random


class FDASourceDataset(ODVGDataset):
    """ODVG dataset with the fixed geometry used by the FDA baseline: images resized to 1024x512 (random crop of
    the same size, i.e. none), returned as float tensors in [0, 255] without normalisation (FDA works on raw
    intensities; normalisation happens after the Fourier transfer)."""

    resize_size = (1024, 512)
    crop_size = (1024, 512)

    def __getitem__(self, index):
        meta = self.metas[index]
        rel_path = meta["filename"]

        abs_path = os.path.join(self.root, rel_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"{abs_path} not found.")
        
        # 1. Load original RGB image
        image = Image.open(abs_path).convert('RGB')
        orig_w, orig_h = image.size
        
        # 2. Extract original boxes
        anno = meta["detection"]
        instances = [obj for obj in anno["instances"]]
        boxes = [obj["bbox"] for obj in instances]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        
        # 3. Text & Caption Logic (Untouched)
        ori_classes = [str(obj["label"]) for obj in instances]
        pos_labels = set(ori_classes)
        neg_labels = list(self.label_index.difference(pos_labels))

        vg_labels = list(pos_labels)
        num_to_add = min(len(neg_labels), self.max_labels-len(pos_labels))
        if num_to_add > 0:
            vg_labels.extend(random.sample(neg_labels, num_to_add))
        
        for i in range(len(vg_labels)-1, 0, -1):
            j = random.randint(0, i)
            vg_labels[i], vg_labels[j] = vg_labels[j], vg_labels[i]

        caption_list = [self.label_map[lb] for lb in vg_labels]
        caption_dict = {item:index for index, item in enumerate(caption_list)}
        caption = ' . '.join(caption_list) + ' .'
        classes = [caption_dict[self.label_map[str(obj["label"])]] for obj in instances]
        classes = torch.tensor(classes, dtype=torch.int64)

        # ==========================================
        # 4. SPATIAL TRANSFORMS (MANUAL)
        # ==========================================
        
        # --- A. Resize Image and Boxes ---
        image = image.resize(self.resize_size, Image.BICUBIC)
        
        ratio_width = self.resize_size[0] / orig_w
        ratio_height = self.resize_size[1] / orig_h
        
        if boxes.shape[0] > 0:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * ratio_width
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * ratio_height

        # --- B. Calculate Crop Coordinates ---
        left = self.resize_size[0] - self.crop_size[0]
        upper = self.resize_size[1] - self.crop_size[1]

        left = np.random.randint(0, high=left) if left > 0 else 0
        upper = np.random.randint(0, high=upper) if upper > 0 else 0
        right = left + self.crop_size[0]
        lower = upper + self.crop_size[1]

        # --- C. Apply Crop to Image ---
        image = image.crop((left, upper, right, lower))
        
        # --- D. Apply Crop to Boxes & Clamp ---
        if boxes.shape[0] > 0:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] - left
            boxes[:, [1, 3]] = boxes[:, [1, 3]] - upper
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, self.crop_size[0])
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, self.crop_size[1])

            # --- E. Convert XYXY to CXCYWH (Normalized 0.0 to 1.0) ---
            cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
            cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
            bw = boxes[:, 2] - boxes[:, 0]
            bh = boxes[:, 3] - boxes[:, 1]
            
            cx = cx / self.crop_size[0]
            cy = cy / self.crop_size[1]
            bw = bw / self.crop_size[0]
            bh = bh / self.crop_size[1]
            
            boxes = torch.stack([cx, cy, bw, bh], dim=1)
            
            # --- F. Safely filter out boxes cut away by the crop ---
            # Keeps only boxes with an area strictly greater than 0
            keep_mask = (bw > 0.0) & (bh > 0.0)
            boxes = boxes[keep_mask]
            classes = classes[keep_mask]
        # ==========================================

        # 5. Final Image Tensor Conversion 
        # Output shape: [C, H, W], Value Range: [0.0, 255.0] (Ready for FDA!)
        image = np.asarray(image, np.float32)
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image.copy()).float()
        
        # 6. Build Target Dictionary
        target = {}
        # Grounding DINO expects [height, width] for size targets
        target["size"] = torch.as_tensor([int(self.crop_size[1]), int(self.crop_size[0])]) 
        target["original_size"] = torch.as_tensor([int(orig_h), int(orig_w)])
        target["cap_list"] = caption_list
        target["caption"] = caption
        target["boxes"] = boxes
        target["labels"] = classes

        return image, target


def add_fda_args(p):
    g = p.add_argument_group("FDA data")
    g.add_argument("--source_root", type=Path, required=True, help="source image root (train/ is appended)")
    g.add_argument("--target_root", type=Path, required=True, help="target image root (train/, val/ are appended)")
    g.add_argument("--source_train_annotations", type=Path, required=True, help="labelled source ODVG jsonl")
    g.add_argument("--target_train_annotations", type=Path, required=True, help="target ODVG jsonl (labels unused)")
    g.add_argument("--target_test_annotations", type=Path, required=True, help="target COCO json for evaluation")
    g.add_argument("--label_map", type=Path, required=True)
    g.add_argument("--init", type=Path, default=None, help="start from this domain expert instead of theta_0")
    g.add_argument("--fda_lb", type=float, default=0.01, help="FDA band size L (fraction of the image size)")
    g.add_argument("--fda_ent", type=float, default=0.0,
                   help="weight of the entropy loss on target images (0 = off, original Table 2 setting; "
                        "the FDA paper uses 0.005)")
    g.add_argument("--fda_ita", type=float, default=2.0, help="robustness exponent of the entropy loss (FDA: 2.0)")
    g.add_argument("--fda_ent_threshold", type=float, default=0.3,
                   help="entropy only on target queries whose best class score exceeds this")
    g.add_argument("--fix_size", default=False)
    g.add_argument("--visualize_predictions", default=False)
    g.add_argument("--save_coco_format", default=None)
    g.add_argument("--zs_th", type=float, default=0.3)


def target_entropy(outputs, caption, categories, ita, threshold):
    """FDA entropy loss (Yang & Soatto, CVPR 2020) on unlabelled target images, adapted to Grounding DINO.

    FDA penalises the robust entropy ``(ent**2 + 1e-8) ** ita`` of the normalised class posterior of every pixel.
    Grounding DINO scores a (query, category) pair with the mean sigmoid of the category's tokens (as PostProcess
    does), so ``ent`` is the binary entropy of these scores, divided by ln 2 (range [0, 1]) and averaged over the
    categories. Only queries whose best score exceeds ``threshold`` contribute: pushing background queries to
    confidence would create false positives. Returns (loss or None if no query passes, number of queries used).
    """
    logits = outputs["pred_logits"]  # [B, Q, 256], -inf on unused tokens
    terms = []
    for j in range(logits.shape[0]):
        pos = create_positive_map(outputs["token"][j], range(len(categories)), categories, caption).to(logits.device)
        pos = pos[pos.sum(-1) > 0]
        if not len(pos):
            continue
        scores = torch.sigmoid(logits[j]) @ (pos / pos.sum(-1, keepdim=True)).T  # [Q, C]
        p = scores.clamp(1e-6, 1 - 1e-6)
        ent = (-(p * p.log() + (1 - p) * (1 - p).log()) / math.log(2)).mean(-1)
        terms.append(((ent ** 2 + 1e-8) ** ita)[scores.max(-1).values > threshold])
    terms = torch.cat(terms) if terms else logits.new_zeros(0)
    return (terms.mean() if len(terms) else None), len(terms)


def run_fda(args, model, postprocessors, device, logger):
    if args.init:
        tensors, _ = load_domain_expert(args.init)
        load_checked(model, tensors, name=f"init {args.init}")
    freeze_all(model)
    unfreeze_modules(model, selected_modules(model, args.trainable))
    keys = trainable_keys(model)
    criterion = build.build_criterion(args)
    optimizer = build_optimizer(args, parameter_groups(model, args))

    source = FDASourceDataset(root=Path(args.source_root) / "train", anno=args.source_train_annotations,
                              label_map_anno=args.label_map, max_labels=args.max_labels, transforms=None)
    target = FDASourceDataset(root=Path(args.target_root) / "train", anno=args.target_train_annotations,
                              label_map_anno=args.label_map, max_labels=args.max_labels, transforms=None)
    test = CocoDetection(img_folder=Path(args.target_root) / "val", ann_file=args.target_test_annotations,
                         transforms=make_coco_transforms("val", fix_size=args.fix_size, args=args),
                         return_masks=args.masks, aux_target_hacks=None)
    source_loader, target_loader = data.train_loader(source, args), data.train_loader(target, args)
    test_loader, test_gt = data.test_loader(test, args), data.coco_api(test)
    normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # target captions name every class of the label map: the target labels must not leak into the entropy term
    target_classes = list(target.label_map.values())
    target_caption = " . ".join(target_classes) + " ."

    def save(name, ap):
        state = model.state_dict()
        info = {"exp_name": args.exp_name, "baseline": "fda", "fda_lb": args.fda_lb, "fda_ent": args.fda_ent,
                "fda_ita": args.fda_ita, "fda_ent_threshold": args.fda_ent_threshold, "select_AP": ap, "checkpoint": name,
                "source_train_annotations": args.source_train_annotations,
                "target_train_annotations": args.target_train_annotations, "seed": args.seed}
        save_domain_expert(args.output_root / f"{name}.safetensors", {k: state[k] for k in keys}, info)

    evaluate(model, postprocessors, test_loader, test_gt, args, device, logger, epoch=0, max_epoch=args.epochs)
    best_ap, result = -float("inf"), None
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(zip(source_loader, target_loader), total=min(len(source_loader), len(target_loader)))
        for step, ((images_s, targets_s), (images_t, _)) in enumerate(pbar):
            images_s, images_t = images_s.tensors.to(device), images_t.tensors.to(device)
            captions = [t["caption"] for t in targets_s]
            cap_list = [t["cap_list"] for t in targets_s]
            targets = [{k: v.to(device) for k, v in t.items() if torch.is_tensor(v)} for t in targets_s]

            restyled = normalize(FDA_source_to_target(images_s, images_t, L=args.fda_lb) / 255.0)
            loss_dict = criterion(model(restyled, captions=captions), targets, cap_list, captions)
            loss = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict if k in criterion.weight_dict)
            loss.backward()
            log = {"TRAIN/epoch": epoch, "TRAIN/loss": loss.item()}
            if args.fda_ent > 0:  # separate backward: the two forward graphs never coexist in memory
                n = len(images_t)
                outputs_t = model(normalize(images_t / 255.0), captions=[target_caption] * n)
                ent, n_queries = target_entropy(outputs_t, target_caption, target_classes, args.fda_ita,
                                                args.fda_ent_threshold)
                if ent is not None:
                    (args.fda_ent * ent).backward()
                log.update({"TRAIN/entropy": ent.item() if ent is not None else 0.0, "TRAIN/entropy_queries": n_queries})
            optimizer.step()
            optimizer.zero_grad()
            logger.log(log, step=epoch * len(pbar) + step)
            if args.debug and (step + 1) % 10 == 0:
                break

        if epoch % args.test_period == 0:
            result = evaluate(model, postprocessors, test_loader, test_gt, args, device, logger,
                              epoch=epoch, max_epoch=args.epochs)
            ap = result["AP"] / 100
            if epoch != 0 and ap > best_ap:
                best_ap = ap
                save("best", ap)
    save("last", None)
    return result
