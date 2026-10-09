import copy
import torch
import wandb
from pycocotools.coco import COCO
from torch.optim.lr_scheduler import OneCycleLR

from argparse import Namespace
from typing import Iterable
from pathlib import Path

from tqdm import tqdm

from groundingdino.datasets.coco_eval import CocoEvaluator
import groundingdino.util.misc as utils
from groundingdino.util.utils import to_device
from tools.coco_utils import save_predictions_as_coco
from tools.visualizer import visualize_predictions
from utils.files import read_json

def train_one_epoch(
    args: Namespace,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_epochs: int,
    max_norm: float = 0,
    scheduler: torch.optim.lr_scheduler = None,
):
    model.train()
    criterion.train()

    # Hyperparameter for the generic head weight
    GENERIC_LOSS_WEIGHT = args.object_loss_w
    CLASS_LOSS_WEIGHT = args.class_loss_w

    step = 0
    pbar = tqdm(loader, desc=f'Train [{epoch}/{max_epochs}]', position=1, colour='green', leave=True)
    
    for samples, targets in pbar:
        samples = samples.to(device)
        
        # Move targets to device
        targets_specific = [{k: v.to(device) for k, v in t.items() if torch.is_tensor(v)} for t in targets]
        
        losses_specific = torch.tensor(0.0)
        weight_dict = criterion.weight_dict
        if not args.objectify:
            captions_specific = [t["caption"] for t in targets]
            cap_list_specific = [t["cap_list"] for t in targets]

            # Forward Pass (Specific)
            outputs_specific = model(samples, captions=captions_specific)
            loss_dict_specific = criterion(outputs_specific, targets_specific, cap_list_specific, captions_specific)

            losses_specific = sum(loss_dict_specific[k] * weight_dict[k] for k in loss_dict_specific.keys() if k in weight_dict)

            # Backward Pass (Specific)
            # We assume gradients accumulate by default  in PyTorch
            (losses_specific * CLASS_LOSS_WEIGHT).backward()

        losses_generic_item = 0.0
        if args.objectify:
            
            # A. Construct Generic Prompt
            # We treat the entire batch as having the single prompt "object ."
            captions_generic = ["object ." for _ in targets]
            # The cap_list must match the prompt tokens
            cap_list_generic = [["object"] for _ in targets]

            # B. Construct Generic Targets
            targets_generic = []
            for t in targets_specific:
                # Deepcopy to avoid modifying the tensors used in Phase 1
                t_gen = {k: v.clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v) for k, v in t.items()}
                
                if 'labels' in t_gen:
                    # CRITICAL FIX:
                    # The caption is "object .". The word "object" is at index 0.
                    # Therefore, we map ALL ground truth boxes to label index 0.
                    t_gen['labels'] = torch.zeros_like(t_gen['labels'])
                
                targets_generic.append(t_gen)

            # Forward Pass (Generic)
            outputs_generic = model(samples, captions=captions_generic)
            loss_dict_generic = criterion(outputs_generic, targets_generic, cap_list_generic, captions_generic)
            
            # Calculate weighted sum
            losses_generic = sum(loss_dict_generic[k] * weight_dict[k] for k in loss_dict_generic.keys() if k in weight_dict)
            
            # Scale and Backward
            # We scale this loss to balance it with the semantic task
            (losses_generic * GENERIC_LOSS_WEIGHT).backward()
            
            losses_generic_item = losses_generic.item()

        # ------------------------------------------------------------------
        # OPTIMIZATION & LOGGING
        # ------------------------------------------------------------------
        
        # Total loss for logging only (gradients are already in the .grad attributes)
        total_loss_item = (losses_specific.item() * CLASS_LOSS_WEIGHT) + (losses_generic_item * GENERIC_LOSS_WEIGHT)

        if wandb.run is not None:
            step_fraction = epoch + (step / len(loader))
            step_int = int(step_fraction * 1000)
            
            log_dict = {
                'TRAIN/epoch': epoch,
                'TRAIN/total_loss': total_loss_item,
                'TRAIN/loss_specific': losses_specific.item()
            }
            if args.objectify:
                log_dict['TRAIN/loss_generic'] = losses_generic_item
            
            wandb.log(log_dict, step=step_int)
        
        # Optimizer Step with Accumulation
        if (step + 1) % args.accumulation_steps == 0:
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            
            optimizer.step()
            
            if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                scheduler.step()
            
            optimizer.zero_grad()
            
        # Criterion maintenance (standard GroundingDINO hooks)
        if getattr(criterion, 'loss_weight_decay', False):
            criterion.loss_weight_decay(epoch=epoch)
        if getattr(criterion, 'tuning_matching', False):
            criterion.tuning_matching(epoch)
            
        if args.debug and (step + 1) % 5 == 0:
            print("DEBUG! " * 5)
            break
        
        step += 1
        
@torch.no_grad()
def evaluate(
    args: Namespace,
    model: torch.nn.Module,
    postprocessors: torch.nn.Module,
    loader: Iterable,
    base_ds,
    epoch: int,
    max_epoch: int,
    device: torch.device,
    output_root: Path
):
    model.eval()

    if args.visualize_predictions:
        vis_count = 0
        max_vis = 20
        vis_output_dir = output_root / "visualization" / f"epoch_{epoch}"
        vis_output_dir.mkdir(parents=True, exist_ok=True)
    
    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    
    coco = COCO(args.coco_val_path)

    cat_ids = sorted(coco.getCatIds())
    category_dict = coco.loadCats(cat_ids)
    cat_list = [item['name'] for item in category_dict]
    caption = " . ".join(cat_list) + ' .'
    
    coco_evaluator = CocoEvaluator(
        coco_gt=base_ds,
        iou_types=iou_types,
        useCats=True,
        class_names=cat_list
    )
    
    # VISUALIZATION SETUP
    if args.visualize_predictions:
        vis_count = 0
        max_vis = 20
        
    if getattr(args, 'save_coco_format', None):
        all_targets_for_coco = []
        all_results_for_coco = []
    
    step = 0
    pbar = tqdm(loader, desc=f'Test [{epoch}/{max_epoch}]', position=2, colour='blue', leave=True)
    for samples, targets in pbar:
        samples = samples.to(device)
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]
        
        bs = samples.tensors.shape[0]

        input_captions = [caption] * bs
        
        outputs = model(samples, captions=input_captions)
    
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        
        if coco_evaluator is not None:
            coco_evaluator.update(res)
            
        # VISUALIZATION CODE
        if args.visualize_predictions and vis_count < max_vis:
            visualize_predictions(
                targets, results, cat_list, 
                vis_output_dir,  # Pass the epoch-specific directory
                vis_count, bs, 
                loader.dataset,
            )
            vis_count += bs
            
        if getattr(args, 'save_coco_format', None):
            cpu_targets = [{k: v.cpu() if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]
            all_targets_for_coco.extend(cpu_targets)
            all_results_for_coco.extend(results)
            
        if args.debug and (step + 1) % 5 == 0:
            print("DEBUG! " * 5)
            break
        step += 1
    
    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        
    if getattr(args, 'save_coco_format', None):
        coco_output_path = Path(args.save_coco_format)
        if coco_output_path.suffix != '.json':
            coco_output_path = coco_output_path.with_suffix('.json')
        coco_output_path.parent.mkdir(parents=True, exist_ok=True)

        save_predictions_as_coco(
            all_targets_for_coco, 
            all_results_for_coco, 
            cat_list, 
            loader.dataset,
            coco_output_path,
            confidence_threshold=args.zs_th
        )
        
    stats = {}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
            
    return stats