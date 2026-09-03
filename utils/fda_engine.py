from groundingdino.datasets import get_coco_api_from_dataset
from utils.builders import build_uda_dataset
from torch.utils.data import SequentialSampler, RandomSampler, BatchSampler, DataLoader
from tqdm import tqdm
from utils.fda import FDA_source_to_target
import groundingdino.util.misc as utils
import torch
import torchvision.transforms as T
import wandb
from engine import evaluate
from utils.checkpoints import save_checkpoint

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def save_debug_stage(tensor_batch, stage_name, filename):
    """Saves the first image in a batch to disk and prints its math stats."""
    # Grab the first image in the batch, move to CPU, and detach
    img = tensor_batch[0].detach().cpu().float()
    
    # --- THE CRITICAL SANITY CHECK ---
    print(f"[{stage_name}] Stats -> Min: {img.min().item():.3f}, Max: {img.max().item():.3f}, Mean: {img.mean().item():.3f}")
    
    # For visualization, if the image has negatives (normalized), we shift it so we can see it
    if img.min() < 0:
        vis_img = img - img.min()
        vis_img = vis_img / (vis_img.max() + 1e-8)
    else:
        # Assuming 0-1 range. If it's 0-255, divide by 255.
        vis_img = torch.clamp(img, 0.0, 1.0) 
        
    vis_img = vis_img.permute(1, 2, 0).numpy() # C, H, W -> H, W, C

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(vis_img)
    ax.set_title(f"{stage_name}\nMin: {img.min().item():.2f} | Max: {img.max().item():.2f}")
    ax.axis('off')
    
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"--> Saved {filename}")

def run_fda(args, model, optimizer, pp, device, criterion):
    train_source, train_target, val_target = build_uda_dataset(args)
    
    train_sampler_source = RandomSampler(train_source)
    train_sampler_source_batch = BatchSampler(
        train_sampler_source,
        batch_size=args.batch_size,
        drop_last=True
    )
    train_loader_source = DataLoader(
        train_source,
        batch_sampler=train_sampler_source_batch,
        num_workers=args.num_workers,
        collate_fn=utils.collate_fn
    )
    
    train_sampler_target = RandomSampler(train_target)
    train_sampler_target_batch = BatchSampler(
        train_sampler_target,
        batch_size=args.batch_size,
        drop_last=True
    )
    train_loader_target = DataLoader(
        train_target,
        batch_sampler=train_sampler_target_batch,
        num_workers=args.num_workers,
        collate_fn=utils.collate_fn
    )
    
    val_sampler_target = SequentialSampler(val_target)
    val_loader_target = DataLoader(
        dataset=val_target,
        sampler=val_sampler_target,
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers
    )
    base_dataset = get_coco_api_from_dataset(val_target)
    
    normalization = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    args.LB = getattr(args, "lb", 0.01)
    
    # Pre-validation before training starts
    _ = evaluate(
        args=args,
        model=model,
        postprocessors=pp,
        loader=val_loader_target,
        base_ds=base_dataset,
        epoch=0,
        max_epoch=args.epochs,
        device=device,
        output_root=args.exp_out
    )
    
    best_ap = -99999
    for epoch in range(args.epochs):
        model.train()
        
        pbar = tqdm(zip(train_loader_source, train_loader_target), total=min(len(train_loader_source), len(train_loader_target)))
        for batch_s, batch_t in pbar:
            images_s, targets_s = batch_s
            images_t, _ = batch_t
            
            images_s = images_s.tensors.to(device)
            images_t = images_t.tensors.to(device)
            
            captions = [t['caption'] for t in targets_s]
            cap_list = [t['cap_list'] for t in targets_s]
            targets = [{k: v.to(device) for k, v in t.items() if torch.is_tensor(v)} for t in targets_s]
            
            # Apply FDA
            src_img = FDA_source_to_target(images_s, images_t, L=args.LB)

            src_img = src_img / 255.0  # Scale to [0, 1] before normalization
            
            # Normalize after FDA
            src_img = normalization(src_img)
            
            weight_dict = criterion.weight_dict
            
            outputs_s = model(src_img, captions=captions)
            
            # Let's break early if the loss explodes so we don't spam your console
            loss_dict = criterion(outputs_s, targets, cap_list, captions)
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
                
            losses.backward()
            optimizer.step()
            optimizer.zero_grad()

            if wandb.run is not None:
                step_fraction = epoch + (pbar.n / len(pbar))
                step_int = int(step_fraction * len(pbar))

                log_dict = {
                    "TRAIN/epoch": epoch,
                    "TRAIN/step": step_int,
                    "TRAIN/loss": losses.item(),
                }

                wandb.log(log_dict, step=step_int)

            if args.debug and (pbar.n + 1) % 10 == 0:
                break
        
        # --- VALIDATION ---
        if epoch % args.test_period == 0:
            stats = evaluate(
                args=args,
                model=model,
                postprocessors=pp,
                loader=val_loader_target,
                base_ds=base_dataset,
                epoch=epoch,
                max_epoch=args.epochs,
                device=device,
                output_root=args.exp_out
            )
            ap = stats['coco_eval_bbox'][0]

            if epoch != 0 and ap > best_ap:
                best_ap = ap
                save_checkpoint(args=args, model=model, svft=args.svft, pth_name='best', keys=args.trainable_keys)