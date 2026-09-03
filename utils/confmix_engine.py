from groundingdino.datasets import get_coco_api_from_dataset
from utils.confmix import ModelEMA
from utils.builders import build_uda_dataset
from torch.utils.data import SequentialSampler, RandomSampler, BatchSampler, DataLoader
from tqdm import tqdm
from utils.general import xyxy2xywh, clip_coords_target
import groundingdino.util.misc as utils
import torch
import numpy as np
import math

def output_to_target(output):
    # Convert model output to target format [batch_id, class_id, x, y, w, h, conf]
    targets = []
    for i, o in enumerate(output):
        for box, conf, cls in zip(o['boxes'].cpu().numpy(), o['scores'].cpu().numpy(), o['labels'].cpu().numpy()):
            targets.append([i, cls, *box.tolist(), conf])
    return np.array(targets)

def run_confmix(args, model, optimizer, pp, device, criterion):
    raise NotImplementedError("This function is not implemented yet.")
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
    
    ema_model = ModelEMA(model=model)
    
    num_batches = min(len(train_loader_source), len(train_loader_target))
    max_iters = args.epochs * num_batches
    
    for epoch in range(args.epochs):
        model.train()
    
        pbar = enumerate(zip(train_loader_source, train_loader_target))
        pbar = tqdm(pbar, total=num_batches, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        
        optimizer.zero_grad()
        for it, ((images_s, targets_s), (images_t, targets_t)) in pbar:
            num_iter = it + num_batches * epoch

            images_s, images_t = images_s.to(device), images_t.to(device)
            
            captions_source = [t["caption"] for t in targets_s]
            caption_list_source = [t["cap_list"] for t in targets_s]
            
            captions_target = [t["caption"] for t in targets_t]
            caption_list_target = [t["cap_list"] for t in targets_t]
            
            targets_s = [{k: v.to(device) for k, v in t.items() if torch.is_tensor(v)} for t in targets_s]
            targets_t = [{k: v.to(device) for k, v in t.items() if torch.is_tensor(v)} for t in targets_t]
            
            r = num_iter / max_iters
            delta = 2 / (1 + math.exp(-5. * r)) - 1
            
            pred_s = model(images_s, captions=captions_source)
            orig_sizes_s = torch.stack([t["original_size"] for t in targets_s]).to(device)
            res_s = pp['bbox'](pred_s, orig_sizes_s)
            
            
            pred_t = model(images_t, captions=captions_target)
            orig_sizes_t = torch.stack([t["original_size"] for t in targets_t]).to(device)
            res_t = pp['bbox'](pred_t, orig_sizes_t)
            
            b, c, h, w = images_s.tensors.shape
            out_s = output_to_target(res_s)
            out_t = output_to_target(res_t)
            
            tar_lb = out_t[(out_t[:,2] < w//2) & (out_t[:,3] >= h//2), :]
            tar_lt = out_t[(out_t[:,2] < w//2) & (out_t[:,3] < h//2), :]
            tar_rb = out_t[(out_t[:,2] >= w//2) & (out_t[:,3] >= h//2), :]
            tar_rt = out_t[(out_t[:,2] >= w//2) & (out_t[:,3] < h//2), :]
            
            target_regions = [tar_lb, tar_lt, tar_rb, tar_rt]
            
            mean_confidences = torch.nan_to_num(torch.as_tensor([np.mean(i[:,6]) for i in target_regions]))
            index = torch.max(mean_confidences, 0)[1]
            
            mask = torch.zeros((b, c, h, w))
            if index == 0:
                # tar_lb
                tar_lb[:,2:6] = clip_coords_target(tar_lb, 0, w//2, h//2, h)
                out_s = out_s[(out_s[:,2] >= w//2) | (out_s[:,3] < h//2), :]
                out_s[(out_s[:,2] >= w//2) & (out_s[:,3] >= h//2), 2:6] = clip_coords_target(out_s[(out_s[:,2] >= w//2) & (out_s[:,3] >= h//2), :], w//2, w, 0, h)
                out_s[(out_s[:,2] >= w//2) & (out_s[:,3] < h//2), 2:6] = clip_coords_target(out_s[(out_s[:,2] >= w//2) & (out_s[:,3] < h//2), :], 0, w, 0, h)
                out_s[out_s[:,2] < w//2, 2:6] = clip_coords_target(out_s[out_s[:,2] < w//2, :], 0, w, 0, h//2)

                mask[:, :, h//2:h+1, 0:w//2] = 1.
            elif index == 1:
                # tar_lt
                tar_lt[:,2:6] = clip_coords_target(tar_lt, 0, w//2, 0, h//2)
                out_s = out_s[(out_s[:,2] >= w//2) | (out_s[:,3] >= h//2), :]
                out_s[(out_s[:,2] >= w//2) & (out_s[:,3] < h//2), 2:6] = clip_coords_target(out_s[(out_s[:,2] >= w//2) & (out_s[:,3] < h//2), :], w//2, w, 0, h)
                out_s[(out_s[:,2] >= w//2) & (out_s[:,3] >= h//2), 2:6] = clip_coords_target(out_s[(out_s[:,2] >= w//2) & (out_s[:,3] >= h//2), :], 0, w, 0, h)
                out_s[out_s[:,2] < w//2, 2:6] = clip_coords_target(out_s[out_s[:,2] < w//2, :], 0, w, h//2, h)

                mask[:, :, 0:h//2, 0:w//2] = 1.
            elif index == 2:
                # tar_rb
                tar_rb[:,2:6] = clip_coords_target(tar_rb, w//2, w, h//2, h)
                out_s = out_s[(out_s[:,2] < w//2) | (out_s[:,3] < h//2), :]
                out_s[(out_s[:,2] < w//2) & (out_s[:,3] >= h//2), 2:6] = clip_coords_target(out_s[(out_s[:,2] < w//2) & (out_s[:,3] >= h//2), :], w//2, w, 0, h)
                out_s[(out_s[:,2] < w//2) & (out_s[:,3] < h//2), 2:6] = clip_coords_target(out_s[(out_s[:,2] < w//2) & (out_s[:,3] < h//2), :], 0, w, 0, h)
                out_s[out_s[:,2] >= w//2, 2:6] = clip_coords_target(out_s[out_s[:,2] >= w//2, :], 0, w, 0, h//2)

                mask[:, :, h//2:h+1, w//2:w+1] = 1.
            elif index == 3:
                # tar_rt
                tar_rt[:,2:6] = clip_coords_target(tar_rt, w//2, w, 0, h//2)
                out_s = out_s[(out_s[:,2] < w//2) | (out_s[:,3] >= h//2), :]
                out_s[(out_s[:,2] < w//2) & (out_s[:,3] < h//2), 2:6] = clip_coords_target(out_s[(out_s[:,2] < w//2) & (out_s[:,3] < h//2), :], w//2, w, 0, h)
                out_s[(out_s[:,2] < w//2) & (out_s[:,3] >= h//2), 2:6] = clip_coords_target(out_s[(out_s[:,2] < w//2) & (out_s[:,3] >= h//2), :], 0, w, 0, h)
                out_s[out_s[:,2] >= w//2, 2:6] = clip_coords_target(out_s[out_s[:,2] >= w//2, :], 0, w, h//2, h)

                mask[:, :, 0:h//2, w//2:w+1] = 1.
                
            # create confmix targets and compute confmix weight
            targets_confmix_s = torch.as_tensor(out_s)
            targets_confmix_t = torch.as_tensor(target_regions[index])

            targets_confmix = torch.cat((targets_confmix_s, targets_confmix_t))

            c_gamma_thres = 0.5
            gamma = (targets_confmix[:,6] > c_gamma_thres).sum() / \
                            (targets_confmix[:,6]).nelement()

            targets_confmix = targets_confmix[:,:6] # remove confidence values
            # normalize
            targets_confmix[:, [2, 4]] /= w
            targets_confmix[:, [3, 5]] /= h

            # create confmix image
            imgs_confmix = images_s*(1-mask) + images_t*mask
            imgs_confmix = imgs_confmix.to(device, non_blocking=True).float() / 255.0
            
            # forward pass on confmix images
            pred_confmix = model(imgs_confmix, captions=captions_source)
            loss_dict = criterion(pred_confmix, targets_s, caption_list_source, captions_source)
            losses = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict.keys() if k in criterion.weight_dict)
            losses.backward()
            
            optimizer.step()
            optimizer.zero_grad()