import copy
import torch
import torch.optim.lr_scheduler as scheduler
from pathlib import Path

import transformers
from groundingdino.datasets.coco import CocoDetection
from groundingdino.datasets.odvg import ODVGDataset
from groundingdino.models.GroundingDINO.backbone.backbone import build_backbone
from groundingdino.models.GroundingDINO.groundingdino import GroundingDINO, PostProcess, SetCriterion
from groundingdino.models.GroundingDINO.matcher import build_matcher
from groundingdino.models.GroundingDINO.transformer import build_transformer
from utils.transforms import make_coco_transforms
from utils.transporter import SimpleTransporter, RotationTransporter, Analogy, MergedTransporter

def build_dataset(args, img_set='train', type='coco', add_split=True):
    img_folder = args.root
    split = 'train' if img_set == 'train' else 'val'
    img_folder = Path(img_folder) / split if add_split else Path(img_folder)
    annotations = args.train_annotations if img_set == 'train' else args.test_annotations


    transforms = make_coco_transforms(
        image_set=img_set,
        fix_size=args.fix_size,
        args=args
    )

    if not type == 'coco':
        label_map = args.label_map
        return ODVGDataset(
            args=args,
            root=img_folder,
            anno=annotations,
            label_map_anno=label_map,
            max_labels=args.max_labels,
            transforms=transforms
        )
    else:
        return CocoDetection(
            img_folder=img_folder,
            ann_file=annotations,
            transforms=transforms,
            return_masks=args.masks,
            aux_target_hacks=None
        )
        
def build_uda_dataset(args):
    transform_train = None
    if not args.uda == 'fda':
        transform_train = make_coco_transforms(
            image_set='train',
            fix_size=args.fix_size,
            args=args
        )
    
    transform_val = make_coco_transforms(
        image_set='val',
        fix_size=args.fix_size,
        args=args
    )
    
    train_source = ODVGDataset(
        args=args,
        root=Path(args.source_root) / 'train',
        anno=args.source_train_annotations,
        label_map_anno=args.label_map,
        max_labels=args.max_labels,
        transforms=transform_train
    )
    
    train_target = ODVGDataset(
        args=args,
        root=Path(args.target_root) / 'train',
        anno=args.target_train_annotations,
        label_map_anno=args.label_map,
        max_labels=args.max_labels,
        transforms=transform_train
    )
    
    val_target = CocoDetection(
        img_folder=Path(args.target_root) / 'val',
        ann_file=args.target_test_annotations,
        transforms=transform_val,
        return_masks=args.masks,
        aux_target_hacks=None
    )
    
    return train_source, train_target, val_target

def build_model(args):
    backbone = build_backbone(args)
    transformer = build_transformer(args)

    dn_labelbook_size = args.dn_labelbook_size
    dec_pred_bbox_embed_share = args.dec_pred_bbox_embed_share
    sub_sentence_present = args.sub_sentence_present
    
    model = GroundingDINO(
        backbone=backbone,
        transformer=transformer,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        iter_update=True,
        query_dim=4,
        num_feature_levels=args.num_feature_levels,
        nheads=args.nheads,
        dec_pred_bbox_embed_share=dec_pred_bbox_embed_share,
        two_stage_type=args.two_stage_type,
        two_stage_bbox_embed_share=args.two_stage_bbox_embed_share,
        two_stage_class_embed_share=args.two_stage_class_embed_share,
        num_patterns=args.num_patterns,
        dn_number=0,
        dn_box_noise_scale=args.dn_box_noise_scale,
        dn_label_noise_ratio=args.dn_label_noise_ratio,
        dn_labelbook_size=dn_labelbook_size,
        text_encoder_type=args.text_encoder_type,
        sub_sentence_present=sub_sentence_present,
        max_text_len=args.max_text_len
    )

    postprocessors = {
        'bbox': PostProcess(
            num_select=args.num_select,
            text_encoder_type=args.text_encoder_type,
            nms_iou_threshold=args.nms_iou_threshold,
            args=args
        )
    }

    return model, postprocessors

def build_loss(args):
    device = torch.device(args.device)
    matcher = build_matcher(args)

    # prepare weight dict
    weight_dict = {
        'loss_ce': args.cls_loss_coef,
        'loss_bbox': args.bbox_loss_coef,
        'loss_giou': args.giou_loss_coef,
        'entropy': args.fda_ent
    }
    clean_weight_dict_wo_dn = copy.deepcopy(weight_dict)
    clean_weight_dict = copy.deepcopy(weight_dict)

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in clean_weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    if args.two_stage_type != 'no':
        interm_weight_dict = {}
        try:
            no_interm_box_loss = args.no_interm_box_loss
        except:
            no_interm_box_loss = False
        _coeff_weight_dict = {
            'loss_ce': 1.0,
            'loss_bbox': 1.0 if not no_interm_box_loss else 0.0,
            'loss_giou': 1.0 if not no_interm_box_loss else 0.0,
            'entropy': 1.0
        }
        try:
            interm_loss_coef = args.interm_loss_coef
        except:
            interm_loss_coef = 1.0
        interm_weight_dict.update({k + f'_interm': v * interm_loss_coef * _coeff_weight_dict[k] for k, v in clean_weight_dict_wo_dn.items()})
        weight_dict.update(interm_weight_dict)

    losses = ['labels', 'boxes']

    criterion = SetCriterion(
        args=args,
        matcher=matcher,
        weight_dict=weight_dict,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        losses=losses
    )
    criterion.to(device)

    return criterion


def build_transporter(
    args,
    transporter_type,
    w_source,
    w_target,
    mask,
    analogy_source,
    analogy_target,
    pretrain
    ):
    if transporter_type == 'simple':
        return SimpleTransporter(w_source=w_source, w_target=w_target)
    elif transporter_type == 'abra':
        return RotationTransporter(
            w_source=w_source,
            w_target=w_target,
            mask=mask,
            alpha_estimate=args.alpha_estimate,
        )
    elif transporter_type == 'analogy':
        return Analogy(
            source_domain=analogy_source,
            target_domain=analogy_target,
            source_class=w_source,
            pretrain=pretrain
        )
    elif transporter_type == 'merged':
        return MergedTransporter(
            w_source=w_source,
            w_target=w_target,
            source_domain=analogy_source,
            target_domain=analogy_target,
            pretrain=pretrain,
            alpha=args.merged_alpha,
            mask=args.abra_mask,
            alpha_estimate=args.alpha_estimate,
        )
    else:
        raise ValueError(f"Unknown transporter type: {transporter_type}")
    
def build_optimizer(args, params):
    if args.optim == 'adamw':
        return torch.optim.AdamW(
            params=params,
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    elif args.optim == 'sgd':
        return torch.optim.SGD(
            params=params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
            nesterov=args.nesterov
        )


def build_scheduler(args, optimizer, train_dataset):
    sch = args.scheduler

    if sch == 'step':
        return scheduler.StepLR(optimizer, args.step_size), optimizer
    elif sch == 'multi':
        return scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=args.step_milestones,
            gamma=0.1
        ), optimizer
    elif sch == 'cosine':
        num_train_step = len(train_dataset) // args.batch_size * args.epochs
        
        return transformers.get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=int(num_train_step * args.warmup_ratio),
            num_training_steps=num_train_step
        ), optimizer
    elif sch == 'sequential':
        base_lr = 2e-4
        min_lr = 1e-6
        wu_factor = min_lr / base_lr
        wu_sched = scheduler.LinearLR(optimizer, start_factor=wu_factor, total_iters=1000)
        cos_sched = scheduler.CosineAnnealingLR(optimizer, T_max=(args.epochs * len(train_dataset) // args.batch_size) - 1000, eta_min=min_lr)
        sch = scheduler.SequentialLR(optimizer, schedulers=[wu_sched, cos_sched], milestones=[1000])
        return sch, optimizer
    
    return None, optimizer