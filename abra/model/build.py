"""Build Grounding DINO, its criterion and post-processor from the model config + CLI arguments.

The vendored Grounding DINO code reads its hyper-parameters as attributes of one namespace, so the
model config file (plain Python assignments, e.g. ``groundingdino/config/cfg_odvg.py``) is merged into
the argparse namespace; CLI arguments win over config values.
"""
import copy
import runpy

import torch

from groundingdino.models.GroundingDINO.backbone.backbone import build_backbone
from groundingdino.models.GroundingDINO.groundingdino import GroundingDINO, PostProcess, SetCriterion
from groundingdino.models.GroundingDINO.matcher import build_matcher
from groundingdino.models.GroundingDINO.transformer import build_transformer
from abra.utils.checkpoint import load_checked


def merge_model_config(args):
    cfg = {k: v for k, v in runpy.run_path(str(args.model_cfg)).items() if not k.startswith("_")}
    for k, v in cfg.items():
        if not hasattr(args, k):
            setattr(args, k, v)
    return args


def build_model(args):
    model = GroundingDINO(
        backbone=build_backbone(args),
        transformer=build_transformer(args),
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        iter_update=True,
        query_dim=4,
        num_feature_levels=args.num_feature_levels,
        nheads=args.nheads,
        dec_pred_bbox_embed_share=args.dec_pred_bbox_embed_share,
        two_stage_type=args.two_stage_type,
        two_stage_bbox_embed_share=args.two_stage_bbox_embed_share,
        two_stage_class_embed_share=args.two_stage_class_embed_share,
        num_patterns=args.num_patterns,
        dn_number=0,
        dn_box_noise_scale=args.dn_box_noise_scale,
        dn_label_noise_ratio=args.dn_label_noise_ratio,
        dn_labelbook_size=args.dn_labelbook_size,
        text_encoder_type=args.text_encoder_type,
        sub_sentence_present=args.sub_sentence_present,
        max_text_len=args.max_text_len,
    )
    postprocessors = {"bbox": PostProcess(num_select=args.num_select, text_encoder_type=args.text_encoder_type,
                                          nms_iou_threshold=args.nms_iou_threshold, args=args)}
    return model, postprocessors


def build_criterion(args):
    weight_dict = {"loss_ce": args.cls_loss_coef, "loss_bbox": args.bbox_loss_coef,
                   "loss_giou": args.giou_loss_coef, "entropy": getattr(args, "fda_ent", 0.0)}
    clean = copy.deepcopy(weight_dict)
    if args.aux_loss:
        for i in range(args.dec_layers - 1):
            weight_dict.update({f"{k}_{i}": v for k, v in clean.items()})
    if args.two_stage_type != "no":
        no_interm_box_loss = getattr(args, "no_interm_box_loss", False)
        coeff = {"loss_ce": 1.0, "loss_bbox": 0.0 if no_interm_box_loss else 1.0,
                 "loss_giou": 0.0 if no_interm_box_loss else 1.0, "entropy": 1.0}
        interm = getattr(args, "interm_loss_coef", 1.0)
        weight_dict.update({f"{k}_interm": v * interm * coeff[k] for k, v in clean.items()})
    criterion = SetCriterion(args=args, matcher=build_matcher(args), weight_dict=weight_dict,
                             focal_alpha=args.focal_alpha, focal_gamma=args.focal_gamma, losses=["labels", "boxes"])
    return criterion.to(torch.device(args.device))


def load_base_weights(model, path):
    """Load the pretrained Grounding DINO checkpoint; returns its (cleaned) state dict (theta_0)."""
    state = torch.load(path, map_location="cpu")["model"]
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    load_checked(model, state, name=f"base weights {path}", allow_unexpected=True)
    return state
