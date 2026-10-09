"""Reentrant gradient checkpointing leaves encoder layer 0 without gradient; --fix_layer0_grad fixes it.

Needs the base Grounding DINO weights in ``$ABRA_WEIGHTS`` (default ``../weights``); ~1 min on CPU.
"""
import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = Path(os.environ.get("ABRA_WEIGHTS", ROOT.parent / "weights"))
pytestmark = pytest.mark.skipif(not (WEIGHTS / "groundingdino_swint_ogc.pth").exists(), reason="base weights not found")


def _layers_without_grad(tmp_path, fix):
    sys.path.insert(0, str(ROOT / "tests" / "golden"))
    import cpu_forward_golden as golden
    import groundingdino.util.misc as misc
    from abra.cli import build_parser
    from abra.data import datasets as data
    from abra.model import build
    from abra.model.layers import adapted_linear_layers, freeze_all
    from abra.model.svft import apply_svft

    golden.make_data(tmp_path)
    args = build_parser().parse_args([
        "train-expert", "--exp_name", "t", "--output_root", str(tmp_path / "o"), "--device", "cpu",
        "--weights", str(WEIGHTS / "groundingdino_swint_ogc.pth"), "--root", str(tmp_path / "cityscapes_synth"),
        "--test_annotations", str(tmp_path / "val.json"), "--train_annotations", str(tmp_path / "train.jsonl"),
        "--label_map", str(tmp_path / "label_map.json"), "--domain", "unused", "--fix_layer0_grad", str(int(fix))])
    args.coco_val_path = args.test_annotations
    build.merge_model_config(args)
    torch.manual_seed(0)
    model, _ = build.build_model(args)
    build.load_base_weights(model, args.weights)
    freeze_all(model)
    layers = adapted_linear_layers(model, "encoder_attn")
    apply_svft(model, layers, 2)
    criterion = build.build_criterion(args)
    samples, targets = misc.collate_fn([data.train_dataset(args)[0]])
    model.train()
    captions = [t["caption"] for t in targets]
    outputs = model(samples, captions=captions)
    loss_dict = criterion(outputs, [{k: v for k, v in t.items() if torch.is_tensor(v)} for t in targets],
                          [t["cap_list"] for t in targets], captions)
    sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict if k in criterion.weight_dict).backward()
    return sorted({".".join(n.split(".")[2:4]) for n in layers if model.get_submodule(n).s.grad is None})


def test_layer0_has_no_gradient_with_original_checkpointing(tmp_path):
    assert _layers_without_grad(tmp_path, fix=False) == ["fusion_layers.0", "layers.0"]


def test_fix_layer0_grad(tmp_path):
    assert _layers_without_grad(tmp_path, fix=True) == []
