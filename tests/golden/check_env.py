"""Quick check that a machine can run the golden scripts. Prints OK/FAIL per item; takes ~1 minute.

    python tests/golden/check_env.py --gdino_weights /path/to/groundingdino_swint_ogc.pth \
        --fog_root golden_images/cityscapes/fog/images --normal_root golden_images/cityscapes/normal/images \
        --weights_dir /path/to/weights
"""
import argparse
import glob
import importlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
failures = []


def check(name, fn):
    try:
        info = fn()
        print(f"OK    {name}" + (f": {info}" if info else ""))
    except Exception as e:  # report everything, keep going
        failures.append(name)
        print(f"FAIL  {name}: {type(e).__name__}: {e}")


def gpu():
    import torch
    assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
    props = torch.cuda.get_device_properties(0)
    return f"torch {torch.__version__}, CUDA {torch.version.cuda}, {props.name}, {props.total_memory / 2**30:.1f} GiB"


def versions():
    out = []
    for mod, want in [("transformers", "4.28.0"), ("numpy", "1.24"), ("torchvision", "0.16")]:
        v = importlib.import_module(mod).__version__
        out.append(f"{mod} {v}" + ("" if v.startswith(want) else f" (expected {want})"))
    return ", ".join(out)


def deform_op():
    import torch
    import MultiScaleDeformableAttention  # noqa: F401  (compiled from groundingdino/models/GroundingDINO/ops)
    from groundingdino.models.GroundingDINO.ms_deform_attn import multi_scale_deformable_attn_pytorch, MultiScaleDeformableAttnFunction
    shapes = torch.as_tensor([(6, 4), (3, 2)], dtype=torch.long, device="cuda")
    start = torch.cat((shapes.new_zeros((1,)), shapes.prod(1).cumsum(0)[:-1]))
    value = torch.rand(1, int(shapes.prod(1).sum()), 2, 8, device="cuda")
    loc = torch.rand(1, 5, 2, 2, 2, 2, device="cuda")
    attn = torch.rand(1, 5, 2, 2, 2, device="cuda")
    attn = attn / attn.sum(-1, keepdim=True).sum(-2, keepdim=True)
    ref = multi_scale_deformable_attn_pytorch(value, shapes, loc, attn)
    cuda = MultiScaleDeformableAttnFunction.apply(value, shapes, start, loc, attn, 64)
    err = (ref - cuda).abs().max().item()
    assert err < 1e-4, f"CUDA op differs from PyTorch reference by {err}"
    return f"max diff vs PyTorch reference {err:.1e}"


def repo_imports():
    import main  # noqa: F401  (imports the whole pipeline, incl. torch_sparse and pycocotools)
    return None


def bert():
    from transformers import AutoTokenizer, BertModel
    AutoTokenizer.from_pretrained("bert-base-uncased")
    BertModel.from_pretrained("bert-base-uncased")
    return "bert-base-uncased available (HF cache or internet)"


def file_exists(path):
    def fn():
        if not Path(path).exists():
            raise FileNotFoundError(path)
    return fn


def roots_named_cityscapes(*roots):
    def fn():
        bad = [str(r) for r in roots if "cityscapes" not in str(r)]
        assert not bad, f"rename so the path contains 'cityscapes' (else main.py does not append /val): {bad}"
    return fn


def images(root, pattern):
    def fn():
        names = {img["file_name"] for f in glob.glob(str(REPO / "annotations" / pattern)) for img in json.load(open(f))["images"]}
        assert names, f"no annotation files match annotations/{pattern}"
        missing = [n for n in names if not (Path(root) / n).exists()]
        assert not missing, f"{len(missing)}/{len(names)} images missing under {root}, e.g. {missing[0]}"
        return f"{len(names)} images found"
    return fn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gdino_weights", type=Path, required=True)
    p.add_argument("--fog_root", type=Path, required=True)
    p.add_argument("--normal_root", type=Path, required=True)
    p.add_argument("--weights_dir", type=Path, required=True, help="dir with cityscape_normal/, cityscape_fog/, cityscapes_experts/")
    args = p.parse_args()

    check("GPU", gpu)
    check("package versions", versions)
    check("deformable attention CUDA op", deform_op)
    check("repo imports (main.py)", repo_imports)
    check("BERT tokenizer/model", bert)
    check("GDINO base weights", file_exists(args.gdino_weights))
    for sub in ["cityscape_normal/best.pth", "cityscape_fog/best.pth"]:
        check(f"weights/{sub}", file_exists(args.weights_dir / sub))
    check("class experts", lambda: f"{len(glob.glob(str(args.weights_dir / 'cityscapes_experts/*/best.pth')))} found")
    check("'cityscapes' in image root paths", roots_named_cityscapes(args.fog_root, args.normal_root))
    check("fog val images", images(args.fog_root / "val", "fog_val/*/annotations_val.json"))
    check("normal val images", images(args.normal_root / "val", "normal_train/*/annotations_val.json"))
    print("\nall checks passed" if not failures else f"\n{len(failures)} check(s) failed: {', '.join(failures)}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
