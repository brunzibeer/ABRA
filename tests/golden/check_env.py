"""Quick check that the software environment can run the golden scripts (GPU, versions, compiled op,
imports, BERT). Prints OK/FAIL per item. Run from the repo root: ``python tests/golden/check_env.py``.
File and image paths are checked by ``run_gpu_golden.py`` itself before it starts."""
import importlib
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
    import abra.commands  # noqa: F401  (imports the whole pipeline, incl. pycocotools)
    import abra.baselines.fda  # noqa: F401
    return None


def bert():
    from transformers import AutoTokenizer, BertModel
    AutoTokenizer.from_pretrained("bert-base-uncased")
    BertModel.from_pretrained("bert-base-uncased")
    return "bert-base-uncased available (HF cache or internet)"


def main():
    check("GPU", gpu)
    check("package versions", versions)
    check("deformable attention CUDA op", deform_op)
    check("repo imports (abra)", repo_imports)
    check("BERT tokenizer/model", bert)
    print("\nall checks passed" if not failures else f"\n{len(failures)} check(s) failed: {', '.join(failures)}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
