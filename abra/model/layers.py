"""Which modules are adapted, and the optimizer parameter groups."""
import re

from torch import nn

LAYER_SELECTORS = {
    "encoder": r"\b[\w\.]*transformer\.encoder[\w\.]*\b",
    "encoder_attn": r"^(?=.*transformer\.encoder)(?=.*attn)[\w\.]+$",
    "decoder": r"^[\w\.]*transformer\.decoder[\w\.]*$",
    "decoder_attn": r"^(?=.*transformer\.decoder)(?=.*attn)[\w\.]+$",
    "attn": r"^(?=.*attn)[\w\.]+$",
    "all": r".+",
}


def selected_modules(model, selector):
    """Names of all modules (containers included) matched by ``selector``, in ``named_modules`` order."""
    pattern = LAYER_SELECTORS[selector]
    return [n for n, _ in model.named_modules() if re.fullmatch(pattern, n)]


def adapted_linear_layers(model, selector):
    """The nn.Linear layers that SVFT adapts (and that class residuals are defined on)."""
    return [n for n in selected_modules(model, selector) if isinstance(model.get_submodule(n), nn.Linear)]


def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_modules(model, names):
    for n in names:
        for p in model.get_submodule(n).parameters():
            p.requires_grad = True


def trainable_keys(model):
    return [n for n, p in model.named_parameters() if p.requires_grad]


def _match(name, keywords):
    return any(k in name for k in keywords)


def parameter_groups(model, args):
    """Optimizer groups: ``default`` puts ``--lp_names`` params at ``--lp_lr``; ``adapt`` adds a backbone group."""
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    lp = args.lp_names
    if args.param_type == "all":
        return [{"params": [p for _, p in params], "lr": args.lr}]
    if args.param_type == "default":
        return [{"params": [p for n, p in params if not _match(n, lp)], "lr": args.lr},
                {"params": [p for n, p in params if _match(n, lp)], "lr": args.lp_lr}]
    if args.param_type == "adapt":
        return [{"params": [p for n, p in params if not _match(n, lp) and "backbone" in n and "transformer" not in n],
                 "lr": args.backbone_lr},
                {"params": [p for n, p in params if not _match(n, lp) and "transformer" in n and "backbone" not in n],
                 "lr": args.lr},
                {"params": [p for n, p in params if _match(n, lp)], "lr": args.lp_lr}]
    raise ValueError(args.param_type)
