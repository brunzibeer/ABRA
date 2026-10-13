"""Command line interface: ``python -m abra <command> [options]``.

Commands
    train-domain   train a domain expert with Objectification (class-agnostic "object" boxes)
    train-expert   train a class expert with SVFT on top of a domain expert -> class residual
    eval           evaluate zero-shot / a domain expert / a transported class residual
    transport      transport a class residual offline (CPU, no data) and save it
    migrate        convert original .pth checkpoints to the new formats
    fda            FDA unsupervised domain adaptation baseline (paper Table 2)
    diagnose       weight-space diagnostics D1-D3 and Option-2 layer scales (no data)
    report         collect results.json files into one table
    prepare-data   build the Cityscapes -> Foggy Cityscapes annotation splits (CPU, seconds)
    data-stats     class-partition statistics of a benchmark (images / instances per class and split)
    convert-bdd    BDD-format label files -> COCO json (SDGOD splits)

Run ``python -m abra <command> -h`` for the options of a command.
"""
import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_CFG = REPO / "groundingdino" / "config" / "cfg_odvg.py"


def str2bool(value):
    value = str(value).lower()
    if value in ("1", "true", "t", "yes", "y"):
        return True
    if value in ("0", "false", "f", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean (0/1), got {value!r}")


# ---------------------------------------------------------------------------- argument groups

def add_run_args(p):
    g = p.add_argument_group("run")
    g.add_argument("--exp_name", required=True, help="name of the run (also used for wandb)")
    g.add_argument("--output_root", type=Path, required=True,
                   help="run directory: checkpoints, run.log, metrics.jsonl, config.json")
    g.add_argument("--device", default="cuda")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--num_workers", type=int, default=2)
    g.add_argument("--debug", type=str2bool, default=False, help="stop every train/eval loop after 5 batches")
    g.add_argument("--wandb_project", default=None, help="mirror metrics to wandb (optional)")
    g.add_argument("--wandb_entity", default=None)


def add_model_args(p):
    g = p.add_argument_group("model")
    g.add_argument("--model_cfg", type=Path, default=DEFAULT_MODEL_CFG, help="Grounding DINO config (python file)")
    g.add_argument("--weights", type=Path, required=True, help="base Grounding DINO checkpoint (theta_0)")


def add_test_data_args(p):
    g = p.add_argument_group("test data")
    g.add_argument("--root", type=Path, help="image root (train/ or val/ is appended, see --add_split)")
    g.add_argument("--test_annotations", type=Path, help="COCO json of the test split")
    g.add_argument("--add_split", type=str2bool, default=None,
                   help="append train/ or val/ to --root (default: only for cityscapes, AODRaw, bdd roots)")
    g.add_argument("--fix_size", type=str2bool, default=False)
    g.add_argument("--visualize_predictions", type=str2bool, default=False)
    g.add_argument("--save_coco_format", type=Path, default=None, help="save predictions as COCO json")
    g.add_argument("--zs_th", type=float, default=0.3, help="score threshold for --save_coco_format")
    g.add_argument("--eval_classes", nargs="+", default=None,
                   help="evaluate only these category names (prompt and AP), e.g. one class on a multi-class "
                        "test json = standard all-images protocol")


def add_train_args(p):
    g = p.add_argument_group("training data")
    g.add_argument("--train_annotations", type=Path, required=True, help="ODVG jsonl")
    g.add_argument("--label_map", type=Path, required=True, help="label id -> class name json")
    g.add_argument("--val_annotations", type=Path, default=None,
                   help="COCO json used for model selection (default: --test_annotations, original behaviour)")
    add_optim_args(p)


def add_optim_args(p):
    g = p.add_argument_group("optimisation")
    g.add_argument("--trainable", default="encoder_attn",
                   choices=["encoder", "decoder", "encoder_attn", "decoder_attn", "attn", "all"])
    g.add_argument("--param_type", default="default", choices=["all", "default", "adapt"])
    g.add_argument("--epochs", type=int, default=10)
    g.add_argument("--batch_size", type=int, default=2)
    g.add_argument("--accumulation_steps", type=int, default=1)
    g.add_argument("--optim", default="adamw", choices=["adamw", "sgd"])
    g.add_argument("--lr", type=float, default=1e-4)
    g.add_argument("--lp_names", nargs="+", default=["ref_point_head", "sampling_offsets"],
                   help="parameters (name substrings) trained with --lp_lr")
    g.add_argument("--lp_lr", type=float, default=1e-5)
    g.add_argument("--backbone_lr", type=float, default=1e-5)
    g.add_argument("--weight_decay", type=float, default=1e-4)
    g.add_argument("--momentum", type=float, default=0.0)
    g.add_argument("--nesterov", type=str2bool, default=False)
    g.add_argument("--scheduler", default=None, choices=["step", "multi", "cosine", "sequential"])
    g.add_argument("--step_size", type=int, default=None)
    g.add_argument("--step_milestones", type=int, nargs="+", default=[8, 11])
    g.add_argument("--warmup_ratio", type=float, default=0.03)
    g.add_argument("--test_period", type=int, default=1, help="evaluate every N epochs")
    g.add_argument("--eval_after_epoch", type=str2bool, default=False,
                   help="evaluate/select after each epoch (default: before, as the original code; "
                        "then the final epoch is never selected as best)")
    g.add_argument("--fix_layer0_grad", type=str2bool, default=False,
                   help="non-reentrant gradient checkpointing so that encoder layer 0 is trained (the original code "
                        "never updated it; default off for reproducibility, see docs/MIGRATION.md)")
    g.add_argument("--class_loss_w", type=float, default=1.0, help="weight of the class loss")
    g.add_argument("--object_loss_w", type=float, default=1.0, help="weight of the 'object' loss (Objectification)")


def add_transport_args(p, residual_required):
    g = p.add_argument_group("transport")
    g.add_argument("--residual", type=Path, required=residual_required, help="class residual (source class expert)")
    g.add_argument("--source_domain", type=Path, default=None,
                   help="parent domain expert of --residual (only if the path stored in the residual is not valid here)")
    g.add_argument("--method", default="abra", choices=["abra", "param_delta", "task_addition"],
                   help="abra (rotation), param_delta (Param Delta baseline) or task_addition (theta_T + tau)")
    g.add_argument("--abra_mask", default=None, choices=["plain", "banded"])
    g.add_argument("--alpha_estimate", type=str2bool, default=False,
                   help="ABRA: scale each layer by ||sigma_T|| / ||sigma_S||")


def build_parser():
    parser = argparse.ArgumentParser(prog="abra", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train-domain", help="train a domain expert (Objectification)")
    add_run_args(p)
    add_model_args(p)
    add_test_data_args(p)
    add_train_args(p)
    p.add_argument("--init", type=Path, default=None, help="start from this domain expert instead of theta_0")
    p.add_argument("--objectify", type=str2bool, default=True,
                   help="train with class-agnostic 'object' boxes (default; 0 trains on the label map classes)")

    p = sub.add_parser("train-expert", help="train a class expert (SVFT residual on a domain expert)")
    add_run_args(p)
    add_model_args(p)
    add_test_data_args(p)
    add_train_args(p)
    p.add_argument("--domain", type=Path, required=True, help="parent domain expert")
    p.add_argument("--svft_diag", type=int, default=2, help="SVFT band half-width (off-diagonals)")
    p.add_argument("--svft_zero_init", type=str2bool, default=False,
                   help="start the SVFT band at zero (expert = parent at step 0) instead of the original random init")
    p.add_argument("--residual_encoding", default="svft_band", choices=["svft_band", "dense"])

    p = sub.add_parser("eval", help="evaluate a model on a test split")
    add_run_args(p)
    add_model_args(p)
    add_test_data_args(p)
    add_transport_args(p, residual_required=False)
    p.add_argument("--domain", type=Path, default=None, help="domain expert to evaluate (target domain for transports)")
    p.add_argument("--alpha", type=float, default=1.0, help="scale of the transported residual")
    p.add_argument("--alpha_file", type=Path, default=None,
                   help="per-layer scales (json layer -> alpha, e.g. from 'abra diagnose'); overrides --alpha")
    p.add_argument("--alpha_search", type=str2bool, default=False,
                   help="ABRA-alpha: pick alpha by AP on --alpha_annotations (default: the test split = oracle)")
    p.add_argument("--alpha_annotations", type=Path, default=None, help="COCO json used to select alpha")
    p.add_argument("--min_alpha", type=float, default=0.1)
    p.add_argument("--max_alpha", type=float, default=5.0)
    p.add_argument("--alpha_step", type=float, default=0.1)
    p.add_argument("--alpha_patience", type=int, default=5)

    p = sub.add_parser("transport", help="transport a class residual offline and save it (dense encoding)")
    add_transport_args(p, residual_required=True)
    p.add_argument("--target_domain", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="output .safetensors")
    p.add_argument("--device", default="cpu", help="device for the legacy Param Delta SVD")

    p = sub.add_parser("migrate", help="convert original .pth checkpoints to the new formats")
    p.add_argument("--domain", type=Path, nargs="*", default=[], help="original domain experts (.pth)")
    p.add_argument("--expert", type=Path, nargs="*", default=[], help="original class experts (.pth)")
    p.add_argument("--parent", type=Path, default=None, help="parent domain expert of --expert (.pth or new)")
    p.add_argument("--out_dir", type=Path, required=True)

    from abra.baselines.fda import add_fda_args
    p = sub.add_parser("fda", help="FDA unsupervised domain adaptation baseline (Table 2)")
    add_run_args(p)
    add_model_args(p)
    add_fda_args(p)
    add_optim_args(p)

    from abra.diagnostics.command import add_diagnose_args
    add_diagnose_args(sub.add_parser("diagnose", help="weight-space diagnostics D1-D3 (no data)"))
    from abra.eval.report import add_report_args
    add_report_args(sub.add_parser("report", help="collect results.json files into one table"))
    from abra.data.prepare import add_prepare_args
    add_prepare_args(sub.add_parser("prepare-data", help="build the Cityscapes -> Foggy Cityscapes annotation splits"))
    from abra.data.stats import add_stats_args
    add_stats_args(sub.add_parser("data-stats", help="class-partition statistics of a benchmark"))
    from abra.data.bdd import add_bdd_args
    add_bdd_args(sub.add_parser("convert-bdd", help="BDD-format label files -> COCO json (SDGOD)"))
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "diagnose":
        from abra.diagnostics.command import diagnose
        return diagnose(args)
    if args.command == "report":
        from abra.eval.report import report
        return report(args)
    if args.command == "prepare-data":
        from abra.data.prepare import prepare
        return prepare(args)
    if args.command == "data-stats":
        from abra.data.stats import data_stats
        return data_stats(args)
    if args.command == "convert-bdd":
        from abra.data.bdd import convert_bdd
        return convert_bdd(args)
    from abra import commands
    handler = {"train-domain": commands.train_domain, "train-expert": commands.train_expert, "eval": commands.evaluate,
               "transport": commands.transport_offline, "migrate": commands.migrate, "fda": commands.fda}[args.command]
    return handler(args)


if __name__ == "__main__":
    main()
