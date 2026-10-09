"""COCO-style evaluation of a Grounding DINO model on one test set.

The text prompt is built from the categories of the test annotations ("bus ." for a single-class split).
"""
from pathlib import Path

import torch
from tqdm import tqdm

from groundingdino.datasets.coco_eval import CocoEvaluator
from groundingdino.util.utils import to_device
from abra.eval.export import save_predictions_as_coco
from abra.eval.visualize import visualize_predictions

MAX_VISUALIZED = 20


@torch.no_grad()
def evaluate(model, postprocessors, loader, coco_gt, args, device, logger=None, epoch=0, max_epoch=0, tag="EVAL"):
    """Run inference on ``loader`` and return COCO stats (in %) plus per-category AP rows."""
    model.eval()
    cat_ids = sorted(coco_gt.getCatIds(catNms=args.eval_classes) if getattr(args, "eval_classes", None)
                     else coco_gt.getCatIds())
    if not cat_ids:
        raise ValueError(f"none of --eval_classes {args.eval_classes} is a category of the test annotations")
    cat_list = [c["name"] for c in coco_gt.loadCats(cat_ids)]
    caption = " . ".join(cat_list) + " ."
    evaluator = CocoEvaluator(coco_gt=coco_gt, iou_types=("bbox",), useCats=True, class_names=cat_list)
    evaluator.coco_eval["bbox"].params.catIds = cat_ids

    vis_count, vis_dir = 0, Path(args.output_root) / "visualization" / f"epoch_{epoch}"
    if args.visualize_predictions:
        vis_dir.mkdir(parents=True, exist_ok=True)
    export_targets, export_results = [], []

    for step, (samples, targets) in enumerate(tqdm(loader, desc=f"Test [{epoch}/{max_epoch}]", colour="blue")):
        samples = samples.to(device)
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]
        outputs = model(samples, captions=[caption] * samples.tensors.shape[0])
        orig_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors["bbox"](outputs, orig_sizes)
        evaluator.update({t["image_id"].item(): r for t, r in zip(targets, results)})

        if args.visualize_predictions and vis_count < MAX_VISUALIZED:
            visualize_predictions(targets, results, cat_list, vis_dir, vis_count, len(targets), loader.dataset)
            vis_count += len(targets)
        if args.save_coco_format:
            export_targets.extend({k: v.cpu() if torch.is_tensor(v) else v for k, v in t.items()} for t in targets)
            export_results.extend(results)
        if args.debug and (step + 1) % 5 == 0:
            print("DEBUG: evaluation stopped after 5 batches")
            break

    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()

    if args.save_coco_format:
        out = Path(args.save_coco_format)
        out = out if out.suffix == ".json" else out.with_suffix(".json")
        out.parent.mkdir(parents=True, exist_ok=True)
        save_predictions_as_coco(export_targets, export_results, cat_list, loader.dataset, out,
                                 confidence_threshold=args.zs_th)

    stats = evaluator.coco_eval["bbox"].stats.tolist()
    result = {"AP": stats[0] * 100, "AP50": stats[1] * 100, "AP75": stats[2] * 100,
              "APs": stats[3] * 100, "APm": stats[4] * 100, "APl": stats[5] * 100,
              "coco_stats": stats, "per_category": evaluator.per_category["bbox"]}
    if logger is not None:
        logger.log({f"{tag}/{k}": result[k] for k in ("AP", "AP50", "AP75", "APs", "APm", "APl")} | {"epoch": epoch})
        logger.log_table(f"{tag}/per_class_AP", ["category", "AP", "AP50", "AP75"], result["per_category"])
    return result
