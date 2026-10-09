"""Training loop shared by domain experts (Objectification) and class experts (SVFT)."""
import copy

import torch
from tqdm import tqdm

from abra.eval.evaluate import evaluate
from abra.train.optim import PER_ITERATION


def _weighted_sum(loss_dict, weight_dict):
    return sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)


def train_one_epoch(model, criterion, loader, optimizer, scheduler, args, device, epoch, logger):
    """One epoch. With ``--objectify`` every box is labelled "object" and the caption is "object .";
    otherwise captions come from the dataset (class names + sampled negatives)."""
    model.train()
    criterion.train()
    weight_dict = criterion.weight_dict
    per_iteration = scheduler is not None and args.scheduler in PER_ITERATION

    for step, (samples, targets) in enumerate(tqdm(loader, desc=f"Train [{epoch}/{args.epochs}]", colour="green")):
        samples = samples.to(device)
        targets_dev = [{k: v.to(device) for k, v in t.items() if torch.is_tensor(v)} for t in targets]
        loss_specific = torch.tensor(0.0)
        loss_generic = 0.0

        if not args.objectify:
            captions = [t["caption"] for t in targets]
            cap_list = [t["cap_list"] for t in targets]
            outputs = model(samples, captions=captions)
            loss_specific = _weighted_sum(criterion(outputs, targets_dev, cap_list, captions), weight_dict)
            (loss_specific * args.class_loss_w).backward()
        else:
            captions = ["object ." for _ in targets]
            cap_list = [["object"] for _ in targets]
            targets_obj = []
            for t in targets_dev:
                t = {k: v.clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v) for k, v in t.items()}
                if "labels" in t:
                    t["labels"] = torch.zeros_like(t["labels"])  # "object" is token 0 of the caption
                targets_obj.append(t)
            outputs = model(samples, captions=captions)
            losses = _weighted_sum(criterion(outputs, targets_obj, cap_list, captions), weight_dict)
            (losses * args.object_loss_w).backward()
            loss_generic = losses.item()

        record = {"TRAIN/epoch": epoch,
                  "TRAIN/total_loss": loss_specific.item() * args.class_loss_w + loss_generic * args.object_loss_w,
                  "TRAIN/loss_specific": loss_specific.item()}
        if args.objectify:
            record["TRAIN/loss_generic"] = loss_generic
        logger.log(record, step=int((epoch + step / len(loader)) * 1000))

        if (step + 1) % args.accumulation_steps == 0:
            if args.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
            optimizer.step()
            if per_iteration:
                scheduler.step()
            optimizer.zero_grad()

        if args.debug and (step + 1) % 5 == 0:
            print("DEBUG: training epoch stopped after 5 batches")
            break


def fit(model, criterion, postprocessors, optimizer, scheduler, loaders, args, device, logger, save):
    """Train for ``args.epochs`` epochs, keeping ``best`` and ``last`` checkpoints via ``save(name, ap)``.

    ``loaders``: dict with ``train``, ``test`` (+ coco ``test_gt``) and optionally ``val`` (+ ``val_gt``).
    Model selection uses ``val`` when given, otherwise ``test`` (the original behaviour, which selects on
    the test split and is flagged in the logs).

    Default order (original code): evaluate *before* each epoch, so the model after the last epoch is
    only evaluated at the end and never selected as best. ``--eval_after_epoch`` evaluates after each epoch
    instead (every epoch's model, including the last, can be selected).
    """
    sel, sel_gt = ("val", "val_gt") if "val" in loaders else ("test", "test_gt")
    if sel == "test":
        print("[select] no --val_annotations: model selection uses the TEST split (original behaviour)")
    best_ap = -float("inf")
    per_epoch_scheduler = scheduler is not None and args.scheduler not in PER_ITERATION

    def select(epoch):
        nonlocal best_ap
        stats = evaluate(model, postprocessors, loaders[sel], loaders[sel_gt], args, device, logger,
                         epoch=epoch, max_epoch=args.epochs, tag=f"SELECT_{sel.upper()}")
        logger.log({"select/split": sel, "select/AP": stats["AP"], "epoch": epoch})
        return stats["AP"] / 100

    for epoch in range(args.epochs):
        if not args.eval_after_epoch:
            ap = select(epoch) if epoch % args.test_period == 0 else None
            if epoch != 0 and ap is not None and ap > best_ap:
                best_ap = ap
                save("best", ap)
            save("last", ap)

        train_one_epoch(model, criterion, loaders["train"], optimizer, scheduler, args, device, epoch, logger)
        if per_epoch_scheduler:
            scheduler.step()
        logger.log({"TRAIN/LR": optimizer.param_groups[0]["lr"], "epoch": epoch})

        if args.eval_after_epoch:
            last = epoch == args.epochs - 1
            ap = select(epoch + 1) if ((epoch + 1) % args.test_period == 0 or last) else None
            if ap is not None and ap > best_ap:
                best_ap = ap
                save("best", ap)
            save("last", ap)

    final = evaluate(model, postprocessors, loaders["test"], loaders["test_gt"], args, device, logger,
                     epoch=args.epochs, max_epoch=args.epochs, tag="TEST")
    if not args.eval_after_epoch:
        save("last", None)
    logger.log({"final/test_AP_last_epoch": final["AP"], "best/select_AP": best_ap * 100})
    return final
