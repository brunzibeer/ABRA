import numpy as np
import torch


def compute_task_vector(
    pretrain: dict[str, torch.Tensor],
    transported: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Compute the task vector as the element-wise difference between the
    transported (fine-tuned) weights and the pretrained weights.

    Keys that exist only in *transported* (new heads, adapters, …) are kept
    as-is so they are still applied when the task vector is scaled.
    Keys that exist only in *pretrain* are intentionally ignored: they were
    not touched by the transporter and carry no task signal.

    All output tensors are moved to *device* so downstream arithmetic and
    model loading are always on the same device.

    Args:
        pretrain:    Clean pretrained state-dict.
        transported: Transported / fine-tuned state-dict.
        device:      Target device for all output tensors.

    Returns:
        A state-dict-like mapping from parameter name to task-vector tensor,
        with every tensor residing on *device*.
    """
    task_vector: dict[str, torch.Tensor] = {}
    for key, t_param in transported.items():
        t_param = t_param.to(device).float()
        if key in pretrain:
            p_param = pretrain[key].to(device).float()
            task_vector[key] = t_param - p_param
        else:
            # Parameter introduced by the transporter – include it unchanged
            task_vector[key] = t_param
    return task_vector


def apply_alpha(
    pretrain: dict[str, torch.Tensor],
    task_vector: dict[str, torch.Tensor],
    alpha: float,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Merge the pretrained weights with a scaled task vector.

    merged[key] = pretrain[key] + alpha * task_vector[key]

    For keys that have no pretrain counterpart (introduced by the transporter)
    the formula reduces to:  merged[key] = alpha * task_vector[key].

    All output tensors are moved to *device* so the result can be loaded
    directly into a model that lives on that device.

    Args:
        pretrain:    Clean pretrained state-dict.
        task_vector: Task vector returned by :func:`compute_task_vector`.
                     Its tensors are already expected on *device*.
        alpha:       Scalar that controls the strength of the task vector.
        device:      Target device for all output tensors.

    Returns:
        A merged state dict with every tensor on *device*, ready to be
        loaded into the model.
    """
    merged: dict[str, torch.Tensor] = {}
    for key, tv_param in task_vector.items():
        # tv_param is already on device (produced by compute_task_vector)
        if key in pretrain:
            p_param = pretrain[key].to(device).float()
            merged[key] = p_param + alpha * tv_param
        else:
            merged[key] = alpha * tv_param
    return merged


def run_alpha_search(
    args,
    pretrain: dict[str, torch.Tensor],
    transported: dict[str, torch.Tensor],
    model: torch.nn.Module,
    post_proc,
    test_loader,
    base_dataset,
    device: torch.device,
    evaluate_fn,
) -> float:
    """Search over alpha values and return the one that maximises validation AP.

    For every alpha in [min_alpha, max_alpha) (step = alpha_step) the function:
      1. Builds merged weights = pretrain + alpha * task_vector
      2. Loads them into *model* (in-place, strict=False)
      3. Evaluates on the validation set
      4. Tracks the best alpha / AP pair

    After the loop the model is left with the *best* merged weights already
    loaded, ready for the final evaluation that follows in main().

    Args:
        args:         Parsed CLI arguments (needs min_alpha, max_alpha,
                      alpha_step, epochs, exp_out).
        pretrain:     Clean pretrained state-dict.
        transported:  Transported state-dict (output of transporter.transport()).
        model:        The model whose weights will be updated in-place.
        post_proc:    Post-processor passed to evaluate_fn.
        test_loader:  Validation DataLoader.
        base_dataset: COCO base dataset for evaluation.
        device:       Target device.
        evaluate_fn:  The ``evaluate`` function from engine.py.

    Returns:
        The best alpha found during the search.
    """
    task_vector = compute_task_vector(pretrain, transported, device)

    alphas = np.arange(args.min_alpha, args.max_alpha, args.alpha_step).tolist()

    best_alpha: float = alphas[0]
    last_ap: float = -float("inf")
    best_ap: float = -float("inf")
    steps_without_improvement: int = 0

    print(f"\n{'='*60}")
    print(f"Alpha search over {len(alphas)} values: "
          f"[{args.min_alpha:.2f}, {args.max_alpha:.2f}) step={args.alpha_step:.2f}")
    if args.alpha_patience is not None:
        print(f"Early stopping patience: {args.alpha_patience} steps")
    print(f"{'='*60}")

    for alpha in alphas:
        merged_weights = apply_alpha(pretrain, task_vector, alpha, device)
        model.load_state_dict(merged_weights, strict=False)

        eval_stats = evaluate_fn(
            args=args,
            model=model,
            postprocessors=post_proc,
            loader=test_loader,
            base_ds=base_dataset,
            epoch=args.epochs,
            max_epoch=args.epochs,
            device=device,
            output_root=args.exp_out,
        )
        ap = eval_stats["coco_eval_bbox"][0]

        if ap > best_ap:
            best_ap = ap
            best_alpha = alpha
            steps_without_improvement = 0
            marker = " ◀ best"
        else:
            if ap < last_ap:
                steps_without_improvement += 1
            marker = ""

        last_ap = ap

        print(f"  alpha={alpha:.3f}  AP={ap:.4f}{marker}")

        if args.alpha_patience is not None and steps_without_improvement >= args.alpha_patience:
            print(f"\n  Early stopping: no improvement for {args.alpha_patience} consecutive steps.")
            break

    print(f"\nBest alpha: {best_alpha:.3f}  (AP={best_ap:.4f})")
    print(f"{'='*60}\n")

    # Leave the model loaded with the best weights for the caller
    best_weights = apply_alpha(pretrain, task_vector, best_alpha, device)
    model.load_state_dict(best_weights, strict=False)

    return best_alpha