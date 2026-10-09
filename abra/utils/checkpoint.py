def load_checked(model, state_dict, name, allow_unexpected=False):
    """``model.load_state_dict(strict=False)`` that refuses to silently ignore the given weights.

    Missing keys are expected (checkpoints hold only a subset of the model). Unexpected keys mean the
    checkpoint does not match the model, so they raise unless ``allow_unexpected`` (then they are reported).
    """
    result = model.load_state_dict(state_dict, strict=False)
    loaded = len(state_dict) - len(result.unexpected_keys)
    print(f"[load] {name}: {loaded}/{len(state_dict)} tensors loaded")
    if result.unexpected_keys:
        msg = (f"[load] {name}: {len(result.unexpected_keys)} keys do not exist in the model, "
               f"e.g. {result.unexpected_keys[:3]}")
        if not allow_unexpected:
            raise RuntimeError(msg)
        print(msg)
    return result
