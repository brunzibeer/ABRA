"""Compose evaluation weights: theta_0 (+ domain expert) (+ alpha * transported residual)."""
import torch

from abra.utils.checkpoint import load_checked


def load_domain_into(model, tensors, name):
    load_checked(model, tensors, name=name)


@torch.no_grad()
def add_residual(model, residual, alpha=1.0):
    """W_layer <- W_layer + alpha_layer * R_layer for every layer of ``residual`` (layer -> tensor).

    ``alpha`` is a float or a dict layer -> float.
    """
    for layer, r in residual.items():
        weight = model.get_submodule(layer).weight
        a = alpha[layer] if isinstance(alpha, dict) else alpha
        weight.copy_(weight + a * r.to(weight))


def snapshot(model, layers):
    """Copy of the current weights of ``layers`` (to restore them, e.g. between alpha values)."""
    return {l: model.get_submodule(l).weight.detach().clone() for l in layers}


@torch.no_grad()
def restore(model, weights):
    for layer, w in weights.items():
        model.get_submodule(layer).weight.copy_(w)

