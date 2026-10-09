"""Built-in transports.

Notation per adapted layer: tau = U_S C V_S^T is the source class residual with SVFT coefficients C in the
basis (U_S, V_S) of the source domain expert theta_S; theta_T = U_T diag(sigma_T) V_T^T. The returned R
is added to theta_T.
"""
import torch

from abra.transport.base import register
from abra.weights.svd import band_mask


@register("task_addition")
def task_addition(residual, ctx):
    """R = tau: add the source residual unchanged (what the original 'analogy' transporter computes)."""
    return {layer: d.double() for layer, d in residual.delta.items()}


@register("abra")
def abra(residual, ctx, mask=None, off_diag=None, norm_preserving=False):
    """Aligned Basis Relocation: R = U_T M(U_T^T tau V_T) V_T^T.

    Without a mask this is the projection of tau onto the target singular subspaces (identical to
    task addition for square layers). ``mask`` restricts the target coefficients to the SVFT band
    (``banded``: same pattern as the source coefficients) or to the diagonal (``plain``).
    ``norm_preserving`` scales each layer by ||sigma_T|| / ||sigma_S|| (original ``--alpha_estimate``).
    """
    out = {}
    for layer, delta in residual.delta.items():
        u_t, sigma_t, vh_t = ctx.target_svd(layer)
        coeffs = u_t.T @ delta.double() @ vh_t.T
        if mask == "banded":
            if residual.coefficients is not None:
                coeffs = coeffs * (residual.coefficients[layer] != 0)
            else:
                coeffs = coeffs * band_mask(coeffs.shape[0], off_diag or residual.info["off_diag"])
        elif mask == "plain":
            coeffs = coeffs * torch.eye(*coeffs.shape, dtype=coeffs.dtype)
        elif mask is not None:
            raise ValueError(f"unknown ABRA mask {mask!r}")
        if norm_preserving:
            sigma_s = residual.source_sigma[layer] if residual.source_sigma else ctx.source_svd(layer)[1]
            coeffs = coeffs * (sigma_t.norm() / (sigma_s.norm() + 1e-8))
        out[layer] = u_t @ coeffs @ vh_t
    return out


@register("param_delta")
def param_delta(residual, ctx):
    """Param Delta baseline: R = U_T C V_T^T, i.e. the source SVFT coefficients reused in the target basis
    without any alignment (the original 'simple' transporter).

    The result depends on the arbitrary sign of every singular vector. For residuals in the new format,
    both bases are canonical (reproducible everywhere). For legacy checkpoints it reproduces the original
    behaviour: the stored coefficients with the float32 SVD of theta_T computed on ``ctx.device``.
    """
    if residual.coefficients is None:
        raise ValueError("param_delta needs SVFT coefficients (dense residuals have none)")
    legacy = residual.info.get("encoding") == "legacy_svft_dense"
    out = {}
    for layer, coeffs in residual.coefficients.items():
        if legacy:
            u_t, _, vh_t = torch.linalg.svd(ctx.target[layer].float().to(ctx.device), full_matrices=False)
            out[layer] = (u_t @ coeffs.to(u_t) @ vh_t).cpu().double()
        else:
            u_t, _, vh_t = ctx.target_svd(layer)
            out[layer] = u_t @ coeffs.double() @ vh_t
    return out
