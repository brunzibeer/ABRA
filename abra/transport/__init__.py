"""Transports: map a source class residual tau_S onto the target domain expert theta_T.

Every transport is a function ``(residual, ctx, **options) -> {layer: R}`` where R is the residual to
add to theta_T, so that the evaluated weights are ``theta_T + alpha * R``. Transports only need
weights (no data, no model) and run on CPU in float64.

To add a method, write a function in ``methods.py`` and decorate it with ``@register("name")``.
"""
from abra.transport.base import TRANSPORTS, TransportContext, register, transport  # noqa: F401
from abra.transport import methods  # noqa: F401  (registers the built-in methods)
