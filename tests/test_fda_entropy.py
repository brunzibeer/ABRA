"""FDA entropy loss on Grounding DINO outputs (synthetic logits, no model or tokenizer download needed)."""
import math

import torch

from abra.baselines.fda import target_entropy

CAPTION = "bus . truck ."
CATEGORIES = ["bus", "truck"]


class Encoding:
    """Stand-in for a tokenizer encoding of CAPTION: [CLS] bus . truck . [SEP] -> tokens 0..5."""

    def char_to_token(self, i):
        return {0: 1, 1: 1, 2: 1, 4: 2, 6: 3, 7: 3, 8: 3, 9: 3, 10: 3, 12: 4}.get(i)


def _outputs(token_logits):
    """[B, Q, 6] logits of the 6 caption tokens -> pred_logits padded with -inf to 256 tokens."""
    b, q, _ = token_logits.shape
    logits = torch.full((b, q, 256), float("-inf"))
    logits[..., :6] = token_logits
    return {"pred_logits": logits, "token": [Encoding()] * b}


def _loss(token_logits, ita=2.0, threshold=0.3):
    return target_entropy(_outputs(token_logits), CAPTION, CATEGORIES, ita, threshold)


def test_confident_queries_have_no_entropy():
    logits = torch.tensor([[[0.0, 20.0, 0.0, -20.0, 0.0, 0.0]]])  # bus certain, truck certainly absent
    loss, n = _loss(logits)
    assert n == 1 and loss.item() < 1e-6


def test_uncertain_query_has_maximal_entropy():
    logits = torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])  # both classes at p = 0.5
    loss, n = _loss(logits, ita=1.0)
    assert n == 1 and math.isclose(loss.item(), 1.0, rel_tol=1e-6)


def test_background_queries_are_ignored():
    logits = torch.full((2, 3, 6), -5.0)  # every score sigmoid(-5) < 0.3
    assert _loss(logits) == (None, 0)


def test_gradient_increases_confidence_and_is_finite():
    logits = torch.tensor([[[0.0, 0.4, 0.0, -0.4, 0.0, 0.0]]], requires_grad=True)
    loss, _ = _loss(logits, ita=1.0)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 0, 1] < 0 < logits.grad[0, 0, 3]  # bus pushed up, truck pushed down


def test_ita_is_the_charbonnier_exponent():
    logits = torch.tensor([[[0.0, 1.0, 0.0, -1.0, 0.0, 0.0]]])
    ent = _loss(logits, ita=0.5)[0].item()  # (ent**2 + 1e-8) ** 0.5 ~ ent
    assert math.isclose(_loss(logits, ita=2.0)[0].item(), ent ** 4, rel_tol=1e-4)
