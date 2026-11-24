from __future__ import annotations

import numpy as np

from captcha2text.images import preprocess, resize_bilinear
from captcha2text.model import ctc_greedy, ids_greedy
from captcha2text.ratelimit import RateLimiter


def test_downsample_by_two_averages_blocks():
    img = np.arange(16, dtype=np.float32).reshape(4, 4, 1)
    out = resize_bilinear(img, 2, 2)[..., 0]
    expected = np.array([[2.5, 4.5], [10.5, 12.5]], dtype=np.float32)
    np.testing.assert_allclose(out, expected)


def test_same_size_is_identity():
    img = np.random.default_rng(0).integers(0, 256, (50, 100, 3)).astype(np.uint8)
    np.testing.assert_allclose(resize_bilinear(img, 50, 100), img.astype(np.float32))


def test_preprocess_shapes_and_range():
    rgb = np.full((60, 160, 3), 255, dtype=np.uint8)
    x = preprocess(rgb, 100, 50)
    assert x.shape == (1, 50, 100, 3) and x.dtype == np.float32
    assert np.allclose(x, 1.0)
    assert preprocess(rgb, 128, 48, channels=1, layout="NCHW").shape == (1, 1, 48, 128)


def test_preprocess_keeps_aspect_ratio_for_auto_width():
    rgb = np.full((60, 160, 3), 255, dtype=np.uint8)
    x = preprocess(rgb, None, 64, channels=1, layout="NCHW", resample="lanczos", mean=0.5, std=0.5)
    assert x.shape == (1, 1, 64, 170)  # int(160 * 64 / 60)
    assert np.allclose(x, 1.0)  # white is 1.0 after (1 - 0.5) / 0.5


def test_ctc_mask_rules_out_classes():
    logits = np.array([[0, 1, 5, 2]], dtype=np.float32)  # blank, a, b, c; b is best
    mask = np.array([True, True, False, True])
    assert ctc_greedy(logits, "abc", mask)[0] == "c"


def _onehot(classes: list[int], n: int, p: float = 0.9) -> np.ndarray:
    logits = np.full((len(classes), n), 0.0, dtype=np.float32)
    for t, k in enumerate(classes):
        logits[t, k] = 10.0
    return logits


def test_ctc_collapses_repeats_and_blanks():
    charset = "ab"
    # path: a a _ a b b _  -> "aab"
    text, conf = ctc_greedy(_onehot([1, 1, 0, 1, 2, 2, 0], 3), charset)
    assert text == "aab"
    assert 0.99 < conf <= 1.0


def test_ctc_all_blank_is_empty():
    assert ctc_greedy(_onehot([0, 0, 0], 3), "ab") == ("", 0.0)


def test_ids_stop_at_end_token():
    tokens = ["", "<start>", "<end>", "k", "e", "r", "y"]
    ids = np.array([3, 4, 5, 5, 6, 2, 2])
    probs = np.array([0.9, 1, 1, 1, 1, 0.5, 1], dtype=np.float32)
    text, conf = ids_greedy(ids, probs, tokens, "<end>")
    assert text == "kerry"
    assert abs(conf - 0.45) < 1e-6


def test_rate_limiter_refills():
    rl = RateLimiter()
    assert rl.check(1, rpm=60, now=0.0) is None
    for _ in range(59):
        rl.check(1, rpm=60, now=0.0)
    wait = rl.check(1, rpm=60, now=0.0)
    assert wait is not None and 0.9 < wait <= 1.0
    assert rl.check(1, rpm=60, now=1.0) is None
    assert rl.check(2, rpm=0) is None
