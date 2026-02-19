"""Unit tests for oracle_reduction.comparator."""

from __future__ import annotations

import imagehash
import numpy as np
from PIL import Image

from oracle_reduction.comparator import compare, find_nearest_canonical
from oracle_reduction.hasher import compute_phash
from oracle_reduction.models import Outcome


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_gradient(width: int, height: int) -> Image.Image:
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    for row in range(height):
        val = int(row * 255 / max(height - 1, 1))
        arr[row, :] = [val, 80, 40]
    return Image.fromarray(arr, mode="RGB")


def make_noise(width: int, height: int, seed: int = 99) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


# ---------------------------------------------------------------------------
# find_nearest_canonical
# ---------------------------------------------------------------------------


class TestFindNearest:
    def test_empty_stored_returns_none(self):
        phash = compute_phash(make_gradient(64, 64))
        best_id, best_dist = find_nearest_canonical(phash, [])
        assert best_id is None
        assert best_dist is None

    def test_finds_closest_match(self):
        original = make_gradient(256, 256)
        similar = original.resize((128, 128), Image.LANCZOS)
        unrelated = make_noise(256, 256, seed=7)

        h_orig = compute_phash(original)
        h_similar = compute_phash(similar)
        h_unrelated = compute_phash(unrelated)

        stored = [
            (1, str(h_orig)),
            (2, str(h_unrelated)),
        ]
        best_id, best_dist = find_nearest_canonical(h_similar, stored)
        assert best_id == 1
        assert best_dist <= 10


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


class TestCompare:
    def _phash_str(self, img: Image.Image) -> str:
        return str(compute_phash(img))

    def test_identical_when_crypto_exists(self):
        img = make_gradient(64, 64)
        phash = compute_phash(img)
        result = compare(
            crypto_hash="deadbeef",
            phash=phash,
            stored_phashes=[],
            crypto_exists=True,
            threshold=10,
        )
        assert result.outcome is Outcome.IDENTICAL
        assert result.canonical_id is None

    def test_new_when_no_stored_images(self):
        img = make_gradient(64, 64)
        phash = compute_phash(img)
        result = compare(
            crypto_hash="aabbccdd",
            phash=phash,
            stored_phashes=[],
            crypto_exists=False,
            threshold=10,
        )
        assert result.outcome is Outcome.NEW
        assert result.canonical_id is None

    def test_variant_when_phash_within_threshold(self):
        original = make_gradient(256, 256)
        half = original.resize((128, 128), Image.LANCZOS)

        h_orig = compute_phash(original)
        h_half = compute_phash(half)

        stored = [(42, str(h_orig))]
        result = compare(
            crypto_hash="unique_hash",
            phash=h_half,
            stored_phashes=stored,
            crypto_exists=False,
            threshold=10,
        )
        assert result.outcome is Outcome.VARIANT
        assert result.canonical_id == 42
        assert result.phash_distance is not None
        assert result.phash_distance <= 10

    def test_new_when_phash_beyond_threshold(self):
        gradient = make_gradient(256, 256)
        noise = make_noise(256, 256, seed=1)

        h_gradient = compute_phash(gradient)
        h_noise = compute_phash(noise)

        stored = [(1, str(h_gradient))]
        result = compare(
            crypto_hash="unique_hash_noise",
            phash=h_noise,
            stored_phashes=stored,
            crypto_exists=False,
            threshold=10,
        )
        assert result.outcome is Outcome.NEW

    def test_threshold_boundary(self):
        """An exact threshold match is still a VARIANT; one over is NEW."""
        original = make_gradient(256, 256)
        h_orig = compute_phash(original)
        stored = [(1, str(h_orig))]

        # distance == 0 → well within threshold → VARIANT
        result = compare("x", h_orig, stored, False, threshold=10)
        assert result.outcome is Outcome.VARIANT
        assert result.phash_distance == 0
