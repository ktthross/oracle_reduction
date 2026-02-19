"""Unit tests for oracle_reduction.hasher."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from oracle_reduction.hasher import (
    compute_crypto_hash,
    compute_phash,
    normalise,
    phash_from_str,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_solid(width: int, height: int, color: tuple[int, int, int] = (128, 64, 32)) -> Image.Image:
    arr = np.full((height, width, 3), color, dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def make_gradient(width: int, height: int) -> Image.Image:
    """Gradient from dark-red at top to bright-red at bottom."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    for row in range(height):
        val = int(row * 255 / (height - 1))
        arr[row, :] = [val, 80, 40]
    return Image.fromarray(arr, mode="RGB")


def make_noise(width: int, height: int, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


# ---------------------------------------------------------------------------
# normalise
# ---------------------------------------------------------------------------


def test_normalise_converts_to_rgb():
    rgba = Image.new("RGBA", (10, 10), (100, 150, 200, 128))
    result = normalise(rgba)
    assert result.mode == "RGB"


# ---------------------------------------------------------------------------
# compute_crypto_hash
# ---------------------------------------------------------------------------


class TestCryptoHash:
    def test_identical_images_produce_same_hash(self):
        img = make_solid(64, 64)
        assert compute_crypto_hash(img) == compute_crypto_hash(img)

    def test_different_images_produce_different_hashes(self):
        img1 = make_solid(64, 64, color=(10, 20, 30))
        img2 = make_solid(64, 64, color=(200, 100, 50))
        assert compute_crypto_hash(img1) != compute_crypto_hash(img2)

    def test_hash_is_64_char_hex(self):
        img = make_solid(32, 32)
        h = compute_crypto_hash(img)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_different_sizes_same_content_different_hashes(self):
        # Different pixel counts → different raw bytes → different hash
        small = make_solid(32, 32, color=(100, 100, 100))
        large = make_solid(64, 64, color=(100, 100, 100))
        assert compute_crypto_hash(small) != compute_crypto_hash(large)


# ---------------------------------------------------------------------------
# compute_phash
# ---------------------------------------------------------------------------


class TestPHash:
    def test_phash_roundtrip_via_string(self):
        img = make_gradient(128, 128)
        h = compute_phash(img)
        restored = phash_from_str(str(h))
        assert (h - restored) == 0

    def test_same_image_different_resolution_low_distance(self):
        """Downscaled version of the same image must be within threshold 10."""
        original = make_gradient(256, 256)
        half = original.resize((128, 128), Image.LANCZOS)
        h_orig = compute_phash(original)
        h_half = compute_phash(half)
        assert (h_orig - h_half) <= 10

    def test_different_images_high_distance(self):
        """A structured gradient vs random noise should be clearly distinct."""
        gradient = make_gradient(256, 256)
        noise = make_noise(256, 256, seed=42)
        h_grad = compute_phash(gradient)
        h_noise = compute_phash(noise)
        assert (h_grad - h_noise) > 10

    @pytest.mark.parametrize("scale", [0.25, 0.5, 2.0])
    def test_phash_robust_across_scales(self, scale: float):
        original = make_gradient(256, 256)
        new_size = (int(256 * scale), int(256 * scale))
        resized = original.resize(new_size, Image.LANCZOS)
        h_orig = compute_phash(original)
        h_resized = compute_phash(resized)
        assert (h_orig - h_resized) <= 10
