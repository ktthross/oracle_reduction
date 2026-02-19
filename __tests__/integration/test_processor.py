"""Integration tests for the full image processing pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from oracle_reduction.config import Config
from oracle_reduction.models import Outcome
from oracle_reduction.processor import ImageProcessor
from oracle_reduction.storage import ImageStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_gradient(width: int, height: int) -> Image.Image:
    """Create a smooth gradient image with recognisable frequency content."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    for row in range(height):
        val = int(row * 255 / max(height - 1, 1))
        arr[row, :] = [val, 80, 40]
    return Image.fromarray(arr, mode="RGB")


def make_noise(width: int, height: int, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def save_png(img: Image.Image, path: Path) -> Path:
    img.save(path, format="PNG")
    return path


@pytest.fixture()
def pipeline(tmp_path: Path):
    """Return a fully wired (config, store, processor) tuple."""
    config = Config(
        phash_threshold=10,
        db_path=str(tmp_path / "test.db"),
        storage_dir=str(tmp_path / "store"),
    )
    store = ImageStore(config.db_path, config.storage_dir)
    processor = ImageProcessor(config, store)
    yield config, store, processor
    store.close()


# ---------------------------------------------------------------------------
# Core workflow
# ---------------------------------------------------------------------------


class TestCoreWorkflow:
    def test_new_image_is_saved_as_canonical(self, pipeline, tmp_path: Path):
        _, store, proc = pipeline
        img_path = save_png(make_gradient(256, 256), tmp_path / "orig.png")

        result = proc.process(img_path)

        assert result.outcome is Outcome.NEW
        assert result.canonical_id is not None
        assert store.get_canonical(result.canonical_id) is not None

    def test_identical_image_is_discarded(self, pipeline, tmp_path: Path):
        _, _, proc = pipeline
        img_path = save_png(make_gradient(256, 256), tmp_path / "orig.png")

        first = proc.process(img_path)
        assert first.outcome is Outcome.NEW

        second = proc.process(img_path)
        assert second.outcome is Outcome.IDENTICAL

    def test_downscaled_variant_is_linked(self, pipeline, tmp_path: Path):
        _, store, proc = pipeline
        orig = make_gradient(256, 256)
        orig_path = save_png(orig, tmp_path / "orig.png")
        half_path = save_png(orig.resize((128, 128), Image.LANCZOS), tmp_path / "half.png")

        r_orig = proc.process(orig_path)
        assert r_orig.outcome is Outcome.NEW

        r_half = proc.process(half_path)
        assert r_half.outcome is Outcome.VARIANT
        assert r_half.canonical_id == r_orig.canonical_id
        assert r_half.variant_id is not None

        variants = store.get_variants_for_canonical(r_orig.canonical_id)
        assert len(variants) == 1
        assert variants[0].id == r_half.variant_id
        assert variants[0].relationship == "downscaled"

    def test_upscaled_variant_is_linked(self, pipeline, tmp_path: Path):
        _, store, proc = pipeline
        orig = make_gradient(128, 128)
        orig_path = save_png(orig, tmp_path / "orig.png")
        large_path = save_png(orig.resize((256, 256), Image.LANCZOS), tmp_path / "large.png")

        r_orig = proc.process(orig_path)
        r_large = proc.process(large_path)

        assert r_large.outcome is Outcome.VARIANT
        variant = store.get_variants_for_canonical(r_orig.canonical_id)[0]
        assert variant.relationship == "upscaled"

    def test_different_image_is_saved_as_new_canonical(self, pipeline, tmp_path: Path):
        _, store, proc = pipeline
        orig_path = save_png(make_gradient(256, 256), tmp_path / "orig.png")
        noise_path = save_png(make_noise(256, 256, seed=42), tmp_path / "noise.png")

        r1 = proc.process(orig_path)
        r2 = proc.process(noise_path)

        assert r2.outcome is Outcome.NEW
        assert r2.canonical_id != r1.canonical_id
        assert len(store.list_canonicals()) == 2

    def test_multiple_variants_all_linked(self, pipeline, tmp_path: Path):
        _, store, proc = pipeline
        orig = make_gradient(256, 256)
        orig_path = save_png(orig, tmp_path / "orig.png")
        r = proc.process(orig_path)

        for scale, name in [(0.5, "half"), (0.25, "quarter"), (2.0, "double")]:
            new_size = (int(256 * scale), int(256 * scale))
            path = save_png(orig.resize(new_size, Image.LANCZOS), tmp_path / f"{name}.png")
            rv = proc.process(path)
            assert rv.outcome is Outcome.VARIANT
            assert rv.canonical_id == r.canonical_id

        variants = store.get_variants_for_canonical(r.canonical_id)
        assert len(variants) == 3


# ---------------------------------------------------------------------------
# Per-call threshold override
# ---------------------------------------------------------------------------


class TestThresholdOverride:
    def test_override_forces_new_when_threshold_zero(self, pipeline, tmp_path: Path):
        """With threshold=0, only byte-identical images are variants."""
        _, _, proc = pipeline
        orig = make_gradient(256, 256)
        orig_path = save_png(orig, tmp_path / "orig.png")
        half_path = save_png(orig.resize((128, 128), Image.LANCZOS), tmp_path / "half.png")

        proc.process(orig_path)
        # half is visually the same but NOT byte-identical; with threshold 0
        # the pHash distance will be > 0, so it becomes NEW.
        result = proc.process(half_path, phash_threshold=0)
        assert result.outcome is Outcome.NEW

    def test_override_takes_precedence_over_config(self, tmp_path: Path):
        """Config threshold=0 but per-call threshold=10 still catches the variant."""
        config = Config(phash_threshold=0, db_path=str(tmp_path / "t.db"),
                        storage_dir=str(tmp_path / "ts"))
        store = ImageStore(config.db_path, config.storage_dir)
        proc = ImageProcessor(config, store)

        orig = make_gradient(256, 256)
        orig_path = save_png(orig, tmp_path / "orig.png")
        half_path = save_png(orig.resize((128, 128), Image.LANCZOS), tmp_path / "half.png")

        proc.process(orig_path)
        result = proc.process(half_path, phash_threshold=10)
        store.close()

        assert result.outcome is Outcome.VARIANT
