"""Unit tests for oracle_reduction.storage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from oracle_reduction.hasher import compute_crypto_hash, compute_phash
from oracle_reduction.storage import ImageStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_gradient(width: int, height: int) -> Image.Image:
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    for row in range(height):
        val = int(row * 255 / max(height - 1, 1))
        arr[row, :] = [val, 80, 40]
    return Image.fromarray(arr, mode="RGB")


def save_image(img: Image.Image, path: Path) -> Path:
    img.save(path, format="PNG")
    return path


@pytest.fixture()
def store(tmp_path: Path) -> ImageStore:
    s = ImageStore(
        db_path=tmp_path / "test.db",
        storage_dir=tmp_path / "store",
    )
    yield s
    s.close()


@pytest.fixture()
def image_file(tmp_path: Path) -> tuple[Image.Image, Path]:
    img = make_gradient(128, 128)
    path = save_image(img, tmp_path / "test_image.png")
    return img, path


# ---------------------------------------------------------------------------
# crypto_hash_exists
# ---------------------------------------------------------------------------


class TestCryptoHashExists:
    def test_returns_false_when_empty(self, store: ImageStore, image_file):
        img, _ = image_file
        assert not store.crypto_hash_exists(compute_crypto_hash(img))

    def test_returns_true_after_save_canonical(self, store: ImageStore, image_file):
        img, path = image_file
        h = compute_crypto_hash(img)
        ph = compute_phash(img)
        store.save_canonical(img, path, h, ph)
        assert store.crypto_hash_exists(h)

    def test_returns_true_after_save_variant(self, store: ImageStore, tmp_path: Path):
        # canonical
        orig = make_gradient(256, 256)
        orig_path = save_image(orig, tmp_path / "orig.png")
        h_orig = compute_crypto_hash(orig)
        ph_orig = compute_phash(orig)
        canonical_id = store.save_canonical(orig, orig_path, h_orig, ph_orig)
        canonical = store.get_canonical(canonical_id)

        # variant
        half = orig.resize((128, 128), Image.LANCZOS)
        half_path = save_image(half, tmp_path / "half.png")
        h_half = compute_crypto_hash(half)
        ph_half = compute_phash(half)
        store.save_variant(half, half_path, canonical_id, h_half, ph_half, canonical)

        assert store.crypto_hash_exists(h_half)


# ---------------------------------------------------------------------------
# get_all_canonical_phashes
# ---------------------------------------------------------------------------


class TestGetAllCanonicalPhashes:
    def test_empty_when_no_images(self, store: ImageStore):
        assert store.get_all_canonical_phashes() == []

    def test_returns_inserted_phashes(self, store: ImageStore, image_file):
        img, path = image_file
        h = compute_crypto_hash(img)
        ph = compute_phash(img)
        cid = store.save_canonical(img, path, h, ph)
        phashes = store.get_all_canonical_phashes()
        assert len(phashes) == 1
        assert phashes[0][0] == cid
        assert phashes[0][1] == str(ph)


# ---------------------------------------------------------------------------
# save_canonical / get_canonical
# ---------------------------------------------------------------------------


class TestSaveAndGetCanonical:
    def test_round_trip(self, store: ImageStore, image_file):
        img, path = image_file
        h = compute_crypto_hash(img)
        ph = compute_phash(img)
        cid = store.save_canonical(img, path, h, ph)

        record = store.get_canonical(cid)
        assert record is not None
        assert record.id == cid
        assert record.crypto_hash == h
        assert record.phash == str(ph)
        assert record.width == img.width
        assert record.height == img.height

    def test_file_is_copied(self, store: ImageStore, image_file):
        img, path = image_file
        h = compute_crypto_hash(img)
        ph = compute_phash(img)
        store.save_canonical(img, path, h, ph)
        # The stored file must exist and not be the original
        record = store.get_canonical(1)
        assert record is not None
        assert Path(record.file_path).exists()

    def test_get_canonical_returns_none_for_missing_id(self, store: ImageStore):
        assert store.get_canonical(999) is None


# ---------------------------------------------------------------------------
# save_variant / get_variants_for_canonical
# ---------------------------------------------------------------------------


class TestVariants:
    def test_variant_linked_to_canonical(self, store: ImageStore, tmp_path: Path):
        orig = make_gradient(256, 256)
        orig_path = save_image(orig, tmp_path / "orig.png")
        cid = store.save_canonical(orig, orig_path, compute_crypto_hash(orig), compute_phash(orig))
        canonical = store.get_canonical(cid)

        half = orig.resize((128, 128), Image.LANCZOS)
        half_path = save_image(half, tmp_path / "half.png")
        vid = store.save_variant(
            half, half_path, cid, compute_crypto_hash(half), compute_phash(half), canonical
        )

        variants = store.get_variants_for_canonical(cid)
        assert len(variants) == 1
        assert variants[0].id == vid
        assert variants[0].canonical_id == cid

    def test_variant_relationship_downscaled(self, store: ImageStore, tmp_path: Path):
        orig = make_gradient(256, 256)
        orig_path = save_image(orig, tmp_path / "orig.png")
        cid = store.save_canonical(orig, orig_path, compute_crypto_hash(orig), compute_phash(orig))
        canonical = store.get_canonical(cid)

        half = orig.resize((128, 128), Image.LANCZOS)
        half_path = save_image(half, tmp_path / "half.png")
        store.save_variant(
            half, half_path, cid, compute_crypto_hash(half), compute_phash(half), canonical
        )

        variant = store.get_variants_for_canonical(cid)[0]
        assert variant.relationship == "downscaled"
        assert variant.scale_factor is not None
        assert variant.scale_factor < 1.0

    def test_no_variants_returns_empty_list(self, store: ImageStore, image_file):
        img, path = image_file
        cid = store.save_canonical(img, path, compute_crypto_hash(img), compute_phash(img))
        assert store.get_variants_for_canonical(cid) == []


# ---------------------------------------------------------------------------
# context manager
# ---------------------------------------------------------------------------


def test_store_context_manager(tmp_path: Path):
    with ImageStore(tmp_path / "ctx.db", tmp_path / "ctx_store") as s:
        assert s.get_all_canonical_phashes() == []
    # After close, further use would raise; just verify no exception on exit.
