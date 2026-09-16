"""Unit tests for ImageStore's audit/cleanup integrity helpers."""

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


def make_gradient(width: int, height: int, tint: int = 0) -> Image.Image:
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    for row in range(height):
        val = int(row * 255 / max(height - 1, 1))
        arr[row, :] = [val, (80 + tint) % 256, 40]
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture()
def store(tmp_path: Path) -> ImageStore:
    s = ImageStore(db_path=tmp_path / "test.db", storage_dir=tmp_path / "store")
    yield s
    s.close()


def add_canonical(store: ImageStore, tmp_path: Path, name: str, tint: int = 0) -> int:
    """Save a canonical image into *store* and return its row id."""
    img = make_gradient(64, 64, tint)
    source = tmp_path / name
    img.save(source, format="PNG")
    return store.save_canonical(
        image=img,
        source_path=source,
        crypto_hash=compute_crypto_hash(img),
        phash=compute_phash(img),
    )


def add_variant(store: ImageStore, tmp_path: Path, canonical_id: int, name: str) -> int:
    """Save a half-size variant of *canonical_id* and return its row id."""
    canonical = store.get_canonical(canonical_id)
    img = make_gradient(32, 32)
    source = tmp_path / name
    img.save(source, format="PNG")
    return store.save_variant(
        image=img,
        source_path=source,
        canonical_id=canonical_id,
        crypto_hash=compute_crypto_hash(img),
        phash=compute_phash(img),
        canonical=canonical,
    )


# ---------------------------------------------------------------------------
# audit_storage
# ---------------------------------------------------------------------------


class TestAuditStorage:
    def test_clean_store_reports_nothing(self, store: ImageStore, tmp_path: Path):
        add_canonical(store, tmp_path, "a.png")
        result = store.audit_storage()
        assert result["orphan_files"] == []
        assert result["missing_files"] == []

    def test_empty_store_is_clean(self, store: ImageStore):
        result = store.audit_storage()
        assert result["orphan_files"] == []
        assert result["missing_files"] == []

    def test_detects_orphan_file(self, store: ImageStore, tmp_path: Path):
        add_canonical(store, tmp_path, "a.png")
        orphan = store.storage_dir / "canonicals" / "stray.png"
        orphan.write_bytes(b"not referenced by any row")

        orphans = store.audit_storage()["orphan_files"]
        assert orphans == [orphan]

    def test_detects_orphan_in_variants_dir(self, store: ImageStore):
        orphan = store.storage_dir / "variants" / "stray.png"
        orphan.write_bytes(b"not referenced by any row")

        assert store.audit_storage()["orphan_files"] == [orphan]

    def test_detects_missing_canonical_file(self, store: ImageStore, tmp_path: Path):
        canonical_id = add_canonical(store, tmp_path, "a.png")
        stored = store.get_canonical(canonical_id).file_path
        stored.unlink()

        missing = store.audit_storage()["missing_files"]
        assert missing == [("canonical", canonical_id, stored)]

    def test_detects_missing_variant_file(self, store: ImageStore, tmp_path: Path):
        canonical_id = add_canonical(store, tmp_path, "a.png")
        variant_id = add_variant(store, tmp_path, canonical_id, "a_small.png")
        stored = store.get_variants_for_canonical(canonical_id)[0].file_path
        stored.unlink()

        missing = store.audit_storage()["missing_files"]
        assert missing == [("variant", variant_id, stored)]

    def test_a_missing_file_is_not_also_an_orphan(self, store: ImageStore, tmp_path: Path):
        canonical_id = add_canonical(store, tmp_path, "a.png")
        store.get_canonical(canonical_id).file_path.unlink()

        result = store.audit_storage()
        assert result["orphan_files"] == []
        assert len(result["missing_files"]) == 1


# ---------------------------------------------------------------------------
# cleanup_orphans
# ---------------------------------------------------------------------------


class TestCleanupOrphans:
    def test_removes_orphan_and_reports_count(self, store: ImageStore, tmp_path: Path):
        add_canonical(store, tmp_path, "a.png")
        orphan = store.storage_dir / "canonicals" / "stray.png"
        orphan.write_bytes(b"stray")

        assert store.cleanup_orphans() == 1
        assert not orphan.exists()

    def test_leaves_referenced_files_alone(self, store: ImageStore, tmp_path: Path):
        canonical_id = add_canonical(store, tmp_path, "a.png")
        kept = store.get_canonical(canonical_id).file_path
        (store.storage_dir / "canonicals" / "stray.png").write_bytes(b"stray")

        store.cleanup_orphans()
        assert kept.exists()

    def test_noop_on_clean_store(self, store: ImageStore, tmp_path: Path):
        add_canonical(store, tmp_path, "a.png")
        assert store.cleanup_orphans() == 0


# ---------------------------------------------------------------------------
# remove_dangling_records
# ---------------------------------------------------------------------------


class TestRemoveDanglingRecords:
    def test_removes_record_for_missing_canonical(self, store: ImageStore, tmp_path: Path):
        canonical_id = add_canonical(store, tmp_path, "a.png")
        store.get_canonical(canonical_id).file_path.unlink()

        assert store.remove_dangling_records() == 1
        assert store.get_canonical(canonical_id) is None

    def test_removes_record_for_missing_variant(self, store: ImageStore, tmp_path: Path):
        canonical_id = add_canonical(store, tmp_path, "a.png")
        add_variant(store, tmp_path, canonical_id, "a_small.png")
        store.get_variants_for_canonical(canonical_id)[0].file_path.unlink()

        assert store.remove_dangling_records() == 1
        assert store.get_variants_for_canonical(canonical_id) == []

    def test_keeps_healthy_records(self, store: ImageStore, tmp_path: Path):
        healthy = add_canonical(store, tmp_path, "a.png")
        broken = add_canonical(store, tmp_path, "b.png", tint=120)
        store.get_canonical(broken).file_path.unlink()

        store.remove_dangling_records()
        assert store.get_canonical(healthy) is not None
        assert store.get_canonical(broken) is None

    def test_missing_canonical_cascades_to_its_variants(self, store: ImageStore, tmp_path: Path):
        """A variant must never outlive the canonical it points at."""
        canonical_id = add_canonical(store, tmp_path, "a.png")
        add_variant(store, tmp_path, canonical_id, "a_small.png")
        store.get_canonical(canonical_id).file_path.unlink()

        # 1 canonical + 1 cascaded variant
        assert store.remove_dangling_records() == 2
        assert store.get_variants_for_canonical(canonical_id) == []

    def test_noop_on_clean_store(self, store: ImageStore, tmp_path: Path):
        add_canonical(store, tmp_path, "a.png")
        assert store.remove_dangling_records() == 0

    def test_audit_is_clean_after_full_cleanup(self, store: ImageStore, tmp_path: Path):
        canonical_id = add_canonical(store, tmp_path, "a.png")
        add_canonical(store, tmp_path, "b.png", tint=120)
        store.get_canonical(canonical_id).file_path.unlink()
        (store.storage_dir / "canonicals" / "stray.png").write_bytes(b"stray")

        store.cleanup_orphans()
        store.remove_dangling_records()
        store.cleanup_orphans()

        result = store.audit_storage()
        assert result["orphan_files"] == []
        assert result["missing_files"] == []
