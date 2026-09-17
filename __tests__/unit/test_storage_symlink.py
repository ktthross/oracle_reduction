"""Unit tests for ImageStore's symlink store mode."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from oracle_reduction.hasher import compute_crypto_hash, compute_phash
from oracle_reduction.storage import ImageStore


def make_gradient(width: int, height: int, tint: int = 0) -> Image.Image:
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    for row in range(height):
        val = int(row * 255 / max(height - 1, 1))
        arr[row, :] = [val, (80 + tint) % 256, 40]
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture()
def linking_store(tmp_path: Path) -> ImageStore:
    s = ImageStore(
        db_path=tmp_path / "test.db",
        storage_dir=tmp_path / "store",
        store_mode="symlink",
    )
    yield s
    s.close()


def add_canonical(store: ImageStore, source: Path, tint: int = 0) -> int:
    img = make_gradient(64, 64, tint)
    img.save(source, format="PNG")
    return store.save_canonical(
        image=img,
        source_path=source,
        crypto_hash=compute_crypto_hash(img),
        phash=compute_phash(img),
    )


class TestStoreMode:
    def test_rejects_an_unknown_mode(self, tmp_path: Path):
        with pytest.raises(ValueError, match="store_mode"):
            ImageStore(tmp_path / "db", tmp_path / "store", store_mode="hardlink")

    def test_defaults_to_copy(self, tmp_path: Path):
        store = ImageStore(tmp_path / "db", tmp_path / "store")
        assert store.store_mode == "copy"
        store.close()


class TestSymlinkPlacement:
    def test_entry_is_a_symlink_to_the_source(self, linking_store: ImageStore, tmp_path: Path):
        source = tmp_path / "a.png"
        canonical_id = add_canonical(linking_store, source)

        stored = linking_store.get_canonical(canonical_id).file_path
        assert stored.is_symlink()
        assert Path(os.readlink(stored)) == source.resolve()

    def test_entry_costs_almost_no_disk(self, linking_store: ImageStore, tmp_path: Path):
        source = tmp_path / "a.png"
        canonical_id = add_canonical(linking_store, source)

        stored = linking_store.get_canonical(canonical_id).file_path
        # lstat is the link itself; stat would follow through to the source.
        assert stored.lstat().st_size < source.stat().st_size

    def test_image_is_still_readable_through_the_entry(
        self, linking_store: ImageStore, tmp_path: Path
    ):
        source = tmp_path / "a.png"
        canonical_id = add_canonical(linking_store, source)

        stored = linking_store.get_canonical(canonical_id).file_path
        with Image.open(stored) as img:
            assert img.size == (64, 64)

    def test_copy_mode_still_copies(self, tmp_path: Path):
        store = ImageStore(tmp_path / "db", tmp_path / "store", store_mode="copy")
        source = tmp_path / "a.png"
        canonical_id = add_canonical(store, source)

        stored = store.get_canonical(canonical_id).file_path
        assert not stored.is_symlink()
        assert stored.stat().st_size == source.stat().st_size
        store.close()


class TestAuditWithSymlinks:
    def test_intact_links_audit_clean(self, linking_store: ImageStore, tmp_path: Path):
        add_canonical(linking_store, tmp_path / "a.png")

        result = linking_store.audit_storage()
        assert result["orphan_files"] == []
        assert result["missing_files"] == []

    def test_a_broken_link_is_reported_missing(self, linking_store: ImageStore, tmp_path: Path):
        source = tmp_path / "a.png"
        canonical_id = add_canonical(linking_store, source)
        source.unlink()  # the original goes away; the link dangles

        missing = linking_store.audit_storage()["missing_files"]
        assert [(kind, row_id) for kind, row_id, _ in missing] == [("canonical", canonical_id)]

    def test_a_broken_link_is_not_mistaken_for_an_orphan(
        self, linking_store: ImageStore, tmp_path: Path
    ):
        """It still has a row, so it is dangling, not unreferenced."""
        source = tmp_path / "a.png"
        add_canonical(linking_store, source)
        source.unlink()

        assert linking_store.audit_storage()["orphan_files"] == []

    def test_an_unreferenced_dead_link_is_swept_as_an_orphan(
        self, linking_store: ImageStore, tmp_path: Path
    ):
        """is_file() is False for a dead link, so is_symlink() has to catch it."""
        dangling = linking_store.storage_dir / "canonicals" / "stray.png"
        os.symlink(tmp_path / "never_existed.png", dangling)

        assert linking_store.audit_storage()["orphan_files"] == [dangling]
        assert linking_store.cleanup_orphans() == 1
        assert not dangling.is_symlink()


class TestCleanupNeverTouchesSources:
    def test_cleanup_orphans_leaves_the_original_alone(
        self, linking_store: ImageStore, tmp_path: Path
    ):
        """The critical safety property of symlink mode."""
        source = tmp_path / "a.png"
        add_canonical(linking_store, source)

        # Strand the entry by dropping its row, then sweep.
        linking_store._conn.execute("DELETE FROM images")
        linking_store._conn.commit()

        assert linking_store.cleanup_orphans() == 1
        assert source.exists(), "cleanup must never delete the user's original file"

    def test_removing_dangling_records_leaves_the_original_alone(
        self, linking_store: ImageStore, tmp_path: Path
    ):
        source = tmp_path / "a.png"
        add_canonical(linking_store, source)
        stored = linking_store.list_canonicals()[0].file_path
        stored.unlink()  # lose the link, keep the source

        linking_store.remove_dangling_records()
        assert source.exists()
