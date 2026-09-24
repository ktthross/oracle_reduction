"""Unit tests for deleting and reclassifying entries in an ImageStore.

The rule these guard is that deletion is confined to ``storage_dir``: in
symlink mode the entry is a link, and unlinking it must never reach through to
the original file the user still has on disk.
"""

from __future__ import annotations

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


@pytest.fixture(params=["copy", "symlink"])
def store(request, tmp_path: Path) -> ImageStore:
    """An ImageStore in each placement mode, so every test covers both."""
    s = ImageStore(
        db_path=tmp_path / "test.db",
        storage_dir=tmp_path / "store",
        store_mode=request.param,
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


def add_variant(store: ImageStore, source: Path, canonical_id: int, tint: int = 7) -> int:
    img = make_gradient(32, 32, tint)
    img.save(source, format="PNG")
    return store.save_variant(
        image=img,
        source_path=source,
        canonical_id=canonical_id,
        crypto_hash=compute_crypto_hash(img),
        phash=compute_phash(img),
        canonical=store.get_canonical(canonical_id),
    )


class TestDeleteVariant:
    def test_returns_none_for_an_unknown_id(self, store: ImageStore):
        assert store.delete_variant(9999) is None

    def test_returns_the_record_it_deleted(self, store: ImageStore, tmp_path: Path):
        canonical_id = add_canonical(store, tmp_path / "a.png")
        variant_id = add_variant(store, tmp_path / "b.png", canonical_id)

        record = store.delete_variant(variant_id)

        assert record is not None
        assert record.id == variant_id
        assert record.canonical_id == canonical_id

    def test_drops_the_row(self, store: ImageStore, tmp_path: Path):
        canonical_id = add_canonical(store, tmp_path / "a.png")
        variant_id = add_variant(store, tmp_path / "b.png", canonical_id)

        store.delete_variant(variant_id)

        assert store.get_variants_for_canonical(canonical_id) == []

    def test_unlinks_the_managed_entry(self, store: ImageStore, tmp_path: Path):
        canonical_id = add_canonical(store, tmp_path / "a.png")
        variant_id = add_variant(store, tmp_path / "b.png", canonical_id)
        entry = store.get_variants_for_canonical(canonical_id)[0].file_path

        store.delete_variant(variant_id)

        assert not entry.exists() and not entry.is_symlink()

    def test_leaves_the_original_file_alone(self, store: ImageStore, tmp_path: Path):
        """The guarantee that makes a one-click delete safe in symlink mode."""
        canonical_id = add_canonical(store, tmp_path / "a.png")
        source = tmp_path / "b.png"
        variant_id = add_variant(store, source, canonical_id)

        store.delete_variant(variant_id)

        assert source.exists()

    def test_leaves_the_canonical_alone(self, store: ImageStore, tmp_path: Path):
        canonical_id = add_canonical(store, tmp_path / "a.png")
        variant_id = add_variant(store, tmp_path / "b.png", canonical_id)

        store.delete_variant(variant_id)

        canonical = store.get_canonical(canonical_id)
        assert canonical is not None
        assert canonical.file_path.exists()

    def test_frees_the_crypto_hash_for_reuse(self, store: ImageStore, tmp_path: Path):
        """A deleted image is no longer a duplicate, so it can come back."""
        canonical_id = add_canonical(store, tmp_path / "a.png")
        variant_id = add_variant(store, tmp_path / "b.png", canonical_id)
        crypto_hash = store.get_variants_for_canonical(canonical_id)[0].crypto_hash

        store.delete_variant(variant_id)

        assert store.crypto_hash_exists(crypto_hash) is False


class TestDeleteCanonical:
    def test_returns_none_for_an_unknown_id(self, store: ImageStore):
        assert store.delete_canonical(9999) is None

    def test_returns_the_canonical_and_its_variants(self, store: ImageStore, tmp_path: Path):
        canonical_id = add_canonical(store, tmp_path / "a.png")
        add_variant(store, tmp_path / "b.png", canonical_id, tint=7)
        add_variant(store, tmp_path / "c.png", canonical_id, tint=9)

        result = store.delete_canonical(canonical_id)

        assert result is not None
        canonical, variants = result
        assert canonical.id == canonical_id
        assert len(variants) == 2

    def test_drops_the_canonical_row(self, store: ImageStore, tmp_path: Path):
        canonical_id = add_canonical(store, tmp_path / "a.png")

        store.delete_canonical(canonical_id)

        assert store.get_canonical(canonical_id) is None

    def test_drops_the_variant_rows_too(self, store: ImageStore, tmp_path: Path):
        canonical_id = add_canonical(store, tmp_path / "a.png")
        add_variant(store, tmp_path / "b.png", canonical_id)

        store.delete_canonical(canonical_id)

        assert store.get_variants_for_canonical(canonical_id) == []

    def test_unlinks_every_managed_entry(self, store: ImageStore, tmp_path: Path):
        canonical_id = add_canonical(store, tmp_path / "a.png")
        add_variant(store, tmp_path / "b.png", canonical_id)
        entries = [store.get_canonical(canonical_id).file_path] + [
            v.file_path for v in store.get_variants_for_canonical(canonical_id)
        ]

        store.delete_canonical(canonical_id)

        assert not any(e.exists() or e.is_symlink() for e in entries)

    def test_leaves_every_original_file_alone(self, store: ImageStore, tmp_path: Path):
        canonical_source = tmp_path / "a.png"
        variant_source = tmp_path / "b.png"
        canonical_id = add_canonical(store, canonical_source)
        add_variant(store, variant_source, canonical_id)

        store.delete_canonical(canonical_id)

        assert canonical_source.exists()
        assert variant_source.exists()

    def test_leaves_an_unrelated_canonical_alone(self, store: ImageStore, tmp_path: Path):
        doomed = add_canonical(store, tmp_path / "a.png", tint=0)
        keeper = add_canonical(store, tmp_path / "b.png", tint=120)

        store.delete_canonical(doomed)

        survivor = store.get_canonical(keeper)
        assert survivor is not None
        assert survivor.file_path.exists()

    def test_drops_the_deleted_image_from_the_phash_index(
        self, store: ImageStore, tmp_path: Path
    ):
        """Without invalidation the index keeps matching images to a dead row."""
        canonical_id = add_canonical(store, tmp_path / "a.png")
        store.canonical_phash_index()  # build the cache before deleting

        store.delete_canonical(canonical_id)

        matched_id, _ = store.canonical_phash_index().nearest(
            compute_phash(make_gradient(64, 64))
        )
        assert matched_id != canonical_id

    def test_survives_a_row_whose_file_is_already_gone(
        self, store: ImageStore, tmp_path: Path
    ):
        canonical_id = add_canonical(store, tmp_path / "a.png")
        store.get_canonical(canonical_id).file_path.unlink()

        assert store.delete_canonical(canonical_id) is not None
        assert store.get_canonical(canonical_id) is None


class TestPromoteVariant:
    def test_returns_none_for_an_unknown_id(self, store: ImageStore):
        assert store.promote_variant(9999) is None

    def test_creates_a_canonical(self, store: ImageStore, tmp_path: Path):
        canonical_id = add_canonical(store, tmp_path / "a.png")
        variant_id = add_variant(store, tmp_path / "b.png", canonical_id)

        promoted_id = store.promote_variant(variant_id)

        assert promoted_id is not None
        assert store.get_canonical(promoted_id) is not None

    def test_removes_the_variant_row(self, store: ImageStore, tmp_path: Path):
        canonical_id = add_canonical(store, tmp_path / "a.png")
        variant_id = add_variant(store, tmp_path / "b.png", canonical_id)

        store.promote_variant(variant_id)

        assert store.get_variants_for_canonical(canonical_id) == []

    def test_carries_the_metadata_across_unchanged(self, store: ImageStore, tmp_path: Path):
        canonical_id = add_canonical(store, tmp_path / "a.png")
        variant_id = add_variant(store, tmp_path / "b.png", canonical_id)
        before = store.get_variants_for_canonical(canonical_id)[0]

        after = store.get_canonical(store.promote_variant(variant_id))

        assert after.filename == before.filename
        assert after.crypto_hash == before.crypto_hash
        assert after.phash == before.phash
        assert (after.width, after.height) == (before.width, before.height)
        assert after.format == before.format

    def test_moves_the_entry_into_the_canonicals_directory(
        self, store: ImageStore, tmp_path: Path
    ):
        canonical_id = add_canonical(store, tmp_path / "a.png")
        variant_id = add_variant(store, tmp_path / "b.png", canonical_id)
        old_entry = store.get_variants_for_canonical(canonical_id)[0].file_path

        promoted = store.get_canonical(store.promote_variant(variant_id))

        assert promoted.file_path.parent.name == "canonicals"
        assert promoted.file_path.exists()
        assert not old_entry.exists() and not old_entry.is_symlink()

    def test_the_promoted_entry_still_resolves_to_the_original(
        self, store: ImageStore, tmp_path: Path
    ):
        """Moving a symlink must not break what it points at."""
        canonical_id = add_canonical(store, tmp_path / "a.png")
        source = tmp_path / "b.png"
        variant_id = add_variant(store, source, canonical_id)

        promoted = store.get_canonical(store.promote_variant(variant_id))

        assert promoted.file_path.read_bytes() == source.read_bytes()

    def test_does_not_collide_with_the_hash_suffix_already_on_the_entry(
        self, store: ImageStore, tmp_path: Path
    ):
        """The store entry's stem carries a hash; naming from it would double it."""
        canonical_id = add_canonical(store, tmp_path / "a.png")
        variant_id = add_variant(store, tmp_path / "b.png", canonical_id)

        promoted = store.get_canonical(store.promote_variant(variant_id))

        assert promoted.file_path.stem.count("_") == 1

    def test_the_promoted_image_becomes_matchable(self, store: ImageStore, tmp_path: Path):
        """The point of promoting: later copies match it instead of the old canonical."""
        canonical_id = add_canonical(store, tmp_path / "a.png", tint=0)
        variant_img = make_gradient(32, 32, tint=7)
        variant_id = add_variant(store, tmp_path / "b.png", canonical_id, tint=7)
        store.canonical_phash_index()  # build the cache before promoting

        promoted_id = store.promote_variant(variant_id)

        matched_id, distance = store.canonical_phash_index().nearest(
            compute_phash(variant_img)
        )
        assert matched_id == promoted_id
        assert distance == 0

    def test_leaves_the_original_file_alone(self, store: ImageStore, tmp_path: Path):
        canonical_id = add_canonical(store, tmp_path / "a.png")
        source = tmp_path / "b.png"
        variant_id = add_variant(store, source, canonical_id)

        store.promote_variant(variant_id)

        assert source.exists()

    def test_survives_a_row_whose_file_is_already_gone(
        self, store: ImageStore, tmp_path: Path
    ):
        canonical_id = add_canonical(store, tmp_path / "a.png")
        variant_id = add_variant(store, tmp_path / "b.png", canonical_id)
        store.get_variants_for_canonical(canonical_id)[0].file_path.unlink()

        assert store.promote_variant(variant_id) is not None
