"""Tests for swapping which copy of a picture is the canonical one.

The canonical is whichever copy arrived first, which says nothing about which
is best.  ``make_canonical`` trades places without removing anything, so these
tests are mostly about what must *survive*: both files, every sibling, and a
set of relationships that still describes the group.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from oracle_reduction.hasher import compute_crypto_hash, compute_phash
from oracle_reduction.storage import ImageStore


def make_pattern(width: int, height: int, seed: int = 1) -> Image.Image:
    rng = np.random.default_rng(seed)
    blocks = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
    return Image.fromarray(blocks, mode="RGB").resize((width, height), Image.NEAREST)


def save(img: Image.Image, path: Path) -> Path:
    img.save(path, format="PNG")
    return path


@pytest.fixture(params=["copy", "symlink"])
def store(request, tmp_path: Path) -> ImageStore:
    s = ImageStore(
        db_path=tmp_path / "test.db",
        storage_dir=tmp_path / "store",
        store_mode=request.param,
    )
    yield s
    s.close()


@pytest.fixture()
def group(store: ImageStore, tmp_path: Path):
    """A small canonical with a big copy and a middling one.

    The 200x200 canonical was merely seen first; the 800x800 copy is the one
    worth keeping and the 400x400 is a sibling that has to come along.
    """
    sources = tmp_path / "sources"
    sources.mkdir()
    base = make_pattern(800, 800)

    small = base.resize((200, 200))
    canonical_id = store.save_canonical(
        small, save(small, sources / "small.png"),
        compute_crypto_hash(small), compute_phash(small),
    )
    canonical = store.get_canonical(canonical_id)

    big_id = store.save_variant(
        base, save(base, sources / "big.png"), canonical_id,
        compute_crypto_hash(base), compute_phash(base), canonical,
    )
    mid = base.resize((400, 400))
    mid_id = store.save_variant(
        mid, save(mid, sources / "mid.png"), canonical_id,
        compute_crypto_hash(mid), compute_phash(mid), canonical,
    )
    return canonical_id, big_id, mid_id


def test_it_finds_the_group_needing_a_swap(store: ImageStore, group):
    canonical_id, big_id, _ = group
    assert store.variants_beating_their_canonical() == [(canonical_id, big_id)], (
        "one row per group, naming the biggest copy rather than every bigger one"
    )


def test_nothing_to_do_once_the_biggest_is_canonical(store: ImageStore, group):
    _, big_id, _ = group
    store.make_canonical(big_id)
    assert store.variants_beating_their_canonical() == []


def test_a_group_already_in_order_is_not_offered(store: ImageStore, tmp_path: Path):
    sources = tmp_path / "s"
    sources.mkdir()
    base = make_pattern(600, 600, seed=11)
    canonical_id = store.save_canonical(
        base, save(base, sources / "big.png"),
        compute_crypto_hash(base), compute_phash(base),
    )
    small = base.resize((100, 100))
    store.save_variant(
        small, save(small, sources / "small.png"), canonical_id,
        compute_crypto_hash(small), compute_phash(small),
        store.get_canonical(canonical_id),
    )
    assert store.variants_beating_their_canonical() == []


def test_the_big_copy_becomes_the_canonical(store: ImageStore, group):
    canonical_id, big_id, _ = group
    new_canonical_id, _ = store.make_canonical(big_id)

    promoted = store.get_canonical(new_canonical_id)
    assert promoted is not None
    assert (promoted.width, promoted.height) == (800, 800)
    assert store.get_canonical(canonical_id) is None, "the old canonical is no longer one"
    assert store.get_variant(big_id) is None, "and the big copy is no longer a variant"


def test_the_old_canonical_becomes_a_variant(store: ImageStore, group):
    _, big_id, _ = group
    new_canonical_id, new_variant_id = store.make_canonical(big_id)

    demoted = store.get_variant(new_variant_id)
    assert demoted is not None
    assert (demoted.width, demoted.height) == (200, 200)
    assert demoted.canonical_id == new_canonical_id
    assert demoted.relationship == "downscaled", "200px is smaller than 800px"
    assert demoted.scale_factor == pytest.approx(0.25)


def test_the_group_keeps_everyone(store: ImageStore, group):
    _, big_id, mid_id = group
    new_canonical_id, _ = store.make_canonical(big_id)

    variants = store.get_variants_for_canonical(new_canonical_id)
    assert len(variants) == 2, "the deposed canonical and the untouched sibling"
    assert mid_id in {v.id for v in variants}


def test_a_siblings_relationship_is_restated(store: ImageStore, group):
    """400px was *bigger* than the 200px canonical and is smaller than 800px.

    Left alone the row would still read "upscaled", which is the sort of stale
    claim that makes the gallery argue with itself.
    """
    canonical_id, big_id, mid_id = group
    before = next(
        v for v in store.get_variants_for_canonical(canonical_id) if v.id == mid_id
    )
    assert before.relationship == "upscaled"

    new_canonical_id, _ = store.make_canonical(big_id)

    after = next(
        v for v in store.get_variants_for_canonical(new_canonical_id) if v.id == mid_id
    )
    assert after.relationship == "downscaled"
    assert after.scale_factor == pytest.approx(0.5)


def test_siblings_survive_cascading_deletes(tmp_path: Path):
    """The old canonical's row is deleted last, and that ordering matters.

    ``image_variants.canonical_id`` is declared ``ON DELETE CASCADE``.  With
    foreign keys enforced, removing the deposed canonical before its variants
    have been re-pointed takes the whole group with it.  Enforcement is off by
    default, so without this the ordering would be load-bearing and untested.
    """
    store = ImageStore(db_path=tmp_path / "fk.db", storage_dir=tmp_path / "store")
    store._conn.execute("PRAGMA foreign_keys=ON")

    sources = tmp_path / "sources"
    sources.mkdir()
    base = make_pattern(800, 800)
    small = base.resize((200, 200))
    canonical_id = store.save_canonical(
        small, save(small, sources / "s.png"),
        compute_crypto_hash(small), compute_phash(small),
    )
    canonical = store.get_canonical(canonical_id)
    big_id = store.save_variant(
        base, save(base, sources / "b.png"), canonical_id,
        compute_crypto_hash(base), compute_phash(base), canonical,
    )
    mid = base.resize((400, 400))
    store.save_variant(
        mid, save(mid, sources / "m.png"), canonical_id,
        compute_crypto_hash(mid), compute_phash(mid), canonical,
    )

    new_canonical_id, _ = store.make_canonical(big_id)
    assert len(store.get_variants_for_canonical(new_canonical_id)) == 2
    store.close()


def test_both_files_stay_in_the_store(store: ImageStore, group):
    _, big_id, _ = group
    new_canonical_id, new_variant_id = store.make_canonical(big_id)

    promoted = store.get_canonical(new_canonical_id)
    demoted = store.get_variant(new_variant_id)
    assert promoted.file_path.exists() or promoted.file_path.is_symlink()
    assert demoted.file_path.exists() or demoted.file_path.is_symlink()
    assert promoted.file_path.parent.name == "canonicals"
    assert demoted.file_path.parent.name == "variants"


def test_the_users_originals_are_untouched(store: ImageStore, group, tmp_path: Path):
    """Nothing here is a delete.  In symlink mode the entries are pointers at
    the user's files, so a careless move would reach through them."""
    _, big_id, _ = group
    store.make_canonical(big_id)

    for name in ("small.png", "big.png", "mid.png"):
        assert (tmp_path / "sources" / name).exists()


def test_the_new_canonical_joins_the_index(store: ImageStore, group):
    _, big_id, _ = group
    new_canonical_id, _ = store.make_canonical(big_id)

    ids = {i for i, _ in store.get_all_canonical_phashes()}
    assert new_canonical_id in ids
    index = store.canonical_phash_index()
    assert index.nearest(compute_phash(make_pattern(800, 800)))[0] == new_canonical_id


def test_an_unknown_variant_changes_nothing(store: ImageStore, group):
    canonical_id, _, _ = group
    assert store.make_canonical(9999) is None
    assert store.get_canonical(canonical_id) is not None
    assert len(store.get_variants_for_canonical(canonical_id)) == 2


def test_swapping_a_lone_copy_works(store: ImageStore, tmp_path: Path):
    """The common case in practice: one canonical, one better copy, no siblings."""
    sources = tmp_path / "sources"
    sources.mkdir()
    base = make_pattern(600, 400, seed=5)
    small = base.resize((150, 100))
    canonical_id = store.save_canonical(
        small, save(small, sources / "s.png"),
        compute_crypto_hash(small), compute_phash(small),
    )
    big_id = store.save_variant(
        base, save(base, sources / "b.png"), canonical_id,
        compute_crypto_hash(base), compute_phash(base),
        store.get_canonical(canonical_id),
    )

    new_canonical_id, new_variant_id = store.make_canonical(big_id)
    assert store.get_canonical(new_canonical_id).width == 600
    assert store.get_variant(new_variant_id).width == 150
    assert len(store.get_variants_for_canonical(new_canonical_id)) == 1
