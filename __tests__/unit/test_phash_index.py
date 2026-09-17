"""Unit tests for PHashIndex — the fast nearest-neighbour path.

The index replaces a linear scan over every stored hash, so the property that
matters most is that it returns *exactly* what the scan returned. Several tests
here assert equivalence rather than specific values.
"""

from __future__ import annotations

import random

import imagehash
import numpy as np
import pytest

from oracle_reduction.comparator import (
    PHashIndex,
    compare,
    find_nearest_canonical,
    pack_phash,
)
from oracle_reduction.models import Outcome


def random_hex() -> str:
    return "".join(random.choice("0123456789abcdef") for _ in range(16))


def flip_bits(phash: imagehash.ImageHash, count: int) -> imagehash.ImageHash:
    """Return *phash* with *count* specific bits flipped."""
    bits = phash.hash.copy()
    for i in range(count):
        bits[i // 8][i % 8] = not bits[i // 8][i % 8]
    return imagehash.ImageHash(bits)


class TestPacking:
    def test_packs_to_a_single_uint64(self):
        packed = pack_phash(imagehash.hex_to_hash(random_hex()))
        assert packed.dtype == np.uint64

    def test_identical_hashes_pack_identically(self):
        h = random_hex()
        assert pack_phash(imagehash.hex_to_hash(h)) == pack_phash(imagehash.hex_to_hash(h))


class TestNearest:
    def test_empty_index_finds_nothing(self):
        assert PHashIndex().nearest(imagehash.hex_to_hash(random_hex())) == (None, None)

    def test_exact_match_has_distance_zero(self):
        target = imagehash.hex_to_hash(random_hex())
        index = PHashIndex()
        index.add(42, target)
        assert index.nearest(target) == (42, 0)

    @pytest.mark.parametrize("flips", [1, 3, 7, 20])
    def test_distance_counts_flipped_bits(self, flips: int):
        base = imagehash.hex_to_hash(random_hex())
        index = PHashIndex()
        index.add(1, base)
        assert index.nearest(flip_bits(base, flips)) == (1, flips)

    def test_picks_the_closest_of_several(self):
        base = imagehash.hex_to_hash(random_hex())
        index = PHashIndex()
        index.add(1, flip_bits(base, 12))
        index.add(2, flip_bits(base, 2))
        index.add(3, flip_bits(base, 25))
        assert index.nearest(base) == (2, 2)

    def test_ties_go_to_the_earliest_entry(self):
        """Matches find_nearest_canonical, so results stay deterministic."""
        base = imagehash.hex_to_hash(random_hex())
        index = PHashIndex()
        index.add(10, flip_bits(base, 4))
        index.add(20, flip_bits(base, 4))
        assert index.nearest(base)[0] == 10


class TestGrowth:
    def test_grows_past_its_initial_capacity(self):
        index = PHashIndex(capacity=2)
        hashes = [imagehash.hex_to_hash(random_hex()) for _ in range(50)]
        for i, h in enumerate(hashes):
            index.add(i, h)

        assert len(index) == 50
        # every entry must still be findable after the buffer doubled
        for i, h in enumerate(hashes):
            assert index.nearest(h) == (i, 0)

    def test_len_tracks_additions(self):
        index = PHashIndex()
        assert len(index) == 0
        index.add(1, imagehash.hex_to_hash(random_hex()))
        assert len(index) == 1


class TestEquivalenceWithLinearScan:
    """The index must be a drop-in replacement, not an approximation."""

    def test_matches_on_random_corpora(self):
        random.seed(11)
        for _ in range(200):
            pairs = [(i + 1, random_hex()) for i in range(random.randint(1, 50))]
            probe = imagehash.hex_to_hash(random_hex())
            index = PHashIndex.from_pairs(pairs)
            assert index.nearest(probe) == find_nearest_canonical(probe, pairs)

    def test_matches_on_near_duplicates(self):
        """Small distances and ties are where a fast path is most likely to differ."""
        random.seed(12)
        for _ in range(200):
            pairs = [(i + 1, random_hex()) for i in range(random.randint(2, 30))]
            index = PHashIndex.from_pairs(pairs)
            probe = flip_bits(
                imagehash.hex_to_hash(random.choice(pairs)[1]), random.randint(0, 6)
            )
            assert index.nearest(probe) == find_nearest_canonical(probe, pairs)

    def test_from_pairs_on_empty_input(self):
        assert len(PHashIndex.from_pairs([])) == 0


class TestCompareAcceptsEitherForm:
    def test_index_and_list_classify_identically(self):
        base = imagehash.hex_to_hash(random_hex())
        pairs = [(1, str(flip_bits(base, 3)))]

        as_list = compare("x", base, pairs, crypto_exists=False, threshold=10)
        as_index = compare(
            "x", base, PHashIndex.from_pairs(pairs), crypto_exists=False, threshold=10
        )
        assert as_list == as_index
        assert as_list.outcome is Outcome.VARIANT

    def test_index_path_respects_the_threshold(self):
        base = imagehash.hex_to_hash(random_hex())
        pairs = [(1, str(flip_bits(base, 20)))]
        result = compare(
            "x", base, PHashIndex.from_pairs(pairs), crypto_exists=False, threshold=10
        )
        assert result.outcome is Outcome.NEW
