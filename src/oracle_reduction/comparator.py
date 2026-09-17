"""Comparison logic: determine whether an incoming image is identical,
a resolution variant, or genuinely new.
"""

from __future__ import annotations

from dataclasses import dataclass

import imagehash
import numpy as np

from .models import Outcome


@dataclass
class ComparisonResult:
    """Outcome of comparing a new image against the stored corpus.

    Attributes:
        outcome: Classification of the incoming image.
        canonical_id: The matching canonical image's DB id, if applicable.
        phash_distance: Hamming distance to the nearest canonical pHash,
            or ``None`` when no canonicals exist yet.
    """

    outcome: Outcome
    canonical_id: int | None = None
    phash_distance: int | None = None


def find_nearest_canonical(
    new_phash: imagehash.ImageHash,
    stored: list[tuple[int, str]],
) -> tuple[int | None, int | None]:
    """Return the canonical whose pHash is closest to *new_phash*.

    Args:
        new_phash: The perceptual hash of the incoming image.
        stored: ``[(canonical_id, phash_hex_str), ...]`` from the database.

    Returns:
        ``(canonical_id, hamming_distance)`` of the closest match, or
        ``(None, None)`` when *stored* is empty.
    """
    if not stored:
        return None, None

    best_id: int | None = None
    best_dist: int | None = None

    for canonical_id, phash_str in stored:
        dist = new_phash - imagehash.hex_to_hash(phash_str)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_id = canonical_id

    return best_id, best_dist



# ---------------------------------------------------------------------------
# Fast nearest-neighbour search
# ---------------------------------------------------------------------------

if hasattr(np, "bitwise_count"):  # numpy >= 2.0
    _popcount = np.bitwise_count
else:  # pragma: no cover - exercised only on numpy 1.x
    _POPCOUNT_BYTE = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

    def _popcount(values: np.ndarray) -> np.ndarray:
        """Set bits per element, via a byte lookup table."""
        as_bytes = values.view(np.uint8).reshape(-1, values.dtype.itemsize)
        return _POPCOUNT_BYTE[as_bytes].sum(axis=1)


def pack_phash(phash: imagehash.ImageHash) -> np.uint64:
    """Pack a 64-bit pHash into a single ``uint64``.

    Byte order does not matter, only that every hash is packed the same way:
    Hamming distance is computed by XOR, which is order-agnostic.
    """
    return np.packbits(phash.hash.flatten()).view(np.uint64)[0]


class PHashIndex:
    """In-memory index over canonical pHashes, for nearest-neighbour search.

    The straightforward approach — fetching every stored hash and comparing one
    at a time — costs about 10 microseconds per stored image *per lookup*, so
    ingesting a corpus is quadratic and a six-figure collection takes many
    hours.  Packing the hashes into one ``uint64`` array and letting numpy XOR
    and popcount the whole array at once turns each lookup into a handful of
    vector operations.

    Entries are appended into a doubling buffer, so growing the index stays
    amortised O(1) rather than rebuilding the array on every insert.
    """

    def __init__(self, capacity: int = 1024) -> None:
        self._hashes = np.empty(max(capacity, 1), dtype=np.uint64)
        self._ids = np.empty(max(capacity, 1), dtype=np.int64)
        self._size = 0

    def __len__(self) -> int:
        return self._size

    @classmethod
    def from_pairs(cls, pairs: list[tuple[int, str]]) -> "PHashIndex":
        """Build an index from ``[(canonical_id, phash_hex), ...]`` rows."""
        index = cls(capacity=len(pairs) or 1)
        for canonical_id, phash_str in pairs:
            index.add(canonical_id, imagehash.hex_to_hash(phash_str))
        return index

    def add(self, canonical_id: int, phash: imagehash.ImageHash) -> None:
        """Append one canonical to the index."""
        if self._size == self._hashes.size:
            self._hashes = np.resize(self._hashes, self._size * 2)
            self._ids = np.resize(self._ids, self._size * 2)
        self._hashes[self._size] = pack_phash(phash)
        self._ids[self._size] = canonical_id
        self._size += 1

    def nearest(self, phash: imagehash.ImageHash) -> tuple[int | None, int | None]:
        """Return ``(canonical_id, hamming_distance)`` of the closest entry.

        Ties go to the earliest entry, matching
        :func:`find_nearest_canonical`.
        """
        if self._size == 0:
            return None, None

        distances = _popcount(self._hashes[: self._size] ^ pack_phash(phash))
        best = int(np.argmin(distances))
        return int(self._ids[best]), int(distances[best])


def compare(
    crypto_hash: str,
    phash: imagehash.ImageHash,
    stored_phashes: list[tuple[int, str]] | PHashIndex,
    crypto_exists: bool,
    threshold: int,
) -> ComparisonResult:
    """Classify an incoming image.

    Args:
        crypto_hash: SHA-256 hex digest of the normalised pixel data.
        phash: Perceptual hash of the image.
        stored_phashes: The stored corpus to match against — either a
            :class:`PHashIndex` (what the pipeline passes, and what keeps
            ingestion from going quadratic) or a plain list of
            ``(canonical_id, phash_str)`` pairs.
        crypto_exists: Whether *crypto_hash* already appears in the DB.
        threshold: Maximum Hamming distance to consider two images the same.

    Returns:
        A :class:`ComparisonResult` with the appropriate outcome.
    """
    if crypto_exists:
        return ComparisonResult(outcome=Outcome.IDENTICAL)

    if isinstance(stored_phashes, PHashIndex):
        best_id, best_dist = stored_phashes.nearest(phash)
    else:
        best_id, best_dist = find_nearest_canonical(phash, stored_phashes)

    if best_id is not None and best_dist is not None and best_dist <= threshold:
        return ComparisonResult(
            outcome=Outcome.VARIANT,
            canonical_id=best_id,
            phash_distance=best_dist,
        )

    return ComparisonResult(outcome=Outcome.NEW, phash_distance=best_dist)
