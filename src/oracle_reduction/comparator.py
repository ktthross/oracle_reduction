"""Comparison logic: determine whether an incoming image is identical,
a resolution variant, or genuinely new.
"""

from __future__ import annotations

from dataclasses import dataclass

import imagehash

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


def compare(
    crypto_hash: str,
    phash: imagehash.ImageHash,
    stored_phashes: list[tuple[int, str]],
    crypto_exists: bool,
    threshold: int,
) -> ComparisonResult:
    """Classify an incoming image.

    Args:
        crypto_hash: SHA-256 hex digest of the normalised pixel data.
        phash: Perceptual hash of the image.
        stored_phashes: All ``(canonical_id, phash_str)`` pairs from the DB.
        crypto_exists: Whether *crypto_hash* already appears in the DB.
        threshold: Maximum Hamming distance to consider two images the same.

    Returns:
        A :class:`ComparisonResult` with the appropriate outcome.
    """
    if crypto_exists:
        return ComparisonResult(outcome=Outcome.IDENTICAL)

    best_id, best_dist = find_nearest_canonical(phash, stored_phashes)

    if best_id is not None and best_dist is not None and best_dist <= threshold:
        return ComparisonResult(
            outcome=Outcome.VARIANT,
            canonical_id=best_id,
            phash_distance=best_dist,
        )

    return ComparisonResult(outcome=Outcome.NEW, phash_distance=best_dist)
