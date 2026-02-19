"""Image hashing utilities.

Two complementary hashes are produced for each image:

* **Cryptographic hash** (SHA-256) — detects byte-perfect duplicates after
  normalising to RGB pixel data, independent of file format.
* **Perceptual hash** (pHash / DCT) — detects visually identical images
  regardless of resolution or minor compression differences.
"""

from __future__ import annotations

import hashlib

import imagehash
from PIL import Image


def normalise(image: Image.Image) -> Image.Image:
    """Return a copy of *image* converted to RGB with EXIF stripped."""
    return image.convert("RGB")


def compute_crypto_hash(image: Image.Image) -> str:
    """Compute a SHA-256 hex digest of the normalised pixel data.

    Args:
        image: Any PIL Image (any mode, any format).

    Returns:
        64-character lowercase hex string.
    """
    rgb = normalise(image)
    return hashlib.sha256(rgb.tobytes()).hexdigest()


def compute_phash(image: Image.Image) -> imagehash.ImageHash:
    """Compute a 64-bit DCT perceptual hash.

    Args:
        image: Any PIL Image.

    Returns:
        An ``imagehash.ImageHash`` that supports ``-`` (Hamming distance)
        and ``str()`` for serialisation.
    """
    return imagehash.phash(image)


def phash_from_str(phash_str: str) -> imagehash.ImageHash:
    """Deserialise a pHash that was previously stored as a hex string."""
    return imagehash.hex_to_hash(phash_str)
