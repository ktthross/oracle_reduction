"""Top-level orchestrator: ties hashing, comparison, and storage together."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from .comparator import compare
from .config import Config
from .hasher import compute_crypto_hash, compute_phash
from .models import Outcome, ProcessingResult
from .storage import ImageStore


class ImageProcessor:
    """Process incoming images through the deduplication pipeline.

    Args:
        config: Pipeline configuration (thresholds, paths).
        store: Shared :class:`~oracle_reduction.storage.ImageStore` instance.
    """

    def __init__(self, config: Config, store: ImageStore) -> None:
        self.config = config
        self.store = store

    def process(
        self,
        image_path: Path,
        phash_threshold: int | None = None,
    ) -> ProcessingResult:
        """Classify and (if appropriate) persist *image_path*.

        The effective threshold is *phash_threshold* when provided; otherwise
        ``self.config.phash_threshold`` (which itself defaults to the value of
        the ``ORACLE_PHASH_THRESHOLD`` env var).

        Args:
            image_path: Path to the incoming image file.
            phash_threshold: Per-call override for the pHash Hamming distance
                threshold.  Pass ``None`` to use the configured default.

        Returns:
            A :class:`~oracle_reduction.models.ProcessingResult` describing
            what happened.
        """
        threshold = phash_threshold if phash_threshold is not None else self.config.phash_threshold
        image_path = Path(image_path)

        with Image.open(image_path) as img:
            # Keep a copy of the image open for hashing; PIL is lazy so we
            # load pixel data now while the file handle is valid.
            img.load()
            crypto_hash = compute_crypto_hash(img)
            phash = compute_phash(img)

            crypto_exists = self.store.crypto_hash_exists(crypto_hash)

            result = compare(
                crypto_hash=crypto_hash,
                phash=phash,
                stored_phashes=self.store.canonical_phash_index(),
                crypto_exists=crypto_exists,
                threshold=threshold,
            )

            if result.outcome is Outcome.IDENTICAL:
                return ProcessingResult(
                    outcome=Outcome.IDENTICAL,
                    image_path=image_path,
                )

            if result.outcome is Outcome.VARIANT:
                canonical = self.store.get_canonical(result.canonical_id)  # type: ignore[arg-type]
                variant_id = self.store.save_variant(
                    image=img,
                    source_path=image_path,
                    canonical_id=result.canonical_id,  # type: ignore[arg-type]
                    crypto_hash=crypto_hash,
                    phash=phash,
                    canonical=canonical,  # type: ignore[arg-type]
                )
                return ProcessingResult(
                    outcome=Outcome.VARIANT,
                    image_path=image_path,
                    canonical_id=result.canonical_id,
                    variant_id=variant_id,
                    phash_distance=result.phash_distance,
                )

            # Outcome.NEW
            canonical_id = self.store.save_canonical(
                image=img,
                source_path=image_path,
                crypto_hash=crypto_hash,
                phash=phash,
            )
            return ProcessingResult(
                outcome=Outcome.NEW,
                image_path=image_path,
                canonical_id=canonical_id,
                phash_distance=result.phash_distance,
            )
