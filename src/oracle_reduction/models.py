"""Data contracts for the image deduplication pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class Outcome(Enum):
    """Result of processing a single incoming image."""

    IDENTICAL = "identical"
    VARIANT = "variant"
    NEW = "new"


@dataclass
class ImageRecord:
    """A canonical (first-seen) image stored in the database."""

    id: int
    file_path: Path
    filename: str
    width: int
    height: int
    format: str
    crypto_hash: str
    phash: str
    file_size: int | None
    created_at: str


@dataclass
class VariantRecord:
    """A resolution variant linked to a canonical image."""

    id: int
    canonical_id: int
    file_path: Path
    filename: str
    width: int
    height: int
    format: str
    crypto_hash: str
    phash: str
    file_size: int | None
    scale_factor: float | None
    relationship: str | None
    created_at: str


@dataclass
class ProcessingResult:
    """Outcome of processing an incoming image through the pipeline."""

    outcome: Outcome
    image_path: Path | None = field(default=None)
    canonical_id: int | None = field(default=None)
    variant_id: int | None = field(default=None)
    phash_distance: int | None = field(default=None)
