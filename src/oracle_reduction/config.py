"""Configuration for the oracle-reduction pipeline.

The pHash similarity threshold can be set via the ORACLE_PHASH_THRESHOLD
environment variable. A per-call override passed to ``ImageProcessor.process``
takes precedence over this default.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

_DEFAULT_PHASH_THRESHOLD = 10
_DEFAULT_DB_PATH = "oracle_reduction.db"
_DEFAULT_STORAGE_DIR = "image_store"
_DEFAULT_STORE_MODE = "copy"


@dataclass
class Config:
    """Pipeline configuration.

    Attributes:
        phash_threshold: Maximum Hamming distance between two pHashes for them
            to be considered the same image.  Loaded from the environment
            variable ``ORACLE_PHASH_THRESHOLD`` if set; otherwise defaults to 10.
        db_path: Path to the SQLite database file.
        storage_dir: Root directory where managed image files are stored.
        store_mode: How incoming files enter the store — ``"copy"`` (the
            default) duplicates the bytes, so the store survives the original
            being deleted; ``"symlink"`` links to the original instead, which
            costs almost no disk but makes the store depend on the source file
            staying put.  Read from ``ORACLE_STORE_MODE`` when set.
    """

    phash_threshold: int = field(
        default_factory=lambda: int(
            os.environ.get("ORACLE_PHASH_THRESHOLD", _DEFAULT_PHASH_THRESHOLD)
        )
    )
    db_path: str = field(default=_DEFAULT_DB_PATH)
    storage_dir: str = field(default=_DEFAULT_STORAGE_DIR)
    store_mode: str = field(
        default_factory=lambda: os.environ.get("ORACLE_STORE_MODE", _DEFAULT_STORE_MODE)
    )
