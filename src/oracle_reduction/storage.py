"""SQLite-backed storage for canonical images and their resolution variants.

File layout under *storage_dir*::

    storage_dir/
    ├── canonicals/   # first-seen (canonical) copies
    └── variants/     # resolution-variant copies
"""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import imagehash
from PIL import Image

from .models import ImageRecord, VariantRecord

_DDL = """
CREATE TABLE IF NOT EXISTS images (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path   TEXT    NOT NULL,
    filename    TEXT    NOT NULL,
    width       INTEGER NOT NULL,
    height      INTEGER NOT NULL,
    format      TEXT    NOT NULL,
    crypto_hash TEXT    NOT NULL UNIQUE,
    phash       TEXT    NOT NULL,
    file_size   INTEGER,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS image_variants (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_id     INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    file_path        TEXT    NOT NULL,
    filename         TEXT    NOT NULL,
    width            INTEGER NOT NULL,
    height           INTEGER NOT NULL,
    format           TEXT    NOT NULL,
    crypto_hash      TEXT    NOT NULL UNIQUE,
    phash            TEXT    NOT NULL,
    file_size        INTEGER,
    scale_factor     REAL,
    relationship     TEXT,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_images_crypto_hash    ON images(crypto_hash);
CREATE INDEX IF NOT EXISTS idx_images_phash          ON images(phash);
CREATE INDEX IF NOT EXISTS idx_variants_canonical_id ON image_variants(canonical_id);
CREATE INDEX IF NOT EXISTS idx_variants_crypto_hash  ON image_variants(crypto_hash);
"""


class ImageStore:
    """Manages canonical images and variants in SQLite + local filesystem.

    Args:
        db_path: Path to the SQLite database file (created if absent).
        storage_dir: Root directory for managed image files (created if absent).
    """

    def __init__(self, db_path: str | Path, storage_dir: str | Path) -> None:
        self.db_path = Path(db_path)
        self.storage_dir = Path(storage_dir)
        self._canonicals_dir = self.storage_dir / "canonicals"
        self._variants_dir = self.storage_dir / "variants"
        self._canonicals_dir.mkdir(parents=True, exist_ok=True)
        self._variants_dir.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_DDL)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def crypto_hash_exists(self, crypto_hash: str) -> bool:
        """Return True if *crypto_hash* appears in either table."""
        row = self._conn.execute(
            "SELECT 1 FROM images WHERE crypto_hash = ?", (crypto_hash,)
        ).fetchone()
        if row:
            return True
        row = self._conn.execute(
            "SELECT 1 FROM image_variants WHERE crypto_hash = ?", (crypto_hash,)
        ).fetchone()
        return row is not None

    def get_all_canonical_phashes(self) -> list[tuple[int, str]]:
        """Return ``[(id, phash_str), ...]`` for every canonical image."""
        rows = self._conn.execute("SELECT id, phash FROM images").fetchall()
        return [(row["id"], row["phash"]) for row in rows]

    def get_canonical(self, canonical_id: int) -> ImageRecord | None:
        """Fetch a canonical image record by primary key."""
        row = self._conn.execute(
            "SELECT * FROM images WHERE id = ?", (canonical_id,)
        ).fetchone()
        return self._to_image_record(row) if row else None

    def get_variants_for_canonical(self, canonical_id: int) -> list[VariantRecord]:
        """Return all variant records linked to *canonical_id*."""
        rows = self._conn.execute(
            "SELECT * FROM image_variants WHERE canonical_id = ?", (canonical_id,)
        ).fetchall()
        return [self._to_variant_record(row) for row in rows]

    def list_canonicals(self) -> list[ImageRecord]:
        """Return all canonical image records."""
        rows = self._conn.execute("SELECT * FROM images ORDER BY id").fetchall()
        return [self._to_image_record(row) for row in rows]

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def save_canonical(
        self,
        image: Image.Image,
        source_path: Path,
        crypto_hash: str,
        phash: imagehash.ImageHash,
    ) -> int:
        """Copy *source_path* into managed storage and insert a canonical row.

        Returns:
            The new row's primary key.
        """
        dest = self._unique_dest(self._canonicals_dir, source_path, crypto_hash)
        shutil.copy2(source_path, dest)

        width, height = image.size
        fmt = image.format or source_path.suffix.lstrip(".").upper() or "UNKNOWN"

        cursor = self._conn.execute(
            """
            INSERT INTO images
                (file_path, filename, width, height, format, crypto_hash, phash, file_size)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(dest),
                source_path.name,
                width,
                height,
                fmt,
                crypto_hash,
                str(phash),
                source_path.stat().st_size,
            ),
        )
        self._conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def save_variant(
        self,
        image: Image.Image,
        source_path: Path,
        canonical_id: int,
        crypto_hash: str,
        phash: imagehash.ImageHash,
        canonical: ImageRecord,
    ) -> int:
        """Copy *source_path* into managed storage and insert a variant row.

        Args:
            image: The opened variant image.
            source_path: Original file location.
            canonical_id: FK to the canonical ``images`` row.
            crypto_hash: SHA-256 of this image's pixel data.
            phash: Perceptual hash of this image.
            canonical: The canonical record (used to compute scale/relationship).

        Returns:
            The new row's primary key.
        """
        dest = self._unique_dest(self._variants_dir, source_path, crypto_hash)
        shutil.copy2(source_path, dest)

        width, height = image.size
        fmt = image.format or source_path.suffix.lstrip(".").upper() or "UNKNOWN"

        canonical_pixels = canonical.width * canonical.height
        variant_pixels = width * height
        if variant_pixels > canonical_pixels:
            relationship = "upscaled"
            scale_factor = (variant_pixels / canonical_pixels) ** 0.5
        elif variant_pixels < canonical_pixels:
            relationship = "downscaled"
            scale_factor = (variant_pixels / canonical_pixels) ** 0.5
        else:
            relationship = "same_resolution"
            scale_factor = 1.0

        cursor = self._conn.execute(
            """
            INSERT INTO image_variants
                (canonical_id, file_path, filename, width, height, format,
                 crypto_hash, phash, file_size, scale_factor, relationship)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                canonical_id,
                str(dest),
                source_path.name,
                width,
                height,
                fmt,
                crypto_hash,
                str(phash),
                source_path.stat().st_size,
                scale_factor,
                relationship,
            ),
        )
        self._conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Integrity
    # ------------------------------------------------------------------

    def audit_storage(self) -> dict[str, list]:
        """Report mismatches between managed files on disk and database rows.

        Returns:
            A dict with two keys:

            ``orphan_files``
                ``[Path, ...]`` — files under ``storage_dir`` that no row
                references.
            ``missing_files``
                ``[(kind, row_id, Path), ...]`` — rows whose ``file_path`` is
                absent from disk, where *kind* is ``"canonical"`` or
                ``"variant"``.
        """
        referenced: set[Path] = set()
        missing_files: list[tuple[str, int, Path]] = []

        for kind, row_id, path in self._iter_referenced():
            referenced.add(path.resolve())
            if not path.exists():
                missing_files.append((kind, row_id, path))

        orphan_files = [
            path
            for directory in (self._canonicals_dir, self._variants_dir)
            for path in sorted(directory.iterdir())
            if path.is_file() and path.resolve() not in referenced
        ]

        return {"orphan_files": orphan_files, "missing_files": missing_files}

    def cleanup_orphans(self) -> int:
        """Delete managed files that no database row references.

        Returns:
            The number of files removed.
        """
        removed = 0
        for path in self.audit_storage()["orphan_files"]:
            try:
                path.unlink()
            except OSError:
                continue
            removed += 1
        return removed

    def remove_dangling_records(self) -> int:
        """Delete rows whose managed file is missing from disk.

        Variants of a removed canonical are deleted as well, so no variant is
        left pointing at a canonical that no longer exists.  ``ON DELETE
        CASCADE`` does not fire here because SQLite enforces foreign keys only
        under ``PRAGMA foreign_keys = ON``, so the cascade is done explicitly.
        Any still-present files belonging to those variants become orphans,
        which :meth:`cleanup_orphans` then removes.

        Returns:
            The number of rows removed.
        """
        missing = self.audit_storage()["missing_files"]
        canonical_ids = [row_id for kind, row_id, _ in missing if kind == "canonical"]
        variant_ids = [row_id for kind, row_id, _ in missing if kind == "variant"]

        removed = 0
        for row_id in variant_ids:
            removed += self._conn.execute(
                "DELETE FROM image_variants WHERE id = ?", (row_id,)
            ).rowcount
        for row_id in canonical_ids:
            removed += self._conn.execute(
                "DELETE FROM image_variants WHERE canonical_id = ?", (row_id,)
            ).rowcount
            removed += self._conn.execute(
                "DELETE FROM images WHERE id = ?", (row_id,)
            ).rowcount

        self._conn.commit()
        return removed

    def _iter_referenced(self):
        """Yield ``(kind, row_id, file_path)`` for every row in both tables."""
        for kind, table in (("canonical", "images"), ("variant", "image_variants")):
            for row in self._conn.execute(f"SELECT id, file_path FROM {table}"):
                yield kind, row["id"], Path(row["file_path"])

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()

    def __enter__(self) -> "ImageStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unique_dest(directory: Path, source: Path, crypto_hash: str) -> Path:
        """Return a destination path that will not collide with existing files."""
        stem = source.stem
        suffix = source.suffix or ".bin"
        return directory / f"{stem}_{crypto_hash[:10]}{suffix}"

    @staticmethod
    def _to_image_record(row: sqlite3.Row) -> ImageRecord:
        return ImageRecord(
            id=row["id"],
            file_path=Path(row["file_path"]),
            filename=row["filename"],
            width=row["width"],
            height=row["height"],
            format=row["format"],
            crypto_hash=row["crypto_hash"],
            phash=row["phash"],
            file_size=row["file_size"],
            created_at=row["created_at"],
        )

    @staticmethod
    def _to_variant_record(row: sqlite3.Row) -> VariantRecord:
        return VariantRecord(
            id=row["id"],
            canonical_id=row["canonical_id"],
            file_path=Path(row["file_path"]),
            filename=row["filename"],
            width=row["width"],
            height=row["height"],
            format=row["format"],
            crypto_hash=row["crypto_hash"],
            phash=row["phash"],
            file_size=row["file_size"],
            scale_factor=row["scale_factor"],
            relationship=row["relationship"],
            created_at=row["created_at"],
        )
