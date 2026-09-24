"""SQLite-backed storage for canonical images and their resolution variants.

File layout under *storage_dir*::

    storage_dir/
    ├── canonicals/   # first-seen (canonical) copies
    └── variants/     # resolution-variant copies
"""

from __future__ import annotations

import os
import shutil
import sqlite3
from pathlib import Path

import imagehash
from PIL import Image

from .comparator import PHashIndex
from .hasher import phash_from_str
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
        store_mode: ``"copy"`` duplicates each incoming file into the store, so
            the store keeps working when the original is deleted.
            ``"symlink"`` links to the original instead: near-zero disk, but an
            entry breaks if its source is moved or removed.  Cleanup only ever
            unlinks entries inside ``storage_dir``, so source files are never
            deleted in either mode.
    """

    def __init__(
        self,
        db_path: str | Path,
        storage_dir: str | Path,
        store_mode: str = "copy",
    ) -> None:
        if store_mode not in ("copy", "symlink"):
            raise ValueError(f"store_mode must be 'copy' or 'symlink', got {store_mode!r}")
        self.db_path = Path(db_path)
        self.storage_dir = Path(storage_dir)
        self.store_mode = store_mode
        self._canonicals_dir = self.storage_dir / "canonicals"
        self._variants_dir = self.storage_dir / "variants"
        self._canonicals_dir.mkdir(parents=True, exist_ok=True)
        self._variants_dir.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_DDL)
        self._conn.commit()
        self._phash_index: PHashIndex | None = None

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

    def canonical_phash_index(self) -> PHashIndex:
        """Return the pHash index, building it from the database on first use.

        Kept in memory and appended to by :meth:`save_canonical`, so a scan
        reads the table once instead of once per image.  Anything that deletes
        canonicals must call :meth:`invalidate_phash_index`.
        """
        if self._phash_index is None:
            self._phash_index = PHashIndex.from_pairs(self.get_all_canonical_phashes())
        return self._phash_index

    def invalidate_phash_index(self) -> None:
        """Drop the cached index so the next lookup rebuilds it."""
        self._phash_index = None

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
        self._place(source_path, dest)

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

        canonical_id = cursor.lastrowid
        if self._phash_index is not None:
            self._phash_index.add(canonical_id, phash)
        return canonical_id  # type: ignore[return-value]

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
        self._place(source_path, dest)

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
            referenced.add(self._entry_key(path))
            # exists() follows symlinks, so a link whose target is gone counts
            # as missing — which is what a broken entry means for the gallery.
            if not path.exists():
                missing_files.append((kind, row_id, path))

        orphan_files = [
            path
            for directory in (self._canonicals_dir, self._variants_dir)
            for path in sorted(directory.iterdir())
            # is_symlink() catches broken links, which is_file() reports as
            # False; without it a dead link would never be swept up.
            if (path.is_file() or path.is_symlink())
            and self._entry_key(path) not in referenced
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
        if removed:
            self.invalidate_phash_index()
        return removed

    def _iter_referenced(self):
        """Yield ``(kind, row_id, file_path)`` for every row in both tables."""
        for kind, table in (("canonical", "images"), ("variant", "image_variants")):
            for row in self._conn.execute(f"SELECT id, file_path FROM {table}"):
                yield kind, row["id"], Path(row["file_path"])

    # ------------------------------------------------------------------
    # Deletion and reclassification
    # ------------------------------------------------------------------

    def delete_variant(self, variant_id: int) -> VariantRecord | None:
        """Delete a variant's row and unlink its managed file.

        Only the entry inside ``storage_dir`` is unlinked.  In symlink mode
        that entry is the link, so the original file it points at is left
        alone; callers that want the original gone must move it themselves.

        Args:
            variant_id: Primary key in ``image_variants``.

        Returns:
            The record as it was before deletion, or ``None`` if no variant
            has that id.
        """
        row = self._conn.execute(
            "SELECT * FROM image_variants WHERE id = ?", (variant_id,)
        ).fetchone()
        if row is None:
            return None

        record = self._to_variant_record(row)
        self._unlink_entry(record.file_path)
        self._conn.execute("DELETE FROM image_variants WHERE id = ?", (variant_id,))
        self._conn.commit()
        return record

    def delete_canonical(
        self, canonical_id: int
    ) -> tuple[ImageRecord, list[VariantRecord]] | None:
        """Delete a canonical, every variant matched to it, and their files.

        The variants go too because they exist only as relatives of this
        image; leaving them would strand rows pointing at a canonical that is
        no longer there.  As with :meth:`delete_variant`, only entries inside
        ``storage_dir`` are unlinked.

        Args:
            canonical_id: Primary key in ``images``.

        Returns:
            ``(canonical, variants)`` as they were before deletion, or ``None``
            if no canonical has that id.
        """
        record = self.get_canonical(canonical_id)
        if record is None:
            return None

        variants = self.get_variants_for_canonical(canonical_id)
        for variant in variants:
            self._unlink_entry(variant.file_path)
        self._unlink_entry(record.file_path)

        self._conn.execute(
            "DELETE FROM image_variants WHERE canonical_id = ?", (canonical_id,)
        )
        self._conn.execute("DELETE FROM images WHERE id = ?", (canonical_id,))
        self._conn.commit()
        self.invalidate_phash_index()
        return record, variants

    def promote_variant(self, variant_id: int) -> int | None:
        """Turn a variant into a canonical image in its own right.

        For a match made in error: two pictures whose pHashes land within the
        threshold but which are not the same image.  The managed entry moves
        from ``variants/`` to ``canonicals/`` and the row moves with it,
        carrying the hashes and dimensions already recorded, so nothing is
        re-read or re-hashed.

        The promoted image joins the pHash index, so later images can match
        against it — which is the point, since it represents a picture the
        index did not previously know about.

        Args:
            variant_id: Primary key in ``image_variants``.

        Returns:
            The new canonical's id, or ``None`` if no variant has that id.
        """
        row = self._conn.execute(
            "SELECT * FROM image_variants WHERE id = ?", (variant_id,)
        ).fetchone()
        if row is None:
            return None

        record = self._to_variant_record(row)
        # Name the destination from the original filename, not from the store
        # entry, whose stem already carries a hash suffix.
        dest = self._unique_dest(
            self._canonicals_dir, Path(record.filename), record.crypto_hash
        )
        if record.file_path.exists() or record.file_path.is_symlink():
            # Same directory tree, so this relocates the entry itself -- in
            # symlink mode the link rather than its target.
            os.replace(record.file_path, dest)

        cursor = self._conn.execute(
            """
            INSERT INTO images
                (file_path, filename, width, height, format, crypto_hash, phash, file_size)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(dest),
                record.filename,
                record.width,
                record.height,
                record.format,
                record.crypto_hash,
                record.phash,
                record.file_size,
            ),
        )
        self._conn.execute("DELETE FROM image_variants WHERE id = ?", (variant_id,))
        self._conn.commit()

        canonical_id = cursor.lastrowid
        if self._phash_index is not None:
            self._phash_index.add(canonical_id, phash_from_str(record.phash))
        return canonical_id  # type: ignore[return-value]

    @staticmethod
    def _unlink_entry(path: Path) -> None:
        """Remove a managed store entry, tolerating one that is already gone.

        ``unlink`` acts on the link rather than its target, so a symlink entry
        -- broken or not -- is removed without touching the original file.
        """
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass

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

    def _place(self, source: Path, dest: Path) -> None:
        """Put *source* into the store at *dest*, honouring ``store_mode``."""
        if self.store_mode == "symlink":
            # Absolute, so the link resolves no matter the working directory.
            os.symlink(Path(source).resolve(), dest)
        else:
            shutil.copy2(source, dest)

    @staticmethod
    def _entry_key(path: Path) -> Path:
        """Identity of a store entry, for comparing disk against database.

        Uses ``abspath`` rather than ``resolve``: in symlink mode the entry is
        the link itself, and resolving would compare the targets instead, so a
        broken link would stop matching its own row.
        """
        return Path(os.path.abspath(path))

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
