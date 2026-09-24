"""A reader and the writer have to be able to run at the same time.

The gallery is browsed while scans run.  Under SQLite's default rollback
journal that combination fails outright, so this is really one test: does a
commit still land while something else is holding a read open.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from oracle_reduction.hasher import compute_crypto_hash, compute_phash
from oracle_reduction.storage import ImageStore


@pytest.fixture()
def store(tmp_path: Path) -> ImageStore:
    s = ImageStore(db_path=tmp_path / "test.db", storage_dir=tmp_path / "store")
    yield s
    s.close()


def save_canonical(store: ImageStore, path: Path, shade: int) -> int:
    img = Image.fromarray(
        np.full((64, 64, 3), shade, dtype=np.uint8), mode="RGB"
    )
    img.save(path, format="PNG")
    with Image.open(path) as opened:
        opened.load()
        return store.save_canonical(
            image=opened,
            source_path=path,
            crypto_hash=compute_crypto_hash(opened),
            phash=compute_phash(opened),
        )


class TestConcurrentAccess:
    def test_journal_mode_is_wal(self, store: ImageStore):
        mode = store._conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"

    def test_write_commits_while_a_read_is_open(self, store: ImageStore, tmp_path: Path):
        """The case that used to raise "database is locked"."""
        save_canonical(store, tmp_path / "first.png", 40)

        reader = sqlite3.connect(str(store.db_path))
        try:
            reader.execute("BEGIN")
            reader.execute("SELECT COUNT(*) FROM images").fetchone()

            # Under a rollback journal the reader's shared lock would stop
            # this commit; under WAL it goes through.
            save_canonical(store, tmp_path / "second.png", 200)
        finally:
            reader.close()

        assert len(store.list_canonicals()) == 2

    def test_reader_sees_rows_committed_after_it_connected(
        self, store: ImageStore, tmp_path: Path
    ):
        reader = sqlite3.connect(str(store.db_path))
        try:
            assert reader.execute("SELECT COUNT(*) FROM images").fetchone()[0] == 0
            save_canonical(store, tmp_path / "later.png", 120)
            assert reader.execute("SELECT COUNT(*) FROM images").fetchone()[0] == 1
        finally:
            reader.close()
