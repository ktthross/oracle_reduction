# oracle-reduction

Image deduplication and resolution-variant detection. Give it images one at a
time; it tells you whether each one is new, a different-resolution copy of
something it already has, or a byte-perfect duplicate — and keeps the ones
worth keeping.

It is the engine under [rapture](../rapture), and works standalone.

## Install

```bash
uv venv
uv pip install -e .
```

## Quick start

```bash
oracle-reduction process photo.png
oracle-reduction list-canonicals
oracle-reduction list-variants --canonical-id 1
```

By default this writes `oracle_reduction.db` and an `image_store/` directory
in the working directory. Point them somewhere else with `--db` and
`--storage-dir`, or the `ORACLE_DB_PATH` and `ORACLE_STORAGE_DIR` environment
variables.

## How an image is classified

Each incoming image is hashed twice:

- **SHA-256** over the normalised RGB pixel data — catches byte-perfect
  duplicates regardless of container format.
- **pHash** (64-bit DCT perceptual hash) — catches the same picture at a
  different resolution or compression level.

From those two:

| Outcome | Meaning | What happens |
|---------|---------|--------------|
| `IDENTICAL` | The crypto hash is already stored | Discarded |
| `VARIANT` | pHash within the threshold of a canonical | Stored, linked to that canonical |
| `NEW` | No match | Stored as a canonical |

Variants record their `scale_factor` and `relationship` (`upscaled`,
`downscaled`, `same_resolution`) relative to their canonical.

### The pHash threshold

Maximum Hamming distance for two images to count as the same picture.
Defaults to 10; set it per call with `--threshold`, or globally with
`ORACLE_PHASH_THRESHOLD`.

| Threshold | Behaviour |
|-----------|-----------|
| 0 | Only perceptually identical images match |
| 5 | Strict — catches rescales and light recompression |
| 10 (default) | Balanced |
| 15+ | Loose — starts grouping visually distinct images |

Lower produces more canonicals and fewer false matches; higher produces fewer
canonicals and risks grouping unrelated pictures.

## Copying vs linking

By default a kept image is **copied** into the store, so the store survives
the original being deleted. `--link` stores a symlink instead — about 130
bytes per image rather than the image — at the cost of the store breaking if
an original is moved or deleted.

```bash
oracle-reduction --link process photo.png
```

Cleanup only ever unlinks entries inside the store. Original files are never
deleted in either mode.

## Storage layout

```
oracle_reduction.db      # images, image_variants
image_store/
  canonicals/            # first-seen images
  variants/              # resolution variants
```

Entries are named `{stem}_{crypto_hash[:10]}{suffix}`, so two files of the
same name do not collide.

The database runs in WAL mode, so the store can be read while something else
is writing to it — a gallery stays browsable during a long ingest.

## Integrity

```bash
oracle-reduction audit     # report problems
oracle-reduction cleanup   # fix them
```

`audit` reports two kinds of drift between disk and database:

- **Orphaned files** — in the store, referenced by no row.
- **Dangling records** — rows whose file is missing. In symlink mode this
  includes links whose target has gone.

`cleanup` removes both. Deleting a canonical takes its variants with it, since
a variant only exists as a relative of its canonical.

## Library use

```python
from oracle_reduction.config import Config
from oracle_reduction.processor import ImageProcessor
from oracle_reduction.storage import ImageStore

config = Config(db_path="images.db", storage_dir="store")
with ImageStore(config.db_path, config.storage_dir) as store:
    result = ImageProcessor(config, store).process(Path("photo.png"))
    print(result.outcome, result.canonical_id, result.phash_distance)
```

Canonical pHashes are held in an in-memory `PHashIndex` and compared with
vectorised XOR + popcount, so ingesting a large corpus does not go quadratic.
Anything that deletes canonicals behind the store's back must call
`invalidate_phash_index()`.

## Tests

```bash
uv pip install -e ".[dev]"
.venv/bin/python -m pytest
```
