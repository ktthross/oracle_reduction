"""Command-line interface for the oracle-reduction pipeline."""

from __future__ import annotations

from pathlib import Path

import click

from .config import Config
from .models import Outcome
from .processor import ImageProcessor
from .storage import ImageStore


@click.group()
@click.option(
    "--db",
    default=None,
    envvar="ORACLE_DB_PATH",
    help="SQLite database path (default: oracle_reduction.db).",
)
@click.option(
    "--storage-dir",
    default=None,
    envvar="ORACLE_STORAGE_DIR",
    help="Root directory for managed image files (default: image_store).",
)
@click.option(
    "--link",
    is_flag=True,
    default=False,
    help="Symlink stored images to their originals instead of copying them.",
)
@click.pass_context
def cli(ctx: click.Context, db: str | None, storage_dir: str | None, link: bool) -> None:
    """Oracle Reduction — image deduplication and variant detection."""
    ctx.ensure_object(dict)
    config = Config()
    if db:
        config.db_path = db
    if storage_dir:
        config.storage_dir = storage_dir
    if link:
        config.store_mode = "symlink"
    ctx.obj["config"] = config
    ctx.obj["store"] = ImageStore(
        config.db_path, config.storage_dir, store_mode=config.store_mode
    )


@cli.command()
@click.argument("image_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--threshold",
    type=int,
    default=None,
    help="pHash Hamming distance threshold (overrides env var / config default).",
)
@click.pass_context
def process(ctx: click.Context, image_path: Path, threshold: int | None) -> None:
    """Process IMAGE_PATH: classify it as identical, variant, or new."""
    config: Config = ctx.obj["config"]
    store: ImageStore = ctx.obj["store"]

    processor = ImageProcessor(config, store)
    result = processor.process(image_path, phash_threshold=threshold)

    outcome_label = {
        Outcome.IDENTICAL: "IDENTICAL — discarded (already in corpus)",
        Outcome.VARIANT: "VARIANT   — saved and linked to canonical",
        Outcome.NEW: "NEW       — saved as canonical",
    }[result.outcome]

    click.echo(f"Outcome    : {outcome_label}")
    if result.canonical_id is not None:
        click.echo(f"Canonical  : id={result.canonical_id}")
    if result.variant_id is not None:
        click.echo(f"Variant    : id={result.variant_id}")
    if result.phash_distance is not None:
        click.echo(f"pHash dist : {result.phash_distance}")


@cli.command("list-variants")
@click.option("--canonical-id", type=int, required=True, help="Canonical image id.")
@click.pass_context
def list_variants(ctx: click.Context, canonical_id: int) -> None:
    """List all resolution variants linked to CANONICAL_ID."""
    store: ImageStore = ctx.obj["store"]

    canonical = store.get_canonical(canonical_id)
    if canonical is None:
        click.echo(f"No canonical image found with id {canonical_id}.", err=True)
        ctx.exit(1)
        return

    click.echo(
        f"Canonical  : id={canonical.id}  {canonical.filename}"
        f"  {canonical.width}x{canonical.height}  ({canonical.format})"
    )
    click.echo(f"Stored at  : {canonical.file_path}")

    variants = store.get_variants_for_canonical(canonical_id)
    if not variants:
        click.echo("No variants found.")
        return

    click.echo(f"\n{len(variants)} variant(s):")
    for v in variants:
        click.echo(
            f"  [{v.id}] {v.filename}  {v.width}x{v.height}"
            f"  {v.relationship}  scale={v.scale_factor:.3f}"
        )


@cli.command("list-canonicals")
@click.pass_context
def list_canonicals(ctx: click.Context) -> None:
    """List all canonical images in the corpus."""
    store: ImageStore = ctx.obj["store"]
    canonicals = store.list_canonicals()
    if not canonicals:
        click.echo("No canonical images in the corpus yet.")
        return
    for img in canonicals:
        click.echo(f"  [{img.id}] {img.filename}  {img.width}x{img.height}  ({img.format})")


@cli.command()
@click.pass_context
def audit(ctx: click.Context) -> None:
    """Report orphaned files and records whose image file is missing."""
    store: ImageStore = ctx.obj["store"]
    result = store.audit_storage()
    orphans = result["orphan_files"]
    dangling = result["missing_files"]

    if orphans:
        click.echo(f"{len(orphans)} orphaned file(s) (on disk but not in DB):")
        for path in orphans:
            click.echo(f"  {path}")
    if dangling:
        click.echo(f"{len(dangling)} dangling record(s) (in DB but file missing):")
        for kind, record_id, path in dangling:
            click.echo(f"  {kind} id={record_id}: {path}")
    if not orphans and not dangling:
        click.echo("Image storage is clean.")


@cli.command()
@click.pass_context
def cleanup(ctx: click.Context) -> None:
    """Delete orphaned files and records whose image file is missing."""
    store: ImageStore = ctx.obj["store"]
    orphan_count = store.cleanup_orphans()
    dangling_count = store.remove_dangling_records()
    click.echo(f"Removed {orphan_count} orphaned file(s).")
    click.echo(f"Removed {dangling_count} dangling DB record(s).")
