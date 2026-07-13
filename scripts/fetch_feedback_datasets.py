#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Fetch checksum-pinned datasets for the feedback-regime validation matrix."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import tomllib

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


@dataclass(frozen=True)
class Source:
    source_id: str
    lane: str
    role: str
    path: PurePosixPath
    url: str
    sha256: str
    size: int
    revision: str
    source_page: str


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sources(path: Path) -> list[Source]:
    with path.open("rb") as handle:
        manifest = tomllib.load(handle)
    if manifest.get("format_version") != 1:
        raise ValueError(f"unsupported feedback source manifest: {path}")

    sources: list[Source] = []
    seen_ids: set[str] = set()
    seen_paths: set[PurePosixPath] = set()
    for entry in manifest.get("source", []):
        relative = PurePosixPath(entry["path"])
        source_id = entry["id"]
        digest = entry["sha256"]
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"source {source_id} has unsafe path {relative}")
        if source_id in seen_ids or relative in seen_paths:
            raise ValueError(f"duplicate source id or path: {source_id} {relative}")
        if not entry["url"].startswith("https://"):
            raise ValueError(f"source {source_id} must use HTTPS")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"source {source_id} has invalid SHA-256")
        if entry["bytes"] < 0:
            raise ValueError(f"source {source_id} has invalid byte count")
        seen_ids.add(source_id)
        seen_paths.add(relative)
        sources.append(
            Source(
                source_id=source_id,
                lane=entry["lane"],
                role=entry["role"],
                path=relative,
                url=entry["url"],
                sha256=digest,
                size=entry["bytes"],
                revision=entry["revision"],
                source_page=entry["source_page"],
            )
        )
    if not sources:
        raise ValueError(f"feedback source manifest is empty: {path}")
    return sources


def verify_existing(source: Source, destination: Path) -> bool:
    if not destination.exists():
        return False
    actual_size = destination.stat().st_size
    actual_digest = sha256_path(destination)
    if actual_size == source.size and actual_digest == source.sha256:
        print(f"cached     {source.source_id:24} {actual_size:>9} bytes")
        return True
    raise FileExistsError(
        f"refusing to replace {destination}: expected {source.size} bytes and "
        f"{source.sha256}, found {actual_size} bytes and {actual_digest}"
    )


def fetch_source(source: Source, root: Path) -> int:
    destination = root.joinpath(*source.path.parts)
    if verify_existing(source, destination):
        return source.size

    destination.parent.mkdir(parents=True, exist_ok=True)
    # Manifest loading rejects every scheme except HTTPS.
    request = urllib.request.Request(  # noqa: S310
        source.url,
        headers={"User-Agent": "muxer-feedback-data/1"},
    )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".part",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            digest = hashlib.sha256()
            size = 0
            with urllib.request.urlopen(  # noqa: S310
                request, timeout=60
            ) as response:
                while chunk := response.read(1024 * 1024):
                    temporary.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
            temporary.flush()
            os.fsync(temporary.fileno())

        actual_digest = digest.hexdigest()
        if size != source.size or actual_digest != source.sha256:
            raise ValueError(
                f"source {source.source_id} failed verification: expected "
                f"{source.size} bytes and {source.sha256}, found {size} bytes "
                f"and {actual_digest}"
            )
        os.replace(temporary_path, destination)
        temporary_path = None
        print(f"downloaded {source.source_id:24} {size:>9} bytes")
        return size
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=SCRIPT_DIR / "feedback_sources.toml",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=REPO_ROOT / "data/feedback/raw",
    )
    parser.add_argument(
        "--lane",
        action="append",
        dest="lanes",
        help="fetch only this lane; repeat to select more than one",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        sources = load_sources(args.manifest)
        known_lanes = {source.lane for source in sources}
        requested_lanes = set(args.lanes or known_lanes)
        unknown_lanes = requested_lanes - known_lanes
        if unknown_lanes:
            names = ", ".join(sorted(unknown_lanes))
            raise ValueError(f"unknown feedback lanes: {names}")
        selected = [source for source in sources if source.lane in requested_lanes]
        total = sum(fetch_source(source, args.destination) for source in selected)
    except (
        OSError,
        ValueError,
        tomllib.TOMLDecodeError,
        urllib.error.URLError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"verified {len(selected)} sources ({total} bytes) under {args.destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
