"""Stage 1 -- pull a bounded slice of Amazon Reviews 2023 item metadata.

Raw JSONL over HTTP, not `datasets` (script loaders were dropped in 4.x), and
stdlib urllib rather than requests -- one fewer dependency between a reader
and the first number.

The size argument for streaming: `meta_Automotive.jsonl` is about 5.3 GB and
we want 40k of its ~2M items. This reads lines until the cap is hit and then
closes the connection, so the remaining ~5.2 GB is never transferred.

Known bias, stated plainly: taking the first N lines is a prefix, not a
random sample. The file's ordering is not documented as random, so the slice
may skew. Fixing it properly means reading the whole file to reservoir-sample,
which trades a ~10 second download for a ~5 GB one per category. The prefix is
the right call for a project that has to run on a laptop, but any per-category
result should be read as "the first 40k Automotive items", not "Automotive".
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterator

from voice_order import config

USER_AGENT = "voice-order-agent (+https://github.com/iam-neelesh8/voice-order-agent)"
_TIMEOUT = 60


def category_url(category: str) -> str:
    cfg = config.load("catalog")
    return str(cfg["source.url_template"]).format(category=category)


def _count_lines(path: Path) -> int:
    with path.open("rb") as fh:
        return sum(1 for _ in fh)


def download_category(
    category: str, limit: int, dest_dir: Path | None = None, force: bool = False
) -> Path:
    """Stream `meta_{category}.jsonl`, stop at `limit` items, write to disk.

    Resumable in the only sense that matters here: if the file already holds
    at least `limit` lines, it is left alone. A partial file from an
    interrupted run is discarded and refetched, because a truncated JSONL line
    is indistinguishable from a malformed record downstream.
    """
    dest_dir = Path(dest_dir or config.load("catalog")["download_dir"])
    if not dest_dir.is_absolute():
        dest_dir = config.ROOT / dest_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"meta_{category}.jsonl"

    if dest.is_file() and not force and _count_lines(dest) >= limit:
        return dest

    url = category_url(category)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    tmp = dest.with_suffix(".jsonl.partial")

    written = 0
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
            with tmp.open("wb") as out:
                for line in response:
                    if not line.strip():
                        continue
                    out.write(line if line.endswith(b"\n") else line + b"\n")
                    written += 1
                    if written >= limit:
                        break
        # Closing the response here is what stops the transfer; everything
        # past this point of the file is never downloaded.
    except urllib.error.HTTPError as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{category}: HTTP {exc.code} from {url}") from exc
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    if written == 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{category}: no records returned from {url}")

    tmp.replace(dest)
    return dest


def iter_raw_items(path: Path) -> Iterator[dict]:
    """Yield decoded JSONL records, skipping malformed lines.

    Malformed lines are counted and reported rather than raised: a single bad
    record in a 40k-line dump should not cost the whole load, but a *lot* of
    them means the file is wrong and you need to know.
    """
    bad = 0
    total = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if isinstance(record, dict):
                yield record
            else:
                bad += 1

    if bad:
        share = bad / total if total else 0
        print(f"  ! {path.name}: skipped {bad}/{total} malformed lines ({share:.1%})")


def download_all(force: bool = False) -> dict[str, Path]:
    """Download every category in `configs/catalog.yaml`. Returns name -> path."""
    cfg = config.load("catalog")
    paths: dict[str, Path] = {}
    for entry in cfg["categories"]:
        name, limit = entry["name"], int(entry["limit"])
        print(f"  {name}: streaming first {limit:,} items ...", flush=True)
        paths[name] = download_category(name, limit, force=force)
    return paths
