"""Fetch a slice of Urdu Wikipedia when the network will not serve the whole file.

Urdu Wikipedia ships as one 167.6 MB parquet. On this machine `hf_hub_download` stalls at
zero bytes on it and `HfFileSystem` streaming dies with `httpx.ReadTimeout`, but **small
HTTP range requests with many retries succeed at about 30 KB/s**. So the file is fetched the
only way that works:

1. the last 1 MB, which contains the parquet **footer** - the metadata describing every
   row group and its byte offsets;
2. the first `--megabytes` of the file, in 5 MB chunks, each retried independently;
3. a local file of the original's exact length, with those two pieces at their original
   offsets and a hole in between.

Offsets in a parquet footer are absolute, so row groups lying entirely inside the downloaded
head are readable from that sparse file exactly as they would be from the whole one. Groups
that reach into the hole are skipped by name rather than read as zeros.

Each chunk is written as it arrives and re-running skips what is already on disk, so an
interrupted fetch resumes instead of restarting.

    python src/fetch.py --megabytes 20
"""

from __future__ import annotations

import argparse
import time
import urllib.error
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
CHUNKS = DATA / "chunks"
URL = (
    "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/"
    "20231101.ur/train-00000-of-00001.parquet"
)
CHUNK_BYTES = 5_000_000
FOOTER_BYTES = 1_000_000
ATTEMPTS = 30


def sparse_path() -> Path:
    return DATA / "urdu_wikipedia.parquet"


def articles_path() -> Path:
    return DATA / "urdu_articles.parquet"


def total_size() -> int:
    request = urllib.request.Request(URL, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:
        return int(response.headers["Content-Length"])


def fetch_range(start: int, end: int, target: Path) -> None:
    """One inclusive byte range, retried. Small ranges are the point - a 5 MB request that
    fails costs 5 MB, while one 167 MB request that fails costs everything."""
    if target.exists() and target.stat().st_size == end - start + 1:
        return
    last = None
    for attempt in range(ATTEMPTS):
        try:
            request = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(request, timeout=300) as response:
                payload = response.read()
            if len(payload) == end - start + 1:
                target.write_bytes(payload)
                return
            last = f"short read: {len(payload)} of {end - start + 1}"
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = f"{type(exc).__name__}: {exc}"
        time.sleep(min(2**attempt, 20))
    raise RuntimeError(f"range {start}-{end} failed after {ATTEMPTS} attempts ({last})")


def fetch(megabytes: int = 20) -> Path:
    CHUNKS.mkdir(parents=True, exist_ok=True)
    size = total_size()
    head_bytes = megabytes * 1_000_000
    print(f"  remote file: {size / 1e6:.1f} MB; taking the first {head_bytes / 1e6:.0f} MB")

    footer = CHUNKS / "footer.bin"
    fetch_range(size - FOOTER_BYTES, size - 1, footer)
    print(f"  footer: {footer.stat().st_size / 1e6:.1f} MB")

    parts = []
    for index, start in enumerate(range(0, head_bytes, CHUNK_BYTES)):
        end = min(start + CHUNK_BYTES, head_bytes) - 1
        chunk = CHUNKS / f"chunk-{index:03d}.bin"
        began = time.time()
        cached = chunk.exists() and chunk.stat().st_size == end - start + 1
        fetch_range(start, end, chunk)
        parts.append(chunk)
        note = "cached" if cached else f"{time.time() - began:.0f}s"
        print(f"  chunk {index}: bytes {start:,}-{end:,}  [{note}]", flush=True)

    target = sparse_path()
    with open(target, "wb") as handle:
        for chunk in parts:
            handle.write(chunk.read_bytes())
        handle.seek(size - FOOTER_BYTES)
        handle.write(footer.read_bytes())
    print(f"  assembled {target.name} at the original length ({size:,} bytes, most of it a hole)")
    return target


def readable_groups(reader, head_bytes: int) -> list[int]:
    """Row groups lying entirely inside the downloaded head.

    A group reaching into the hole would read as zeros - not an error, just wrong data - so
    the boundary is computed from the footer rather than discovered by trying.
    """
    usable = []
    for i in range(reader.num_row_groups):
        group = reader.metadata.row_group(i)
        columns = [group.column(c) for c in range(group.num_columns)]
        starts = [c.file_offset for c in columns if c.file_offset > 0]
        if not starts:
            continue
        # Row-group size is not exposed directly on every pyarrow version; the sum of its
        # column chunks is the same quantity and is always available.
        span = sum(c.total_compressed_size for c in columns)
        if min(starts) + span <= head_bytes:
            usable.append(i)
    return usable


def extract(megabytes: int = 20) -> Path:
    """Read the usable row groups out of the sparse file into a small, self-contained one."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    reader = pq.ParquetFile(sparse_path())
    usable = readable_groups(reader, megabytes * 1_000_000)
    print(f"  {reader.num_row_groups} row groups in the file; {len(usable)} fully downloaded")
    if not usable:
        raise RuntimeError("no complete row group in the downloaded head; fetch more megabytes")

    table = pa.concat_tables([reader.read_row_group(i, columns=["title", "text"]) for i in usable])
    pq.write_table(table, articles_path())
    print(
        f"  wrote {articles_path().name}: {table.num_rows:,} articles, "
        f"{articles_path().stat().st_size / 1e6:.1f} MB"
    )
    return articles_path()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--megabytes", type=int, default=20)
    args = parser.parse_args()
    fetch(args.megabytes)
    extract(args.megabytes)


if __name__ == "__main__":
    main()
