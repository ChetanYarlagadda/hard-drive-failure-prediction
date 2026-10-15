"""Fetch selected days of Backblaze Drive Stats straight out of the quarterly zips with HTTP range requests.

Each quarterly archive (1.2-2.1 GB) holds one CSV per day. Reading the zip's central directory (a few kB at the end
of the file) gives every member's offset, compressed size and CRC-32, so a single day can be downloaded on its own
(about 13-15 MB compressed, 130 MB as CSV), inflated, CRC-checked and reduced to the columns this project uses.
Each day is stored once as a small Parquet file ("bronze" layer) in ``data/raw/bronze/``.
"""

from __future__ import annotations

import io
import json
import struct
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Iterable

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
import requests

BASE_URL = "https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data"
SMART_IDS = (1, 3, 4, 5, 7, 9, 10, 12, 184, 187, 188, 189, 190, 192, 193, 194, 197, 198, 199, 240, 241, 242)
META_COLUMNS = ("date", "serial_number", "model", "capacity_bytes", "failure", "datacenter")
BRONZE_COLUMNS = META_COLUMNS + tuple(f"smart_{i}_raw" for i in SMART_IDS)
BRONZE_VERSION = "b1"  # bump when BRONZE_COLUMNS or the parsing changes

Fetcher = Callable[[str, int, int], bytes]  # (url, first byte, last byte inclusive) -> bytes


@dataclass(frozen=True)
class ZipMember:
    """One file inside a remote zip archive, as listed in its central directory."""

    name: str
    method: int
    crc32: int
    compressed_size: int
    size: int
    header_offset: int


def quarter_of(day: date) -> str:
    """Backblaze archive name of the quarter containing ``day``, e.g. 'Q4_2025'."""
    return f"Q{(day.month - 1) // 3 + 1}_{day.year}"


def zip_url(quarter: str) -> str:
    """URL of the quarterly Drive Stats archive."""
    return f"{BASE_URL}/data_{quarter}.zip"


def date_range(start: date, end: date) -> list[date]:
    """Every day from ``start`` to ``end`` inclusive."""
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def http_fetcher(session: requests.Session | None = None, retries: int = 4, timeout: int = 120) -> Fetcher:
    """Range-request fetcher with retries and a check that the server honoured the range."""
    s = session or requests.Session()

    def fetch(url: str, first: int, last: int) -> bytes:
        for attempt in range(retries):
            try:
                r = s.get(url, headers={"Range": f"bytes={first}-{last}"}, timeout=timeout)
                r.raise_for_status()
                if r.status_code != 206 or len(r.content) != last - first + 1:
                    raise IOError(f"range {first}-{last} of {url}: status {r.status_code}, {len(r.content)} bytes")
                return r.content
            except (requests.RequestException, IOError):
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)
        raise AssertionError("unreachable")

    return fetch


def remote_size(url: str, session: requests.Session | None = None) -> int:
    """Content-Length of a remote file."""
    r = (session or requests).head(url, timeout=60, allow_redirects=True)
    r.raise_for_status()
    return int(r.headers["Content-Length"])


def parse_central_directory(eocd_tail: bytes, fetch_cd: Callable[[int, int], bytes]) -> dict[str, ZipMember]:
    """Members of a zip given the archive's last bytes (containing the end-of-central-directory record).

    ``fetch_cd(offset, size)`` returns the central directory bytes. Zip64 archives are not supported (the Drive
    Stats archives are below 4 GiB).
    """
    i = eocd_tail.rfind(b"PK\x05\x06")
    if i < 0:
        raise ValueError("end of central directory record not found")
    n_entries, cd_size, cd_offset = struct.unpack("<HII", eocd_tail[i + 10:i + 20])
    if 0xFFFFFFFF in (cd_size, cd_offset):
        raise ValueError("zip64 archives are not supported")
    cd = fetch_cd(cd_offset, cd_size)
    members: dict[str, ZipMember] = {}
    p = 0
    for _ in range(n_entries):
        if struct.unpack("<I", cd[p:p + 4])[0] != 0x02014B50:
            raise ValueError(f"bad central directory signature at {p}")
        method = struct.unpack("<H", cd[p + 10:p + 12])[0]
        crc, csize, usize = struct.unpack("<III", cd[p + 16:p + 28])
        nlen, elen, clen = struct.unpack("<HHH", cd[p + 28:p + 34])
        offset = struct.unpack("<I", cd[p + 42:p + 46])[0]
        name = cd[p + 46:p + 46 + nlen].decode("utf-8")
        members[name] = ZipMember(name, method, crc, csize, usize, offset)
        p += 46 + nlen + elen + clen
    return members


def list_members(url: str, fetch: Fetcher, size: int) -> dict[str, ZipMember]:
    """Central directory of a remote zip (two small range requests)."""
    tail_start = max(0, size - 65_557)  # EOCD (22 bytes) + maximum comment length
    tail = fetch(url, tail_start, size - 1)
    return parse_central_directory(tail, lambda off, n: fetch(url, off, off + n - 1))


def read_member(url: str, m: ZipMember, fetch: Fetcher) -> bytes:
    """Download and inflate one zip member, verifying its size and CRC-32."""
    local = fetch(url, m.header_offset, m.header_offset + 29)
    if struct.unpack("<I", local[:4])[0] != 0x04034B50:
        raise ValueError(f"bad local header for {m.name}")
    nlen, elen = struct.unpack("<HH", local[26:30])
    start = m.header_offset + 30 + nlen + elen
    comp = fetch(url, start, start + m.compressed_size - 1) if m.compressed_size else b""
    if m.method == 0:
        raw = comp
    elif m.method == 8:
        raw = zlib.decompressobj(-15).decompress(comp)
    else:
        raise ValueError(f"unsupported compression method {m.method} for {m.name}")
    if len(raw) != m.size or zlib.crc32(raw) != m.crc32:
        raise IOError(f"size/CRC mismatch for {m.name}")
    return raw


def parse_daily_csv(raw: bytes, columns: Iterable[str] = BRONZE_COLUMNS) -> pa.Table:
    """Parse one Drive Stats daily CSV, keeping ``columns`` (SMART raw values as float64)."""
    cols = list(columns)
    types: dict[str, pa.DataType] = {"date": pa.date32(), "serial_number": pa.string(), "model": pa.string(),
                                     "capacity_bytes": pa.int64(), "failure": pa.int8(), "datacenter": pa.string()}
    types.update({c: pa.float64() for c in cols if c.startswith("smart_")})
    header = raw[:raw.index(b"\n")].decode("utf-8").strip().split(",")
    present = [c for c in cols if c in header]
    t = pacsv.read_csv(io.BytesIO(raw), read_options=pacsv.ReadOptions(use_threads=True, block_size=1 << 24),
                       convert_options=pacsv.ConvertOptions(include_columns=present,
                                                            column_types={c: types[c] for c in present}))
    for c in cols:  # columns missing from a given day's schema become all-null
        if c not in present:
            t = t.append_column(c, pa.nulls(t.num_rows, types.get(c, pa.string())))
    return t.select(cols)


def bronze_dir(raw_root: Path) -> Path:
    """Folder of the per-day bronze Parquet files."""
    return raw_root / f"bronze-{BRONZE_VERSION}"


def download_days(days: list[date], raw_root: Path, workers: int = 4, fetch: Fetcher | None = None,
                  log: Callable[[str], None] = print) -> dict:
    """Download, verify and convert every day in ``days`` that is not cached yet; returns a manifest."""
    out = bronze_dir(raw_root)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    fetch = fetch or http_fetcher()
    todo = [d for d in days if not (out / f"{d.isoformat()}.parquet").exists()]
    directories: dict[str, tuple[str, dict[str, ZipMember]]] = {}
    for q in sorted({quarter_of(d) for d in todo}):
        url = zip_url(q)
        directories[q] = (url, list_members(url, fetch, remote_size(url)))
    pa.set_cpu_count(2)
    pa.set_io_thread_count(2)

    def one(d: date) -> tuple[str, dict]:
        q = quarter_of(d)
        url, members = directories[q]
        name = next((n for n in members if n.endswith(f"/{d.isoformat()}.csv") or n == f"{d.isoformat()}.csv"), None)
        if name is None:
            raise FileNotFoundError(f"{d} not in {url}")
        m = members[name]
        t0 = time.perf_counter()
        raw = read_member(url, m, fetch)
        t1 = time.perf_counter()
        table = parse_daily_csv(raw)
        tmp = out / f"{d.isoformat()}.parquet.tmp"
        pq.write_table(table, tmp, compression="zstd")
        tmp.replace(out / f"{d.isoformat()}.parquet")
        return d.isoformat(), {"archive": f"data_{q}.zip", "member": name, **{k: v for k, v in asdict(m).items()
                               if k in ("crc32", "compressed_size", "size")}, "rows": table.num_rows,
                               "download_s": round(t1 - t0, 2), "parse_s": round(time.perf_counter() - t1, 2)}

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for key, info in pool.map(one, todo):
            manifest[key] = info
            done += 1
            if done % 10 == 0 or done == len(todo):
                log(f"bronze: {done}/{len(todo)} days ({key}, {info['rows']:,} rows)")
                manifest_path.write_text(json.dumps(dict(sorted(manifest.items())), indent=1))
    manifest_path.write_text(json.dumps(dict(sorted(manifest.items())), indent=1))
    return {d.isoformat(): manifest[d.isoformat()] for d in days if d.isoformat() in manifest}
