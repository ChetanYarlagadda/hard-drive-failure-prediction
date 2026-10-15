"""Range-request zip reading and daily CSV parsing, against a zip archive built in memory."""

import io
import zipfile
from datetime import date

import pytest

from drivefail.download import BRONZE_COLUMNS, list_members, parse_daily_csv, quarter_of, read_member

CSV = (b"date,serial_number,model,capacity_bytes,failure,datacenter,smart_5_raw,smart_9_raw,smart_187_raw\n"
       b"2025-10-01,S1,ST16000NM001G,16000900661248,0,phx1,0,1000,\n"
       b"2025-10-01,S2,TOSHIBA MG08ACA16TA,16000900661248,1,sac0,8,2000,3\n")


def _archive() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("data_Q4_2025/2025-10-01.csv", CSV, compress_type=zipfile.ZIP_DEFLATED)
        z.writestr("data_Q4_2025/2025-10-02.csv", CSV.replace(b"2025-10-01", b"2025-10-02"),
                   compress_type=zipfile.ZIP_STORED)
    return buf.getvalue()


def _fetcher(blob: bytes, log: list):
    def fetch(url: str, first: int, last: int) -> bytes:
        log.append((first, last))
        return blob[first:last + 1]
    return fetch


def test_members_round_trip_with_range_requests() -> None:
    blob, calls = _archive(), []
    fetch = _fetcher(blob, calls)
    members = list_members("mem://zip", fetch, len(blob))
    assert set(members) == {"data_Q4_2025/2025-10-01.csv", "data_Q4_2025/2025-10-02.csv"}
    assert read_member("mem://zip", members["data_Q4_2025/2025-10-01.csv"], fetch) == CSV  # deflated member
    assert read_member("mem://zip", members["data_Q4_2025/2025-10-02.csv"], fetch).startswith(b"date,")  # stored
    assert all(last - first + 1 <= len(blob) for first, last in calls)


def test_corrupted_member_fails_crc_check() -> None:
    blob = bytearray(_archive())
    members = list_members("mem://zip", _fetcher(bytes(blob), []), len(blob))
    m = members["data_Q4_2025/2025-10-02.csv"]  # stored, so a flipped byte survives decompression
    pos = blob.find(b"2025-10-02,S2")
    blob[pos + 11] ^= 0x01
    with pytest.raises(IOError, match="CRC"):
        read_member("mem://zip", m, _fetcher(bytes(blob), []))


def test_parse_keeps_schema_and_adds_missing_columns() -> None:
    t = parse_daily_csv(CSV)
    assert t.column_names == list(BRONZE_COLUMNS)
    assert t.num_rows == 2
    d = t.to_pydict()
    assert d["failure"] == [0, 1] and d["smart_187_raw"] == [None, 3.0]
    assert d["smart_197_raw"] == [None, None]  # absent from the CSV header
    assert d["date"][0] == date(2025, 10, 1)


def test_quarter_of() -> None:
    assert quarter_of(date(2025, 12, 31)) == "Q4_2025"
    assert quarter_of(date(2026, 1, 1)) == "Q1_2026"
