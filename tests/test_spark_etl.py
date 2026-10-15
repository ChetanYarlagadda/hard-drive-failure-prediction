"""The Spark ETL on a tiny synthetic fleet: drive records, labels, censoring, rolling features and the Arrow writer."""

import shutil
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from drivefail.download import BRONZE_COLUMNS

pytest.importorskip("pyspark")
if shutil.which("java") is None:
    pytest.skip("Spark needs a Java runtime", allow_module_level=True)

from drivefail.spark_etl import (  # noqa: E402
    EtlConfig, drive_sampling, drive_table, feature_table, load_bronze, read_parquet_dir, spark_session,
    write_parquet_via_arrow,
)

D0 = date(2025, 10, 1)
MODEL = "ST16000NM001G"
CFG = EtlConfig(start=D0, end=D0 + timedelta(days=44), horizon=7, train_start=D0 + timedelta(days=10),
                train_end=D0 + timedelta(days=20), test_start=D0 + timedelta(days=28),
                test_end=D0 + timedelta(days=38), models=(MODEL,), drive_rate=1.0)


def _day(f: pd.DataFrame) -> pd.Series:
    return (pd.to_datetime(f.date) - pd.Timestamp(D0)).dt.days


def _bronze(tmp_path) -> list:
    """F fails on day 40 with reallocated sectors growing from day 31; H is healthy; L leaves on day 35."""
    files = []
    for d in range(45):
        rows = [("F", d <= 40, int(d == 40), max(0, d - 30)), ("H", True, 0, 2), ("L", d <= 35, 0, 0)]
        rows = [(s, f, x) for s, present, f, x in rows if present]
        n = len(rows)
        cols = {c: [None] * n for c in BRONZE_COLUMNS}
        cols.update({"date": [D0 + timedelta(days=d)] * n, "serial_number": [r[0] for r in rows],
                     "model": [f"Seagate {MODEL}"] * n, "capacity_bytes": [16 * 10**12] * n,
                     "failure": [r[1] for r in rows], "datacenter": ["dc1"] * n,
                     "smart_5_raw": [float(r[2]) for r in rows], "smart_9_raw": [24.0 * (1000 + d)] * n,
                     "smart_194_raw": [30.0] * n, "smart_197_raw": [0.0] * n, "smart_198_raw": [0.0] * n})
        schema = pa.schema([("date", pa.date32()), ("serial_number", pa.string()), ("model", pa.string()),
                            ("capacity_bytes", pa.int64()), ("failure", pa.int8()), ("datacenter", pa.string())]
                           + [(c, pa.float64()) for c in BRONZE_COLUMNS if c.startswith("smart_")])
        f = tmp_path / f"{d:02d}.parquet"
        pq.write_table(pa.table(cols, schema=schema), f)
        files.append(f)
    return files


@pytest.fixture(scope="module")
def etl(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("etl")
    spark = spark_session(tmp / "spark", memory="1g", shuffle_partitions=2)
    df = load_bronze(spark, _bronze(tmp))
    drives = drive_table(df, CFG.models)
    sample = drive_sampling(drives, CFG)
    keep = sample[sample.keep][["serial_number", "group", "fail_day", "last_day", "weight"]].copy()
    keep["fail_day"] = keep.fail_day.astype("Int32")
    keep["last_day"] = keep.last_day.astype("int32")
    n = write_parquet_via_arrow(feature_table(df, spark.createDataFrame(pa.Table.from_pandas(keep, preserve_index=False)),
                                              CFG), tmp / "features")
    feats = read_parquet_dir(tmp / "features")
    spark.stop()
    return drives.set_index("serial_number"), sample.set_index("serial_number"), feats, n


def test_drive_records(etl) -> None:
    drives, sample, _, _ = etl
    assert drives.loc["F", "failed"] == 1 and pd.Timestamp(drives.loc["F", "fail_date"]) == pd.Timestamp(D0 + timedelta(40))
    assert pd.Timestamp(drives.loc["L", "last_date"]) == pd.Timestamp(D0 + timedelta(35)) and drives.loc["L", "failed"] == 0
    assert drives.loc["H", "model"] == MODEL  # manufacturer prefix stripped
    assert sample.loc["F", "fails_in_window"] and not sample.loc["H", "fails_in_window"]


def test_labels_and_censoring(etl) -> None:
    _, _, f, n = etl
    assert n == len(f)
    day = _day(f).to_numpy()
    assert set(np.unique(day)) == set(range(10, 21)) | set(range(28, 39))  # train and test periods only
    fail = f[f.serial_number == "F"].assign(day=day[f.serial_number == "F"]).set_index("day")
    assert fail.label[fail.index >= 34].eq(1).all() and fail.label[fail.index < 34].eq(0).all()  # 40 - 7 + 1 = 34
    assert (fail.loc[38, "days_to_failure"]) == 2
    leave = f[f.serial_number == "L"].assign(day=day[f.serial_number == "L"]).set_index("day")
    assert leave.censored[leave.index >= 30].all() and not leave.censored[leave.index < 30].any()  # 35 - 7 + 1 = 29
    assert f[f.serial_number == "H"].label.eq(0).all() and not f[f.serial_number == "H"].censored.any()


def test_rolling_features(etl) -> None:
    _, _, f, _ = etl
    day = _day(f)
    r = f[(f.serial_number == "F") & (day == 38)].iloc[0]
    assert r.smart_5 == 8 and r.smart_5_d7 == 7 and r.smart_5_d30 == 8
    assert r.crit_inc_days_30 == 8  # increases on days 31..38
    assert r.temp_mean_7 == 30 and np.isclose(r.age_days, 1038)
    h = f[(f.serial_number == "H") & (day == 15)].iloc[0]
    assert h.smart_5 == 2 and h.smart_5_d7 == 0 and h.crit_sum == 2 and h.crit_inc_days_30 == 0
    assert set(f.group) <= {"A", "B", "C"} and f.weight.eq(1.0).all()
