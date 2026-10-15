"""PySpark ETL over the daily snapshots: cleaning, failure-rate aggregates, per-drive survival records, and the model
table with rolling SMART features and "fails within the next H days" labels.

Spark runs in local mode with two cores. Parquet output is written by pyarrow inside Spark tasks (``mapInArrow``)
rather than by Spark's Hadoop writer, which needs native Windows binaries (winutils) that this project does not
install; small aggregates are collected with Arrow.
"""

from __future__ import annotations

import os
import sys
import zlib
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from drivefail.features import CATEGORICAL, COUNTER_FEATURES, COUNTERS, CRITICAL, MANUFACTURERS, OTHER_FEATURES

_PREFIX = r"^(TOSHIBA|WDC|HGST|Seagate|SEAGATE|Hitachi|HITACHI)\s+"


@dataclass(frozen=True)
class EtlConfig:
    """Time windows, label horizon, drive groups and sampling of one ETL run."""

    start: date
    end: date
    horizon: int                  # label: the drive fails within the next `horizon` days (today included)
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    models: tuple[str, ...]       # drive models used for modelling and survival (normalised names)
    drive_rate: float = 0.15      # share of drives without a failure in the label window kept for modelling
    group_cuts: tuple[int, int] = (55, 70)  # crc32(serial) % 100: < 55 train (A), < 70 validation (B), else test (C)

    def __post_init__(self) -> None:
        if self.train_end + timedelta(days=self.horizon - 1) >= self.test_start:
            raise ValueError("test period must start after every training label is known")
        if self.test_end + timedelta(days=self.horizon - 1) > self.end:
            raise ValueError("test labels need data beyond the end date")


def spark_session(tmp_dir: Path, memory: str = "4g", shuffle_partitions: int = 64) -> SparkSession:
    """Local two-core Spark session (UI off, Arrow on, spill files under ``tmp_dir``)."""
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    spark = (SparkSession.builder.master("local[2]").appName("drivefail")
             .config("spark.driver.memory", memory)
             .config("spark.driver.extraJavaOptions",
                     "-XX:ParallelGCThreads=2 -XX:ConcGCThreads=1 -XX:CICompilerCount=2")
             .config("spark.driver.maxResultSize", "2g")
             .config("spark.ui.enabled", "false")
             .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
             .config("spark.sql.adaptive.enabled", "true")
             .config("spark.sql.session.timeZone", "UTC")
             .config("spark.sql.execution.arrow.pyspark.enabled", "true")
             .config("spark.local.dir", str(tmp_dir))
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def normalise_model(col: F.Column) -> F.Column:
    """Strip the manufacturer prefix Backblaze puts on some model names ('TOSHIBA MG08ACA16TA' -> 'MG08ACA16TA')."""
    return F.trim(F.regexp_replace(col, _PREFIX, ""))


def manufacturer_col(model: F.Column) -> F.Column:
    """Spark version of :func:`drivefail.features.manufacturer`."""
    m = F.upper(model)
    return (F.when(m.startswith("ST"), "Seagate").when(m.startswith("MG"), "Toshiba")
            .when(m.rlike("^(HUH|HMS|HUS|HDS)"), "HGST").when(m.rlike("^(WUH|WD)"), "WDC").otherwise("Other"))


def load_bronze(spark: SparkSession, files: list[Path], dedupe: bool = False) -> DataFrame:
    """Daily snapshots with normalised model names.

    Each bronze file holds one day, so (serial, date) duplicates can only occur within a file; the pipeline checks
    for them with pyarrow first and asks Spark to drop them (a full shuffle) only if any exist.
    """
    df = spark.read.parquet(*[str(f) for f in files]).withColumn("model", normalise_model(F.col("model")))
    return df.dropDuplicates(["serial_number", "date"]) if dedupe else df


def daily_model_counts(df: DataFrame) -> pd.DataFrame:
    """Drives reporting, failures and summed power-on hours per day and model (all models)."""
    return (df.groupBy("date", "model")
            .agg(F.count(F.lit(1)).alias("drives"), F.sum("failure").cast("long").alias("failures"),
                 F.sum("smart_9_raw").alias("hours_sum"), F.count("smart_9_raw").alias("hours_n"),
                 F.max("capacity_bytes").alias("capacity_bytes"))
            .toPandas())


def drive_table(df: DataFrame, models: tuple[str, ...]) -> pd.DataFrame:
    """One record per drive of the given models: first/last day seen, failure day, power-on hours at entry and exit."""
    return (df.where(F.col("model").isin(list(models))).groupBy("serial_number")
            .agg(F.min_by("model", "date").alias("model"), F.min("date").alias("first_date"),
                 F.max("date").alias("last_date"), F.count(F.lit(1)).alias("n_days"),
                 F.max("failure").cast("int").alias("failed"),
                 F.min(F.when(F.col("failure") == 1, F.col("date"))).alias("fail_date"),
                 F.min_by("smart_9_raw", "date").alias("hours_first"),
                 F.max_by("smart_9_raw", "date").alias("hours_last"),
                 F.min_by("datacenter", "date").alias("datacenter"),
                 F.max("capacity_bytes").alias("capacity_bytes"))
            .toPandas())


def drive_sampling(drives: pd.DataFrame, cfg: EtlConfig) -> pd.DataFrame:
    """Drive group (A train / B validation / C test), sampling flag and weight for every drive.

    Every drive that fails inside the label window is kept; other drives are kept with probability
    ``cfg.drive_rate`` (a hash of the serial number) and weighted by its inverse, so weighted counts estimate the
    whole fleet.
    """
    d = drives[["serial_number", "first_date", "last_date", "fail_date"]].copy()
    start = pd.Timestamp(cfg.start)
    d["fail_day"] = (pd.to_datetime(d.fail_date) - start).dt.days
    d["last_day"] = (pd.to_datetime(d.last_date) - start).dt.days
    lo, hi = (cfg.train_start - cfg.start).days, (cfg.test_end - cfg.start).days + cfg.horizon - 1
    positive = d.fail_day.between(lo, hi)
    g = d.serial_number.map(lambda s: zlib.crc32(s.encode()) % 100)
    keep_hash = d.serial_number.map(lambda s: zlib.crc32(f"{s}#keep".encode()) % 10000)
    a, b = cfg.group_cuts
    d["group"] = np.where(g < a, "A", np.where(g < b, "B", "C"))
    d["fails_in_window"] = positive
    d["keep"] = positive | (keep_hash < round(cfg.drive_rate * 10000))
    d["weight"] = np.where(positive, 1.0, 1.0 / cfg.drive_rate)
    return d[["serial_number", "group", "fail_day", "last_day", "fails_in_window", "keep", "weight"]]


def feature_table(df: DataFrame, sampled: DataFrame, cfg: EtlConfig) -> DataFrame:
    """Rolling SMART features and labels for every drive-day of the sampled drives in the train and test periods.

    ``sampled`` holds the kept drives (serial_number, group, fail_day, last_day, weight) from
    :func:`drive_sampling`; it is broadcast-joined so only those drives are shuffled and sorted. Changes are taken
    against the report 7 or 30 rows earlier (drives report daily); temperature and increase-count windows use
    calendar days (``rangeBetween``). The label of drive-day t is 1 when the drive's first recorded failure falls
    in [t, t + H - 1]; rows whose label window runs past the drive's last report without a failure are flagged
    ``censored``.
    """
    models = sorted(cfg.models)
    model_code = F.lit(None).cast("int")
    for i, m in enumerate(models):
        model_code = F.when(F.col("model") == m, F.lit(i)).otherwise(model_code)
    mfg_code = F.lit(None).cast("int")
    for i, m in enumerate(MANUFACTURERS):
        mfg_code = F.when(F.col("mfg") == m, F.lit(i)).otherwise(mfg_code)

    label_end = (cfg.test_end - cfg.start).days
    base = (df.where(F.col("model").isin(models))
            .select("serial_number", "date", "model", "capacity_bytes",
                    *[F.col(f"smart_{c}_raw").alias(f"smart_{c}") for c in
                      sorted(set(COUNTERS) | {9, 12, 192, 193, 194})])
            .withColumn("day", F.datediff(F.col("date"), F.lit(cfg.start)))
            .where(F.col("day") <= label_end)  # later days only matter through fail_day / last_day
            .join(F.broadcast(sampled.select("serial_number", "group", "fail_day", "last_day", "weight")),
                  "serial_number", "inner")
            .withColumn("mfg", manufacturer_col(F.col("model"))))

    ordered = Window.partitionBy("serial_number").orderBy("day")
    growing = ordered.rowsBetween(Window.unboundedPreceding, Window.currentRow)
    w7, w30 = ordered.rangeBetween(-6, 0), ordered.rangeBetween(-29, 0)
    crit = sum(F.coalesce(F.col(f"smart_{c}"), F.lit(0.0)) for c in CRITICAL)

    def change(x: F.Column, k: int) -> F.Column:
        """Change since the report k rows earlier, or since the drive's first report if it has fewer."""
        return x - F.coalesce(F.lag(x, k).over(ordered), F.first(x, ignorenulls=True).over(growing))

    cols = [F.col(c) for c in ("serial_number", "date", "day", "model", "group", "fail_day", "last_day", "weight")]
    for c in COUNTERS:
        x = F.col(f"smart_{c}")
        cols += [x, change(x, 7).alias(f"smart_{c}_d7"), change(x, 30).alias(f"smart_{c}_d30")]
    cols += [
        crit.alias("crit_sum"),
        (crit - F.lag(crit).over(ordered)).alias("crit_d1"),
        (F.col("smart_9") / 24).alias("age_days"),
        F.col("smart_12"), change(F.col("smart_12"), 30).alias("smart_12_d30"),
        F.col("smart_192"), F.col("smart_193"), change(F.col("smart_193"), 7).alias("smart_193_d7"),
        F.col("smart_194").alias("temp"), F.max("smart_194").over(w30).alias("temp_max_30"),
        F.avg("smart_194").over(w7).alias("temp_mean_7"),
        (F.col("capacity_bytes") / 1e12).alias("capacity_tb"),
        model_code.alias("model_code"), mfg_code.alias("mfg_code"),
    ]
    feat = base.select(*cols)
    feat = feat.withColumn("crit_inc_days_30", F.sum(F.when(F.col("crit_d1") > 0, 1).otherwise(0)).over(w30))

    h = cfg.horizon
    t0, t1 = (cfg.train_start - cfg.start).days, (cfg.train_end - cfg.start).days
    s0, s1 = (cfg.test_start - cfg.start).days, label_end
    feat = (feat.withColumn("period", F.when(F.col("day").between(t0, t1), "train")
                            .when(F.col("day").between(s0, s1), "test"))
            .where(F.col("period").isNotNull())
            .where(F.col("fail_day").isNull() | (F.col("day") <= F.col("fail_day")))
            .withColumn("label", (F.col("fail_day").isNotNull() & (F.col("fail_day") <= F.col("day") + h - 1)).cast("int"))
            .withColumn("censored", F.col("fail_day").isNull() & (F.col("last_day") < F.col("day") + h - 1))
            .withColumn("days_to_failure", (F.col("fail_day") - F.col("day")).cast("int")))
    feature_cols = COUNTER_FEATURES + OTHER_FEATURES
    return feat.select(
        "serial_number", "date", "model", "group", "period", "label", "censored", "weight", "days_to_failure",
        *[F.col(c).cast("float").alias(c) for c in feature_cols], *[F.col(c).cast("int").alias(c) for c in CATEGORICAL])


def write_parquet_via_arrow(df: DataFrame, out_dir: Path) -> int:
    """Write ``df`` as Parquet files (one per Spark partition) using pyarrow inside the tasks; returns the row count."""
    out_dir.mkdir(parents=True, exist_ok=True)
    target = str(out_dir)

    def write(batches):  # runs in the Spark Python workers
        import uuid

        import pyarrow as pa_
        import pyarrow.parquet as pq_

        path, writer, n = os.path.join(target, f"part-{uuid.uuid4().hex}.parquet"), None, 0
        for b in batches:
            if b.num_rows == 0:
                continue
            t = pa_.Table.from_batches([b])
            writer = writer or pq_.ParquetWriter(path, t.schema, compression="zstd")
            writer.write_table(t)
            n += b.num_rows
        if writer is not None:
            writer.close()
        yield pa_.RecordBatch.from_pydict({"rows": [n]})

    return int(sum(r["rows"] for r in df.mapInArrow(write, "rows long").collect()))


def read_parquet_dir(path: Path, columns: list[str] | None = None, filters: list | None = None) -> pd.DataFrame:
    """Load a folder written by :func:`write_parquet_via_arrow` into pandas."""
    files = sorted(path.glob("part-*.parquet"))
    tables = [pq.read_table(f, columns=columns, filters=filters) for f in files]
    tables = [t for t in tables if t.num_rows]
    return pa.concat_tables(tables).to_pandas() if tables else pd.DataFrame(columns=columns)
