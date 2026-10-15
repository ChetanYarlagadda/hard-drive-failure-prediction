"""Feature definitions shared by the Spark ETL, the models and the figures."""

from __future__ import annotations

# SMART error counters: raw value, 7-day increase and 30-day increase are used for each.
COUNTERS: dict[int, str] = {
    5: "reallocated sectors",
    187: "reported uncorrectable errors",
    188: "command timeouts",
    197: "current pending sectors",
    198: "offline uncorrectable sectors",
    199: "UDMA CRC errors",
    184: "end-to-end errors",
    10: "spin retries",
    189: "high-fly writes",
}
# The five counters Backblaze highlights as failure indicators (used by the rule baseline).
CRITICAL = (5, 187, 188, 197, 198)

COUNTER_FEATURES = [f for c in COUNTERS for f in (f"smart_{c}", f"smart_{c}_d7", f"smart_{c}_d30")]
OTHER_FEATURES = [
    "crit_sum",            # sum of the five critical raw counters
    "crit_inc_days_30",    # days in the last 30 on which the critical sum increased
    "age_days",            # power-on hours (SMART 9) / 24
    "smart_12",            # power cycles
    "smart_12_d30",
    "smart_192",           # power-off retracts
    "smart_193",           # load/unload cycles
    "smart_193_d7",
    "temp",                # SMART 194, degrees C
    "temp_max_30",
    "temp_mean_7",
    "capacity_tb",
]
CATEGORICAL = ["model_code", "mfg_code"]
FEATURES = COUNTER_FEATURES + OTHER_FEATURES + CATEGORICAL

MANUFACTURERS = ("HGST", "Seagate", "Toshiba", "WDC")


def manufacturer(model: str) -> str:
    """Manufacturer of a normalised Backblaze model name (prefix rules)."""
    m = model.upper()
    if m.startswith("ST"):
        return "Seagate"
    if m.startswith("MG"):
        return "Toshiba"
    if m.startswith(("HUH", "HMS", "HUS", "HDS")):
        return "HGST"
    if m.startswith(("WUH", "WD")):
        return "WDC"
    return "Other"


def label(feature: str) -> str:
    """Readable name of a feature for figures and tables."""
    named = {"crit_sum": "Sum of critical counters (5,187,188,197,198)",
             "crit_inc_days_30": "Days with a critical-counter increase (30 d)",
             "age_days": "Drive age (power-on days)", "smart_12": "Power cycles", "smart_12_d30": "Power cycles, 30-d increase",
             "smart_192": "Power-off retracts", "smart_193": "Load/unload cycles",
             "smart_193_d7": "Load/unload cycles, 7-d increase", "temp": "Temperature (C)",
             "temp_max_30": "Max temperature, 30 d (C)", "temp_mean_7": "Mean temperature, 7 d (C)",
             "capacity_tb": "Capacity (TB)", "model_code": "Drive model", "mfg_code": "Manufacturer"}
    if feature in named:
        return named[feature]
    if feature.startswith("smart_"):
        parts = feature.split("_")
        c = int(parts[1])
        suffix = {"d7": ", 7-d increase", "d30": ", 30-d increase"}.get(parts[2], "") if len(parts) > 2 else ""
        return f"SMART {c} {COUNTERS.get(c, '')}{suffix}".replace("  ", " ")
    return feature
