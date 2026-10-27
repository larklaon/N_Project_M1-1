"""Independently check analysis outputs using Python's standard library.

Run after analysis.py: python scripts/verify_results.py
"""

from __future__ import annotations

from collections import defaultdict
import csv
from datetime import date, timedelta
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import struct
from urllib.parse import unquote, urlsplit

from download_data import DAY_SHA256


ROOT = Path(__file__).resolve().parents[1]


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def close(actual: float, expected: float) -> None:
    check(math.isclose(float(actual), expected, rel_tol=1e-9, abs_tol=1e-6),
          f"Mismatch: {actual} != {expected}")


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    raw_path = ROOT / "data/raw/day.csv"
    check(hashlib.sha256(raw_path.read_bytes()).hexdigest() == DAY_SHA256, "Raw checksum mismatch.")
    raw = rows(raw_path)
    dates = [date.fromisoformat(row["dteday"]) for row in raw]
    check(dates == [date(2011, 1, 1) + timedelta(days=i) for i in range(731)], "Calendar mismatch.")
    values = [int(row["cnt"]) for row in raw]
    summary = json.loads((ROOT / "data/processed/summary.json").read_text(encoding="utf-8"))
    check(summary["count"] == len(raw), "Row count mismatch.")
    close(summary["total"], sum(values))
    close(summary["mean"], statistics.mean(values))
    close(summary["median"], statistics.median(values))
    for key, function in (("min", min), ("max", max)):
        value = function(values)
        check(summary[key] == {"date": dates[values.index(value)].isoformat(), "count": value},
              f"{key} mismatch.")

    grouping = {
        "annual": (lambda d, r: (d.year,), ["year"]),
        "monthly": (lambda d, r: (d.year, d.month), ["year", "month"]),
        "weather": (lambda d, r: (d.year, int(r["weathersit"])), ["year", "weathersit"]),
        "weather_all": (lambda d, r: (int(r["weathersit"]),), ["weathersit"]),
    }
    for name, (key_function, keys) in grouping.items():
        groups = defaultdict(list)
        for day, row, value in zip(dates, raw, values):
            groups[key_function(day, row)].append(value)
        exported = rows(ROOT / f"data/processed/{name}.csv")
        check(len(exported) == len(groups), f"{name}: group count mismatch.")
        for row in exported:
            group = groups[tuple(int(row[key]) for key in keys)]
            close(row["days"], len(group))
            close(row["daily_mean"], statistics.mean(group))
            if "total" in row:
                close(row["total"], sum(group))

    means = {}
    for is_weekend in (False, True):
        members = [r for d, r in zip(dates, raw) if (d.weekday() >= 5) == is_weekend]
        means[is_weekend] = {c: statistics.mean(int(r[c]) for r in members)
                             for c in ("cnt", "casual", "registered")}
        exported = next(r for r in rows(ROOT / "data/processed/day_type.csv")
                        if r["is_weekend"] == str(is_weekend))
        close(exported["days"], len(members))
        for original, result in (("cnt", "total_mean"), ("casual", "casual_mean"),
                                 ("registered", "registered_mean")):
            close(exported[result], means[is_weekend][original])

    daily = rows(ROOT / "data/processed/daily_analysis.csv")
    check(len(daily) == len(raw), "Processed row count mismatch.")
    for source, result in zip(raw, daily):
        check(result["dteday"] == source["dteday"], "An original date was modified.")
        for column in source.keys() - {"dteday"}:
            check(math.isclose(float(result[column]), float(source[column]),
                               rel_tol=0, abs_tol=1e-14),
                  f"An original numeric value was modified: {column}.")
    # Recalculate boundary and event-day windows without pandas or the analysis implementation.
    for index in (0, 6, 29, 30, 667, 668, 730):
        for window in (7, 30):
            actual = daily[index][f"ma{window}"]
            if index < window - 1:
                check(actual == "", "Incomplete moving window should be missing.")
            else:
                close(actual, statistics.mean(values[index - window + 1:index + 1]))
        if index:
            close(daily[index]["daily_change_pct"], (values[index] / values[index - 1] - 1) * 100)
        if index >= 30:
            previous = values[index - 30:index]
            close(daily[index]["rolling_z"],
                  (values[index] - statistics.mean(previous)) / statistics.stdev(previous))
    local_dates = []
    for index in range(30, len(values)):
        previous = values[index - 30:index]
        z = (values[index] - statistics.mean(previous)) / statistics.stdev(previous)
        if abs(z) > 3:
            local_dates.append(dates[index].isoformat())
    check(local_dates == [r["date"] for r in rows(ROOT / "data/processed/flagged_days.csv")],
          "Local anomaly candidates mismatch.")
    check(len(local_dates) == summary["quality"]["local_flag_count"], "Flag count mismatch.")

    report = (ROOT / "README.md").read_text(encoding="utf-8")
    annual = {year: [v for d, v in zip(dates, values) if d.year == year] for year in (2011, 2012)}
    growth = (statistics.mean(annual[2012]) / statistics.mean(annual[2011]) - 1) * 100
    close(summary["annual_mean_change_pct"], growth)
    tokens = [f"{sum(values):,}", f"{statistics.mean(values):,.1f}", f"{growth:.1f}%",
              f"{means[True]['casual'] / means[False]['casual']:.2f}배",
              f"{min(values):,}건", f"{max(values):,}건"]
    for token in tokens:
        check(token in report, f"Core result absent from report: {token}")
    links = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", report)
    for link in links:
        if not urlsplit(link).scheme and not link.startswith("#"):
            check((ROOT / unquote(link.split("#")[0])).is_file(), f"Broken local link: {link}")
    images = sorted((ROOT / "images").glob("*.png"))
    check(len(images) == 3, "Expected three PNG figures.")
    for path in images:
        data = path.read_bytes()
        check(data[:8] == b"\x89PNG\r\n\x1a\n", "Invalid PNG signature.")
        width, height = struct.unpack(">II", data[16:24])
        check(width >= 1200 and height >= 600, "Figure resolution is too low.")
    print("PASS: source integrity; calendar; original-value preservation; independent statistics;")
    print("      sample moving windows and changes; anomaly dates; core report numbers;")
    print(f"      {len(links)} report links checked ({sum(not urlsplit(x).scheme for x in links)} local); 3 PNG figures.")


if __name__ == "__main__":
    main()
