"""Validate and analyze daily bike rentals; export reproducible tables and figures.

Run: python analysis.py
The original data is preserved. All paths are relative to this script's location.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.download_data import DAY_SHA256


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "data" / "processed"
IMAGES = ROOT / "images"
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
WEATHER = {1: "Clear / partly cloudy", 2: "Mist / cloudy", 3: "Light rain / snow"}
COLORS = {2011: "#D4813D", 2012: "#087E8B"}
REQUIRED = [
    "instant", "dteday", "season", "yr", "mnth", "holiday", "weekday",
    "workingday", "weathersit", "temp", "atemp", "hum", "windspeed",
    "casual", "registered", "cnt",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_and_validate() -> tuple[pd.DataFrame, dict]:
    path = ROOT / "data" / "raw" / "day.csv"
    if not path.exists():
        raise FileNotFoundError("Run python scripts/download_data.py first.")
    require(hashlib.sha256(path.read_bytes()).hexdigest() == DAY_SHA256,
            "Raw data checksum mismatch; preserve the source file and review modifications.")
    df = pd.read_csv(path)
    require(list(df.columns) == REQUIRED, "Unexpected source schema.")
    missing = {column: int(count) for column, count in df.isna().sum().items()}
    require(sum(missing.values()) == 0, "Missing values detected; review before analysis.")
    duplicate_rows = int(df.duplicated().sum())
    df["date"] = pd.to_datetime(df["dteday"], format="%Y-%m-%d", errors="raise")
    duplicate_dates = int(df["date"].duplicated().sum())
    require(duplicate_rows == 0 and duplicate_dates == 0, "Duplicate rows or dates detected.")
    df = df.sort_values("date").reset_index(drop=True)
    expected = pd.date_range("2011-01-01", "2012-12-31", freq="D")
    require(len(df) == 731 and pd.DatetimeIndex(df["date"]).equals(expected),
            "Expected 731 consecutive daily observations across 2011–2012.")
    numeric = df[REQUIRED].drop(columns="dteday")
    require(bool(np.isfinite(numeric.to_numpy()).all()), "Non-finite numeric value detected.")
    for column in ("cnt", "casual", "registered"):
        require(bool(((df[column] >= 0) & (df[column] % 1 == 0)).all()),
                f"{column} must contain nonnegative integers.")
    require(bool((df["cnt"] == df["casual"] + df["registered"]).all()),
            "Rental count does not equal casual + registered.")
    for column in ("temp", "atemp", "hum", "windspeed"):
        require(bool(df[column].between(0, 1).all()), f"{column} is outside its normalized range.")
    for column in ("yr", "holiday", "workingday"):
        require(bool(df[column].isin([0, 1]).all()), f"Unexpected {column} code.")
    require(bool(df["season"].isin([1, 2, 3, 4]).all()), "Unexpected season code.")
    require(bool(df["weathersit"].isin([1, 2, 3, 4]).all()), "Unexpected weather code.")
    require(bool((df["yr"] + 2011 == df["date"].dt.year).all()), "Year code mismatch.")
    require(bool((df["mnth"] == df["date"].dt.month).all()), "Month code mismatch.")
    require(bool((df["weekday"] == (df["date"].dt.dayofweek + 1) % 7).all()),
            "Weekday code mismatch.")
    expected_work = (df["date"].dt.dayofweek < 5) & (df["holiday"] == 0)
    require(bool((df["workingday"] == expected_work.astype(int)).all()), "Working-day mismatch.")

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["weekday_index"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["weekday_index"] >= 5  # Holidays on weekdays remain weekdays.
    df["ma7"] = df["cnt"].rolling(7, min_periods=7).mean()
    df["ma30"] = df["cnt"].rolling(30, min_periods=30).mean()
    df["daily_change_pct"] = df["cnt"].pct_change(fill_method=None) * 100

    # A whole-period IQR flags candidates, not errors. Genuine low/high demand stays in the data.
    q1, q3 = df["cnt"].quantile([0.25, 0.75])
    lower, upper = float(q1 - 1.5 * (q3 - q1)), float(q3 + 1.5 * (q3 - q1))
    df["iqr_flag"] = ~df["cnt"].between(lower, upper)
    # Compare with the previous 30 days only; the current day is excluded from its own baseline.
    baseline = df["cnt"].shift(1).rolling(30, min_periods=30)
    df["previous30_mean"] = baseline.mean()
    df["previous30_std"] = baseline.std(ddof=1)
    df["rolling_z"] = (
        (df["cnt"] - df["previous30_mean"]) / df["previous30_std"].replace(0, np.nan)
    )
    df["local_flag"] = df["rolling_z"].abs() > 3
    quality = {
        "rows": len(df), "columns_raw": len(REQUIRED), "missing_raw": missing,
        "duplicate_rows": duplicate_rows, "duplicate_dates": duplicate_dates,
        "missing_dates": 0, "invalid_counts": 0, "count_identity_failures": 0,
        "removed_rows": 0, "imputed_values": 0,
        "iqr_lower": lower, "iqr_upper": upper,
        "iqr_flag_count": int(df["iqr_flag"].sum()),
        "local_flag_count": int(df["local_flag"].sum()),
        "derived_missing_expected": {
            "ma7": 6, "ma30": 29, "daily_change_pct": 1,
            "previous30_mean": 30, "previous30_std": 30, "rolling_z": 30,
        },
    }
    for column, count in quality["derived_missing_expected"].items():
        require(int(df[column].isna().sum()) == count, f"Unexpected missing values in {column}.")
    return df, quality


def export_tables(df: pd.DataFrame, quality: dict) -> dict:
    annual = df.groupby("year").agg(days=("cnt", "size"), total=("cnt", "sum"),
                                   daily_mean=("cnt", "mean"), daily_std=("cnt", "std"))
    monthly = df.groupby(["year", "month"]).agg(
        days=("cnt", "size"), total=("cnt", "sum"), daily_mean=("cnt", "mean"))
    month_comparison = monthly["daily_mean"].unstack("year")
    month_comparison["yoy_pct"] = (month_comparison[2012] / month_comparison[2011] - 1) * 100
    weekday = df.groupby("weekday_index")[["cnt", "casual", "registered"]].mean()
    day_type = df.groupby("is_weekend").agg(
        days=("cnt", "size"), total_mean=("cnt", "mean"),
        casual_mean=("casual", "mean"), registered_mean=("registered", "mean"))
    weather = df.groupby(["year", "weathersit"]).agg(
        days=("cnt", "size"), daily_mean=("cnt", "mean"), median=("cnt", "median"))
    weather_all = df.groupby("weathersit").agg(days=("cnt", "size"), daily_mean=("cnt", "mean"))
    flagged = df.loc[df["local_flag"], ["date", "cnt", "previous30_mean", "rolling_z"]]
    for name, table in {
        "daily_analysis": df, "annual": annual, "monthly": monthly,
        "month_comparison": month_comparison, "weekday": weekday,
        "day_type": day_type, "weather": weather, "weather_all": weather_all,
        "flagged_days": flagged,
    }.items():
        table.to_csv(OUTPUT / f"{name}.csv", index=name != "daily_analysis" and name != "flagged_days",
                     date_format="%Y-%m-%d", float_format=None if name == "daily_analysis" else "%.6f")

    def extreme(column: str, method: str) -> dict:
        index = getattr(df[column], method)()
        row = df.loc[index]
        result = {"date": row["date"].strftime("%Y-%m-%d"), "count": int(row["cnt"])}
        if column != "cnt":
            result["change_pct"] = float(row[column])
        return result

    stats = {
        "period": [df["date"].min().strftime("%Y-%m-%d"), df["date"].max().strftime("%Y-%m-%d")],
        "count": len(df), "total": int(df["cnt"].sum()), "mean": float(df["cnt"].mean()),
        "median": float(df["cnt"].median()), "min": extreme("cnt", "idxmin"),
        "max": extreme("cnt", "idxmax"),
        "annual": annual.reset_index().to_dict(orient="records"),
        "annual_total_change_pct": float((annual.loc[2012, "total"] / annual.loc[2011, "total"] - 1) * 100),
        "annual_mean_change_pct": float((annual.loc[2012, "daily_mean"] / annual.loc[2011, "daily_mean"] - 1) * 100),
        "months_with_yoy_growth": int((month_comparison["yoy_pct"] > 0).sum()),
        "day_type": day_type.reset_index().to_dict(orient="records"),
        "weather": weather.reset_index().to_dict(orient="records"),
        "largest_drop": extreme("daily_change_pct", "idxmin"),
        "largest_rise": extreme("daily_change_pct", "idxmax"),
        "quality": quality,
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return stats


def style() -> None:
    # English plot labels keep figures portable without an OS-specific Korean font dependency.
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 13,
        "axes.labelsize": 10, "figure.facecolor": "#FFFFFF", "axes.facecolor": "#FFFFFF",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#B8C0C8", "text.color": "#243442", "axes.labelcolor": "#243442",
        "xtick.color": "#536473", "ytick.color": "#536473", "savefig.dpi": 180,
        "axes.formatter.useoffset": False,
    })


def save(fig: plt.Figure, filename: str) -> None:
    fig.savefig(IMAGES / filename, bbox_inches="tight", metadata={"Software": "Matplotlib"})
    plt.close(fig)


def plot_trend(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.8), layout="constrained")
    ax.plot(df["date"], df["cnt"], color="#BEC8CE", linewidth=0.8, label="Daily rentals")
    ax.plot(df["date"], df["ma7"], color="#55A9AC", linewidth=1.3, label="Trailing 7-day mean")
    ax.plot(df["date"], df["ma30"], color="#135767", linewidth=2.3, label="Trailing 30-day mean")
    row = df.loc[df["cnt"].idxmin()]
    ax.scatter([row["date"]], [row["cnt"]], s=32, color="#BF5738", zorder=5)
    ax.annotate(f"Minimum: {int(row['cnt']):,}\n{row['date']:%Y-%m-%d}",
                (row["date"], row["cnt"]), xytext=(-100, 40), textcoords="offset points",
                fontsize=9, arrowprops={"arrowstyle": "-", "color": "#BF5738"})
    ax.set(title="Daily bike rentals and moving averages | 2011–2012", ylabel="Rentals / day", xlabel="Date")
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(axis="y", alpha=0.16)
    ax.legend(loc="upper left", ncol=3, frameon=False, fontsize=9)
    save(fig, "01_rental_trend.png")


def plot_calendar(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    monthly = df.groupby(["year", "month"])["cnt"].mean().unstack("year")
    for year in (2011, 2012):
        axes[0].plot(monthly.index, monthly[year], marker="o", markersize=4,
                     color=COLORS[year], label=str(year), linewidth=2)
    axes[0].set(title="A. Same-month comparison", xlabel="Month", ylabel="Mean rentals / day",
                xticks=list(range(1, 13)), ylim=(0, 8500))
    axes[0].legend(frameon=False)
    weekday = df.groupby("weekday_index")[["registered", "casual"]].mean()
    x = np.arange(7)
    axes[1].bar(x, weekday["registered"], label="Registered", color="#087E8B", width=0.64)
    axes[1].bar(x, weekday["casual"], bottom=weekday["registered"], label="Casual", color="#D4813D", width=0.64)
    axes[1].set(title="B. Weekday means by user type", xlabel="Day of week", ylabel="Mean rentals / day",
                xticks=x, xticklabels=WEEKDAYS, ylim=(0, 6500))
    axes[1].legend(frameon=False, ncol=2)
    for ax in axes:
        ax.grid(axis="y", alpha=0.16)
        ax.set_axisbelow(True)
    save(fig, "02_calendar_patterns.png")


def plot_weather(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
    codes = sorted(df["weathersit"].unique())
    x = np.arange(len(codes))
    for offset, year in ((-0.19, 2011), (0.19, 2012)):
        group = df[df["year"] == year].groupby("weathersit")["cnt"].agg(["mean", "size"])
        bars = ax.bar(x + offset, group.reindex(codes)["mean"], width=0.35,
                      label=str(year), color=COLORS[year])
        for bar, code in zip(bars, codes):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 110,
                    f"{group.loc[code, 'mean']:,.0f}\nn={int(group.loc[code, 'size'])}",
                    ha="center", va="bottom", fontsize=9, color="#354A58")
    ax.set(title="Weather groups by year | association, not causation", ylabel="Mean rentals / day",
           xlabel="Recorded weather category", xticks=x, xticklabels=[WEATHER[c] for c in codes],
           ylim=(0, 7300))
    ax.grid(axis="y", alpha=0.16)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    save(fig, "03_weather_patterns.png")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)
    df, quality = load_and_validate()
    stats = export_tables(df, quality)
    style()
    plot_trend(df)
    plot_calendar(df)
    plot_weather(df)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("Exported 3 figures, 9 CSV tables, and summary.json. Raw records are unchanged.")


if __name__ == "__main__":
    main()
