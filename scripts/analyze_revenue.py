#!/usr/bin/env python3
"""Repeatable pipeline that ingests raw sales data and produces cleaned artifacts."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT_DIR / "_in"
OUT_DIR = ROOT_DIR / "_out"
CLEAN_DIR = OUT_DIR / "clean"
AGG_DIR = OUT_DIR / "aggregates"
VIZ_DIR = OUT_DIR / "viz"

RAW_DATE_CANDIDATES = [
    "date",
    "order_date",
    "sale_date",
    "timestamp",
    "created_at",
    "order_datetime",
]
RAW_REGION_CANDIDATES = ["region", "country", "geo", "location", "territory"]
RAW_REVENUE_CANDIDATES = ["revenue", "amount", "total", "grand_total", "sales"]
RAW_PRICE_CANDIDATES = ["price", "unit_price", "sale_price"]
RAW_QUANTITY_CANDIDATES = ["quantity", "qty", "units", "volume"]


def normalize_column_name(name: str) -> str:
    """Return a snake_case version of a column name that avoids awkward characters."""
    normalized = re.sub(r"[^0-9a-z]+", "_", name.lower())
    normalized = re.sub(r"_+", "_", normalized)
    return normalized.strip("_")


DATE_CANDIDATES = [normalize_column_name(name) for name in RAW_DATE_CANDIDATES]
REGION_CANDIDATES = [normalize_column_name(name) for name in RAW_REGION_CANDIDATES]
REVENUE_CANDIDATES = [normalize_column_name(name) for name in RAW_REVENUE_CANDIDATES]
PRICE_CANDIDATES = [normalize_column_name(name) for name in RAW_PRICE_CANDIDATES]
QUANTITY_CANDIDATES = [normalize_column_name(name) for name in RAW_QUANTITY_CANDIDATES]


def discover_input_files(input_dir: Path) -> list[Path]:
    """List every file in `_in/` so they can be processed sequentially."""
    return sorted(path for path in input_dir.iterdir() if path.is_file())


def create_output_dirs() -> None:
    """Ensure `_out/clean`, `_out/aggregates`, and `_out/viz` exist."""
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_DIR.mkdir(parents=True, exist_ok=True)


def load_raw_data(path: Path) -> pd.DataFrame:
    """Load tabular data in supported formats (csv, xlsx, parquet)."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xls", ".xlsx"}:
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file format: {path.name}")


def match_first_column(columns: list[str], candidates: list[str]) -> str | None:
    """Return the first column name from the provided candidates that exists."""
    column_set = set(columns)
    for candidate in candidates:
        if candidate in column_set:
            return candidate
    return None


def sanitize_region(region: str) -> str:
    """Return a filesystem-safe name derived from the region label."""
    safe = re.sub(r"[^0-9a-z]+", "_", region.lower())
    safe = re.sub(r"_+", "_", safe)
    return safe.strip("_") or "region"


def clean_dataframe(data_frame: pd.DataFrame, source_name: str) -> tuple[pd.DataFrame, dict[str, int]]:
    """Normalize, filter, and annotate a single raw dataset."""
    df = data_frame.copy()
    stats: dict[str, int] = {
        "source": source_name,
        "original_rows": len(df),
        "duplicates_removed": 0,
        "missing_required": 0,
        "non_positive_removed": 0,
        "final_rows": 0,
    }

    # normalize column names
    rename_map: dict[str, str] = {}
    used_names: set[str] = set()
    for idx, column in enumerate(df.columns):
        normalized = normalize_column_name(str(column)) or f"column_{idx}"
        unique_name = normalized
        counter = 1
        while unique_name in used_names:
            unique_name = f"{normalized}_{counter}"
            counter += 1
        used_names.add(unique_name)
        rename_map[column] = unique_name
    df = df.rename(columns=rename_map)

    duplicates_before = len(df)
    df = df.drop_duplicates()
    stats["duplicates_removed"] = duplicates_before - len(df)

    date_col = match_first_column(list(df.columns), DATE_CANDIDATES)
    region_col = match_first_column(list(df.columns), REGION_CANDIDATES)
    revenue_col = match_first_column(list(df.columns), REVENUE_CANDIDATES)

    if not date_col or not region_col:
        raise ValueError(f"{source_name} is missing date or region columns after normalization")

    df = df.rename(columns={date_col: "date", region_col: "region"})

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["region"] = df["region"].astype(str).str.strip()
    df["region"] = df["region"].replace("", pd.NA)

    if revenue_col:
        df["revenue"] = pd.to_numeric(df[revenue_col], errors="coerce")
    else:
        price_col = match_first_column(list(df.columns), PRICE_CANDIDATES)
        qty_col = match_first_column(list(df.columns), QUANTITY_CANDIDATES)
        if not price_col or not qty_col:
            raise ValueError(
                f"{source_name} lacks enough information to compute revenue (price+quantity or revenue column required)"
            )
        price = pd.to_numeric(df[price_col], errors="coerce")
        quantity = pd.to_numeric(df[qty_col], errors="coerce")
        df["revenue"] = price * quantity

    before_missing = len(df)
    df = df.dropna(subset=["date", "region", "revenue"])
    stats["missing_required"] = before_missing - len(df)

    before_non_positive = len(df)
    df = df[df["revenue"] > 0]
    stats["non_positive_removed"] = before_non_positive - len(df)

    stats["final_rows"] = len(df)

    # final validation
    if df.empty:
        print(f"[{source_name}] no rows remain after cleaning")
    else:
        print(
            f"[{source_name}] cleaned rows: {stats['final_rows']} "
            f"(removed {stats['duplicates_removed']} duplicates, "
            f"{stats['missing_required']} missing, "
            f"{stats['non_positive_removed']} non-positive)"
        )
    return df, stats


def aggregate_revenue_by_region_month(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary table of revenue grouped by region and year-month."""
    temp = df.copy()
    temp["year_month"] = temp["date"].dt.to_period("M").astype(str)
    aggregated = (
        temp.groupby(["region", "year_month"], as_index=False)["revenue"]
        .sum()
        .sort_values(["region", "year_month"])
    )
    print(f"Aggregated {len(aggregated)} region-month combinations")
    return aggregated


def plot_revenue_timeseries(agg_df: pd.DataFrame, viz_dir: Path) -> None:
    """Create an overall trend plot with one line per region."""
    if agg_df.empty:
        print("Skipping overall plot: no aggregated data available")
        return

    pivot = agg_df.pivot(index="year_month", columns="region", values="revenue").fillna(0)
    pivot.index = pd.to_datetime(pivot.index)
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(ax=ax, marker="o")
    ax.set_title("Monthly Revenue by Region")
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue")
    ax.legend(title="Region")
    fig.tight_layout()
    overall_path = viz_dir / "revenue_by_region_month_overall.png"
    fig.savefig(overall_path, dpi=150)
    plt.close(fig)
    print(f"Saved overview chart to {overall_path.relative_to(ROOT_DIR)}")


def plot_region_timeseries(agg_df: pd.DataFrame, viz_dir: Path) -> None:
    """Save a dedicated plot for each region to highlight its own dynamics."""
    if agg_df.empty:
        print("Skipping region plots: no aggregated data available")
        return

    for region in agg_df["region"].unique():
        region_df = agg_df[agg_df["region"] == region].sort_values("year_month")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(
            pd.to_datetime(region_df["year_month"]),
            region_df["revenue"],
            marker="o",
            linewidth=2,
        )
        ax.set_title(f"Revenue Timeline — {region}")
        ax.set_xlabel("Month")
        ax.set_ylabel("Revenue")
        ax.grid(axis="y", linestyle=":")
        fig.tight_layout()
        region_fname = f"revenue_by_region_month_{sanitize_region(region)}.png"
        region_path = viz_dir / region_fname
        fig.savefig(region_path, dpi=150)
        plt.close(fig)
        print(f"Saved region chart to {region_path.relative_to(ROOT_DIR)}")


def write_report(
    processed_files: list[dict[str, int]],
    cleaned_df: pd.DataFrame,
    aggregated_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Create a short Markdown report describing the pipeline results."""
    lines: list[str] = [
        "# Revenue Analysis Report",
        "",
        "## Data Sources",
        "",
        "Processed files:",
    ]
    for stats in processed_files:
        lines.append(
            f"- `{stats['source']}`: {stats['original_rows']} rows → {stats['final_rows']} kept "
            f"(duplicates: {stats['duplicates_removed']}, "
            f"missing/non-positive: {stats['missing_required'] + stats['non_positive_removed']})."
        )

    lines.extend(
        [
            "",
            "## Cleaning Notes",
            "",
            "- Standardized column names to snake_case.",
            "- Converted the detected date column to datetime and normalized every region label.",
            "- Derived revenue either from a provided revenue column or by multiplying price and quantity.",
            "- Dropped rows missing any date, region, or revenue or featuring non-positive revenue.",
            "",
        ]
    )

    monthly_volume = aggregated_df.groupby("year_month", as_index=False)["revenue"].sum()
    monthly_volume = monthly_volume.sort_values("year_month")
    if not monthly_volume.empty:
        start_month = monthly_volume.iloc[0]
        last_month = monthly_volume.iloc[-1]
        peak_month = monthly_volume.loc[monthly_volume["revenue"].idxmax()]
        lines.extend(
            [
                "## Key Insights",
                "",
                f"- Region totals in descending order:",
            ]
        )
        region_totals = (
            aggregated_df.groupby("region", as_index=False)["revenue"]
            .sum()
            .sort_values("revenue", ascending=False)
        )
        for region in region_totals.itertuples(index=False):
            lines.append(f"  - {region.region}: {region.revenue:.2f}")
        lines.extend(
            [
                "",
                f"- Monthly revenue moved from {start_month['revenue']:.2f} in {start_month['year_month']} "
                f"to {last_month['revenue']:.2f} in {last_month['year_month']}.",
                f"- Peak consolidated revenue was {peak_month['revenue']:.2f} during {peak_month['year_month']}.",
                "",
            ]
        )

    lines.extend(
        [
            "## Artifacts",
            "",
            f"- Cleaned datasets in `{CLEAN_DIR.relative_to(ROOT_DIR)}`.",
            f"- Aggregated revenue table at `{AGG_DIR.relative_to(ROOT_DIR)}/revenue_by_region_month.csv`.",
            f"- Visualizations saved to `{VIZ_DIR.relative_to(ROOT_DIR)}`.",
            "",
        ]
    )

    report_path = output_dir / "REPORT.md"
    report_path.write_text("\n".join(lines))
    print(f"Wrote report to {report_path.relative_to(ROOT_DIR)}")


def main() -> None:
    """Run the full ingestion, cleaning, aggregation, and reporting pipeline."""
    create_output_dirs()
    input_files = discover_input_files(INPUT_DIR)
    if not input_files:
        print("No raw data files detected in `_in/`. Provide data to continue.")
        return

    cleaned_frames: list[pd.DataFrame] = []
    file_stats: list[dict[str, int]] = []

    for path in input_files:
        try:
            raw_df = load_raw_data(path)
        except Exception as exc:
            print(f"Failed to load `{path.name}`: {exc}")
            continue
        cleaned_df, stats = clean_dataframe(raw_df, path.name)
        if cleaned_df.empty:
            continue
        cleaned_frames.append(cleaned_df)
        file_stats.append(stats)
        clean_path = CLEAN_DIR / f"clean_{path.stem}.csv"
        cleaned_df.to_csv(clean_path, index=False)
        print(f"Saved cleaned data to {clean_path.relative_to(ROOT_DIR)}")

    if not cleaned_frames:
        print("No cleaned data available to aggregate.")
        return

    combined = pd.concat(cleaned_frames, ignore_index=True)
    aggregated = aggregate_revenue_by_region_month(combined)
    aggregated_path = AGG_DIR / "revenue_by_region_month.csv"
    aggregated.to_csv(aggregated_path, index=False)
    print(f"Saved aggregated revenue to {aggregated_path.relative_to(ROOT_DIR)}")

    plot_revenue_timeseries(aggregated, VIZ_DIR)
    plot_region_timeseries(aggregated, VIZ_DIR)

    write_report(file_stats, combined, aggregated, AGG_DIR)


if __name__ == "__main__":
    main()
