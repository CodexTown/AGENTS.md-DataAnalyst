# Revenue Analysis Report

## Data Sources

Processed files:
- `sales_data.csv`: 13 rows → 11 kept (duplicates: 0, missing/non-positive: 2).

## Cleaning Notes

- Standardized column names to snake_case.
- Converted the detected date column to datetime and normalized every region label.
- Derived revenue either from a provided revenue column or by multiplying price and quantity.
- Dropped rows missing any date, region, or revenue or featuring non-positive revenue.

## Key Insights

- Region totals in descending order:
  - North: 4400.00
  - South: 4000.00
  - West: 4000.00
  - East: 1700.00

- Monthly revenue moved from 4400.00 in 2023-01 to 5600.00 in 2023-03.
- Peak consolidated revenue was 5600.00 during 2023-03.

## Artifacts

- Cleaned datasets in `_out/clean`.
- Aggregated revenue table at `_out/aggregates/revenue_by_region_month.csv`.
- Visualizations saved to `_out/viz`.
