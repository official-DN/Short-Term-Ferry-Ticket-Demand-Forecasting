"""
Key performance indicators (KPIs) for ferry demand forecasts.

Expected input CSV columns:
    Actual, Predicted, Horizon
Optional columns:
    Lower, Upper, Actual_Datetime, Forecast_Datetime

Example:
    python indicator.py --input forecast_predictions.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def calculate_kpis(df, peak_quantile=0.90):
    required = ["Actual", "Predicted", "Horizon"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Required: {required}")

    data = df.copy()
    for col in ["Actual", "Predicted"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["Actual", "Predicted", "Horizon"]
    )
    if data.empty:
        raise ValueError("No valid rows available.")

    data["Error"] = data["Actual"] - data["Predicted"]
    data["Absolute_Error"] = data["Error"].abs()

    # Peak-demand threshold is based on the actual demand distribution.
    peak_threshold = data["Actual"].quantile(peak_quantile)
    actual_peak = data["Actual"] >= peak_threshold
    missed_peak = actual_peak & (data["Predicted"] < peak_threshold)
    peak_miss_rate = (
        missed_peak.sum() / actual_peak.sum() * 100 if actual_peak.sum() else np.nan
    )

    # Accuracy-like percentage: 100 - MAPE. MAPE excludes zero actual values.
    nonzero = data["Actual"] != 0
    mape = (
        (data.loc[nonzero, "Absolute_Error"] / data.loc[nonzero, "Actual"].abs()).mean()
        * 100 if nonzero.any() else np.nan
    )
    forecast_accuracy = max(0.0, 100.0 - mape) if pd.notna(mape) else np.nan

    rows = []
    for horizon, group in data.groupby("Horizon", sort=False):
        row = {
            "Horizon": horizon,
            "Samples": len(group),
            "Forecast Accuracy (%)": forecast_accuracy,
            "MAE": group["Absolute_Error"].mean(),
            "Mean Error (Bias)": group["Error"].mean(),
            "Peak Miss Rate (%)": peak_miss_rate,
        }

        # Error drift: change in MAE across horizons compared with shortest horizon.
        rows.append(row)

    result = pd.DataFrame(rows)
    baseline_mae = result["MAE"].iloc[0] if len(result) else np.nan
    result["Error Drift (%)"] = (
        (result["MAE"] - baseline_mae) / baseline_mae * 100
        if pd.notna(baseline_mae) and baseline_mae != 0 else np.nan
    )

    # Confidence-band width is only calculated when interval columns are present.
    if {"Lower", "Upper"}.issubset(data.columns):
        data["Lower"] = pd.to_numeric(data["Lower"], errors="coerce")
        data["Upper"] = pd.to_numeric(data["Upper"], errors="coerce")
        data["Band_Width"] = data["Upper"] - data["Lower"]
        band = data.groupby("Horizon")["Band_Width"].mean()
        result["Confidence Band Width"] = result["Horizon"].map(band)
    else:
        result["Confidence Band Width"] = np.nan

    # Lead time: forecast horizon in minutes if the horizon labels are parseable.
    result["Forecast Lead Time"] = result["Horizon"].astype(str)
    return result


def main():
    parser = argparse.ArgumentParser(description="Calculate ferry forecasting KPIs.")
    parser.add_argument("--input", required=True, help="Forecast predictions CSV")
    parser.add_argument("--output", default="model_scores/kpi_results.csv")
    parser.add_argument("--peak-quantile", type=float, default=0.90)
    args = parser.parse_args()

    if not 0 < args.peak_quantile < 1:
        raise ValueError("--peak-quantile must be between 0 and 1")

    df = pd.read_csv(args.input)
    result = calculate_kpis(df, peak_quantile=args.peak_quantile)
    result.to_csv(args.output, index=False)
    print("\\n=== Forecast KPIs ===")
    print(result.to_string(index=False))
    print(f"\\nSaved KPI results to: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()