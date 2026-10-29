"""
Display model performance scores for regression forecasts and optional
classification scores for peak-event/demand-level predictions.

Regression CSV columns:
    Actual, Predicted, Model
Optional:
    Target, Horizon

Classification scores require:
    Actual_Class, Predicted_Class, Model

Run:
    python show_score.py --input forecast_predictions.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def regression_scores(df):
    required = ["Actual", "Predicted", "Model"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Required: {required}")

    data = df.copy()
    data["Actual"] = pd.to_numeric(data["Actual"], errors="coerce")
    data["Predicted"] = pd.to_numeric(data["Predicted"], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["Actual", "Predicted", "Model"]
    )
    if data.empty:
        raise ValueError("No valid regression rows.")

    group_cols = ["Model"]
    if "Target" in data.columns:
        group_cols.insert(0, "Target")
    if "Horizon" in data.columns:
        group_cols.insert(len(group_cols), "Horizon")

    rows = []
    for keys, group in data.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        y_true = group["Actual"].to_numpy(dtype=float)
        y_pred = group["Predicted"].to_numpy(dtype=float)
        nonzero = y_true != 0
        mape = (
            np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100
            if nonzero.any() else np.nan
        )
        row.update({
            "Samples": len(group),
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "MAPE (%)": mape,
            "R2": r2_score(y_true, y_pred) if len(group) > 1 else np.nan,
        })
        rows.append(row)
    return pd.DataFrame(rows).sort_values("MAE")


def classification_scores(df, peak_quantile=0.90):
    data = df.copy()
    derived_peak_classes = not {
        "Actual_Class",
        "Predicted_Class",
    }.issubset(data.columns)
    if derived_peak_classes:
        required = ["Actual", "Predicted", "Model"]
        if not set(required).issubset(data.columns):
            return None

        data["Actual"] = pd.to_numeric(data["Actual"], errors="coerce")
        data["Predicted"] = pd.to_numeric(data["Predicted"], errors="coerce")
        data = data.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["Actual", "Predicted", "Model"]
        )
        if data.empty:
            raise ValueError("No valid rows available for classification scores.")

        threshold_groups = [
            column for column in ["Target", "Horizon"] if column in data.columns
        ]
        if threshold_groups:
            data["Peak_Threshold"] = data.groupby(
                threshold_groups, dropna=False
            )["Actual"].transform(lambda values: values.quantile(peak_quantile))
        else:
            data["Peak_Threshold"] = data["Actual"].quantile(peak_quantile)

        data["Actual_Class"] = (data["Actual"] >= data["Peak_Threshold"]).astype(int)
        data["Predicted_Class"] = (
            data["Predicted"] >= data["Peak_Threshold"]
        ).astype(int)
    else:
        required = ["Actual_Class", "Predicted_Class", "Model"]
        if not set(required).issubset(data.columns):
            return None
        data = data.dropna(subset=required)
        if data.empty:
            raise ValueError("No valid rows available for classification scores.")

    group_cols = [
        column for column in ["Target", "Horizon", "Model"] if column in data.columns
    ]
    rows = []
    for keys, group in data.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        y_true = group["Actual_Class"]
        y_pred = group["Predicted_Class"]
        row.update({
            "Samples": len(group),
            "Actual Positive Samples": int((y_true == 1).sum())
            if derived_peak_classes else np.nan,
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision (weighted)": precision_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "Recall (weighted)": recall_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "F1 (weighted)": f1_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
        })
        if derived_peak_classes:
            row.update({
                "Peak Threshold": group["Peak_Threshold"].iloc[0],
                "Peak Precision": precision_score(
                    y_true, y_pred, pos_label=1, zero_division=0
                ),
                "Peak Recall": recall_score(
                    y_true, y_pred, pos_label=1, zero_division=0
                ),
                "Peak F1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
            })
        rows.append(row)

    result = pd.DataFrame(rows)
    sort_column = "Peak F1" if derived_peak_classes else "F1 (weighted)"
    return result.sort_values(sort_column, ascending=False)


def main():
    parser = argparse.ArgumentParser(description="Show model performance scores.")
    parser.add_argument("--input", required=True, help="Forecast/model predictions CSV")
    parser.add_argument("--output-dir", default="model_scores")
    parser.add_argument(
        "--peak-quantile",
        type=float,
        default=0.90,
        help=(
            "Actual-demand quantile used as the peak threshold when class labels "
            "are absent (default: 0.90)"
        ),
    )
    args = parser.parse_args()

    if not 0 < args.peak_quantile < 1:
        parser.error("--peak-quantile must be between 0 and 1")

    source = Path(args.input)
    if not source.is_file():
        raise FileNotFoundError(f"CSV not found: {source}")

    df = pd.read_csv(source)
    aliases = {
        "actual": "Actual",
        "predicted": "Predicted",
        "model": "Model",
        "target": "Target",
        "horizon": "Horizon",
        "horizon_minutes": "Horizon",
        "actual_class": "Actual_Class",
        "predicted_class": "Predicted_Class",
    }
    rename = {
        column: aliases[column.lower()]
        for column in df.columns
        if column.lower() in aliases and aliases[column.lower()] not in df.columns
    }
    df = df.rename(columns=rename)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    reg = regression_scores(df)
    reg.to_csv(output / "regression_scores.csv", index=False)
    print("\n=== Regression Model Scores ===")
    print(reg.to_string(index=False))

    cls = classification_scores(df, peak_quantile=args.peak_quantile)
    if cls is not None:
        cls.to_csv(output / "classification_scores.csv", index=False)
        print("\n=== Classification Scores ===")
        if {"Actual_Class", "Predicted_Class"}.issubset(df.columns):
            print("Using the Actual_Class and Predicted_Class columns from the input.")
        else:
            print(
                "Peak classes derived from Actual/Predicted using the "
                f"{args.peak_quantile:.0%} actual-demand quantile."
            )
        print(cls.to_string(index=False))
    else:
        print(
            "\nClassification scores skipped: provide Actual_Class and "
            "Predicted_Class, or Actual and Predicted values to derive peak classes."
        )

    print(f"\nScores saved in: {output.resolve()}")


if __name__ == "__main__":
    main()