"""Train, compare, and export used vehicle price models."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from src.charts import (
    actual_vs_predicted_chart,
    feature_impact_chart,
    grouped_feature_importance,
    model_comparison_chart,
    price_distribution_chart,
    save_chart,
)
from src.data import load_vehicle_data
from src.modeling import train_and_evaluate


def save_figures(frame, result, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    best_r2 = float(result.metrics.loc[0, "R2"])

    save_chart(model_comparison_chart(result.metrics, result.model_name), output_dir / "model_comparison.png")
    save_chart(price_distribution_chart(frame["price"]), output_dir / "price_distribution.png")
    save_chart(
        actual_vs_predicted_chart(result.predictions, result.model_name, r2=best_r2),
        output_dir / "actual_vs_predicted.png",
    )
    impact = grouped_feature_importance(result.model, result.feature_importance)
    if impact is None:
        impact = result.feature_importance
    save_chart(
        feature_impact_chart(impact, top=15, title=f"Top feature impacts ({result.model_name})"),
        output_dir / "feature_importance.png",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train used vehicle price prediction models.")
    parser.add_argument("--data", required=True, help="Path to a used vehicle CSV file")
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation proportion, default 0.2")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    artifacts = root / "artifacts"
    reports = root / "reports" / "figures"
    artifacts.mkdir(exist_ok=True)
    frame = load_vehicle_data(args.data)
    result = train_and_evaluate(frame, test_size=args.test_size)
    input_profile = {}
    for column in result.numeric_features:
        series = frame[column].dropna()
        input_profile[column] = {
            "kind": "numeric",
            "minimum": float(series.quantile(0.01)),
            "maximum": float(series.quantile(0.99)),
            "default": float(series.median()),
        }
    for column in result.categorical_features:
        values = frame[column].dropna().astype(str).value_counts().head(100).index.tolist()
        input_profile[column] = {"kind": "categorical", "values": values, "default": values[0] if values else "Unknown"}

    result.metrics.to_csv(artifacts / "model_metrics.csv", index=False)
    result.feature_importance.to_csv(artifacts / "feature_importance.csv", index=False)
    joblib.dump({
        "model": result.model,
        "model_name": result.model_name,
        "numeric_features": result.numeric_features,
        "categorical_features": result.categorical_features,
        "metrics": result.metrics,
        "input_profile": input_profile,
    }, artifacts / "vehicle_price_model.joblib", compress=3)
    save_figures(frame, result, reports)
    print(f"Trained on {len(frame):,} rows.")
    print(f"Selected model: {result.model_name}")
    print(result.metrics.to_string(index=False, float_format=lambda value: f"{value:,.4f}"))
    print(f"Saved model: {artifacts / 'vehicle_price_model.joblib'}")


if __name__ == "__main__":
    main()
