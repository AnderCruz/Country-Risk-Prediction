import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


FEATURES = [
    "gdp_per_capita",
    "inflation",
    "life_expectancy",
    "population",
    "population_growth",
    "unemployment",
    "exports",
    "gdp_lag1",
    "inflation_lag1",
    "life_expectancy_lag1",
    "economic_risk",
    "governance_risk",
]

PREDICTION_COLUMN = "prediction"

PSI_WARNING_THRESHOLD = 0.10
PSI_DRIFT_THRESHOLD = 0.20


def calculate_psi(
    reference: pd.Series,
    production: pd.Series,
    bins: int = 10,
) -> float:
    """
    Calculate Population Stability Index (PSI).

    Bins are defined from reference quantiles so that the
    production distribution is evaluated against the
    reference population.
    """

    reference = reference.dropna().astype(float)
    production = production.dropna().astype(float)

    if reference.empty or production.empty:
        return float("nan")

    quantiles = np.linspace(0, 1, bins + 1)

    edges = np.unique(
        reference.quantile(quantiles).values
    )

    if len(edges) < 3:
        return 0.0

    edges[0] = -np.inf
    edges[-1] = np.inf

    reference_counts, _ = np.histogram(
        reference,
        bins=edges,
    )

    production_counts, _ = np.histogram(
        production,
        bins=edges,
    )

    reference_pct = (
        reference_counts / reference_counts.sum()
    )

    production_pct = (
        production_counts / production_counts.sum()
    )

    epsilon = 1e-10

    reference_pct = np.clip(
        reference_pct,
        epsilon,
        None,
    )

    production_pct = np.clip(
        production_pct,
        epsilon,
        None,
    )

    psi = np.sum(
        (production_pct - reference_pct)
        * np.log(
            production_pct / reference_pct
        )
    )

    return float(psi)


def classify_psi(psi: float) -> str:
    """
    Classify drift according to PSI thresholds.
    """

    if np.isnan(psi):
        return "INSUFFICIENT_DATA"

    if psi < PSI_WARNING_THRESHOLD:
        return "STABLE"

    if psi < PSI_DRIFT_THRESHOLD:
        return "WARNING"

    return "SIGNIFICANT_DRIFT"


def calculate_distribution_drift(
    reference: pd.Series,
    production: pd.Series,
) -> dict:
    """
    Calculate PSI and KS statistics for two distributions.
    """

    reference_values = (
        reference
        .dropna()
        .astype(float)
    )

    production_values = (
        production
        .dropna()
        .astype(float)
    )

    if (
        reference_values.empty
        or production_values.empty
    ):
        return {
            "psi": float("nan"),
            "ks_statistic": float("nan"),
            "ks_pvalue": float("nan"),
            "status": "INSUFFICIENT_DATA",
            "reference_count": len(reference_values),
            "production_count": len(production_values),
        }

    psi = calculate_psi(
        reference_values,
        production_values,
    )

    ks_result = ks_2samp(
        reference_values,
        production_values,
    )

    ks_statistic = float(
        ks_result.statistic
    )

    ks_pvalue = float(
        ks_result.pvalue
    )

    return {
        "psi": psi,
        "ks_statistic": ks_statistic,
        "ks_pvalue": ks_pvalue,
        "status": classify_psi(psi),
        "reference_count": len(reference_values),
        "production_count": len(production_values),
    }


def calculate_data_drift(
    reference: pd.DataFrame,
    production: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate data drift for all model input features.
    """

    results = []

    for feature in FEATURES:

        drift = calculate_distribution_drift(
            reference[feature],
            production[feature],
        )

        results.append(
            {
                "feature": feature,
                **drift,
            }
        )

    return pd.DataFrame(results)


def calculate_prediction_drift(
    reference_predictions: pd.DataFrame,
    production: pd.DataFrame,
) -> dict:
    """
    Calculate prediction drift between reference predictions
    and production predictions.
    """

    if PREDICTION_COLUMN not in reference_predictions.columns:
        raise ValueError(
            "Missing prediction column in reference predictions: "
            f"{PREDICTION_COLUMN}"
        )

    if PREDICTION_COLUMN not in production.columns:
        raise ValueError(
            "Missing prediction column in production dataset: "
            f"{PREDICTION_COLUMN}"
        )

    return calculate_distribution_drift(
        reference_predictions[PREDICTION_COLUMN],
        production[PREDICTION_COLUMN],
    )


def determine_overall_status(
    data_drift: pd.DataFrame,
    prediction_drift: dict,
) -> str:
    """
    Determine the overall monitoring status.

    CRITICAL:
        Any significant feature drift OR significant prediction drift.

    WARNING:
        No significant drift, but at least one warning.

    STABLE:
        No warning or significant drift.
    """

    significant_features = (
        data_drift["status"]
        == "SIGNIFICANT_DRIFT"
    ).sum()

    warning_features = (
        data_drift["status"]
        == "WARNING"
    ).sum()

    prediction_status = (
        prediction_drift["status"]
    )

    if (
        significant_features > 0
        or prediction_status == "SIGNIFICANT_DRIFT"
    ):
        return "CRITICAL"

    if (
        warning_features > 0
        or prediction_status == "WARNING"
    ):
        return "WARNING"

    if (
        prediction_status == "INSUFFICIENT_DATA"
        or (data_drift["status"] == "INSUFFICIENT_DATA").any()
    ):
        return "WARNING"

    return "STABLE"


def build_report(
    reference_path: Path,
    production_path: Path,
    reference_predictions_path: Path,
    reference: pd.DataFrame,
    production: pd.DataFrame,
    reference_predictions: pd.DataFrame,
    data_drift: pd.DataFrame,
    prediction_drift: dict,
) -> dict:
    """
    Build a machine-readable monitoring report.
    """

    significant_features = int(
        (
            data_drift["status"]
            == "SIGNIFICANT_DRIFT"
        ).sum()
    )

    warning_features = int(
        (
            data_drift["status"]
            == "WARNING"
        ).sum()
    )

    stable_features = int(
        (
            data_drift["status"]
            == "STABLE"
        ).sum()
    )

    overall_status = determine_overall_status(
        data_drift,
        prediction_drift,
    )

    alert_triggered = (
        overall_status
        in {"WARNING", "CRITICAL"}
    )

    return {
        "project": "country-risk-prediction",
        "monitoring": {
            "overall_status": overall_status,
            "alert_triggered": alert_triggered,
        },
        "datasets": {
            "reference": str(reference_path),
            "production": str(production_path),
            "reference_predictions": str(
                reference_predictions_path
            ),
            "reference_rows": int(len(reference)),
            "production_rows": int(len(production)),
            "reference_prediction_rows": int(
                len(reference_predictions)
            ),
        },
        "thresholds": {
            "psi_warning": PSI_WARNING_THRESHOLD,
            "psi_significant": PSI_DRIFT_THRESHOLD,
        },
        "data_drift": {
            "stable_features": stable_features,
            "warning_features": warning_features,
            "significant_drift_features": significant_features,
            "features": data_drift.to_dict(
                orient="records"
            ),
        },
        "prediction_drift": prediction_drift,
    }


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Calculate data drift and prediction drift "
            "for the Country Risk Prediction model."
        )
    )

    parser.add_argument(
        "--reference",
        required=True,
        help="Path to reference feature dataset.",
    )

    parser.add_argument(
        "--production",
        required=True,
        help="Path to production dataset.",
    )

    parser.add_argument(
        "--reference-predictions",
        required=True,
        help="Path to reference predictions dataset.",
    )

    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional path for machine-readable JSON "
            "monitoring report."
        ),
    )

    return parser.parse_args()


def main() -> int:

    args = parse_arguments()

    reference_path = Path(
        args.reference
    )

    production_path = Path(
        args.production
    )

    reference_predictions_path = Path(
        args.reference_predictions
    )

    reference = pd.read_csv(
        reference_path
    )

    production = pd.read_csv(
        production_path
    )

    reference_predictions = pd.read_csv(
        reference_predictions_path
    )

    missing_reference = [
        feature
        for feature in FEATURES
        if feature not in reference.columns
    ]

    missing_production = [
        feature
        for feature in FEATURES
        if feature not in production.columns
    ]

    if missing_reference:
        raise ValueError(
            "Missing features in reference dataset: "
            f"{missing_reference}"
        )

    if missing_production:
        raise ValueError(
            "Missing features in production dataset: "
            f"{missing_production}"
        )

    data_drift = calculate_data_drift(
        reference,
        production,
    )

    prediction_drift = calculate_prediction_drift(
        reference_predictions,
        production,
    )

    report = build_report(
        reference_path=reference_path,
        production_path=production_path,
        reference_predictions_path=reference_predictions_path,
        reference=reference,
        production=production,
        reference_predictions=reference_predictions,
        data_drift=data_drift,
        prediction_drift=prediction_drift,
    )

    overall_status = report["monitoring"]["overall_status"]

    alert_triggered = report["monitoring"]["alert_triggered"]

    print("=" * 80)
    print("COUNTRY RISK PREDICTION - MODEL MONITORING")
    print("=" * 80)

    print("\nDATASETS")

    print(
        f"\nReference dataset          : "
        f"{reference_path}"
    )

    print(
        f"Production dataset         : "
        f"{production_path}"
    )

    print(
        f"Reference predictions     : "
        f"{reference_predictions_path}"
    )

    print(
        f"Reference rows             : "
        f"{len(reference)}"
    )

    print(
        f"Production rows            : "
        f"{len(production)}"
    )

    print(
        f"Reference predictions     : "
        f"{len(reference_predictions)}"
    )

    print("\n" + "=" * 80)
    print("DATA DRIFT")
    print("=" * 80)

    print(
        data_drift.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print(
        f"\nStable features           : "
        f"{report['data_drift']['stable_features']}"
    )

    print(
        f"Warning features          : "
        f"{report['data_drift']['warning_features']}"
    )

    print(
        f"Significant drift features: "
        f"{report['data_drift']['significant_drift_features']}"
    )

    print("\n" + "=" * 80)
    print("PREDICTION DRIFT")
    print("=" * 80)

    print(
        f"\nReference count : "
        f"{prediction_drift['reference_count']}"
    )

    print(
        f"Production count: "
        f"{prediction_drift['production_count']}"
    )

    print(
        f"PSI             : "
        f"{prediction_drift['psi']:.6f}"
    )

    print(
        f"KS statistic    : "
        f"{prediction_drift['ks_statistic']:.6f}"
    )

    print(
        f"KS p-value      : "
        f"{prediction_drift['ks_pvalue']:.6f}"
    )

    print(
        f"Status          : "
        f"{prediction_drift['status']}"
    )

    print("\n" + "=" * 80)
    print("OVERALL MONITORING STATUS")
    print("=" * 80)

    print(
        f"\nData drift status      : "
        f"{'SIGNIFICANT_DRIFT' if report['data_drift']['significant_drift_features'] > 0 else 'NO_SIGNIFICANT_DRIFT'}"
    )

    print(
        f"Prediction drift status: "
        f"{prediction_drift['status']}"
    )

    print(
        f"Overall status          : "
        f"{overall_status}"
    )

    print(
        f"Alert triggered         : "
        f"{'YES' if alert_triggered else 'NO'}"
    )

    if args.output:

        output_path = Path(
            args.output
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                report,
                file,
                indent=2,
                allow_nan=False,
            )

        print(
            f"\nMonitoring report saved: "
            f"{output_path}"
        )

    print()

    if overall_status == "CRITICAL":
        print(
            "ALERT: Significant model drift detected."
        )
        return 1

    if overall_status == "WARNING":
        print(
            "WARNING: Model monitoring requires attention."
        )
        return 1

    print(
        "OK: No significant model drift detected."
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())