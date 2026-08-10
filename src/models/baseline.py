import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def evaluate_naive_risk_baseline(
    df: pd.DataFrame,
):
    """
    Naive one-step-ahead Country Risk benchmark.

    Prediction:
        Country Risk(t+1) = Country Risk(t)
    """

    data = df.copy()

    # -------------------------------------------------------------------------
    # Sort by country and year
    # -------------------------------------------------------------------------

    data = data.sort_values(
        ["country", "date"]
    )

    # -------------------------------------------------------------------------
    # Previous year's Country Risk
    # -------------------------------------------------------------------------

    data["risk_lag1"] = (
        data.groupby("country")[
            "country_risk_index"
        ].shift(1)
    )

    # -------------------------------------------------------------------------
    # Keep observations with target and prediction
    # -------------------------------------------------------------------------

    data = data.dropna(
        subset=[
            "future_country_risk",
            "risk_lag1",
        ]
    )

    # -------------------------------------------------------------------------
    # Temporal split
    # -------------------------------------------------------------------------

    years = sorted(
        data["date"].unique()
    )

    split_index = int(
        len(years) * 0.80
    )

    test_years = years[
        split_index:
    ]

    test = data[
        data["date"].isin(test_years)
    ]

    # -------------------------------------------------------------------------
    # Predictions
    # -------------------------------------------------------------------------

    y_true = test[
        "future_country_risk"
    ]

    predictions = test[
        "risk_lag1"
    ]

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------

    mae = mean_absolute_error(
        y_true,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            predictions,
        )
    )

    r2 = r2_score(
        y_true,
        predictions,
    )

    # -------------------------------------------------------------------------
    # Report
    # -------------------------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("NAIVE COUNTRY RISK BASELINE")
    print("=" * 60)

    print(
        f"MAE : {mae:.4f}"
    )

    print(
        f"RMSE: {rmse:.4f}"
    )

    print(
        f"R2  : {r2:.4f}"
    )

    print(
        f"Test period: "
        f"{test_years[0]} - {test_years[-1]}"
    )

    print(
        f"Test observations: "
        f"{len(test)}"
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }