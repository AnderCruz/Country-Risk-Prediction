import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# =============================================================================
# PERFORMANCE ANALYSIS
# =============================================================================


def analyse_model_performance(
    df: pd.DataFrame,
    target_column: str,
    prediction_column: str,
) -> dict:
    """
    Analyse model performance globally, by year and by country.

    Parameters
    ----------
    df:
        Dataset containing actual and predicted values.

    target_column:
        Name of the actual target column.

    prediction_column:
        Name of the prediction column.

    Returns
    -------
    dict
        Global, yearly and country-level performance metrics.
    """

    required_columns = {
        "country",
        "date",
        target_column,
        prediction_column,
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing columns: "
            f"{sorted(missing_columns)}"
        )

    data = df[
        [
            "country",
            "date",
            target_column,
            prediction_column,
        ]
    ].dropna()

    if data.empty:
        raise ValueError(
            "No valid observations available "
            "for performance analysis."
        )

    # =========================================================================
    # GLOBAL PERFORMANCE
    # =========================================================================

    y_true = data[target_column]
    y_pred = data[prediction_column]

    global_metrics = {
        "mae": mean_absolute_error(
            y_true,
            y_pred,
        ),
        "rmse": mean_squared_error(
            y_true,
            y_pred,
        ) ** 0.5,
        "r2": r2_score(
            y_true,
            y_pred,
        ),
    }

    # =========================================================================
    # PERFORMANCE BY YEAR
    # =========================================================================

    yearly_results = []

    for year, group in data.groupby("date"):

        if len(group) < 2:
            r2 = float("nan")
        else:
            r2 = r2_score(
                group[target_column],
                group[prediction_column],
            )

        yearly_results.append(
            {
                "date": year,
                "mae": mean_absolute_error(
                    group[target_column],
                    group[prediction_column],
                ),
                "rmse": mean_squared_error(
                    group[target_column],
                    group[prediction_column],
                ) ** 0.5,
                "r2": r2,
                "n_observations": len(group),
            }
        )

    yearly_metrics = pd.DataFrame(
        yearly_results
    )

    # =========================================================================
    # PERFORMANCE BY COUNTRY
    # =========================================================================

    country_results = []

    for country, group in data.groupby("country"):

        if len(group) < 2:
            r2 = float("nan")
        else:
            r2 = r2_score(
                group[target_column],
                group[prediction_column],
            )

        country_results.append(
            {
                "country": country,
                "mae": mean_absolute_error(
                    group[target_column],
                    group[prediction_column],
                ),
                "rmse": mean_squared_error(
                    group[target_column],
                    group[prediction_column],
                ) ** 0.5,
                "r2": r2,
                "n_observations": len(group),
            }
        )

    country_metrics = pd.DataFrame(
        country_results
    )

    return {
        "global": global_metrics,
        "yearly": yearly_metrics,
        "country": country_metrics,
    }


# =============================================================================
# BUILD PERFORMANCE DATASET
# =============================================================================


def build_performance_dataset(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    predictions,
    source_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a dataset containing actual and predicted values
    for the model test set.

    The function preserves country and date information
    from the original test observations.
    """

    if len(X_test) != len(y_test):
        raise ValueError(
            "X_test and y_test must have "
            "the same number of observations."
        )

    if len(y_test) != len(predictions):
        raise ValueError(
            "y_test and predictions must have "
            "the same number of observations."
        )

    test_data = source_df.loc[
        X_test.index,
        [
            "country",
            "date",
        ],
    ].copy()

    test_data["actual"] = y_test.values
    test_data["prediction"] = predictions

    return test_data.reset_index(
        drop=True
    )