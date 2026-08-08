from pathlib import Path

import pandas as pd

from config import PROJECT_ROOT


REPORTS_DIR = PROJECT_ROOT / "reports"


def generate_eda_report(df: pd.DataFrame) -> None:
    """
    Generate basic EDA reports.
    """

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------
    # Descriptive statistics
    # -------------------------------------------------------

    df.describe(include="all").to_csv(
        REPORTS_DIR / "descriptive_statistics.csv"
    )

    # -------------------------------------------------------
    # Missing values
    # -------------------------------------------------------

    missing = (
        df.isna()
        .sum()
        .to_frame("missing_values")
    )

    missing["percentage"] = (
        missing["missing_values"] / len(df) * 100
    )

    missing.to_csv(
        REPORTS_DIR / "missing_values.csv"
    )

    # -------------------------------------------------------
    # Correlation Matrix
    # -------------------------------------------------------

    correlation = df.select_dtypes(include="number").corr()

    correlation.to_csv(
        REPORTS_DIR / "correlation_matrix.csv"
    )

    print("\nEDA reports generated successfully.")