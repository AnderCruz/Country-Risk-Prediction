import pandas as pd
from api.client import WorldBankClient
from config import RAW_DATA_DIR
from indicators import (
    ECONOMIC_INDICATORS,
    GOVERNANCE_INDICATORS,
)
from pathlib import Path

client = WorldBankClient()


# =============================================================================
# DATAFRAME
# =============================================================================

def create_dataframe(data: list, column_name: str) -> pd.DataFrame:
    """
    Convert World Bank JSON into DataFrame.
    """

    df = pd.DataFrame(data)

    df["country"] = df["country"].apply(
        lambda x: x["value"]
    )

    df = df[
        [
            "country",
            "countryiso3code",
            "date",
            "value",
        ]
    ]

    df = df.rename(
        columns={
            "value": column_name
        }
    )

    return df


# =============================================================================
# SAVE
# =============================================================================

def save_dataframe(
    df: pd.DataFrame,
    filename: str,
) -> None:

    RAW_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    filepath = RAW_DATA_DIR / filename

    df.to_csv(
        filepath,
        index=False,
    )

    print(f"Saved: {filepath}")


# =============================================================================
# DOWNLOAD
# =============================================================================

def download_indicators(indicators: dict) -> None:
    """
    Download all indicators.

    - Skip files that already exist.
    - Continue if one indicator fails.
    """

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for name, indicator in indicators.items():

        filepath = RAW_DATA_DIR / f"{name}.csv"

        # ---------------------------------------------------
        # Cache
        # ---------------------------------------------------

        if filepath.exists():

            print(f"✓ {name} already exists")

            continue

        print(f"\nDownloading {name}")

        try:

            data = client.download(indicator)

            df = create_dataframe(
                data,
                name,
            )

            save_dataframe(
                df,
                f"{name}.csv",
            )

            print(f"✓ {name} downloaded")

        except Exception as e:

            print(f"✗ Failed downloading {name}")

            print(e)

            continue


def download_all_datasets() -> None:
    """
    Download every dataset used by the project.
    """

    print("\nDownloading Economic Indicators")
    download_indicators(ECONOMIC_INDICATORS)
    
    # print("\nDownloading Governance Indicators")

    # download_indicators(GOVERNANCE_INDICATORS)


# =============================================================================
# LOAD
# =============================================================================

def load_all_datasets() -> dict:

    datasets = {}

    for file in sorted(
        RAW_DATA_DIR.glob("*.csv")
    ):

        datasets[file.stem] = pd.read_csv(file)

    return datasets