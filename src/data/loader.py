import pandas as pd
from pathlib import Path
from api.client import WorldBankClient
from data.sources.wgi import WGISource

from config import (
    WORLD_BANK_DIR,
)

from indicators import (
    ECONOMIC_INDICATORS,
)

client = WorldBankClient()

# =============================================================================
# DATAFRAME
# =============================================================================

def create_dataframe(
    data: list,
    column_name: str,
) -> pd.DataFrame:
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

    WORLD_BANK_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    filepath = WORLD_BANK_DIR / filename

    df.to_csv(
        filepath,
        index=False,
    )

    print(f"Saved: {filepath}")


# =============================================================================
# DOWNLOAD
# =============================================================================

def download_indicators(
    indicators: dict,
) -> None:
    """
    Download World Bank indicators.
    """

    WORLD_BANK_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for name, indicator in indicators.items():

        filepath = WORLD_BANK_DIR / f"{name}.csv"

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


# =============================================================================
# DOWNLOAD ALL
# =============================================================================

def load_all_datasets():
    """
    Load datasets from all data sources.
    """

    datasets = {}

    # ---------------------------------------------------------------
    # WORLD BANK
    # ---------------------------------------------------------------

    for file in sorted(
        WORLD_BANK_DIR.glob("*.csv")
    ):

        datasets[file.stem] = pd.read_csv(
            file
        )

    # ---------------------------------------------------------------
    # GOVERNANCE / WGI
    # ---------------------------------------------------------------

    governance_dir = (
        WORLD_BANK_DIR.parent / "governance"
    )

    for file in sorted(
        governance_dir.glob("*.csv")
    ):

        datasets[file.stem] = pd.read_csv(
            file
        )

    return datasets