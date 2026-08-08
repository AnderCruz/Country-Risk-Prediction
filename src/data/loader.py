import pandas as pd

from api.client import WorldBankClient
from config import INDICATORS, RAW_DATA_DIR


client = WorldBankClient()


def create_dataframe(data: list, column_name: str) -> pd.DataFrame:
    """
    Convert World Bank JSON data into a clean DataFrame.
    """

    df = pd.DataFrame(data)

    df["country"] = df["country"].apply(lambda x: x["value"])

    df = df[
        [
            "country",
            "countryiso3code",
            "date",
            "value",
        ]
    ]

    df = df.rename(columns={"value": column_name})

    return df


def save_dataframe(df: pd.DataFrame, filename: str) -> None:
    """
    Save dataframe into data/raw.
    """

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    filepath = RAW_DATA_DIR / filename

    df.to_csv(filepath, index=False)

    print(f"Saved: {filepath}")


def download_all_datasets() -> None:
    """
    Download every indicator defined in config.py.
    """

    for name, indicator in INDICATORS.items():

        print(f"\nDownloading {name}")

        data = client.download(indicator)

        df = create_dataframe(data, name)

        save_dataframe(df, f"{name}.csv")


def load_all_datasets() -> dict[str, pd.DataFrame]:
    """
    Load every CSV inside data/raw.
    """

    datasets = {}

    for file in sorted(RAW_DATA_DIR.glob("*.csv")):
        datasets[file.stem] = pd.read_csv(file)

    return datasets