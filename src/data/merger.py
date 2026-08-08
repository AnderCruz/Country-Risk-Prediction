import pandas as pd

from config import MERGE_COLUMNS


def merge_datasets(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge all datasets using the columns defined in MERGE_COLUMNS.

    Parameters
    ----------
    datasets : dict
        Dictionary where:
        key = dataset name
        value = pandas DataFrame

    Returns
    -------
    pd.DataFrame
        Merged dataset.
    """

    if not datasets:
        raise ValueError("No datasets found.")

    dataframes = list(datasets.values())

    merged = dataframes[0]

    for dataframe in dataframes[1:]:

        merged = merged.merge(
            dataframe,
            on=MERGE_COLUMNS,
            how="inner"
        )

    return merged