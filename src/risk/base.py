from abc import ABC, abstractmethod

import pandas as pd

from sklearn.preprocessing import StandardScaler


class RiskComponent(ABC):
    """
    Base class for every country risk component.

    Examples
    --------
    EconomicRisk
    GovernanceRisk
    FiscalRisk
    ExternalRisk
    """

    def __init__(self):

        self.scaler = StandardScaler()

        self.features = []

        self.weights = {}

    def fit(self, df: pd.DataFrame):

        self.scaler.fit(
            df[self.features]
        )

        return self

    @abstractmethod
    def transform(
        self,
        df: pd.DataFrame,
    ):
        """
        Must return dataframe
        containing the component score.
        """

    def fit_transform(
        self,
        df: pd.DataFrame,
    ):

        self.fit(df)

        return self.transform(df)