import pandas as pd

from sklearn.decomposition import PCA

from risk.base import RiskComponent


class PCAEconomicRisk(RiskComponent):
    """
    Economic Risk calculated using Principal Component Analysis.
    """

    def __init__(self):

        super().__init__()

        self.features = [
            "gdp_per_capita",
            "inflation",
            "unemployment",
            "life_expectancy",
            "exports",
        ]

        self.pca = PCA(n_components=1)

    def fit(self, df: pd.DataFrame):

        scaled = self.scaler.fit_transform(
            df[self.features]
        )

        self.pca.fit(scaled)

        return self

    def transform(self, df: pd.DataFrame):

        data = df.copy()

        scaled = self.scaler.transform(
            data[self.features]
        )

        score = self.pca.transform(scaled)

        data["economic_risk_pca"] = score[:, 0]

        print("\nEconomic Risk (PCA)")
        print("-" * 40)
        print(data["economic_risk_pca"].describe())

        return data