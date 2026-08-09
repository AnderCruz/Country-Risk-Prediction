import pandas as pd

from risk.base import RiskComponent


class EconomicRisk(RiskComponent):
    """
    Economic Risk Component

    Calculates the economic dimension of Country Risk.
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

        self.weights = {
            "gdp_per_capita": 0.40,
            "inflation": -0.25,
            "unemployment": -0.20,
            "life_expectancy": 0.10,
            "exports": 0.05,
        }

    def transform(self, df: pd.DataFrame):

        data = df.copy()

        scaled = self.scaler.transform(
            data[self.features]
        )

        scaled = pd.DataFrame(
            scaled,
            columns=self.features,
            index=data.index,
        )

        risk = 0

        for feature, weight in self.weights.items():

            risk += scaled[feature] * weight

        data["economic_risk"] = risk

        print("\nEconomic Risk")
        print("-" * 40)
        print(data["economic_risk"].describe())

        return data