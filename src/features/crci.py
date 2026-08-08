import pandas as pd
from sklearn.preprocessing import StandardScaler


class CountryRiskIndex:
    """
    Country Risk Composite Index

    Version 2.0

    Economic Score based on normalized macroeconomic indicators.
    """

    def __init__(self):

        self.scaler = StandardScaler()

        self.features = [
            "gdp_per_capita",
            "gdp_growth",
            "inflation",
            "unemployment",
            "life_expectancy",
            "exports",
        ]

        self.weights = {
            "gdp_per_capita": 0.30,
            "gdp_growth": 0.20,
            "inflation": -0.20,
            "unemployment": -0.15,
            "life_expectancy": 0.10,
            "exports": 0.05,
        }

    def fit(self, df: pd.DataFrame):

        self.scaler.fit(df[self.features])

        return self

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

        economic_score = 0

        for feature, weight in self.weights.items():

            economic_score += (
                scaled[feature] * weight
            )

        data["economic_score"] = economic_score

        print("\nEconomic Score")
        print("----------------------------")
        print(data["economic_score"].describe())

        return data

    def fit_transform(self, df: pd.DataFrame):

        self.fit(df)

        return self.transform(df)