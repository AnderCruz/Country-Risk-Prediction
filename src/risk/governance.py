import pandas as pd
from sklearn.preprocessing import StandardScaler


class GovernanceRisk:
    """
    Governance Risk Component.

    Uses the six Worldwide Governance Indicators:

    - Voice & Accountability
    - Political Stability
    - Government Effectiveness
    - Regulatory Quality
    - Rule of Law
    - Control of Corruption

    Higher governance values represent better governance.
    Therefore, the resulting Governance Risk is inverted:
    higher values = higher risk.
    """

    def __init__(self):

        self.features = [
            "voice_accountability",
            "political_stability",
            "government_effectiveness",
            "regulatory_quality",
            "rule_of_law",
            "control_corruption",
        ]

        self.weights = {
            "voice_accountability": 1 / 6,
            "political_stability": 1 / 6,
            "government_effectiveness": 1 / 6,
            "regulatory_quality": 1 / 6,
            "rule_of_law": 1 / 6,
            "control_corruption": 1 / 6,
        }

        self.scaler = StandardScaler()

    def fit(self, df: pd.DataFrame):

        available = df[
            self.features
        ].dropna()

        self.scaler.fit(
            available
        )

        return self

    def transform(
        self,
        df: pd.DataFrame,
    ):

        data = df.copy()

        # -------------------------------------------------------------
        # Only calculate Governance Risk where all six WGI indicators
        # are available.
        # -------------------------------------------------------------

        mask = data[
            self.features
        ].notna().all(axis=1)

        data["governance_risk"] = pd.NA

        if mask.any():

            values = data.loc[
                mask,
                self.features,
            ]

            scaled = self.scaler.transform(
                values
            )

            scaled = pd.DataFrame(
                scaled,
                columns=self.features,
                index=values.index,
            )

            governance_score = 0

            for feature, weight in self.weights.items():

                governance_score += (
                    scaled[feature] * weight
                )

            # Better governance = lower risk
            data.loc[
                mask,
                "governance_risk",
            ] = -governance_score

        data["governance_risk"] = pd.to_numeric(
            data["governance_risk"],
            errors="coerce",
        )

        print("\nGovernance Risk")
        print("-" * 30)

        print(
            data["governance_risk"].describe()
        )

        print(
            f"\nValid Governance Risk observations: "
            f"{data['governance_risk'].notna().sum()}"
        )

        return data

    def fit_transform(
        self,
        df: pd.DataFrame,
    ):

        self.fit(df)

        return self.transform(df)