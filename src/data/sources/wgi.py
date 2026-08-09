from pathlib import Path

import pandas as pd


class WGISource:
    """
    Worldwide Governance Indicators source.

    Reads the official WGI Excel workbook and creates
    a single governance dataset.
    """

    DATA_DIR = Path("data/external/wgi")
    OUTPUT_DIR = Path("data/raw/governance")

    INDICATORS = {
        "va": "voice_accountability",
        "pv": "political_stability",
        "ge": "government_effectiveness",
        "rq": "regulatory_quality",
        "rl": "rule_of_law",
        "cc": "control_corruption",
    }

    VALUE_COLUMN = (
        "Governance estimate "
        "(approx. -2.5 to +2.5)"
    )

    def __init__(self):

        self.DATA_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ------------------------------------------------------------------
    # FIND WGI FILE
    # ------------------------------------------------------------------

    def get_excel_file(self):

        files = list(
            self.DATA_DIR.glob("*.xlsx")
        )

        if not files:

            raise FileNotFoundError(
                f"No WGI Excel file found in "
                f"{self.DATA_DIR}"
            )

        return files[0]

    # ------------------------------------------------------------------
    # READ INDICATOR
    # ------------------------------------------------------------------

    def read_indicator(
        self,
        excel,
        sheet_name,
        column_name,
    ):

        df = pd.read_excel(
            excel,
            sheet_name=sheet_name,
        )

        required_columns = [
            "Economy (name)",
            "Economy (code)",
            "Year",
            self.VALUE_COLUMN,
        ]

        missing = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing:

            raise ValueError(
                f"Missing columns in sheet "
                f"'{sheet_name}': {missing}"
            )

        df = df[
            required_columns
        ].copy()

        df = df.rename(
            columns={
                "Economy (name)": "country",
                "Economy (code)": "countryiso3code",
                "Year": "date",
                self.VALUE_COLUMN: column_name,
            }
        )

        return df

    # ------------------------------------------------------------------
    # BUILD DATASET
    # ------------------------------------------------------------------

    def build_dataset(self):

        file = self.get_excel_file()

        print(
            f"\nLoading WGI: {file.name}"
        )

        excel = pd.ExcelFile(file)

        datasets = []

        for sheet_name, column_name in self.INDICATORS.items():

            print(
                f"Reading {column_name}"
            )

            df = self.read_indicator(
                excel,
                sheet_name,
                column_name,
            )

            print(
                f"Rows: {len(df)}"
            )

            datasets.append(df)

        # --------------------------------------------------------------
        # MERGE SIX GOVERNANCE INDICATORS
        # --------------------------------------------------------------

        governance = datasets[0]

        for df in datasets[1:]:

            governance = pd.merge(
                governance,
                df,
                on=[
                    "country",
                    "countryiso3code",
                    "date",
                ],
                how="outer",
            )

        governance = governance.sort_values(
            by=[
                "countryiso3code",
                "date",
            ]
        ).reset_index(drop=True)

        return governance

    # ------------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------------

    def save(self, df):

        output_file = (
            self.OUTPUT_DIR
            / "governance.csv"
        )

        df.to_csv(
            output_file,
            index=False,
        )

        print(
            f"\nGovernance dataset saved:"
            f"\n{output_file}"
        )

        print(
            f"\nShape: {df.shape}"
        )

        return output_file

    # ------------------------------------------------------------------
    # DOWNLOAD / BUILD
    # ------------------------------------------------------------------

    def process(self):

        df = self.build_dataset()

        self.save(df)

        return df