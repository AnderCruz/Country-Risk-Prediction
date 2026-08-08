import pandas as pd


class DataValidator:

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe

    def check_shape(self):

        print("\n" + "=" * 60)
        print("DATASET SHAPE")
        print("=" * 60)

        print(f"Rows: {self.df.shape[0]}")
        print(f"Columns: {self.df.shape[1]}")

    def check_types(self):

        print("\n" + "=" * 60)
        print("COLUMN TYPES")
        print("=" * 60)

        print(self.df.dtypes)

    def check_missing(self):

        print("\n" + "=" * 60)
        print("MISSING VALUES")
        print("=" * 60)

        print(self.df.isna().sum())

    def check_duplicates(self):

        print("\n" + "=" * 60)
        print("DUPLICATED ROWS")
        print("=" * 60)

        print(self.df.duplicated().sum())

    def check_countries(self):

        print("\n" + "=" * 60)
        print("COUNTRIES")
        print("=" * 60)

        print(self.df["country"].nunique())

    def check_period(self):

        print("\n" + "=" * 60)
        print("TIME PERIOD")
        print("=" * 60)

        print(f"From {self.df['date'].min()} to {self.df['date'].max()}")

    def run(self):

        self.check_shape()

        self.check_types()

        self.check_missing()

        self.check_duplicates()

        self.check_countries()

        self.check_period()