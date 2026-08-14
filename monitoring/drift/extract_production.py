import argparse
import base64
import json
from pathlib import Path

import pandas as pd


FEATURES = [
    "gdp_per_capita",
    "inflation",
    "life_expectancy",
    "population",
    "population_growth",
    "unemployment",
    "exports",
    "gdp_lag1",
    "inflation_lag1",
    "life_expectancy_lag1",
    "economic_risk",
    "governance_risk",
]


def decode_capture_data(data: str, encoding: str) -> dict:
    """
    Decode a SageMaker Data Capture payload.
    """

    if encoding.upper() != "BASE64":
        raise ValueError(
            f"Unsupported capture encoding: {encoding}"
        )

    decoded = base64.b64decode(data).decode("utf-8")

    return json.loads(decoded)


def extract_record(record: dict) -> pd.DataFrame:
    """
    Extract feature rows and predictions from one
    SageMaker Data Capture record.
    """

    capture_data = record["captureData"]

    input_data = capture_data["endpointInput"]

    output_data = capture_data.get("endpointOutput")

    metadata = record.get("eventMetadata", {})

    input_payload = decode_capture_data(
        input_data["data"],
        input_data["encoding"],
    )

    dataframe_split = input_payload["dataframe_split"]

    columns = dataframe_split["columns"]
    rows = dataframe_split["data"]

    dataframe = pd.DataFrame(
        rows,
        columns=columns,
    )

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in dataframe.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing expected features: "
            + ", ".join(missing_features)
        )

    dataframe = dataframe[FEATURES].copy()

    predictions = []

    if output_data is not None:

        output_payload = decode_capture_data(
            output_data["data"],
            output_data["encoding"],
        )

        predictions = output_payload.get(
            "predictions",
            [],
        )

    if predictions:

        if len(predictions) != len(dataframe):

            raise ValueError(
                "Number of predictions does not match "
                "number of input rows: "
                f"{len(predictions)} predictions vs "
                f"{len(dataframe)} rows."
            )

        dataframe["prediction"] = predictions

    else:

        dataframe["prediction"] = pd.NA

    dataframe["event_id"] = metadata.get(
        "eventId"
    )

    dataframe["inference_time"] = metadata.get(
        "inferenceTime"
    )

    return dataframe


def extract_production_data(
    input_path: Path,
) -> pd.DataFrame:
    """
    Read one or more SageMaker Data Capture JSONL files.
    """

    if input_path.is_file():

        files = [input_path]

    elif input_path.is_dir():

        files = sorted(
            input_path.rglob("*.jsonl")
        )

    else:

        raise FileNotFoundError(
            f"Input path does not exist: {input_path}"
        )

    if not files:

        raise FileNotFoundError(
            f"No JSONL files found under: {input_path}"
        )

    records = []

    for file_path in files:

        with file_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            for line_number, line in enumerate(
                file,
                start=1,
            ):

                line = line.strip()

                if not line:
                    continue

                try:

                    record = json.loads(line)

                    dataframe = extract_record(
                        record
                    )

                    records.append(dataframe)

                except Exception as exc:

                    raise ValueError(
                        f"Failed to process "
                        f"{file_path}:"
                        f"{line_number}: "
                        f"{exc}"
                    ) from exc

    if not records:

        raise ValueError(
            "No valid production records found."
        )

    production = pd.concat(
        records,
        ignore_index=True,
    )

    return production


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Extract production data from "
            "SageMaker Data Capture JSONL files."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "JSONL file or directory containing "
            "SageMaker Data Capture files."
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV path.",
    )

    args = parser.parse_args()

    input_path = Path(
        args.input
    )

    output_path = Path(
        args.output
    )

    production = extract_production_data(
        input_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    production.to_csv(
        output_path,
        index=False,
    )

    print("=" * 70)
    print("PRODUCTION DATA EXTRACTION")
    print("=" * 70)

    print(
        f"Input : {input_path}"
    )

    print(
        f"Output: {output_path}"
    )

    print(
        f"Rows  : {len(production)}"
    )

    print(
        f"Columns: {len(production.columns)}"
    )

    print("\nColumns:")

    for column in production.columns:

        print(
            f"- {column}"
        )

    print("\nProduction data:")

    print(
        production.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
