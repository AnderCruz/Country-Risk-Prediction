import argparse
import json
import random
import subprocess
import tempfile
from pathlib import Path

import pandas as pd


ENDPOINT = "country-risk-prediction-v7-v3"
REGION = "us-east-1"

REFERENCE_DATASET = Path(
    "monitoring/drift/reference/reference_dataset.csv"
)

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

DEFAULT_REQUESTS = 100
ROWS_PER_REQUEST = 3


def load_reference():
    reference = pd.read_csv(
        REFERENCE_DATASET
    )

    missing = [
        feature
        for feature in FEATURES
        if feature not in reference.columns
    ]

    if missing:
        raise ValueError(
            f"Missing reference features: {missing}"
        )

    return reference[FEATURES].copy()


def build_payload(
    reference,
    scenario,
    rng,
):
    """
    Build controlled traffic from the real
    reference distribution.

    baseline:
        Bootstrap sample from reference.

    moderate:
        Small controlled shifts designed to
        generate WARNING-level drift.

    severe:
        Strong controlled shifts designed to
        generate SIGNIFICANT_DRIFT.
    """

    sampled = reference.sample(
        n=ROWS_PER_REQUEST,
        replace=True,
        random_state=rng.randint(
            0,
            2**32 - 1,
        ),
    ).copy()

    if scenario == "baseline":

        # Preserve reference distribution.
        pass

    elif scenario == "moderate":

        # Small multiplicative shifts.
        # Avoid population_growth because its
        # reference distribution contains extreme
        # outliers and produces disproportionate PSI.

        sampled["gdp_per_capita"] *= rng.uniform(
            0.90,
            0.95,
        )

        sampled["inflation"] *= rng.uniform(
            1.05,
            1.10,
        )

        sampled["unemployment"] *= rng.uniform(
            1.05,
            1.10,
        )

        sampled["population"] *= rng.uniform(
            0.95,
            1.05,
        )

    elif scenario == "severe":

        sampled["gdp_per_capita"] *= rng.uniform(
            0.45,
            0.60,
        )

        sampled["inflation"] *= rng.uniform(
            1.40,
            1.70,
        )

        sampled["unemployment"] *= rng.uniform(
            1.50,
            2.00,
        )

        sampled["population"] *= rng.uniform(
            1.40,
            1.80,
        )

        # Strong shift on population growth.
        sampled["population_growth"] += rng.uniform(
            20.0,
            40.0,
        )

    else:
        raise ValueError(
            f"Unsupported scenario: {scenario}"
        )

    return {
        "dataframe_split": {
            "columns": FEATURES,
            "data": sampled[
                FEATURES
            ].values.tolist(),
        }
    }


def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Generate controlled traffic for "
            "Country Risk Prediction monitoring."
        )
    )

    parser.add_argument(
        "--scenario",
        choices=[
            "baseline",
            "moderate",
            "severe",
        ],
        required=True,
        help="Traffic scenario.",
    )

    parser.add_argument(
        "--requests",
        type=int,
        default=DEFAULT_REQUESTS,
        help="Number of endpoint requests.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    return parser.parse_args()


def main():

    args = parse_arguments()

    reference = load_reference()

    rng = random.Random(
        args.seed
    )

    print("=" * 70)
    print(
        "GENERATING CONTROLLED SAGEMAKER TRAFFIC"
    )
    print("=" * 70)

    print(
        f"Endpoint          : {ENDPOINT}"
    )

    print(
        f"Scenario          : {args.scenario}"
    )

    print(
        f"Requests          : {args.requests}"
    )

    print(
        f"Rows per request  : {ROWS_PER_REQUEST}"
    )

    print(
        f"Expected rows     : "
        f"{args.requests * ROWS_PER_REQUEST}"
    )

    print(
        f"Reference rows    : {len(reference)}"
    )

    print(
        f"Random seed       : {args.seed}"
    )

    successful = 0
    generated_rows = 0

    with tempfile.TemporaryDirectory() as tmp:

        tmp = Path(tmp)

        for i in range(
            1,
            args.requests + 1,
        ):

            payload = build_payload(
                reference,
                args.scenario,
                rng,
            )

            generated_rows += len(
                payload[
                    "dataframe_split"
                ]["data"]
            )

            input_file = (
                tmp / f"request-{i:03d}.json"
            )

            output_file = (
                tmp / f"response-{i:03d}.json"
            )

            with open(
                input_file,
                "w",
            ) as f:
                json.dump(
                    payload,
                    f,
                )

            result = subprocess.run(
                [
                    "aws",
                    "sagemaker-runtime",
                    "invoke-endpoint",
                    "--endpoint-name",
                    ENDPOINT,
                    "--region",
                    REGION,
                    "--content-type",
                    "application/json",
                    "--body",
                    f"fileb://{input_file}",
                    str(output_file),
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                successful += 1
            else:
                print(
                    f"Request {i} failed:"
                )
                print(result.stderr)

            if i % 10 == 0:
                print(
                    f"Progress: "
                    f"{i}/{args.requests}"
                )

    print("\n" + "=" * 70)
    print(
        "TRAFFIC GENERATION COMPLETE"
    )
    print("=" * 70)

    print(
        f"Successful requests: "
        f"{successful}/{args.requests}"
    )

    print(
        f"Generated rows: "
        f"{generated_rows}"
    )


if __name__ == "__main__":
    main()
