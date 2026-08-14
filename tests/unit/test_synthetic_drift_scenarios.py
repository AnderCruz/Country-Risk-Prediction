import random

import pandas as pd

from monitoring.drift.calculate_drift import (
    calculate_data_drift,
    determine_overall_status,
)
from monitoring.drift.generate_test_traffic import (
    build_payload,
    load_reference,
)


def generate_synthetic_dataset(
    reference: pd.DataFrame,
    scenario: str,
    seed: int = 42,
    requests: int = 100,
) -> pd.DataFrame:

    rng = random.Random(seed)

    rows = []

    for _ in range(requests):

        payload = build_payload(
            reference,
            scenario,
            rng,
        )

        dataframe = payload["dataframe_split"]

        rows.extend(
            dataframe["data"]
        )

    return pd.DataFrame(
        rows,
        columns=dataframe["columns"],
    )


def test_baseline_scenario_is_stable():

    reference = load_reference()

    production = generate_synthetic_dataset(
        reference,
        "baseline",
    )

    drift = calculate_data_drift(
        reference,
        production,
    )

    significant = (
        drift["status"] == "SIGNIFICANT_DRIFT"
    ).sum()

    warnings = (
        drift["status"] == "WARNING"
    ).sum()

    assert significant == 0
    assert warnings == 0

    status = determine_overall_status(
        drift,
        {"status": "STABLE"},
    )

    assert status == "STABLE"


def test_moderate_scenario_is_warning():

    reference = load_reference()

    production = generate_synthetic_dataset(
        reference,
        "moderate",
    )

    drift = calculate_data_drift(
        reference,
        production,
    )

    significant = (
        drift["status"] == "SIGNIFICANT_DRIFT"
    ).sum()

    warnings = (
        drift["status"] == "WARNING"
    ).sum()

    assert significant == 0
    assert warnings > 0

    status = determine_overall_status(
        drift,
        {"status": "STABLE"},
    )

    assert status == "WARNING"


def test_severe_scenario_is_critical():

    reference = load_reference()

    production = generate_synthetic_dataset(
        reference,
        "severe",
    )

    drift = calculate_data_drift(
        reference,
        production,
    )

    significant = (
        drift["status"] == "SIGNIFICANT_DRIFT"
    ).sum()

    assert significant > 0

    status = determine_overall_status(
        drift,
        {"status": "STABLE"},
    )

    assert status == "CRITICAL"
