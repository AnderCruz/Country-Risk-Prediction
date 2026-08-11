import pandas as pd
import pytest

from ml.performance import analyse_model_performance


def test_analyse_model_performance_returns_expected_structure():

    df = pd.DataFrame(
        {
            "country": [
                "Brazil",
                "Brazil",
                "Argentina",
                "Argentina",
            ],
            "date": [
                2020,
                2021,
                2020,
                2021,
            ],
            "future_country_risk": [
                10.0,
                20.0,
                30.0,
                40.0,
            ],
            "prediction": [
                11.0,
                19.0,
                32.0,
                38.0,
            ],
        }
    )

    result = analyse_model_performance(
        df,
        "future_country_risk",
        "prediction",
    )

    assert isinstance(result, dict)

    assert "global" in result
    assert "yearly" in result
    assert "country" in result

    assert isinstance(
        result["global"],
        dict,
    )

    assert isinstance(
        result["yearly"],
        pd.DataFrame,
    )

    assert isinstance(
        result["country"],
        pd.DataFrame,
    )


def test_analyse_model_performance_calculates_global_metrics():

    df = pd.DataFrame(
        {
            "country": [
                "Brazil",
                "Brazil",
                "Argentina",
                "Argentina",
            ],
            "date": [
                2020,
                2021,
                2020,
                2021,
            ],
            "target": [
                10.0,
                20.0,
                30.0,
                40.0,
            ],
            "prediction": [
                11.0,
                19.0,
                32.0,
                38.0,
            ],
        }
    )

    result = analyse_model_performance(
        df,
        "target",
        "prediction",
    )

    metrics = result["global"]

    assert metrics["mae"] == pytest.approx(
        1.5
    )

    assert metrics["rmse"] == pytest.approx(
        1.58113883
    )


def test_analyse_model_performance_calculates_yearly_metrics():

    df = pd.DataFrame(
        {
            "country": [
                "Brazil",
                "Brazil",
                "Argentina",
                "Argentina",
            ],
            "date": [
                2020,
                2020,
                2021,
                2021,
            ],
            "target": [
                10.0,
                20.0,
                30.0,
                40.0,
            ],
            "prediction": [
                11.0,
                18.0,
                32.0,
                39.0,
            ],
        }
    )

    result = analyse_model_performance(
        df,
        "target",
        "prediction",
    )

    yearly = result["yearly"]

    assert len(yearly) == 2

    assert set(yearly["date"]) == {
        2020,
        2021,
    }

    assert all(
        yearly["n_observations"] == 2
    )


def test_analyse_model_performance_calculates_country_metrics():

    df = pd.DataFrame(
        {
            "country": [
                "Brazil",
                "Brazil",
                "Argentina",
                "Argentina",
            ],
            "date": [
                2020,
                2021,
                2020,
                2021,
            ],
            "target": [
                10.0,
                20.0,
                30.0,
                40.0,
            ],
            "prediction": [
                11.0,
                18.0,
                32.0,
                39.0,
            ],
        }
    )

    result = analyse_model_performance(
        df,
        "target",
        "prediction",
    )

    country = result["country"]

    assert len(country) == 2

    assert set(country["country"]) == {
        "Brazil",
        "Argentina",
    }

    assert all(
        country["n_observations"] == 2
    )


def test_analyse_model_performance_rejects_missing_columns():

    df = pd.DataFrame(
        {
            "country": ["Brazil"],
            "date": [2020],
            "target": [10.0],
        }
    )

    with pytest.raises(ValueError):

        analyse_model_performance(
            df,
            "target",
            "prediction",
        )


def test_analyse_model_performance_rejects_empty_data():

    df = pd.DataFrame(
        {
            "country": [],
            "date": [],
            "target": [],
            "prediction": [],
        }
    )

    with pytest.raises(ValueError):

        analyse_model_performance(
            df,
            "target",
            "prediction",
        )


from ml.performance import build_performance_dataset


def test_build_performance_dataset_preserves_test_metadata():

    source_df = pd.DataFrame(
        {
            "country": [
                "Brazil",
                "Brazil",
                "Argentina",
                "Argentina",
            ],
            "date": [
                2020,
                2021,
                2020,
                2021,
            ],
            "target": [
                10.0,
                20.0,
                30.0,
                40.0,
            ],
        }
    )

    X_test = pd.DataFrame(
        {
            "feature_1": [
                1.0,
                2.0,
            ],
        },
        index=[1, 3],
    )

    y_test = pd.Series(
        [20.0, 40.0],
        index=[1, 3],
    )

    predictions = [
        19.0,
        41.0,
    ]

    result = build_performance_dataset(
        X_test,
        y_test,
        predictions,
        source_df,
    )

    assert len(result) == 2

    assert list(result["country"]) == [
        "Brazil",
        "Argentina",
    ]

    assert list(result["date"]) == [
        2021,
        2021,
    ]

    assert list(result["actual"]) == [
        20.0,
        40.0,
    ]

    assert list(result["prediction"]) == [
        19.0,
        41.0,
    ]


def test_build_performance_dataset_rejects_mismatched_lengths():

    source_df = pd.DataFrame(
        {
            "country": [
                "Brazil",
                "Brazil",
            ],
            "date": [
                2020,
                2021,
            ],
        }
    )

    X_test = pd.DataFrame(
        {
            "feature_1": [1.0, 2.0],
        },
        index=[0, 1],
    )

    y_test = pd.Series(
        [10.0, 20.0],
    )

    predictions = [
        11.0,
    ]

    with pytest.raises(ValueError):

        build_performance_dataset(
            X_test,
            y_test,
            predictions,
            source_df,
        )


def test_build_performance_dataset_rejects_mismatched_X_test_and_y_test():

    source_df = pd.DataFrame(
        {
            "country": [
                "Brazil",
                "Brazil",
            ],
            "date": [
                2020,
                2021,
            ],
        }
    )

    X_test = pd.DataFrame(
        {
            "feature_1": [1.0],
        },
        index=[0],
    )

    y_test = pd.Series(
        [10.0, 20.0],
        index=[0, 1],
    )

    predictions = [
        10.0,
        20.0,
    ]

    with pytest.raises(ValueError):

        build_performance_dataset(
            X_test,
            y_test,
            predictions,
            source_df,
        )


def test_build_performance_dataset_rejects_mismatched_y_test_and_predictions():

    source_df = pd.DataFrame(
        {
            "country": [
                "Brazil",
                "Brazil",
            ],
            "date": [
                2020,
                2021,
            ],
        }
    )

    X_test = pd.DataFrame(
        {
            "feature_1": [
                1.0,
                2.0,
            ],
        },
        index=[0, 1],
    )

    y_test = pd.Series(
        [10.0, 20.0],
        index=[0, 1],
    )

    predictions = [
        10.0,
    ]

    with pytest.raises(ValueError):

        build_performance_dataset(
            X_test,
            y_test,
            predictions,
            source_df,
        )