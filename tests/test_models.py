import pandas as pd
import pytest
import mlflow

from models.baseline import evaluate_naive_risk_baseline
from models.train import train_model
from sklearn.ensemble import RandomForestRegressor
from models.evaluate import evaluate_model

def test_naive_risk_baseline_returns_metrics():
    df = pd.DataFrame(
        {
            "country": [
                "Brazil", "Brazil", "Brazil", "Brazil", "Brazil",
                "Argentina", "Argentina", "Argentina", "Argentina", "Argentina",
            ],
            "date": [
                2018, 2019, 2020, 2021, 2022,
                2018, 2019, 2020, 2021, 2022,
            ],
            "country_risk_index": [
                10.0, 20.0, 30.0, 40.0, 50.0,
                15.0, 25.0, 35.0, 45.0, 55.0,
            ],
            "future_country_risk": [
                20.0, 30.0, 40.0, 50.0, 60.0,
                25.0, 35.0, 45.0, 55.0, 65.0,
            ],
        }
    )

    result = evaluate_naive_risk_baseline(df)

    assert isinstance(result, dict)

    assert "mae" in result
    assert "rmse" in result
    assert "r2" in result

    assert result["mae"] >= 0
    assert result["rmse"] >= 0


def test_naive_risk_baseline_uses_previous_year_risk():
    df = pd.DataFrame(
        {
            "country": [
                "Brazil", "Brazil", "Brazil", "Brazil", "Brazil",
            ],
            "date": [2018, 2019, 2020, 2021, 2022],
            "country_risk_index": [
                10.0, 20.0, 30.0, 40.0, 50.0,
            ],
            "future_country_risk": [
                20.0, 30.0, 40.0, 50.0, 60.0,
            ],
        }
    )

    result = evaluate_naive_risk_baseline(df)

    assert result["mae"] == pytest.approx(20.0)
    assert result["rmse"] == pytest.approx(20.0)


def test_naive_risk_baseline_is_calculated_per_country():
    df = pd.DataFrame(
        {
            "country": [
                "Brazil", "Brazil", "Brazil", "Brazil", "Brazil",
                "Argentina", "Argentina", "Argentina", "Argentina", "Argentina",
            ],
            "date": [
                2018, 2019, 2020, 2021, 2022,
                2018, 2019, 2020, 2021, 2022,
            ],
            "country_risk_index": [
                10.0, 20.0, 30.0, 40.0, 50.0,
                100.0, 110.0, 120.0, 130.0, 140.0,
            ],
            "future_country_risk": [
                20.0, 30.0, 40.0, 50.0, 60.0,
                110.0, 120.0, 130.0, 140.0, 150.0,
            ],
        }
    )

    result = evaluate_naive_risk_baseline(df)

    assert result["mae"] == pytest.approx(20.0)
    assert result["rmse"] == pytest.approx(20.0)


def test_train_model_returns_expected_outputs(tmp_path, monkeypatch):
    df = pd.DataFrame(
        {
            "date": [2018, 2019, 2020, 2021, 2022, 2023],
            "feature_1": [1, 2, 3, 4, 5, 6],
            "feature_2": [10, 20, 30, 40, 50, 60],
            "target": [2, 4, 6, 8, 10, 12],
        }
    )

    import models.train as train_module

    monkeypatch.setattr(
        train_module,
        "MODEL_DIR",
        tmp_path,
    )

    model, X_test, y_test, predictions = train_model(
        df,
        ["feature_1", "feature_2"],
        "target",
    )

    assert isinstance(model, RandomForestRegressor)
    assert len(X_test) == len(y_test)
    assert len(y_test) == len(predictions)


def test_train_model_saves_model(tmp_path, monkeypatch):
    df = pd.DataFrame(
        {
            "date": [2018, 2019, 2020, 2021, 2022, 2023],
            "feature_1": [1, 2, 3, 4, 5, 6],
            "feature_2": [10, 20, 30, 40, 50, 60],
            "target": [2, 4, 6, 8, 10, 12],
        }
    )

    import models.train as train_module

    monkeypatch.setattr(
        train_module,
        "MODEL_DIR",
        tmp_path,
    )

    train_model(
        df,
        ["feature_1", "feature_2"],
        "target",
    )

    model_path = tmp_path / "random_forest.pkl"

    assert model_path.exists()


def test_train_model_saves_model(tmp_path, monkeypatch):
    df = pd.DataFrame(
        {
            "date": [2018, 2019, 2020, 2021, 2022, 2023],
            "feature_1": [1, 2, 3, 4, 5, 6],
            "feature_2": [10, 20, 30, 40, 50, 60],
            "target": [2, 4, 6, 8, 10, 12],
        }
    )

    import models.train as train_module

    monkeypatch.setattr(
        train_module,
        "MODEL_DIR",
        tmp_path,
    )

    train_model(
        df,
        ["feature_1", "feature_2"],
        "target",
    )

    model_path = tmp_path / "random_forest.pkl"

    assert model_path.exists()


def test_train_model_requires_at_least_two_years(tmp_path, monkeypatch):
    df = pd.DataFrame(
        {
            "date": [2020],
            "feature_1": [1],
            "feature_2": [10],
            "target": [2],
        }
    )

    import models.train as train_module

    monkeypatch.setattr(
        train_module,
        "MODEL_DIR",
        tmp_path,
    )

    with pytest.raises(
        ValueError,
        match="Not enough years available",
    ):
        train_model(
            df,
            ["feature_1", "feature_2"],
            "target",
        )


def test_evaluate_model_calculates_metrics():
    y_true = [10.0, 20.0, 30.0]
    y_pred = [12.0, 18.0, 33.0]

    result = evaluate_model(
        y_true,
        y_pred,
    )

    assert result["mae"] == pytest.approx(2.3333333333)
    assert result["rmse"] == pytest.approx(2.3804761428)
    assert result["r2"] == pytest.approx(0.915)


def test_evaluate_model_returns_expected_metrics():
    y_true = [10.0, 20.0, 30.0]
    y_pred = [10.0, 20.0, 30.0]

    result = evaluate_model(
        y_true,
        y_pred,
    )

    assert set(result.keys()) == {
        "mae",
        "rmse",
        "r2",
    }

    assert result["mae"] == pytest.approx(0.0)
    assert result["rmse"] == pytest.approx(0.0)
    assert result["r2"] == pytest.approx(1.0)


def test_evaluate_model_logs_metrics_to_mlflow():
    with mlflow.start_run():
        result = evaluate_model(
            [10.0, 20.0, 30.0],
            [12.0, 18.0, 33.0],
        )

        run = mlflow.active_run()

        metrics = mlflow.get_run(
            run.info.run_id
        ).data.metrics

    assert metrics["mae"] == pytest.approx(
        result["mae"]
    )

    assert metrics["rmse"] == pytest.approx(
        result["rmse"]
    )

    assert metrics["r2"] == pytest.approx(
        result["r2"]
    )


