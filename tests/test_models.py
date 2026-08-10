import pandas as pd
import pytest
import mlflow

from models.baseline import evaluate_naive_risk_baseline
from models.train import train_model
from sklearn.ensemble import RandomForestRegressor
from models.evaluate import evaluate_model
from models.experiments import run_experiments
from models.importance import feature_importance_report

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


def test_run_experiments_runs_all_experiments(monkeypatch, tmp_path):
    import models.experiments as experiments_module

    monkeypatch.setattr(
        experiments_module,
        "REPORT_DIR",
        tmp_path,
    )

    train_calls = []
    evaluate_calls = []

    class DummyRun:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    monkeypatch.setattr(
        experiments_module.mlflow,
        "start_run",
        lambda run_name: DummyRun(),
    )

    monkeypatch.setattr(
        experiments_module.mlflow,
        "log_param",
        lambda *args, **kwargs: None,
    )

    def fake_train_model(df, features, target_column):
        train_calls.append(
            {
                "features": features,
                "target": target_column,
            }
        )

        return (
            object(),
            pd.DataFrame(),
            pd.Series([1.0]),
            pd.Series([1.0]),
        )

    def fake_evaluate_model(y_test, predictions):
        evaluate_calls.append(True)

        return {
            "mae": 1.0,
            "rmse": 2.0,
            "r2": 0.5,
        }

    monkeypatch.setattr(
        experiments_module,
        "train_model",
        fake_train_model,
    )

    monkeypatch.setattr(
        experiments_module,
        "evaluate_model",
        fake_evaluate_model,
    )

    df = pd.DataFrame({"date": [2020]})

    result = run_experiments(
        df,
        "future_country_risk",
    )

    assert len(train_calls) == 5
    assert len(evaluate_calls) == 5
    assert len(result) == 5


def test_run_experiments_returns_expected_experiment_names(
    monkeypatch,
    tmp_path,
):
    import models.experiments as experiments_module

    monkeypatch.setattr(
        experiments_module,
        "REPORT_DIR",
        tmp_path,
    )

    class DummyRun:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    monkeypatch.setattr(
        experiments_module.mlflow,
        "start_run",
        lambda run_name: DummyRun(),
    )

    monkeypatch.setattr(
        experiments_module.mlflow,
        "log_param",
        lambda *args, **kwargs: None,
    )

    monkeypatch.setattr(
        experiments_module,
        "train_model",
        lambda df, features, target_column: (
            object(),
            pd.DataFrame(),
            pd.Series([1.0]),
            pd.Series([1.0]),
        ),
    )

    monkeypatch.setattr(
        experiments_module,
        "evaluate_model",
        lambda y_test, predictions: {
            "mae": 1.0,
            "rmse": 2.0,
            "r2": 0.5,
        },
    )

    result = run_experiments(
        pd.DataFrame({"date": [2020]}),
        "future_country_risk",
    )

    expected = [
        "Baseline",
        "Baseline + Lag",
        "Baseline + Lag + Economic Risk",
        "Baseline + Lag + Economic Risk PCA",
        "Full Risk Model",
    ]

    assert result["experiment"].tolist() == expected


def test_run_experiments_rounds_metrics(
    monkeypatch,
    tmp_path,
):
    import models.experiments as experiments_module

    monkeypatch.setattr(
        experiments_module,
        "REPORT_DIR",
        tmp_path,
    )

    class DummyRun:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    monkeypatch.setattr(
        experiments_module.mlflow,
        "start_run",
        lambda run_name: DummyRun(),
    )

    monkeypatch.setattr(
        experiments_module.mlflow,
        "log_param",
        lambda *args, **kwargs: None,
    )

    monkeypatch.setattr(
        experiments_module,
        "train_model",
        lambda df, features, target_column: (
            object(),
            pd.DataFrame(),
            pd.Series([1.0]),
            pd.Series([1.0]),
        ),
    )

    monkeypatch.setattr(
        experiments_module,
        "evaluate_model",
        lambda y_test, predictions: {
            "mae": 1.234567,
            "rmse": 2.345678,
            "r2": 0.987654,
        },
    )

    result = run_experiments(
        pd.DataFrame({"date": [2020]}),
        "future_country_risk",
    )

    assert result["mae"].iloc[0] == 1.2346
    assert result["rmse"].iloc[0] == 2.3457
    assert result["r2"].iloc[0] == 0.9877


def test_run_experiments_saves_results_csv(
    monkeypatch,
    tmp_path,
):
    import models.experiments as experiments_module

    monkeypatch.setattr(
        experiments_module,
        "REPORT_DIR",
        tmp_path,
    )

    class DummyRun:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    monkeypatch.setattr(
        experiments_module.mlflow,
        "start_run",
        lambda run_name: DummyRun(),
    )

    monkeypatch.setattr(
        experiments_module.mlflow,
        "log_param",
        lambda *args, **kwargs: None,
    )

    monkeypatch.setattr(
        experiments_module,
        "train_model",
        lambda df, features, target_column: (
            object(),
            pd.DataFrame(),
            pd.Series([1.0]),
            pd.Series([1.0]),
        ),
    )

    monkeypatch.setattr(
        experiments_module,
        "evaluate_model",
        lambda y_test, predictions: {
            "mae": 1.0,
            "rmse": 2.0,
            "r2": 0.5,
        },
    )

    run_experiments(
        pd.DataFrame({"date": [2020]}),
        "future_country_risk",
    )

    output = tmp_path / "experiments.csv"

    assert output.exists()

    saved = pd.read_csv(output)

    assert len(saved) == 5
    assert list(saved.columns) == [
        "experiment",
        "n_features",
        "mae",
        "rmse",
        "r2",
    ]


def test_feature_importance_report_sorts_by_importance(
    tmp_path,
    monkeypatch,
):
    import models.importance as importance_module

    monkeypatch.setattr(
        importance_module,
        "REPORTS_DIR",
        tmp_path,
    )

    class DummyModel:
        feature_importances_ = [0.2, 0.7, 0.1]

    feature_names = [
        "feature_a",
        "feature_b",
        "feature_c",
    ]

    feature_importance_report(
        DummyModel(),
        feature_names,
    )

    result = pd.read_csv(
        tmp_path / "feature_importance.csv"
    )

    assert result["feature"].tolist() == [
        "feature_b",
        "feature_a",
        "feature_c",
    ]

    assert result["importance"].tolist() == pytest.approx(
        [0.7, 0.2, 0.1]
    )


def test_feature_importance_report_contains_all_features(
    tmp_path,
    monkeypatch,
):
    import models.importance as importance_module

    monkeypatch.setattr(
        importance_module,
        "REPORTS_DIR",
        tmp_path,
    )

    class DummyModel:
        feature_importances_ = [0.4, 0.3, 0.2, 0.1]

    feature_names = [
        "gdp",
        "inflation",
        "exports",
        "population",
    ]

    feature_importance_report(
        DummyModel(),
        feature_names,
    )

    result = pd.read_csv(
        tmp_path / "feature_importance.csv"
    )

    assert len(result) == 4
    assert set(result["feature"]) == set(feature_names)


def test_feature_importance_report_creates_csv(
    tmp_path,
    monkeypatch,
):
    import models.importance as importance_module

    monkeypatch.setattr(
        importance_module,
        "REPORTS_DIR",
        tmp_path,
    )

    class DummyModel:
        feature_importances_ = [0.6, 0.4]

    feature_importance_report(
        DummyModel(),
        ["feature_a", "feature_b"],
    )

    output = tmp_path / "feature_importance.csv"

    assert output.exists()

    result = pd.read_csv(output)

    assert list(result.columns) == [
        "feature",
        "importance",
    ]


