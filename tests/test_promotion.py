import pytest

from ml.promotion import (
    CHAMPION_ALIAS,
    promote_model,
)


# =============================================================================
# TEST HELPERS
# =============================================================================


class FakeRunData:

    def __init__(self, tags):

        self.tags = tags


class FakeRun:

    def __init__(self, tags):

        self.data = FakeRunData(tags)


class FakeModelVersion:

    def __init__(
        self,
        version="7",
        run_id="run-123",
    ):

        self.name = "country-risk-prediction-model"
        self.version = version
        self.run_id = run_id


class FakeMlflowClient:

    def __init__(
        self,
        validation_status="passed",
        version="7",
        run_id="run-123",
    ):

        self.validation_status = (
            validation_status
        )

        self.version = version
        self.run_id = run_id

        self.alias_calls = []

    def get_model_version(
        self,
        name,
        version,
    ):

        return FakeModelVersion(
            version=version,
            run_id=self.run_id,
        )

    def get_run(
        self,
        run_id,
    ):

        return FakeRun(
            {
                "validation_status":
                    self.validation_status
            }
        )

    def set_registered_model_alias(
        self,
        name,
        alias,
        version,
    ):

        self.alias_calls.append(
            {
                "name": name,
                "alias": alias,
                "version": version,
            }
        )


# =============================================================================
# PROMOTION TESTS
# =============================================================================


def test_promote_validated_model(
    monkeypatch,
):

    client = FakeMlflowClient(
        validation_status="passed",
        version="7",
    )

    monkeypatch.setattr(
        "ml.promotion.mlflow.MlflowClient",
        lambda: client,
    )

    result = promote_model(
        model_name="country-risk-prediction-model",
        model_version="7",
    )

    assert result.version == "7"

    assert len(
        client.alias_calls
    ) == 1

    assert client.alias_calls[0] == {
        "name":
            "country-risk-prediction-model",
        "alias":
            CHAMPION_ALIAS,
        "version":
            "7",
    }


def test_failed_validation_blocks_promotion(
    monkeypatch,
):

    client = FakeMlflowClient(
        validation_status="failed",
        version="7",
    )

    monkeypatch.setattr(
        "ml.promotion.mlflow.MlflowClient",
        lambda: client,
    )

    with pytest.raises(
        ValueError,
        match="Model validation failed",
    ):

        promote_model(
            model_name="country-risk-prediction-model",
            model_version="7",
        )

    assert (
        client.alias_calls
        == []
    )


def test_missing_validation_status_blocks_promotion(
    monkeypatch,
):

    client = FakeMlflowClient(
        validation_status=None,
        version="7",
    )

    monkeypatch.setattr(
        "ml.promotion.mlflow.MlflowClient",
        lambda: client,
    )

    with pytest.raises(
        ValueError,
        match="Model validation failed",
    ):

        promote_model(
            model_name="country-risk-prediction-model",
            model_version="7",
        )

    assert (
        client.alias_calls
        == []
    )


def test_missing_model_version(
):

    with pytest.raises(
        ValueError,
        match="model_version must be provided",
    ):

        promote_model(
            model_name="country-risk-prediction-model",
            model_version=None,
        )


def test_model_without_run_blocks_promotion(
    monkeypatch,
):

    client = FakeMlflowClient(
        validation_status="passed",
        version="7",
        run_id=None,
    )

    monkeypatch.setattr(
        "ml.promotion.mlflow.MlflowClient",
        lambda: client,
    )

    with pytest.raises(
        ValueError,
        match="no associated run",
    ):

        promote_model(
            model_name="country-risk-prediction-model",
            model_version="7",
        )

    assert (
        client.alias_calls
        == []
    )