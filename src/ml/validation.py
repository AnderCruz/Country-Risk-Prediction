def validate_model(
    model_metrics: dict,
    baseline_metrics: dict,
) -> dict:
    """
    Validate a machine learning model against a naive baseline.

    The model passes validation only when it improves all three
    evaluation metrics:

        MAE  < baseline MAE
        RMSE < baseline RMSE
        R²   > baseline R²

    Parameters
    ----------
    model_metrics:
        Dictionary containing model MAE, RMSE and R².

    baseline_metrics:
        Dictionary containing baseline MAE, RMSE and R².

    Returns
    -------
    dict
        Validation status and metric improvements.
    """

    required_metrics = {
        "mae",
        "rmse",
        "r2",
    }

    if not required_metrics.issubset(model_metrics):
        missing = required_metrics - set(model_metrics)

        raise ValueError(
            f"Missing model metrics: {sorted(missing)}"
        )

    if not required_metrics.issubset(baseline_metrics):
        missing = required_metrics - set(baseline_metrics)

        raise ValueError(
            f"Missing baseline metrics: {sorted(missing)}"
        )

    mae_improvement = (
        baseline_metrics["mae"]
        - model_metrics["mae"]
    )

    rmse_improvement = (
        baseline_metrics["rmse"]
        - model_metrics["rmse"]
    )

    r2_improvement = (
        model_metrics["r2"]
        - baseline_metrics["r2"]
    )

    mae_passed = (
        model_metrics["mae"]
        < baseline_metrics["mae"]
    )

    rmse_passed = (
        model_metrics["rmse"]
        < baseline_metrics["rmse"]
    )

    r2_passed = (
        model_metrics["r2"]
        > baseline_metrics["r2"]
    )

    passed = (
        mae_passed
        and rmse_passed
        and r2_passed
    )

    return {
        "passed": passed,
        "mae_improvement": mae_improvement,
        "rmse_improvement": rmse_improvement,
        "r2_improvement": r2_improvement,
        "mae_passed": mae_passed,
        "rmse_passed": rmse_passed,
        "r2_passed": r2_passed,
    }
