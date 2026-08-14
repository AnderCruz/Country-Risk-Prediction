import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import boto3
import pandas as pd

from monitoring.drift.calculate_drift import (
    build_report,
    calculate_data_drift,
    calculate_prediction_drift,
    determine_overall_status,
)


CLOUDWATCH_NAMESPACE = "CountryRisk/ModelMonitoring"

ENDPOINT_NAME = "country-risk-prediction-v7-v3"


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run Country Risk model monitoring."
    )

    parser.add_argument(
        "--bucket",
        required=True,
    )

    parser.add_argument(
        "--prefix",
        required=True,
    )

    parser.add_argument(
        "--reference",
        required=True,
    )

    parser.add_argument(
        "--reference-predictions",
        required=True,
    )

    parser.add_argument(
        "--region",
        default="us-east-1",
    )

    parser.add_argument(
        "--output-prefix",
        default="monitoring/reports",
    )

    parser.add_argument(
        "--output",
        default="/tmp/monitoring_report.json",
    )

    return parser.parse_args()


def load_production_dataset(
    bucket: str,
    prefix: str,
    region: str,
) -> Path:

    print("\n[1/6] Loading production data from S3...")

    output_path = Path(
        "/tmp/production_dataset_s3.csv"
    )

    command = [
        sys.executable,
        "monitoring/drift/load_production.py",
        "--bucket",
        bucket,
        "--prefix",
        prefix,
        "--region",
        region,
        "--output",
        str(output_path),
    ]

    subprocess.run(
        command,
        check=True,
    )

    if not output_path.exists():
        raise FileNotFoundError(
            f"Production dataset was not created: "
            f"{output_path}"
        )

    production = pd.read_csv(output_path)

    if production.empty:
        raise ValueError(
            "Production dataset is empty."
        )

    print(
        f"Production rows loaded: {len(production)}"
    )

    return output_path


def validate_datasets(
    reference: pd.DataFrame,
    production: pd.DataFrame,
    reference_predictions: pd.DataFrame,
):

    print("\n[2/6] Validating datasets...")

    required_features = [
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

    missing_reference = [
        feature
        for feature in required_features
        if feature not in reference.columns
    ]

    missing_production = [
        feature
        for feature in required_features
        if feature not in production.columns
    ]

    if missing_reference:
        raise ValueError(
            "Missing features in reference dataset: "
            f"{missing_reference}"
        )

    if missing_production:
        raise ValueError(
            "Missing features in production dataset: "
            f"{missing_production}"
        )

    if "prediction" not in reference_predictions.columns:
        raise ValueError(
            "Reference predictions dataset must contain "
            "'prediction' column."
        )

    if "prediction" not in production.columns:
        raise ValueError(
            "Production dataset must contain "
            "'prediction' column."
        )

    print(
        f"Reference rows           : {len(reference)}"
    )

    print(
        f"Production rows          : {len(production)}"
    )

    print(
        f"Reference predictions    : "
        f"{len(reference_predictions)}"
    )

    print("Dataset validation: OK")


def calculate_monitoring(
    reference: pd.DataFrame,
    production: pd.DataFrame,
    reference_predictions: pd.DataFrame,
):

    print("\n[3/6] Calculating data drift...")

    data_drift = calculate_data_drift(
        reference,
        production,
    )

    print(
        f"Data drift features: "
        f"{len(data_drift)}"
    )

    print("\n[4/6] Calculating prediction drift...")

    prediction_drift = calculate_prediction_drift(
        reference_predictions,
        production,
    )

    print(
        f"Prediction drift: "
        f"{prediction_drift['status']}"
    )

    overall_status = determine_overall_status(
        data_drift,
        prediction_drift,
    )

    print(
        f"Overall monitoring status: "
        f"{overall_status}"
    )

    return (
        data_drift,
        prediction_drift,
        overall_status,
    )


def status_to_metric(
    status: str,
) -> int:

    mapping = {
        "STABLE": 0,
        "WARNING": 1,
        "CRITICAL": 2,
    }

    if status not in mapping:
        raise ValueError(
            f"Unknown monitoring status: {status}"
        )

    return mapping[status]


def publish_cloudwatch_metric(
    status: str,
    endpoint: str,
    region: str,
):

    print("\n[5/6] Publishing CloudWatch metric...")

    cloudwatch = boto3.client(
        "cloudwatch",
        region_name=region,
    )

    metric_value = status_to_metric(
        status
    )

    cloudwatch.put_metric_data(
        Namespace=CLOUDWATCH_NAMESPACE,
        MetricData=[
            {
                "MetricName": "MonitoringStatus",
                "Dimensions": [
                    {
                        "Name": "Endpoint",
                        "Value": endpoint,
                    },
                    {
                        "Name": "Environment",
                        "Value": "production",
                    },
                ],
                "Timestamp": datetime.now(
                    timezone.utc
                ),
                "Value": metric_value,
                "Unit": "Count",
            }
        ],
    )

    print(
        f"CloudWatch metric published:"
        f" MonitoringStatus={metric_value}"
    )


def publish_report_to_s3(
    report: dict,
    bucket: str,
    region: str,
    output_prefix: str,
):

    print("\n[6/6] Publishing monitoring report to S3...")

    s3 = boto3.client(
        "s3",
        region_name=region,
    )

    timestamp = datetime.now(
        timezone.utc
    )

    date_path = timestamp.strftime(
        "%Y/%m/%d"
    )

    timestamp_string = timestamp.strftime(
        "%Y%m%dT%H%M%SZ"
    )

    key = (
        f"{output_prefix}/"
        f"{date_path}/"
        f"monitoring_report_"
        f"{timestamp_string}.json"
    )

    body = json.dumps(
        report,
        indent=2,
    )

    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=body.encode("utf-8"),
        ContentType="application/json",
    )

    print(
        f"S3 report published:"
    )

    print(
        f"s3://{bucket}/{key}"
    )

    return key


def main() -> int:

    args = parse_arguments()

    print("=" * 80)
    print("COUNTRY RISK PREDICTION - MODEL MONITORING")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. Load production data
    # ------------------------------------------------------------------

    production_path = load_production_dataset(
        bucket=args.bucket,
        prefix=args.prefix,
        region=args.region,
    )

    # ------------------------------------------------------------------
    # 2. Load reference datasets
    # ------------------------------------------------------------------

    reference_path = Path(
        args.reference
    )

    reference_predictions_path = Path(
        args.reference_predictions
    )

    reference = pd.read_csv(
        reference_path
    )

    production = pd.read_csv(
        production_path
    )

    reference_predictions = pd.read_csv(
        reference_predictions_path
    )

    # ------------------------------------------------------------------
    # 3. Validate
    # ------------------------------------------------------------------

    validate_datasets(
        reference,
        production,
        reference_predictions,
    )

    # ------------------------------------------------------------------
    # 4. Calculate monitoring
    # ------------------------------------------------------------------

    (
        data_drift,
        prediction_drift,
        overall_status,
    ) = calculate_monitoring(
        reference,
        production,
        reference_predictions,
    )

    # ------------------------------------------------------------------
    # 5. Build machine-readable report
    # ------------------------------------------------------------------

    report = build_report(
        reference_path=reference_path,
        production_path=production_path,
        reference_predictions_path=reference_predictions_path,
        reference=reference,
        production=production,
        reference_predictions=reference_predictions,
        data_drift=data_drift,
        prediction_drift=prediction_drift,
    )

    # ------------------------------------------------------------------
    # 6. Save local report
    # ------------------------------------------------------------------

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w"
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
        )

    # ------------------------------------------------------------------
    # 7. Publish CloudWatch metric
    # ------------------------------------------------------------------

    publish_cloudwatch_metric(
        status=overall_status,
        endpoint=ENDPOINT_NAME,
        region=args.region,
    )

    # ------------------------------------------------------------------
    # 8. Publish report to S3
    # ------------------------------------------------------------------

    s3_key = publish_report_to_s3(
        report=report,
        bucket=args.bucket,
        region=args.region,
        output_prefix=args.output_prefix,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("MODEL MONITORING COMPLETED")
    print("=" * 80)

    print()
    print(
        f"Overall status : {overall_status}"
    )

    print(
        f"Alert triggered: "
        f"{report['monitoring']['alert_triggered']}"
    )

    print(
        f"Data drift     : "
        f"{report['data_drift']['significant_drift_features']} "
        f"significant / "
        f"{report['data_drift']['warning_features']} warning"
    )

    print(
        f"Prediction drift: "
        f"{report['prediction_drift']['status']}"
    )

    print(
        f"Local report   : {output_path}"
    )

    print(
        f"S3 report      : "
        f"s3://{args.bucket}/{s3_key}"
    )

    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )