import argparse
import json
from datetime import datetime, timezone

import boto3


STATUS_VALUES = {
    "STABLE": 0,
    "WARNING": 1,
    "CRITICAL": 2,
}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Publish model monitoring status to CloudWatch."
    )

    parser.add_argument(
        "--report",
        required=True,
        help="Path to monitoring report JSON.",
    )

    parser.add_argument(
        "--endpoint",
        required=True,
        help="SageMaker endpoint name.",
    )

    parser.add_argument(
        "--region",
        default="us-east-1",
        help="AWS region.",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    with open(args.report) as f:
        report = json.load(f)

    status = report["monitoring"]["overall_status"]

    if status not in STATUS_VALUES:
        raise ValueError(
            f"Unknown monitoring status: {status}"
        )

    metric_value = STATUS_VALUES[status]

    cloudwatch = boto3.client(
        "cloudwatch",
        region_name=args.region,
    )

    cloudwatch.put_metric_data(
        Namespace="CountryRisk/ModelMonitoring",
        MetricData=[
            {
                "MetricName": "MonitoringStatus",
                "Dimensions": [
                    {
                        "Name": "Endpoint",
                        "Value": args.endpoint,
                    },
                    {
                        "Name": "Environment",
                        "Value": "production",
                    },
                ],
                "Timestamp": datetime.now(timezone.utc),
                "Value": metric_value,
                "Unit": "Count",
            }
        ],
    )

    print("=" * 80)
    print("CLOUDWATCH MONITORING METRIC PUBLISHED")
    print("=" * 80)
    print()
    print(f"Endpoint : {args.endpoint}")
    print(f"Status   : {status}")
    print(f"Metric   : {metric_value}")
    print()
    print("Namespace: CountryRisk/ModelMonitoring")
    print("Metric   : MonitoringStatus")


if __name__ == "__main__":
    main()