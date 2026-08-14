import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import boto3


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Publish model monitoring report to S3."
    )

    parser.add_argument(
        "--report",
        required=True,
        help="Path to monitoring JSON report.",
    )

    parser.add_argument(
        "--bucket",
        required=True,
        help="S3 bucket for monitoring reports.",
    )

    parser.add_argument(
        "--prefix",
        default="monitoring/reports",
        help="S3 prefix for monitoring reports.",
    )

    parser.add_argument(
        "--region",
        default="us-east-1",
        help="AWS region.",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    report_path = Path(args.report)

    if not report_path.exists():
        raise FileNotFoundError(
            f"Monitoring report not found: {report_path}"
        )

    with report_path.open() as f:
        report = json.load(f)

    timestamp = datetime.now(timezone.utc)

    date_path = timestamp.strftime("%Y/%m/%d")
    timestamp_name = timestamp.strftime(
        "%Y%m%dT%H%M%SZ"
    )

    s3_key = (
        f"{args.prefix}/"
        f"{date_path}/"
        f"monitoring_report_{timestamp_name}.json"
    )

    s3 = boto3.client(
        "s3",
        region_name=args.region,
    )

    s3.put_object(
        Bucket=args.bucket,
        Key=s3_key,
        Body=json.dumps(
            report,
            indent=2,
        ).encode("utf-8"),
        ContentType="application/json",
    )

    print("=" * 80)
    print("MODEL MONITORING REPORT PUBLISHED")
    print("=" * 80)
    print()
    print(f"Local report : {report_path}")
    print(f"S3 bucket   : {args.bucket}")
    print(f"S3 key      : {s3_key}")
    print()
    print(
        f"s3://{args.bucket}/{s3_key}"
    )


if __name__ == "__main__":
    main()