import argparse
import subprocess
import tempfile
from pathlib import Path

import boto3


def list_capture_files(
    bucket: str,
    prefix: str,
    region: str,
) -> list[str]:
    s3 = boto3.client(
        "s3",
        region_name=region,
    )

    paginator = s3.get_paginator(
        "list_objects_v2"
    )

    files = []

    for page in paginator.paginate(
        Bucket=bucket,
        Prefix=prefix,
    ):
        for obj in page.get("Contents", []):
            key = obj["Key"]

            if key.endswith(".jsonl"):
                files.append(key)

    return sorted(files)


def download_capture_files(
    bucket: str,
    keys: list[str],
    output_dir: Path,
    region: str,
) -> None:
    s3 = boto3.client(
        "s3",
        region_name=region,
    )

    for index, key in enumerate(keys, start=1):

        destination = (
            output_dir /
            f"capture-{index:06d}.jsonl"
        )

        print(
            f"Downloading: s3://{bucket}/{key}"
        )

        s3.download_file(
            bucket,
            key,
            str(destination),
        )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Download SageMaker Data Capture "
            "files from S3 and extract production data."
        )
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
        "--region",
        default="us-east-1",
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    print("=" * 70)
    print("SAGEMAKER PRODUCTION DATA LOADER")
    print("=" * 70)

    keys = list_capture_files(
        bucket=args.bucket,
        prefix=args.prefix,
        region=args.region,
    )

    print(
        f"\nCapture files found: {len(keys)}"
    )

    if not keys:
        raise RuntimeError(
            "No SageMaker Data Capture files found."
        )

    output = Path(args.output)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.TemporaryDirectory(
        prefix="country-risk-capture-"
    ) as temp_dir:

        temp_path = Path(temp_dir)

        download_capture_files(
            bucket=args.bucket,
            keys=keys,
            output_dir=temp_path,
            region=args.region,
        )

        print(
            "\nRunning production extractor..."
        )

        subprocess.run(
            [
                "python",
                "monitoring/drift/extract_production.py",
                "--input",
                str(temp_path),
                "--output",
                str(output),
            ],
            check=True,
        )

    print("\n" + "=" * 70)
    print("PRODUCTION DATASET CREATED")
    print("=" * 70)

    print(
        f"Output: {output}"
    )


if __name__ == "__main__":
    main()
