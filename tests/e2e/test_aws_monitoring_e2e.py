import os
import subprocess

import boto3
import pytest


REGION = "us-east-1"

ENDPOINT = "country-risk-prediction-v7-v3"

BUCKET = "country-risk-prediction-monitoring-2026"

CAPTURE_PREFIX = (
    "data-capture/"
    "country-risk-prediction-v7-v3/"
)

ALARM_NAME = (
    "CountryRisk-ModelMonitoring-CRITICAL"
)

SNS_TOPIC = (
    "arn:aws:sns:"
    "us-east-1:"
    "287127678337:"
    "country-risk-model-monitoring-alerts"
)


@pytest.mark.aws
def test_sagemaker_endpoint_exists():

    sagemaker = boto3.client(
        "sagemaker",
        region_name=REGION,
    )

    response = sagemaker.describe_endpoint(
        EndpointName=ENDPOINT,
    )

    assert response["EndpointName"] == ENDPOINT

    assert response["EndpointStatus"] == "InService"


@pytest.mark.aws
def test_monitoring_s3_bucket_exists():

    s3 = boto3.client(
        "s3",
        region_name=REGION,
    )

    response = s3.list_objects_v2(
        Bucket=BUCKET,
        Prefix=CAPTURE_PREFIX,
        MaxKeys=1,
    )

    assert response["ResponseMetadata"][
        "HTTPStatusCode"
    ] == 200


@pytest.mark.aws
def test_cloudwatch_alarm_exists_and_has_sns_action():

    cloudwatch = boto3.client(
        "cloudwatch",
        region_name=REGION,
    )

    response = cloudwatch.describe_alarms(
        AlarmNames=[ALARM_NAME],
    )

    alarms = response["MetricAlarms"]

    assert len(alarms) == 1

    alarm = alarms[0]

    assert alarm["AlarmName"] == ALARM_NAME

    assert SNS_TOPIC in alarm["AlarmActions"]


@pytest.mark.aws
def test_sns_subscription_exists():

    sns = boto3.client(
        "sns",
        region_name=REGION,
    )

    response = sns.list_subscriptions_by_topic(
        TopicArn=SNS_TOPIC,
    )

    subscriptions = response[
        "Subscriptions"
    ]

    assert len(subscriptions) >= 1

    protocols = {
        subscription["Protocol"]
        for subscription in subscriptions
    }

    assert "email" in protocols


@pytest.mark.aws
def test_cloudwatch_monitoring_metric_exists():

    cloudwatch = boto3.client(
        "cloudwatch",
        region_name=REGION,
    )

    response = cloudwatch.list_metrics(
        Namespace="CountryRisk/ModelMonitoring",
        MetricName="MonitoringStatus",
        Dimensions=[
            {
                "Name": "Endpoint",
                "Value": ENDPOINT,
            },
            {
                "Name": "Environment",
                "Value": "production",
            },
        ],
    )

    assert len(response["Metrics"]) >= 1


@pytest.mark.aws
def test_cloudwatch_alarm_has_executed_sns_action():

    cloudwatch = boto3.client(
        "cloudwatch",
        region_name=REGION,
    )

    response = cloudwatch.describe_alarm_history(
        AlarmName=ALARM_NAME,
        HistoryItemType="Action",
        MaxRecords=20,
    )

    actions = response[
        "AlarmHistoryItems"
    ]

    successful_sns_actions = [
        item
        for item in actions
        if (
            "Successfully executed action"
            in item.get("HistorySummary", "")
            and SNS_TOPIC
            in item.get("HistorySummary", "")
        )
    ]

    assert successful_sns_actions, (
        "No successful CloudWatch -> SNS "
        "action found in alarm history."
    )
