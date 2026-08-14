# Production Monitoring Evidence

## Purpose

This document records the validation of the production monitoring architecture running on AWS.

The objective was to verify that a controlled model drift event can travel through the complete monitoring chain:

```text
Synthetic Traffic
      ↓
SageMaker Endpoint
      ↓
SageMaker Data Capture
      ↓
Amazon S3
      ↓
Production Dataset
      ↓
Drift Detection
      ↓
MonitoringStatus
      ↓
CloudWatch
      ↓
CloudWatch Alarm
      ↓
SNS