# Country Risk Prediction — MLOps Architecture

## Overview

The Country Risk Prediction platform is an end-to-end MLOps system for predicting sovereign country risk from macroeconomic and governance indicators.

The platform covers the complete ML lifecycle:

- data ingestion and preparation
- feature engineering
- model training and validation
- experiment tracking and model registry
- model promotion
- containerized serving
- AWS SageMaker deployment
- production inference
- data capture
- automated drift detection
- CloudWatch monitoring
- SNS alerting
- automated testing
- CI/CD

The architecture is designed to demonstrate how a machine learning model can be moved from experimentation to monitored production operation.

---

## Architecture

```mermaid
flowchart TB

    A[World Bank / Public Data] --> B[Data Ingestion]

    B --> C[Data Validation]
    C --> D[Feature Engineering]

    D --> E[Training Pipeline]

    E --> F[MLflow Tracking]
    F --> G[Model Validation]

    G --> H{Validation Gate}

    H -->|Pass| I[MLflow Model Registry]
    H -->|Fail| X[Reject Model]

    I --> J[Model Promotion]

    J --> K[Docker Image]
    K --> L[GitHub Actions CI/CD]

    L --> M[AWS SageMaker Endpoint]

    M --> N[Production Inference]

    N --> O[SageMaker Data Capture]
    O --> P[Amazon S3]

    P --> Q[Production Dataset Extraction]

    Q --> R[Drift Detection]

    R --> S[PSI]
    R --> T[Kolmogorov-Smirnov Test]

    S --> U[Monitoring Status]
    T --> U

    U --> V[CloudWatch Custom Metric]

    V --> W{CloudWatch Alarm}

    W -->|CRITICAL| Y[SNS Alert]
    W -->|OK| Z[Normal Operation]

    AA[Synthetic Drift Tests] --> M
    AA --> R

    AB[Unit Tests] --> L
    AC[Integration Tests] --> L
    AD[AWS E2E Tests] --> M