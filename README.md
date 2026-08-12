# 🌍 Country Risk Prediction

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-active--development-orange.svg)](#current-status)

An end-to-end **Data Science & MLOps** platform for predicting country-level economic and governance risk metrics using real-world macroeconomic indicators.

This repository covers the complete machine learning lifecycle: automated data ingestion, validation, feature engineering, model training, experiment tracking, artifact versioning, containerisation, automated testing, and scalable cloud deployment on AWS SageMaker with CloudWatch monitoring.

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Architecture & Workflow](#-architecture--workflow)
- [Current Implementation](#-current-implementation)
- [Feature Store & Data Dictionary](#-feature-store--data-dictionary)
- [Model Serving & API Payload](#-model-serving--api-payload)
- [Testing & Quality Assurance](#-testing--quality-assurance)
- [Model Versioning & Artifact Storage](#-model-versioning--artifact-storage)
- [AWS Cloud Infrastructure](#-aws-cloud-infrastructure)
- [Monitoring & Observability](#-monitoring--observability)
- [Baseline Model Performance](#-baseline-model-performance)
- [Quickstart & Local Development](#-quickstart--local-development)
- [Project Structure](#-project-structure)
- [Project Roadmap](#-roadmap)
- [Current Validation Status](#-current-status)
- [Author & Contact](#-author)

---

## 🎯 Project Overview

Understanding economic risk on a global scale requires combining historical macroeconomic trends with governance metrics. The primary objective of this project is to build a robust, reproducible, and production-ready machine learning framework that ingests data from the **World Bank API**, processes complex time-series features, and serves real-time predictions via cloud endpoints.

### Key Highlights
* **Automated Data Pipeline:** Pulls macroeconomic indicators directly from public APIs with automated schema validation and cleaning.
* **Experiment Tracking:** Uses **MLflow** for tracking hyperparameters, evaluation metrics, and model versions.
* **Large File Versioning:** Manages heavy model binaries outside of Git using **DVC** and **Amazon S3**.
* **Containerised Cloud Deployment:** Packages serving environments into Docker containers hosted on **AWS ECR** and served through **AWS SageMaker**.
* **Production Observability:** Endpoint health is continuously monitored using **Amazon CloudWatch** with automated 5XX alarm triggers.

---

## 🏗️ Architecture & Workflow

The pipeline represents a complete MLOps workflow from source data to endpoint deployment:

```text
                           World Bank API
                                 │
                                 ▼
                          Data Ingestion
                                 │
                                 ▼
                         Data Validation
                                 │
                                 ▼
                          Data Cleaning
                                 │
                                 ▼
                        Feature Engineering
                                 │
                                 ▼
                       Exploratory Analysis
                                 │
                                 ▼
                        Model Development
                                 │
                                 ▼
                              MLflow
                                 │
                                 ▼
                          Model Registry
                                 │
                                 ▼
                           DVC + Amazon S3
                                 │
                                 ▼
                            Docker Image
                                 │
                                 ▼
                             Amazon ECR
                                 │
                                 ▼
                        Amazon SageMaker
                            Endpoint
                                 │
                   ┌─────────────┴─────────────┐
                   ▼                           ▼
            Single Inference             Batch Inference
                   │                           │
                   └─────────────┬─────────────┘
                                 ▼
                             CloudWatch
                                 │
                                 ▼
                          Error Monitoring
                                 │
                                 ▼
                           5XX Alarm


🚀 MLOps Infrastructure: Docker containerisation (`Dockerfile.sagemaker`), AWS ECR repository integration, SageMaker endpoint deployment, and boto3 client orchestration.

---

## 📋 Feature Store & Data Dictionary

The model accepts macroeconomic indicators and generated time-lagged variables to produce a risk score:

| Feature | Type | Description |
|---|---|---|
| `gdp_per_capita` | `float` | Gross Domestic Product per capita (USD) |
| `inflation` | `float` | Annual inflation rate (%) |
| `life_expectancy` | `float` | Life expectancy at birth (years) |
| `population` | `float` | Total country population |
| `population_growth` | `float` | Annual population growth rate (%) |
| `unemployment` | `float` | Unemployment rate (% of total labor force) |
| `exports` | `float` | Exports of goods and services (% of GDP) |
| `gdp_lag1` | `float` | Previous period GDP per capita |
| `inflation_lag1` | `float` | Previous period inflation rate |
| `life_expectancy_lag1` | `float` | Previous period life expectancy |
| `economic_risk` | `float` | Composite macroeconomic risk metric |
| `governance_risk` | `float` | Composite governance/institutional risk metric |

---

## 🔌 Model Serving & API Payload

The trained model is packaged using MLflow and served through a custom **FastAPI / Uvicorn** server inside the container listening on **Port 8080**.

### Example JSON Payload (`dataframe_records` format):

```json
{
  "dataframe_records": [
    {
      "gdp_per_capita": 15000.0,
      "inflation": 3.0,
      "life_expectancy": 75.0,
      "population": 50000000.0,
      "population_growth": 1.0,
      "unemployment": 6.0,
      "exports": 25.0,
      "gdp_lag1": 14500.0,
      "inflation_lag1": 3.2,
      "life_expectancy_lag1": 74.8,
      "economic_risk": 0.30,
      "governance_risk": 0.25
    }
  ]
}
Both single-record queries and batch inferences are supported and validated.🧪 Testing & Quality AssuranceAutomated unit, functional, and integration tests ensure system stability prior to cloud promotion.Total Test Suite: 65 passed, 1 warningRunning TestsBash# Run the entire unit and functional test suite
pytest -q

# Run live AWS SageMaker integration tests
pytest -m integration -q
Integration tests verify live payload delivery to the SageMaker endpoint for both single and batch requests.📦 Model Versioning & Artifact StorageTo avoid tracking large binary files in Git, DVC (Data Version Control) handles model storage with Amazon S3 serving as the remote backend.PlaintextGit Repository
 │
 └── model.skops.dvc
          │
          ▼
     DVC Remote
          │
          ▼
     Amazon S3 Bucket
          │
          ▼
     model.skops (96 MB Artifact)

☁️ AWS Cloud InfrastructureThe model is deployed as a real-time HTTP endpoint on AWS SageMaker.Active Endpoint Name: country-risk-prediction-v7-v3Deployment PipelineBuild runtime image using Dockerfile.sagemaker.Push container image to Amazon ECR.Register model instance in SageMaker Model Registry.Spin up real-time HTTPS SageMaker Endpoint.📊 Monitoring & ObservabilityEndpoint operational metrics are published to Amazon CloudWatch:InvocationsModelLatencyInvocation5XXErrorsInvocationModelErrorsAlarm Configuration (country-risk-prediction-v7-v3-5xx)Metric: Invocation5XXErrorsStatistic: SumThreshold: >= 1 error within a 5-minute evaluation period.Current Alarm State: OK (0 errors reported post-deployment).📈 Baseline Model PerformanceThe initial Random Forest baseline evaluation yields the following performance metrics on test evaluation splits:MetricScoreDescriptionMAE8.6462Mean Absolute ErrorRMSE14.8474Root Mean Squared ErrorR²0.1907Coefficient of Determination💻 Quickstart & Local Development1. Clone & Set Up EnvironmentBash# Clone repository
git clone [https://github.com/AnderCruz/Country-Risk-Prediction.git](https://github.com/AnderCruz/Country-Risk-Prediction.git)
cd Country-Risk-Prediction

# Create and activate virtual environment
python -m venv .venv

# Linux / macOS:
source .venv/bin/activate

# Windows:
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
2. Pipeline CommandsBash# Execute full data ingestion & training pipeline
python src/main.py

# Run unit tests
pytest -q

# Execute live SageMaker inference script
python scripts/inference.py
📁 Project StructurePlaintextCountry-Risk-Prediction/
│
├── data/
│   ├── raw/                  # Raw World Bank API pulls
│   ├── processed/            # Cleaned and engineered features
│   └── external/             # Supplementary metadata
│
├── docker/
│   └── model/
│       ├── MLmodel           # MLflow model package definition
│       ├── serve.py          # Custom FastAPI serving engine
│       ├── requirements.txt  # Container dependencies
│       └── model.skops.dvc   # DVC tracker for model file
│
├── notebooks/                # Exploratory Data Analysis & experiments
│
├── scripts/
│   ├── __init__.py
│   └── inference.py          # Boto3 client for endpoint testing
│
├── src/
│   ├── api/                  # API utilities and schema validators
│   ├── data/                 # Ingestion & validation modules
│   ├── features/             # Feature transformers and lag generators
│   ├── models/               # Model training and promotion scripts
│   ├── visualization/        # Performance plots and metrics dashboard
│   ├── utils/                # AWS & MLflow helpers
│   ├── config.py             # Central project configuration
│   └── main.py               # Main execution entrypoint
│
├── tests/                    # Unit, functional, and integration tests
├── Dockerfile.sagemaker      # AWS SageMaker runtime container definition
├── dvc.yaml                  # DVC pipeline stages
├── dvc.lock                  # DVC lockfile
├── pytest.ini                # Pytest settings and custom markers
├── requirements.txt          # Python dependencies
└── README.md

## 🔄 CI/CD & MLOps Deployment

GitHub Pull Request
        │
        ▼
      CI
        │
        ├── Dependencies
        ├── Import validation
        ├── DVC validation
        ├── Unit/integration tests
        └── Docker build
                │
                ▼
             main
                │
                ▼
              CD
                │
                ▼
          GitHub OIDC
                │
                ▼
          AWS IAM Role
                │
        ┌───────┴────────┐
        ▼                ▼
      DVC/S3            ECR
        │                │
        └───────┬────────┘
                ▼
        SageMaker Model
                │
                ▼
       Endpoint Config
                │
                ▼
       SageMaker Endpoint
                │
                ▼
          Verification



🛣️ Roadmap
[ ] Advanced Modeling: Benchmark XGBoost, LightGBM, and CatBoost models.
[ ] Explainability: Integrate SHAP values for global and local feature importance.
[ ] CI/CD Pipeline: GitHub Actions workflow for automated testing, linting, and deployment.
[ ] Drift Monitoring: Continuous monitoring for data drift and model performance decay.
[ ] Frontend Dashboard: Interactive Streamlit Web UI for real-time risk simulation.

✅ Current StatusStatus: Active DevelopmentUnit Tests: 65 PassedIntegration Tests: 2 PassedInference Pipeline: Single & Batch PASSSageMaker Endpoint Status: InServiceCloudWatch Alarm: OK

---

👤 AuthorAnderson CruzData Scientist | Machine Learning & Predictive Analytics
💼 LinkedIn: linkedin.com/in/anderjcruz
🐙 GitHub: github.com/AnderCruzEOF