# 🌍 MLOps Country Risk Prediction

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python\&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.9.0-F7931E?logo=scikit-learn\&logoColor=white)](https://scikit-learn.org/)
[![MLflow](https://img.shields.io/badge/MLflow-3.15.1-0194E2?logo=mlflow\&logoColor=white)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker\&logoColor=white)](https://www.docker.com/)
[![AWS](https://img.shields.io/badge/AWS-SageMaker%20%7C%20ECR%20%7C%20S3%20%7C%20CloudWatch-232F3E?logo=amazonaws\&logoColor=white)](https://aws.amazon.com/)
[![Tests](https://img.shields.io/badge/tests-65%20passed-success)](#testing-and-quality-assurance)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> **An end-to-end production-oriented MLOps platform for country risk prediction, combining macroeconomic data, machine learning, experiment tracking, model versioning, containerized inference, AWS SageMaker deployment, automated testing, and cloud observability.**

---

## 📌 Table of Contents

* [Project Overview](#-project-overview)
* [Business Problem](#-business-problem)
* [Project Objectives](#-project-objectives)
* [Solution Overview](#-solution-overview)
* [Architecture](#-architecture)
* [End-to-End ML Lifecycle](#-end-to-end-ml-lifecycle)
* [Data Pipeline](#-data-pipeline)
* [Feature Engineering](#-feature-engineering)
* [Machine Learning Model](#-machine-learning-model)
* [Experiment Tracking with MLflow](#-experiment-tracking-with-mlflow)
* [Model Serialization and Versioning](#-model-serialization-and-versioning)
* [Model Serving](#-model-serving)
* [Docker and Containerization](#-docker-and-containerization)
* [AWS Cloud Deployment](#-aws-cloud-deployment)
* [SageMaker Endpoint](#-sagemaker-endpoint)
* [Monitoring and Observability](#-monitoring-and-observability)
* [Testing and Quality Assurance](#-testing-and-quality-assurance)
* [CI/CD](#-cicd)
* [API Inference](#-api-inference)
* [Feature Dictionary](#-feature-dictionary)
* [Project Structure](#-project-structure)
* [Local Development](#-local-development)
* [Production Deployment Flow](#-production-deployment-flow)
* [Current Production Status](#-current-production-status)
* [Roadmap](#-roadmap)
* [Key MLOps Practices Demonstrated](#-key-mlops-practices-demonstrated)
* [Author](#-author)

---

# 🎯 Project Overview

**MLOps Country Risk Prediction** is an end-to-end machine learning engineering project designed to demonstrate how a data science model can be transformed into a reproducible, versioned, tested, containerized, deployed, and monitored production service.

The project goes beyond model development.

It implements the complete lifecycle:

```text
Data
 ↓
Data Ingestion
 ↓
Data Validation
 ↓
Data Cleaning
 ↓
Feature Engineering
 ↓
Model Training
 ↓
Experiment Tracking
 ↓
Model Versioning
 ↓
Artifact Storage
 ↓
Containerization
 ↓
Cloud Deployment
 ↓
Real-Time Inference
 ↓
Monitoring
```

The system uses macroeconomic and governance-related indicators to estimate a country-level risk score.

The primary objective is not only predictive performance, but also **reproducibility, reliability, traceability, deployment automation, and operational observability**.

---

# 💼 Business Problem

Country risk assessment is relevant to organizations involved in:

* International investment
* Financial analysis
* Credit risk assessment
* Portfolio allocation
* International expansion
* Sovereign risk analysis
* Economic research
* Strategic decision-making

Traditional country risk analysis often combines multiple economic, demographic, institutional, and governance indicators.

This project explores how these heterogeneous indicators can be transformed into a machine learning pipeline capable of producing a standardized country risk prediction.

The machine learning problem is formulated as a supervised regression task.

The model receives a set of macroeconomic and risk-related features and produces a continuous numerical risk prediction.

---

# 🎯 Project Objectives

The project was designed around five major objectives.

### 1. Build a reproducible machine learning pipeline

The same data processing and training workflow should be executable repeatedly with predictable results.

### 2. Implement production-oriented MLOps practices

The project incorporates:

* Data versioning
* Model versioning
* Experiment tracking
* Automated testing
* Containerization
* Cloud deployment
* Monitoring
* Infrastructure integration

### 3. Separate development from production

Model development occurs independently from model serving.

The final model artifact is packaged and deployed through a dedicated inference environment.

### 4. Make models deployable

The trained model is converted into a production artifact and exposed through an HTTP inference service.

### 5. Demonstrate the complete ML lifecycle

The project intentionally covers the transition:

> **From notebook → reproducible pipeline → versioned model → container → cloud endpoint → monitored production service.**

---

# 🏗️ Solution Overview

The architecture combines open economic data, Python-based machine learning, MLflow, DVC, Docker, and AWS.

```text
                         ┌─────────────────────┐
                         │   World Bank API    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Data Ingestion    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Data Validation     │
                         │ & Data Cleaning     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Feature Engineering │
                         │ & Lag Generation    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Model Training      │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │      MLflow         │
                         │ Experiment Tracking │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Model Artifact      │
                         │    Versioning       │
                         └──────────┬──────────┘
                                    │
                           ┌────────┴────────┐
                           ▼                 ▼
                       DVC + S3          MLflow
                           │
                           ▼
                    ┌───────────────┐
                    │ Docker Image  │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │   AWS ECR     │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ AWS SageMaker │
                    │   Endpoint    │
                    └───────┬───────┘
                            │
                 ┌──────────┴──────────┐
                 ▼                     ▼
          Single Inference       Batch Inference
                 │                     │
                 └──────────┬──────────┘
                            ▼
                    ┌───────────────┐
                    │  CloudWatch   │
                    │  Monitoring   │
                    └───────┬───────┘
                            │
                            ▼
                       5XX Alarms
```

---

# 🔄 End-to-End ML Lifecycle

The project follows a structured ML lifecycle.

## Phase 1 — Data Acquisition

Macroeconomic indicators are retrieved from the World Bank API.

The ingestion layer is responsible for:

* API communication
* Data extraction
* Schema handling
* Country identification
* Time-period alignment
* Raw data persistence

---

## Phase 2 — Data Validation

Before entering the modeling pipeline, data is validated for:

* Expected columns
* Data types
* Missing values
* Numeric constraints
* Duplicate observations
* Structural consistency

This prevents malformed upstream data from silently propagating into the model.

---

## Phase 3 — Data Cleaning

The cleaning process handles:

* Missing observations
* Invalid values
* Data type normalization
* Temporal consistency
* Feature alignment

The objective is to create a stable dataset suitable for downstream feature engineering.

---

## Phase 4 — Feature Engineering

The pipeline generates model-ready features from the original economic indicators.

This includes temporal features such as lagged variables.

For example:

```text
GDP(t)
      │
      └──► GDP(t-1)

Inflation(t)
      │
      └──► Inflation(t-1)
```

Lagged variables allow the model to incorporate historical information rather than relying exclusively on current-period observations.

---

# 🧮 Feature Engineering

The current model uses **12 numerical features**.

| Feature                | Description                                       |
| ---------------------- | ------------------------------------------------- |
| `gdp_per_capita`       | GDP per capita                                    |
| `inflation`            | Annual inflation rate                             |
| `life_expectancy`      | Life expectancy at birth                          |
| `population`           | Total population                                  |
| `population_growth`    | Annual population growth rate                     |
| `unemployment`         | Unemployment rate                                 |
| `exports`              | Exports as a percentage of GDP                    |
| `gdp_lag1`             | Previous-period GDP per capita                    |
| `inflation_lag1`       | Previous-period inflation                         |
| `life_expectancy_lag1` | Previous-period life expectancy                   |
| `economic_risk`        | Composite economic risk indicator                 |
| `governance_risk`      | Composite governance/institutional risk indicator |

All production inference inputs are numerical and validated before reaching the model.

---

# 🤖 Machine Learning Model

The initial production pipeline uses a **Random Forest regression model** as the baseline estimator.

Random Forest was selected as a strong baseline because it:

* Handles nonlinear relationships
* Captures feature interactions
* Requires limited assumptions about functional form
* Works well with heterogeneous numerical features
* Provides a useful benchmark for future model comparison

The project is deliberately structured so that the model implementation can evolve without redesigning the surrounding MLOps infrastructure.

This allows future experimentation with:

* XGBoost
* LightGBM
* CatBoost
* Gradient Boosting
* Ensemble approaches
* Explainable ML techniques

---

# 📊 Baseline Model Performance

The baseline Random Forest model achieved the following evaluation results:

| Metric   |       Score |
| -------- | ----------: |
| **MAE**  |  **8.6462** |
| **RMSE** | **14.8474** |
| **R²**   |  **0.1907** |

### Interpretation

The baseline provides a reproducible reference point for future modeling experiments.

The relatively modest R² also highlights an important aspect of the project:

> The objective is not to present an artificially optimized model, but to establish a transparent and reproducible ML system where future models can be benchmarked against a known baseline.

This makes the repository suitable for continued experimentation and model improvement.

---

# 🧪 Experiment Tracking with MLflow

**MLflow** is used to track the machine learning lifecycle.

The project records model metadata including:

* Model version
* Parameters
* Evaluation metrics
* Model artifacts
* Python environment
* Dependency information
* Serialization format

The current production artifact metadata includes:

```text
MLflow:       3.15.1
Python:       3.12.3
scikit-learn: 1.9.0
Serialization: skops
```

MLflow provides traceability between:

```text
Experiment
    ↓
Training Run
    ↓
Model Artifact
    ↓
Model Version
    ↓
Deployment
```

This is critical for reproducibility and production governance.

---

# 📦 Model Serialization and Versioning

Large model binaries are intentionally kept outside the Git repository.

The project uses:

* **DVC** for model/data versioning
* **Amazon S3** for artifact storage
* **MLflow** for model metadata and lifecycle tracking
* **skops** for secure scikit-learn model serialization

The current production model artifact is approximately **100 MB**.

The model is stored in S3 using a versioned structure:

```text
s3://country-risk-prediction-mlops-models-2026/
└── models/
    └── country-risk-prediction/
        └── v7/
            ├── model.skops
            ├── MLmodel
            └── environment metadata
```

This architecture keeps Git focused on source code while using dedicated storage systems for large ML artifacts.

---

# 🐳 Docker and Containerization

The inference environment is containerized using Docker.

The SageMaker-specific image is defined through:

```text
Dockerfile.sagemaker
```

The container includes:

* Python runtime
* Model artifact
* ML dependencies
* FastAPI
* Uvicorn
* Prediction logic
* Request validation

Containerization provides environment consistency across:

```text
Development
     ↓
Testing
     ↓
Container Build
     ↓
AWS ECR
     ↓
SageMaker
```

This eliminates many environment-related inconsistencies between development and production.

---

# 🔌 Model Serving

The model is served through a custom **FastAPI/Uvicorn** inference server.

The application listens on:

```text
Port 8080
```

The serving layer is responsible for:

1. Receiving inference requests
2. Validating payload structure
3. Validating required features
4. Converting input into the expected dataframe format
5. Executing model inference
6. Returning predictions

The service supports both:

* Single-record inference
* Batch inference

---

# 📡 API Inference

The inference endpoint accepts requests using a dataframe-records structure.

Example:

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
```

The same inference contract can be used locally and through the deployed SageMaker endpoint.

---

# ☁️ AWS Cloud Deployment

The production infrastructure is built around AWS managed services.

### AWS Services

| Service               | Purpose                                  |
| --------------------- | ---------------------------------------- |
| **Amazon S3**         | ML artifact and model storage            |
| **AWS ECR**           | Docker image registry                    |
| **Amazon SageMaker**  | Model hosting and inference              |
| **Amazon CloudWatch** | Monitoring and operational observability |
| **AWS IAM**           | Secure service permissions               |
| **AWS STS/OIDC**      | Secure CI/CD authentication              |

The deployment process follows:

```text
Source Code
    ↓
Docker Build
    ↓
Amazon ECR
    ↓
SageMaker Model
    ↓
Endpoint Configuration
    ↓
SageMaker Endpoint
    ↓
Inference Verification
    ↓
CloudWatch Monitoring
```

---

# 🚀 SageMaker Endpoint

The current deployed model is:

```text
country-risk-prediction-v7-v3
```

The SageMaker model uses a dedicated ECR container image and an IAM execution role.

The production deployment includes:

```text
Model
 ↓
Container
 ↓
SageMaker Model
 ↓
Endpoint Configuration
 ↓
Real-Time Endpoint
```

This provides a production-style HTTPS inference interface suitable for real-time prediction requests.

---

# 📈 Monitoring and Observability

Production observability is implemented through **Amazon CloudWatch**.

The endpoint exposes operational metrics including:

* Invocations
* Model latency
* Invocation latency
* 5XX errors
* Model errors

A dedicated alarm monitors endpoint failures.

### 5XX Alarm

```text
Alarm:
country-risk-prediction-v7-v3-5xx

Metric:
Invocation5XXErrors

Statistic:
Sum

Threshold:
>= 1 error

Evaluation Period:
5 minutes
```

The current post-deployment status is:

```text
Alarm State: OK
5XX Errors: 0
```

The monitoring layer demonstrates that deployment does not end when the endpoint becomes available.

The system must also be observable after deployment.

---

# 🧪 Testing and Quality Assurance

Testing is integrated throughout the project.

The test suite includes:

* Unit tests
* Functional tests
* Integration tests
* API validation tests
* Model inference tests
* AWS endpoint integration tests

Current test status:

```text
65 tests passed
2 integration tests passed
```

Run the standard test suite:

```bash
pytest -q
```

Run AWS integration tests:

```bash
pytest -m integration -q
```

Integration tests validate real communication with the deployed SageMaker endpoint.

This ensures that the system is tested not only at the function level, but also at the infrastructure integration level.

---

# 🔄 CI/CD

The repository is structured around a CI/CD-oriented development workflow.

The conceptual pipeline is:

```text
Developer
    │
    ▼
GitHub Pull Request
    │
    ▼
Continuous Integration
    │
    ├── Dependency validation
    ├── Import validation
    ├── DVC validation
    ├── Automated tests
    └── Docker build
    │
    ▼
Protected Main Branch
    │
    ▼
Continuous Deployment
    │
    ▼
GitHub OIDC
    │
    ▼
AWS IAM Role
    │
    ├───────────────┐
    ▼               ▼
   S3              ECR
    │               │
    └───────┬───────┘
            ▼
      SageMaker
            │
            ▼
       Verification
            │
            ▼
       CloudWatch
```

The architecture separates source control, testing, artifact management, container registry, deployment, and monitoring.

---

# 🔐 Security and Reproducibility

The project avoids embedding cloud credentials directly into application code.

AWS access is managed through IAM-based authentication and role-based permissions.

The production architecture therefore separates:

```text
Application Code
        +
Model Artifacts
        +
Container Images
        +
Cloud Permissions
```

This reduces the coupling between application logic and infrastructure credentials.

The use of versioned artifacts also improves reproducibility by allowing a deployed model to be traced back to a specific artifact version.

---

# 📁 Project Structure

```text
MLOps-Country-Risk-Prediction/
│
├── data/
│   ├── raw/
│   │   └── Raw World Bank API data
│   │
│   ├── processed/
│   │   └── Cleaned and engineered datasets
│   │
│   └── external/
│       └── Supplementary metadata
│
├── docker/
│   └── model/
│       ├── MLmodel
│       ├── serve.py
│       ├── requirements.txt
│       └── model.skops.dvc
│
├── notebooks/
│   └── Exploratory analysis and experiments
│
├── scripts/
│   ├── __init__.py
│   └── inference.py
│
├── src/
│   ├── api/
│   │   └── API utilities and validation
│   │
│   ├── data/
│   │   └── Data ingestion and validation
│   │
│   ├── features/
│   │   └── Feature engineering
│   │
│   ├── models/
│   │   └── Model training and promotion
│   │
│   ├── visualization/
│   │   └── Metrics and visualizations
│   │
│   ├── utils/
│   │   └── AWS and MLflow utilities
│   │
│   ├── config.py
│   └── main.py
│
├── tests/
│   ├── Unit tests
│   ├── Functional tests
│   └── Integration tests
│
├── Dockerfile.sagemaker
├── dvc.yaml
├── dvc.lock
├── pytest.ini
├── requirements.txt
└── README.md
```

---

# 💻 Local Development

## 1. Clone the Repository

```bash
git clone https://github.com/AnderCruz/MLOps-Country-Risk-Prediction.git

cd MLOps-Country-Risk-Prediction
```

---

## 2. Create a Virtual Environment

### Linux / macOS

```bash
python -m venv .venv

source .venv/bin/activate
```

### Windows

```powershell
python -m venv .venv

.venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Run the Main Pipeline

```bash
python src/main.py
```

---

## 5. Run Tests

```bash
pytest -q
```

---

## 6. Run Integration Tests

AWS credentials and access to the deployed endpoint are required.

```bash
pytest -m integration -q
```

---

## 7. Run Inference

```bash
python scripts/inference.py
```

---

# 🔬 Reproducibility

The project combines several mechanisms to improve reproducibility.

### Source Code

Git provides version control for:

* Python source code
* Configuration
* Tests
* Infrastructure definitions
* Pipeline definitions

### Data and Model Artifacts

DVC and Amazon S3 provide versioning for large ML artifacts.

### Experiment Metadata

MLflow tracks:

* Parameters
* Metrics
* Model versions
* Environment metadata

### Runtime Environment

Docker provides a reproducible production runtime.

Together:

```text
Git
 +
DVC
 +
MLflow
 +
Docker
 +
AWS
```

create a reproducible ML delivery workflow.

---

# 🏭 Production Deployment Flow

The complete deployment lifecycle can be summarized as:

```text
1. Develop
      │
      ▼
2. Validate
      │
      ▼
3. Test
      │
      ▼
4. Train
      │
      ▼
5. Track with MLflow
      │
      ▼
6. Version model with DVC
      │
      ▼
7. Store artifact in S3
      │
      ▼
8. Build Docker image
      │
      ▼
9. Push image to ECR
      │
      ▼
10. Register SageMaker model
      │
      ▼
11. Deploy endpoint
      │
      ▼
12. Execute inference tests
      │
      ▼
13. Monitor with CloudWatch
```

This represents the central philosophy of the project:

> **A machine learning model is not finished when it achieves a good metric. It is finished when it can be reliably delivered, reproduced, served, monitored, and maintained.**

---

# 📊 Current Production Status

| Component               | Status            |
| ----------------------- | ----------------- |
| Data ingestion          | ✅ Implemented     |
| Data validation         | ✅ Implemented     |
| Feature engineering     | ✅ Implemented     |
| Model training          | ✅ Implemented     |
| MLflow tracking         | ✅ Implemented     |
| Model versioning        | ✅ Implemented     |
| DVC                     | ✅ Implemented     |
| S3 artifact storage     | ✅ Implemented     |
| Docker containerization | ✅ Implemented     |
| Amazon ECR              | ✅ Implemented     |
| SageMaker deployment    | ✅ Implemented     |
| Real-time inference     | ✅ Operational     |
| Batch inference support | ✅ Implemented     |
| Automated testing       | ✅ 65 tests passed |
| Integration testing     | ✅ 2 tests passed  |
| CloudWatch monitoring   | ✅ Implemented     |
| 5XX alarm               | ✅ OK              |
| Production endpoint     | ✅ InService       |

---

# 🧠 What This Project Demonstrates

This project is intentionally broader than a traditional machine learning notebook.

It demonstrates practical experience across:

### Data Science

* Exploratory data analysis
* Feature engineering
* Regression modeling
* Model evaluation
* Time-dependent features

### Machine Learning Engineering

* Modular Python architecture
* Model serialization
* Inference services
* API validation
* Testing
* Reproducibility

### MLOps

* MLflow
* DVC
* Model versioning
* Artifact management
* Experiment tracking
* CI/CD
* Production deployment
* Monitoring

### Cloud Engineering

* Amazon S3
* Amazon ECR
* Amazon SageMaker
* Amazon CloudWatch
* AWS IAM
* OIDC-based authentication

### Software Engineering

* Git-based development
* Automated testing
* Docker
* Modular architecture
* Separation of concerns
* Production-oriented project structure

---

# 🛣️ Roadmap

The project is designed to evolve beyond the baseline implementation.

## Advanced Modeling

* [ ] Benchmark XGBoost
* [ ] Benchmark LightGBM
* [ ] Benchmark CatBoost
* [ ] Hyperparameter optimization
* [ ] Cross-validation strategy
* [ ] Model selection framework

## Explainability

* [ ] SHAP integration
* [ ] Global feature importance
* [ ] Local prediction explanations
* [ ] Model interpretability reports

## Advanced MLOps

* [ ] Automated model promotion
* [ ] Model approval workflow
* [ ] Automated rollback
* [ ] Model registry governance
* [ ] Automated retraining

## Monitoring

* [ ] Data drift detection
* [ ] Feature distribution monitoring
* [ ] Model performance monitoring
* [ ] Prediction drift monitoring
* [ ] Automated drift alerts

## Product Layer

* [ ] Interactive Streamlit dashboard
* [ ] Country risk simulation interface
* [ ] Historical risk visualization
* [ ] Model explanation dashboard
* [ ] REST API documentation

---

# 🏆 Key MLOps Practices Demonstrated

The project demonstrates the following production-oriented principles:

### 1. Reproducibility

The same pipeline can be executed repeatedly using versioned source code, data artifacts, model artifacts, and environments.

### 2. Traceability

A production model can be traced through:

```text
Model Version
      ↓
MLflow Metadata
      ↓
Artifact
      ↓
DVC
      ↓
S3
      ↓
Docker Image
      ↓
SageMaker Endpoint
```

### 3. Separation of Concerns

The project separates:

* Data processing
* Feature engineering
* Model training
* Model serving
* Infrastructure
* Testing
* Monitoring

### 4. Production Readiness

The model is not limited to experimentation.

It is exposed through a real cloud endpoint and monitored using AWS infrastructure.

### 5. Automated Quality Assurance

The project contains a structured test suite covering both application logic and infrastructure integration.

### 6. Cloud-Native Deployment

The model is packaged as a Docker container and deployed through AWS managed services.

---

# 🎓 Learning Outcomes

This project was developed to consolidate practical knowledge in:

* Machine Learning
* Data Science
* MLOps
* MLflow
* DVC
* Docker
* FastAPI
* AWS
* SageMaker
* ECR
* S3
* CloudWatch
* CI/CD
* Model serving
* Automated testing
* Production ML architecture

The central learning objective was to bridge the gap between:

> **"I trained a machine learning model."**

and:

> **"I built and deployed a reproducible machine learning system."**

---

# 👤 Author

## Anderson Cruz

**Data Scientist | Machine Learning & Predictive Analytics | MLOps**

Focused on building production-oriented machine learning systems combining:

* Data Science
* Machine Learning
* Deep Learning
* MLOps
* Cloud Computing
* Predictive Analytics

### Profiles

🐙 GitHub:
https://github.com/AnderCruz

💼 LinkedIn:
https://linkedin.com/in/anderjcruz

---

# ⭐ Project Positioning

This repository is part of my **Data Science and MLOps portfolio** and represents a complete production-oriented machine learning workflow.

The project demonstrates that modern Data Science requires more than model development.

A successful ML solution must connect:

```text
Business Problem
       ↓
Data
       ↓
Features
       ↓
Model
       ↓
Experimentation
       ↓
Versioning
       ↓
Testing
       ↓
Deployment
       ↓
Monitoring
       ↓
Continuous Improvement
```

**This project implements that complete lifecycle.**
