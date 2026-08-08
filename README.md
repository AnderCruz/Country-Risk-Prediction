# 🌍 Country Risk Prediction

An end-to-end Data Science project that predicts country-level economic risk using macroeconomic indicators from the World Bank.

This project demonstrates a complete Machine Learning pipeline, from automated data ingestion to model training, following software engineering best practices.

---

# Project Overview

The objective is to build a scalable platform capable of collecting, processing and modelling economic indicators to support country risk analysis.

Current version includes:

- Automated data ingestion from the World Bank API
- Data validation and quality checks
- Data cleaning
- Feature engineering
- Exploratory Data Analysis (EDA)
- Baseline Machine Learning model (Random Forest)
- Modular and scalable architecture

Future versions will integrate governance indicators, additional machine learning algorithms and an interactive dashboard.

---

# Architecture

```
                World Bank API
                       │
                       ▼
              Data Ingestion Layer
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
               Machine Learning
                       │
                       ▼
                 Model Evaluation
```

---

# Project Structure

```
Country-Risk-Prediction/

├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── docs/
│
├── notebooks/
│
├── reports/
│
├── src/
│   ├── api/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── visualization/
│   ├── utils/
│   ├── config.py
│   └── main.py
│
├── tests/
│
├── requirements.txt
│
└── README.md
```

---

# Technologies

- Python
- Pandas
- NumPy
- Scikit-Learn
- Requests
- Jupyter Notebook
- World Bank API
- Git

---

# Machine Learning Pipeline

Current pipeline:

```
Download Data

↓

Validation

↓

Cleaning

↓

Feature Engineering

↓

EDA

↓

Train Model

↓

Model Evaluation
```

Current baseline model:

- Random Forest Regressor

Evaluation Metrics:

- MAE
- RMSE
- R² Score

---

# Current Features

Current dataset includes:

- GDP per Capita
- Population
- Inflation
- Life Expectancy
- GDP Growth
- Population Growth
- Lag Features

---

# Future Improvements

Planned roadmap:

- Worldwide Governance Indicators (WGI)
- Country Risk Score
- XGBoost
- LightGBM
- CatBoost
- Hyperparameter Tuning
- SHAP Explainability
- Streamlit Dashboard
- Docker
- CI/CD with GitHub Actions
- Unit Testing
- MLflow Experiment Tracking

---

# Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Country-Risk-Prediction.git
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate it

Linux

```bash
source .venv/bin/activate
```

Windows

```bash
.venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the project

```bash
python src/main.py
```

---

# Project Status

Current Version

**v1.0.0**

Status

✅ Active Development

---

# Author

**Anderson Cruz**

Data Scientist | Machine Learning | Predictive Analytics

LinkedIn:

https://linkedin.com/in/anderjcruz

GitHub:

https://github.com/anderjcruz