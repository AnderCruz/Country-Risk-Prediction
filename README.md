# 🌍 Country Risk Prediction

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange)
![Status](https://img.shields.io/badge/Status-Active-success)
![Version](https://img.shields.io/badge/Version-v1.0.0-blue)

An end-to-end Data Science project that predicts country risk using macroeconomic indicators from the World Bank.

The objective is to build a scalable Machine Learning platform capable of collecting, processing and modelling economic indicators to support country risk analysis.

---

# Project Architecture

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
                Exploratory Data Analysis
                           │
                           ▼
                 Machine Learning Model
                           │
                           ▼
                   Model Evaluation
```

---

# Current Features

- Automated data ingestion from the World Bank API
- Modular project architecture
- Data validation
- Data cleaning
- Feature engineering
- Automated EDA reports
- Random Forest baseline model
- Model evaluation

---

# Technologies

- Python 3.12
- Pandas
- NumPy
- Scikit-Learn
- Requests
- Jupyter Notebook
- World Bank API
- Git

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

# Machine Learning Pipeline

```
Download Data
      │
      ▼
Validation
      │
      ▼
Cleaning
      │
      ▼
Feature Engineering
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Train Random Forest
      │
      ▼
Evaluate Model
```

---

# Dataset

Current indicators

- GDP per Capita
- Population
- Inflation
- Life Expectancy

Engineered Features

- GDP Growth
- Population Growth
- GDP Lag
- Inflation Lag
- Life Expectancy Lag

---

# Current Results

Baseline Model

| Metric | Value |
|--------|------:|
| MAE | 8.6462 |
| RMSE | 14.8474 |
| R² | 0.1907 |

---

# Future Roadmap

## Version 1.1

- Worldwide Governance Indicators (WGI)
- Country Risk Score

## Version 1.2

- XGBoost
- LightGBM
- CatBoost
- Model Benchmark

## Version 1.3

- SHAP Explainability
- Feature Importance

## Version 1.4

- Streamlit Dashboard

## Version 2.0

- Docker
- MLflow
- GitHub Actions
- Unit Tests
- CI/CD Pipeline

---

# Installation

Clone repository

```bash
git clone https://github.com/AnderCruz/Country-Risk-Prediction.git
```

Enter the project

```bash
cd Country-Risk-Prediction
```

Create virtual environment

```bash
python -m venv .venv
```

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

# Current Status

**Version:** v1.0.0

**Status:** Active Development

---

# Author

**Anderson Cruz**

Data Scientist | Machine Learning | Predictive Analytics

LinkedIn

https://linkedin.com/in/anderjcruz

GitHub

https://github.com/AnderCruz