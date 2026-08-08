# Country Risk Prediction Platform

## Objective

Build an end-to-end Machine Learning platform capable of predicting sovereign country risk using macroeconomic and governance indicators from international public data sources.

---

## Data Sources

### World Bank API

Current indicators:

- GDP per Capita
- Population
- Inflation
- Life Expectancy

Future sources:

- Worldwide Governance Indicators (WGI)
- IMF
- OECD

---

## Project Pipeline

World Bank API
        ↓
Download Indicators
        ↓
Raw CSV Files
        ↓
Merge Datasets
        ↓
Data Validation
        ↓
Feature Engineering
        ↓
Machine Learning Dataset
        ↓
Model Training
        ↓
Prediction
        ↓
Dashboard

---

## Project Structure

Country-Risk-Prediction/

data/
    raw/
    processed/

src/
    api/
    data/
    features/
    models/
    visualization/
    utils/

docs/

tests/

---

## Future Improvements

- Automated pipeline
- Model explainability (SHAP)
- Dashboard
- REST API
- Docker
- CI/CD