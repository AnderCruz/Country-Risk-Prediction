# Country Risk Composite Index (CRCI)

Version: 1.0

---

# 1. Purpose

The Country Risk Composite Index (CRCI) is an explainable indicator designed to estimate the overall economic and institutional risk of sovereign countries using publicly available data.

The CRCI combines macroeconomic fundamentals and governance indicators into a single reproducible score.

Unlike proprietary country risk methodologies, the CRCI is fully transparent and based on open data.

---

# 2. Objectives

The CRCI aims to:

- Measure country risk using open data.
- Produce a reproducible index.
- Support Machine Learning models.
- Provide explainable predictions.
- Serve as an alternative to proprietary methodologies.

---

# 3. Conceptual Framework

Country Risk

↓

Economic Strength

+

Institutional Quality

↓

Composite Score

---

# 4. Economic Dimension

Indicators

- GDP per Capita
- GDP Growth
- Inflation
- Population Growth

Future Indicators

- Government Debt
- Current Account
- Exports
- Imports
- Unemployment

---

# 5. Governance Dimension

Worldwide Governance Indicators

- Voice and Accountability
- Political Stability
- Government Effectiveness
- Regulatory Quality
- Rule of Law
- Control of Corruption

---

# 6. Data Preparation

Before calculating the CRCI all variables will be:

- validated
- cleaned
- standardized
- checked for missing values

Continuous variables will be normalized.

---

# 7. Initial Formula

Version 1.0

CRCI = Average(Standardized Indicators)

Each indicator receives the same weight.

---

# 8. Future Versions

Version 2.0

Principal Component Analysis (PCA)

↓

Estimate optimal weights.

---

Version 3.0

Machine Learning Feature Importance

↓

Dynamic weighting.

---

# 9. Validation

The CRCI will be evaluated by:

- Internal consistency
- Correlation analysis
- Machine Learning performance
- Sensitivity analysis

---

# 10. Expected Outputs

The project will generate:

- Country Risk Dataset
- Country Risk Composite Index
- Machine Learning Models
- Explainability Reports
- Dashboard