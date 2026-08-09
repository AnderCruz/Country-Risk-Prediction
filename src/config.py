from pathlib import Path

# =============================================================================
# PROJECT
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# =============================================================================
# DATA
# =============================================================================

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

WORLD_BANK_DIR = RAW_DATA_DIR / "world_bank"

GOVERNANCE_DIR = RAW_DATA_DIR / "governance"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

REPORTS_DIR = PROJECT_ROOT / "reports"

MODEL_DIR = PROJECT_ROOT / "models"

# =============================================================================
# MERGE
# =============================================================================

MERGE_COLUMNS = [
    "country",
    "countryiso3code",
    "date",
]

# =============================================================================
# MACHINE LEARNING
# =============================================================================

FEATURE_COLUMNS = [
    "gdp_per_capita",
    "gdp_per_capita_growth",
    "inflation",
    "life_expectancy",
    "population",
    "population_growth",
    "unemployment",
    "exports",
    "gdp_lag1",
    "inflation_lag1",
    "life_expectancy_lag1",
    "economic_risk",
    "economic_risk_pca",
    "governance_risk",
]

TARGET_COLUMN = "future_country_risk"