from pathlib import Path

# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

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

    # Economic Indicators
    "gdp_per_capita",
    "inflation",
    "life_expectancy",
    "population",
    "population_growth",
    "unemployment",
    "exports",

    # Time Series Features
    "gdp_lag1",
    "inflation_lag1",
    "life_expectancy_lag1",

    # Composite Indicator
    "economic_score",

]

TARGET_COLUMN = "gdp_growth"