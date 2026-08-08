from pathlib import Path

# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

EXTERNAL_DATA_DIR = DATA_DIR / "external"

# =============================================================================
# WORLD BANK INDICATORS
# =============================================================================

INDICATORS = {
    "gdp_per_capita": "NY.GDP.PCAP.CD",
    "population": "SP.POP.TOTL",
    "inflation": "FP.CPI.TOTL.ZG",
    "life_expectancy": "SP.DYN.LE00.IN",
}

# =============================================================================
# MERGE CONFIGURATION
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
    "population",
    "inflation",
    "life_expectancy",
    "population_growth",
    "gdp_lag1",
    "inflation_lag1",
    "life_expectancy_lag1",
]

TARGET_COLUMN = "gdp_growth"