"""
Country Risk Prediction

Indicator definitions used throughout the project.

Author: Anderson Cruz
"""

# =============================================================================
# WORLD BANK - ECONOMIC INDICATORS
# =============================================================================

ECONOMIC_INDICATORS = {

    # GDP
    "gdp_per_capita": "NY.GDP.PCAP.CD",
    "gdp_growth": "NY.GDP.MKTP.KD.ZG",

    # Inflation
    "inflation": "FP.CPI.TOTL.ZG",

    # Population
    "population": "SP.POP.TOTL",
    "population_growth": "SP.POP.GROW",

    # Employment
    "unemployment": "SL.UEM.TOTL.ZS",

    # Trade
    "exports": "NE.EXP.GNFS.ZS",

    # Social
    "life_expectancy": "SP.DYN.LE00.IN",

}


# =============================================================================
# WORLDWIDE GOVERNANCE INDICATORS (WGI)
# =============================================================================

GOVERNANCE_INDICATORS = {

    "voice_accountability": "VA.EST",

    "political_stability": "PV.EST",

    "government_effectiveness": "GE.EST",

    "regulatory_quality": "RQ.EST",

    "rule_of_law": "RL.EST",

    "control_corruption": "CC.EST",

}