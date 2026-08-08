"""
World Bank datasource.

Responsible for downloading macroeconomic indicators.
"""

from api.client import WorldBankClient


class WorldBankDatasource:

    def __init__(self):

        self.client = WorldBankClient()

    def download(self, indicator):

        return self.client.download(indicator)