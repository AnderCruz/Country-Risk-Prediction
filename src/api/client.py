import requests


class WorldBankClient:

    BASE_URL = "https://api.worldbank.org/v2/country/all/indicator"

    def download(self, indicator: str) -> list:
        """
        Download all pages from a World Bank indicator.
        """

        url = f"{self.BASE_URL}/{indicator}"

        page = 1

        all_records = []

        while True:

            params = {
                "format": "json",
                "page": page,
                "per_page": 1000,
            }

            response = requests.get(
                url,
                params=params,
                timeout=30,
            )

            response.raise_for_status()

            data = response.json()

            metadata = data[0]

            records = data[1]

            print(
                f"Downloading page {page}/{metadata['pages']}"
            )

            all_records.extend(records)

            if page >= metadata["pages"]:
                break

            page += 1

        return all_records