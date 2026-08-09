import time
import requests

class WorldBankClient:

    BASE_URL = "https://api.worldbank.org/v2/country/all/indicator"

    def download(self, indicator: str) -> list:

        url = f"{self.BASE_URL}/{indicator}"

        page = 1

        all_records = []

        while True:

            params = {
                "format": "json",
                "page": page,
                "per_page": 1000,
            }

            success = False

            for attempt in range(3):

                try:

                    response = requests.get(
                        url,
                        params=params,
                        timeout=90,
                    )

                    response.raise_for_status()

                    success = True

                    break

                except requests.exceptions.RequestException:

                    print(
                        f"Retry {attempt+1}/3 (page {page})..."
                    )

                    time.sleep(3)

            if not success:

                raise Exception(
                    f"Failed downloading {indicator}"
                )

            data = response.json()

            # -------------------------------------------------------------
            # Validate response
            # -------------------------------------------------------------

            if len(data) < 2:

                print(f"\nInvalid response for indicator: {indicator}")
                print(data)

                raise Exception(
                    f"World Bank API returned no data for {indicator}"
                )

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