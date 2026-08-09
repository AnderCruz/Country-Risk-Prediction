from data.sources.base import DataSource

class WorldBankSource(DataSource):
    """
    World Bank data source.
    """

    def download(self):
        raise NotImplementedError(
            "World Bank source integration will be implemented "
            "after the current pipeline is stable."
        )

    def load(self):
        raise NotImplementedError(
            "World Bank source integration will be implemented "
            "after the current pipeline is stable."
        )