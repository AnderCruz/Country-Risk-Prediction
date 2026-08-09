from abc import ABC, abstractmethod


class DataSource(ABC):
    """
    Base class for every data source.
    """

    @abstractmethod
    def download(self):
        pass

    @abstractmethod
    def load(self):
        pass