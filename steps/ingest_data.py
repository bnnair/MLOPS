import logging
import pandas as pd

from zenml import step


class IngestData():
    """_
    Ingesting the data from the data path
    """

    def __init__(self, data_path: str):
        """
        Args:
            data_path (str): _description_
        """
        self.data_path = data_path

    def get_data(self):
        """
        Ingesting the data from the data path 

        Returns:
            pd.DataFrame: returns the pandas dataframe
        """
        logging.info(f"Ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path)


@step
def ingest_data(data_path: str) -> pd.DataFrame:
    """
    Ingesting the data from the data path

    Args:
        data_path (str): path to the data

    Returns:
        pd.DataFrame: Data Frame
    """
    try:
        ingest_data = IngestData(data_path)
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(f"Error while ingesting data : {e}")
        raise e
