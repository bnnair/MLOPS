import logging
import pandas as pd

from zenml import step
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig
import mlflow

from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def train_model(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: ModelNameConfig,
) -> RegressorMixin:
    """
    Trains the model on ingested data

    Args:
        x_train (pd.DataFrame): training data
        x_test (pd.DataFrame): testing data
        y_train (pd.Series): training target data
        y_test (pd.Series): testing target data

    Returns:
        RegressorMixin: the regression model returned
    """

    model = None
    try:
        if config.selected_model == "LinearRegression":
            mlflow.autolog()
            model = LinearRegressionModel()
            trained_model = model.train(x_train=x_train, y_train=y_train)
            return trained_model
        else:
            raise ValueError(
                "Model {} not supported".format(config.model_name))
    except Exception as e:
        logging.error("Error in training model {}".format(e))
        raise e
