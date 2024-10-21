import logging
from abc import ABC, abstractmethod

import pandas as pd
from sklearn.linear_model import LinearRegression


class Model(ABC):
    """
    Abstract base class for all models
    """

    @abstractmethod
    def train(self, x_train, y_train):
        """
        Train the model on the given data

        Args:
            x_train (_type_): Training Data
            y_train (_type_): Target Data
        """
        pass

    @abstractmethod
    def optimize(self, trial, x_train, y_train, x_test, y_test):
        """
        Optimizes the hyperparameters of the model.

        Args:
            trial: Optuna trial object
            x_train: Training data
            y_train: Target data
            x_test: Testing data
            y_test: Testing target
        """
        pass


class LinearRegressionModel(Model):
    """
    LinearRegressionModel that implements the Model interface.
    """

    def train(self, x_train, y_train, **kwargs):
        """
        Trains the model
        Args:
            x_train (_type_): Training data
            y_train (_type_): Target data
        Returns:
            None
        """
        try:
            logging.info("Model training started --Linear Regression.")
            reg = LinearRegression(**kwargs)
            reg.fit(x_train, y_train)
            logging.info("Model training completed")
            return reg
        except Exception as e:
            logging.info(
                "Error occured while training the model : {}".format(e))
            raise e

    # For linear regression, there might not be hyperparameters that we want to tune,
    # so we can simply return the score
    def optimize(self, trial, x_train, y_train, x_test, y_test):
        reg = self.train(x_train, y_train)
        return reg.score(x_test, y_test)
