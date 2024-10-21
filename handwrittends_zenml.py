# pip install zen["Server"]
# zenml integration install sklearn -y
# pip install pyparsing=2.4.2 ----used only for colab env
# import Ipython
# Ipython.Application.Instance().kernel.do_shutdown(restart=True)

# NGROK_TOKEN="XXXXX"    ---set you NGROK token if you are working on Colab

# From zenml.environment import Environment
# if Environment.in_google_colab():   only for colab

# !pip install pyngrok
# !ngrok authtoken {NGROK_TOKEN}


# !rm -rf .zen
# zenml init


from sklearn.datasets import load_digits
from sklearn.base import ClassifierMixin
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

from zenml import step
from zenml import pipeline
from typing_extensions import Annotated
import pandas as pd
from typing import Tuple
import numpy as np

from zenml.environment import Environment


@step
def importer() -> Tuple[
    Annotated[np.ndarray, "X_train"],
    Annotated[np.ndarray, "X_test"],
    Annotated[np.ndarray, "y_train"],
    Annotated[np.ndarray, "y_test"],
]:
    """ load the digits dataset as numpy arrays """
    digits = load_digits()
    data = digits.images.reshape((len(digits.images), -1))
    X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=0.2, shuffle=True
    )
    return X_train, X_test, y_train, y_test


@step
def svc_trainer(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> ClassifierMixin:
    """ Train an Sklearn SVC Classifier"""
    model = SVC(gamma=0.001)
    model.fit(X_train, y_train)
    return model


@step
def evaluator(
    X_test: np.ndarray,
    y_test: np.ndarray,
    model: ClassifierMixin,
) -> float:
    """ Calculate the test set accuracy of the Sklearn model """
    test_acc = model.score(X_test, y_test)
    print(f" Test accuracy : {test_acc}")
    return test_acc


@pipeline
def digits_pipeline():
    """ Links all the steps together in a pipeline"""
    X_train, X_test, y_train, y_test = importer()
    model = svc_trainer(X_train=X_train, y_train=y_train)
    evaluator(X_test=X_test, y_test=y_test, model=model)


digits_svc_pipeline = digits_pipeline()

# only for colab

# def start_zenml_dashboard(port=8237):
#     if Environment.in_google_colab():
#         from pyngrok import ngrok
#         public_url = ngrok.connect(port)
#         print(f"\xlb[31mIn Colab, use this URL instead: {public_url}! \xlb [0m ")
#         !zenml up - -blocking - -port {port}

#     else:
#         !zenml up - -port {port}

# start_zenml_dashboard()
