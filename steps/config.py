from zenml.steps import BaseParameters


class ModelNameConfig(BaseParameters):
    """ Model configurations """

    selected_model: str = "LinearRegression"
    fine_tuning: bool = False
