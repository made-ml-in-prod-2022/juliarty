import numpy as np
import pandas as pd
from .model_params import ModelParams
from typing import Union
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

SklearnClassifierModel = Union[LogisticRegression, RandomForestClassifier]


def train(
    features: Union[pd.DataFrame, np.ndarray],
    target: Union[pd.Series, np.ndarray],
    model_params: ModelParams,
) -> SklearnClassifierModel:
    """
    Returns a model described in `model_params` trained on features provided.
    """
    if model_params.model_type == "LogisticRegression":
        model = LogisticRegression(**model_params.params)
    else:
        model = RandomForestClassifier(**model_params.params)

    model.fit(features, target)
    return model
