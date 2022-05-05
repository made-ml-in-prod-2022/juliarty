import pandas as pd
import numpy as np
import logging

from typing import Tuple, Union
from sklearn.model_selection import train_test_split
from .features_params import FeatureParams
from .split_params import SplittingParams

logger = logging.getLogger(__name__)


def get_data(
    data_path: str, feature_params: FeatureParams, with_target=False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
    """
    Gets features and target from **.csv** file according `feature_params`.

    Args:
        with_target: Whether to add target column to output
        feature_params: Describes all features and target to be included.
        data_path: Path to a csv file.

    Returns: Tuple[pd.DataFrame, pd.Series] (features, target) or pd.DataFrame (features).
    """
    df = pd.read_csv(data_path)

    if with_target:
        return df[feature_params.all_features], df[feature_params.target]
    else:
        return df[feature_params.all_features]


def split_data(
    features: Union[pd.DataFrame, np.array],
    target: Union[pd.Series, np.array],
    split_params: SplittingParams,
) -> Tuple[
    Union[pd.DataFrame, np.array],
    Union[pd.DataFrame, np.array],
    Union[pd.Series, np.array],
    Union[pd.Series, np.array],
]:
    """
    Splits features, target on train, test.

    Returns: Tuple (features_train, features_test, target_train, target_test)
    """
    features_train, features_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=split_params.test_size,
        random_state=split_params.random_state,
    )
    logger.info(f"Split data. Test size: {split_params.test_size}")
    return features_train, features_test, y_train, y_test
