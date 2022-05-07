import numpy as np
import pandas as pd

from ml_project.src.pipelines.data import generate_train_data, FeatureParams
from ml_project.src.pipelines.data.features_params import (
    NumericalFeatureParams,
    CategoricalFeatureParams,
)


def check_numerical_feature(values: pd.Series, feature_params: NumericalFeatureParams):
    if feature_params.type == "discrete":
        assert values.dtype == np.int64 or values.dtype == np.int32
    elif feature_params.type == "continuous":
        assert values.dtype == np.float64 or values.dtype == np.float32
    else:
        assert False

    assert np.all(
        np.logical_and(
            values >= feature_params.min,
            values <= feature_params.max,
        )
    )


def check_categorical_feature(
    values: pd.Series, feature_params: CategoricalFeatureParams
):
    assert np.all(values.apply(lambda x: x in feature_params.categories))


class TestDataModule:
    def test_data_generation(self):
        discrete_feature_range = [-100, 100]
        categorical_features_categories = [0, 1, 2]
        continuous_feature_range = [0, 1]
        target_categories = [0, 1]
        discrete_1 = NumericalFeatureParams(
            name="discrete_1",
            type="discrete",
            min=discrete_feature_range[0],
            max=discrete_feature_range[1],
        )

        continuous_1 = NumericalFeatureParams(
            name="continuous_1",
            type="continuous",
            min=continuous_feature_range[0],
            max=continuous_feature_range[1],
        )

        categorical_1 = CategoricalFeatureParams(
            name="cat_1", categories=categorical_features_categories
        )
        target_feature_params = CategoricalFeatureParams(
            name="target", categories=target_categories
        )

        feature_params = FeatureParams(
            all_features=[
                discrete_1.name,
                continuous_1.name,
                categorical_1.name,
            ],
            numerical_features=[discrete_1, continuous_1],
            categorical_features=[categorical_1],
            target=target_feature_params,
            features_to_drop=[],
        )

        features, target = generate_train_data(100, feature_params)

        check_numerical_feature(features[discrete_1.name], discrete_1)
        check_numerical_feature(features[continuous_1.name], continuous_1)

        check_categorical_feature(features[categorical_1.name], categorical_1)
        check_categorical_feature(target, target_feature_params)
