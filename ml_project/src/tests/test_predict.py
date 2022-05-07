from ..pipelines.data import FeatureParams, generate_train_data
from ..pipelines.models import ModelParams
from ..pipelines.models.train_model import get_model
from ..pipelines.predict_pipeline_params import (
    PredictPipelineParams,
    get_predict_pipeline_params,
)

from ..pipelines.predict_pipeline import start_predict_pipeline
import yaml
import pytest
import os
import pickle

from ..pipelines.utils import create_directory


TEST_DIR = os.path.dirname(__file__)
TEST_DATA_PATH = os.path.join(TEST_DIR, "test_data")


def load_config(config_path: str) -> PredictPipelineParams:
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
        return get_predict_pipeline_params(config_dict)


def create_model(
    model_path: str, model_params: ModelParams, feature_params: FeatureParams
) -> None:
    create_directory(model_path)
    model = get_model(model_params)
    samples_num = 128
    features, targets = generate_train_data(samples_num, feature_params)

    model.fit(features, targets)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)


def remove_model(model_path: str) -> None:
    os.remove(model_path)


class TestPredict:
    def test_get_predict_pipeline_params(self):
        test_config_path = os.path.join(TEST_DATA_PATH, "predict_config.yaml")
        try:
            load_config(test_config_path)
        except Exception:
            pytest.fail("Problem with a configuration file.")

    def test_predict_pipeline(self):
        test_config_path = os.path.join(TEST_DATA_PATH, "predict_config.yaml")
        pipeline_params = load_config(test_config_path)
        create_model(
            pipeline_params.model_path, pipeline_params.model, pipeline_params.features
        )
        start_predict_pipeline(pipeline_params)
        remove_model(pipeline_params.model_path)
