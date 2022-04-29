from ..pipelines.train_pipeline_params import (
    get_training_pipeline_params,
    TrainingPipelineParams,
)
from ..pipelines.train_pipeline import start_training_pipeline
import yaml
import pytest
import os

TEST_DIR = os.path.dirname(__file__)
TEST_DATA_PATH = os.path.join(TEST_DIR, "test_data")


def load_config(config_path: str) -> TrainingPipelineParams:
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
        return get_training_pipeline_params(config_dict)


class TestTrain:
    def test_get_training_pipeline_params(self):
        test_config_path = os.path.join(TEST_DATA_PATH, "train_config.yaml")
        try:
            load_config(test_config_path)
        except Exception:
            pytest.fail("Problem with a configuration file.")

    def test_train_pipeline(self):
        test_config_path = os.path.join(TEST_DATA_PATH, "train_config.yaml")
        pipeline_params = load_config(test_config_path)
        start_training_pipeline(pipeline_params)
