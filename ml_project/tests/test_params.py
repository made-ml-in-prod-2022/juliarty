from ..src import get_training_pipeline_params
import yaml
import pytest
import os

TEST_DIR = os.path.dirname(__file__)
TEST_DATA_PATH = os.path.join(TEST_DIR, "test_data")


class TestParams:
    def test_framework(self):
        assert True

    def test_get_training_pipeline_params(self):
        test_config_path = os.path.join(TEST_DATA_PATH, "config.yaml")
        try:
            config_dict = yaml.safe_load(open(test_config_path))
            get_training_pipeline_params(config_dict)
        except Exception:
            pytest.fail("Problem with a configuration file.")
