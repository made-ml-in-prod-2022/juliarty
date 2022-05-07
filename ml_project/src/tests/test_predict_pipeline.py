from ..pipelines.predict_pipeline_params import (
    PredictPipelineParams,
    get_predict_pipeline_params,
)

from ..pipelines.predict_pipeline import start_predict_pipeline
import yaml
import pytest
import os


TEST_DIR = os.path.dirname(__file__)
TEST_DATA_PATH = os.path.join(TEST_DIR, "test_data")


def load_config(config_path: str) -> PredictPipelineParams:
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
        return get_predict_pipeline_params(config_dict)


class TestPredictPipeline:
    def test_get_predict_pipeline_params(self):
        test_config_path = os.path.join(TEST_DATA_PATH, "predict_config.yaml")
        try:
            load_config(test_config_path)
        except Exception:
            pytest.fail("Problem with a configuration file.")

    def test_predict_pipeline(self):
        test_config_path = os.path.join(TEST_DATA_PATH, "predict_config.yaml")
        pipeline_params = load_config(test_config_path)
        from ml_project.src.tests.utils import create_model

        create_model(
            pipeline_params.model_path, pipeline_params.model, pipeline_params.features
        )
        start_predict_pipeline(pipeline_params)

        os.remove(pipeline_params.model_path)
