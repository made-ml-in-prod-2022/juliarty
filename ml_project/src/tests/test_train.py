import csv

from ..pipelines.data import generate_train_data, FeatureParams
from ..pipelines.train_pipeline_params import (
    get_training_pipeline_params,
    TrainingPipelineParams,
)
from ..pipelines.train_pipeline import start_training_pipeline
import yaml
import pytest
import os

from ..pipelines.utils import create_directory

TEST_DIR = os.path.dirname(__file__)
TEST_DATA_PATH = os.path.join(TEST_DIR, "test_data")


def load_config(config_path: str) -> TrainingPipelineParams:
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
        return get_training_pipeline_params(config_dict)


def create_test_dataset(dataset_path: str, feature_params: FeatureParams) -> None:
    """
    Saves a randomly generated dateset with features that correspond to `features_params`.
    The dataset is saved in the file `file_path` in .csv format.
    """
    create_directory(dataset_path)
    samples_num = 128
    features, target = generate_train_data(samples_num, feature_params)
    with open(dataset_path, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=",")
        csv_writer.writerow(
            list(features.columns.values) + [feature_params.target.name]
        )
        for i in range(len(features)):
            # Generator is used because iloc() and loc() change the type of all features to float64
            csv_writer.writerow(
                [features[name][i] for name in list(features.columns.values)]
                + [target[i]]
            )


def remove_test_dataset(dataset_path: str) -> None:
    os.remove(dataset_path)


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
        create_test_dataset(pipeline_params.input_data_path, pipeline_params.features)
        start_training_pipeline(pipeline_params)
        remove_test_dataset(pipeline_params.input_data_path)
