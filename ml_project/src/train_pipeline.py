import os

import hydra
from hydra.utils import get_original_cwd
import logging
import models
import data
from train_pipeline_params import get_training_pipeline_params
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="train_pipeline.yaml")
def start_training(cfg: DictConfig) -> None:
    # By default, hydra has .outputs/ as a working directory
    # That doesn't let to use relative paths in the config.yaml
    os.chdir(get_original_cwd())

    logger.info("Started training.")

    logger.info("Parse the config.")
    pipeline_params = get_training_pipeline_params(dict(cfg))

    logger.info("Load data.")
    features, target = data.get_data(
        pipeline_params.input_data_path, pipeline_params.features
    )

    logger.info("Preprocess data.")
    features_train, features_test, target_train, target_test = data.split_data(
        features, target, pipeline_params.split
    )

    logger.info("Fit model.")
    model = models.train(features_train, target_train, pipeline_params.model)


if __name__ == "__main__":
    start_training()
