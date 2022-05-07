import csv
import pickle

from ml_project.src.pipelines.data import FeatureParams, generate_train_data
from ml_project.src.pipelines.models import ModelParams
from ml_project.src.pipelines.models.train_model import get_model
from ml_project.src.pipelines.utils import create_directory


def create_random_dataset(dataset_path: str, feature_params: FeatureParams) -> None:
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
