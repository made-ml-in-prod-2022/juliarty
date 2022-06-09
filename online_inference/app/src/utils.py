import yaml
from marshmallow_dataclass import class_schema
from ml_project.pipelines.data import generate_train_data, FeatureParams


def sample_data(features_config_path: str = "config/features.yaml") -> dict:
    with open(features_config_path, "r") as f:
        dict_config = yaml.load(f, Loader=yaml.FullLoader)
    feature_params = class_schema(FeatureParams)().load(dict_config)
    data, _ = generate_train_data(1, feature_params)
    return {column: data[column].tolist()[0] for column in data.columns}
