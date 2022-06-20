import json
import os
import pickle
import airflow
import logging
import pandas as pd

from typing import Tuple
from airflow.sensors.python import PythonSensor
from airflow.utils import yaml
from marshmallow_dataclass import class_schema
from airflow import DAG
from airflow.operators.python import PythonOperator
from ml_project.pipelines.data import FeatureParams
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from ml_project.pipelines.data import load_data
from ml_project.pipelines.preprocessing import (
    create_transformer,
    PreprocessingParams,
)
from ml_project.pipelines.models import train, ModelParams
from ml_project.pipelines.utils import get_pipeline
from ml_project.pipelines.train_pipeline import pickle_object

from utils import wait_for_data_func
from config import (
    CONFIG_FEATURES_PATH,
    CONFIG_PREPROCESSING_PATH,
    CONFIG_MODEL_PATH,
    AIRFLOW_DATA_PATH,
    DATA_PATH_FORMAT,
    TARGET_PATH_FORMAT,
)

logger = logging.getLogger(__name__)

processed_path_format = os.path.join(
    AIRFLOW_DATA_PATH, "processed/{0}/processed_data.csv"
)
train_data_path_format = os.path.join(AIRFLOW_DATA_PATH, "processed/{0}/train_data.csv")
val_data_path_format = os.path.join(AIRFLOW_DATA_PATH, "processed/{0}/val_data.csv")
output_model_path_format = os.path.join(AIRFLOW_DATA_PATH, "models/{0}/model.pkl")
output_metrics_path_format = os.path.join(AIRFLOW_DATA_PATH, "models/{0}/metrics.json")


def _prepare_train_data(execution_time: str) -> None:
    logger.info(f"Preparing train data. Datetime: {execution_time}")
    processed_data_path = processed_path_format.format(execution_time)
    data_path = DATA_PATH_FORMAT.format(execution_time)
    target_path = TARGET_PATH_FORMAT.format(execution_time)

    features_df = pd.read_csv(data_path)
    target_df = pd.read_csv(target_path)

    train_df = pd.concat([features_df, target_df], axis=1)

    logger.info(f"Saving data to {processed_data_path}")
    if not os.path.exists(os.path.dirname(processed_data_path)):
        os.makedirs(os.path.dirname(processed_data_path))

    train_df.to_csv(processed_data_path, sep=",", index=False)


def _split_train_data(execution_time: str, train_split_ratio: str = 0.8):
    processed_df = pd.read_csv(processed_path_format.format(execution_time))
    train_df, val_df = train_test_split(processed_df, train_size=train_split_ratio)
    train_df.to_csv(train_data_path_format.format(execution_time), index=False, sep=",")
    val_df.to_csv(val_data_path_format.format(execution_time), index=False, sep=",")


def _get_configs(
    features_config_path: str, preprocessing_config_path: str, model_config_path: str
) -> Tuple[FeatureParams, PreprocessingParams, ModelParams]:
    with open(features_config_path, "r") as f:
        feature_dict_config = yaml.load(f, Loader=yaml.FullLoader)
    feature_params = class_schema(FeatureParams)().load(feature_dict_config)

    with open(preprocessing_config_path, "r") as f:
        preprocessing_dict_config = yaml.load(f, Loader=yaml.FullLoader)
    preprocessing_params = class_schema(PreprocessingParams)().load(
        preprocessing_dict_config
    )

    with open(model_config_path, "r") as f:
        model_dict_config = yaml.load(f, Loader=yaml.FullLoader)
    model_params = class_schema(ModelParams)().load(model_dict_config)

    return feature_params, preprocessing_params, model_params


def _train_model(
    execution_time: str,
    features_config_path: str = CONFIG_FEATURES_PATH,
    preprocessing_config_path: str = CONFIG_PREPROCESSING_PATH,
    model_config_path: str = CONFIG_MODEL_PATH,
):
    logger.info("Training model.")

    feature_params, preprocessing_params, model_params = _get_configs(
        features_config_path, preprocessing_config_path, model_config_path
    )

    features, target = load_data(
        train_data_path_format.format(execution_time), feature_params, True
    )

    logger.info("Preprocessing data.")
    categorical_features = [f.name for f in feature_params.categorical_features]
    numerical_features = [f.name for f in feature_params.numerical_features]

    transformer = create_transformer(
        params=preprocessing_params,
        categorical_features=categorical_features,
        numerical_features=numerical_features,
    )

    features_train = transformer.fit_transform(features)
    logger.info(f"Fit transformer: {transformer.__str__()}")

    logger.info("Fitting model.")
    model = train(features_train, target, model_params)
    pipeline = get_pipeline(transformer, model)

    output_model_path = output_model_path_format.format(execution_time)
    logger.info(f"Saving model to {output_model_path}.")

    if not os.path.exists(os.path.dirname(output_model_path)):
        os.makedirs(os.path.dirname(output_model_path))

    pickle_object(pipeline, output_model_path)


def _validate_model(
    execution_time: str, features_config_path: str = CONFIG_FEATURES_PATH,
):
    metrics_to_score_func = {
        "f1-score": f1_score,
        "accuracy": accuracy_score,
    }

    logger.info("Validating model.")

    with open(output_model_path_format.format(execution_time), "rb") as f:
        model = pickle.load(f)

    with open(features_config_path, "r") as f:
        feature_dict_config = yaml.load(f, Loader=yaml.FullLoader)
    feature_params = class_schema(FeatureParams)().load(feature_dict_config)

    features, target = load_data(
        val_data_path_format.format(execution_time), feature_params, True
    )

    target_predicted = model.predict(features)

    metric_scores = {}

    for metric_name in metrics_to_score_func.keys():
        score_func = metrics_to_score_func[metric_name]
        metric_scores[metric_name] = score_func(target_predicted, target)

    with open(output_metrics_path_format.format(execution_time), "w") as f:
        json.dump(metric_scores, f)

    logger.info(f"Metrics: {metric_scores}")


with DAG(
    dag_id="train_model",
    start_date=airflow.utils.dates.days_ago(1),
    schedule_interval="@weekly",
) as dag:

    wait_for_data = PythonSensor(
        task_id="wait_for_data",
        python_callable=wait_for_data_func,
        timeout=6000,
        poke_interval=10,
        retries=100,
        mode="poke",
        op_kwargs={
            "execution_time": "{{ execution_date | ds }}",
            "data_path_format": DATA_PATH_FORMAT,
            "target_path_format": TARGET_PATH_FORMAT,
        },
    )

    prepare_train_data = PythonOperator(
        task_id="prepare_train_data",
        python_callable=_prepare_train_data,
        op_kwargs={"execution_time": "{{ execution_date | ds }}"},
    )

    split_data = PythonOperator(
        task_id="split_train",
        python_callable=_split_train_data,
        op_kwargs={"execution_time": "{{ execution_date | ds }}"},
    )

    train_model = PythonOperator(
        task_id="train_model",
        python_callable=_train_model,
        op_kwargs={"execution_time": "{{ execution_date | ds }}"},
    )

    validate_model = PythonOperator(
        task_id="validate_model",
        python_callable=_validate_model,
        op_kwargs={"execution_time": "{{ execution_date | ds }}"},
    )

    wait_for_data >> prepare_train_data >> split_data >> train_model >> validate_model
