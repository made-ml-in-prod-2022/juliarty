import csv
import os

import pickle
import airflow
import logging

from airflow import AirflowException
from airflow.models import Variable
from airflow.sensors.python import PythonSensor
from airflow.utils import yaml
from marshmallow_dataclass import class_schema
from airflow import DAG
from airflow.operators.python import PythonOperator
from ml_project.pipelines.data import FeatureParams
from ml_project.pipelines.data import load_data
from config import (
    CONFIG_FEATURES_PATH,
    AIRFLOW_DATA_PATH,
    DATA_PATH_FORMAT,
)
from utils import wait_for_data_func

logger = logging.getLogger(__name__)

output_predictions_path_format = os.path.join(
    AIRFLOW_DATA_PATH, "predictions/{0}/predictions.csv"
)


def _predict(execution_time: str, features_config_path: str = CONFIG_FEATURES_PATH):
    model_path = Variable.get("model_path")

    if not os.path.exists(model_path):
        logger.error(f"There is no model with the path: {model_path}.")
        raise AirflowException()

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(features_config_path, "r") as f:
        feature_dict_config = yaml.load(f, Loader=yaml.FullLoader)
    feature_params = class_schema(FeatureParams)().load(feature_dict_config)

    features = load_data(DATA_PATH_FORMAT.format(execution_time), feature_params, False)

    logger.info("Predicting...")
    target_predicted = model.predict(features)

    logger.info("Saving results...")
    output_path = output_predictions_path_format.format(execution_time)

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    with open(output_path, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=";")
        for label in target_predicted:
            csv_writer.writerow([label])


with DAG(
    dag_id="inference",
    start_date=airflow.utils.dates.days_ago(3),
    schedule_interval="@daily",
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
            "target_path_format": None,
        },
    )

    predict = PythonOperator(
        task_id="inference",
        python_callable=_predict,
        op_kwargs={"execution_time": "{{ execution_date | ds }}"},
    )

    wait_for_data >> predict
