import os
import airflow
import logging
import csv
import random
import yaml
from marshmallow_dataclass import class_schema

from airflow import DAG
from airflow.operators.python import PythonOperator
from ml_project.pipelines.data import generate_train_data, FeatureParams

logger = logging.getLogger(__name__)


def _generate_synthetic_data(
    exec_time: str, features_config_path: str = "/opt/airflow/dags/config/features.yaml"
) -> None:
    """
    Args:
        exec_time: execution datetime in jinja datetime format for ds
    """
    logger.info("Starting data generation.")

    rows_num = random.randint(a=100, b=1000)
    features_path = f"/opt/airflow/data/raw/{exec_time}/data.csv"
    target_path = f"/opt/airflow/data/raw/{exec_time}/target.csv"

    if not os.path.exists(os.path.dirname(features_path)):
        logger.info(f"Current dir: {os.path.curdir}")
        os.makedirs(os.path.dirname(features_path))

    if not os.path.exists(os.path.dirname(target_path)):
        os.makedirs(os.path.dirname(target_path))

    with open(features_config_path, "r") as f:
        dict_config = yaml.load(f, Loader=yaml.FullLoader)
    feature_params = class_schema(FeatureParams)().load(dict_config)

    logger.info(f"Generating {rows_num} rows.")
    features, target = generate_train_data(rows_num, feature_params)

    logger.info(f"Writing features to {features_path}.")
    with open(features_path, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=",")
        csv_writer.writerow(list(features.columns.values))
        for i in range(len(features)):
            csv_writer.writerow(
                [features[name][i] for name in list(features.columns.values)]
            )

    logger.info(f"Writing targets to {target_path}.")
    with open(target_path, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=",")
        csv_writer.writerow([feature_params.target.name])
        for i in range(len(features)):
            csv_writer.writerow([target[i]])

    logger.info("Data has been generated successfully.")


with DAG(
    dag_id="data_generating",
    start_date=airflow.utils.dates.days_ago(3),
    schedule_interval="@daily",
) as dag:
    PythonOperator(
        task_id="generate_data",
        python_callable=_generate_synthetic_data,
        op_kwargs={"exec_time": "{{ execution_date | ds }}"},
    )
