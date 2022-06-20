import os

AIRFLOW_DAGS_PATH = "/opt/airflow/dags/"
AIRFLOW_DATA_PATH = "/opt/airflow/data/"
CONFIG_FEATURES_PATH = f"{AIRFLOW_DAGS_PATH}/config/features.yaml"
CONFIG_PREPROCESSING_PATH = f"{AIRFLOW_DAGS_PATH}/config/preprocessing.yaml"
CONFIG_MODEL_PATH = f"{AIRFLOW_DAGS_PATH}/config/model.yaml"

DATA_PATH_FORMAT = os.path.join(AIRFLOW_DATA_PATH, "raw/{0}/data.csv")
TARGET_PATH_FORMAT = os.path.join(AIRFLOW_DATA_PATH, "raw/{0}/target.csv")
