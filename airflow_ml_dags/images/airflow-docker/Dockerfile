ARG AIRFLOW_BASE_IMAGE="apache/airflow:2.3.0-python3.8"
FROM ${AIRFLOW_BASE_IMAGE}

RUN pip install --user --no-cache-dir \
    apache-airflow-providers-docker==2.6.0 ml_in_prod_juliarty_ml_project_1~=0.2.0 \
    PyYAML~=6.0


USER root
ENV PYTHONPATH=/home/airflow/.local/lib/python3.8/site-packages