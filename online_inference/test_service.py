import random
import logging
import click
import requests
from app.src.utils import sample_data

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

endpoints = ['predict', 'health']


def generate_endpoint_url(host: str, port: int, endpoint: str):
    return '/'.join([f"http://{host}:{port}", endpoint])


@click.command()
@click.option('--host', default='0.0.0.0', help='Host IP-address')
@click.option('--port', default=9999, help='Host port number')
@click.option('--req-num', default=10, help='Number of requests to send')
def send_requests(host, port, req_num):
    for i in range(req_num):
        endpoint = random.choice(endpoints)
        generated_url = generate_endpoint_url(host, port, endpoint)
        if endpoint == 'predict':
            response = requests.post(generated_url, json=sample_data("app/config/features.yaml"))
        elif endpoint == 'health':
            response = requests.get(generated_url)
        else:
            logger.error(f"There is no such endpoint: {endpoint}.")
            continue

        logger.info(f'Sending request: {generated_url}')
        logger.info(f'Status: {response.status_code}\nContent: {response.content}')


if __name__ == '__main__':
    send_requests()

