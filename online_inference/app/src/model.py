import logging
import os
import pickle
import gdown

from sklearn.pipeline import Pipeline


MODEL_URL_ENV_VAR_NAME = "MODEL_GDRIVE_URL"

logger = logging.getLogger(__name__)


def download_artefacts(model_url: str, model_path: str) -> None:
    """
    If model and transformer don't exist, they will be downloaded from Google Drive.
    Args:
        model_url: The url to download from model.
        model_path: The path to write the model.
    """
    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))
    logger.info("Downloading the model...")
    gdown.download(model_url, model_path, quiet=False)


def get_inference_pipeline(model_path: str) -> Pipeline:
    model_url = os.environ[MODEL_URL_ENV_VAR_NAME]
    download_artefacts(model_url, model_path)
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model
