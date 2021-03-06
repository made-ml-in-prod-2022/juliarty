import os

from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd


def create_directory(file_path: str) -> None:
    """
    Creates a directory for a `file_path` if a specified directory doesn't exist.
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.mkdir(directory)


def init_hydra() -> None:
    # By default, hydra has .outputs/ as a working directory
    # That doesn't let to use relative paths in the config.yaml
    if HydraConfig.initialized():
        os.chdir(get_original_cwd())
