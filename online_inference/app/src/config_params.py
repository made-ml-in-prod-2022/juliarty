import dataclasses
from os import PathLike
from typing import Union

import yaml
from marshmallow_dataclass import class_schema
import logging

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ConfigParams:
    model_path: str


ConfigParamsSchema = class_schema(ConfigParams)


def get_config_params(path: Union[str, PathLike]) -> ConfigParams:
    logger.info("Getting config file.")
    with open(path, "r") as f:
        dict_config = yaml.load(f, Loader=yaml.FullLoader)

    return ConfigParamsSchema().load(dict_config)
