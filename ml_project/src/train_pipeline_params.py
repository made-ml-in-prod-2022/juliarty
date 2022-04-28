from dataclasses import dataclass
from data.features_params import FeatureParams
from data.split_params import SplittingParams
from preprocessing.preprocessing_params import PreprocessingParams
from models.model_params import ModelParams
from marshmallow_dataclass import class_schema


@dataclass()
class TrainingPipelineParams:
    input_data_path: str
    output_model_path: str
    metric_path: str
    features: FeatureParams
    model: ModelParams
    preprocessing: PreprocessingParams
    split: SplittingParams


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)


def get_training_pipeline_params(dict_config: dict) -> TrainingPipelineParams:
    return TrainingPipelineParamsSchema().load(dict_config)
