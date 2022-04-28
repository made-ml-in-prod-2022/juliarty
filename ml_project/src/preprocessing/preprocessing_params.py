from dataclasses import dataclass, field
from typing import List


@dataclass()
class PreprocessingParams:
    categorical_features: List[str] = field(default_factory=list)
    transformer_type: str = field(default="OneHotTransformer")
    random_state: int = field(default=42)
