from dataclasses import dataclass
from typing import List, Optional


@dataclass()
class FeatureParams:
    all_features: List[str]
    features_to_drop: List[str]
    target: Optional[str]
