from typing import List, Optional, Dict, Any

from pydantic import field_validator, field_serializer, confloat

from common.data.utils import CustomBaseModel, encode_dict
from common.data import Model
from common.data.model import Metric


class ReportRequestDto(CustomBaseModel):
    classification_type: str
    classification_output: str
    semantic_types: List[str]
    preferences: Dict[Metric, confloat(ge=0, le=1)]
    dataset_name: Optional[str] = None
    dataset_id: str

    class Settings:
        bson_encoders = {
            Dict: encode_dict
        }

    class Config:
        pass

    @field_validator("preferences", mode="before")
    def validate_preferences(cls, v: Any) -> dict[Metric, Any]:
        return Model.validate_metrics(v)

    @field_serializer("preferences")
    def serialize_preferences(self, preferences: dict[Metric, Any], info) -> Dict[str, Any]:
        return {metric.value: value for metric, value in preferences.items()}
