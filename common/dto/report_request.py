from typing import List, Optional, Dict, Any

from pydantic import field_validator

from common.data.utils import CustomBaseModel
from common.data import Model


class ReportRequestDto(CustomBaseModel):
    classification_type: str
    classification_output: str
    semantic_types: List[str]
    preferences: Dict[str, Any]
    dataset_name: Optional[str] = None
    dataset_id: str

    @field_validator("preferences", mode="before")
    def validate_preferences(cls, v: dict[str, float]) -> dict[str, Any]:
        return Model.validate_metrics(v)
