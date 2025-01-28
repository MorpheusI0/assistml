from typing import List

from pydantic import BaseModel


class ReportRequestDto(BaseModel):
    classif_type: str
    classif_output: str
    sem_types: List[str]
    accuracy_range: float
    precision_range: float
    recall_range: float
    trtime_range: float
    dataset_name: str
