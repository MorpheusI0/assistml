from typing import Optional, List, Any

from beanie import Document, Link
from pydantic import Field, field_validator

from .dataset import Dataset
from .model import Model
from .task import TaskType
from .utils import CustomBaseModel, alias_generator


class Summary(CustomBaseModel):
    acceptable_models: int
    nearly_acceptable_models: int
    distrust_score: float
    warnings: List[str]


class PerformanceReport(CustomBaseModel):
    accuracy: str
    precision: str
    recall: str
    training_time: str


class ModelReport(CustomBaseModel):
    name: str
    language: str
    plattform: str
    nr_hparams: int
    nr_dependencies: int
    implementation: str
    out_analysis: str
    preprocessing: str
    overall_score: float
    performance: PerformanceReport


class Report(CustomBaseModel):
    summary: Summary
    acceptable_models: List[ModelReport]
    nearly_acceptable_models: List[ModelReport] = Field(list)


class Query(Document):
    made_at: str
    task_type: TaskType
    dataset: Link[Dataset]
    semantic_types: List[str]
    preferences: dict[str, float]
    report: Optional[Report] = None

    class Settings:
        name = "queries"
        keep_nulls = False
        alias_generator = alias_generator

    @field_validator("preferences", mode="before")
    def validate_preferences(cls, v: dict[str, float]) -> dict[str, Any]:
        return Model.validate_metrics(v)
