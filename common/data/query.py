from typing import Optional, List, Any, Dict

from beanie import Document, Link
from pydantic import Field, field_validator, confloat

from .dataset import Dataset
from .model import Model, Metric
from .task import TaskType
from .utils import CustomBaseModel, alias_generator, encode_dict


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
    preferences: Dict[Metric, confloat(ge=0, le=1)]
    report: Optional[Report] = None

    class Settings:
        name = "queries"
        keep_nulls = False
        validate_on_save = True
        alias_generator = alias_generator
        bson_encoders = {
            Dict: encode_dict
        }

    @field_validator("preferences", mode="before")
    def validate_preferences(cls, v: Any) -> dict[Metric, Any]:
        return Model.validate_metrics(v)

    @field_validator("task_type", mode="before")
    def convert_task_type(cls, value):
        if isinstance(value, str):
            try:
                return TaskType(value)
            except KeyError:
                raise ValueError(f"Invalid task_type: {value}")
        return value
