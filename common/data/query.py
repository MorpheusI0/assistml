from typing import Optional, List

from beanie import Document
from pydantic import Field, BaseModel


class Summary(BaseModel):
    acceptable_models: int
    nearly_acceptable_models: int
    distrust_score: float
    warnings: List[str]


class PerformanceReport(BaseModel):
    accuracy: str
    precision: str
    recall: str
    training_time: str


class ModelReport(BaseModel):
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


class Report(BaseModel):
    summary: Summary
    acceptable_models: List[ModelReport]


class Query(Document):
    number: int
    made_at: str = Field(alias="madeat")
    classif_type: str
    classif_output: str
    dataset: str
    semantic_types: List[str]
    accuracy_range: float
    precision_range: float
    recall_range: float
    traintime_range: float
    report: Optional[Report] = None

    class Settings:
        name = "queries"
        keep_nulls = False
