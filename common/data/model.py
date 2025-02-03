from typing import Dict, Union, Optional, Any, List

from beanie import Document, Link
from pydantic import Field, BaseModel

from .implementation import Implementation
from .task import Task


class Parameter(BaseModel):
    name: str
    data_type: str
    implementation: Link[Implementation]
    value: Any
    default_value: Optional[Any] = None


class Metrics(BaseModel):
    area_under_curve: Optional[float] = None
    average_cost: Optional[float] = None
    f_measure: Optional[float] = None
    kappa: Optional[float] = None
    kononenko_branko_relative_information_score: Optional[float] = None
    mean_absolute_error: Optional[float] = None
    mean_prior_absolute_error: Optional[float] = None
    precision: Optional[float] = None
    accuracy: Optional[float] = None
    prior_entropy: Optional[float] = None
    recall: Optional[float] = None
    relative_absolute_error: Optional[float] = None
    root_mean_prior_squared_error: Optional[float] = None
    root_mean_squared_error: Optional[float] = None
    root_relative_squared_error: Optional[float] = None
    total_cost: Optional[float] = None
    training_time: Optional[float] = None


class Setup(BaseModel):
    hyper_parameters: List[Parameter]# = Field(alias="Hyper_Parameters")
    setup_string: Optional[str] = None
    implementation: Link[Implementation]
    task: Link[Task]


class EnrichedModel(BaseModel):
    fam_name: str
    rows: str
    columns_change: str

    numeric_ratio: float
    categorical_ratio: float
    datetime_ratio: float
    text_ratio: float

    training_time_std: float
    performance_score: float
    performance_gap: int

    quantile_accuracy: str
    quantile_error: str
    quantile_precision: str
    quantile_recall: str
    quantile_training_time: str

    nr_hyperparams: int
    nr_hyperparams_label: str


class Model(Document):
    mlsea_uri: Optional[str] = None
    setup: Setup = Field(alias="Setup")
    metrics: Metrics = Field(alias="Metrics")
    enriched_model: Optional[EnrichedModel] = Field(alias="Enriched_Model", default=None)


    class Settings:
        name = "models"
        keep_nulls = False
