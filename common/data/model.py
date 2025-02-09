from enum import Enum
from typing import Optional, Any, List, Type, Dict

from beanie import Document, Link
from pydantic import field_validator
from pymongo import IndexModel

from .implementation import Implementation
from .task import Task
from .utils import CustomBaseModel, alias_generator


class Parameter(CustomBaseModel):
    name: str
    data_type: str
    implementation: Link[Implementation]
    value: Any
    default_value: Optional[Any] = None


class Metric(Enum):
    AREA_UNDER_CURVE = ("area_under_curve", "Area under curve")
    AVERAGE_COST = ("average_cost", "Average cost")
    F_MEASURE = ("f_measure", "F-measure")
    KAPPA = ("kappa", "Kappa")
    KONONENKO_BRANKO_RELATIVE_INFORMATION_SCORE = (
        "kononenko_branko_relative_information_score", "Kononenko Branko relative information score"
    )
    MEAN_ABSOLUTE_ERROR = ("mean_absolute_error", "Mean absolute error")
    MEAN_PRIOR_ABSOLUTE_ERROR = ("mean_prior_absolute_error", "Mean prior absolute error")
    PRECISION = ("precision", "Precision")
    ACCURACY = ("accuracy", "Accuracy")
    PRIOR_ENTROPY = ("prior_entropy", "Prior entropy")
    RECALL = ("recall", "Recall")
    RELATIVE_ABSOLUTE_ERROR = ("relative_absolute_error", "Relative absolute error")
    ROOT_MEAN_PRIOR_SQUARED_ERROR = ("root_mean_prior_squared_error", "Root mean prior squared error")
    ROOT_MEAN_SQUARED_ERROR = ("root_mean_squared_error", "Root mean squared error")
    ROOT_RELATIVE_SQUARED_ERROR = ("root_relative_squared_error", "Root relative squared error")
    TOTAL_COST = ("total_cost", "Total cost")
    TRAINING_TIME = ("training_time", "Training time")

    def __init__(self, key: str, display_name: str, datatype: Type = float):
        self.key = key
        self.display_name = display_name
        self.datatype = datatype

    @classmethod
    def from_key(cls, key: str) -> "Metric":
        for metric in cls:
            if metric.key == key:
                return metric
        raise ValueError(f"Key {key} is not a valid metric")


class Setup(CustomBaseModel):
    hyper_parameters: List[Parameter]
    setup_string: Optional[str] = None
    implementation: Link[Implementation]
    task: Link[Task] = None

    class Config:
        populate_by_name=True


class EnrichedModel(CustomBaseModel):
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
    setup: Setup
    metrics: Dict[str, Any]
    enriched_model: Optional[EnrichedModel] = None

    class Settings:
        name = "models"
        keep_nulls = False
        validate_on_save = True
        use_enum_values = True
        indexes = [
            IndexModel("mlseaUri", name="mlseaUri_", unique=True,
                       partialFilterExpression={"mlseaUri": {"$exists": True}}),
            IndexModel("setup.task.$id", name="setup.task.$id_"),
        ]

    class Config:
        arbitrary_types_allowed=True,
        populate_by_name = True
        alias_generator = alias_generator

    @field_validator("metrics", mode="before")
    def validate_metrics(cls, v: Any) -> Dict[str, Any]:
        if not isinstance(v, dict):
            raise ValueError("Metrics must be a dictionary")

        validated: Dict[str, Any] = {}
        for key, value in v.items():
            if isinstance(key, Metric):
                metric = key
            elif isinstance(key, str):
                try:
                    metric = Metric.from_key(key)
                except ValueError:
                    raise KeyError(f"Key {key} is not a valid metric")
            else:
                raise ValueError(f"Key {key} is not a valid metric")

            expected_type = metric.datatype
            if not isinstance(value, expected_type):
                try:
                    value = expected_type(value)
                except Exception:
                    raise ValueError(
                        f"Value {value} for metric {metric.name} must be of type {expected_type}, but is {type(value)}"
                    )

            validated[metric.key] = value
        return validated
