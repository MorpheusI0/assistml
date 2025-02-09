from enum import Enum
from typing import Dict, List, ForwardRef, Optional

from beanie import Document, BackLink
from pydantic import Field
from pymongo import IndexModel

from common.data.utils import CustomBaseModel, alias_generator

Task = ForwardRef("Task")


class TargetFeatureType(str, Enum):
    BINARY = "binary"
    MULTICLASS = "multiclass"
    REGRESSION = "regression"

class Info(CustomBaseModel):
    mlsea_uri: Optional[str] = None
    dataset_name: str
    target_label: str
    target_feature_type: TargetFeatureType
    observations: int
    analyzed_observations: int
    features: int
    numeric_ratio: float
    categorical_ratio: float
    datetime_ratio: float
    unstructured_ratio: float
    analyzed_features: list[str]
    discarded_features: list[str]
    analysis_time: float


class Quantiles(CustomBaseModel):
    q0: float
    q1: float
    q2: float
    q3: float
    q4: float
    iqr: float


class Outliers(CustomBaseModel):
    number: int
    actual_values: list[float]


class Distribution(CustomBaseModel):
    normal: bool
    exponential: bool


class NumericalFeature(CustomBaseModel):
    monotonous_filtering: float
    anova_f1: float
    anova_pvalue: float
    mutual_info: Optional[float] = None  # Does not exist for regression
    missing_values: int
    min_orderm: float
    max_orderm: float
    quartiles: Quantiles
    outliers: Outliers
    distribution: Distribution


class CategoricalFeature(CustomBaseModel):
    missing_values: int
    nr_levels: int
    levels: Dict[str, str]
    imbalance: float
    mutual_info: float
    monotonous_filtering: float


class UnstructuredFeature(CustomBaseModel):
    missing_values: int
    vocab_size: int
    relative_vocab: float
    vocab_concentration: float
    entropy: float
    min_vocab: int
    max_vocab: int


class DatetimeFeature(CustomBaseModel):
    pass


class Features(CustomBaseModel):
    numerical_features: Dict[str, NumericalFeature]
    categorical_features: Dict[str, CategoricalFeature]
    unstructured_features: Dict[str, UnstructuredFeature]
    datetime_features: Dict[str, DatetimeFeature]


class Dataset(Document):
    info: Info
    features: Features
    tasks: List[BackLink[Task]] = Field(json_schema_extra={"original_field": "dataset"})

    class Settings:
        name = "datasets"
        keep_nulls = False
        validate_on_save = True
        indexes = [
            IndexModel("info.mlseaUri", name="info.mlseaUri_", unique=True,
                       partialFilterExpression={"info.mlseaUri": {"$exists": True}})
        ]

    class Config:
        arbitrary_types_allowed = True
        populate_by_name = True
        use_enum_values = True
        alias_generator = alias_generator


#from .task import Task
#Dataset.update_forward_refs()
