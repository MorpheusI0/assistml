from typing import Dict, List, ForwardRef, Optional

from beanie import Document, BackLink
from pydantic import BaseModel, Field

Task = ForwardRef("Task")


class Info(BaseModel):
    mlsea_uri: Optional[str] = None
    dataset_name: str
    target_label: str
    target_feature_type: str
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


class Quantiles(BaseModel):
    q0: float
    q1: float
    q2: float
    q3: float
    q4: float
    iqr: float

    class Config:
        ser_json_inf_nan = 'constants'


class Outliers(BaseModel):
    number: int
    actual_values: list[float] = Field(alias="Actual_Values")


class Distribution(BaseModel):
    normal: bool
    exponential: bool


class NumericalFeature(BaseModel):
    monotonous_filtering: float
    anova_f1: float
    anova_pvalue: float
    mutual_info: float
    missing_values: int
    min_orderm: float = Field(allow_inf_nan=True)
    max_orderm: float = Field(allow_inf_nan=True)
    quartiles: Quantiles = Field(alias="Quartiles")
    outliers: Outliers = Field(alias="Outliers")
    distribution: Distribution = Field(alias="Distribution")

    class Config:
        ser_json_inf_nan = 'constants'


class CategoricalFeature(BaseModel):
    missing_values: int
    nr_levels: int
    levels: Dict[str, str] = Field(alias="Levels")
    imbalance: float
    mutual_info: float
    monotonous_filtering: float

    class Config:
        ser_json_inf_nan = 'constants'


class UnstructuredFeature(BaseModel):
    missing_values: int
    vocab_size: int
    relative_vocab: float
    vocab_concentration: float
    entropy: float
    min_vocab: int
    max_vocab: int

    class Config:
        ser_json_inf_nan = 'constants'


class DatetimeFeature(BaseModel):
    pass


class Features(BaseModel):
    numerical_features: Optional[Dict[str, NumericalFeature]] = Field(alias="Numerical_Features", default=None)
    categorical_features: Optional[Dict[str, CategoricalFeature]] = Field(alias="Categorical_Features", default=None)
    unstructured_features: Optional[Dict[str, UnstructuredFeature]] = Field(alias="Unstructured_Features", default=None)
    datetime_features: Optional[Dict[str, DatetimeFeature]] = Field(alias="Datetime_Features", default=None)


class Dataset(Document):
    info: Info = Field(alias="Info")
    features: Features = Field(alias="Features")
    tasks: List[BackLink[Task]] = Field(json_schema_extra={"original_field": "dataset"})

    class Settings:
        name = "datasets"
        #keep_nulls = False
        #validate_on_save = True. # breaks dataset insertion

#from .task import Task
#Dataset.update_forward_refs()
