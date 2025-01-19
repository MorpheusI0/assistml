from typing import Dict, Union, Optional

import pydantic
from beanie import Document, Link
from pydantic import Field

from .task import Task


class Preprocessing(pydantic.BaseModel):
    categorical_encoding: str
    numerical_encoding: str
    categorical_selection: str
    date_encoding: str
    text_encoding: str
    numerical_selection: str
    datetime_encoding: str
    datetime_selection: str
    text_selection: str


class DataMetaData(pydantic.BaseModel):
    rows: int
    classification_type: str
    classification_output: str
    class_variable: str
    categorical_colummns: int
    numeric_columns: int
    datetime_columns: int
    text_columns: int
    list_of_columns_used: list[str]
    dataset_name: str
    cols_pre_preprocessing: int
    preprocessing: Preprocessing = Field(alias="Preprocessing")
    cols_afr_preprocessing: int


class Dependencies(pydantic.BaseModel):
    nr_dependencies: int
    libraries: Dict[str, str] = Field(alias="Libraries")


class TrainingCharacteristics(pydantic.BaseModel):
    hyper_parameters: Dict[str, Union[str, int]] = Field(alias="Hyper_Parameters", default_factory=dict)
    test_size: float
    seed_value: int
    cross_validation_folds: int
    sampling: str
    algorithm_implementation: str
    language: str
    language_version: str
    cores: str
    ghZ: str
    deployment: str
    implementation: str
    dependencies: Dependencies = Field(alias="Dependencies")
    platform: str


class Metrics(pydantic.BaseModel):
    accuracy: float
    error: float
    precision: float
    recall: float
    fscore: float
    cross_validation_training_time: float
    test_time_per_unit: float
    confusion_matrix_rowstrue_colspred: list[int]
    test_file: str
    training_time: float


class Info(pydantic.BaseModel):
    mlsea_uri: Optional[str] = None
    name: str
    spec_version: str
    use_case: str


class BaseModel(pydantic.BaseModel):
    data_meta_data: DataMetaData = Field(alias="Data_Meta_Data")
    training_characteristics: TrainingCharacteristics = Field(alias="Training_Characteristics")
    metrics: Metrics = Field(alias="Metrics")
    info: Info = Field(alias="Info")


class EnrichedModel(pydantic.BaseModel):
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
    base_model: BaseModel = Field(alias="Base_Model")
    enriched_model: Optional[EnrichedModel] = Field(alias="Enriched_Model")
    task: Link[Task]

    class Settings:
        name = "models"
        keep_nulls = False
