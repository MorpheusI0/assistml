from abc import ABC
from enum import Enum
from typing import List, ForwardRef, Optional

from beanie import Document, Link, BackLink
from pydantic import Field, BaseModel

from .dataset import Dataset
from .implementation import Implementation

Model = ForwardRef("Model")


class TaskType(str, Enum):
    SUPERVISED_CLASSIFICATION = "Supervised_Classification"
    SUPERVISED_REGRESSION = "Supervised_Regression"
    LEARNING_CURVE = "Learning_Curve"
    SUPERVISED_DATASTREAM_CLASSIFICATION = "Supervised_Datastream_Classification"
    CLUSTERING = "Clustering"
    MACHINE_LEARNING_CHALLENGE = "Machine_Learning_Challenge"
    SURVIVAL_ANALYSIS = "Survival_Analysis"
    SUBGROUP_DISCOVERY = "Subgroup_Discovery"
    MULTITASK_REGRESSION = "Multitask_Regression"

class Task(Document):
    mlsea_uri: Optional[str] = None
    task_type: TaskType
    dataset: Optional[Link[Dataset]]
    related_implementations: Optional[List[Link[Implementation]]] = Field(default_factory=list)
    #models: Optional[List[BackLink[Model]]] = Field(json_schema_extra={"original_field": "task"})

    class Settings:
        name = "tasks"
        keep_nulls = False
        validate_on_save = True
        use_enum_values = True
        is_root = True


class SupervisedTask(Task, ABC):
    target_name: str

class ClassificationTask(SupervisedTask):
    class_labels: Optional[List[str]] = None

class RegressionTask(SupervisedTask):
    pass

class ClusteringTask(Task):
    target_name: Optional[str] = None

class LearningCurveTask(ClassificationTask):
    pass

#from .model import Model
#Task.model_rebuild()
