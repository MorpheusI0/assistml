from typing import List, ForwardRef, Optional

from beanie import Document, Link, BackLink
from pydantic import Field, BaseModel

from .dataset import Dataset
from .implementation import Implementation

Model = ForwardRef("Model")


class UseCaseSet(BaseModel):
    task_type: str
    task_output: str


class Task(Document):
    mlsea_uri: Optional[str] = None
    use_case_set: UseCaseSet = Field(alias="UseCaseSet")
    dataset: Optional[Link[Dataset]]
    related_implementations: Optional[List[Link[Implementation]]] = Field(default_factory=list)
    #models: Optional[List[BackLink[Model]]] = Field(json_schema_extra={"original_field": "task"})

    class Settings:
        name = "tasks"
        keep_nulls = False
        validate_on_save = True

#from .model import Model
#Task.model_rebuild()
