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
    related_implementations: List[Link[Implementation]] = Field(original_field="task", default=list)
    models: List[BackLink[Model]] = Field(original_field="task")

    class Settings:
        name = "tasks"
        keep_nulls = False

#from .model import Model
#Task.update_forward_refs()
