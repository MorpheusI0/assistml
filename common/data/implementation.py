from __future__ import annotations

from typing import Optional, ForwardRef, List, Any, Dict, OrderedDict
from beanie import Document, BackLink, Link
from pydantic import Field
from pymongo import IndexModel

from .utils import CustomBaseModel, alias_generator

Task = ForwardRef("Task")


class Software(CustomBaseModel):
    mlsea_uri: Optional[str] = None
    name: str
    version: str

class Parameter(CustomBaseModel):
    default_value: Optional[Any] = None
    type: str
    description: Optional[str] = None

class Implementation(Document):
    mlsea_uri: Optional[str] = None
    title: str
    dependencies: List[Software]
    parameters: Dict[str, Parameter]
    components: Optional[Dict[str, Link[Implementation]]] = None
    description: Optional[str] = None
    task: List[BackLink[Task]] = Field(json_schema_extra={"original_field": "relatedImplementations"})

    class Settings:
        name = "implementations"
        keep_nulls = False
        validate_on_save = True
        indexes = [
            IndexModel("mlseaUri", name="mlseaUri_", unique=True,
                       partialFilterExpression={"mlseaUri": {"$exists": True}})
        ]

    class Config:
        arbitrary_types_allowed = True,
        populate_by_name = True
        alias_generator = alias_generator

#from .task import Task
#Implementation.update_forward_refs()
