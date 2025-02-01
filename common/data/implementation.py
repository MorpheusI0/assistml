from __future__ import annotations

from typing import Optional, ForwardRef, List, Any, Dict, OrderedDict
from beanie import Document, BackLink, Link
from pydantic import BaseModel, Field

Task = ForwardRef("Task")


class Software(BaseModel):
    mlsea_uri: Optional[str] = None
    name: str
    version: str

class Parameter(BaseModel):
    default_value: Optional[Any] = None
    type: str
    description: Optional[str] = None

class Implementation(Document):
    mlsea_uri: Optional[str] = None
    title: str
    dependencies: List[Software]
    parameters: Dict[str, Parameter]
    components: Optional[Dict[str, Link[Implementation]]] = Field(default_factory=dict)
    description: Optional[str] = None
    task: List[BackLink[Task]] = Field(json_schema_extra={"original_field": "related_implementations"})

    class Settings:
        name = "implementations"
        keep_nulls = False
        validate_on_save = True

#from .task import Task
#Implementation.update_forward_refs()
