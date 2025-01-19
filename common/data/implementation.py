from typing import Optional, ForwardRef, List

from beanie import Document, BackLink
from pydantic import BaseModel, Field

Task = ForwardRef("Task")


class Software(BaseModel):
    mlsea_uri: Optional[str] = None
    name: str
    version: str


class Implementation(Document):
    mlsea_uri: Optional[str] = None
    title: str
    dependencies: List[Software]
    task: BackLink[Task] = Field(original_field="implementation")

#from .task import Task
#Implementation.update_forward_refs()
