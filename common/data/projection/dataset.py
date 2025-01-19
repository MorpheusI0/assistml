from beanie import PydanticObjectId
from pydantic import BaseModel, Field

from common.data.dataset import Info, Features


class EmptyView(BaseModel):
    id: PydanticObjectId

    class Settings:
        projection = {"id": "$_id"}


class InfoView(BaseModel):
    id: PydanticObjectId
    info: Info = Field(alias="Info")

    class Settings:
        projection = {"id": "$_id", "Info": 1}


class DatasetNameAndFeaturesView(BaseModel):
    id: PydanticObjectId
    dataset_name: str
    features: Features = Field(alias="Features")

    class Settings:
        projection = {"id": "$_id", "dataset_name": "$Info.dataset_name", "Features": 1}
