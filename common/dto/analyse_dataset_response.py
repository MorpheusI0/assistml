from typing import Optional

from pydantic import BaseModel, Field

from common.data.dataset import Info, Features


class DatasetInfoDto(BaseModel):
    info: Info = Field(alias="Info")
    features: Features = Field(alias="Features")

class AnalyseDatasetResponseDto(BaseModel):
    data_profile: Optional[DatasetInfoDto] = None
    db_write_status: str
