from pydantic import BaseModel


class AnalyseDatasetRequestDto(BaseModel):
    class_label: str
    class_feature_type: str
    feature_type_list: str
