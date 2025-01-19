from beanie import PydanticObjectId
from pydantic import BaseModel


class EnrichedModelMetricsView(BaseModel):
    id: PydanticObjectId
    accuracy: float
    precision: float
    recall: float
    training_time_std: float
    performance_score: float

    class Settings:
        projection = {
            "id": "$_id",
            "accuracy": "$Model.Base_Model.Metrics.accuracy",
            "precision": "$Model.Base_Model.Metrics.precision",
            "recall": "$Model.Base_Model.Metrics.recall",
            "training_time_std": "$Model.Enriched_Model.training_time_std",
            "performance_score": "$Model.Enriched_Model.performance_score",
        }
