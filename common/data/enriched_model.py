from beanie import Document


# TODO: remove this
class EnrichedModel(Document):
    model_name: str
    fam_name: str
    dataset_name: str
    use_case: str
    rows: str
    columns_change: str
    numeric_ratio: float
    numerical_encoding: str
    categorical_ratio: float
    categorical_encoding: str
    datetime_ratio: float
    datetime_encoding: str
    text_ratio: float
    text_encoding: str
    training_time_std: float
    performance_score: float
    performance_gap: float
    classification_type: str
    classification_output: str
    error: float
    quantile_error: str
    accuracy: float
    quantile_accuracy: str
    platform: str
    precision: float
    quantile_precision: str
    recall: float
    quantile_recall: str
    training_time: float
    quantile_training_time: str
    sampling: str
    language: str
    test_size: str
    nr_hyperparams: int
    nr_hyperparams_label: str
    nr_of_features: int
    nr_dependencies: int
    algorithm_implementation: str

    class Settings:
        name = "enriched_models"
        keep_nulls = False
