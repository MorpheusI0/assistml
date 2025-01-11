import os

from flask import current_app, json
from pymongo import MongoClient


def query_usecase(classif_type: str, classif_output: str) -> dict:
    """
    Generate list describing the use case task.

    :param classif_type: String to say if it's binary or multiclass.
    :param classif_output: String to say if a single prediction or probabilities are expected.
    :return: Dictionary with strings to describe the type of task and output in a standard string.
    """
    verbose = current_app.config['VERBOSE']

    usecase = {"tasktype": "", "output": ""}
    if classif_type.lower() in ["binary", "binary classification"]:
        usecase["tasktype"] = "Binary"
    elif classif_type.lower() in ["multiclass", "multiclass classification", "multi-class", "categorical"]:
        usecase["tasktype"] = "Multi-Class"

    if classif_output.lower() in ["single", "single prediction", "single_prediction"]:
        usecase["output"] = "single"
    elif classif_output.lower() in ["probs", "class probabilities", "probabilities", "multiple"]:
        usecase["output"] = "probs"

    if verbose:
        current_app.logger.info("Inside query_usecase()")
        current_app.logger.info(usecase)

    return usecase


def query_data(semantic_types: list, dataset_name: str, use_case: str) -> dict:
    """
    Generates a summary of all metadata and also gives details of each feature according to four semantic types: numeric, categorical, datetime or unstructured text.

    :param semantic_types: List with annotations of each feature semantic type (N, C, D, U, T).
    :param dataset_name: String identifying the dataset name.
    :param use_case: String identifying the use case (to be used with Mongo repository).
    :return: Dictionary describing all data features based on semantic types.
    """

    verbose = current_app.config['VERBOSE']

    if verbose:
        current_app.logger.info("Inside query_data()")
        current_app.logger.info("semantic_types: " + str(semantic_types))
        current_app.logger.info("dataset_name: " + dataset_name)
        current_app.logger.info("use_case: " + use_case)
        current_app.logger.info(" ")

    client = MongoClient(
        host=current_app.config['MONGO_HOST'],
        port=int(current_app.config['MONGO_PORT']),
        username=current_app.config['MONGO_USER'],
        password=current_app.config['MONGO_PASS']
    )
    db = client["assistml"]
    collection = db["datasets"]

    query = {"Info.use_case": use_case, "Info.dataset_name": dataset_name}
    projection = {"_id": 0}
    current_data = collection.find_one(query, projection)

    data_features = {
        "dataset_name": current_data["Info"]["dataset_name"],
        "features": current_data["Info"]["features"],
        "analyzed_observations": current_data["Info"]["analyzed_observations"],
        "observations": current_data["Info"]["observations"],
        "numeric_ratio": current_data["Info"]["numeric_ratio"],
        "categorical_ratio": current_data["Info"]["categorical_ratio"],
        "datetime_ratio": current_data["Info"]["datetime_ratio"],
        "unstructured_ratio": current_data["Info"]["unstructured_ratio"]
    }

    if "Numerical_Features" in current_data["Features"]:
        data_features["numerical_features"] = current_data["Features"]["Numerical_Features"]

    if "Categorical_Features" in current_data["Features"]:
        data_features["categorical_features"] = current_data["Features"]["Categorical_Features"]

    if "Datetime_Features" in current_data["Features"]:
        data_features["datetime_features"] = current_data["Features"]["Datetime_Features"]

    if "Unstructured_Features" in current_data["Features"]:
        data_features["unstructured_features"] = current_data["Features"]["Unstructured_Features"]

    working_dir = os.path.expanduser(current_app.config['WORKING_DIR'])
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)
    file_path = os.path.join(working_dir, "python_data_features.json")
    with open(file_path, 'w') as f:
        json.dump(data_features, f, indent=3)

    return data_features


def query_settings(lang: str, algofam: str, platform: str, tuning_limit: int) -> dict:
    """
    Generate preferred training settings for new dataset.

    :param lang: Preferred language.
    :param algofam: Must be in 3-char format.
    :param platform: Must match one of the predefined platforms or option "other".
    :param tuning_limit: Threshold number of hyperparameters that are considered acceptable.
    :return: Dictionary of technical settings for training the ML model.
    """
    if algofam not in ["DLN", "RFR", "DTR", "NBY", "LGR", "SVM", "KNN", "GBE", "GLM"]:
        raise ValueError("Error: Algorithm unknown.")

    pform = ""
    if platform.lower() in ["scikit", "sklearn", "scikit-learn"]:
        pform = "scikit"
    elif platform.lower() in ["h2o", "h2o_cluster_version", "mojo"]:
        pform = "h2o"
    elif platform.lower() in ["rweka", "weka", "r"]:
        pform = "weka"

    return {"language": lang, "algorithm": algofam, "platform": pform, "hparams": tuning_limit}


def query_preferences(accuracy_range: float, precision_range: float, recall_range: float, trtime_range: float) -> dict:
    """
    Generate query performance preferences to define acceptable performances.

    :param accuracy_range: The width of acceptable accuracy for values going from 1 to 0.
    :param precision_range: The width of acceptable precision for values going from 1 to 0.
    :param recall_range: The width of acceptable recall for values going from 1 to 0.
    :param trtime_range: The width of acceptable training time for standardized values going from 1 to 0.
    :return: Dictionary with ranges for accuracy, precision, recall, and training time.
    """
    preferences = {
        "acc_width": accuracy_range,
        "pre_width": precision_range,
        "rec_width": recall_range,
        "tra_width": trtime_range
    }
    return preferences
