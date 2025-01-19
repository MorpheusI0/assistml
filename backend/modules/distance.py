import pandas as pd
from typing import List, Dict

from quart import current_app
from pymongo import MongoClient


def retrieve_settings(selected_models: Dict[str, List[str]]) -> Dict[str, pd.DataFrame]:
    """
    Fetches training settings subtree from MongoDB and other model info currently from bundled model_data.

    Parameters:
    selected_models (Dict[str, List[str]]): List of acceptable and nearly acceptable models generated by cluster_models().

    Returns:
    Dict[str, pd.DataFrame]: List of dataframes with all information for acceptable and nearly acceptable models.
    """
    verbose = current_app.config['VERBOSE']

    # Refreshing connection to base_models
    client = MongoClient(
        host=current_app.config['MONGO_HOST'],
        port=int(current_app.config['MONGO_PORT']),
        username=current_app.config['MONGO_USER'],
        password=current_app.config['MONGO_PASS']
    )
    base_models_db = client["assistml"]["base_models"]
    enriched_models_db = client["assistml"]["enriched_models"]

    # Retrieve_Settings for ACCMODELS
    accmodel_settings = pd.json_normalize(list(base_models_db.find(
        {"Model.Info.name": {"$in": selected_models["acceptable_models"]}},
        {"Model.Training_Characteristics": 1, "_id": 0}
    )))

    if verbose:
        current_app.logger.info(f"Obtained from Mongo for ACC. Length should be a positive number: {len(accmodel_settings)}")

    accmodels_performance = pd.DataFrame(list(enriched_models_db.find(
        {"model_name": {"$in": selected_models["acceptable_models"]}}
    )))

    # Taking all fields in enriched_models, except for those already used in previous phases
    # accmodels_performance = accmodels_performance.drop(columns=["a1", "b2", "c3"])  # FIXME: set correct columns

    # descripe accessmodel_settings structure
    current_app.logger.info(f"accmodel_settings: {accmodel_settings}")

    # The number of hyper params
    accmodels_nr_hyperparams_df = accmodel_settings["Model.Training_Characteristics.Hyper_Parameters.nr_hyperparams"]
    # Specific fiels of Training_Characteristics. Should be avoided and instead be integrated in the Python scripts building enriched_models
    accmodels_specific_fields_df = accmodel_settings[["Model.Training_Characteristics." + characteristic for characteristic in ["cores", "deployment", "ghZ", "implementation", "language_version"]]]

    # Adding data from base_models Training_Characteristics to complement what is in enriched_models
    accmodels_performance = pd.concat([accmodels_performance, accmodels_nr_hyperparams_df, accmodels_specific_fields_df], axis=1)

    # Changing names of field containing similar data
    accmodels_performance.rename(columns={
        "number.rows": "rows",
        "test.size": "test_size_label",
        "number.custom.params": "nr_customparams_label",
        "Model.Training_Characteristics.Hyper_Parameters.nr_hyperparams": "nr_hyperparamas",
        "No of Cross Validation Folds Used": "cross_validation_folds"
    }, inplace=True)

    # Retrieve_Settings for NACCMODELS
    if selected_models["nearly_acceptable_models"][0] == "none":
        return {"accmodels": accmodels_performance}
    else:
        naccmodel_settings = pd.json_normalize(list(base_models_db.find(
            {"Model.Info.name": {"$in": selected_models["nearly_acceptable_models"]}},
            {"Model.Training_Characteristics": 1, "_id": 0}
        )))

        if verbose:
            current_app.logger.info(f"Obtained from Mongo for NACC. Length should be a positive number: {len(naccmodel_settings)}")

        naccmodels_performance = pd.DataFrame(list(enriched_models_db.find(
            {"model_name": {"$in": selected_models["nearly_acceptable_models"]}}
        )))

        # Taking all fields in enriched_models, except for those already used in previous phases
        #naccmodels_performance = naccmodels_performance.drop(
        #    columns=["a1", "b2", "c3"])  # FIXME: set correct columns

        # The number of hyper params
        naccmodels_nr_hyperparams_df = naccmodel_settings["Model.Training_Characteristics.Hyper_Parameters.nr_hyperparams"]
        # Specific fiels of Training_Characteristics. Should be avoided and instead be integrated in the Python scripts building enriched_models
        naccmodels_specific_fields_df = naccmodel_settings[["Model.Training_Characteristics." + characteristic for characteristic in ["cores", "deployment", "ghZ", "implementation", "language_version"]]]

        # Adding data from base_models Training_Characteristics to complement what is in enriched_models
        naccmodels_performance = pd.concat([naccmodels_performance, naccmodels_nr_hyperparams_df, naccmodels_specific_fields_df], axis=1)

        # Changing names of field containing similar data
        naccmodels_performance.rename(columns={
            "number.rows": "rows",
            "test.size": "test_size_label",
            "number.custom.params": "nr_customparams_label",
            "Model.Training_Characteristics.Hyper_Parameters.nr_hyperparams": "nr_hyperparamas",
            "No of Cross Validation Folds Used": "cross_validation_folds"
        }, inplace=True)

        return {
            "accmodels": accmodels_performance,
            "naccmodels": naccmodels_performance
        }
