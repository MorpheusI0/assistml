import os
from typing import List, Dict, Union

import pandas as pd
from flask import current_app
from pymongo import MongoClient


def rank_models(groupcodes: Dict[str, List[str]], preferences: Dict[str, List[float]]) -> Dict[str, Union[pd.DataFrame, None]]:
    """
    Rank acceptable and nearly acceptable models based on performance score and user preferences.

    Parameters:
    groupcodes (Dict[str, List[str]]): Dictionary containing lists of acceptable and nearly acceptable model codes.
    preferences (Dict[str, List[float]]): Dictionary containing user performance preferences.

    Returns:
    Dict[str, pd.DataFrame]: Dictionary containing ranked dataframes for acceptable and nearly acceptable models.
    """
    verbose = current_app.config['VERBOSE']

    working_dir = os.path.expanduser(current_app.config['WORKING_DIR'])
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)

    accms_ranked_csv_path = os.path.join(working_dir, "accms_ranked.csv")
    naccms_ranked_csv_path = os.path.join(working_dir, "naccms_ranked.csv")

    userpref = sorted(preferences.items(), key=lambda x: x[1])

    sort_priorities = []
    for key, _ in userpref:
        if key.startswith("pre"):
            sort_priorities.append("precision")
        elif key.startswith("rec"):
            sort_priorities.append("recall")
        elif key.startswith("acc"):
            sort_priorities.append("accuracy")
        elif key.startswith("tra"):
            sort_priorities.append("training_time_std")

    cols_2_include = ["model_name", "fam_name", "accuracy", "precision", "recall", "training_time_std", "performance_score"]

    if verbose:
        current_app.logger.info("Connecting to mongo to get enriched models")
    client = MongoClient(
        host=current_app.config['MONGO_HOST'],
        port=int(current_app.config['MONGO_PORT']),
        username=current_app.config['MONGO_USER'],
        password=current_app.config['MONGO_PASS']
    )
    enriched_models = client.assistml.enriched_models

    # Rank of acceptable models
    accmodels_data = enriched_models.find({"model_name": {"$in": groupcodes["acceptable_models"]}})
    accmodels_data = pd.DataFrame(accmodels_data)[cols_2_include]

    current_app.logger.info("")
    current_app.logger.info("Obtained accmodels_data")
    current_app.logger.info(accmodels_data.columns)

    accmodels_data["accuracy"] = accmodels_data["accuracy"].round(4)
    accmodels_data["precision"] = accmodels_data["precision"].round(4)
    accmodels_data["recall"] = accmodels_data["recall"].round(4)
    accmodels_data["training_time_std"] = accmodels_data["training_time_std"].round(4)
    accmodels_data["performance_score"] = accmodels_data["performance_score"].astype(float)

    accmodels_data = accmodels_data.sort_values(
        by=["performance_score"] + sort_priorities,
        ascending=[False] * (len(sort_priorities) + 1)
    )

    accmodels_data.to_csv(accms_ranked_csv_path, index=False)

    if verbose:
        current_app.logger.info(f"Finished creating acceptable models rank. Score range: {accmodels_data['performance_score'].max()} - {accmodels_data['performance_score'].min()}")

    # Rank of nearly acceptable models
    if groupcodes["nearly_acceptable_models"][0] == "none":
        #  There are no NACC models to rank
        return {"accmodels_rank": accmodels_data, "naccmodels_rank": None}

    # else: There are nacc models to rank
    naccmodels_data = enriched_models.find({"model_name": {"$in": groupcodes["nearly_acceptable_models"]}})
    naccmodels_data = pd.DataFrame(naccmodels_data)[cols_2_include]

    naccmodels_data["accuracy"] = naccmodels_data["accuracy"].round(4)
    naccmodels_data["precision"] = naccmodels_data["precision"].round(4)
    naccmodels_data["recall"] = naccmodels_data["recall"].round(4)
    naccmodels_data["training_time_std"] = naccmodels_data["training_time_std"].round(4)
    naccmodels_data["performance_score"] = naccmodels_data["performance_score"].astype(float)

    naccmodels_data = naccmodels_data.sort_values(
        by=["performance_score"] + sort_priorities,
        ascending=[False] * (len(sort_priorities) + 1)
    )

    naccmodels_data.to_csv(naccms_ranked_csv_path, index=False)

    if verbose:
        current_app.logger.info(f"Finished creating nearly acceptable models rank. Score range: {naccmodels_data['performance_score'].max()} - {naccmodels_data['performance_score'].min()}")

    return {"accmodels_rank": accmodels_data, "naccmodels_rank": naccmodels_data}


def shortlist_models(ranked_models: Dict[str, Union[pd.DataFrame, None]]) -> Dict[str, pd.DataFrame]:
    """
    Make a stratified selection of models as final result.

    Parameters:
    ranked_models (Dict[str, pd.DataFrame]): Dictionary containing ranked dataframes for acceptable and nearly acceptable models.

    Returns:
    Dict[str, pd.DataFrame]: Dictionary containing shortlisted dataframes for acceptable and nearly acceptable models.
    """
    # Set retain ratio
    retain = 0.1

    # Shortlisting the acceptable models
    accmodels = ranked_models["accmodels_rank"]
    accmodels["fam_name"] = accmodels["fam_name"].astype("category")

    accms_sample = accmodels["fam_name"].value_counts().apply(lambda x: max(1, int(x * retain)))

    accmodels_choice = pd.DataFrame()
    for fam_name, sample_size in accms_sample.items():
        accmodels_tmp = accmodels[accmodels["fam_name"] == fam_name]
        accmodels_choice = pd.concat([accmodels_choice, accmodels_tmp.head(sample_size)], ignore_index=True)

    accmodels_choice = accmodels_choice.sort_values(by="performance_score", ascending=False)

    # Shortlisting the nearly acceptable models
    if ranked_models["naccmodels_rank"] is None:
        current_app.logger.info(f"Finished choosing. Acceptable models: {len(accmodels_choice)}")
        return {"accms_choice": accmodels_choice, "naccms_choice": None}
    else:
        naccmodels = ranked_models["naccmodels_rank"]
        naccmodels["fam_name"] = naccmodels["fam_name"].astype("category")

        naccms_sample = naccmodels["fam_name"].value_counts().apply(lambda x: max(1, int(x * retain)))

        naccmodels_choice = pd.DataFrame()
        for fam_name, sample_size in naccms_sample.items():
            naccmodels_tmp = naccmodels[naccmodels["fam_name"] == fam_name]
            naccmodels_choice = pd.concat([naccmodels_choice, naccmodels_tmp.head(sample_size)], ignore_index=True)

        naccmodels_choice = naccmodels_choice.sort_values(by="performance_score", ascending=False)

        current_app.logger.info(f"Finished choosing. Acceptable models: {len(accmodels_choice)}, Nearly acceptable models: {len(naccmodels_choice)}")
        return {"accms_choice": accmodels_choice, "naccms_choice": naccmodels_choice}
