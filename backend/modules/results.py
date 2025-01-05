import pandas as pd
from typing import Dict, List, Any, Union

from flask import current_app
from pymongo import MongoClient

from modules.distance import retrieve_settings
from modules.explain import explain_python


def generate_model_report(picked_model: str, details: pd.DataFrame) -> Dict[str, Any]:
    """
    Creates a dictionary with human-readable details of a single model.

    Parameters:
    picked_model (str): String with the model code.
    details (pd.DataFrame): DataFrame containing the query details relevant for Hamming.

    Returns:
    Dict[str, Any]: Dictionary with model details in human-readable form.
    """
    verbose = current_app.config['VERBOSE']
    if verbose:
        current_app.logger.info(f"Entered generate_model_report for {picked_model}")
        current_app.logger.info(details)

    client = MongoClient(
        host=current_app.config['MONGO_HOST'],
        port=int(current_app.config['MONGO_PORT']),
        username=current_app.config['MONGO_USER'],
        password=current_app.config['MONGO_PASS']
    )
    base_models_db = client["assistml"]["base_models"]
    enriched_models_db = client["assistml"]["enriched_models"]

    # Pull all MongoDB and model_data data useful about this model
    accmodel_json = base_models_db.find_one({"Model.Info.name": picked_model})
    accmodel_data = enriched_models_db.find_one({"model_name": picked_model})

    # Process the data and generate the report
    model_name = {
        "DTR": "Model trained with Decision Trees",
        "RFR": "Model trained with Random Forests",
        "LGR": "Model trained with Logistic Regression",
        "SVM": "Model trained with Support Vector Machines",
        "NBY": "Model trained with Naive Bayes",
        "DLN": "Model trained with Deep Learning",
        "GBE": "Model trained with Gradient Boosting Ensemble",
        "GLM": "Model trained with General Linear Model"
    }.get(accmodel_data["fam_name"], "Unknown model family")

    if accmodel_json["Model"]["Metrics"].get("Explainability") is None:
        # try to get the explainability data from the explainability module
        accmodel_json = explain_python(accmodel_json["Model"]["Info"]["name"])
        # if the explainability data is still not available, skip the explainability part
        if accmodel_json["Model"]["Metrics"].get("Explainability") is None:
            accmodel_json["Model"]["Metrics"]["Explainability"] = {
                "None": {
                    "total": 0.0
                }
            }

    feats_totals = {name: metric["total"] for name, metric in accmodel_json["Model"]["Metrics"]["Explainability"].items()}

    confmat_text = f"Suitable to detect variations in data like {', '.join([name for name, total in feats_totals.items() if total < 0.05])}. " \
                   f"Unsuitable to detect variations in data like {', '.join([name for name, total in feats_totals.items() if total > 0.4])}."

    preprocessing_description = ""
    for datatype in ["numerical", "categorical", "datetime", "text"]:
        if accmodel_json["Model"]["Data_Meta_Data"]["Preprocessing"][f"{datatype}_encoding"] not in ["None", "none"]:
            preprocessing_description += f"The {datatype} data is read (with) {accmodel_json['Model']['Data_Meta_Data']['Preprocessing'][f'{datatype}_encoding']}. "
        if accmodel_json["Model"]["Data_Meta_Data"]["Preprocessing"][f"{datatype}_selection"] not in ["None", "none"]:
            preprocessing_description += f"The {datatype} data was filtered based on {accmodel_json['Model']['Data_Meta_Data']['Preprocessing'][f'{datatype}_selection']}. "

    current_app.logger.info(f"Generated report for {picked_model}")

    report = {
        "name": model_name,
        "language": accmodel_json["Model"]["Training_Characteristics"]["language"],
        "platform": accmodel_json["Model"]["Training_Characteristics"]["algorithm_implementation"],
        "nr_hparams": accmodel_json["Model"]["Training_Characteristics"]["Hyper_Parameters"]["nr_hyperparams"],
        "nr_dependencies": accmodel_json["Model"]["Training_Characteristics"]["Dependencies"]["nr_dependencies"],
        "implementation": accmodel_json["Model"]["Training_Characteristics"]["implementation"],
        "deployment": accmodel_json["Model"]["Training_Characteristics"]["deployment"],
        "cores": accmodel_json["Model"]["Training_Characteristics"]["cores"],
        "power": accmodel_json["Model"]["Training_Characteristics"]["ghZ"],
        "out_analysis": confmat_text,
        "preprocessing": preprocessing_description,
        "overall_score": round(details["performance_score"], 4),
        "performance": {
            "accuracy": accmodel_data["quantile_accuracy"][4:],
            "precision": accmodel_data["quantile_precision"][4:],
            "recall": accmodel_data["quantile_recall"][4:],
            "training_time": accmodel_data["quantile_training_time"][4:]
        },
        "code": picked_model,
        "rules": "None"
    }

    if verbose:
        current_app.logger.info(f"Generated report for {picked_model}")

    return report


def add_rules(model_report: Dict[str, Any], found_rules: List[Any]) -> List[str]:
    """
    Selects relevant rules to include for the model based on its described report contents.

    Parameters:
    model_report (Dict[str, Any]): Dictionary containing the report of a single model.
    found_rules (List[Dict[str, Any]]): Full list of generated rules of acceptable and nearly acceptable models.

    Returns:
    List[str]: Array of strings containing the rules that mention the algorithm family used in the supplied model.
    """
    verbose = current_app.config['VERBOSE']
    if verbose:
        current_app.logger.info("Entered add_rules()")
        current_app.logger.info(found_rules[:5])

    # Find association rules for the specified model components
    model_leads = model_report["code"].split("_")
    relevant_rules = []

    for rule in found_rules:
        if not rule.get("full_rule") or not model_leads[0]:
            continue
        if model_leads[0] in rule["full_rule"]:  # Checks if the fam name matches that of the model being reported
            if verbose:
                current_app.logger.info(f"Found a rule for family name {model_leads[0]}")
            if len(relevant_rules) < 10:  # To restrict the maximum number of rules that are given back
                rulename = rule["full_rule"].replace(":", "").replace("'", "")
                rule_begins = [i for i, char in enumerate(rulename) if char == "["]
                rule_stops = [i for i, char in enumerate(rulename) if char == "]"]
                rel_rule = f"IF ML solution has {rulename[rule_begins[0]:rule_stops[0]]} then {rulename[rule_begins[1]:rule_stops[1]]}. "
                relevant_rules.append(rel_rule)

        for key in ["nr_dependencies", "nr_hparams"]:
            if f"{key}_{model_report[key]}" in rule["full_rule"]:
                if len(relevant_rules) < 10:  # To restrict the maximum number of rules that are given back
                    rulename = rule["full_rule"].replace(":", "").replace("'", "")
                    rule_begins = [i for i, char in enumerate(rulename) if char == "["]
                    rule_stops = [i for i, char in enumerate(rulename) if char == "]"]
                    rel_rule = f"IF ML solution has {rulename[rule_begins[0]:rule_stops[0]]} then {rulename[rule_begins[1]:rule_stops[1]]}. "
                    relevant_rules.append(rel_rule)

    return relevant_rules


def generate_results(models_choice: Dict[str, pd.DataFrame], usecase_rules: List[Dict[str, Union[str, float, List[str]]]], warnings: List[str], distrust_points: int, query_record: Dict[str, str], distrust_basis: int) -> Dict[str, Any]:
    """
    Produces object and JSON with results of the analysis process.

    Parameters:
    models_choice (Dict[str, pd.DataFrame]): List of two dataframes containing only the selected acc and nearly acc models.
    usecase_rules (List[Dict[str, Any]]): Full list of all generated rules for both acc and nearly acc models.
    warnings (List[str]): List of warnings generated during the process.
    distrust_points (int): Number of distrust points accumulated.
    query_record (Dict[str, Any]): Dictionary containing the query fields relevant for Hamming calculations.
    distrust_basis (int): Basis for calculating the distrust score.

    Returns:
    Dict[str, Any]: Dictionary with human-readable details of chosen acceptable and nearly acceptable models.
    """
    verbose = current_app.config['VERBOSE']

    # Produces final report with all the results

    if models_choice["naccms_choice"] is None:
        models_choice_dfs = retrieve_settings(selected_models={
            "acceptable_models": models_choice["accms_choice"]["model_name"].astype(str).tolist(),
            "nearly_acceptable_models": ["none"]
        })
        if verbose:
            current_app.logger.info(" ")
            current_app.logger.info(f"Obtained dataframes to get distance details for {len(models_choice_dfs['accmodels'])} acceptable models")
    else:
        models_choice_dfs = retrieve_settings(selected_models={
            "acceptable_models": models_choice["accms_choice"]["model_name"].astype(str).tolist(),
            "nearly_acceptable_models": models_choice["naccms_choice"]["model_name"].astype(str).tolist()
        })
        if verbose:
            current_app.logger.info(" ")
            current_app.logger.info(f"Obtained dataframes to get distance details for {len(models_choice_dfs['accmodels'])} acceptable models and {len(models_choice_dfs['naccmodels'])} nearly acceptable models")

    if verbose:
        current_app.logger.info("Acceptable Solutions")

    accms_report = []
    for i in range(len(models_choice["accms_choice"])):
        model_name = models_choice["accms_choice"]["model_name"][i]
        details = models_choice["accms_choice"].iloc[i, 1:]

        model_report = generate_model_report(model_name, details)
        model_report["rules"] = add_rules(model_report, usecase_rules)

        accms_report.append(model_report)

    results = {
        "summary": {
            "query_issued": query_record,
            "acceptable_models": len(accms_report),
            "distrust_score": round(distrust_points / distrust_basis, 4),
            "warnings": warnings
        },
        "acceptable_models": accms_report
    }

    if any(choice != "none" for choice in models_choice["naccms_choice"]):
        #  Case there are nearly acceptable solutions chosen
        if verbose:
            current_app.logger.info("Nearly Acceptable Solutions")

        naccms_report = []
        for j in range(len(models_choice["naccms_choice"])):
            model_name = models_choice["naccms_choice"]["model_name"][j]
            details = models_choice["naccms_choice"].iloc[j, 1:]

            model_report = generate_model_report(model_name, details)
            model_report["rules"] = add_rules(model_report, usecase_rules)

            naccms_report.append(model_report)

        results["summary"]["nearly_acceptable_models"] = len(naccms_report)
        results["nearly_acceptable_models"] = naccms_report

    return results
