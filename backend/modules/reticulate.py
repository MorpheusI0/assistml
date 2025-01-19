import json
import os
import sys
import time
from typing import List, Dict, Union

from quart import current_app
from pymongo import MongoClient

from modules import data_encoder, association_python, analysis


def python_rules(model_codes: List[str]) -> List[Dict[str, Union[str, float, List[str]]]]:
    """
    Generates rules for the given model codes using Python scripts and retrieves the rules from MongoDB.

    Parameters:
    model_codes (List[str]): List of model codes for which to generate rules.

    Returns:
    List[Dict[str, Any]]: List of rules retrieved from MongoDB.
    """
    verbose = current_app.config['VERBOSE']

    if verbose:
        current_app.logger.info("Entered python_rules() with the following chosen models")
        current_app.logger.info(model_codes)
        current_app.logger.info("")

    working_dir = os.path.expanduser(current_app.config['WORKING_DIR'])
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)

    python_rules_json_path = os.path.join(working_dir, "python_rules.json")

    current_app.logger.info("Calling python data_encoder.py...")
    # Triggers execution of python modules with mlxtend to generate rules
    sys.argv = ["data_encoder.py", "[fam_name,nr_hyperparams_label,performance_gap,quantile_accuracy,quantile_recall,quantile_precision,platform,quantile_training_time,nr_dependencies]", f'[{",".join(model_codes)}]']
    data_encoder.main()

    current_app.logger.info("Calling python association_python.py")
    sys.argv = ["association_python.py", "0", "0.7", "0.25"]
    association_python.main()

    current_app.logger.info("Calling python analysis.py")
    sys.argv = ["analysis.py", "0.5", "0.01", "1.2"]
    analysis.main()

    if verbose:
        current_app.logger.info("Retrieving last added rules summary from Mongo")

    rulestamp = time.strftime("%Y%m%d-%H%M")
    if verbose:
        current_app.logger.info("Retrieving rules for experiment inserted at:")
        current_app.logger.info(rulestamp)

    client = MongoClient(
        host=current_app.config['MONGO_HOST'],
        port=int(current_app.config['MONGO_PORT']),
        username=current_app.config['MONGO_USER'],
        password=current_app.config['MONGO_PASS']
    )
    db = client.assistml
    rules_collection = db.rules

    current_setofrules = list(rules_collection.find_one(
        {"Rules": {"$exists": True}, "Experiment.created": rulestamp},  # FIXME: minute might not be correct
        {"Rules": True}
    )["Rules"].values())

    if len(current_setofrules) > 0:
        if verbose:
            current_app.logger.info("Storing rules as json")
        with open(python_rules_json_path, 'w') as f:
            json.dump(current_setofrules, f, indent=3)
        return current_setofrules
    else:
        if verbose:
            current_app.logger.info("No rules were found nor filtered")
        return []
