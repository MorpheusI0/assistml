import sys
from typing import Dict, Any

from quart import current_app
from pymongo import MongoClient

from modules import explainability


def explain_python(model_code: str) -> Dict[str, Any]:
    """
    Generates an explanation for the given model code by running a Python script and retrieving the results from MongoDB.

    Parameters:
    model_code (str): The code of the model to generate explanations for.

    Returns:
    Dict[str, Any]: A dictionary containing the explanation details retrieved from MongoDB.
    """
    verbose = current_app.config['VERBOSE']

    if verbose:
        current_app.logger.info(f"Generating explanation for {model_code}")

    sys.argv = ["explainability.py", "-m", model_code]
    explainability.main()

    if verbose:
        current_app.logger.info(f"Retrieving explanations from Mongo for {model_code}")

    client = MongoClient(
        host=current_app.config['MONGO_HOST'],
        port=int(current_app.config['MONGO_PORT']),
        username=current_app.config['MONGO_USER'],
        password=current_app.config['MONGO_PASS']
    )
    base_models_db = client["assistml"]["base_models"]
    explanation = base_models_db.find_one({"Model.Info.name": model_code})

    return explanation
