import json
import os
from typing import Dict

from flask import request, jsonify, current_app
from pymongo import MongoClient
from assistml.api import assistml_bp as bp
from modules.cluster import cluster_models
from modules.query import query_usecase, query_data, query_preferences
import time

from modules.rank import rank_models, shortlist_models
from modules.results import generate_results
from modules.reticulate import python_rules
from modules.select import choose_models


@bp.route('/assistml', methods=['POST'])
def assistml():
    """
        ---
        post:
          summary: AssistML analysis for new usecase/data
          description: Recommends ML models for a given query based on a base of known trained models.
          parameters:
            - in: body
              name: body
              schema:
                type: object
                properties:
                  classif_type:
                    type: string
                    description: String to say if it is binary or multiclass
                  classif_output:
                    type: string
                    description: String to say if the result should be a single prediction or class probabilities
                  sem_types:
                    type: array
                    items:
                      type: string
                    description: JSON array with annotations for the semantic types of the data features of dataset
                  accuracy_range:
                    type: number
                    format: float
                    description: Float to state the range of top models in the accuracy dimension to be considered acceptable (0 to 1)
                  precision_range:
                    type: number
                    format: float
                    description: Float to state the range of top models in the precision dimension to be considered acceptable (0 to 1)
                  recall_range:
                    type: number
                    format: float
                    description: Float to state the range of top models in the recall dimension to be considered acceptable (0 to 1)
                  trtime_range:
                    type: number
                    format: float
                    description: Float to state the range of top models in the training time dimension to be considered acceptable (0 to 1)
                  dataset_name:
                    type: string
                    description: CSV file of the new use case with sample data to analyze
                  usecase:
                    type: string
                    description: Name of the new use case for which the query is issued
                  deployment:
                    type: string
                    description: String to say if the MLS should be deployed in a single host or cluster
                  lang:
                    type: string
                    description: String to state the preferred programming language
                  algofam:
                    type: string
                    description: String with 3-char code to specify preferred algorithm family
                  platform:
                    type: string
                    description: String to state preferred execution platform
                  tuning_limit:
                    type: integer
                    description: Int to state an acceptable number of hyper parameters to tune
                  implementation:
                    type: string
                    description: String to say how the implementation should be: single language or multi language
          responses:
            200:
              description: A JSON object containing the recommended models and additional information
              schema:
                type: object
        """
    data = request.get_json()

    classif_type = data.get('classif_type')
    classif_output = data.get('classif_output')
    sem_types = data.get('sem_types')
    accuracy_range = data.get('accuracy_range')
    precision_range = data.get('precision_range')
    recall_range = data.get('recall_range')
    trtime_range = data.get('trtime_range')
    dataset_name = data.get('dataset_name')
    usecase = data.get('usecase')
    deployment = data.get('deployment')
    lang = data.get('lang')
    algofam = data.get('algofam')
    platform = data.get('platform')
    tuning_limit = data.get('tuning_limit')
    implementation = data.get('implementation')

    verbose = current_app.config['VERBOSE']
    start_time = time.time()
    current_app.logger.info("Connecting to mongo to get base models")

    client = MongoClient(
        host=current_app.config['MONGO_HOST'],
        port=int(current_app.config['MONGO_PORT']),
        username=current_app.config['MONGO_USER'],
        password=current_app.config['MONGO_PASS']
    )
    base_models_collection = client["assistml"]["base_models"]
    enriched_models_collection = client["assistml"]["enriched_models"]
    queries_collection = client["assistml"]["queries"]

    defaults = base_models_collection.find({}, {"Model.Training_Characteristics.Dependencies.Platforms":1, "Model.Training_Characteristics.Dependencies.Libraries":1, "Model.Training_Characteristics.Hyper_Parameters.nr_hyperparams":1, "Model.Training_Characteristics.implementation":1})

    current_app.logger.info("Connecting to mongo to get enriched models")
    more_defaults = enriched_models_collection.find({})

    current_app.logger.info(" ")

    queryId = queries_collection.count_documents({}) + 1

    current_app.logger.info("Forming query record with fields...")
    current_app.logger.info(f"Query NR {queryId} Issued at {time.strftime('%Y%m%d-%H%M')} For classification {classif_type}")

    query_record: Dict[str, str] = {
        "number": queryId,
        "madeat": time.strftime('%Y%m%d-%H%M'),
        "classif_type": classif_type,
        "classif_output": classif_output,
        "dataset": dataset_name,
        "semantic_types": sem_types,
        "accuracy_range": accuracy_range,
        "precision_range": precision_range,
        "recall_range": recall_range,
        "traintime_range": trtime_range
    }

    warnings = []
    distrust_pts = 0

    queries_collection.insert_one(query_record)

    # remove non serializable objectId
    query_record.pop('_id', None)

    current_app.logger.info("#### QUERY FUNCTIONS ####")

    if verbose:
        current_app.logger.info("API call values")
        current_app.logger.info(classif_type)
        current_app.logger.info(classif_output)
        current_app.logger.info(" ")

    usecase_info = query_usecase(classif_type, classif_output)

    data_feats = query_data(semantic_types=sem_types, dataset_name=dataset_name, use_case=usecase)

    if verbose:
        current_app.logger.info(
            f"Retrieved descriptive data features for new dataset : \ncols {data_feats['features']} and \nrows {data_feats['analyzed_observations']} from originally {data_feats['observations']}")

    usecase_preferences = query_preferences(accuracy_range, precision_range, recall_range, trtime_range)

    if verbose:
        current_app.logger.info("Created performance preferences list:")
        current_app.logger.info(usecase_preferences)

    current_app.logger.info(" ")
    current_app.logger.info("#### SELECT FUNCTIONS ####")

    usecase_models, similarity_level = choose_models(task_type=usecase_info['tasktype'], output_type=usecase_info['output'],
                                   data_features=data_feats)

    if similarity_level is None:
        current_app.logger.info("No similar models could be found")
        return jsonify("No similar models could be found")

    distrust_pts += (3 - similarity_level)

    warnings.append([
                        "Dataset similarity level 0. Only the type of task and output match. Distrust Pts increased by 3",
                        "Dataset similarity level 1. Datasets used shared data types. Distrust Pts increased by 2",
                        "Dataset similarity level 2. Datasets used have similar ratios of data types. Distrust Pts increased by 1",
                        "Dataset similarity level 3. Datasets used have features with similar meta feature values. Distrust Pts increased by 0"
                    ][similarity_level])

    current_app.logger.info(f"assist(): Selected models: {len(usecase_models)} found with similarity level {similarity_level} .")

    current_app.logger.info("#### CLUSTER FUNCTIONS ####")

    usecase_mgroups = cluster_models(selected_models=usecase_models, preferences=usecase_preferences)
    current_app.logger.info(f"Acc Models: {len(usecase_mgroups['acceptable_models'])}")

    if usecase_mgroups['nearly_acceptable_models'][0] in ["none"]:
        current_app.logger.info(f"Nearly Acc Models: {usecase_mgroups['nearly_acceptable_models'][0]}")
    else:
        current_app.logger.info(f"Nearly Acc Models: {len(usecase_mgroups['nearly_acceptable_models'])}")

    if usecase_mgroups['distrust']['acceptable_models'] > 0:
        distrust_pts += usecase_mgroups['distrust']['acceptable_models']
        current_app.logger.info(f"Added {usecase_mgroups['distrust']['acceptable_models']} distrust points from ACC cut")
        warnings.append(
            f"The selection of ACC solutions was not as clean as possible. Distrust Pts increased by {usecase_mgroups['distrust']['acceptable_models']}")

    if usecase_mgroups['distrust']['nearly_acceptable_models'] > 0:
        distrust_pts += usecase_mgroups['distrust']['nearly_acceptable_models']
        current_app.logger.info(f"Added {usecase_mgroups['distrust']['nearly_acceptable_models']} distrust points from NACC cut")
        if verbose:
            current_app.logger.info(
                f"The selection of NACC solutions was not as clean as possible. Distrust Pts+{usecase_mgroups['distrust']['nearly_acceptable_models']}")
        warnings.append(
            f"The selection of NACC solutions was not as clean as possible. Distrust Pts increased by {usecase_mgroups['distrust']['nearly_acceptable_models']}")

    current_app.logger.info("#### RULES FUNCTIONS #### \n\n\n")

    if usecase_mgroups['nearly_acceptable_models'][0] in ["none"]:
        usecase_rules = python_rules(usecase_mgroups['acceptable_models'])
    else:
        usecase_rules = python_rules(usecase_mgroups['acceptable_models'] + usecase_mgroups['nearly_acceptable_models'])

    current_app.logger.info(f"Finished rules generation. {len(usecase_rules)} rules generated.")

    current_app.logger.info("#### RANK FUNCTIONS ####")

    ranked_models = rank_models(usecase_mgroups, usecase_preferences)

    models_choice = shortlist_models(ranked_models)

    current_app.logger.info("#### RESULTS FUNCTIONS ####")

    distrust_basis = 3 + 3 + 3
    current_app.logger.info("Distrust score calculation")
    current_app.logger.info(f"{distrust_pts} over {distrust_basis}")

    models_report = generate_results(models_choice, usecase_rules, warnings, distrust_pts, query_record, distrust_basis)

    working_dir = os.path.expanduser(current_app.config['WORKING_DIR'])
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)
    models_report_path = os.path.join(working_dir, "models_report.json")
    with open(models_report_path, 'w') as f:
        json.dump(models_report, f, indent=3)

    queries_collection.update_one(
        filter={"number": queryId},
        update={"$set": {"report": json.dumps(models_report)}},
        upsert=False,
    )

    current_app.logger.info("ASSISTML TERMINATED")

    end_time = time.time()
    time_taken = end_time - start_time
    current_app.logger.info(f"Time taken for end to end execution {time_taken}")

    return jsonify(models_report)
