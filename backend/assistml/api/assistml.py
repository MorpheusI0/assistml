import json
import os
import time

from pydantic import ValidationError
from quart import request, jsonify, current_app

from assistml.api import bp
from assistml.model_recommender import generate_report
from common.dto import ReportRequestDto
from modules.rank import rank_models, shortlist_models
from modules.results import generate_results
from modules.rules import rules


@bp.route('/assistml', methods=['POST'])
async def assistml():
    """
        ---
        post:
          summary: AssistML analysis for new data
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
    try:
        data = await request.get_json()
        report_request = ReportRequestDto(**data)
    except ValidationError as e:
        return jsonify({"error": f"Invalid request payload: {e}"}), 400
    except Exception as e:
        return jsonify({"error": f"An error occurred: {e}"}), 400


    await generate_report(report_request)

    raise NotImplementedError("Not implemented yet")

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
        usecase_rules = rules(usecase_mgroups['acceptable_models'])
    else:
        usecase_rules = rules(usecase_mgroups['acceptable_models'] + usecase_mgroups['nearly_acceptable_models'])

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
        filter={"number": query_number},
        update={"$set": {"report": json.dumps(models_report)}},
        upsert=False,
    )

    current_app.logger.info("ASSISTML TERMINATED")

    end_time = time.time()
    time_taken = end_time - start_time
    current_app.logger.info(f"Time taken for end to end execution {time_taken}")

    return jsonify(models_report)
