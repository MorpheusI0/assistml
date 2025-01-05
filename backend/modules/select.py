from typing import List, Dict, Any, Tuple, Optional

from flask import current_app
from pymongo import MongoClient


def choose_models(task_type: str, output_type: str, data_features: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    """
    Filters out models by filtering based on characteristics of the usecase task and dataset.

    :param task_type: String to describe the kind of classification task (binary or multiclass).
    :param output_type: String to describe the type of output expected.
    :param data_features: Dictionary describing all the details of the dataset features.
    :return: A tuple containing a list of dictionaries with the model rows that match the description criteria and the similarity level.
    """
    verbose = current_app.config['VERBOSE']
    if verbose:
        current_app.logger.info("Entered choose_models() with values:")
        current_app.logger.info(task_type)
        current_app.logger.info(output_type)

    client = MongoClient(
        host=current_app.config['MONGO_HOST'],
        port=int(current_app.config['MONGO_PORT']),
        username=current_app.config['MONGO_USER'],
        password=current_app.config['MONGO_PASS']
    )
    datasetsMongo = client.assistml.datasets
    base_models = client.assistml.base_models
    enriched_models = client.assistml.enriched_models

    # Determine dataset similarity 0
    if verbose:
        current_app.logger.info("Checking similarity 0")

    sim_0 = list(base_models.find({}, {"Model.Data_Meta_Data.classification_output": 1, "Model.Data_Meta_Data.classification_type": 1, "Model.Info.name": 1, "_id": 0}))

    # Finding the actual models with similar output and type from base models
    sim_0_codes = [model["Model"]["Info"]["name"] for model in sim_0 if model["Model"]["Data_Meta_Data"]["classification_type"] == task_type and model["Model"]["Data_Meta_Data"]["classification_output"] == output_type]

    if len(sim_0_codes) > 0:
        sim_level = 0
        if verbose:
            current_app.logger.info(f"Models with similarity 0: {len(sim_0_codes)}")

        # Determine dataset similarity 1

        # Feature ratios from the new dataset being analyzed
        type_ratios = {
            "numeric_ratio": data_features["numeric_ratio"],
            "categorical_ratio": data_features["categorical_ratio"],
            "datetime_ratio": data_features["datetime_ratio"],
            "unstructured_ratio": data_features["unstructured_ratio"]
        }

        # Getting the data type ratios for the new dataset being used in the query.
        data_repo = list(datasetsMongo.find({}, {"Info": 1, "_id": 0}))
        data_repo = [{k: v for k, v in data["Info"].items() if k in ["use_case", "numeric_ratio", "categorical_ratio", "datetime_ratio", "unstructured_ratio"]} for data in data_repo]

        if verbose:
            current_app.logger.info("Checking similarity 1")
            current_app.logger.info("Looking for matches to\n" + ", ".join([name for name, ratio in type_ratios.items() if ratio != 0]))

        sim_1_usecases = []
        for data in data_repo:
            overlap_datatypes = [key for key in
                                 ["numeric_ratio", "categorical_ratio", "datetime_ratio", "unstructured_ratio"]
                                 if data[key] != 0 and key in [k for k in type_ratios.keys() if type_ratios[k] != 0]]

            if len([key for key in ["numeric_ratio", "categorical_ratio", "datetime_ratio", "unstructured_ratio"] if
                    data[key] != 0]) == len(overlap_datatypes):
                sim_1_usecases.append(data["use_case"])
            else:
                current_app.logger.info(f"Dataset {data['use_case']} has NO similarity 1")

        # Keeping only the models trained on a dataset with similarity 1
        # Sim 1 codes can only be those already obtained in sim_0_codes
        sim_1_codes = [model["Model"]["Info"]["name"] for model in base_models.find({"Model.Info.name": {"$in": sim_0_codes}}, {"Model.Info.name": 1, "Model.Info.use_case": 1, "_id": 0}) if model["Model"]["Info"]["use_case"] in sim_1_usecases]
        sim_2_codes = []
        sim_3_codes = []

        if len(sim_1_codes) > 0:
            sim_level = 1
            if verbose:
                current_app.logger.info(f"Models with similarity 1: {len(sim_1_codes)}")

            # Determine dataset similarity 2
            if verbose:
                current_app.logger.info("")
                current_app.logger.info("Checking similarity 2")

            decile_upper = {k: v + 0.05 for k, v in type_ratios.items() if v != 0}
            decile_lower = {k: v - 0.05 for k, v in type_ratios.items() if v != 0}

            sim_1_datarepo = [data for data in data_repo if data["use_case"] in sim_1_usecases]

            sim_2_usecases = []
            # Checking whether the data type ratios of each dataset are within 1 decile of distance from the ratios of the new dataset
            for data in sim_1_datarepo:
                overlap_datatypes = [key for key in
                                     ["numeric_ratio", "categorical_ratio", "datetime_ratio", "unstructured_ratio"]
                                     if data[key] != 0]

                sim_decile = False
                for datatype in overlap_datatypes:
                    if decile_lower[datatype] <= data[datatype] <= decile_upper[datatype]:
                        sim_decile = True
                    else:
                        sim_decile = False
                        break

                if sim_decile:
                    current_app.logger.info(f"Use case {data['use_case']} has similarity level 2")
                    sim_2_usecases.append(data["use_case"])
                else:
                    current_app.logger.info(f"Use case {data['use_case']} has NO similarity level 2")

            sim_2_codes = [model["Model"]["Info"]["name"] for model in base_models.find({"Model.Info.name": {"$in": sim_1_codes}}, {"Model.Info.name": 1, "Model.Info.use_case": 1, "_id": 0}) if model["Model"]["Info"]["use_case"] in sim_2_usecases]

            if len(sim_2_codes) > 0:
                sim_level = 2
                if verbose:
                    current_app.logger.info(f"Models with similarity 2: {len(sim_2_codes)}")

                # Determine dataset similarity 3
                if verbose:
                    current_app.logger.info("")
                    current_app.logger.info("Checking similarity 3")
                    current_app.logger.info("")

                # Collecting feature numbers from S2 datasets
                all_datasets_feature_information = [{k: data["Info"][k] for k in ["dataset_name", "features"]} for data in datasetsMongo.find({"Info.use_case": {"$in": sim_2_usecases}}, {"Info.dataset_name": 1, "Info.features": 1, "_id": 0})]

                # Getting all numerical metafeatures of the new dataset
                new_dataset_numerical_features_analysis = []
                for feature_name, feature_analytics in data_features["numerical_features"]:
                    new_dataset_numerical_features_analysis.append({
                        "feature_name": feature_name,
                        "monotonous_filtering": feature_analytics["monotonous_filtering"],
                        "anova_f1": feature_analytics["anova_f1"],
                        "anova_pvalue": feature_analytics["anova_pvalue"],
                        "mutual_info": feature_analytics["mutual_info"],
                        "missing_values": feature_analytics["missing_values"],
                        "min_orderm": feature_analytics["min_orderm"],
                        "max_orderm": feature_analytics["max_orderm"]
                    })

                # Getting all categorical metafeatures of the new dataset
                new_dataset_categorical_features_analysis = []
                for feature_name, feature_analytics in data_features["categorical_features"]:
                    new_dataset_categorical_features_analysis.append({
                        "feature_name": feature_name,
                        "missing_values": feature_analytics["missing_values"],
                        "nr_levels": feature_analytics["nr_levels"],
                        "imbalance": feature_analytics["imbalance"],
                        "mutual_info": feature_analytics["mutual_info"],
                        "monotonous_filtering": feature_analytics["monotonous_filtering"]
                    })

                s3_datasets = []
                for current_dataset_features_information in all_datasets_feature_information:
                    current_app.logger.info(f"Checking similarity 3 for dataset {current_dataset_features_information['dataset_name']}")

                    # Getting all numerical metafeatures of current dataset
                    current_dataset_numerical_features = datasetsMongo.find_one({"Info.dataset_name": current_dataset_features_information["dataset_name"]}, {"Features.Numerical_Features": 1, "_id": 0})["Features"]["Numerical_Features"]

                    current_dataset_numerical_features_analysis = [{"feature_name": feature_name, **{k: feature_analytics[k] for k in ["monotonous_filtering", "anova_f1", "anova_pvalue", "mutual_info", "missing_values", "min_orderm", "max_orderm"]}} for feature_name, feature_analytics in current_dataset_numerical_features]

                    # Getting all categorical metafeatures of current dataset
                    current_dataset_categorical_features = datasetsMongo.find_one({"Info.dataset_name": current_dataset_features_information["dataset_name"]}, {"Features.Categorical_Features": 1, "_id": 0})["Features"]["Categorical_Features"]

                    current_dataset_categorical_features_analysis = [{"feature_name": feature_name, **{k: feature_analytics[k] for k in ["missing_values", "nr_levels", "imbalance", "mutual_info", "monotonous_filtering"]}} for feature_name, feature_analytics in current_dataset_categorical_features]

                    # Find for every numerical feature in the NEW dataset, a similar feature from the current dataset
                    if verbose:
                        current_app.logger.info("Numerical features sim 3 check")

                    num_found = set() # changed from list to set
                    for new_numerical_feature_analysis in new_dataset_numerical_features_analysis:
                        # Checking monotonous filtering values
                        mono_matches = [current_feat for current_feat in current_dataset_numerical_features_analysis if new_numerical_feature_analysis["monotonous_filtering"] - 0.05 <= current_feat["monotonous_filtering"] <= new_numerical_feature_analysis["monotonous_filtering"] + 0.05]

                        # Checking mutual information values of those picked from monotonous values
                        mutual_matches = [current_feat for current_feat in mono_matches if new_numerical_feature_analysis["mutual_info"] - 0.02 <= current_feat["mutual_info"] <= new_numerical_feature_analysis["mutual_info"] + 0.02]
                        mutual_matches_names = [match['feature_name'] for match in mutual_matches]

                        current_app.logger.info(f"Found {len(mutual_matches)} matches for numerical feature {new_numerical_feature_analysis['feature_name']}: {mutual_matches_names}")
                        num_found.update(mutual_matches_names)

                    # Find for every categorical feature in the NEW dataset, a similar feature from the current dataset
                    if verbose:
                        current_app.logger.info("Categorical features sim 3 check")

                    cat_found = set() # changed from list to set
                    for new_categorical_feature_analysis in new_dataset_categorical_features_analysis:
                        # Checking monotonous filtering values
                        mono_matches = [current_feat for current_feat in current_dataset_categorical_features_analysis if new_categorical_feature_analysis["monotonous_filtering"] - 0.05 <= current_feat["monotonous_filtering"] <= new_categorical_feature_analysis["monotonous_filtering"] + 0.05]

                        # Checking mutual information values of those picked from monotonous values
                        mutual_matches = [current_feat for current_feat in mono_matches if new_categorical_feature_analysis["mutual_info"] - 0.02 <= current_feat["mutual_info"] <= new_categorical_feature_analysis["mutual_info"] + 0.02]
                        mutual_matches_names = [match['feature_name'] for match in mutual_matches]

                        current_app.logger.info(f"Found {len(mutual_matches)} matches for categorical feature {new_categorical_feature_analysis['feature_name']}: {mutual_matches_names}")
                        cat_found.update(mutual_matches_names)

                    s3_ratio = (len(num_found) + len(cat_found)) / (len(current_dataset_numerical_features) + len(current_dataset_categorical_features))

                    current_app.logger.info(f"Dataset {current_dataset_features_information['dataset_name']} has {round(s3_ratio * 100, 2)}% of S3 features")

                    if s3_ratio >= 0.5:
                        s3_datasets.append(current_dataset_features_information["dataset_name"])

                current_app.logger.info(f"s3_datasets: {s3_datasets}")

                sim_3_codes = [model["Model"]["Info"]["name"] for model in base_models.find({"Model.Info.name": {"$in": sim_2_codes}}, {"Model.Info.name": 1, "Model.Info.use_case": 1, "_id": 0, "Model.Data_Meta_Data.dataset_name": 1}) if model["Model"]["Data_Meta_Data"]["dataset_name"] in s3_datasets]

                if len(sim_3_codes) > 0:
                    sim_level = 3
                    current_app.logger.info("There are models with similarity level 3")

        sim_codes = [sim_0_codes, sim_1_codes, sim_2_codes, sim_3_codes][sim_level]

        sim_models = list(enriched_models.find({"model_name": {"$in": sim_codes}}, {"model_name": 1, "accuracy": 1, "precision": 1, "recall": 1, "training_time_std": 1, "performance_score": 1, "_id": 0}))

        # Casting the metrics as numerics
        for model in sim_models:
            model["accuracy"] = float(model["accuracy"])
            model["precision"] = float(model["precision"])
            model["recall"] = float(model["recall"])
            model["training_time_std"] = float(model["training_time_std"])
            model["performance_score"] = float(model["performance_score"])

        # Return only the columns for clustering to avoid fetching all data before it is actually needed when retrieve_settings() does that for the calculation of hamming distances
        return sim_models, sim_level
    else:
        # Finish assistml
        current_app.logger.info("Choose.models() Finished. No dataset similarity of any kind could be determined")
        return [], None
