from typing import List, Dict, Any, Tuple, Optional

from beanie.odm.operators.find.comparison import In
from quart import current_app

from common.data import Dataset, Task, Model
from common.data.projection import dataset as dataset_projection
from common.data.projection import task as task_projection
from common.data.projection import model as model_projection


async def choose_models(task_type: str, output_type: str, data_features: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Optional[int]]:
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

    # Determine dataset similarity 0
    if verbose:
        current_app.logger.info("Checking similarity 0")

    sim_0_tasks = await Task.find(
        Task.use_case_set.task_type == task_type,
        Task.use_case_set.task_output == output_type
    ).project(task_projection.EmptyView).to_list()

    # Finding tasks with similar output and type from tasks collection
    if len(sim_0_tasks) == 0:
        # Finish assistml
        current_app.logger.info("Choose.models() Finished. No dataset similarity of any kind could be determined")
        return [], None

    sim_level = 0
    if verbose:
        current_app.logger.info(f"Tasks with similarity 0: {len(sim_0_tasks)}")

    # Determine dataset similarity 1

    # Feature ratios from the new dataset being analyzed
    data_types = ["numeric", "categorical", "datetime", "unstructured"]
    ratio_keys = [data_type + "_ratio" for data_type in data_types]
    new_data_ratios = {key: data_features[key] for key in ratio_keys}

    # Getting the data type ratios for the new dataset being used in the query.
    data_repo = await Dataset.find().project(dataset_projection.InfoView).to_list()

    if verbose:
        current_app.logger.info("Checking similarity 1")
        current_app.logger.info("Looking for matches to\n" + ", ".join([name for name, ratio in new_data_ratios.items() if ratio != 0]))

    sim_1_datasets = []
    for dataset in data_repo:
        overlap_datatypes = [key for key in ratio_keys if dataset.info[key] != 0 and new_data_ratios[key] != 0]

        if len([key for key in ratio_keys if dataset.info[key] != 0]) == len(overlap_datatypes):
            sim_1_datasets.append(dataset)
        else:
            current_app.logger.info(f"Dataset {dataset.info.dataset_name} ({dataset.id}) has NO similarity 1")

    # Keeping only the models trained on a dataset with similarity 1
    # Sim 1 codes can only be those already obtained in sim_0_tasks
    sim_1_tasks = await Task.find(
        In(Task.id, [task.id for task in sim_0_tasks]),
        In(Task.dataset.id, [dataset.id for dataset in sim_1_datasets])
    ).project(task_projection.EmptyView).to_list()
    sim_2_tasks = []
    sim_3_tasks = []

    if len(sim_1_tasks) > 0:
        sim_level = 1
        if verbose:
            current_app.logger.info(f"Tasks with similarity 1: {len(sim_1_tasks)}")

        # Determine dataset similarity 2
        if verbose:
            current_app.logger.info("")
            current_app.logger.info("Checking similarity 2")

        decile_upper = {k: v + 0.05 for k, v in new_data_ratios.items() if v != 0}
        decile_lower = {k: v - 0.05 for k, v in new_data_ratios.items() if v != 0}

        sim_1_datarepo = [data for data in data_repo if data.id in [dataset.id for dataset in sim_1_datasets]]

        sim_2_datasets = []
        # Checking whether the data type ratios of each dataset are within 1 decile of distance from the ratios of the new dataset
        for dataset in sim_1_datarepo:
            overlap_datatypes = [key for key in ratio_keys if dataset.info[key] != 0]

            sim_decile = False
            for datatype in overlap_datatypes:
                if decile_lower[datatype] <= dataset.info[datatype] <= decile_upper[datatype]:
                    sim_decile = True
                else:
                    sim_decile = False
                    break

            if sim_decile:
                current_app.logger.info(f"Dataset {dataset.info.dataset_name} ({dataset.id}) has similarity level 2")
                sim_2_datasets.append(dataset)
            else:
                current_app.logger.info(f"Dataset {dataset.info.dataset_name} ({dataset.id}) has NO similarity level 2")

        sim_2_tasks = await Task.find(
            In(Task.id, [task.id for task in sim_1_tasks]),
            In(Task.dataset.id, [dataset.id for dataset in sim_2_datasets])
        ).project(task_projection.EmptyView).to_list()

        if len(sim_2_tasks) > 0:
            sim_level = 2
            if verbose:
                current_app.logger.info(f"Tasks with similarity 2: {len(sim_2_tasks)}")

            # Determine dataset similarity 3
            if verbose:
                current_app.logger.info("")
                current_app.logger.info("Checking similarity 3")
                current_app.logger.info("")

            # Collecting feature numbers from S2 datasets
            sim_2_datasets_expanded = await Dataset.find(
                In(Dataset.id, [dataset.id for dataset in sim_2_datasets])
            ).project(dataset_projection.DatasetNameAndFeaturesView).to_list()

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

            sim_3_datasets = []
            for current_dataset in sim_2_datasets_expanded:
                current_app.logger.info(f"Checking similarity 3 for dataset {current_dataset.dataset_name}")

                # Getting all numerical metafeatures of current dataset
                current_dataset_numerical_features = current_dataset.features.numerical_features

                current_dataset_numerical_features_analysis = [{"feature_name": feature_name, **{k: feature_analytics[k] for k in ["monotonous_filtering", "anova_f1", "anova_pvalue", "mutual_info", "missing_values", "min_orderm", "max_orderm"]}} for feature_name, feature_analytics in current_dataset_numerical_features]

                # Getting all categorical metafeatures of current dataset
                current_dataset_categorical_features = current_dataset.features.categorical_features

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

                sim_3_ratio = (len(num_found) + len(cat_found)) / (len(current_dataset_numerical_features) + len(current_dataset_categorical_features))

                current_app.logger.info(f"Dataset {current_dataset.info.dataset_name} ({current_dataset.id}) has {round(sim_3_ratio * 100, 2)}% of Sim3 features")

                if sim_3_ratio >= 0.5:
                    sim_3_datasets.append(current_dataset)

            current_app.logger.info(f"sim_3_datasets: {sim_3_datasets}")

            sim_3_tasks = await Task.find(
                In(Task.id, [task.id for task in sim_2_tasks]),
                In(Task.dataset.id, [dataset.id for dataset in sim_3_datasets])
            ).project(task_projection.EmptyView).to_list()

            if len(sim_3_tasks) > 0:
                sim_level = 3
                current_app.logger.info("There are models with similarity level 3")

    sim_tasks = [sim_0_tasks, sim_1_tasks, sim_2_tasks, sim_3_tasks][sim_level]

    # TODO: Only fetch models with the metrics inside requested boundaries
    sim_models = await Model.find(
        In(Model.task.id, [task.id for task in sim_tasks])
    ).project(model_projection.EnrichedModelMetricsView).to_list()

    # Return only the columns for clustering to avoid fetching all data before it is actually needed when retrieve_settings() does that for the calculation of hamming distances
    return sim_models, sim_level
