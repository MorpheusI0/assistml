from common.data import Dataset, Implementation, Model, Task
from common.data.projection import model as model_projection
from common.data.task import TaskType

RATIO_FIELD_NAMES = [
    "categoricalRatio",
    "numericalRatio",
    "datetimeRatio",
    "unstructuredRatio"
]

def _get_sim_0_tasks_pipeline(task_type: TaskType):
    pipeline = [{
        "$match": {
            "taskType": task_type.value
        }
    }]
    return pipeline

def _get_sim_1_ratio_conditions():
    sim_1_conditions = []
    for ratio_field_name in RATIO_FIELD_NAMES:
        cur_dataset_ratio_field_path = f"$dataset.info.{ratio_field_name}"
        new_dataset_ratio_field_path = f"$newDataset.info.{ratio_field_name}"

        sim_1_conditions.append({
            "$or": [
                {
                    "$and": [
                        {"$expr": {"$ne": [cur_dataset_ratio_field_path, 0]}},
                        {"$expr": {"$ne": [new_dataset_ratio_field_path, 0]}}
                    ]
                }, {
                    "$and": [
                        {"$expr": {"$eq": [cur_dataset_ratio_field_path, 0]}},
                        {"$expr": {"$eq": [new_dataset_ratio_field_path, 0]}}
                    ]
                }
            ]
        })

    return sim_1_conditions

def _get_sim_1_tasks_pipeline(task_type: TaskType, new_dataset: Dataset):
    pipeline = _get_sim_0_tasks_pipeline(task_type)
    pipeline.extend([
         {
            "$lookup": {
                "from": Dataset.get_collection_name(),
                "localField": "dataset.$id",
                "foreignField": "_id",
                "as": "dataset"
            }
        }, {
            "$unwind": {
                "path": "$dataset"
            }
        }, {
            "$lookup": {
                "from": Dataset.get_collection_name(),
                "let": {
                    "newDatasetId": {"$toObjectId": str(new_dataset.id)}
                },
                "pipeline": [
                    {
                        "$match": {
                            "$expr": {"$eq": ["$_id", "$$newDatasetId"]}
                        }
                    }
                ],
                "as": "newDataset"
            }
        }, {
            "$unwind": {
                "path": "$newDataset"
            }
        }, {
            "$match": {
                "$expr": {"$ne": ["$dataset._id", "$newDataset._id"]}
            }
        }, {
            "$match": {
                "$and": _get_sim_1_ratio_conditions()
            }
        }
    ])

    return pipeline

def _get_sim_2_ratio_conditions(feature_ratio_tolerance: float):
    sim_2_conditions = []
    for ratio_field_name in RATIO_FIELD_NAMES:
        cur_dataset_ratio_field_path = f"$dataset.info.{ratio_field_name}"
        new_dataset_ratio_field_path = f"$newDataset.info.{ratio_field_name}"

        sim_2_conditions.append({
            "$or": [
                {
                    "$and": [
                        {"$expr": {"$gte": [cur_dataset_ratio_field_path, {"$subtract": [new_dataset_ratio_field_path, feature_ratio_tolerance]}]}},
                        {"$expr": {"$lte": [cur_dataset_ratio_field_path, {"$add": [new_dataset_ratio_field_path, feature_ratio_tolerance]}]}}
                    ]
                }, {
                    "$and": [
                        {"$expr": {"$eq": [cur_dataset_ratio_field_path, 0]}},
                        {"$expr": {"$eq": [new_dataset_ratio_field_path, 0]}}
                    ]
                }
            ]
        })

    return sim_2_conditions

def _get_sim_2_tasks_pipeline(task_type: TaskType, new_dataset: Dataset, feature_ratio_tolerance: float):
    pipeline = _get_sim_1_tasks_pipeline(task_type, new_dataset)
    pipeline.append({
        "$match": {
            "$and": _get_sim_2_ratio_conditions(feature_ratio_tolerance)
        }
    })

    return pipeline

def _build_matching_features_field_definition(features_field_name: str, new_features_field_name: str, monotonous_filtering_tolerance: float, mutual_info_tolerance: float):
    return {
        "$filter": {
            "input": f"${features_field_name}",
            "as": "feat",
            "cond": {
                "$and": [
                    # handle exception (in some cases there is no mutual_information)
                    {"$ne": ["$$feat.v.mutualInfo", None]},

                    # check if there is a matching element in newFeatures
                    {
                        "$anyElementTrue": {
                            "$map": {
                                "input": f"${new_features_field_name}",
                                "as": "newFeat",
                                "in": {
                                    "$and": [
                                        {"$gte": ["$$feat.v.monotonousFiltering", {"$subtract": ["$$newFeat.v.monotonousFiltering", monotonous_filtering_tolerance]}]},
                                        {"$lte": ["$$feat.v.monotonousFiltering", {"$add": ["$$newFeat.v.monotonousFiltering", monotonous_filtering_tolerance]}]},
                                        {"$gte": ["$$feat.v.mutualInfo", {"$subtract": ["$$newFeat.v.mutualInfo", mutual_info_tolerance]}]},
                                        {"$lte": ["$$feat.v.mutualInfo", {"$add": ["$$newFeat.v.mutualInfo", mutual_info_tolerance]}]}
                                    ]
                                }
                            }
                        }
                    }
                ]
            }
        }
    }

def _get_sim_3_tasks_pipeline(task_type: TaskType, new_dataset: Dataset, feature_ratio_tolerance: float, monotonous_filtering_tolerance: float, mutual_info_tolerance: float, similarity_ratio_tolerance: float):
    pipeline = _get_sim_2_tasks_pipeline(task_type, new_dataset, feature_ratio_tolerance)
    pipeline.extend([
        {
            "$addFields": {
                "numericalFeatures": { "$objectToArray": "$dataset.features.numericalFeatures"},
                "newNumericalFeatures": { "$objectToArray": "$newDataset.features.numericalFeatures"},
                "categoricalFeatures": { "$objectToArray": "$dataset.features.CategoricalFeatures"},
                "newCategoricalFeatures": { "$objectToArray": "$newDataset.features.CategoricalFeatures"}
            }
        }, {
            "$addFields": {
                "matchingNumerical": _build_matching_features_field_definition(
                    "numericalFeatures", "newNumericalFeatures", monotonous_filtering_tolerance, mutual_info_tolerance),
                "matchingCategorical": _build_matching_features_field_definition(
                    "categoricalFeatures", "newCategoricalFeatures", monotonous_filtering_tolerance, mutual_info_tolerance)
            }
        }, {
            "$addFields": {
                "totalMatches": { "$add": [ { "$size": "$matchingNumerical" }, { "$size": "$matchingCategorical" } ] },
                "totalFeatures": { "$add": [ { "$size": "$numericalFeatures" }, { "$size": "$categoricalFeatures" } ] },
                "similarity3": { "$cond": [
                    { "$gt": [ { "$add": [ { "$size": "$numericalFeatures" }, { "$size": "$categoricalFeatures" } ] }, 0 ] },
                    { "$divide": [
                        { "$add": [ { "$size": "$matchingNumerical" }, { "$size": "$matchingCategorical" } ] },
                        { "$add": [ { "$size": "$numericalFeatures" }, { "$size": "$categoricalFeatures" } ] }
                    ] },
                    0
                  ]
                }
            }
        }, {
            "$match": {
                "similarity3": { "$gte": similarity_ratio_tolerance }
            }
        }
    ])
    return pipeline

def _get_models_aggregation_extension():
    pipeline = [
        {
            "$set": {
                "dataset": {
                    "$mergeObjects": [
                        { "$literal": { "$ref": Dataset.get_collection_name()}},
                        {
                            "$arrayToObject": [
                                [{"k": {"$literal": "$id"}, "v": "$dataset._id"}]
                            ]
                        }
                    ]
                }
            }
        }, {
            "$lookup": {
                "from": Model.get_collection_name(),
                "localField": "_id",
                "foreignField": "setup.task.$id",
                "as": "models"
            }
        }, {
            "$unwind": {
                "path": "$models",
                "preserveNullAndEmptyArrays": False
            }
        }, {
            "$replaceRoot": {
                "newRoot": {
                    "$mergeObjects": [
                        "$models",
                        {
                            "setup": {
                                "$mergeObjects": [
                                    { "$ifNull": ["$models.setup", {}]},
                                    { "task": "$$ROOT" }
                                ]
                            }
                        }
                    ]
                }
            }
        }, {
            "$lookup": {
                "from": Implementation.get_collection_name(),
                "localField": "setup.implementation.$id",
                "foreignField": "_id",
                "as": "setup.implementation"
            }
        }, {
            "$unwind": {
                "path": "$setup.implementation",
            }
        }, {
            "$unset": "setup.task.models"
        }
    ]

    return pipeline

# Public functions to get models

async def get_sim_0_models(task_type: TaskType):
    pipeline = _get_sim_0_tasks_pipeline(task_type)
    pipeline.extend(_get_models_aggregation_extension())
    return await Task.find(with_children=True).aggregate(pipeline, projection_model=model_projection.FullyJoinedModelView).to_list()

async def get_sim_1_models(task_type: TaskType, new_dataset: Dataset):
    pipeline = _get_sim_1_tasks_pipeline(task_type, new_dataset)
    pipeline.extend(_get_models_aggregation_extension())
    return await Task.find(with_children=True).aggregate(pipeline, projection_model=model_projection.FullyJoinedModelView).to_list()

async def get_sim_2_models(task_type: TaskType, new_dataset: Dataset, feature_ratio_tolerance: float):
    pipeline = _get_sim_2_tasks_pipeline(task_type, new_dataset, feature_ratio_tolerance)
    pipeline.extend(_get_models_aggregation_extension())
    return await Task.find(with_children=True).aggregate(pipeline, projection_model=model_projection.FullyJoinedModelView).to_list()

async def get_sim_3_models(task_type: TaskType, new_dataset: Dataset, feature_ratio_tolerance: float, monotonous_filtering_tolerance: float, mutual_info_tolerance: float, similarity_ratio_tolerance: float):
    pipeline = _get_sim_3_tasks_pipeline(task_type, new_dataset, feature_ratio_tolerance, monotonous_filtering_tolerance, mutual_info_tolerance, similarity_ratio_tolerance)
    pipeline.extend(_get_models_aggregation_extension())
    return await Task.find(with_children=True).aggregate(pipeline, projection_model=model_projection.FullyJoinedModelView).to_list()
