import asyncio
from typing import List, Optional

from bson import ObjectId
from quart import current_app

from common.data import Dataset, DatasetSimilarity, Model, Task
from common.data.task import TaskType
from common.data.projection.model import ModelView

RATIO_FIELD_NAMES = [
    "categoricalRatio",
    "numericalRatio",
    "datetimeRatio",
    "unstructuredRatio"
]

def _get_sim_1_ratio_conditions():
    sim_1_conditions = []
    for ratio_field_name in RATIO_FIELD_NAMES:
        cur_dataset_ratio_field_path = f"$info.{ratio_field_name}"
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

def _get_sim_2_ratio_conditions(feature_ratio_tolerance: float):
    sim_2_conditions = []
    for ratio_field_name in RATIO_FIELD_NAMES:
        cur_dataset_ratio_field_path = f"$info.{ratio_field_name}"
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

def _get_similar_models_pipeline(
        query_id: ObjectId,
        task_type: TaskType,
        similarity_level: int,
        limit: int = None,
        offset_id: ObjectId = None):
    pipeline = [
        *([{
            "$match": {
                "_id": {"$gt": offset_id}
            }
        }] if offset_id else []),
        {
            "$sort": {
                "_id": 1
            }
        }, {
            "$lookup": {
                "from": Task.get_collection_name(),
                "localField": "setup.task.$id",
                "foreignField": "_id",
                "pipeline": [
                    {
                        "$match": {
                            "taskType": task_type.value
                        }
                    }
                ],
                "as": "task"
            }
        }, {
            "$unwind": {
                "path": "$task",
                "preserveNullAndEmptyArrays": False
            }
        }, {
            "$lookup": {
                "from": DatasetSimilarity.get_collection_name(),
                "let": {
                    "queryId": query_id,
                    "datasetId": "$task.dataset.$id"
                },
                "pipeline": [
                    {
                        "$match": {
                            "$expr": {
                                "$and": [
                                    {"$eq": ["$queryId", "$$queryId"]},
                                    {"$eq": ["$datasetId", "$$datasetId"]}
                                ]
                            }
                        }
                    }, {
                        "$match": {
                            **({"hasSim1": True} if similarity_level >= 1 else {}),
                            **({"hasSim2": True} if similarity_level >= 2 else {}),
                            **({"hasSim3": True} if similarity_level >= 3 else {})
                        }
                    }
                ],
                "as": "dataset"
            }
        }, {
            "$unwind": {
                "path": "$dataset",
                "preserveNullAndEmptyArrays": False
            }
        },
        *([{
            "$limit": limit
        }] if limit else []),
    ]
    return pipeline

def _max_size_stage(size_mb: int):
    return {
        "$match": {
            "$expr": {
                "$lte": [
                    { "$bsonSize": "$$ROOT"},
                    size_mb * 1024 * 1024
                ]
            }
        }
    }

def _get_dataset_similarity_pipeline(
        query_id: ObjectId,
        new_dataset: Dataset,
        feature_ratio_tolerance: float,
        monotonous_filtering_tolerance: float,
        mutual_info_tolerance: float,
        similarity_ratio_tolerance: float
):
    pipeline = [
        _max_size_stage(8),
        {
            "$addFields": {
                "queryId": query_id,
                "datasetId": "$_id"
            }
        }, {
            "$unset": "_id"
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
                "path": "$newDataset",
                "preserveNullAndEmptyArrays": False
            }
        }, {
            "$match": {
                "$expr": {"$ne": ["$datasetId", "$newDataset._id"]}
            }
        }, {
            "$addFields": {
                "hasSim1": {
                    "$expr": {
                        "$and": _get_sim_1_ratio_conditions()
                    }
                },
                "hasSim2": {
                    "$expr": {
                        "$and": _get_sim_2_ratio_conditions(feature_ratio_tolerance)
                    }
                }
            }
        }, {
            "$addFields": {
                "numericalFeatures": { "$objectToArray": "$features.numericalFeatures"},
                "newNumericalFeatures": { "$objectToArray": "$newDataset.features.numericalFeatures"},
                "categoricalFeatures": { "$objectToArray": "$features.categoricalFeatures"},
                "newCategoricalFeatures": { "$objectToArray": "$newDataset.features.categoricalFeatures"}
            }
        }, {
            "$addFields": {
                "matchingNumerical": _build_matching_features_field_definition(
                    "numericalFeatures", "newNumericalFeatures",
                    monotonous_filtering_tolerance, mutual_info_tolerance),
                "matchingCategorical": _build_matching_features_field_definition(
                    "categoricalFeatures", "newCategoricalFeatures",
                    monotonous_filtering_tolerance, mutual_info_tolerance)
            }
        }, {
            "$addFields": {
                "totalMatches": { "$add": [ { "$size": "$matchingNumerical" }, { "$size": "$matchingCategorical" } ] },
                "totalFeatures": { "$add": [ { "$size": "$numericalFeatures" }, { "$size": "$categoricalFeatures" } ] },
                "similarity3": { "$cond": [
                    {
                        "$gt": [
                            { "$add": [ { "$size": "$numericalFeatures" }, { "$size": "$categoricalFeatures" } ] }, 0
                        ]
                    },
                    { "$divide": [
                        { "$add": [ { "$size": "$matchingNumerical" }, { "$size": "$matchingCategorical" } ] },
                        { "$add": [ { "$size": "$numericalFeatures" }, { "$size": "$categoricalFeatures" } ] }
                    ] },
                    0
                  ]
                }
            }
        }, {
            "$addFields": {
                "hasSim3": {
                    "$expr": { "$gte": ["$similarity3", similarity_ratio_tolerance] }
                }
            }
        }, {
            "$unset": [
                "info",
                "features",
                "newDataset",
                "numericalFeatures",
                "categoricalFeatures",
                "newNumericalFeatures",
                "newCategoricalFeatures"
            ]
        },
        _max_size_stage(14),
        {
            "$set": {
                "createdAt": { "$toDate": "$$NOW" },
            }
        }, {
            "$merge": {
                "into": "dataset_similarities",
                "on": ["queryId", "datasetId"],
                "whenMatched": "replace",
                "whenNotMatched": "insert"
            }
        }
    ]
    return pipeline

async def _execute_with_retry(async_func, max_retries: int = 5):
    backoff_time = 1  # seconds
    for try_no in range(max_retries):
        try:
            return await async_func
        except Exception as e:
            current_app.logger.error(f"Error executing function: {e}")
            if try_no == max_retries - 1:
                raise e
            current_app.logger.info(f"Retrying in {backoff_time} seconds...")
            await asyncio.sleep(backoff_time)
            backoff_time *= 2
            current_app.logger.info("Retrying...")

# Public functions to get models

async def calculate_dataset_similarity(
        query_id: ObjectId,
        new_dataset: Dataset,
        feature_ratio_tolerance: float,
        monotonous_filtering_tolerance: float,
        mutual_info_tolerance: float,
        similarity_ratio_tolerance: float
):
    pipeline = _get_dataset_similarity_pipeline(query_id, new_dataset, feature_ratio_tolerance,
                                                monotonous_filtering_tolerance, mutual_info_tolerance,
                                                similarity_ratio_tolerance)
    return await _execute_with_retry(Dataset.find().aggregate(pipeline).to_list())


async def get_similar_models(query_id: ObjectId, task_type: TaskType, similarity_level: int):
    # check if similar datasets exists
    count = await DatasetSimilarity.find({
        "queryId": query_id,
        **({"hasSim1": True} if similarity_level >= 1 else {}),
        **({"hasSim2": True} if similarity_level >= 2 else {}),
        **({"hasSim3": True} if similarity_level >= 3 else {})
    }).count()
    if count == 0:
        return []

    models_limit: Optional[int] = current_app.config["PROCESS_MODEL_LIMIT"]
    current_app.logger.info(f"{count} similar datasets found with similarity level {similarity_level}. Fetching {f'up to {models_limit}' if models_limit is not None else 'all'} related models...")
    models: List[ModelView] = []
    batch_size = 1_000
    offset_id = None

    while True:
        next_batch_size = min(batch_size, models_limit - len(models)) if models_limit is not None else batch_size
        pipeline = _get_similar_models_pipeline(query_id, task_type, similarity_level, next_batch_size, offset_id)

        batch = await _execute_with_retry(
            Model
                .find(with_children=True)
                .aggregate(aggregation_pipeline=pipeline, projection_model=ModelView)
                .to_list()
             )
        if not batch:
            break
        models.extend(batch)
        offset_id = batch[-1].id
        current_app.logger.info(f"Retrieved {len(models)} models so far")
        if models_limit is not None and len(models) >= models_limit:
            break
        if len(batch) < batch_size:
            break

    return models

async def clear_dataset_similarity_context(query_id: ObjectId):
    await _execute_with_retry(DatasetSimilarity.find({"queryId": query_id}).delete())
    current_app.logger.info("Cleared dataset similarity context")
