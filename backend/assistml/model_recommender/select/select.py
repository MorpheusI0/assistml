from assistml.model_recommender.select.aggregation_pipelines import get_sim_0_models, get_sim_1_models, \
    get_sim_2_models, get_sim_3_models
from common.data import Dataset, Query
from common.data.projection.model import FullyJoinedModelView

TOLERANCES = {"feature_ratio": 0.05, "monotonous_filtering": 0.05, "mutual_info": 0.02, "similarity_ratio": 0.5}


async def select_models_on_dataset_similarity(query: Query) -> tuple[list[FullyJoinedModelView], int]:
    new_dataset: Dataset = await query.dataset.fetch()
    if not new_dataset:
        raise ValueError("Dataset not found")

    sim_3_models = await get_sim_3_models(query.task_type, new_dataset, TOLERANCES["feature_ratio"],
                                          TOLERANCES["monotonous_filtering"], TOLERANCES["mutual_info"],
                                          TOLERANCES["similarity_ratio"])
    if len(sim_3_models) > 0:
        return sim_3_models, 3

    sim_2_models = await get_sim_2_models(query.task_type, new_dataset, TOLERANCES["feature_ratio"])
    if len(sim_2_models) > 0:
        return sim_2_models, 2

    sim_1_models = await get_sim_1_models(query.task_type, new_dataset)
    if len(sim_1_models) > 0:
        return sim_1_models, 1

    sim_0_models = await get_sim_0_models(query.task_type)
    return sim_0_models, 0
