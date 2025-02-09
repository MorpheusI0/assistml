import time

from assistml.model_recommender.cluster import cluster_models
from assistml.model_recommender.query import handle_query
from assistml.model_recommender.select import select_models_on_dataset_similarity
from common.dto import ReportRequestDto


async def generate_report(request: ReportRequestDto):
    """
    Generate a report based on the given request.
    """
    warnings = []
    distrust_pts = 0
    start_time = time.time()

    query = await handle_query(request)

    models, similarity_level = await select_models_on_dataset_similarity(query)
    if len(models) == 0:
        raise ValueError("No models found")

    distrust_pts += (3 - similarity_level)

    warnings.append([
            "Dataset similarity level 0. Only the type of task and output match. Distrust Pts increased by 3",
            "Dataset similarity level 1. Datasets used shared data types. Distrust Pts increased by 2",
            "Dataset similarity level 2. Datasets used have similar ratios of data types. Distrust Pts increased by 1",
            "Dataset similarity level 3. Datasets used have features with similar meta feature values. Distrust Pts increased by 0"
        ][similarity_level])

    acceptable_models, nearly_acceptable_models, distrust_pts_metrics, distrust_pts_acc, distrust_pts_nacc = cluster_models(models, query.preferences)

    if distrust_pts_metrics > 0:
        distrust_pts += distrust_pts_metrics
        warnings.append(f"Not all requested metric boundaries could be applied. Distrust points increased by {distrust_pts_metrics}")

    if distrust_pts_acc > 0:
        distrust_pts += distrust_pts_acc
        warnings.append(f"Acceptable models distrust points increased by {distrust_pts_acc}")

    if distrust_pts_nacc > 0:
        distrust_pts += distrust_pts_nacc
        warnings.append(f"Nearly acceptable models distrust points increased by {distrust_pts_nacc}")