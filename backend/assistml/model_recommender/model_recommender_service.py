import time
from typing import List

from assistml.model_recommender.cluster import cluster_models
from assistml.model_recommender.ranking import Report
from assistml.model_recommender.query import handle_query
from assistml.model_recommender.ranking.report import DistrustPointCategory
from assistml.model_recommender.select import select_models_on_dataset_similarity
from common.dto import ReportRequestDto
from common.data.model import Metric


async def generate_report(request: ReportRequestDto):
    """
    Generate a report based on the given request.
    """
    start_time = time.time()

    query = await handle_query(request)
    report = Report(query)

    models, similarity_level = await select_models_on_dataset_similarity(query)
    if len(models) == 0:
        raise ValueError("No models found")

    report.set_distrust_points(DistrustPointCategory.DATASET_SIMILARITY, 3-similarity_level)

    acceptable_models, nearly_acceptable_models, distrust_pts_metrics, distrust_pts_acc, distrust_pts_nacc = cluster_models(models, query.preferences)
    await report.set_models(acceptable_models, nearly_acceptable_models)
    report.set_distrust_points(DistrustPointCategory.METRICS_SUPPORT, distrust_pts_metrics)
    report.set_distrust_points(DistrustPointCategory.CLUSTER_INSIDE_RATIO_ACC, distrust_pts_acc)
    report.set_distrust_points(DistrustPointCategory.CLUSTER_INSIDE_RATIO_NACC, distrust_pts_nacc)

    selected_metrics: List[Metric] = list(query.preferences.keys())
    final_report = await report.generate_report()
    pass
