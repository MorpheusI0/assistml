import asyncio
from collections import defaultdict
from enum import Enum
from typing import DefaultDict, Dict, List, Literal, Optional, Tuple

from beanie import Link, PydanticObjectId

from assistml.model_recommender.ranking.implementation_group import ImplementationGroup
from assistml.model_recommender.ranking.metric_analytics import MetricAnalytics
from assistml.utils.document_cache import DocumentCache
from common.data import Dataset, Implementation, Query
from common.data.model import Metric
from common.data.projection.model import FullyJoinedModelView


class DistrustPointCategory(str, Enum):
    DATASET_SIMILARITY = "dataset_similarity"
    METRICS_SUPPORT = "metrics_support"
    CLUSTER_INSIDE_RATIO_ACC = "cluster_inside_ratio_acc"
    CLUSTER_INSIDE_RATIO_NACC = "cluster_inside_ratio_nacc"

ModelGroup = Literal['acceptable_models', 'nearly_acceptable_models']

class Report:
    _query: Query
    _distrust_points: Dict[DistrustPointCategory, int]
    _implementation_groups: Dict[ModelGroup, Optional[Dict[PydanticObjectId, ImplementationGroup]]]
    _implementation_groups_locks: Dict[ModelGroup, DefaultDict[PydanticObjectId, asyncio.Lock]]
    _document_cache: DocumentCache
    _metric_analytics: MetricAnalytics
    _ranked_implementation_groups: Optional[Dict[ModelGroup, List[Tuple[float, ImplementationGroup]]]]
    _rank_implementations_lock: asyncio.Lock

    def __init__(self, query: Query):
        self._query = query
        self._distrust_points = {category: 0 for category in DistrustPointCategory}
        self._implementation_groups = {
            "acceptable_models": None,
            "nearly_acceptable_models": None,
        }
        self._implementation_groups_locks = {
            "acceptable_models": defaultdict(asyncio.Lock),
            "nearly_acceptable_models": defaultdict(asyncio.Lock),
        }
        self._document_cache = DocumentCache()
        self._metric_analytics = MetricAnalytics()
        self._ranked_implementation_groups = None
        self._rank_implementations_lock = asyncio.Lock()

    def set_distrust_points(self, category: DistrustPointCategory, points: int):
        self._distrust_points[category] = points

    def get_distrust_warnings(self):
        warnings = [[
            "Dataset similarity level 3. Datasets used have features with similar meta feature values. Distrust Pts increased by 0",
            "Dataset similarity level 2. Datasets used have similar ratios of data types. Distrust Pts increased by 1",
            "Dataset similarity level 1. Datasets used shared data types. Distrust Pts increased by 2",
            "Dataset similarity level 0. Only the type of task and output match. Distrust Pts increased by 3",
                    ][self._distrust_points[DistrustPointCategory.DATASET_SIMILARITY]]]
        if self._distrust_points[DistrustPointCategory.METRICS_SUPPORT] > 0:
            warnings.append(f"Not all requested metric boundaries could be applied. Distrust points increased by {self._distrust_points[DistrustPointCategory.METRICS_SUPPORT]}")
        if self._distrust_points[DistrustPointCategory.CLUSTER_INSIDE_RATIO_ACC] > 0:
            warnings.append(f"Acceptable models distrust points increased by {self._distrust_points[DistrustPointCategory.CLUSTER_INSIDE_RATIO_ACC]}")
        if self._distrust_points[DistrustPointCategory.CLUSTER_INSIDE_RATIO_NACC] > 0:
            warnings.append(f"Nearly acceptable models distrust points increased by {self._distrust_points[DistrustPointCategory.CLUSTER_INSIDE_RATIO_NACC]}")
        return warnings


    async def set_models(
            self,
            acceptable_models: List[FullyJoinedModelView],
            nearly_acceptable_models: List[FullyJoinedModelView]
    ):
        model_groups: Dict[ModelGroup, List[FullyJoinedModelView]] = {
            "acceptable_models": acceptable_models,
            "nearly_acceptable_models": nearly_acceptable_models,
        }
        tasks = []
        for model_group, models in model_groups.items():
            self._implementation_groups[model_group] = {}
            for model in models:
                implementation = model.setup.implementation
                if isinstance(implementation, Link):
                    implementation_id = implementation.to_ref().id
                elif isinstance(implementation, Implementation):
                    implementation_id = implementation.id
                else:
                    raise ValueError(f"Unknown implementation type: {type(implementation)}")
                async with self._implementation_groups_locks[model_group][implementation_id]:
                    if implementation_id not in self._implementation_groups[model_group]:
                        self._implementation_groups[model_group][implementation_id] = await ImplementationGroup.create(
                            implementation, self._document_cache, self._metric_analytics)
                tasks.append(self._implementation_groups[model_group][implementation_id].add_model(model))
        await asyncio.gather(*tasks)
        
    def _rank_datasets(self, dataset: Dataset):
        for implementation_group in self._implementation_groups.values():
            for group in implementation_group.values():
                group.rank_datasets(dataset)

    async def _rank_implementations(self, selected_metrics: List[Metric]):
        if any([group is None for group in self._implementation_groups.values()]):
            raise ValueError("Models not set")
        
        new_dataset = await self._document_cache.get_dataset(self._query.dataset)
        self._metric_analytics.fit_normalizers()
        self._rank_datasets(new_dataset)
        self._ranked_implementation_groups = {}
        for model_group, implementation_groups in self._implementation_groups.items():
            ranked_implementation_groups = []
            for implementation_group in implementation_groups.values():
                overall_score = implementation_group.calculate_overall_score(selected_metrics)
                ranked_implementation_groups.append((overall_score, implementation_group))
            ranked_implementation_groups.sort(key=lambda x: x[0], reverse=True)
            self._ranked_implementation_groups[model_group] = ranked_implementation_groups
        
    async def generate_report(self, top_k: int = 5, top_n: int = 3):
        selected_metrics = [metric for metric in self._query.preferences.keys()]
        async with self._rank_implementations_lock:
            if self._ranked_implementation_groups is None:
                await self._rank_implementations(selected_metrics)
        reports = {}
        for model_group, ranked_implementation_groups in self._ranked_implementation_groups.items():
            reports[model_group] = [
                implementation_group.generate_report(top_n)
                for score, implementation_group in ranked_implementation_groups[:top_k]
            ]
        return reports
        
            