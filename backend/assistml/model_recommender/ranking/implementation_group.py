import asyncio
from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Tuple, Union

from beanie import Link, PydanticObjectId

from assistml.model_recommender.ranking.hyperparameter_analytics import HyperparameterAnalytics
from assistml.model_recommender.ranking.implementation_dataset_group import ImplementationDatasetGroup
from assistml.model_recommender.ranking.metric_analytics import MetricAnalytics
from assistml.utils.document_cache import DocumentCache
from common.data import Implementation
from common.data import Dataset
from common.data.model import Metric
from common.data.projection.model import FullyJoinedModelView


class ImplementationGroup:

    _implementation: Implementation
    _dataset_groups: Dict[PydanticObjectId, ImplementationDatasetGroup]
    _document_cache: DocumentCache
    _hyperparameter_analytics: HyperparameterAnalytics
    _check_dataset_lock: DefaultDict[PydanticObjectId, asyncio.Lock]
    _ranked_dataset_groups: Optional[List[Tuple[float, ImplementationDatasetGroup]]]
    _aggregated_metrics: Optional[Dict[Metric, Dict[str, float]]]

    def __init__(
            self,
            implementation: Implementation,
            document_cache: DocumentCache,
            metric_analytics: MetricAnalytics
    ):
        self._implementation = implementation
        self._dataset_groups = {}
        self._document_cache = document_cache
        self._metric_analytics = metric_analytics
        self._hyperparameter_analytics = HyperparameterAnalytics(implementation, document_cache)
        self._check_dataset_lock = defaultdict(asyncio.Lock)
        self._ranked_dataset_groups = None
        self._aggregated_metrics = None

    @classmethod
    async def create(
            cls,
            implementation_ref: Union[Link[Implementation], Implementation],
            document_cache: DocumentCache,
            metric_analytics: MetricAnalytics
    ) -> "ImplementationGroup":
        implementation = await document_cache.get_implementation(implementation_ref)
        return cls(implementation, document_cache, metric_analytics)

    async def add_model(self, model: FullyJoinedModelView):
        dataset = model.setup.task.dataset
        if isinstance(dataset, Dataset):
            dataset_id = dataset.id
        elif isinstance(dataset, Link):
            dataset_id = dataset.to_ref().id
        else:
            raise ValueError(f"Unknown dataset type: {type(dataset)}")
        if dataset_id in self._dataset_groups:
            await self._dataset_groups[dataset_id].add_model(model)
            return

        async with self._check_dataset_lock[dataset_id]:
            if dataset_id not in self._dataset_groups: # check again after lock
                self._dataset_groups[dataset_id] = await ImplementationDatasetGroup.create(
                    self._implementation, dataset, self._document_cache, self._metric_analytics,
                    self._hyperparameter_analytics)
        await self._dataset_groups[dataset_id].add_model(model)

    def rank_datasets(self, dataset: Dataset) -> None:
        if not self._hyperparameter_analytics.are_standardizers_fitted():
            self._hyperparameter_analytics.fit_standardizers()
        dataset_similarities = []
        for dataset_group in self._dataset_groups.values():
            similarity_score = dataset_group.get_dataset_similarity(dataset)
            dataset_similarities.append((similarity_score, dataset_group))
        dataset_similarities.sort(key=lambda x: x[0], reverse=True)
        self._ranked_dataset_groups = dataset_similarities

    def _aggregate_metrics(self, selected_metrics: List[Metric]) -> None:
        if self._ranked_dataset_groups is None:
            raise ValueError("Rank datasets first")
        dataset_group_metrics = []
        for similarity_score, dataset_group in self._ranked_dataset_groups:
            best_configuration_metrics = dataset_group.get_metrics_of_best_configuration(selected_metrics)
            dataset_group_metrics.append((similarity_score, best_configuration_metrics))
        self._aggregated_metrics = self._metric_analytics.aggregate_list(dataset_group_metrics)

    def get_aggregated_metrics(self, selected_metrics: List[Metric]) -> Dict[Metric, Dict[str, float]]:
        if self._aggregated_metrics is None:
            self._aggregate_metrics(selected_metrics)
        return self._aggregated_metrics

    def calculate_overall_score(self, selected_metrics: List[Metric], lambda_penalty: float = 0.5) -> float:
        if self._aggregated_metrics is None:
            self._aggregate_metrics(selected_metrics)
        metric_weights = {metric: 1.0 if metric in selected_metrics else 0.0 for metric in Metric}
        score_vector = self._metric_analytics.calculate_overall_score(self._aggregated_metrics, metric_weights)
        return score_vector['mean'] - lambda_penalty * score_vector['std']

    def generate_report(self, top_n: int = 3) -> List[Dict[Metric, Dict[str, float]]]:
        if self._ranked_dataset_groups is None:
            raise ValueError("Rank datasets first")
        return [
            dataset_group.generate_report()
            for _, dataset_group in self._ranked_dataset_groups[:top_n]
        ]

    def __repr__(self):
        return f"ImplementationGroup(implementation={self._implementation.title}, dataset_groups={self._dataset_groups})"

    def __str__(self):
        return f"ImplementationGroup: {self._implementation.title}, #datasets={len(self._dataset_groups)}"
