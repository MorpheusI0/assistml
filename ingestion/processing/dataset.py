from typing import List, Optional

import pandas as pd
from sklearn.datasets import fetch_openml

from common.data import Dataset
from mlsea import MLSeaRepository, DatasetDto
from processing.task import TaskProcessor

from common.data_profiler import DataProfiler, ReadMode


class DatasetProcessor:
    def __init__(self, mlsea: MLSeaRepository):
        self._mlsea = mlsea

    async def process(self, dataset_ids: List[int] = None, recursive: bool = False, head: int = None):
        datasets_df = self._mlsea.retrieve_datasets_from_openml(dataset_ids)
        print(datasets_df.head())

        if head is not None:
            datasets_df = datasets_df.head(head)

        for dataset_dto in datasets_df.itertuples(index=False):
            dataset_dto = DatasetDto(*dataset_dto)

            dataset: Dataset = await DatasetProcessor._ensure_dataset_exists(dataset_dto)

            if recursive:
                task_processor = TaskProcessor(self._mlsea)
                await task_processor.process_all(dataset, recursive, head)

    @staticmethod
    async def _ensure_dataset_exists(dataset_dto: DatasetDto):
        dataset: Optional[Dataset] = await Dataset.find_one(Dataset.info.mlsea_uri == dataset_dto.mlsea_dataset_uri)

        if dataset is not None:
            return dataset

        profiled_dataset = DatasetProcessor._profile_dataset(dataset_dto.openml_dataset_id,
                                                             dataset_dto.default_target_feature_label)

        dataset = Dataset(**profiled_dataset)
        dataset.info.mlsea_uri = dataset_dto.mlsea_dataset_uri
        await dataset.insert()
        return dataset

    @staticmethod
    def _profile_dataset(openml_dataset_id, default_target_feature_label: str):
        raw_data = fetch_openml(data_id=openml_dataset_id, parser='auto', as_frame=True)
        df: pd.DataFrame = raw_data['frame']
        details = raw_data['details']

        feature_annotations = [
            'T' if feature_name == default_target_feature_label else
            'N' if feature_type in ['int64', 'float64', 'numeric'] else
            'C' if feature_type in ['category'] else
            'D' if feature_type in ['datetime64'] else
            'U'
            for feature_name, feature_type in df.dtypes.items()
        ]
        feature_annotations_string = '[' + ','.join(feature_annotations) + ']'
        target_feature_type = DatasetProcessor._recognize_classification_output_type(df, default_target_feature_label)

        data_profiler = DataProfiler(
            dataset_name=details['name'],
            target_label=default_target_feature_label,
            target_feature_type=target_feature_type
        )
        data_info = data_profiler.analyse_dataset(ReadMode.READ_FROM_DATAFRAME, feature_annotations_string,
                                                  dataset_df=df)
        return data_info

    @staticmethod
    def _recognize_classification_output_type(df: pd.DataFrame, target_feature: str):
        if df.dtypes[target_feature] == 'category':
            if len(df[target_feature].cat.categories) == 2:
                return 'binary'
            elif len(df[target_feature].cat.categories) > 2:
                return 'multi-class'
            else:
                return 'single-class'
        elif df.dtypes[target_feature] in ['int64', 'float64', 'numeric']:
            return 'regression'
        else:
            raise ValueError("Unrecognized type")
