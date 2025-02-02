import re
from typing import Optional

import openml.tasks

from common.data import Dataset, Task
from common.data.task import TaskType, ClassificationTask, RegressionTask, ClusteringTask, LearningCurveTask
from mlsea import MLSeaRepository
from mlsea.dtos import TaskDto
from processing.model import ModelProcessor
from processing.implementation import ImplementationProcessor


MLSO_TT_BASE_URI = "http://w3id.org/mlso/vocab/ml_task_type#"

class TaskProcessor:
    def __init__(self, mlsea: MLSeaRepository):
        self._mlsea = mlsea

    async def process_all(self, dataset: Dataset, recursive: bool = False, head: int = None):
        dataset_id = int(dataset.info.mlsea_uri.split('/')[-1])
        dataset_tasks_df = self._mlsea.retrieve_all_tasks_from_openml_for_dataset(dataset_id)
        print(dataset_tasks_df.head())

        if head is not None:
            dataset_tasks_df = dataset_tasks_df.head(head)

        for task_dto in dataset_tasks_df.itertuples(index=False):
            task_dto = TaskDto(*task_dto)

            task: Task = await TaskProcessor._ensure_task_exists(task_dto, dataset)

            if recursive:
                implementation_processor = ImplementationProcessor(self._mlsea)
                await implementation_processor.process_all(task, recursive, head)

                model_processor = ModelProcessor(self._mlsea)
                await model_processor.process_all(task, recursive, head)

    @staticmethod
    async def _ensure_task_exists(task_dto: TaskDto, dataset: Dataset):
        task: Optional[Task] = await Task.find_one(Task.mlsea_uri == task_dto.mlsea_task_uri)

        if task is not None:
            return task

        task = TaskProcessor._parse_task(task_dto, dataset)

        await task.insert()
        return task

    @staticmethod
    def _parse_task_type(task_type_concept: str) -> TaskType:
        if not task_type_concept.startswith(MLSO_TT_BASE_URI):
            raise ValueError(f"Task type concept {task_type_concept} does not start with {MLSO_TT_BASE_URI}")

        task_type_string = task_type_concept[len(MLSO_TT_BASE_URI):]

        # MLSO-TT is not consistently in Capitalized_Snake_Case
        task_type_string = re.sub(r'(?<!^)(?<!_)([A-Z])', r'_\1', task_type_string)

        # MLSO-TT is not consistent with following OpenML task types
        if task_type_string == "Learning_Curve_Estimation":
            task_type_string = "Learning_Curve"

        return TaskType(task_type_string)

    @staticmethod
    def _parse_task(task_dto, dataset) -> Task:
        openml_task: openml.tasks.OpenMLTask = openml.tasks.get_task(task_dto.openml_task_id)

        task_type = TaskProcessor._parse_task_type(task_dto.task_type)
        task: Task
        if task_type == TaskType.SUPERVISED_CLASSIFICATION:
            openml_task: openml.tasks.OpenMLClassificationTask
            task = ClassificationTask(
                task_type=task_type,
                dataset=dataset,
                mlsea_uri=task_dto.mlsea_task_uri,
                target_name=openml_task.target_name,
                class_labels=openml_task.class_labels
            )
        elif task_type == TaskType.SUPERVISED_REGRESSION:
            openml_task: openml.tasks.OpenMLRegressionTask
            task = RegressionTask(
                task_type=task_type,
                dataset=dataset,
                mlsea_uri=task_dto.mlsea_task_uri,
                target_name=openml_task.target_name
            )
        elif task_type == TaskType.CLUSTERING:
            openml_task: openml.tasks.OpenMLClusteringTask
            task = ClusteringTask(
                task_type=task_type,
                dataset=dataset,
                mlsea_uri=task_dto.mlsea_task_uri,
                target_name=openml_task.target_name
            )
        elif task_type == TaskType.LEARNING_CURVE:
            openml_task: openml.tasks.OpenMLLearningCurveTask
            task = LearningCurveTask(
                task_type=task_type,
                dataset=dataset,
                mlsea_uri=task_dto.mlsea_task_uri,
                target_name=openml_task.target_name,
                class_labels=openml_task.class_labels
            )
        else:
            task = Task(
                task_type=task_type,
                dataset=dataset,
                mlsea_uri=task_dto.mlsea_task_uri
            )

        return task
