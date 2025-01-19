from typing import Optional
from urllib.parse import urlparse

from common.data import Dataset, Task
from common.data.task import UseCaseSet
from mlsea import MLSeaRepository
from mlsea.dtos import TaskDto
from processing.model import ModelProcessor
from processing.implementation import ImplementationProcessor


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

        task = Task(
            use_case_set=UseCaseSet(
                task_type=urlparse(task_dto.task_type).fragment,  # TODO: parse task type
                task_output='single'  # TODO: parse whether it delivers single predictions or class probabilities
            ),
            dataset=dataset,
            mlsea_uri=task_dto.mlsea_task_uri
        )
        await task.insert()
        return task
