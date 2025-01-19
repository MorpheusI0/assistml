from typing import Optional

from common.data import Task, Model
from common.data.model import BaseModel, Info, Metrics, TrainingCharacteristics
from mlsea import MLSeaRepository
from mlsea.dtos import RunDto


class ModelProcessor:
    def __init__(self, mlsea: MLSeaRepository):
        self._mlsea = mlsea

    async def process_all(self, task: Task, recursive: bool = False, head: int = None):
        openml_task_id = int(task.mlsea_uri.split('/')[-1])
        task_runs_df = self._mlsea.retrieve_all_runs_from_openml_for_task(openml_task_id)
        print(task_runs_df.head())

        if head is not None:
            task_runs_df = task_runs_df.head(head)

        for run_dto in task_runs_df.itertuples(index=False):
            run_dto = RunDto(*run_dto)

            openml_run_id = int(run_dto.openml_run_url.split('/')[-1])
            run_metrics_df = self._mlsea.retrieve_all_metrics_from_openml_for_run(openml_run_id)

            print(run_metrics_df.head())

            await ModelProcessor._ensure_base_model_exists(run_dto, task)

    @staticmethod
    async def _ensure_base_model_exists(run_dto: RunDto, task: Task):
        model: Optional[Model] = await Model.find_one(Model.base_model.info.mlsea_uri == run_dto.mlsea_run_uri)

        if model is not None:
            return model

        # TODO: Implement the creation of a new model
