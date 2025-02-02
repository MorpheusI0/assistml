from typing import Optional

import openml.runs

from common.data import Task, Model
from common.data.model import BaseModel, Info, Metrics, TrainingCharacteristics
from mlsea import MLSeaRepository
from mlsea.dtos import RunDto


RUN_BASE_URI = "http://w3id.org/mlsea/openml/run/"

class ModelProcessor:
    def __init__(self, mlsea: MLSeaRepository):
        self._mlsea = mlsea

    async def process_all(self, task: Task, recursive: bool = False, head: int = None, offset_id: int = 0):
        openml_task_id = int(task.mlsea_uri.split('/')[-1])
        count = 0
        while True:
            task_runs_df = self._mlsea.retrieve_all_runs_from_openml_for_task(openml_task_id, batch_size=100, offset_id=offset_id)
            if task_runs_df.empty:
                break

            if head is not None:
                task_runs_df = task_runs_df.head(head)

            for run_dto in task_runs_df.itertuples(index=False):
                run_dto = RunDto(*run_dto)

                print(f"Processing run {run_dto.openml_run_id}")

                #openml_run_id #int(run_dto.openml_run_url.split('/')[-1])
                run_metrics_df = self._mlsea.retrieve_all_metrics_from_openml_for_run(run_dto.openml_run_id)

                #print(run_metrics_df.head())

                await ModelProcessor._ensure_base_model_exists(run_dto, task)

                count += 1
                offset_id = run_dto.openml_run_id

            if head is not None and count >= head:
                break

    @staticmethod
    async def _ensure_base_model_exists(run_dto: RunDto, task: Task):
        model: Optional[Model] = await Model.find_one(Model.base_model.info.mlsea_uri == run_dto.mlsea_run_uri)

        if model is not None:
            return model

        base_model = await ModelProcessor._generate_base_model(run_dto, task)

    @staticmethod
    async def _generate_base_model(run_dto: RunDto, task: Task) -> BaseModel:
        openml_run_id = int(run_dto.openml_run_url.split('/')[-1])
        openml_run = openml.runs.get_run(openml_run_id)
        openml_setup = openml.setups.get_setup(openml_run.setup_id)

        if openml_setup.parameters is not None:
            for parameter in openml_setup.parameters.values():
                #print(parameter)
                pass

        #training_characteristics = TrainingCharacteristics(

