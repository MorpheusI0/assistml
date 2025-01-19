import re
from typing import Optional, List

from common.data.implementation import Software

from common.data import Task, Implementation
from mlsea import MLSeaRepository
from mlsea.dtos import ImplementationDto, SoftwareDto


class ImplementationProcessor:
    def __init__(self, mlsea: MLSeaRepository):
        self._mlsea = mlsea

    async def process_all(self, task: Task, recursive: bool = False, head: int = None):
        openml_task_id = int(task.mlsea_uri.split('/')[-1])
        task_implementations_df = self._mlsea.retrieve_all_implementations_from_openml_for_task(openml_task_id)
        print(task_implementations_df.head())

        if head is not None:
            task_implementations_df = task_implementations_df.head(head)

        for implementation_dto in task_implementations_df.itertuples(index=False):
            implementation_dto = ImplementationDto(*implementation_dto)

            software_df = self._mlsea.retrieve_dependencies_from_openml_for_implementation(
                implementation_dto.openml_flow_id)
            software_dtos = [SoftwareDto(*software_dto) for software_dto in software_df.itertuples(index=False)]

            await ImplementationProcessor._ensure_implementation_exists(implementation_dto, software_dtos, task)

    @staticmethod
    async def _ensure_implementation_exists(implementation_dto: ImplementationDto, software_dtos: List[SoftwareDto],
                                            task: Task):
        implementation: Optional[Implementation] = await Implementation.find_one(
            Implementation.mlsea_uri == implementation_dto.mlsea_implementation_uri)

        if implementation is not None:
            return implementation

        dependencies = [
            parsed_dependency
            for software_dto in software_dtos
            for parsed_dependency in ImplementationProcessor._transform_software_dto(software_dto)
        ]

        implementation = Implementation(
            mlsea_uri=implementation_dto.mlsea_implementation_uri,
            title=implementation_dto.title,
            task=task,
            dependencies=dependencies
        )  # TODO: add hyperparameters with default values
        await implementation.insert()
        return implementation

    @staticmethod
    def _transform_software_dto(software_dto: SoftwareDto) -> List[Software]:
        dependency_exceptions = [
            {
                'original': 'Shark machine learning library',
                'transformed': {'name': 'Shark', 'version': None}
            },
            {
                'original': 'Build on top of Weka API (Jar version 3.?.?)',
                'transformed': {'name': 'Weka', 'version': '3.?.?'}
            },
            {
                'original': 'MLR 2.4',
                'transformed': {'name': 'MLR', 'version': '2.4'}
            }
        ]

        # handle exceptions
        for exception in dependency_exceptions:
            if software_dto.software_requirement == exception['original']:
                return [Software(mlsea_uri=software_dto.mlsea_software_uri, **exception['transformed'])]

        requirements = software_dto.software_requirement.split(' ')
        pattern = r"([a-zA-Z\d]+)([_<>=!]*)([a-zA-Z\d.+-]*)"

        parsed_dependencies = []
        for requirement in requirements:
            match = re.match(pattern, requirement)
            if match:
                name, operator, version = match.groups()
                parsed_dependencies.append(
                    Software(mlsea_uri=software_dto.mlsea_software_uri, name=name, version=version))

            else:
                raise ValueError(
                    f"Could not parse software requirement: {requirement} for software: {software_dto.mlsea_software_uri}")

        return parsed_dependencies
