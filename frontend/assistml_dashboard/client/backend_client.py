import glob
import os
from typing import Optional

import httpx
from pydantic import ValidationError

from common.dto import AnalyseDatasetRequestDto, AnalyseDatasetResponseDto, ReportRequestDto, ReportResponseDto


class BackendClient:

    def __init__(self, config: dict):
        self.base_url = config['BACKEND_BASE_URL']
        self.working_dir = config['WORKING_DIR']

    async def analyse_dataset(self, class_label: str, class_feature_type: str, feature_type_list: str) -> (Optional[AnalyseDatasetResponseDto], Optional[str]):
        url = f"{self.base_url}/analyse-dataset"
        upload_dir = os.path.join(self.working_dir, "uploads")  # TODO: skip saving file on disk
        os.makedirs(upload_dir, exist_ok=True)
        csv_files = glob.glob(os.path.join(upload_dir, "*.csv"))
        arff_files = glob.glob(os.path.join(upload_dir, "*.arff"))
        all_files = sorted(csv_files + arff_files, key=os.path.getmtime)
        file = all_files[-1].split("/")[-1]
        payload = AnalyseDatasetRequestDto(
            class_label=class_label,
            class_feature_type=class_feature_type,
            feature_type_list=feature_type_list
        )
        async with httpx.AsyncClient() as client:
            with open(os.path.join(upload_dir, file), "rb") as dataset_uploaded:
                file_dict = {
                    "json": (None, payload.model_dump_json(), "application/json"),
                    "file": (str(file), dataset_uploaded, "text/plain")
                }
                response = await client.post(url, files=file_dict)
                if response.status_code != 200:
                    return None, f"Failed to analyse dataset: {response.text}"
                response_json = response.json()
                try:
                    return AnalyseDatasetResponseDto(**response_json), None
                except ValidationError as e:
                    return None, f"Error while parsing response: {e}"

    async def report(self, class_feature_type, feature_type_list, classification_output, accuracy_slider,
                                precision_slider,
                                recall_slider, trtime_slider, csv_filename) -> (Optional[ReportResponseDto], Optional[str]):
        feature_type_list = feature_type_list.replace(' ', '')
        feature_type_list = feature_type_list.replace("'", '')
        feature_type_list = feature_type_list.replace('"', '')
        feature_type_list = list(feature_type_list.strip('[]').split(','))

        url = f"{self.base_url}/assistml"
        query_dto = ReportRequestDto(
            classif_type=class_feature_type,
            classif_output=classification_output,
            sem_types=feature_type_list,
            accuracy_range=accuracy_slider,
            precision_range=precision_slider,
            recall_range=recall_slider,
            trtime_range=trtime_slider,
            dataset_name=csv_filename
        )
        async with httpx.AsyncClient() as client:
            response = await client.post(url=url, data=query_dto.model_dump(),
                                     headers={'Content-Type': 'application/json'})

            if response.status_code != 200:
                return None, response.text

            response_json = response.json()
            try:
                return ReportResponseDto(**response_json), None
            except ValidationError as e:
                return None, f"Error while parsing response: {e}"
