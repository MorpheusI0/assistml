import csv
import io
import os

import arff
import pandas as pd
from pydantic import ValidationError
from quart import current_app
from werkzeug.datastructures.file_storage import FileStorage

from common.dto import AnalyseDatasetRequestDto, DatasetInfoDto, DbWriteStatusDto
from common.data_profiler import DataProfiler, ReadMode
from common.data import Dataset
from common.data.projection import dataset as dataset_projection


async def profile_dataset(request: AnalyseDatasetRequestDto, file: FileStorage) -> (DatasetInfoDto, DbWriteStatusDto):
    if current_app.config["SAVE_UPLOADS"]:
        await _save_file_to_disk(file)

    try:
        df = await _load_file(file)
    except ValueError as e:
        raise ValueError(f"Error while loading file: {e}")

    current_app.logger.debug("Sample of uploaded data")
    current_app.logger.debug("\n" + str(df.head()))

    data_profiler = DataProfiler(file.filename, request.class_label, request.class_feature_type)
    mode = ReadMode.READ_FROM_DATAFRAME

    try:
        dataset_profile = data_profiler.analyse_dataset(mode, request.feature_type_list, dataset_df=df)
        dataset_profile = DatasetInfoDto(**dataset_profile)
    except ValidationError as e:
        raise ValueError(f"Error while parsing dataset profile: {e}")
    except Exception as e:
        raise ValueError(f"Error while profiling dataset: {e}")

    db_write_status = await _write_result_to_db(dataset_profile)

    return dataset_profile, db_write_status

async def _save_file_to_disk(file):
    current_app.logger.info(f"Saving file {file.filename} to disk")
    working_dir = os.path.expanduser(current_app.config["WORKING_DIR"])
    upload_dir = os.path.join(working_dir, "uploads")
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    file_path = os.path.join(upload_dir, file.filename)
    await file.save(file_path)
    file.seek(0)
    current_app.logger.info(f"Just saved {file.filename} to {file_path}")

async def _load_file(file: FileStorage) -> pd.DataFrame:
    current_app.logger.info(f"Loading file {file.filename}")
    data = file.stream.read()
    decoded_data = data.decode("utf-8")
    if file.filename.endswith(".csv"):
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(decoded_data.splitlines()[0])
        df = pd.read_csv(io.StringIO(decoded_data), delimiter=str(dialect.delimiter))

    elif file.filename.endswith(".arff"):
        current_app.logger.info(f"Loading ARFF file...")
        data = arff.load(decoded_data)
        df = pd.DataFrame(data['data'], columns=[x[0] for x in data['attributes']])

    else:
        raise ValueError(f"File format {file.filename} not supported")

    return df

async def _write_result_to_db(data_profile: DatasetInfoDto) -> DbWriteStatusDto:
    similar_dataset = await _check_for_similar_dataset_in_db(data_profile)
    if similar_dataset is not None:
        return DbWriteStatusDto(
            status=(f"Information about the dataset {data_profile.info.dataset_name} already available in the "
                    f"database. Skipping insertion."),
            dataset_id=str(similar_dataset.id)
        )
    else:
        new_dataset = Dataset(**data_profile.model_dump())
        await new_dataset.insert()
        return DbWriteStatusDto(
            status=f"Information about the dataset {data_profile.info.dataset_name} written to the database.",
            dataset_id=str(new_dataset.id)
        )


async def _check_for_similar_dataset_in_db(data_profile) -> dataset_projection.EmptyView:
    similar_datasets = Dataset.find({
        "info.datasetName": data_profile.info.dataset_name,
        "info.observations": data_profile.info.observations,
        "info.nrTotalFeatures": data_profile.info.nr_total_features,
        "info.nrAnalyzedFeatures": data_profile.info.nr_analyzed_features,
        "info.numericRatio": data_profile.info.numeric_ratio,
        "info.categoricalRatio": data_profile.info.categorical_ratio,
        "info.datetimeRatio": data_profile.info.datetime_ratio,
        "info.unstructuredRatio": data_profile.info.unstructured_ratio,
    }).project(dataset_projection.EmptyView)

    return await similar_datasets.first_or_none()
