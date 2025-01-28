import csv
import json
import os

import arff
import pandas as pd
from quart import request, jsonify, current_app
from werkzeug.datastructures import FileStorage

from assistml.api import bp
from common.data_profiler import DataProfiler
from common.data_profiler import ReadMode
from common.data import Dataset
from common.data.projection import dataset as dataset_projection


@bp.route('/analyse-dataset', methods=['POST'])
async def analyse_dataset():
    form = await request.form
    data = form.get("json")
    if data is None:
        return jsonify({"error": "No JSON data provided"}), 400

    try:
        data = json.loads(data)
    except json.JSONDecodeError as e:
        return jsonify({"error": str(e)}), 400

    files = await request.files
    file = files.get("file")
    if file is None:
        return jsonify({"error": "No file part"}), 400

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if current_app.config["SAVE_UPLOADS"]:
        await _save_file_to_disk(file)

    try:
        df = await _load_file(file)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    current_app.logger.info("Sample of uploaded data")
    current_app.logger.info("\n" + str(df.head()))

    class_label = data['class_label']
    class_feature_type = data['class_feature_type']
    feature_type_list = data['feature_type_list']

    data_profiler = DataProfiler(file.filename, class_label, class_feature_type)
    mode = ReadMode.READ_FROM_DATAFRAME

    data_info = data_profiler.analyse_dataset(mode, str(feature_type_list), dataset_df=df)
    db_write_status = await _write_result_to_db(data_info)

    return jsonify({
        'data_profile': data_info if data_info else None,
        'db_write_status': db_write_status,
    })


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


async def _write_result_to_db(data_profile):
    if await _check_data_availability_in_db(data_profile):
        status = (f"Information about the dataset {data_profile['Info']['dataset_name']} already available in the "
                  f"database. Skipping insertion.")
    else:
        new_dataset = Dataset(**data_profile)
        await new_dataset.save()
        status = f"Information about the dataset {data_profile['Info']['dataset_name']} written to the database."

    current_app.logger.info(status)
    return status


async def _check_data_availability_in_db(data_profile):
    similar_datasets = Dataset.find(
        Dataset.info.dataset_name == data_profile['Info']['dataset_name'],
        Dataset.info.observations == data_profile['Info']['observations'],
        Dataset.info.features == data_profile['Info']['features'],
        Dataset.info.numeric_ratio == data_profile['Info']['numeric_ratio'],
        Dataset.info.categorical_ratio == data_profile['Info']['categorical_ratio'],
        Dataset.info.datetime_ratio == data_profile['Info']['datetime_ratio'],
        Dataset.info.unstructured_ratio == data_profile['Info']['unstructured_ratio'],
    ).project(dataset_projection.EmptyView)

    return await similar_datasets.exists()


async def _load_file(file: FileStorage) -> pd.DataFrame:
    current_app.logger.info(f"Loading file {file.filename}")
    data = file.stream.read()
    decoded_data = data.decode("utf-8")
    if file.filename.endswith(".csv"):
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(file.stream.read(1024).decode("utf-8"))
        df = pd.read_csv(decoded_data, delimiter=str(dialect.delimiter))

    elif file.filename.endswith(".arff"):
        current_app.logger.info(f"Loading ARFF file...")
        data = arff.load(decoded_data)
        df = pd.DataFrame(data['data'], columns=[x[0] for x in data['attributes']])

    else:
        raise ValueError(f"File format {file.filename} not supported")

    return df

