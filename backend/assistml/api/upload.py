import csv
import os

import arff
import pandas as pd
from quart import request, jsonify, current_app

from assistml.api import bp


@bp.route('/upload', methods=['POST'])
async def upload_data():
    """
    Route to upload data of CSV files to the server.
    """
    files = await request.files
    if len(files.keys()) == 0:
        return jsonify({"error": "No file part"}), 400

    filename = list(files.keys())[0]
    file = files[filename]
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        #temp_file_path = f"/tmp/{file.filename}"
        #file.save(temp_file_path)
        #current_app.logger.info(f"Temporary file path: {temp_file_path}")

        working_dir = os.path.expanduser(current_app.config["WORKING_DIR"])
        upload_dir = os.path.join(working_dir, "uploads")
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        file_path = os.path.join(upload_dir, file.filename)

        await file.save(file_path)
        file.close()
        current_app.logger.info(f"Just uploaded {file.filename}")
        if file.filename.endswith(".csv"):
            sniffer = csv.Sniffer()
            with open(file_path) as csvfile:
                dialect = sniffer.sniff(csvfile.read(1024))

            df = pd.read_csv(file_path, delimiter=str(dialect.delimiter))
            current_app.logger.info("Sample of uploaded data")
            current_app.logger.info("\n" + str(df.head()))

        elif file.filename.endswith(".arff"):

            # scipy.io.arff does not support strings nor datetime types
            # data, meta = scipy.io.arff.loadarff(file_path)

            with open(file_path, "r") as arfffile:
                data = arff.load(arfffile)
            df = pd.DataFrame(data['data'], columns=[x[0] for x in data['attributes']])
            current_app.logger.info("Sample of uploaded data")
            current_app.logger.info("\n" + str(df.head()))

        return jsonify({"message": f"File {file.filename} uploaded successfully"}), 200
