import os

import pandas as pd
from flask import request, jsonify, current_app
from assistml.api import upload_bp as bp


@bp.route('/upload', methods=['POST'])
def upload_data():
    """
    Route to upload data of CSV files to the server.
    """
    files = request.files.to_dict()
    if len(files.keys()) == 0:
        return jsonify({"error": "No file part"}), 400

    filename = list(files.keys())[0]
    file = files[filename]
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        temp_file_path = f"/tmp/{file.filename}"
        file.save(temp_file_path)
        current_app.logger.info(f"Temporary file path: {temp_file_path}")

        newcsv = pd.read_csv(temp_file_path)
        current_app.logger.info("Sample of uploaded data")
        current_app.logger.info(newcsv.head())

        working_dir = os.path.expanduser(current_app.config["WORKING_DIR"])
        upload_dir = os.path.join(working_dir, "uploads")
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)

        file.save(os.path.join(upload_dir, file.filename))
        current_app.logger.info(f"Just uploaded {file.filename}")

        return jsonify({"message": f"File {file.filename} uploaded successfully"}), 200
