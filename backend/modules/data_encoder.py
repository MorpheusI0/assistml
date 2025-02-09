import os
import sys
import pandas as pd
import pymongo
from quart import current_app


def _process_cmd_arg(cmd_argument, available_elements_db):
    annotation_list = cmd_argument.replace(' ', '').replace("'", '').replace('"', '').strip('[]').split(',')
    annotation_list = list(filter(None,annotation_list))

    recognized_elements = [element for element in annotation_list if element in available_elements_db]
    return recognized_elements


def process_and_save_data(feature_annotation_list: str = "", model_names_list: str = "", working_dir: str = ""):

    data = pd.DataFrame(list(collection_enriched.find({},{"_id":0})))
    column_names_enriched = data.columns
    model_names_enriched = collection_enriched.distinct("model_name")

    data_new = pd.DataFrame([])
    # Select model names (Row names) for analysis
    if len(sys.argv) >2:
        model_names_list = sys.argv[2]
        models_list = process_cmd_arg(model_names_list, model_names_enriched)
        if(len(models_list) != 0):
            current_app.logger.info("Found model names list in cmd argument")
            for model_name in models_list:
                subset = data[data.model_name == model_name]
                data_new = pd.concat([data_new, subset], ignore_index=True)
            current_app.logger.info(data_new)
        else:
            current_app.logger.info("Using all models for csv generation")

    # Select column names for analysis
    new_file_name=""
    if len(sys.argv) >1:
        feature_annotation_list = sys.argv[1]
        column_list = process_cmd_arg(feature_annotation_list, column_names_enriched)
        if(len(column_list) != 0):
            current_app.logger.info("Found feature annotation list in cmd argument")
            data_new = data_new.loc[:,column_list]
        else:
            current_app.logger.info("Using default feature annotation list")
            data_new = data_new.loc[:,["fam_name","rows","sampling","language","nr_hyperparams_label","performance_gap","test_size","categorical_encoding"]]
    else:
        current_app.logger.info("Using default feature annotation list")
    current_app.logger.info(data_new)


    output_file_name="merged_data_" + new_file_name +"selectedcols.csv"
    working_dir = os.path.expanduser(current_app.config['WORKING_DIR'])
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)
    output_file_path = os.path.join(working_dir, output_file_name)
    data_new.to_csv(output_file_path, index=False)
