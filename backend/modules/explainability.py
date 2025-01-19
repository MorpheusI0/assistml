#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse

import numpy as np
import pandas as pd
from quart import current_app
from pymongo import MongoClient
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix


def main():
    import os
    #os.chdir("../")
    working_dir = os.path.expanduser(current_app.config['WORKING_DIR'])
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)
    data_base_path = os.path.join(working_dir, "1_data")
    pickle_file_base_path = os.path.join(working_dir, "3_pickle")
    json_file_base_path = os.path.join(working_dir, "3_pickle")

    for path in [working_dir, data_base_path, pickle_file_base_path, json_file_base_path]:
        if not os.path.exists(path):
            os.makedirs(path)
    #os.chdir(os.path.join(os.getcwd(), "2_code"))

    # parse input parameters
    parser = argparse.ArgumentParser(
        description="This script is used to perform analysis on confusion matrix for its explainability")
    parser.add_argument("-m", "--modelName", required=True, type=str,
                        help="The name of the model")
    args = parser.parse_args()
    modelName = args.modelName

    myclient = MongoClient(
        host=current_app.config['MONGO_HOST'],
        port=int(current_app.config['MONGO_PORT']),
        username=current_app.config['MONGO_USER'],
        password=current_app.config['MONGO_PASS']
    )
    dbname = myclient["assistml"]
    collectionname = dbname["base_models"]
    collection_datasets = dbname["datasets"]

    model = collectionname.find({"Model.Info.name": modelName}).next()
    use_case = model["Model"]["Info"]["use_case"]
    dataset = collection_datasets.find({"Info.use_case": use_case}).next()

    missing_values = ["n/a", "na", "--", "NA", "?", "", " ", -1, "NAN", "NaN"]
    test_dataset = model['Model']['Metrics']['test_file']



    test_dataset_path = os.path.join(data_base_path, test_dataset)
    try:
        df = pd.read_csv(test_dataset_path, na_values=missing_values, index_col=False)
    except FileNotFoundError:
        current_app.logger.error(f"File {test_dataset_path} not found")
        return  # TODO: handle not existing test files
    used_columns = model['Model']['Data_Meta_Data']['list_of_columns_used']

    class_variable = model['Model']['Data_Meta_Data']['class_variable']
    df.rename(columns={'0': class_variable}, inplace=True)
    current_app.logger.info(df.columns)
    try:
        df.pop('Unnamed: 0')
    except:
        pass
    current_app.logger.info(df.columns)


    y = df.pop(class_variable)
    # try:
    #     y.replace({1: 0, 2: 1}, inplace=True)
    # except:
    #     pass
    if list(y.unique()) == [1, 2]:
        y.replace({1: 0, 2: 1}, inplace=True)
    X = df

    import joblib

    filename = os.path.join(pickle_file_base_path, model['Model']['Info']['name'] + '.pkl')
    # with open(filename, 'wb') as file:
    #    joblib.dump(lr, file)
    try:
        with open(filename, 'rb') as file:
            dt_pkl = joblib.load(file)
    except FileNotFoundError:
        current_app.logger.error(f"Model {filename} not found")
        return  # TODO: handle not existing pkl files

    y_pred = dt_pkl.predict(X)
    #y_pred = [0 if i == 1 else 1 for i in y_pred]
    accuracy = dt_pkl.score(X, y)
    current_app.logger.info("Accuracy:", accuracy)
    current_app.logger.info("Error:", 1 - accuracy)
    metrics = {}
    prec_recall = precision_recall_fscore_support(y, y_pred, average=None)
    p, r, f, sp = prec_recall
    # t = (stop - start)
    # t1 = (stop1 - start1)
    # t2 = (stop3 - start3)
    s = accuracy_score(y, y_pred)
    c = confusion_matrix(y, y_pred)
    d = c.tolist()
    # metrics['accuracy'] = s
    # metrics['error'] = 1 - s
    # metrics['precision'] = p[1]
    # metrics['recall'] = r[1]
    # metrics['fscore'] = f[1]
    # metrics["single_training_time"] = t
    # metrics["cross_validated_training_time"] = t1
    # metrics["test_time_per_unit"] = t2 / 11303
    # metrics["confusion_matrix_rowstrue_colspred"] = d
    # metrics["test_file"] = "steelplates_test_25.csv"
    # current_app.logger.info("Score= {}".format(s))
    # current_app.logger.info("Error= {}".format(1 - s))
    # current_app.logger.info("Precision,Recall,F_beta,Support {}".format(prec_recall))
    # current_app.logger.info(confusion_matrix(y, y_pred))

    num_columns = list(dataset['Features']['Numerical_Features'].keys())
    cat_columns = list(dataset['Features']['Categorical_Features'].keys())

    samples = 0
    samples_values = []
    for i in range(len(c)):
        for j in range(len(c[i])):
            samples_values.append(c[i][j])
    samples_values = np.array(samples_values)
    samples = samples_values.min()

    if samples > 100:
        samples = 100
    current_app.logger.info(c)
    current_app.logger.info(samples)
    # samples = math.ceil(samples * 0.1)
    TP = []
    FP = []
    TN = []
    FN = []
    current_app.logger.info(set(y_pred))
    current_app.logger.info(list(y.unique()))
    explainability = []
    for i in range(len(y_pred)):
        if y[i] == y_pred[i] == 1:
            if len(TP) < samples:
                x = X.loc[i].tolist()
                x.append(y[i])
                x.append(y_pred[i])
                x.append('TP')
                TP.append(x)
        if y_pred[i] == 1 and y[i] != y_pred[i]:
            if len(FP) < samples:
                x = X.loc[i].tolist()
                x.append(y[i])
                x.append(y_pred[i])
                x.append('FP')
                FP.append(x)
        if y[i] == y_pred[i] == 0:
            if len(TN) < samples:
                x = X.loc[i].tolist()
                x.append(y[i])
                x.append(y_pred[i])
                x.append('TN')
                TN.append(x)
        if y_pred[i] == 0 and y[i] != y_pred[i]:
            if len(FN) < samples:
                x = X.loc[i].tolist()
                x.append(y[i])
                x.append(y_pred[i])
                x.append('FN')
                FN.append(x)
    cm = [TP, FP, TN, FN]
    current_app.logger.info(len(cm[3]))
    for i in range(4):
        for j in range(samples):
            explainability.append(cm[i][j])


    columns = used_columns
    try:
        columns.remove(class_variable)
    except:
        pass

    columns.append('Y_True')
    columns.append('Y_Prediction')
    columns.append('Classification_Outcome')
    # cat_columns.append('Y_True')
    # cat_columns.append('Y_Prediction')
    # cat_columns.append('Classification_Outcome')
    num_columns_modified = []
    cat_columns_modified = []
    for column in used_columns:
        for num_column in num_columns:
            if column in num_column:
                num_columns_modified.append(column)
        for cat_column in cat_columns:
            if column in cat_column:
                cat_columns_modified.append(column)
    num_columns = num_columns_modified
    cat_columns = cat_columns_modified
    if not (len(num_columns) == 0):
        TP_df_num = pd.DataFrame(TP, columns=columns)[num_columns]
        TP_df_num = pd.concat([TP_df_num[col].sort_values(ascending=True, ignore_index=True) for col in TP_df_num],
                              axis=1,
                              ignore_index=True)
        TP_df_num.columns = num_columns

        FP_df_num = pd.DataFrame(FP, columns=columns)[num_columns]
        FP_df_num = pd.concat([FP_df_num[col].sort_values(ascending=True, ignore_index=True) for col in FP_df_num],
                              axis=1,
                              ignore_index=True)
        FP_df_num.columns = num_columns

        TN_df_num = pd.DataFrame(TN, columns=columns)[num_columns]
        TN_df_num = pd.concat([TN_df_num[col].sort_values(ascending=True, ignore_index=True) for col in TN_df_num],
                              axis=1,
                              ignore_index=True)
        TN_df_num.columns = num_columns

        FN_df_num = pd.DataFrame(FN, columns=columns)[num_columns]
        FN_df_num = pd.concat([FN_df_num[col].sort_values(ascending=True, ignore_index=True) for col in FN_df_num],
                              axis=1,
                              ignore_index=True)
        FN_df_num.columns = num_columns

    if not len(cat_columns) == 0:
        FP_df_cat = pd.DataFrame(FP, columns=columns)[cat_columns]
        FP_df_cat = pd.concat([FP_df_cat[col].sort_values(ascending=True, ignore_index=True) for col in FP_df_cat],
                              axis=1,
                              ignore_index=True)
        FP_df_cat.columns = cat_columns

        TN_df_cat = pd.DataFrame(TN, columns=columns)[cat_columns]
        TN_df_cat = pd.concat([TN_df_cat[col].sort_values(ascending=True, ignore_index=True) for col in TN_df_cat],
                              axis=1,
                              ignore_index=True)
        TN_df_cat.columns = cat_columns

        FN_df_cat = pd.DataFrame(FN, columns=columns)[cat_columns]
        FN_df_cat = pd.concat([FN_df_cat[col].sort_values(ascending=True, ignore_index=True) for col in FN_df_cat],
                              axis=1,
                              ignore_index=True)
        FN_df_cat.columns = cat_columns

        TP_df_cat = pd.DataFrame(TP, columns=columns)[cat_columns]
        TP_df_cat = pd.concat([TP_df_cat[col].sort_values(ascending=True, ignore_index=True) for col in TP_df_cat],
                              axis=1,
                              ignore_index=True)
        TP_df_cat.columns = cat_columns

    if not len(num_columns) == 0:
        positives_subtract = FP_df_num.subtract(TP_df_num).abs()
        negatives_subtract = FN_df_num.subtract(TN_df_num).abs()
        positives_subtract_sum = positives_subtract.sum()
        negatives_subtract_sum = negatives_subtract.sum()
        TP_mean = TP_df_num.mean()
        FP_mean = FP_df_num.mean()
        TN_mean = TN_df_num.mean()
        FN_mean = FN_df_num.mean()
        P_mean = (TP_mean + FP_mean) / 2
        N_mean = (TN_mean + FN_mean) / 2
        gap_num_positives = positives_subtract_sum / P_mean
        gap_num_negatives = negatives_subtract_sum / N_mean
        gap_num_positives_lst = list(gap_num_positives)
        gap_num_negatives_lst = list(gap_num_negatives)

    gap_negatives_dict = {}
    gap_positives_dict = {}

    if not len(num_columns) == 0:
        for i, col in enumerate(num_columns):
            gap_negatives_dict[col] = [gap_num_negatives_lst[i]]
            gap_positives_dict[col] = [gap_num_positives_lst[i]]

    if not len(cat_columns) == 0:
        gap_cat_negatives_list = []
        for col in FN_df_cat:
            gap_cat_negatives_list.append(((FN_df_cat[col].value_counts() / len(FN_df_cat[col])) - (
                    TN_df_cat[col].value_counts() / len(TN_df_cat[col]))).abs().sum())
        gap_cat_positives_list = []
        for col in FP_df_cat:
            gap_cat_positives_list.append(((FP_df_cat[col].value_counts() / len(FP_df_cat[col])) - (
                    TP_df_cat[col].value_counts() / len(TP_df_cat[col]))).abs().sum())

    if not len(cat_columns) == 0:
        for i, col in enumerate(cat_columns):
            gap_negatives_dict[col] = [gap_cat_negatives_list[i]]
            gap_positives_dict[col] = [gap_cat_positives_list[i]]

    gap_positives = pd.DataFrame(gap_positives_dict)
    gap_negatives = pd.DataFrame(gap_negatives_dict)

    gap_average = (gap_positives + gap_negatives) / 2

    new_columns = num_columns + cat_columns
    new_explainability = []
    new_explainability.append(gap_positives.values[0])
    new_explainability.append(gap_negatives.values[0])
    new_explainability.append(gap_average.values[0])
    index = ['positives', 'negatives', 'total']
    new_explainability_out = pd.DataFrame(new_explainability, columns=new_columns, index=index)

    current_app.logger.info(new_explainability_out)
    model['Model']['Metrics']['Explainability'] = {}
    model['Model']['Metrics']['Explainability'] = new_explainability_out.to_dict()
    collectionname.delete_one({"_id": model["_id"]})
    new_model = {}
    new_model['Model'] = model['Model']
    collectionname.insert_one(new_model)

    new_explainability_out.to_csv(r'../explain_DTR_steelplates_001.csv', index=False)

    explainability_out = pd.DataFrame(explainability, columns=columns)

    # explainability_out.to_csv(r'/home/mohammoa/HiWi/asm-2/3_pickle/explainability_steelplates.csv', index=False)

    # import csv
    # with open("/home/mohammoa/HiWi/asm-2/3_pickle/explainability.csv","w") as my_csv:
    #     csvWriter = csv.writer(my_csv,delimiter=',')
    #     csvWriter.writerows(TP)
    #     csvWriter.writerows(TN)
    #     csvWriter.writerows(FP)
    #     csvWriter.writerows(FN)

    # if os.path.exists('/home/mohammoa/HiWi/asm-2/3_pickle/DTR_steelplates_001.json'):
    #    with open('/home/mohammoa/HiWi/asm-2/3_pickle/DTR_steelplates_001.json', 'r') as f:
    #        models = json.load(f)
    #    models["DTR_steelplates_001"]["Metrics"] = metrics
    #    with open('/home/mohammoa/HiWi/asm-2/3_pickle/DTR_steelplates_001.json', 'w') as f:
    #        json.dump(models, f, indent=2)


if __name__ == '__main__':

    main()
