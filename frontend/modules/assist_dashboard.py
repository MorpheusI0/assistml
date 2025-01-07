import base64
import csv
import glob
import io
import json
import os
import re

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import pandas as pd
import plotly.express as px
import pymongo
import requests
from dash.dependencies import Input, Output, State

from data_profiler import DataProfiler

json_renderer_theme = {
    "scheme": "monokai",
    "author": "wimer hazenberg (http://www.monokai.nl)",
    "base00": "#272822",
    "base01": "#383830",
    "base02": "#49483e",
    "base03": "#75715e",
    "base04": "#a59f85",
    "base05": "#f8f8f2",
    "base06": "#f5f4f1",
    "base07": "#f9f8f5",
    "base08": "#f92672",
    "base09": "#fd971f",
    "base0A": "#f4bf75",
    "base0B": "#a6e22e",
    "base0C": "#a1efe4",
    "base0D": "#66d9ef",
    "base0E": "#ae81ff",
    "base0F": "#cc6633",
}

numerical_features_keys = []
response_json = {}
categorical_features_keys = ['Name', 'Missing values percentage', 'Mutual info', 'Monotonous filtering']
working_dir = os.path.expanduser(os.getenv('WORKING_DIR', "~/.assistml/dashboard"))
os.makedirs(working_dir, exist_ok=True)

def background_color(grade):
    color = 'White'
    if type(grade) == str:
        if grade == 'A+':
            color = 'lightblue'
        elif grade == 'A':
            color = 'green'
        elif grade == 'B':
            color = 'lightgreen'
        elif grade == 'C':
            color = 'gold'
        elif grade == 'D':
            color = 'darkgoldenrod'
        else:
            color = 'red'

    if type(grade) == float:
        if grade >= 0.95:
            color = 'lightblue'
        elif grade >= 0.90:
            color = 'green'
        elif grade >= 0.85:
            color = 'lightgreen'
        elif grade >= 0.75:
            color = 'gold'
        elif grade >= 0.65:
            color = 'darkgoldenrod'
        else:
            color = 'red'

    return color


def generate_acceptable_or_nearly_table(acceptable_model):
    if not 'rules' in acceptable_model:
        rules = "No notes are provided for this solution"
    else:
        rules = acceptable_model["rules"]
    place_holder = "There should be some text here! "
    deploy_text = "Deployed in " + str(acceptable_model["deployment"]) + " with " + str(
        acceptable_model["cores"]) + " cores and power: " + str(acceptable_model["power"]) + " GhZ"
    # for difference in acceptable_model["differences"]:
    #     if 'TRUE' in difference:
    #         column_name = difference.replace(" TRUE", "")
    #         remarks_string = remarks_string + '(' + column_name + '),   '
    # remarks_string = remarks_string[:-1]
    model_code = re.search('([A-Z]{3})_[a-zA-Z0-9]+_\w+', acceptable_model["code"])
    model_code = model_code[1]

    return html.Table(
        [
            html.Tr(
                [
                    html.Td(model_code, rowSpan='9',
                            style={'border-style': 'solid', 'border-width': '1px', 'textAlign': 'center',
                                   'color': 'black'}),
                    html.Td(acceptable_model["name"], colSpan='8',
                            style={'border-style': 'solid', 'border-width': '2px', 'textAlign': 'center',
                                   'font-weight': 'bold', 'backgroundColor': '#8DAD26', 'color': 'black'})
                ]
            ),
            html.Tr(
                [
                    html.Td('Accuracy', rowSpan='1', colSpan='1',
                            style={'border-style': 'solid', 'border-width': '1px', 'font-weight': 'bold',
                                   'color': 'black', }),
                    html.Td(acceptable_model["performance"]["accuracy"], rowSpan='1', colSpan='1',
                            style={'border-style': 'solid', 'border-width': '1px', 'color': 'black',
                                   'backgroundColor': background_color(acceptable_model["performance"]["accuracy"]), }),
                    html.Td('Precision', rowSpan='1', colSpan='1',
                            style={'border-style': 'solid', 'border-width': '1px', 'font-weight': 'bold',
                                   'color': 'black', }),
                    html.Td(acceptable_model["performance"]["precision"], rowSpan='1', colSpan='1',
                            style={'border-style': 'solid', 'border-width': '1px', 'color': 'black',
                                   'backgroundColor': background_color(
                                       acceptable_model["performance"]["precision"]), }),
                    html.Td('Recall', rowSpan='1', colSpan='1',
                            style={'border-style': 'solid', 'border-width': '1px', 'font-weight': 'bold',
                                   'color': 'black', }),
                    html.Td(acceptable_model["performance"]["recall"], rowSpan='1', colSpan='1',
                            style={'border-style': 'solid', 'border-width': '1px', 'color': 'black',
                                   'backgroundColor': background_color(acceptable_model["performance"]["recall"]), }),
                    html.Td('Training Time', rowSpan='1', colSpan='1',
                            style={'border-style': 'solid', 'border-width': '1px', 'color': 'black', }),
                    html.Td(acceptable_model["performance"]["training_time"], rowSpan='1', colSpan='1',
                            style={'border-style': 'solid', 'border-width': '1px', 'font-weight': 'bold',
                                   'color': 'black', 'backgroundColor': background_color(
                                    acceptable_model["performance"]["training_time"]), }),
                ]
                , style={'border-style': 'solid', 'border-width': '2px', 'color': 'black'}
            ),
            html.Tr(
                [
                    html.Td('Output analysis', rowSpan='1', colSpan='8',
                            style={'border-style': 'solid', 'border-width': '2px', 'font-weight': 'bold', }),
                ]
            ),
            html.Tr(
                [
                    html.Td(acceptable_model["out_analysis"], rowSpan='1', colSpan='8',
                            style={'border-style': 'solid', 'border-width': '1px'}),
                ]
            ),
            html.Tr(
                [
                    html.Td('Data Preprocessing', rowSpan='1', colSpan='8',
                            style={'border-style': 'solid', 'border-width': '2px', 'font-weight': 'bold', }),
                ]
            ),
            html.Tr(
                [
                    html.Td(acceptable_model["preprocessing"], rowSpan='1', colSpan='8',
                            style={'border-style': 'solid', 'border-width': '1px'}),
                ]
            ),
            html.Tr(
                [
                    html.Td('ML solution patterns', rowSpan='1', colSpan='8',
                            style={'border-style': 'solid', 'border-width': '2px', 'font-weight': 'bold', }),
                ]
            ),
            html.Tr(
                [
                    html.Td(rules, rowSpan='1', colSpan='8',
                            style={'border-style': 'solid', 'border-width': '1px'}),
                ]
            ),
            html.Tr(
                [
                    html.Td('Deployment description', rowSpan='1', colSpan='8',
                            style={'border-style': 'solid', 'border-width': '2px', 'font-weight': 'bold', }),
                ]
            ),
            html.Tr(
                [
                    html.Td('(overall) Score: ' + str(acceptable_model["overall_score"]), rowSpan='1', colSpan='1',
                            style={'border-style': 'solid', 'border-width': '1px',
                                   'backgroundColor': background_color(acceptable_model["overall_score"]), }),
                    html.Td(deploy_text, rowSpan='1', colSpan='8',
                            style={'border-style': 'solid', 'border-width': '1px', }),
                ]
            ),
            html.Tr(
                [
                    html.Td(acceptable_model["code"], rowSpan='1', colSpan='1',
                            style={'border-style': 'solid', 'border-width': '1px'}),
                    html.Td('Language', rowSpan='1', colSpan='1',
                            style={'border-style': 'solid', 'border-width': '1px', 'font-weight': 'bold', }),
                    html.Td(acceptable_model["language"], rowSpan='1', colSpan='1',
                            style={'border-style': 'solid', 'border-width': '1px'}),
                    html.Td('Implementation', rowSpan='1', colSpan='1',
                            style={'border-style': 'solid', 'border-width': '1px', 'font-weight': 'bold', }),
                    html.Td(acceptable_model["platform"], rowSpan='1', colSpan='1',
                            style={'border-style': 'solid', 'border-width': '1px'}),
                    html.Td('Nr Dependencies', rowSpan='1', colSpan='1',
                            style={'border-style': 'solid', 'border-width': '1px', 'font-weight': 'bold', }),
                    html.Td(acceptable_model["nr_dependencies"], rowSpan='1', colSpan='1',
                            style={'border-style': 'solid', 'border-width': '1px'}),
                    html.Td('Nr Parameters', rowSpan='1', colSpan='1',
                            style={'border-style': 'solid', 'border-width': '1px', 'font-weight': 'bold', }),
                    html.Td(acceptable_model["nr_hparams"], rowSpan='1', colSpan='1',
                            style={'border-style': 'solid', 'border-width': '1px', 'color': 'black', }),
                ]
                , style={'border-style': 'solid', 'border-width': '2px', }
            ),
        ],
        style={'border-collapse': 'collapse', 'border-spacing': '0.1', 'border-width': '2px', 'color': 'black', },
    )


def generate_table(acceptable_models, nearly_acceptable_models):
    return html.Div([
        html.Br(),
        html.H5(children='Acceptable Models', style={'font-weight': 'bold', }),
        html.Div([generate_acceptable_or_nearly_table(acceptable_model) for acceptable_model in acceptable_models],
                 style={'display': 'inline-block'}),
        html.Br(),
        html.H5(children='Nearly Acceptable Models', style={'font-weight': 'bold', }),
        html.Div([generate_acceptable_or_nearly_table(nearly_acceptable_model) for nearly_acceptable_model in
                  nearly_acceptable_models], style={'display': 'inline-block'}),
    ],
        style={'width': '90%', 'display': 'inline-block'})


def generate_summary(summary):
    distrust = "The distrust score for is: " + str((summary["distrust_score"]) * 100) + "%"
    warnings = summary["warnings"]
    warnings_string = ""
    try:
        no_of_acceptable = summary["acceptable_models"]
    except:
        no_of_acceptable = 0
    try:
        no_of_nearly_acceptable = summary["nearly_acceptable_models"]
    except:
        no_of_nearly_acceptable = 0
    if (len(warnings) > 0):
        distrust += " and the reason for this is the following:"
        for warning in warnings:
            warnings_string += "\n* " + warning
    else:
        distrust += "."
    return html.Div([
        html.H1('Query results'),
        html.Div([
            html.P(
                "There is " + str(no_of_acceptable) + " acceptable models that match your query and " + str(
                    no_of_nearly_acceptable) + " nearly acceptable models."),
            html.P(distrust),
            dcc.Markdown(warnings_string)
        ])
    ])


def plotting_response(acceptable_models, nearly_acceptable_models):
    training_time_std = []
    recall = []
    accuracy = []
    acceptable_or_nearly = []
    models_names = []
    client = pymongo.MongoClient(
        host=os.getenv('MONGO_HOST', 'localhost'),
        port=int(os.getenv('MONGO_PORT', 27017)),
        username=os.getenv('MONGO_USER', 'admin'),
        password=os.getenv('MONGO_PASS', 'admin')
    )
    db = client["assistml"]
    enriched_models = db["enriched_models"]

    for acceptable_model in acceptable_models:
        model_name = acceptable_model["code"]
        model = enriched_models.find_one({'model_name': model_name})
        training_time_std.append(model["training_time_std"])
        recall.append(model["recall"])
        accuracy.append(model["accuracy"])
        acceptable_or_nearly.append("Acceptable")
        models_names.append(model["model_name"])
    for nearly_acceptable_model in nearly_acceptable_models:
        model_name = nearly_acceptable_model["code"]
        model = enriched_models.find_one({'model_name': model_name})
        training_time_std.append(model["training_time_std"])
        recall.append(model["recall"])
        accuracy.append(model["accuracy"])
        acceptable_or_nearly.append("Nearly Acceptable")
        models_names.append(model["model_name"])

    dictionary = {'Model Name': models_names, 'Recommended Model Type': acceptable_or_nearly,
                  'Training Time': training_time_std, 'Recall': recall, 'Accuracy': accuracy}
    df = pd.DataFrame(dictionary)
    fig = px.scatter_3d(df, text="Model Name", x='Training Time', y='Recall', z='Accuracy',
                        color='Recommended Model Type', color_discrete_sequence=["green", "orange"])

    fig.update_layout(margin=dict(l=5, r=5, t=5, b=5))
    return dcc.Graph(id='response_plot', figure=fig)


def upload_dataset_R_backend(file_content):
    url = os.getenv('BACKEND_BASE_URL', 'http://localhost:8080') + "/upload"
    upload_dir = os.path.join(working_dir, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    csv_files = sorted(glob.glob(os.path.join(upload_dir,"*.csv")),key=os.path.getmtime)
    # csv_files.sort(key=os.path.getctime)
    file = csv_files[-1].split("/")[-1]
    print(file)
    print("\n")
    with open(os.path.join(upload_dir, file), "r") as dataset_uploaded:
        file_dict = {str(file): dataset_uploaded}
        print(file_dict)
        response = requests.post(url, files=file_dict)
        print(response.text)
        return response.status_code


def api_call_R_backend(class_feature_type, feature_type_list, classification_output, accuracy_slider, precision_slider,
                       recall_slider, trtime_slider, use_case, csv_filename):
    # algorithm_family, deployment_type,
    #                   platform, implementation_lang,
    #                   language, tuning_slider,
    global response_json
    feature_type_list = feature_type_list.replace(' ', '')
    feature_type_list = feature_type_list.replace("'", '')
    feature_type_list = feature_type_list.replace('"', '')
    feature_type_list = list(feature_type_list.strip('[]').split(','))

    url = os.getenv('BACKEND_BASE_URL', 'http://localhost:8080') + "/assistml"
    params_json = {
        "classif_type": class_feature_type,
        "classif_output": classification_output,
        # "deployment": deployment_type,
        # "implementation": implementation_lang,
        "sem_types": feature_type_list,
        # "lang": language,
        # "algofam": algorithm_family,
        # "platform": platform,
        # "tuning_limit": tuning_slider,
        "accuracy_range": accuracy_slider,
        "precision_range": precision_slider,
        "recall_range": recall_slider,
        "trtime_range": trtime_slider,
        "dataset_name": csv_filename,
        "usecase": use_case
        # "dataset": "dataset.csv"
    }
    print(params_json)
    response = requests.post(url=url, data=json.dumps(params_json), headers={'Content-Type': 'application/json'})
    print(response.text)

    if response.status_code == 200:
        response_json = response.json()
        print(response_json)
        try:
            acceptable_models = response_json["acceptable_models"]
        except:
            print("There are no acceptable models!")
            acceptable_models = {}
        try:
            nearly_acceptable_models = response_json["nearly_acceptable_models"]
        except:
            print("There are no nearly acceptable models!")
            nearly_acceptable_models = {}
        summary = response_json["summary"]
        table_html = generate_table(acceptable_models, nearly_acceptable_models)
        generated_summary = generate_summary(summary)
        response_plot = plotting_response(acceptable_models, nearly_acceptable_models)
        return html.Div([
            generated_summary,
            response_plot,
            table_html,
            # html.H5(children='Query Result', style={'font-weight': 'bold', }),
            # dash_renderjson.DashRenderjson(id="input", data=response_json, max_depth=0, theme=json_renderer_theme,
            #                               invert_theme=True),
            # html.Div([query_issued_tag], ),
        ])
    else:
        return html.Div([
            html.H6(children='Execution of Remote R backend terminated with status code ' + str(response.status_code),
                    style={'font-weight': 'bold', }),
        ])

        # Sample response to be used for testing.
        # TODO: Remove in future.
        # response_json = '{"summary" : { "query_issued" : {"number" : 72,"madeat" : "2020-10-27 11:52:34",             "classif_type" : "binary",             "classif_output" : "single",             "deployment" : "single_host",             "dataset" : "dataset.csv",             "semantic_types" : [                  "N",                  "N",                  "N",                  "N",                  "N",                  "N",                  "N",                  "N",                  "N",                  "N",                  "N",                  "C",                  "C",                  "N",                  "N",                  "N",                  "N",                  "N",                  "N",                  "N",                  "C",                  "N",                  "N",                  "N",                  "N",                  "N",                  "N",                  "T"             ],             "accuracy_range" : 0.1,             "precision_range" : 0.15,             "recall_range" : 0.2,             "traintime_range" : 0.25,             "pref_language" : "python",             "pref_algofam" : "DTR",             "pref_platform" : "scikit",             "tuning_limit" : 3,             "pref_implementation" : "single_language"         },         "acceptable_models" : 7,         "nearly_acceptable_models" : 5,         "distrust_score" : 0.4286,         "warnings" : [              "Dataset similarity level 1 only. Distrust Pts +2",              "The selection of acceptable models was not as clean as possible. Distrust Pts+1",              "The selection of nearly acceptable models was not as clean as possible. Distrust Pts+3"         ]     },     "acceptable_models" : [          {             "name" : "Model trained with Random Forests",             "language" : "python",             "platform" : "sklearn.ensemble.ExtraTreeClassifier",             "nr_hparams" : 14,             "similarity_to_query" : 0.333333333333333,             "differences" : [                  "deployment FALSE",                  "fam_name TRUE",                  "first_datatype TRUE",                  "implementation FALSE",                  "language FALSE",                  "nr_hyperparams_label TRUE",                  "rows TRUE",                  "second_datatype FALSE"             ],             "overall_score" : 0.9914,             "performance" : {                 "accuracy" : "A+",                 "precision" : "A+",                 "recall" : "A",                 "training time" : "C"             },             "code" : "RFR_kick_011"         },          {             "name" : "Model trained with Random Forests",             "language" : "python",             "platform" : "sklearn.ensemble.RandomForestClassifier",             "nr_hparams" : 6,             "similarity_to_query" : 0.333333333333333,             "differences" : [                  "deployment FALSE",                  "fam_name TRUE",                  "first_datatype TRUE",                  "implementation FALSE",                  "language FALSE",                  "nr_hyperparams_label TRUE",                  "rows TRUE",                  "second_datatype FALSE"             ],             "overall_score" : 0.9875,             "performance" : {                 "accuracy" : "A",                 "precision" : "A",                 "recall" : "A",                 "training time" : "D"             },             "code" : "RFR_kick_030"         },          {             "name" : "Model trained with Random Forests",             "language" : "python",             "platform" : "sklearn.ensemble.RandomForestClassifier",             "nr_hparams" : 6,             "similarity_to_query" : 0.333333333333333,             "differences" : [                  "deployment FALSE",                  "fam_name TRUE",                  "first_datatype TRUE",                  "implementation FALSE",                  "language FALSE",                  "nr_hyperparams_label TRUE",                  "rows TRUE",                  "second_datatype FALSE"             ],             "overall_score" : 0.9875,             "performance" : {                 "accuracy" : "A",                 "precision" : "A",                 "recall" : "A",                 "training time" : "C"             },             "code" : "RFR_kick_028"         },          {             "name" : "Model trained with Decision Trees",             "language" : "python",             "platform" : "sklearn.tree.DecisionTreeClassifier",             "nr_hparams" : 3,             "similarity_to_query" : 0.5,             "differences" : [                  "deployment FALSE",                  "fam_name FALSE",                  "first_datatype TRUE",                  "implementation FALSE",                  "language FALSE",                  "nr_hyperparams_label TRUE",                  "rows TRUE",                  "second_datatype FALSE"             ],             "overall_score" : 0.9539,             "performance" : {                 "accuracy" : "A",                 "precision" : "A",                 "recall" : "A",                 "training time" : "C"             },             "code" : "DTR_kick_019"         },          {             "name" : "Model trained with Decision Trees",             "language" : "python",             "platform" : "sklearn.tree.ExtraTreeClassifier",             "nr_hparams" : 3,             "similarity_to_query" : 0.5,             "differences" : [                  "deployment FALSE",                  "fam_name FALSE",                  "first_datatype TRUE",                  "implementation FALSE",                  "language FALSE",                  "nr_hyperparams_label TRUE",                  "rows TRUE",                  "second_datatype FALSE"             ],             "overall_score" : 0.9537,             "performance" : {                 "accuracy" : "A",                 "precision" : "A",                 "recall" : "A",                 "training time" : "C"             },             "code" : "DTR_kick_012"         },          {             "name" : "Model trained with Naive Bayes",             "language" : "python",             "platform" : "sklearn.naive_bayes.BernoulliNB",             "nr_hparams" : 2,             "similarity_to_query" : 0.333333333333333,             "differences" : [                  "deployment FALSE",                  "fam_name TRUE",                  "first_datatype TRUE",                  "implementation FALSE",                  "language FALSE",                  "nr_hyperparams_label TRUE",                  "rows TRUE",                  "second_datatype FALSE"             ],             "overall_score" : 0.9203,             "performance" : {                 "accuracy" : "D",                 "precision" : "A",                 "recall" : "A",                 "training time" : "A+"             },             "code" : "NBY_bank_001"         },          {             "name" : "Model trained with Support Vector Machines",             "language" : "python",             "platform" : "sklearn.kernel_approximation.Nystroem + LinearSVC",             "nr_hparams" : 14,             "similarity_to_query" : 0.333333333333333,             "differences" : [                  "deployment FALSE",                  "fam_name TRUE",                  "first_datatype TRUE",                  "implementation FALSE",                  "language FALSE",                  "nr_hyperparams_label TRUE",                  "rows TRUE",                  "second_datatype FALSE"             ],             "overall_score" : 0.907,             "performance" : {                 "accuracy" : "B",                 "precision" : "B",                 "recall" : "A",                 "training time" : "A+"             },             "code" : "SVM_kick_004"         }     ],     "nearly_acceptable_models" : [          {             "name" : "Model trained with Decision Trees",             "language" : "python",             "platform" : "sklearn.tree.DecisionTreeClassifier",             "nr_hparams" : 3,             "similarity_to_query" : 0.5,             "differences" : [                  "deployment FALSE",                  "fam_name FALSE",                  "first_datatype TRUE",                  "implementation FALSE",                  "language FALSE",                  "nr_hyperparams_label TRUE",                  "rows TRUE",                  "second_datatype FALSE"             ],             "overall_score" : 0.8178,             "performance" : {                 "accuracy" : "E",                 "precision" : "B",                 "recall" : "B",                 "training time" : "B"             },             "code" : "DTR_kick_016"         },          {             "name" : "Model trained with Random Forests",             "language" : "python",             "platform" : "sklearn.tree.ExtraTreeClassifier",             "nr_hparams" : 3,             "similarity_to_query" : 0.333333333333333,             "differences" : [                  "deployment FALSE",                  "fam_name TRUE",                  "first_datatype TRUE",                  "implementation FALSE",                  "language FALSE",                  "nr_hyperparams_label TRUE",                  "rows TRUE",                  "second_datatype FALSE"             ],             "overall_score" : 0.813,             "performance" : {                 "accuracy" : "E",                 "precision" : "B",                 "recall" : "B",                 "training time" : "C"             },             "code" : "RFR_kick_008"         },          {             "name" : "Model trained with Gradient Boosting Ensemble",             "language" : "python",             "platform" : "h2o.estimators.gbm",             "nr_hparams" : 4,             "similarity_to_query" : 0.333333333333333,             "differences" : [                  "deployment FALSE",                  "fam_name TRUE",                  "first_datatype TRUE",                  "implementation FALSE",                  "language FALSE",                  "nr_hyperparams_label TRUE",                  "rows TRUE",                  "second_datatype FALSE"             ],             "overall_score" : 0.787,             "performance" : {                 "accuracy" : "B",                 "precision" : "D",                 "recall" : "B",                 "training time" : "C"             },             "code" : "GBE_bank_001"         },          {             "name" : "Model trained with Deep Learning",             "language" : "python",             "platform" : "h2o.estimators.deeplearning",             "nr_hparams" : 10,             "similarity_to_query" : 0.333333333333333,             "differences" : [                  "deployment FALSE",                  "fam_name TRUE",                  "first_datatype TRUE",                  "implementation FALSE",                  "language FALSE",                  "nr_hyperparams_label TRUE",                  "rows TRUE",                  "second_datatype FALSE"             ],             "overall_score" : 0.7815,             "performance" : {                 "accuracy" : "D",                 "precision" : "E",                 "recall" : "B",                 "training time" : "B"             },             "code" : "DLN_bank_014"         },          {             "name" : "Model trained with Deep Learning",             "language" : "python",             "platform" : "h2o.estimators.deeplearning",             "nr_hparams" : 10,             "similarity_to_query" : 0.333333333333333,             "differences" : [                  "deployment FALSE",                  "fam_name TRUE",                  "first_datatype TRUE",                  "implementation FALSE",                  "language FALSE",                  "nr_hyperparams_label TRUE",                  "rows TRUE",                  "second_datatype FALSE"             ],             "overall_score" : 0.7794,             "performance" : {                 "accuracy" : "C",                 "precision" : "D",                 "recall" : "B",                 "training time" : "B"             },             "code" : "DLN_bank_011"         }     ] }'


def construct_json_numerical_features(numerical_feature, feature_name, missing_values_percent):
    selected_feature_dict = {}
    selected_feature_dict['Name'] = feature_name
    # selected_feature_dict['Number of outliers'] = numerical_feature['Outliers']['number']
    # selected_feature_dict['Minimum value'] = numerical_feature['Quartiles']['q0']
    # selected_feature_dict['Maximum value'] = numerical_feature['Quartiles']['q4']
    # if 'numeric' in class_feature_type:
    # selected_feature_dict['Correlation'] = numerical_feature['correlation']
    selected_feature_dict['Mutual info'] = round(numerical_feature['mutual_info'], 3)
    selected_feature_dict['Missing values percentage'] = round(missing_values_percent, 3)
    selected_feature_dict['Monotonous filtering'] = round(numerical_feature['monotonous_filtering'], 3)
    # else:
    # selected_feature_dict['Chi square test - True'] = numerical_feature['Correlation']['p_val']
    #    selected_feature_dict['mutual_info'] = numerical_feature['mutual_info']
    # if numerical_feature['Distribution']['normal'] == True:
    #    selected_feature_dict['Distribution'] = 'normal'
    # elif numerical_feature['Distribution']['exponential'] == True:
    #    selected_feature_dict['Distribution'] = 'exponential'
    # else:
    #    selected_feature_dict['Distribution'] = 'none'
    return selected_feature_dict


def suggest_features_to_user(dataset_info_json, class_feature_type):
    global numerical_features_keys
    selected_features = {}
    rows_nr = dataset_info_json['Info']['observations']

    # Numerical features
    selected_features['numerical_features'] = []
    numerical_features = dataset_info_json['Features']['Numerical_Features']
    if "numeric" in class_feature_type:
        for feature in numerical_features:
            missing_values_nr = numerical_features[feature]['missing_values']
            missing_values_percent = (missing_values_nr / rows_nr) * 100
            numerical_features_keys = ['Name', 'Missing values percentage', 'Mutual info', 'Monotonous filtering']
            # correlation = abs(numerical_features[feature]['correlation'])
            # pvalue_anova = numerical_features[feature]['anova_pvalue']
            monotonous_filtering = numerical_features[feature]['monotonous_filtering']
            mutual_info = numerical_features[feature]['mutual_info']
            if missing_values_percent < 20 and mutual_info >= 0.01 and monotonous_filtering > 0.5 and monotonous_filtering < 0.9:
                selected_feature_dict = construct_json_numerical_features(numerical_features[feature], feature,
                                                                          missing_values_percent)
                selected_features['numerical_features'].append(selected_feature_dict)
    else:
        for feature in numerical_features:
            missing_values_nr = numerical_features[feature]['missing_values']
            missing_values_percent = (missing_values_nr / rows_nr) * 100
            numerical_features_keys = ['Name', 'Missing values percentage', 'Mutual info', 'Monotonous filtering']
            # chisq_correlated = numerical_features[feature]['Correlation']['chisq_correlated']
            # pvalue_anova = numerical_features[feature]['anova_pvalue']
            mutual_info = numerical_features[feature]['mutual_info']
            monotonous_filtering = numerical_features[feature]['monotonous_filtering']
            # or pvalue_anova<0.05
            # if chisq_correlated == 'True' :
            if missing_values_percent < 20 and mutual_info >= 0.01 and monotonous_filtering > 0.5 and monotonous_filtering < 0.9:
                selected_feature_dict = construct_json_numerical_features(numerical_features[feature], feature,
                                                                          missing_values_percent)
                selected_features['numerical_features'].append(selected_feature_dict)

    # Categorical features
    selected_features['categorical_features'] = []
    categorical_features = dataset_info_json['Features']['Categorical_Features']
    for feature in categorical_features:
        missing_values_nr = categorical_features[feature]['missing_values']
        missing_values_percent = (missing_values_nr / rows_nr) * 100
        # chisq_correlated = categorical_features[feature]['Correlation']['chisq_correlated']
        mutual_info = categorical_features[feature]['mutual_info']
        monotonous_filtering = categorical_features[feature]['monotonous_filtering']
        # if chisq_correlated == 'True':
        if missing_values_percent < 20 and mutual_info >= 0.01 and monotonous_filtering > 0.5 and monotonous_filtering < 0.9:
            selected_feature_dict = {}
            selected_feature_dict['Name'] = feature
            # selected_feature_dict['Imbalance'] = categorical_features[feature]['imbalance']
            # selected_feature_dict['Number of levels'] = categorical_features[feature]['nr_levels']
            selected_feature_dict['Mutual info'] = round(categorical_features[feature]['mutual_info'], 3)
            selected_feature_dict['Missing values percentage'] = round(missing_values_percent, 3)
            selected_feature_dict['Monotonous filtering'] = round(categorical_features[feature]['monotonous_filtering'], 3)
            selected_features['categorical_features'].append(selected_feature_dict)
    return selected_features


def construct_output_html(suggested_features):
    numerical_features_table = html.Div(style={'width': '90%'},
                                        children=[
                                            html.H5('List of important numerical features',
                                                    style={'font-weight': 'bold'}
                                                    ),
                                            dash_table.DataTable(
                                                data=suggested_features['numerical_features'],
                                                columns=[{'id': i, 'name': i} for i in numerical_features_keys],

                                                style_header={
                                                    'backgroundColor': 'rgb(204, 229, 255)',
                                                    "text-align": "left", 'justify': 'left',
                                                    'fontWeight': 'bold',

                                                },
                                                style_data_conditional=[
                                                    {
                                                        'if': {'row_index': 'odd'},
                                                        "text-align": "left", 'justify': 'left',
                                                        'backgroundColor': 'rgb(248, 248, 248)'
                                                    }
                                                ],
                                                style_data={
                                                    "text-align": "left", 'justify': 'left',
                                                },
                                                style_table={
                                                },
                                            ),
                                        ])
    categorical_features_table = html.Div(style={'width': '90%'},
                                          children=[
                                              html.H5('List of important categorical features',
                                                      style={'font-weight': 'bold'}),
                                              dash_table.DataTable(
                                                  data=suggested_features['categorical_features'],
                                                  columns=[{'id': i, 'name': i} for i in categorical_features_keys],

                                                  style_header={
                                                      'backgroundColor': 'rgb(204, 229, 255)',
                                                      "text-align": "left", 'justify': 'left',
                                                      'fontWeight': 'bold',

                                                  },
                                                  style_data_conditional=[
                                                      {
                                                          'if': {'row_index': 'odd'},
                                                          "text-align": "left", 'justify': 'left',
                                                          'backgroundColor': 'rgb(248, 248, 248)'
                                                      }
                                                  ],
                                                  style_data={
                                                      "text-align": "left", 'justify': 'left',
                                                  },
                                                  style_table={
                                                  },
                                              ),
                                          ])
    if suggested_features['numerical_features'] and suggested_features['categorical_features']:
        content = html.Div([numerical_features_table, categorical_features_table], )
    elif not suggested_features['numerical_features']:
        content = html.Div([categorical_features_table], )
    elif not suggested_features['categorical_features']:
        content = html.Div([numerical_features_table], )
    else:
        content = html.Div([])
    return content


### Main code ###

external_stylesheets = ["https://cdnjs.cloudflare.com/ajax/libs/normalize/7.0.0/normalize.min.css",
                        "https://cdnjs.cloudflare.com/ajax/libs/skeleton/2.0.4/skeleton.min.css",
                        "//fonts.googleapis.com/css?family=Raleway:400,300,600",
                        "https://codepen.io/bcd/pen/KQrXdb.css",
                        "https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css",
                        "https://codepen.io/dmcomfort/pen/JzdzEZ.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

SIDEBAR_STYLE = {
    "border-radius": "10px",
    "margin": "10px",
    "position": "relative",
    "left": 0,
    "top": 0,
    "bottom": 0,
    "width": "30rem",
    "padding": "1rem 1rem",
    "border": "1px solid #D1D1D",
    "float": "left",
}

# the styles for the main content position it to the right of the sidebar and_core
# add some padding.
CONTENT_STYLE = {
    'width': '70%',
    "float": "right",
}

header = html.Div([
    html.H1(
        children='Assist ML')],
    style={
        'textAlign': 'center',
        'color': '#000000',
        'backgroundColor': '#8DAD26'
    }
)

# Database details
# same id as the api 192.168.221.146
myclient = pymongo.MongoClient(
            host=os.getenv('MONGO_HOST', 'localhost'),
            port=int(os.getenv('MONGO_PORT', 27017)),
            username=os.getenv('MONGO_USER', 'admin'),
            password=os.getenv('MONGO_PASS', 'admin')
        )
dbname = myclient["assistml"]
collection_datasets = dbname["datasets"]
enriched_datasets = dbname["enriched_models"]
print("Connected to database !!")

use_case_list = []
use_case_list = collection_datasets.distinct("Info.use_case")
use_case_list.insert(0, "----no selection----")

nr_hyperparams_list = enriched_datasets.distinct("nr_hyperparams")
nr_hyperparams_max = max(nr_hyperparams_list)

language_list = enriched_datasets.distinct("language")
algo_fam_list = enriched_datasets.distinct("fam_name")
platform_list = enriched_datasets.distinct("platform")

classification_type_mode = list(enriched_datasets.aggregate([
    {
        '$match': {
            "classification_output": {'$exists': True}
        }
    }, {
        '$group': {
            '_id': "$classification_output",
            'count': {'$sum': 1}
        }
    }, {
        '$sort': {"count": -1}
    }, {
        '$limit': 1
    }
]))
classification_type_mode = classification_type_mode[0]["_id"]
fam_name_mode = list(enriched_datasets.aggregate([
    {
        '$match': {
            "fam_name": {'$exists': True}
        }
    }, {
        '$group': {
            '_id': "$fam_name",
            'count': {'$sum': 1}
        }
    }, {
        '$sort': {"count": -1}
    }, {
        '$limit': 1
    }
]))
fam_name_mode = fam_name_mode[0]["_id"]

deployment_type_mode = list(enriched_datasets.aggregate([
    {
        '$match': {
            "classification_output": {'$exists': True}
        }
    }, {
        '$group': {
            '_id': "$classification_output",
            'count': {'$sum': 1}
        }
    }, {
        '$sort': {"count": -1}
    }, {
        '$limit': 1
    }
]))
deployment_type_mode = deployment_type_mode[0]["_id"]
if 'single' in deployment_type_mode:
    deployment_type_mode = "single_host"

platform_mode = list(enriched_datasets.aggregate([
    {
        '$match': {
            "platform": {'$exists': True}
        }
    }, {
        '$group': {
            '_id': "$platform",
            'count': {'$sum': 1}
        }
    }, {
        '$sort': {"count": -1}
    }, {
        '$limit': 1
    }
]))
platform_mode = platform_mode[0]["_id"]

nr_hyperparams_mode = list(enriched_datasets.aggregate([
    {
        '$match': {
            "nr_hyperparams": {'$exists': True}
        }
    }, {
        '$group': {
            '_id': "$nr_hyperparams",
            'count': {'$sum': 1}
        }
    }, {
        '$sort': {"count": -1}
    }, {
        '$limit': 1
    }
]))
nr_hyperparams_mode = nr_hyperparams_mode[0]["_id"]

language_mode = list(enriched_datasets.aggregate([
    {
        '$match': {
            "language": {'$exists': True}
        }
    }, {
        '$group': {
            '_id': "$language",
            'count': {'$sum': 1}
        }
    }, {
        '$sort': {"count": -1}
    }, {
        '$limit': 1
    }
]))
language_mode = language_mode[0]["_id"]

use_case = html.Div(
    [
        dbc.Label("Use case",
                  width=7, color="#FFFAF0",
                  style={"text-align": "center", 'justify': 'left', 'font-size': '20px', 'font-weight': 'bold',
                         'width': '100%', "background-color": "transparent", "color": "black"}),
        dbc.Label("Select a use case from the list or type a new one",
                  width=7, color="#FFFAF0",
                  style={"text-align": "left", 'justify': 'left', 'font-size': '15px', 'font-weight': 'bold',
                         'width': '100%', "background-color": "transparent", "color": "black"}),
        dcc.Dropdown(id="use_case",
                     options=[
                         dict({'label': use_case, 'value': use_case}) for use_case in use_case_list
                     ],
                     placeholder="Select use-case name here",
                     style={'width': '100%', 'color': 'black', },
                     clearable=False, ),
    ], )

csv_upload = html.Div(
    children=[
        dbc.Label("Upload dataset as a csv file here",
                  width=7, color="#FFFAF0",
                  style={"text-align": "left", 'justify': 'left', 'font-size': '15px', 'font-weight': 'bold',
                         'width': '100%', "background-color": "transparent", "color": "black"}),
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
            },
            # Allow multiple files to be uploaded
            multiple=False
        ),
        dcc.Loading(
            id="upload_loading",
            type="default",
            children=html.Div(id="output_data_upload")
        ),
        html.Br(),
    ])

class_label = html.Div(
    [
        dbc.Label("Select Label of Target Class",
                  width=7, color="#FFFAF0",
                  style={"text-align": "left", 'justify': 'left', 'font-size': '15px', 'font-weight': 'bold',
                         'width': '100%', "background-color": "transparent", "color": "black"}),
        dcc.Dropdown(id="class_label", placeholder='Select target class label',
                     style={'width': '100%', 'color': 'black'},
                     options=[], ),
        html.Br(),
    ])

class_feature_type = html.Div(
    [
        dbc.Label("Select Datatype of Target Class",
                  width=7, color="#FFFAF0",
                  style={"text-align": "left", 'justify': 'left', 'font-size': '15px', 'font-weight': 'bold',
                         'width': '100%', "background-color": "transparent", "color": "black"}),
        dcc.Dropdown(id="class_feature_type",
                     options=[
                         {'label': 'Binary', 'value': 'binary'},
                         {'label': 'Categorical', 'value': 'categorical'},
                         {'label': 'Numerical', 'value': 'numerical'}
                     ],
                     placeholder="Select a Datatype",
                     style={'width': '100%', 'color': 'black'}),
        html.Br(),
    ])

feature_type_list = html.Div(
    [
        dbc.Label("Enter Feature Annotation List",
                  width=7, color="#FFFAF0",
                  style={"text-align": "left", 'justify': 'left', 'font-size': '15px', 'font-weight': 'bold',
                         'width': '100%', "background-color": "transparent", "color": "black"}),
        dbc.Input(id="feature_type_list", placeholder='Enter datatype of all features as a list', value='', type="text",
                  style={'width': '100%', 'color': 'black'}),
        html.Br(),
    ])

classification_type = html.Div(
    [
        dbc.Label("Output Type",
                  width=7, color="#FFFAF0",
                  style={"text-align": "left", 'justify': 'left', 'font-size': '15px', 'font-weight': 'bold',
                         'width': '100%', "background-color": "transparent", "color": "black"}),
        dcc.Dropdown(id="classification_type",
                     options=[
                         {'label': 'single', 'value': 'single'},
                         {'label': 'probabilities', 'value': 'probs'},
                     ],
                     placeholder="Output Type",
                     style={'width': '100%', 'color': 'black'},
                     value=classification_type_mode, ),
        html.Br(),
    ])

# algorithm_family = html.Div(
#     [
#         dbc.Label("Select Algorithm Family",
#                   width=7, color="#FFFAF0",
#                   style={"text-align": "left", 'justify': 'left', 'font-size': '15px', 'font-weight': 'bold',
#                          'width': '100%', "background-color": "transparent", "color": "black"}),
#         dcc.Dropdown(id="algorithm_family",
#                      options=[
#                          {'label': algo_fam, 'value': algo_fam} for algo_fam in algo_fam_list
#                      ],
#                      placeholder="Select a Algorithm Family",
#                      style={'width': '100%', 'color': 'black'},
#                      value=fam_name_mode),
#         html.Br(),
#     ])
#
# deployment_type = html.Div(
#     [
#         dbc.Label("Select Type of Deployment",
#                   width=7, color="#FFFAF0",
#                   style={"text-align": "left", 'justify': 'left', 'font-size': '15px', 'font-weight': 'bold',
#                          'width': '100%', "background-color": "transparent", "color": "black"}),
#         dcc.Dropdown(id="deployment_type",
#                      options=[
#                          {'label': 'single host', 'value': 'single_host'},
#                          {'label': 'cluster', 'value': 'cluster'},
#                      ],
#                      placeholder="Select a Deployment Type",
#                      style={'width': '100%', 'color': 'black'},
#                      value=deployment_type_mode),
#         html.Br(),
#     ])
#
# platform = html.Div(
#     [
#         dbc.Label("Select Platform",
#                   width=7, color="#FFFAF0",
#                   style={"text-align": "left", 'justify': 'left', 'font-size': '15px', 'font-weight': 'bold',
#                          'width': '100%', "background-color": "transparent", "color": "black"}),
#         dcc.Dropdown(id="platform",
#                      options=[
#                          {'label': platform, 'value': platform} for platform in platform_list
#                      ],
#                      placeholder="Select a Platform",
#                      style={'width': '100%', 'color': 'black'},
#                      value=platform_mode),
#         html.Br(),
#     ])
#
# implementation_lang = html.Div(
#     [
#         dbc.Label("Select Implementation Language Type",
#                   width=7, color="#FFFAF0",
#                   style={"text-align": "left", 'justify': 'left', 'font-size': '15px', 'font-weight': 'bold',
#                          'width': '100%', "background-color": "transparent", "color": "black"}),
#         dcc.Dropdown(id="implementation_lang",
#                      options=[
#                          {'label': 'single language', 'value': 'single_language'},
#                          {'label': 'multi language', 'value': 'multi_language'},
#                      ],
#                      placeholder="Select a Implementation Language Type",
#                      style={'width': '100%', 'color': 'black'},
#                      value='single_language'),
#         html.Br(),
#     ])
#
# language = html.Div(
#     [
#         dbc.Label("Select Implementation Language",
#                   width=7, color="#FFFAF0",
#                   style={"text-align": "left", 'justify': 'left', 'font-size': '15px', 'font-weight': 'bold',
#                          'width': '100%', "background-color": "transparent", "color": "black"}),
#         dcc.Dropdown(id="language",
#                      options=[
#                          {'label': language, 'value': language} for language in language_list
#                      ],
#                      placeholder="Select a Implementation Language",
#                      style={'width': '100%', 'color': 'black'},
#                      value=language_mode),
#         html.Br(),
#     ])
# tuning_limit = html.Div(
#     [
#         dbc.Label("Select Hyperparameters Tuning Range",
#                   width=7, color="#FFFAF0",
#                   style={"text-align": "left", 'justify': 'left', 'font-size': '15px', 'font-weight': 'bold',
#                          'width': '100%', "background-color": "transparent", "color": "black"}),
#         dcc.Slider(
#             id='tuning_slider',
#             min=0,
#             max=int(nr_hyperparams_max),
#             step=1,
#             marks={
#                 0: '0',
#                 round(int(nr_hyperparams_max) / 4): str(round(int(nr_hyperparams_max) / 4)),
#                 round(int(nr_hyperparams_max) / 2): str(round(int(nr_hyperparams_max) / 2)),
#                 round(int(nr_hyperparams_max) * 0.75): str(round(int(nr_hyperparams_max) * 0.75)),
#                 nr_hyperparams_max: str(nr_hyperparams_max)
#             },
#             value=int(nr_hyperparams_mode),
#         ),
#         html.Div(id='tuning_slider_value'),
#         html.Br(),
#
#     ])
#
#
# @app.callback(
#     dash.dependencies.Output('tuning_slider_value', 'children'),
#     [dash.dependencies.Input('tuning_slider', 'value')])
# def update_output(value):
#     return 'Selected Tuning Limit: "{}"'.format(value)
#

accuracy_range = html.Div(
    [
        dbc.Label("Select Accuracy",
                  width=7, color="#FFFAF0",
                  style={"text-align": "left", 'justify': 'left', 'font-size': '15px', 'font-weight': 'bold',
                         'width': '100%', "background-color": "transparent", "color": "black"}),
        dcc.Slider(
            id='accuracy_slider',
            min=0,
            max=1,
            step=0.01,
            marks={
                0: '0',
                0.25: '0.25',
                0.5: '0.5',
                0.75: '0.75',
                1: '1'
            },
            value=0.45,
        ),
        html.Div(id='accuracy_slider_value'),
        html.Br(),
    ])


@app.callback(
    dash.dependencies.Output('accuracy_slider_value', 'children'),
    [dash.dependencies.Input('accuracy_slider', 'value')])
def update_output(value):
    return 'Selected Accuracy: "{}"'.format(value)


precision_range = html.Div(
    [
        dbc.Label("Select Precision",
                  width=7, color="#FFFAF0",
                  style={"text-align": "left", 'justify': 'left', 'font-size': '15px', 'font-weight': 'bold',
                         'width': '100%', "background-color": "transparent", "color": "black"}),
        dcc.Slider(
            id='precision_slider',
            min=0,
            max=1,
            step=0.01,
            marks={
                0: '0',
                0.25: '0.25',
                0.5: '0.5',
                0.75: '0.75',
                1: '1'
            },
            value=0.45,
        ),
        html.Div(id='precision_slider_value'),
        html.Br(),
    ])


@app.callback(
    dash.dependencies.Output('precision_slider_value', 'children'),
    [dash.dependencies.Input('precision_slider', 'value')])
def update_output(value):
    return 'Selected Precision: "{}"'.format(value)


recall_range = html.Div(
    [
        dbc.Label("Select Recall",
                  width=7, color="#FFFAF0",
                  style={"text-align": "left", 'justify': 'left', 'font-size': '15px', 'font-weight': 'bold',
                         'width': '100%', "background-color": "transparent", "color": "black"}),
        dcc.Slider(
            id='recall_slider',
            min=0,
            max=1,
            step=0.01,
            marks={
                0: '0',
                0.25: '0.25',
                0.5: '0.5',
                0.75: '0.75',
                1: '1'
            },
            value=0.45,
        ),
        html.Div(id='recall_slider_value'),
        html.Br(),
    ])


@app.callback(
    dash.dependencies.Output('recall_slider_value', 'children'),
    [dash.dependencies.Input('recall_slider', 'value')])
def update_output(value):
    return 'Selected Recall: "{}"'.format(value)


trtime_range = html.Div(
    [
        dbc.Label("Select Training Time",
                  width=7, color="#FFFAF0",
                  style={"text-align": "left", 'justify': 'left', 'font-size': '15px', 'font-weight': 'bold',
                         'width': '100%', "background-color": "transparent", "color": "black"}),
        dcc.Slider(
            id='trtime_slider',
            min=0,
            max=1,
            step=0.01,
            marks={
                0: '0',
                0.25: '0.25',
                0.5: '0.5',
                0.75: '0.75',
                1: '1'
            },
            value=0.45,
        ),
        html.Div(id='trtime_slider_value'),
        html.Br(),
    ])


@app.callback(
    dash.dependencies.Output('trtime_slider_value', 'children'),
    [dash.dependencies.Input('trtime_slider', 'value')])
def update_output(value):
    return 'Selected Training Time: "{}"'.format(value)


submit_button = html.Div([
    dbc.Button("Analyse Dataset", id="submit_button",
               color="primary", className="mr-1",
               style={"justify": "center", 'block': 'True', 'width': '100%', "background-color": "rgb(176,196,222)",
                      "font-color": "black"},
               ),
    dcc.Loading(
        id="submit_btn_loading",
        type="default",
        children=html.Div(id="submit_btn_load_output", style={"font-weight": "bold", },
                          ),
    )]
)

dataset_characteristics = html.Div([
    dbc.Label("Dataset Charactersitics",
              width=7, color="#FFFAF0",
              style={"text-align": "center", 'justify': 'left', 'font-size': '20px', 'font-weight': 'bold',
                     'width': '100%', "background-color": "transparent", "color": "black"}),
    csv_upload,
    class_label,
    class_feature_type,
    feature_type_list,
])

classifier_preferences = html.Div([
    dbc.Label("Classifier Preferences",
              width=7, color="#FFFAF0",
              style={"text-align": "center", 'justify': 'left', 'font-size': '20px', 'font-weight': 'bold',
                     'width': '100%', "background-color": "transparent", "color": "black"}),
    classification_type,
    # algorithm_family,
    # deployment_type,
    # platform,
    # implementation_lang,
    # language,
    # tuning_limit,
    accuracy_range,
    precision_range,
    recall_range,
    trtime_range,
])

sidebar = html.Div(
    children=[
        html.H5(children='Fill in required details and upload dataset', style={'font-weight': 'bold', }),
        use_case,
        html.Br(),
        html.H5(id='user_usecase'),
        dataset_characteristics,
        html.Br(),
        classifier_preferences,
        submit_button,
    ],
    style={
        "border-radius": "10px",
        "margin": "10px",
        "position": "relative",
        "left": 0,
        "top": 0,
        "bottom": 0,
        "width": '25%',
        "padding": "1rem 1rem",
        "float": "left",
    }
)

content = html.Div(
    children=[
        html.H6(id='api_call_response', ),
        html.H6(id='result_section', children='Analysis results will get displayed here !!!!',
                style={'font-weight': 'bold', }),
        html.H6(id='query_issued_tag', ),
        dbc.Collapse(id='query_issued_value', ),
    ],
    style=CONTENT_STYLE,
)

app.layout = html.Div(
    [
        html.Div([header]),
        html.Div(
            children=[sidebar]
        ),
        html.Div(
            children=[content]
        )
    ],
)

'''nr_features = df.shape[1] - 1
    if(nr_features < 20 ):
        for i in range(1,nr_features):
            return_html.children.append(
                 dbc.Input(id="feature_type_"+str(i),placeholder='Enter datatype of feature '+str(i),value='',type="text", style= {'width': '60%', 'color':'black'}),)
    else:
        return_html.children.append(
                 dbc.Input(id="feature_type_list",placeholder='Enter datatype of all features as a list',value='',type="text", style= {'width': '60%', 'color':'black'}),)
'''


@app.callback(
    Output('user_usecase', 'children'),
    Input('use_case', 'value'),
    prevent_initial_call=True
)
def create_use_case_input(selected_value):
    if ("no selection" in selected_value):
        user_defined_usecase = dbc.Input(placeholder='Enter use-case name here', value='', type="text",
                                         style={'width': '100%', 'color': 'black', 'font-size': '15px'})
        return user_defined_usecase


@app.callback(
    Output('use_case', 'options'),
    Input('result_section', 'children'),
    prevent_initial_call=True
)
def update_usecase_dropdown(submit_btn_clicks):
    use_case_list = collection_datasets.distinct("Info.use_case")
    use_case_list.insert(0, "----no selection----")
    options = [{'label': use_case, 'value': use_case} for use_case in use_case_list]
    return options


@app.callback(Output('output_data_upload', 'children'),
              Output('class_label', 'options'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename')],
              prevent_initial_call=True)
def update_output(list_of_contents, filename):
    content_type, content_string = list_of_contents.split(',')
    decoded = base64.b64decode(content_string)
    upload_dir = os.path.join(working_dir, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    with open(os.path.join(upload_dir, filename), 'w') as csv_file:
        for line in str(decoded.decode('utf-8')).splitlines():
            csv_file.write(line)
            csv_file.write('\n')
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            decoded_content = decoded.decode('utf-8')
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(decoded_content.splitlines()[0])
            df = pd.read_csv(io.StringIO(decoded_content), delimiter=str(dialect.delimiter))
            feature_list = df.columns
            options = [{'label': feature_name, 'value': feature_name} for feature_name in feature_list]
            error_code = upload_dataset_R_backend(decoded)
            # Use error code to print message accordingly
            if error_code == 200:
                return filename + ' is uploaded successfully', options
            else:
                return filename + ' is upload failed with error code ' + error_code, options

        else:
            return 'Invalid input file. Try again by uploading csv file'
    except Exception as e:
        print(e)
        return 'There was an error processing this file.'


@app.callback(
    Output('submit_btn_load_output', 'children'),
    Output('result_section', 'children'),
    Output('api_call_response', 'children'),
    [
        Input('submit_button', 'n_clicks')
    ],
    [
        State('use_case', 'value'),
        State('user_usecase', 'children'),
        State('upload-data', 'contents'),
        State('class_label', 'value'),
        State('class_feature_type', 'value'),
        State('feature_type_list', 'value'),
        State('upload-data', 'filename'),

        State('classification_type', 'value'),
        # State('algorithm_family', 'value'),
        # State('deployment_type', 'value'),
        # State('platform', 'value'),
        # State('implementation_lang', 'value'),
        # State('language', 'value'),
        # State('tuning_slider', 'value'),
        State('accuracy_slider', 'value'),
        State('precision_slider', 'value'),
        State('recall_slider', 'value'),
        State('trtime_slider', 'value'),
    ],
    prevent_initial_call=True
)
def trigger_data_profiler(submit_btn_clicks, use_case, user_use_case, csv_file_contents, class_label,
                          class_feature_type, feature_type_list, csv_filename,
                          classification_type, accuracy_slider, precision_slider, recall_slider, trtime_slider):
    # algorithm_family, deployment_type, platform, implementation_lang,language, tuning_slider,
    mode = 2
    print(type(submit_btn_clicks))
    content_type, content_string = str(csv_file_contents).split(',')
    print(type(content_string))
    if "no selection" in use_case:
        use_case = user_use_case['props']['value']
    data_profiler = DataProfiler(mode, '', content_string, csv_filename, class_label, class_feature_type, use_case,
                                 str(feature_type_list))
    json_output, db_write_status = data_profiler.analyse_dataset()
    if len(json_output) == 0:
        print("Error in execution of data_profiler.py")
        return db_write_status, "Feature suggestion not possible", ""
    print("Execution of data_profiler.py Complete")
    json_output = json.loads(json_output)
    # TODO handle the suggested features here.
    suggested_features = suggest_features_to_user(json_output, class_feature_type)
    suggested_features = construct_output_html(suggested_features)
    response_api_call = api_call_R_backend(class_feature_type, feature_type_list, classification_type, accuracy_slider,
                                           precision_slider, recall_slider, trtime_slider, use_case, csv_filename)
    # algorithm_family,
    # deployment_type, platform,
    # implementation_lang, language, tuning_slider,
    print('Execution result printed to dashboard')
    return db_write_status, suggested_features, response_api_call


@app.callback(
    Output("query_issued_value", "is_open"),
    [Input("query_issued_tag", "n_clicks")],
    [State("query_issued_value", "is_open")],
    prevent_initial_call=True
)
def toggle_accordion(n_clicks, is_open):
    ctx = dash.callback_context
    if not ctx.triggered:
        return False
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "query_issued_tag" and n_clicks:
        print(is_open)
        return not is_open
    else:
        return False


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')
