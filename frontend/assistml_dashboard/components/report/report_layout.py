from typing import List

import pandas as pd
from dash import html, dcc
import plotly.express as px

from assistml_dashboard.components.report.model_report_table_layout import create_model_report_table_layout
from common.data.enriched_model import EnrichedModel
from common.data.query import Summary, ModelReport
from common.dto import ReportResponseDto


def _generate_summary(summary: Summary):
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

async def _generate_plotting_response(acceptable_models: List[ModelReport], nearly_acceptable_models: List[ModelReport]):
    training_time_std = []
    recall = []
    accuracy = []
    acceptable_or_nearly = []
    models_names = []

    for acceptable_model in acceptable_models:
        model_name = acceptable_model["code"]
        model = await EnrichedModel.find_one(EnrichedModel.model_name == model_name)  # TODO: Use new data structure
        training_time_std.append(model.training_time_std)
        recall.append(model.recall)
        accuracy.append(model.accuracy)
        acceptable_or_nearly.append("Acceptable")
        models_names.append(model.model_name)
    for nearly_acceptable_model in nearly_acceptable_models:
        model_name = nearly_acceptable_model["code"]
        model = await EnrichedModel.find_one(EnrichedModel.model_name == model_name)  # TODO: Use new data structure
        training_time_std.append(model.training_time_std)
        recall.append(model.recall)
        accuracy.append(model.accuracy)
        acceptable_or_nearly.append("Nearly Acceptable")
        models_names.append(model.model_name)

    dictionary = {'Model Name': models_names, 'Recommended Model Type': acceptable_or_nearly,
                  'Training Time': training_time_std, 'Recall': recall, 'Accuracy': accuracy}
    df = pd.DataFrame(dictionary)
    fig = px.scatter_3d(df, text="Model Name", x='Training Time', y='Recall', z='Accuracy',
                        color='Recommended Model Type', color_discrete_sequence=["green", "orange"])

    fig.update_layout(margin=dict(l=5, r=5, t=5, b=5))
    return dcc.Graph(id='response_plot', figure=fig)

def _generate_table(acceptable_models: [], nearly_acceptable_models):
    return html.Div([
        html.Br(),
        html.H5(children='Acceptable Models', style={'font-weight': 'bold', }),
        html.Div([create_model_report_table_layout(acceptable_model) for acceptable_model in acceptable_models],
                 style={'display': 'inline-block'}),
        html.Br(),
        html.H5(children='Nearly Acceptable Models', style={'font-weight': 'bold', }),
        html.Div([create_model_report_table_layout(nearly_acceptable_model) for nearly_acceptable_model in
                  nearly_acceptable_models], style={'display': 'inline-block'}),
    ],
        style={'width': '90%', 'display': 'inline-block'})

async def create_report_layout(report: ReportResponseDto, response):
    if report is None:
        return html.Div([
                html.H6(
                    children='Execution of Remote R backend terminated with status code ' + str(response.status_code),
                    style={'font-weight': 'bold', }),
            ])

    return html.Div([
        _generate_summary(report.summary),
        await _generate_plotting_response(report.acceptable_models, report.nearly_acceptable_models),
        _generate_table(report.acceptable_models, report.nearly_acceptable_models)
    ])
