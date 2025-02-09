from dash import html, dcc
import dash_bootstrap_components as dbc
from pydantic import BaseModel, Field

from common.data import EnrichedModel


class _DefaultClassification(BaseModel):
    type: str = Field(None, alias="_id")
    count: int

async def _retrieve_default_classification_type():
    return await EnrichedModel.aggregate(
        [
            {
                "$match": {
                    "classification_output": {"$exists": True}
                }
            },
            {
                "$group": {
                    "_id": "$classification_output",
                    'count': {'$sum': 1}
                }
            }, {
                "$sort": {"count": -1}
            }, {
                "$limit": 1
            }
        ],
        projection_model=_DefaultClassification
    ).to_list()

async def create_classifier_preferences():
    #default_option = await _retrieve_default_classification_type()
    #default_option_type = default_option[0].type
    default_option_type = None
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
                         value=default_option_type, ),
            html.Br(),
        ])

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

    return html.Div([
    dbc.Label("Classifier Preferences",
              width=7, color="#FFFAF0",
              style={"text-align": "center", 'justify': 'left', 'font-size': '20px', 'font-weight': 'bold',
                     'width': '100%', "background-color": "transparent", "color": "black"}),
    classification_type,
    accuracy_range,
    precision_range,
    recall_range,
    trtime_range,
])