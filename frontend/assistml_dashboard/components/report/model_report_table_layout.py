import re

from dash import html

from common.data.query import ModelReport


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

def create_model_report_table_layout(model_report: ModelReport):
    if not 'rules' in model_report:
        rules = "No notes are provided for this solution"
    else:
        rules = model_report["rules"]
    place_holder = "There should be some text here! "
    deploy_text = "Deployed in " + str(model_report["deployment"]) + " with " + str(
        model_report["cores"]) + " cores and power: " + str(model_report["power"]) + " GhZ"
    model_code = re.search('([A-Z]{3})_[a-zA-Z0-9]+_\w+', model_report["code"])
    model_code = model_code[1]

    return html.Table(
        [
            html.Tr(
                [
                    html.Td(model_code, rowSpan='9',
                            style={'border-style': 'solid', 'border-width': '1px', 'textAlign': 'center',
                                   'color': 'black'}),
                    html.Td(model_report["name"], colSpan='8',
                            style={'border-style': 'solid', 'border-width': '2px', 'textAlign': 'center',
                                   'font-weight': 'bold', 'backgroundColor': '#8DAD26', 'color': 'black'})
                ]
            ),
            html.Tr(
                [
                    html.Td('Accuracy', rowSpan='1', colSpan='1',
                            style={'border-style': 'solid', 'border-width': '1px', 'font-weight': 'bold',
                                   'color': 'black', }),
                    html.Td(model_report["performance"]["accuracy"], rowSpan='1', colSpan='1',
                            style={'border-style': 'solid', 'border-width': '1px', 'color': 'black',
                                   'backgroundColor': background_color(model_report["performance"]["accuracy"]), }),
                    html.Td('Precision', rowSpan='1', colSpan='1',
                            style={'border-style': 'solid', 'border-width': '1px', 'font-weight': 'bold',
                                   'color': 'black', }),
                    html.Td(model_report["performance"]["precision"], rowSpan='1', colSpan='1',
                            style={'border-style': 'solid', 'border-width': '1px', 'color': 'black',
                                   'backgroundColor': background_color(
                                       model_report["performance"]["precision"]), }),
                    html.Td('Recall', rowSpan='1', colSpan='1',
                            style={'border-style': 'solid', 'border-width': '1px', 'font-weight': 'bold',
                                   'color': 'black', }),
                    html.Td(model_report["performance"]["recall"], rowSpan='1', colSpan='1',
                            style={'border-style': 'solid', 'border-width': '1px', 'color': 'black',
                                   'backgroundColor': background_color(model_report["performance"]["recall"]), }),
                    html.Td('Training Time', rowSpan='1', colSpan='1',
                            style={'border-style': 'solid', 'border-width': '1px', 'color': 'black', }),
                    html.Td(model_report["performance"]["training_time"], rowSpan='1', colSpan='1',
                            style={'border-style': 'solid', 'border-width': '1px', 'font-weight': 'bold',
                                   'color': 'black', 'backgroundColor': background_color(
                                    model_report["performance"]["training_time"]), }),
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
                    html.Td(model_report["out_analysis"], rowSpan='1', colSpan='8',
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
                    html.Td(model_report["preprocessing"], rowSpan='1', colSpan='8',
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
                    html.Td('(overall) Score: ' + str(model_report["overall_score"]), rowSpan='1', colSpan='1',
                            style={'border-style': 'solid', 'border-width': '1px',
                                   'backgroundColor': background_color(model_report["overall_score"]), }),
                    html.Td(deploy_text, rowSpan='1', colSpan='8',
                            style={'border-style': 'solid', 'border-width': '1px', }),
                ]
            ),
            html.Tr(
                [
                    html.Td(model_report["code"], rowSpan='1', colSpan='1',
                            style={'border-style': 'solid', 'border-width': '1px'}),
                    html.Td('Language', rowSpan='1', colSpan='1',
                            style={'border-style': 'solid', 'border-width': '1px', 'font-weight': 'bold', }),
                    html.Td(model_report["language"], rowSpan='1', colSpan='1',
                            style={'border-style': 'solid', 'border-width': '1px'}),
                    html.Td('Implementation', rowSpan='1', colSpan='1',
                            style={'border-style': 'solid', 'border-width': '1px', 'font-weight': 'bold', }),
                    html.Td(model_report["platform"], rowSpan='1', colSpan='1',
                            style={'border-style': 'solid', 'border-width': '1px'}),
                    html.Td('Nr Dependencies', rowSpan='1', colSpan='1',
                            style={'border-style': 'solid', 'border-width': '1px', 'font-weight': 'bold', }),
                    html.Td(model_report["nr_dependencies"], rowSpan='1', colSpan='1',
                            style={'border-style': 'solid', 'border-width': '1px'}),
                    html.Td('Nr Parameters', rowSpan='1', colSpan='1',
                            style={'border-style': 'solid', 'border-width': '1px', 'font-weight': 'bold', }),
                    html.Td(model_report["nr_hparams"], rowSpan='1', colSpan='1',
                            style={'border-style': 'solid', 'border-width': '1px', 'color': 'black', }),
                ]
                , style={'border-style': 'solid', 'border-width': '2px', }
            ),
        ],
        style={'border-collapse': 'collapse', 'border-spacing': '0.1', 'border-width': '2px', 'color': 'black', },
    )
