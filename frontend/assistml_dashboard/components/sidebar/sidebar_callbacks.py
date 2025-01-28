from flash import Flash, Input, Output, State
from quart import g

from assistml_dashboard.client import BackendClient
from assistml_dashboard.components.report import create_report_layout, create_suggested_feature_layout
from assistml_dashboard.components.sidebar.classifier_preferences_callbacks import register_classifier_preferences_callbacks
from assistml_dashboard.components.sidebar.dataset_characteristics_callbacks import register_dataset_characteristics_callbacks


def register_sidebar_callbacks(app: Flash):
    backend: BackendClient = g.backend_client

    register_dataset_characteristics_callbacks(app)
    register_classifier_preferences_callbacks(app)

    @app.callback(
        [
            Output('submit_btn_load_output', 'children'),
            Output('result_section', 'children'),
            Output('report_section', 'children'),
        ],
        [
            Input('submit_button', 'n_clicks'),
        ],
        [
            State('parsed-data', 'data'),
            State('class_label', 'value'),
            State('class_feature_type', 'value'),
            State('feature_type_list', 'value'),
            State('upload-data', 'filename'),

            State('classification_type', 'value'),
            State('accuracy_slider', 'value'),
            State('precision_slider', 'value'),
            State('recall_slider', 'value'),
            State('trtime_slider', 'value'),
        ],
        prevent_initial_call=True
    )
    async def trigger_data_profiler(submit_btn_clicks, serialized_dataframe, class_label, class_feature_type,
                              feature_type_list, csv_filename, classification_type, accuracy_slider, precision_slider,
                              recall_slider, trtime_slider):
        print(type(submit_btn_clicks))
        response, error = await backend.analyse_dataset(class_label, class_feature_type, feature_type_list)
        if response is None:
            return error, "Feature suggestion not possible", ""

        if response.data_profile is None:
            return response.db_write_status, "Feature suggestion not possible", ""

        suggested_features = create_suggested_feature_layout(response.data_profile, class_feature_type)
        report, error = await backend.report(class_feature_type, feature_type_list, classification_type, accuracy_slider,
                                       precision_slider, recall_slider, trtime_slider, csv_filename)

        if report is None:
            return response.db_write_status, suggested_features, "Error in execution of report.py"

        report_layout = await create_report_layout(report, error)

        return response.db_write_status, suggested_features, report_layout
