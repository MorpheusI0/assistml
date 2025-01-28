from flash import Flash, Input, Output


def register_classifier_preferences_callbacks(app: Flash):
    @app.callback(
        Output('accuracy_slider_value', 'children'),
        [Input('accuracy_slider', 'value')])
    async def update_output(value):
        return 'Selected Accuracy: "{}"'.format(value)

    @app.callback(
        Output('precision_slider_value', 'children'),
        [Input('precision_slider', 'value')])
    async def update_output(value):
        return 'Selected Precision: "{}"'.format(value)

    @app.callback(
        Output('recall_slider_value', 'children'),
        [Input('recall_slider', 'value')])
    async def update_output(value):
        return 'Selected Recall: "{}"'.format(value)

    @app.callback(
        Output('trtime_slider_value', 'children'),
        [Input('trtime_slider', 'value')])
    async def update_output(value):
        return 'Selected Training Time: "{}"'.format(value)
