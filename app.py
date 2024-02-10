import time
import pandas as pd
import random as rand
import numpy as np
import base64
from datetime import datetime, date

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import json
# import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

import utils.dash_reusable_components as drc
import utils.figures as figs
from utils.views import *
from utils.calibrate import *

app = dash.Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
    prevent_initial_callbacks='initial_duplicate',
)
app.config.suppress_callback_exceptions=True
app.title = "BREX Linear"

app.layout = html.Div(children=[
    dcc.Location(id='url', refresh=False),
    html.Div(className="banner", children=[
        html.Div(className="container scalable", children=[
            html.H2(id="banner-title", children=[
                html.A(id="banner-logo", href="https://bmex.dev", children=[
                    html.Img(src=app.get_asset_url("BMEX-logo-3.png"))
                ],),
            ],),
        ],),
    ],),
    *view(),
    dcc.Store(id="samples-memory", data={'params': [], 'last posterior': None}),
    dcc.Interval(id='interval', interval=2000, n_intervals=0, max_intervals=10, disabled=True),
])

def update_main_fig(figure, count):
    figure['layout']['title'] = f'n={count}'
    figure['layout']['xaxis'] = dict(title='x', range=[0, 160], fixedrange=True)
    figure['layout']['yaxis'] = dict(title='y', range=[0, 130], fixedrange=True)


@app.callback(
    [
        Output('interval', 'disabled', allow_duplicate=True),
        Output('interval', 'n_intervals', allow_duplicate=True),
    ],
    Input('btn', 'n_clicks'),
)
def disable_interval(n_clicks):
    if n_clicks == None or n_clicks == 0:
        print("disabled")
        return [True, 0]
    print("enabled")
    return [False, 0]

@app.callback(
    [
        Output('graph', 'figure', allow_duplicate=True),
        Output("div-graphs", "children", allow_duplicate=True),
        Output('samples-memory', 'data', allow_duplicate=True),
        Output('div-lineplot', 'children', allow_duplicate=True),
    ],
    Input('interval', 'n_intervals'),
    State('graph', 'figure'),
    State('tbl', 'derived_virtual_data'),
    State('samples-memory', 'data'),

)
def fit(n_intervals, figure, table_data, samples):
    start_time = time.time()
    if table_data is None or len(table_data) == 0 or table_data[0]['x'] is None:
        raise PreventUpdate
    x = np.array(figure['data'][1]['x'], dtype=int)
    y = np.array(figure['data'][1]['y'], dtype=int)
    dy = np.array(figure['data'][1]['error_y']['array'], dtype=int)

    if samples['last posterior'] is None:
        resume = None
    else:
        resume = (samples['params'][-1], samples['last posterior'])

    print('resume', resume)
    graph_points, params, posteriors = calibrate(x, y, dy, resume=resume)
    x_func, lower, median, upper = graph_points

    line_trace = go.Scatter(
        x=x_func,
        y=median,
        line=dict(color='#274653'),
        mode='lines', hoverinfo="skip",
    )
    upper_trace = go.Scatter(
        name='Upper Bound',
        x=x_func,
        y=upper,
        mode='lines',
        marker=dict(color="#444"),
        line=dict(width=0),
        showlegend=False
    )
    lower_trace = go.Scatter(
        name='Lower Bound',
        x=x_func,
        y=lower,
        marker=dict(color="#444"),
        line=dict(width=0),
        mode='lines',
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty',
        showlegend=False
    )

    figure['data'] = figure['data'][:2]
    figure['data'].append(line_trace)
    figure['data'].append(upper_trace)
    figure['data'].append(lower_trace)
    figure['layout']['xaxis'] = dict(title='x', range=[0,160], fixedrange=True)
    figure['layout']['yaxis'] = dict(title='y', range=[0,130], fixedrange=True)

    image_string = figs.corner_plot(params)

    old_params = samples['params']

    if len(old_params) > 100_000:
        old_params = old_params[20_000:]
    if len(old_params) > 0:
        params = np.array(list(old_params) + list(params))

    line_plot_string = figs.line_plot(params[:,0], "0")
    line_plot_string2 = figs.line_plot(params[:,1], "1")

    return [
        figure,
        [html.P(f"Time to generate: {time.time() - start_time:.2f} seconds"),
        html.Img(src='data:image/png;base64,{}'.format(image_string), id="corner_plot")],      
        {'params': params, 'last posterior': posteriors[-1]},
        [html.Img(src='data:image/png;base64,{}'.format(line_plot_string), id="line_plot"),
         html.Img(src='data:image/png;base64,{}'.format(line_plot_string2), id="line_plot2"),]
    ]
    

# OLD
# @app.callback(
#     Output('graph', 'figure', allow_duplicate=True),
#     Output("div-graphs", "children", allow_duplicate=True),
#     Input('btn', 'n_clicks'),
#     State('graph', 'figure'),
# )
# def fit(n_clicks, figure):
#     start_time = time.time()

#     if n_clicks == None or n_clicks == 0:
#         raise PreventUpdate
#     x = np.array(figure['data'][1]['x'], dtype=int)
#     y = np.array(figure['data'][1]['y'], dtype=int)
#     dy = np.array(figure['data'][1]['error_y']['array'], dtype=int)
#     prior = [0, 1]
#     if len(figure['data']) > 2:
#         b = figure['data'][2]['y'][0]
#         m = (figure['data'][2]['y'][100]-b)/100
#         prior = [b, m]
#     line_trace, upper_trace, lower_trace, alpha_rand = cali_plot(x, y, dy, prior)
#     figure['data'] = figure['data'][:2]
#     figure['data'].append(line_trace)
#     figure['data'].append(upper_trace)
#     figure['data'].append(lower_trace)
#     figure['layout']['xaxis'] = dict(title='x', range=[0,160], fixedrange=True)
#     figure['layout']['yaxis'] = dict(title='y', range=[0,130], fixedrange=True)

#     image_string = figs.corner_plot(alpha_rand)
#     return figure,\
#         [html.Img(src='data:image/png;base64,{}'.format(image_string), id="corner_plot"), html.P(f"Time to generate: {time.time() - start_time:.2f} seconds")]


# KINDA OLD
# @app.callback(
#     Output('interval', 'disabled'),
#     Input('btn', 'n_clicks'),
# )
# def disable_interval(n_clicks):
#     if n_clicks == None or n_clicks == 0:
#         print("disabled")
#         return True
#     print("enabled")
#     return False

# @app.callback(
#     Output('graph', 'figure', allow_duplicate=True),
#     Output("div-graphs", "children", allow_duplicate=True),
#     Input('interval', 'n_intervals'),
#     State('graph', 'figure'),
#     State('tbl', 'derived_virtual_data'),
# )
# def fit(n_intervals, figure, table_data):
#     start_time = time.time()
#     if table_data == None or len(table_data) == 0:
#         raise PreventUpdate
#     x = np.array(figure['data'][1]['x'], dtype=int)
#     y = np.array(figure['data'][1]['y'], dtype=int)
#     dy = np.array(figure['data'][1]['error_y']['array'], dtype=int)
#     prior = [0, 1]
#     if len(figure['data']) > 2:
#         b = figure['data'][2]['y'][0]
#         m = (figure['data'][2]['y'][100]-b)/100
#         prior = [b, m]
#     line_trace, upper_trace, lower_trace, alpha_rand = cali_plot(x, y, dy, prior)
#     figure['data'] = figure['data'][:2]
#     figure['data'].append(line_trace)
#     figure['data'].append(upper_trace)
#     figure['data'].append(lower_trace)
#     figure['layout']['xaxis'] = dict(title='x', range=[0,160], fixedrange=True)
#     figure['layout']['yaxis'] = dict(title='y', range=[0,130], fixedrange=True)

#     image_string = figs.corner_plot(alpha_rand)
#     return figure,\
#         [html.Img(src='data:image/png;base64,{}'.format(image_string), id="corner_plot"), html.P(f"Time to generate: {time.time() - start_time:.2f} seconds")]







# @app.callback(
    
#     Output("samples-memory", "data"),
#     [
#         State("sample-input", "value"),
#         State("model-dropdown", "value"),
#         State("seed-input", "value"),
#         State("bin-input", "value"),
#         Input("sample-button", "n_clicks"),
#     ]
# )
# def main_output(n_samples, model, seed, bins, n_clicks):
#     start_time = time.time()
#     if n_clicks == None:
#         raise PreventUpdate
#     if n_samples == None:
#         n_samples = 10000
#     samples_df = sample.generate_samples(n_samples, model, seed)
#     image_string = figs.corner_plot(samples_df, bins)
#     return \
#         [html.Img(src='data:image/png;base64,{}'.format(image_string), id="corner_plot"), html.P(f"Time to generate: {time.time() - start_time:.2f} seconds")], \
#         json.dumps(samples_df.to_dict(orient="records"))


@app.callback(
    [
        Output('graph', 'figure', allow_duplicate=True),
        Output('tbl', 'data', allow_duplicate=True),
        Output("div-graphs", "children", allow_duplicate=True),
        Output('samples-memory', 'data', allow_duplicate=True),
        Output('interval', 'disabled', allow_duplicate=True),
    ],
    Input('btn_clear', 'n_clicks'),
)
def clear(n_clicks):
    if n_clicks == None or n_clicks == 0:
        raise PreventUpdate
    fig = px.imshow(np.zeros(shape=(130, 160, 4)), origin='lower')
    fig.add_trace(go.Scatter(x=[], y=[], marker=dict(color='#e76f51', size=10), name='',
                             error_y = dict(type='data', symmetric=True, array=[], color='#e76f51', thickness=1, width=2,), mode='markers'),
                             )
    fig.update_layout( 
        margin=dict(r=20, t=10, l=25, b=20),
        xaxis=dict(title='x', range=[0,160], fixedrange=True),
        yaxis=dict(title='y', range=[0,130], fixedrange=True),
        showlegend=False,
    )
    fig['data'][0]['hovertemplate'] = 'x: %{x}<br>y: %{y}'
    fig['data'][0]['name'] = ''

    return [
        fig, 
        [{'x': None, 'y': None, 'dy': None}], 
        None,
        {'params': [], 'last posterior': None},
        True,
    ]

@app.callback(
    [
        Output('graph', 'figure', allow_duplicate=True),
        Output('tbl', 'data', allow_duplicate=True),
    ],
    [    
        Input('graph', 'clickData'),
        Input('tbl', 'derived_virtual_data'),
    ],
    State('graph', 'figure'),
)
def update(click_data, derived_virtual_data, figure):
    triggered_id = dash.callback_context.triggered_id
    if triggered_id == 'tbl':
        if derived_virtual_data == None or len(derived_virtual_data) == 0:
            # If no data is available, return an empty row
            return [figure, {'x': None, 'y': None, 'dy': None}]
        
        # Plot points in table onto graph
        plottable_table = [row for row in derived_virtual_data if all(row[col] is not None for col in row)] # Remove empty rows
        x_tbl = [row['x'] for row in plottable_table]
        y_tbl = [row['y'] for row in plottable_table]
        dy_tbl = [row['dy'] for row in plottable_table]
        figure['data'][1]['x'] = x_tbl
        figure['data'][1]['y'] = y_tbl
        figure['data'][1]['error_y']['array'] = dy_tbl

        # Check if the last row is not empty, then add a new empty row
        last_row = derived_virtual_data[-1]
        if any(last_row.values()):
            derived_virtual_data.append({'x': None, 'y': None, 'dy': None})
        # figure['layout']['xaxis']['range'] = [0,160]
        # figure['layout']['yaxis']['range'] = [0,130]
        return [figure, derived_virtual_data]
    # else:
        # if derived_virtual_data is None or len(derived_virtual_data) == 0:
        #     return [{'x': None, 'y': None, 'dy': None}]
    if click_data is not None and 'points' in click_data:
        x_click = click_data['points'][0]['x']
        y_click = click_data['points'][0]['y']
        existing_data = figure['data'][1]

        # Update the table data with the clicked point
        derived_virtual_data[-1]['x'] = x_click
        derived_virtual_data[-1]['y'] = y_click
        derived_virtual_data[-1]['dy'] = 10 # Reset uncertainty for the new point

        # Add a new empty row
        derived_virtual_data.append({'x': None, 'y': None, 'dy': None})
        figure['layout']['xaxis'] = dict(title='x', range=[0,160], fixedrange=True)
        figure['layout']['yaxis'] = dict(title='y', range=[0,130], fixedrange=True)

    return [figure, derived_virtual_data]

# Running the server
if __name__ == "__main__":
    app.run_server(debug=True, dev_tools_hot_reload=True, use_reloader=True)


   