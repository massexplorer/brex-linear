from dash import dcc
from dash import html
# import dash_bootstrap_components as dbc
import utils.dash_reusable_components as drc
import colorlover as cl
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
import pandas as pd
from functools import partial
from plotly.tools import mpl_to_plotly
from dash import dash_table

def view():

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


    return html.Div(id="app-container", children=[
        html.Div(id="left-column", children=[
            html.Div(id="div-btns", children=[
                html.Button(id='btn', children='Fit', n_clicks=0),
                html.Button(id='btn_clear', children='Clear', n_clicks=0),
            ]),
            dash_table.DataTable(
                data = [{'x': None, 'y': None, 'dy': None}],
                columns = [{"name": 'x', "id": 'x'}, {"name": 'y', "id": 'y'}, {"name": '\u03B4y', "id": 'dy'}], 
                id='tbl', editable=True, row_deletable=True, 
                style_data_conditional=[{'if': {'state': 'active'}, 'backgroundColor': 'rgba(0, 0, 0, 0)', 'border': '1px solid #e76f51', 'color': 'black'}],
            ),
        ]),
        html.Div(id='center-column', children=[
            # dcc.Loading(children=[
            #     html.Div(id="div-graph", children=[
            dcc.Graph(id="graph", figure=fig, config={'scrollZoom': False, 'responsive': False, 
                                                      'autosizable': False, 'displayModeBar': False, 'displaylogo': False, 'staticPlot': False, 'doubleClick': False, 
                                                      'showTips': False, 'showAxisDragHandles': False, 'showAxisRangeEntryBoxes': False, 'showLink': False,})
            # 'displayModeBar': False,
            #     ])
            # ])
        ]),
        html.Div(id="right-column", children=[
            html.Div(id="div-graphs"),
            html.Div(id="div-lineplot"),
        ]),
    ]),                 
  


