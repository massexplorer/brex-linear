from dash import dcc
from dash import html
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
from dash import dash_table
import utils.figures as figs

def view():

    fig = figs.main_plot()


    return html.Div(id="app-container", children=[
        html.Div(id="left-column", children=[
            html.Div(id="div-btns", children=[
                html.Button(id='btn', children='Fit', n_clicks=0),
                html.Button(id='btn_clear', children='Clear', n_clicks=0),
            ]),
            dash_table.DataTable(
                data = [{'x': None, 'y': None, 'dy': None}],
                columns = [{"name": 'x', "id": 'x'}, {"name": 'y', "id": 'y'}, 
                           {"name": '\u03B4y', "id": 'dy'}], 
                id='tbl', editable=True, row_deletable=True, 
                style_data_conditional=[{'if': {'state': 'active'}, 
                                        #  'backgroundColor': 'rgba(0, 0, 0, 0)', 
                                         'border': '1px solid #e76f51', }],
                # style_data={
                #     'color': '#a5b1cd',
                #     'backgroundColor': '#282b38',
                # },
                style_header={
                    'fontWeight': 'bold',
                },
                style_cell={
                    'backgroundColor': '#282b38',
                    'color': '#a5b1cd',
                    'border': '1px solid #a5b1cd',
                },
            ),
        ]),
        html.Div(id='center-column', children=[
            dcc.Graph(id="graph", figure=fig, 
                      config={'scrollZoom': False, 'responsive': False, 
                              'autosizable': False, 'displayModeBar': False, 
                              'displaylogo': False, 'staticPlot': False, 
                              'doubleClick': False, 'showTips': False, 
                              'showAxisDragHandles': False, 
                              'showAxisRangeEntryBoxes': False, 
                              'showLink': False,})
        ]),
        html.Div(id="right-column", children=[
            html.Div(id="div-corner"),
            html.Div(id="div-lineplot"),
        ]),
    ]),                 
  


