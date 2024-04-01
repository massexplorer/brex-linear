import plotly.graph_objs as go
import plotly.express as px
import numpy as np
import base64
import io
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
# import corner
import pymc as pm
import arviz as az

def pair_plot(trace):
    plt.rcParams.update({'text.color' : "#a5b1cd", 'axes.labelcolor' : "#a5b1cd",
                         'xtick.color': "#a5b1cd", 'ytick.color': "#a5b1cd",})
    figure = plt.figure(facecolor='#282b38')
    az.plot_pair(
    trace,
    var_names=['intercept', 'slope'],
    kind="hexbin",
    marginals=True,
    figsize=(10, 10),
    gridsize=35,)

    # Set the background color of all axes
    axes = plt.gcf().get_axes()
    for ax in axes:
        ax.set_facecolor('#282b38')

    plt.text(.6, .75, f"N = {len(trace['posterior']['slope'])}", fontsize=18,
             transform=plt.gcf().transFigure)
    # Convert the plot to an image
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_string = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()
    return image_string
        

def add_fit(fig, x_func, median, upper, lower):
    line_trace = go.Scatter(
        x=x_func, y=median,
        line=dict(color='#e9c46a'),
        mode='lines', hoverinfo="skip",
    )
    upper_trace = go.Scatter(
        name='Upper Bound',
        x=x_func, y=upper,
        mode='lines',
        marker=dict(color="#444"),
        line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    )
    lower_trace = go.Scatter(
        name='Lower Bound',
        x=x_func, y=lower,
        marker=dict(color="#444"),
        line=dict(width=0),
        mode='lines',
        fillcolor='rgba(233, 197, 106, 0.15)',
        fill='tonexty',
        showlegend=False, hoverinfo="skip",
    )

    fig['data'] = fig['data'][:2]
    fig['data'].append(line_trace)
    fig['data'].append(upper_trace)
    fig['data'].append(lower_trace)
    return fig

def reset_axes(fig):
    fig['layout']['xaxis'] = dict(title='x', range=[0,160], fixedrange=True,
                                  gridcolor='#888888', gridwidth=1, showline=True, mirror=True)
    fig['layout']['yaxis'] = dict(title='y', range=[0,130], fixedrange=True,
                                  gridcolor='#888888', gridwidth=1, showline=True, mirror=True)
    return fig

def main_plot():
    fig = px.imshow(np.zeros(shape=(130, 160, 4)), origin='lower')
    fig.add_trace(
        go.Scatter(x=[], y=[], marker=dict(color='#e76f51', size=10), name='',
                   error_y = dict(type='data', symmetric=True, array=[], 
                   color='#e76f51', thickness=1, width=2,), mode='markers'),
    )
    fig = reset_axes(fig)
    fig.update_layout( 
        margin=dict(r=20, t=10, l=25, b=20),
        showlegend=False, plot_bgcolor='#282b38', paper_bgcolor='#282b38',
        font=dict(color='#a5b1cd'),
    )
    fig['data'][0]['hovertemplate'] = 'x: %{x}<br>y: %{y}'
    fig['data'][0]['name'] = ''
    return fig

# def corner_plot(samples, bins=12):
#     SUB = str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉')
#     plt.rcParams.update({'text.color' : "#a5b1cd", 'axes.labelcolor' : "#a5b1cd",
#                         'xtick.color': "#a5b1cd", 'ytick.color': "#a5b1cd",})
#     figure = plt.figure(facecolor='#282b38')
#     corner.corner(np.array(samples), bins=bins, quantiles=None,
#                   labels=[f"\u03B1{i}".translate(SUB) for i in range(len(samples))], 
#                         label_kwargs={"fontsize":20}, hist_kwargs= {"linewidth":2, "color":'#e76f51'}, 
#                         smooth=(1.7), smooth1d=1.0, 
#                         color='#e76f51',
#                         show_titles=True, facecolor='#282b38', title_kwargs={"fontsize":20},
#                         fig=figure, hist2d_kwargs={"color":'#e76f51'},)


#     # Set the background color of all axes
#     axes = plt.gcf().get_axes()
#     for ax in axes:
#         ax.set_facecolor('#282b38')

#     plt.text(.6, .75, f"N = {len(samples)}", fontsize=18,
#              transform=plt.gcf().transFigure)
#     # Convert the plot to an image
#     buffer = io.BytesIO()
#     plt.savefig(buffer, format='png')
#     buffer.seek(0)
#     image_string = base64.b64encode(buffer.read()).decode('utf-8')
#     plt.close()
#     return image_string

def line_plot(samples, title_n):
    # samples = samples[::2]
    plt.rcParams.update({'text.color' : "#a5b1cd", 'axes.labelcolor' : "#a5b1cd",
                        'xtick.color': "#a5b1cd", 'ytick.color': "#a5b1cd",})
    plt.figure(facecolor='#282b38')
    plt.plot(np.arange(len(samples)), np.array(samples), color='#e76f51', 
             linewidth=2)
    ax = plt.gca()
    ax.set_facecolor('#282b38')
    # set frame color
    for spine in plt.gca().spines.values():
        spine.set_edgecolor('#282b38')
    plt.xlabel("Iteration", fontsize=20)
    plt.ylabel("Parameter Value", fontsize=20)
    plt.title(f"Parameter {title_n} Convergence", fontsize=20)
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_string = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()
    return image_string

