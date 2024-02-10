import colorlover as cl
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
import base64
import io
import time
from functools import partial
from plotly.tools import mpl_to_plotly
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
import corner

def corner_plot(samples, bins=12):
    SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    fig = corner.corner(np.array(samples),
                        bins=bins,
                        labels=[f"\u03B1{i}".translate(SUB) for i in range(len(samples))], 
                        label_kwargs={"fontsize":20}, 
                        hist_kwargs= {"linewidth":2}, 
                        quantiles=None, 
                        # smooth=(1.7), 
                        # smooth1d=1.0, 
                        show_titles=True)
    plt.text(.6, .75, f"N = {len(samples)}", fontsize=18, transform=plt.gcf().transFigure)
    # Convert the plot to an image
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_string = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()
    return image_string

def line_plot(samples, title_n):
    samples = samples[::100]
    plt.plot(np.arange(len(samples)), np.array(samples))
    plt.xlabel("Iteration", fontsize=20)
    plt.ylabel("Parameter Value", fontsize=20)
    plt.title(f"Parameter {title_n} Convergence", fontsize=20)
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_string = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()
    return image_string

