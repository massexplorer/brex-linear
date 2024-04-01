import numpy as np
import plotly.graph_objs as go
import pymc as pm
import arviz as az
import logging
logger = logging.getLogger("pymc3")
logger.setLevel(logging.ERROR)

def metro_sample(x, y_obs, y_err, cum_trace=None, n_samples=10000, batch_size=100):
    with pm.Model() as linear_model:
        intercept = pm.Normal('intercept', mu=0, sigma=10)
        slope = pm.Normal('slope', mu=0, sigma=10)
        mu = intercept + slope * x
        y = pm.Normal('y_obs', mu=mu, sigma=y_err, observed=y_obs)
        step = pm.Metropolis([intercept, slope])
        
        if cum_trace is not None:
            n = len(np.array(cum_trace['posterior']['slope']).flatten())
            if n >= n_samples:
                return 'stop'
            
        trace = pm.sample(draws=batch_size, step=step, trace=None, chains=1, 
                          discard_tuned_samples=False)

        # if cum_trace is None:
        #     cum_trace = trace
        # else:
        #     cum_trace = az.concat([cum_trace, trace], dim='chain')

        # round all floats to 2 decimal places
        # return cum_trace
        return trace


def calibrate(x, y, dy, cum_trace=None, batch_size=100):
    x_func = np.arange(0, 161)

    #Doing the Metropolis sampling for 100 values
    cum_trace = metro_sample(x, y, dy, cum_trace=cum_trace, batch_size=batch_size)
    m = np.array(cum_trace['posterior']['slope']).flatten()
    b = np.array(cum_trace['posterior']['intercept']).flatten()
    
    y_func = x_func[:, np.newaxis] * m[np.newaxis, :] + b[np.newaxis, :]

    lower = np.percentile(y_func, 2.5, axis = 1)
    median = np.percentile(y_func, 50, axis = 1)
    upper = np.percentile(y_func, 97.5, axis = 1)
    
    return [x_func, lower, median, upper], cum_trace
