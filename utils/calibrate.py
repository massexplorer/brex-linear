import numpy as np
import plotly.graph_objs as go
import pymc as pm
import arviz as az

def linear(x, m, b):
    return m * x + b

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

        if cum_trace is None:
            cum_trace = trace
        else:
            cum_trace = az.concat([cum_trace, trace], dim='chain')

        return cum_trace


def calibrate(x, y, dy, cum_trace=None, batch_size=100):
    x_func = np.arange(0, 161)

    #Doing the Metropolis sampling for 1000 values
    cum_trace = metro_sample(x, y, dy, cum_trace=cum_trace, batch_size=batch_size)
    m = np.array(cum_trace['posterior']['slope']).flatten()
    b = np.array(cum_trace['posterior']['intercept']).flatten()
    
    func_rand = linear(x_func, m, b)
    lower = np.percentile(func_rand, 2.5, axis = 0)
    median = np.percentile(func_rand, 50, axis = 0)
    upper = np.percentile(func_rand, 97.5, axis = 0)
    
    return [x_func, lower, median, upper], cum_trace
