import numpy as np
import plotly.graph_objs as go


def linear(x,params):
    return params[0]*10+params[1]*x


def prior_linear(params_vals,arguments):
    params0,params0_Cov_Inv_matrix=arguments
    mu=np.array(params_vals)-np.array(params0)
    params_size=len(params_vals)
    return (2*np.pi)**(-params_size/2)*np.linalg.det(params0_Cov_Inv_matrix)**(-1)*np.exp(-np.dot(mu,np.dot(params0_Cov_Inv_matrix,mu))/2)


def likelihood(params,arguments):
#Assumed format for data=[xvals,yvals]
    data, model, sigmas = arguments
    llh_log_val=0

    for i in range(len(data[0])):
        llh_log_val+=-0.5*((data[1][i]-model(data[0][i],params))/sigmas[i])**2\
        -np.log(2*np.pi*sigmas[i]**2)/2
      
    return np.exp(llh_log_val)


def metropolis(data, sigma, prior_arguments, n, step_size, 
               resume=None, model=linear, prior=prior_linear, likelihood=likelihood):
    # step_size should be a list the size of the parameters of the model
    likelihood_arguments = [data, model, sigma]
    cov_step_size = np.diag(step_size)**2
    
    if resume is None:
        # Set the initial state of the chain
        params_current = prior_arguments[0]
        params_list = []
        posterior_list = []
        posterior_current = (likelihood(params_current,likelihood_arguments))*\
                    (prior(params_current, prior_arguments))
        # Run the Metropolis-Hastings algorithm for burning
        burn_samples = 1000
        for i in range(burn_samples):
            # Propose a new state for the chain
            params_proposed=np.random.multivariate_normal(params_current,cov_step_size)
            posterior_proposed=(likelihood(params_proposed,likelihood_arguments))*(prior(params_proposed,\
                                                                                prior_arguments))
            # Calculate the acceptance probability
            acceptance_prob = min(1, posterior_proposed / posterior_current)
            # Accept or reject the proposal
            if np.random.uniform() < acceptance_prob:
                params_current = params_proposed
                posterior_current=posterior_proposed
    else:
        params_current = resume[0]
        posterior_current = resume[1]
        params_list = [params_current]
        posterior_list = [posterior_current]

    for i in range(n):
        params_proposed=np.random.multivariate_normal(params_current,cov_step_size)
        posterior_proposed=(likelihood(params_proposed,likelihood_arguments))*\
        (prior(params_proposed,prior_arguments))
        # Calculate the acceptance probability
        acceptance_prob = min(1, posterior_proposed / posterior_current)
        # Accept or reject the proposal
        if np.random.uniform() < acceptance_prob:
            params_current = params_proposed
            posterior_current=posterior_proposed
        # Store the current state
        params_list.append(params_current)
        posterior_list.append(posterior_current)

    return [
        np.array(params_list),
        np.array(posterior_list),
    ]


def calibrate(x, y, dy, resume=None):
    x_func = np.arange(0, 161)

    #Setting up the prior
    prior_arguments = [[0, 1], np.linalg.inv(np.diag([2**2, 2**2]))]

    #Doing the Metropolis sampling for 10000 values
    results = metropolis([x,y], dy, prior_arguments, 20000, [.2,.2], resume=resume)
    params = results[0]
    posteriors = results[1]

    #Taking 10000 samples from the visited posterior to estimate the percentiles in the model predictions
    # rng = np.random.default_rng()
    # alpha_rand = rng.choice(all_chains,(20000),replace=False)
    # print(alpha_rand.shape, alpha_rand[0], alpha_rand[1], alpha_rand[2])
    # func_rand = [linear(x_func,alpha) for alpha in alpha_rand]

    func_rand = [linear(x_func,alpha) for alpha in params]

    lower = np.percentile(func_rand, 2.5, axis = 0)
    median = np.percentile(func_rand, 50, axis = 0)
    upper = np.percentile(func_rand, 97.5, axis = 0)
    
    return [x_func, lower, median, upper], params, posteriors


# def cali_plot(x, y, dy, resume):
#     x_func, lower, median, upper, alpha_rand = calibrate(x, y, dy, prior)

#     line_trace = go.Scatter(
#         x=x_func,
#         y=median,
#         line=dict(color='#274653'),
#         mode='lines', hoverinfo="skip",
#     )
#     upper_trace = go.Scatter(
#         name='Upper Bound',
#         x=x_func,
#         y=upper,
#         mode='lines',
#         marker=dict(color="#444"),
#         line=dict(width=0),
#         showlegend=False
#     )
#     lower_trace = go.Scatter(
#         name='Lower Bound',
#         x=x_func,
#         y=lower,
#         marker=dict(color="#444"),
#         line=dict(width=0),
#         mode='lines',
#         fillcolor='rgba(68, 68, 68, 0.3)',
#         fill='tonexty',
#         showlegend=False
#     )

#     return [line_trace, upper_trace, lower_trace, alpha_rand]
