import numpy as np
import plotly.graph_objs as go

def linear(x,params):
    return params[0]+params[1]*x

def likelihood(params,arguments):
#Assumed format for data=[xvals,yvals]
    data, model, sigmas = arguments
    likelihood_log_val=0

    for i in range(len(data[0])):
        likelihood_log_val=likelihood_log_val-1/2*((data[1][i] - model(data[0][i],params)) / sigmas[i])**2\
        -np.log(2*np.pi*sigmas[i]**2)/2
      
        
    return np.exp(likelihood_log_val)

def prior_linear(params_vals,arguments):
    params0,params0_Cov_Inv_matrix=arguments
    mu=np.array(params_vals)-np.array(params0)
    params_size=len(params_vals)
    return (2*np.pi)**(-params_size/2)*np.linalg.det(params0_Cov_Inv_matrix)**(-1)*np.exp(-np.dot(mu,np.dot(params0_Cov_Inv_matrix,mu))/2)

def metropolis(data, sigma, prior, prior_arguments, likelihood, model,\
               num_iterations, step_size):
    # step_size should be a list the size of the parameters of the model
    likelihood_arguments=[data, model, sigma]
    initial_parameters=prior_arguments[0]
    # thermalizing
    burn_samples=1000
    # Set the initial state of the chain
    params_current=initial_parameters
    params_list=[]
    posterior_list=[]
    acceptance_times=0
    cov_step_size=np.diag(step_size)**2
    posterior_current=(likelihood(params_current,likelihood_arguments))*(prior(params_current, prior_arguments))
    # Run the Metropolis-Hastings algorithm for burning
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
    for i in range(num_iterations):
        params_proposed=np.random.multivariate_normal(params_current,cov_step_size)
        posterior_proposed=(likelihood(params_proposed,likelihood_arguments))*\
        (prior(params_proposed,prior_arguments))
        # Calculate the acceptance probability
        acceptance_prob = min(1, posterior_proposed / posterior_current)
        # Accept or reject the proposal
        if np.random.uniform() < acceptance_prob:
            params_current = params_proposed
            posterior_current=posterior_proposed
            acceptance_times=acceptance_times+1
        # Store the current state
        params_list.append(params_current)
        posterior_list.append(posterior_current)
    #Rule of thumb acceptance is around 50%. 
    #You could plot the accuracy of the estimations as a function of this rate, that would be interesting to see. 
    # print(acceptance_times/num_iterations*100,"%")
    
    return(np.array(params_list),np.array(posterior_list),\
           acceptance_times/num_iterations*100)


def calibrate(x, y, dy, prior):
    x_func = np.arange(0, 161)

    #Setting up the prior
    prior_arguments=[prior, np.linalg.inv(np.diag([2**2,2**2]))]

    #Doing the Metropolis sampling for 100000 values
    results=metropolis([x,y],dy, prior_linear,\
                            prior_arguments, likelihood, linear, 20000, [2,2])
    all_chains =results[0]

    #Taking 10000 samples from the visited posterior to estimate the percentiles in the model predictions

    rng = np.random.default_rng()
    alpha_rand = rng.choice(all_chains,(20000),replace=False)
    print(alpha_rand.shape, alpha_rand[0], alpha_rand[1], alpha_rand[2])
    func_rand=[linear(x_func,alpha) for alpha in alpha_rand]

    lower = np.percentile(func_rand, 2.5, axis = 0)
    median = np.percentile(func_rand, 50, axis = 0)
    upper = np.percentile(func_rand, 97.5, axis = 0)
    

    return [x_func, lower, median, upper, alpha_rand]


def cali_plot(x, y, dy, prior):
    x_func, lower, median, upper, alpha_rand = calibrate(x, y, dy, prior)

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

    return [line_trace, upper_trace, lower_trace, alpha_rand]


# def generator_function(username):
#     count = 0
#     while True:
#         yield f"Hello, {username}! Count: {count}"
#         count += 1

# def generator_function():
#     # Emit the new samples to all connected clients
#     socketio.emit('update_samples', {'new_samples': new_samples})
#     count = 1
#     while True:
#         yield count
#         count += 1