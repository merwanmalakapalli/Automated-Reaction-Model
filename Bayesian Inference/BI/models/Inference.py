import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
import jax.random as random
import jax.numpy as jnp
from jax import grad, jacfwd
import numpy as np
from scipy.stats import norm
from typing import Callable, Dict, Tuple, Optional
from numpyro.infer import init_to_mean, init_to_feasible, init_to_uniform
import io
import sys
import time
import threading
import re
from contextlib import redirect_stdout
 
class BayesianKineticAI():
    """
    Reaction
 
    Args:
        
 
    Methods:
 
    """
 
    def __init__(self,
                 idf_threshold: float = 0.9,
                 plikelhood_pts: int = 50):
        self.idf_threshold = idf_threshold
        self.profile_likelihood_points = plikelhood_pts
        self.params_corr = None
        self.non_identifiable_params = []

    """
        Analyzes correlations between sampled parameters to identify potential identifiability issues.

        Args:
            mcmc_samples (Dict): Dictionary of posterior samples from MCMC, where each key is a parameter name
                                 and the value is a 1D jax.numpy array of samples.

        Returns:
            Tuple[jnp.ndarray, list]: A tuple containing:
                - correlation_matrix (jnp.ndarray): 2D array with Pearson correlation coefficients between parameters.
                - non_identifiable_params (list): List of parameter names that are highly correlated (above threshold).
        """
    def analyze_parameter_correlations(self, mcmc_samples: Dict) -> Tuple[jnp.ndarray, list]:
        """
 
        """
        param_names = [k for k in mcmc_samples.keys() if k != 'sigma']
        samples = jnp.stack([mcmc_samples[k] for k in param_names], axis=1)
        
        correlation_matrix = jnp.corrcoef(samples.T)
        
        non_identifiable = []
        n_params = correlation_matrix.shape[0]
        for i in range(n_params):
            for j in range(i+1, n_params):
                if abs(correlation_matrix[i,j]) > self.idf_threshold:
                    non_identifiable.extend([param_names[i], param_names[j]])
                    
        self.params_corr = correlation_matrix
        self.non_identifiable_params = list(set(non_identifiable))
        
        return correlation_matrix, non_identifiable

    """
       Performs profile likelihood analysis for a single parameter to assess structural identifiability.

       Args:
           function (Callable): Forward model function taking (X, params) as input.
           param_name (str): Name of the parameter to profile.
           mcmc_samples (Dict): Posterior samples from MCMC.
           X (jnp.ndarray): Input data for the model.
           Y (jnp.ndarray): Observed output data.

       Returns:
           Tuple[jnp.ndarray, jnp.ndarray]: 
               - param_range: Range of values for the parameter.
               - profile_likes: Log-likelihoods evaluated at each fixed parameter value.
       """
    def profile_likelihood_analysis(self,
                                  function: Callable,
                                  param_name: str,
                                  mcmc_samples: Dict,
                                  X: jnp.ndarray,
                                  Y: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """   
        """
        param_samples = mcmc_samples[param_name]
        param_min, param_max = jnp.min(param_samples), jnp.max(param_samples)
        param_range = jnp.linspace(param_min, param_max, self.profile_likelihood_points)
        
        profile_likes = []
        for param_val in param_range:
            def fixed_param_model(X, params):
                params[param_name] = param_val
                return function(X, params)
            
            sigma = jnp.median(mcmc_samples['sigma'])
            pred = fixed_param_model(X, mcmc_samples)
            log_like = jnp.sum(norm.logpdf(Y, pred, sigma))
            profile_likes.append(log_like)
            
        return param_range, jnp.array(profile_likes)

    """
        Runs Bayesian inference using NUTS and optionally checks for parameter identifiability issues.

        Args:
            function (Callable): Forward model function taking (X, params) as input.
            priors (Callable): Function returning a dictionary of parameters sampled using numpyro.
            X (jnp.ndarray): Input data.
            Y (jnp.ndarray): Observed output data.
            num_samples (int): Number of MCMC samples to draw.
            num_warmup (int): Number of warm-up iterations for MCMC.
            check_identifiability (bool): Whether to analyze parameter identifiability.

        Returns:
            MCMC: Trained MCMC object containing posterior samples.
        """
    def run_bayesian_inference(self, function, priors, X, Y, num_samples=2000, num_warmup=1000, check_identifiability: bool = False):
        # rng_key = random.PRNGKey(random.randint(0, 1000))  # Generate a random PRNG key
        rng_key = random.PRNGKey(0)
        rng_key, rng_key_ = random.split(rng_key)

        def model(X: jnp.ndarray, y: jnp.ndarray = None) -> None:
            # Samples prior parameters
            params = priors()

            # Predicted output from ODE model
            mu = function(X, params)

            # Samples noise
            sigma = numpyro.sample("sigma", dist.LogNormal(0, 1))  # we  eed to add an arg for this
 
            numpyro.sample("y", dist.Normal(mu, sigma), obs=y)
        
        # MCMC algorithm
        kernel = NUTS(model)
        
        # kernel = NUTS(model, init_strategy=None)
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
        mcmc.run(rng_key_, X, Y)
        if check_identifiability:
            corr_matrix, non_ident = self.analyze_parameter_correlations(mcmc.get_samples())
            
            if non_ident:
                print("Warning: Potential identifiability issues detected for parameters: {non_ident}")
                
                for param in non_ident:
                    param_vals, prof_likes = self.profile_likelihood_analysis(
                        function, param, mcmc.get_samples(), X, Y
                    )
                    if jnp.std(prof_likes) < 0.1:  
                        print(f"Parameter {param} shows structural non-identifiability")
                        
        return mcmc

    """
       Predicts output values using the posterior samples obtained from a fitted Bayesian inference model.

       Args:
           BI_model (MCMC): MCMC object containing posterior samples.
           function (Callable): Forward model function taking (X, params) as input.
           priors (Callable): Function returning a dictionary of model parameters for sampling.
           X (jnp.ndarray): New input data for prediction.
           num_samples (int): Number of posterior predictive samples to generate (unused here but can be extended).
           num_warmup (int): Number of warm-up steps (unused here but can be extended).

       Returns:
           jnp.ndarray: Posterior predictive samples for the output variable.
       """
    def bayesian_inference_predict(self, BI_model, function, priors, X, num_samples=2000, num_warmup=1000):
        rng_key = random.PRNGKey(0)
        rng_key, rng_key_         = random.split(rng_key)
        
        def model2(X: jnp.ndarray, y:jnp.ndarray = None) -> None:
            params = priors() #irrelevant
            mu     = function(X, params)
            sigma  = numpyro.sample("sigma", dist.LogNormal(0,1))
 
            # yield prediction
            numpyro.sample("y", dist.Normal(mu, sigma), obs=y)
 
 
        # Predictive overrides sampling with posterior samples
        predictive_obj                = Predictive(model2, posterior_samples=BI_model.get_samples(), parallel=True)
        y_prredicted                  = predictive_obj(rng_key, X)
 
        return y_prredicted['y']
        #    numpyro.sample("y", dist.Normal(mu, sigma), obs=y)


        # predictive_obj                = Predictive(model2, posterior_samples=BI_model.get_samples(), parallel=True)
        # y_prredicted                  = predictive_obj(rng_key, X)
        #
        # return y_prredicted['y']

