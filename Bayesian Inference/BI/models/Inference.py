import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
import jax.random as random
import jax.numpy as jnp
from jax import grad, jacfwd
import numpy as np
from scipy.stats import norm
from typing import Callable, Dict, Tuple, Optional
 
 
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
 
    def run_bayesian_inference(self, function, priors, X, Y, num_samples=2000, num_warmup=1000, check_identifiability: bool = False):
        # rng_key = random.PRNGKey(random.randint(0, 1000))  # Generate a random PRNG key
        rng_key = random.PRNGKey(0)
        rng_key, rng_key_ = random.split(rng_key)
 
        def model(X: jnp.ndarray, y: jnp.ndarray = None) -> None:
            params = priors()
            mu = function(X, params)
 
            sigma = numpyro.sample("sigma", dist.LogNormal(0, 1))  # we  eed to add an arg for this
 
            numpyro.sample("y", dist.Normal(mu, sigma), obs=y)
        
        kernel = NUTS(model)
        # from numpyro.infer import init_to_mean, init_to_feasible, init_to_uniform
        # kernel = NUTS(model, init_strategy=None)
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
        mcmc.run(rng_key_, X, Y)
        if check_identifiability:
            corr_matrix, non_ident = self.analyze_parameter_correlations(mcmc.get_samples())
            
            if non_ident:
                print(f"Warning: Potential identifiability issues detected for parameters: {non_ident}")
                
                for param in non_ident:
                    param_vals, prof_likes = self.profile_likelihood_analysis(
                        function, param, mcmc.get_samples(), X, Y
                    )
                    if jnp.std(prof_likes) < 0.1:  
                        print(f"Parameter {param} shows structural non-identifiability")
                        
        return mcmc
 
 
    def bayesian_inference_predict(self, BI_model, function, priors, X, num_samples=2000, num_warmup=1000):
        rng_key = random.PRNGKey(0)
        rng_key, rng_key_         = random.split(rng_key)
        
        def model2(X: jnp.ndarray, y:jnp.ndarray = None) -> None:
            params = priors()
            mu     = function(X, params)
            sigma  = numpyro.sample("sigma", dist.LogNormal(0,1))
 
            numpyro.sample("y", dist.Normal(mu, sigma), obs=y)
 
 
        predictive_obj                = Predictive(model2, posterior_samples=BI_model.get_samples(), parallel=True)
        y_prredicted                  = predictive_obj(rng_key, X)
 
        return y_prredicted['y']
            numpyro.sample("y", dist.Normal(mu, sigma), obs=y)


        predictive_obj                = Predictive(model2, posterior_samples=BI_model.get_samples(), parallel=True)
        y_prredicted                  = predictive_obj(rng_key, X)

        return y_prredicted['y']

