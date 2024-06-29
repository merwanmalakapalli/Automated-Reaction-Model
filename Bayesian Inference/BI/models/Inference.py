import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax.random as random
import jax.numpy as jnp
from numpyro.infer import Predictive

class BayesianKineticAI():
    """
    Reaction

    Args:
        

    Methods:

    """

    def __init__(self):
        pass

    def run_bayesian_inference(self, function, priors, X, Y, num_samples=2000, num_warmup=1000):
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
        mcmc.print_summary()

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

