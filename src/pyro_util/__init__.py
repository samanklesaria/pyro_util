import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro import handlers
import jax
import jax.numpy as jnp
import xarray as xr
from typing import Optional, Union
import pandas as pd
import numpy as np
from patsy import dmatrices, Term
import arviz as az

def fit_nuts(model, *args, num_samples=1000, **kwargs):
    "Run four chains with 500 warmup samples using the NUTS kernel."
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=num_samples, num_chains=4)
    mcmc.run(jax.random.PRNGKey(0), *args, **kwargs)
    return mcmc

def from_numpyro(df, model, mcmc, *args):
    post_pred = Predictive(model, mcmc.get_samples())(jax.random.PRNGKey(1), *args, predictive=True)
    prior = Predictive(model, num_samples=1000)(jax.random.PRNGKey(2), *args, predictive=True)
    dimwrap = extract_dims()
    with dimwrap:
        with handlers.seed(rng_seed=1):
            model(*args, predictive=False)
    result = az.from_numpyro(mcmc,
        prior=prior,
        posterior_predictive=post_pred,
        dims=dimwrap.dims)
    result.constant_data = xr.Dataset.from_dataframe(df)
    result._groups.append('constant_data')
    return result

class extract_dims(handlers.Messenger):
    """
    This effect handler tracks the plates associated with each sample site.
    The resulting map from sample names to lists of plates is stored
    in `self.dims`.
    """
    def __init__(self, fn=None):
        self.dims = {}
        super().__init__(fn)

    def process_message(self, msg):
        if msg["type"] in ("sample", "deterministic"):
            dims = [a.name for a in msg['cond_indep_stack']]
            dims.reverse()
            self.dims[msg['name']] = dims

def glm(formula: str, df: pd.DataFrame, family: dist.Distribution = dist.Normal,
    groups: Optional[str] = None, weights: Optional[str] = None, predictive: bool = False):
    """
    Create a generalized linear model following `formula`.
    We assume that the associated link function has already been applied to the RHS,
    so e.g. count observations with Poisson families are on a log scale.
    If the `groups` argument is provided, separate variance parameters are found for each group.
    """
    y, X = dmatrices(formula, df)
    y = jnp.array(y[:,0])
    info = X.design_info.factor_infos
    mu = 0.0
    for (k,v) in X.design_info.term_slices.items():

        # The intercept term has a normal prior around the average y value
        if k == Term([]):
            baseline_y = y[baseline_mask(df, info)]
            mu = mu + numpyro.sample(k.name(), dist.Normal(baseline_y.mean(), 2.5 * baseline_y.std()))

        # Priors for coefficients for continuous factors are normal about zero with scale $2.5 \sigma_y / \sigma_x$.
        # This is equivalent to standardizing the predictors and letting the scales be $2.5 \sigma_y$.
        # In this case, the cofficients represent the change in $y$ we would get by increasing each covariate by a
        # single standard deviation. Allowing each change in $x$ to produce a change in $y$ covering almost all of
        # the observed variance (2.5 standard deviations) therefore gives us a weakly informative prior.
        else:
            all_categorical = all(info[f].type == 'categorical' for f in k.factors)
            subX = X[:, v]
            mask = subX != 0.0
            stdx, stdy = ([], [])
            for i in range(mask.shape[1]):
                stdx.append(1.0 if all_categorical else np.sqrt(np.square(subX[mask[:, i], i]).mean()))
                stdy.append(y[mask[:, i]].std())
            K = subX.shape[1]
            if K == 1:
                beta = numpyro.sample(k.name(), dist.Normal(0.0, 2.5 * stdy[0] / stdx[0]))
                mu = mu + jnp.array(subX[:, 0]) * beta
            else:
                with numpyro.plate(k.name() + "s", K):
                    beta = numpyro.sample(k.name(),
                        dist.Normal(jnp.zeros(K), 2.5 * jnp.stack(stdy) / jnp.stack(stdx)))
                mu = mu + jnp.array(subX) @ beta

    # The scale of the noise is given an Exponential prior with a mean matching the observed standard deviation.
    # We certainly wouldn't expect noise scales larger than this, as regressing on covariates should allow us to
    # *reduce* the variance rather than increasing it.
    if family is not dist.Poisson:
        if groups is not None:
            with numpyro.plate("groups", df[groups].nunique()):
               sigmas = numpyro.sample("sigma", dist.Exponential(1 / y.std()))
               sigma = sigmas[df[groups].cat.codes.to_numpy()]
        else:
            sigma = numpyro.sample("sigma", dist.Exponential(1 / y.std()))
    if family is dist.Poisson:
        data_dist = dist.Poisson(jnp.exp(mu))
        obs = jnp.exp(y) # Assuming data is log counts
    elif family is dist.NegativeBinomial2:
        data_dist = dist.NegativeBinomial2(jnp.exp(mu), 1 / sigma)
        obs = jnp.exp(y) # Assuming data is log counts
    elif family in [dist.StudentT, dist.Normal, dist.Cauchy, dist.Laplace]:
        data_dist = family(mu, sigma)
        obs = y
    else:
        raise Exception("Unknown family")
    with handlers.scale(scale=jnp.array(df[weights]) if weights is not None else 1.0):
        with numpyro.plate("obs", X.shape[0]):
            numpyro.deterministic("mu", mu)
            return numpyro.sample('y', data_dist, obs=None if predictive else obs)

def baseline_mask(df, info):
    "Find the elements of `df` which have categorical covariates at baseline levels."
    mask = np.ones(df.shape[0], dtype=bool)
    for (k,v) in info.items():
        if v.type == 'categorical':
            mask &= (df[k.name()] == v.categories[0]).to_numpy()
    return mask
