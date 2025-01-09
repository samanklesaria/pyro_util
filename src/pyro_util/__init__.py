# type: ignore
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro import handlers
import numpyro
from numpyro.contrib.hsgp.laplacian import eigenfunctions
from numpyro.contrib.hsgp.spectral_densities import (
    diag_spectral_density_squared_exponential
)
import jax
import jax.numpy as jnp
import xarray as xr
from typing import Optional, Union
import pandas as pd
from patsy import dmatrices, Term
import arviz as az
from collections import namedtuple

def fit_nuts(model, *args, num_samples=1000, **kwargs):
    "Run four chains with 500 warmup samples using the NUTS kernel."
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=num_samples, num_chains=4)
    mcmc.run(jax.random.PRNGKey(0), *args, **kwargs)
    return mcmc

def from_numpyro(df, model, mcmc, *args, predictive=True):
    post_pred = Predictive(model, mcmc.get_samples())(jax.random.PRNGKey(1), *args, predictive=True) if predictive else None
    prior = Predictive(model, num_samples=1000)(jax.random.PRNGKey(2), *args, predictive=True) if predictive else None
    dimwrap = extract_dims()
    with dimwrap:
        with handlers.seed(rng_seed=1):
            model(*args, predictive=False)
    result = az.from_numpyro(mcmc,
        prior=prior,
        posterior_predictive=post_pred,
        dims=dimwrap.dims)
    if df is not None:
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
    y, design = dmatrices(formula, df)
    y = jnp.array(y[:,0])
    X = jnp.array(design)
    mle_params = jnp.linalg.solve(X.T @ X, X.T @ y)
    stdy = y.std()
    stds = 2.5 * stdy / X.std(axis=0)
    mu = 0.0 # Observation mean for each unit
    for (k,v) in design.design_info.term_slices.items():
        subX = X[:, v]
        loc = mle_params[v] if k == Term([]) else jnp.zeros(1)
        K = subX.shape[1]
        if K == 1:
            beta = numpyro.sample(k.name(), dist.Normal(loc[0], 2.5 * stdy if  k == Term([]) else stds[v][0]))
            mu = mu + jnp.array(subX[:, 0]) * beta
        else: # Stack categorical factors into their own plate
            with numpyro.plate(k.name() + "s", K):
                beta = numpyro.sample(k.name(), dist.Normal(loc, 2.5 * stdy if  k == Term([]) else stds[v]))
                mu = mu + subX @ beta

    if family is not dist.Poisson:
        if groups is not None:
            with numpyro.plate("groups", df[groups].nunique()):
               sigmas = numpyro.sample("sigma", dist.Exponential(1 / stdy))
               sigma = sigmas[df[groups].cat.codes.to_numpy()]
        else:
            sigma = numpyro.sample("sigma", dist.Exponential(1 / stdy))
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

class hsgp(namedtuple("hsgp", "spd beta ell m")):
    __slots__ = ()
    def at(self, x):
        phi = eigenfunctions(x=x, ell=self.ell, m=self.m)
        return phi @ (self.spd * self.beta)

def hsgp_rbf(
    prefix: str,
    alpha: float,
    ell: float,
    m: int,
    length: Union[float,list[float]]) -> hsgp:
    dim = len(length) if hasattr(length.__class__, "__len__") else 1
    spd = jnp.sqrt(diag_spectral_density_squared_exponential(
            alpha=alpha, length=length, ell=ell, m=m, dim=dim))
    with handlers.scope(prefix=prefix):
        with numpyro.plate("basis", len(spd)):
            beta = numpyro.sample("beta", dist.Normal())
    return hsgp(spd, beta, ell, m)
