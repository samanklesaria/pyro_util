# type: ignore
import inspect
from collections import namedtuple
from typing import Optional, Union

import arviz as az
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pandas as pd
import xarray as xr
from numpyro import handlers
from numpyro.contrib.hsgp.laplacian import eigenfunctions
from numpyro.contrib.hsgp.spectral_densities import \
    diag_spectral_density_squared_exponential
from numpyro.infer import MCMC, NUTS, Predictive
from pandas.io.parsers.readers import Callable
from patsy import Term, dmatrices


def fit_nuts(model, *args, num_samples=1000, **kwargs):
    "Run four chains with 500 warmup samples using the NUTS kernel."
    nuts_kernel = NUTS(model)
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=500,
        num_samples=num_samples,
        num_chains=4)
    mcmc.run(jax.random.PRNGKey(0), *args, **kwargs)
    return mcmc


def from_numpyro(df: Optional[pd.DataFrame], model: Callable, mcmc: MCMC, *args, predictive=True):
    """Create `InferenceData` from a `MCMC` run of `model` with sections for posterior and prior
    predictive samples as well as constant data from the dataframe `df` if it is provided.
    """
    post_pred = Predictive(
        model,
        mcmc.get_samples())(
        jax.random.PRNGKey(1),
        *args,
        predictive=True) if predictive else None
    prior = Predictive(
        model,
        num_samples=1000)(
        jax.random.PRNGKey(2),
        *args,
        predictive=True) if predictive else None
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
    """Track the plates associated with each sample site.
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
        prior: dist.Distribution = dist.Normal, groups: Optional[str] = None,
        weights: Optional[str] = None, obs_mask: jax.Array = None, predictive: bool = False):
    """Run a generalized linear model following `formula` on the dataframe `df` with family
    `family`. If the `groups` argument is provided, find separate variance parameters for each group.
    Use the distribution `prior` as the prior distribution for coefficients.
    If `predictive=False`, condition on the endogenous variable in `df`. Otherwise, sample a
    result. Return the endogenous variable.
    """
    y, design = dmatrices(formula, df)
    y = jnp.array(y[:, 0])
    X = jnp.array(design)
    mle_params = jnp.linalg.solve(X.T @ X, X.T @ y)
    stdy = y.std() if dist is dist.Normal else 1.0
    stds = 2.5 * stdy / X.std(axis=0)
    mu = 0.0  # Observation mean for each unit
    for (k, v) in design.design_info.term_slices.items():
        subX = X[:, v]
        loc = mle_params[v] if k == Term([]) else jnp.zeros(1)
        K = subX.shape[1]
        if K == 1:
            beta = numpyro.sample(
                k.name(),
                prior(loc[0], 2.5 * stdy if k == Term([]) else stds[v][0]))
            mu = mu + jnp.array(subX[:, 0]) * beta
        else:  # Stack categorical factors into their own plate
            with numpyro.plate(k.name() + "s", K):
                beta = numpyro.sample(
                    k.name(),
                    prior(loc, 2.5 * stdy if k == Term([]) else stds[v]))
                mu = mu + subX @ beta

    dist_args = set(inspect.getfullargspec(family).args)
    if 'scale' in dist_args:
        if groups is not None:
            with numpyro.plate("groups", df[groups].nunique()):
                sigmas = numpyro.sample("sigma", dist.Exponential(1 / stdy))
                sigma = sigmas[df[groups].cat.codes.to_numpy()]
        else:
            sigma = numpyro.sample("sigma", dist.Exponential(1 / stdy))
    if family is dist.Poisson:
        data_dist = dist.Poisson(jnp.exp(mu))
    elif family is dist.Binomial:
        data_dist = dist.Bernoulli(logits=mu)
    elif family is dist.NegativeBinomial2:
        data_dist = dist.NegativeBinomial2(jnp.exp(mu), 1 / sigma)
    elif set(['loc', 'scale']) <= dist_args:
        data_dist = family(mu, sigma)
    else:
        raise Exception("Unknown family")
    with handlers.scale(scale=jnp.array(df[weights]) if weights is not None else 1.0):
        with numpyro.plate("obs", X.shape[0]):
            numpyro.deterministic("mu", mu)
            return numpyro.sample(
                'y', data_dist, obs=None if predictive else y, obs_mask=obs_mask)


class hsgp(namedtuple("hsgp", "spd beta ell m")):
    """Represents a sample of a Hilbert Space Gaussian Process.

    Stored parameters:
    `ell`:  length of the interval from zero being approximated
    `m`:    number of eigenvectors used in the approximation
    `spd`:  root of the spectral densities
    `beta`: sampled eigenvector coefficients
    """
    __slots__ = ()

    def at(self, x):
        phi = eigenfunctions(x=x, ell=self.ell, m=self.m)
        return phi @ (self.spd * self.beta)


def hsgp_rbf(
        prefix: str, alpha: float, ell: float,
        m: int, length: Union[float, list[float]]) -> hsgp:
    """Sample a `hsgp` with an RBF kernel.

    Parameters:
    `prefix`:   Prefix fpr sample sites of eigenvector coefficients
    `alpha`:    Uniform scaling for the kernel
    `ell`:      Kernel lengthscale
    `length`:   Length of the interval from zero for which the approximation should be accurate
    """
    dim = len(length) if hasattr(length.__class__, "__len__") else 1
    spd = jnp.sqrt(diag_spectral_density_squared_exponential(
        alpha=alpha, length=length, ell=ell, m=m, dim=dim))
    with handlers.scope(prefix=prefix):
        with numpyro.plate("basis", len(spd)):
            beta = numpyro.sample("beta", dist.Normal())
    return hsgp(spd, beta, ell, m)
