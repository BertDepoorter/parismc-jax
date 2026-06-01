import numpy as np
from scipy.optimize import minimize

# import JAX libraries
from .utils import (
    jax_weighted_cov
)

import jax.numpy as jnp
from jax import jit
import jax
import optax
jax.config.update("jax_enable_x64", True)

# Default hyperparameters for the optax-based beta optimization.
_BETA_LOWER = 0.5      # beta is an anisotropic *shrink* factor applied to Sigma
_BETA_UPPER = 1.0 + 1e-3
_BETA_SLOPE = 50.0     # sharpness of the sigmoid relaxation of the boundary indicator
_BETA_STEPS = 100      # fixed number of optimizer steps (keeps the routine jittable)
_BETA_LR = 1e-2


def negative_BN_log_like(beta, diff, Adiff, W, W_sum, original_log_det_cov,
                         original_zeromean_samples, mean, ndim, integral_num=None,
                         slope=_BETA_SLOPE):
    """Truncated-Gaussian penalized negative log-likelihood as a function of ``beta``.

    This is the *smooth* surrogate of the original objective. The non-differentiable
    boundary indicator ``any(z out of [0,1])`` is replaced by a product of sigmoids so
    the objective is differentiable and usable with gradient-based (optax) optimizers.

    ``beta`` is the per-axis scale applied to Sigma. A zero-mean draw ``z ~ N(0, Sigma)``
    maps to ``z/beta + mean``; the in-bounds condition ``0 <= z/beta + mean <= 1`` is
    equivalent to ``-mean*beta <= z <= (1-mean)*beta``.

    ``integral_num`` is accepted for backward-compatibility but unused: the Monte-Carlo
    fraction is estimated directly from ``original_zeromean_samples``.
    """
    lo = (0.0 - mean) * beta            # (ndim,)
    hi = (1.0 - mean) * beta            # (ndim,)
    # Smooth in-bounds membership per sample/dimension, then product over dims.
    within = (jax.nn.sigmoid(slope * (original_zeromean_samples - lo[None, :]))
              * jax.nn.sigmoid(slope * (hi[None, :] - original_zeromean_samples)))
    portion = jnp.mean(jnp.prod(within, axis=1))
    portion = jnp.clip(portion, 1e-12, 1.0)

    A = jnp.einsum('ij,ji->i', diff * beta[None, :] * beta[None, :], Adiff)
    A_inner_W = A @ W

    scaling_factor = jnp.prod(beta ** 2)
    gaussian_normterm = jnp.exp(-0.5 * original_log_det_cov) * jnp.sqrt(scaling_factor) / jnp.sqrt((2 * np.pi) ** ndim)
    return 0.5 * A_inner_W + W_sum * jnp.log(portion / gaussian_normterm)


def find_max_beta(diff, Adiff, W, W_sum, original_log_det_cov, original_zeromean_samples,
                  mean, ndim, integral_num=None, *,
                  n_steps=_BETA_STEPS, lr=_BETA_LR, slope=_BETA_SLOPE,
                  lower=_BETA_LOWER, upper=_BETA_UPPER):
    """Minimize the truncated-Gaussian objective over ``beta`` with optax.

    Uses a fixed number of projected Adam steps (projection = clip onto the box
    ``[lower, upper]``). Because the step count is static and every operation is pure,
    the whole routine is ``jit``- and ``vmap``-able across processes -- a property the
    old SciPy ``Powell``/``minimize`` path could never offer.

    Returns
    -------
    beta : jnp.ndarray of shape (ndim,)
    """
    obj = lambda b: negative_BN_log_like(
        b, diff, Adiff, W, W_sum, original_log_det_cov,
        original_zeromean_samples, mean, ndim, slope=slope)

    beta0 = jnp.ones((ndim,))
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(beta0)

    def body(carry, _):
        beta, state = carry
        value, grad = jax.value_and_grad(obj)(beta)
        updates, state = optimizer.update(grad, state, beta)
        beta = optax.apply_updates(beta, updates)
        beta = jnp.clip(beta, lower, upper)            # projection onto the box
        return (beta, state), value

    (beta, _), _ = jax.lax.scan(body, (beta0, opt_state), None, length=n_steps)
    # Guard against non-finite results (e.g. degenerate covariance): fall back to 1.
    beta = jnp.where(jnp.all(jnp.isfinite(beta)), beta, jnp.ones((ndim,)))
    return beta

def oracle_approximating_shrinkage(cov_matrix, n_samples):
    """
    Estimate the Oracle Approximating Shrinkage covariance matrix.

    Parameters
    ----------
    cov_matrix : array-like of shape (n_features, n_features)
        The sample covariance matrix.

    n_samples : int
        The number of samples used to compute the sample covariance.

    Returns
    -------
    shrunk_cov : ndarray of shape (n_features, n_features)
        The Oracle Approximating Shrinkage covariance matrix.

    shrinkage : float
        Shrinkage coefficient used in the computation.
    """


    p = cov_matrix.shape[0]
    mu = jnp.trace(cov_matrix) / p
    sum_of_squares = (cov_matrix**2).sum() # which is Tr(\Sigma^2)
    squared_trace = (mu * p)**2 # which is Tr(\Sigma)^2

    numerator = (1 - 2/p)*sum_of_squares + squared_trace
    denominator = (n_samples + 1 - 2/p)*sum_of_squares + (1 - n_samples/p)*squared_trace
    # jittable: avoid Python branching / in-place `.flat` mutation on immutable arrays.
    safe_denom = jnp.where(denominator == 0, 1.0, denominator)
    shrinkage = jnp.where(denominator == 0, 1.0, jnp.minimum(numerator / safe_denom, 1.0))

    # Shrink towards mu * I: off-diagonals scaled by (1 - s), diagonal pulled toward mu.
    shrunk_cov = (1.0 - shrinkage) * cov_matrix + shrinkage * mu * jnp.eye(p)

    return shrunk_cov, shrinkage