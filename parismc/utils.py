from jax.scipy.special import gammainc
from scipy.optimize import bisect
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)

# define a JAX-native weighted covariance function
def jax_weighted_cov(x, y=None, aweights=None, ddof=0):
    """Compute the weighted covariance matrix of x and y using JAX. 
    Function does not exists with those parameters in JAX"""
    if y is None:
        y = x
    if aweights is None:
        aweights = jnp.ones(x.shape[0])
    # Normalize weights
    aweights = aweights / jnp.sum(aweights)
    # Compute weighted means
    mean_x = jnp.sum(aweights[:, None] * x, axis=0)
    mean_y = jnp.sum(aweights[:, None] * y, axis=0)
    # Center the data
    x_centered = x - mean_x
    y_centered = y - mean_y
    # Compute weighted covariance
    cov_matrix = jnp.dot((aweights[:, None] * x_centered).T, y_centered)
    return cov_matrix

def weighting_seeds_manypoint(points_array, means_array, inv_covariance, norm_term, proposalcoeff_array):
    """Mixture density at each point under a single shared covariance.

    For every point ``x_i`` returns ``sum_k norm_term * coeff_k *
    exp(-0.5 (x_i - mu_k)^T C^{-1} (x_i - mu_k))`` summed over the means ``mu_k``.

    Shapes
    ------
    points_array        : (n_points, ndim)
    means_array         : (n_means, ndim)
    inv_covariance      : (ndim, ndim)
    norm_term           : scalar
    proposalcoeff_array : (n_means,)
    returns             : (n_points,)
    """
    def per_mean(k):
        diff = points_array - means_array[k]                       # (n_points, ndim)
        exponent = -0.5 * jnp.einsum('ij,jk,ik->i', diff, inv_covariance, diff)
        return norm_term * jnp.exp(exponent) * proposalcoeff_array[k]
    return jax.vmap(per_mean)(jnp.arange(means_array.shape[0])).sum(axis=0)


def weighting_seeds_onepoint_with_onemean(points_array, means_array, inv_covariance, norm_term):
    """Paired single-Gaussian density: point ``i`` evaluated against mean ``i``.

    ``points_array`` and ``means_array`` are aligned row-by-row, sharing one
    covariance. Returns ``norm_term * exp(-0.5 (x_i - mu_i)^T C^{-1} (x_i - mu_i))``.

    Shapes
    ------
    points_array   : (n, ndim)
    means_array    : (n, ndim)   (row-aligned with points_array)
    inv_covariance : (ndim, ndim)
    returns        : (n,)
    """
    diff = points_array - means_array                              # (n, ndim)
    exponent = -0.5 * jnp.einsum('ij,jk,ik->i', diff, inv_covariance, diff)
    return norm_term * jnp.exp(exponent)


def weighting_seeds_manycov(points_array, means_array, inv_covariances_array, norm_terms_array, proposalcoeff_array):
    """Mixture density at each point under per-mean covariances (the cross-term kernel).

    For every point ``x_k`` returns ``sum_m norm_m * coeff_m *
    exp(-0.5 (x_k - mu_m)^T C_m^{-1} (x_k - mu_m))`` summed over all means ``mu_m``,
    where each mean carries its own covariance history.

    Shapes
    ------
    points_array          : (n_points, ndim)
    means_array           : (n_means, ndim)
    inv_covariances_array : (n_means, ndim, ndim)
    norm_terms_array      : (n_means,)
    proposalcoeff_array   : (n_means,)
    returns               : (n_points,)
    """
    def per_point(point):
        diff = point[None, :] - means_array                        # (n_means, ndim)
        exponent = -0.5 * jnp.einsum('mj,mjk,mk->m', diff, inv_covariances_array, diff)
        return jnp.sum(norm_terms_array * jnp.exp(exponent) * proposalcoeff_array)
    return jax.vmap(per_point)(points_array)


def find_sigma_level(ndim, prob):
    """Helper method to compute sigma level for a given probability."""
    if not (0 <= prob <= 1):
        raise ValueError("Probability must be between 0 and 1.")        
    def analytical_probability(k):
        return gammainc(ndim / 2, k**2 / 2) - prob
    return bisect(analytical_probability, 0, 10, xtol=1e-6)


