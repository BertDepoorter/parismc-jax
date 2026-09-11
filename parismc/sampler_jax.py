"""JAX-native PARIS sampler.

The sampler module is re-implemented as :class:`JaxSampler`, which follows the same overall logic as the original
but complies with JAX's static shape and functional programming constraints. The API is unchanged, but the 
internl design is quite different. The original's dynamic list of processes with in-place mutation and 
variable-dependent lengths is replaced by a fixed-size array of processes with a traced boolean mask. All updates 
are now purely functional, with the state carried in an immutable :class:`SamplerState` pytree. 

The expensive log-density function is vmapped over all processes inside a jitted step. 
This replaces the original's multiprocessing pool. The boundary correction optimization 
is now done with optax instead of scipy, and the merging is done on the host device 
after computing the distance matrix in JAX.

More details onn design choices:
1. **A traced boolean ``alive`` mask** of fixed length ``n_proc_max`` replaces the
   shrinking Python list of processes. Merging never changes array shapes; it just flips
   mask bits on the host between jitted blocks. Every per-process kernel is ``vmap``-ed over
   ``n_proc_max`` and dead rows are excluded from reductions/selection by the mask. This complies 
   with JAX's static shape requirements for tracing the pyTree during JIT. 

2. **optax optimizer** for the boundary Beta-correction (``find_max_beta``), not the old
   ``scipy``/``jax.scipy`` ``Powell``/``BFGS`` path. The objective is smoothed with sigmoid
   relaxation of the boundary indicator so it is differentiable, then minimized with a
   fixed number of projected Adam steps. This is fully ``jit``/``vmap`` friendly.

3. **No in-place mutation.** All state lives in an immutable :class:`SamplerState`
   pytree. Updates are purely functional (``.at[...].set/add``). A single jitted ``step`` is
   driven by ``lax.scan`` over a block of iterations while clustering and I/O happen on host
   between blocks. This balances JIt-advantages with the need for killing redundant processes efficiently.

Design simplifications 
---------------------------------------------------
* ``batch_point_num == 1`` (as in the original's actual runtime path). Because every live
  process appends exactly one point per iteration and merging keeps the survivor's fill
  level, **the fill level is identical across all processes** -- so it is a single scalar
  ``nf`` rather than a per-process ``element_num_list``. This is what makes the sliding
  windows shape-static.
* Rejection sampling is **fixed-budget**: draw ``trail_size`` candidates at once, take the
  first in-bounds one (``argmax`` of the validity mask), fall back to candidate 0 if none
  are valid -- matching the original's behaviour, which force-accepted the first valid
  point (the importance-weight rejection check was overridden to ``True``).
* Merging uses the **distance** strategy (Mahalanobis), computed via
  :func:`parismc.clustering.pairwise_mahalanobis_matrix` in JAX with the greedy assignment
  on host. The weight-based merge can be layered on top of the same host.

The expensive ``log_density_func``, i.e some sort of likelihood, is vmapped over all
processes inside the jitted step and stays on-device, replacing ``multiprocessing.Pool``. 
A full JAX rewrite of the clustering/merging is possible but the current hybrid approach keeps
dynamic control over the number of processes, which is a major part of PARIS's efficiency.
"""
# handy python stuff
from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import partial
from typing import Callable, NamedTuple, Optional, Tuple

# set-up JAX and numpy
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# Load custom modules
from .optimization import find_max_beta, oracle_approximating_shrinkage
from .utils import jax_weighted_cov
from .clustering import pairwise_mahalanobis_matrix, get_cluster_indices_from_matrix

logger = logging.getLogger(__name__)


##===========================================================================
## Configuration and state
##===========================================================================

@dataclass(frozen=True)
class JaxSamplerConfig:
    """Hyperparameters for :class:`JaxSampler`.

    Fields (``n_proc_max``, ``n_max``, ``alpha``, ``trail_size``, ``integral_num``, ``gamma``) 
    determine shapes and should be static at compile time.
    """
    n_proc_max: int                 # static upper bound on number of processes (seeds)
    n_max: int                      # static buffer length per process (max accepted points)
    alpha: int = 1000               # sliding-window length for the IS denominator
    gamma: int = 100                # covariance update cadence (iterations)
    trail_size: int = 1000          # rejection-sampling budget per process per iteration
    integral_num: int = 4096        # MC samples for the boundary fraction / beta fit
    boundary_limiting: bool = True  # constrain proposals to [0, 1]^d
    use_beta: bool = True           # enable the Beta boundary correction
    use_beta_threshold: float = 0.1 # enable beta only when oob fraction exceeds this
    cov_jitter: float = 1e-10       # diagonal jitter for covariance stability
    merge_dist: float = 4.0         # Mahalanobis merge radius R_m
    min_log_det_cov: float = -500.0 # reject covariances with log-det below this
    beta_steps: int = 100           # optax steps for the beta fit
    beta_lr: float = 1e-2           # optax learning rate for the beta fit
    seed: int = 0


class SamplerState(NamedTuple):
    """Immutable pytree carrying the full sampler state across a scan.

    Leading axis ``P = n_proc_max`` everywhere there's a per-process quantity appearing.
    ``N = n_max`` is th buffer length per process, ``H = n_max // gamma + 1`` the
    covariance-history length, ``D`` the dimension.
    """
    points: jnp.ndarray        # (P, N, D)  accepted points (== mixture means)
    logdens: jnp.ndarray       # (P, N)     log-density at each accepted point
    wdeno: jnp.ndarray         # (P, N)     running IS-weight denominator (sliding window)
    inv_cov_hist: jnp.ndarray  # (P, H, D, D) history of inverse proposal covariances
    norm_hist: jnp.ndarray     # (P, H)     matching Gaussian normalizers
    now_cov: jnp.ndarray       # (P, D, D)  current proposal covariance
    now_inv_cov: jnp.ndarray   # (P, D, D)  current inverse covariance
    now_norm: jnp.ndarray      # (P,)       current Gaussian normalizer
    now_mean: jnp.ndarray      # (P, D)     current weighted mean (updated every step)
    init_inv_cov: jnp.ndarray  # (P, D, D)  fallback inverse covariance
    init_norm: jnp.ndarray     # (P,)       fallback normalizer
    max_logden: jnp.ndarray    # (P,)       running max log-density per process
    alive: jnp.ndarray         # (P,) bool  traced liveness mask
    nf: jnp.ndarray            # scalar int32  shared fill level (points 0..nf-1 valid)
    it: jnp.ndarray            # scalar int32  iteration counter
    loglike_norm: jnp.ndarray  # scalar     log-density normalizer c (overflow guard)
    key: jnp.ndarray           # PRNG key (threaded -- never global state)


##===========================================================================
# Sampler
##===========================================================================
class JaxSampler:
    """Adaptive importance sampler with a jitted inner step. Orchestration happens on host.

    Parameters
    ----------
    ndim : int
    log_density_func : callable
        Maps an array of points ``(batch, ndim)`` to log-densities ``(batch,)``. Must be
        JAX-traceable (jittable). It is ``vmap``-batched over processes internally.
    config : JaxSamplerConfig
    prior_transform : callable, optional
        Maps ``[0,1]^d`` to physical space. This is applied only when returning the samples.
    """

    def __init__(self,
                 ndim: int,
                 log_density_func: Callable[[jnp.ndarray], jnp.ndarray],
                 config: JaxSamplerConfig,
                 prior_transform: Optional[Callable] = None) -> None:
        if ndim <= 0:
            raise ValueError("ndim must be positive")
        self.ndim = ndim
        self.log_density_func = log_density_func
        self.prior_transform = prior_transform
        self.cfg = config
        self.state: Optional[SamplerState] = None

        # Static shapes baked in as Python ints so XLA treats them as constants.
        self.P = config.n_proc_max
        self.N = config.n_max
        self.D = ndim
        self.A = min(config.alpha, config.n_max)
        self.H = config.n_max // config.gamma + 1

        # Build the jitted single-iteration step once.
        self._step = self._build_step()

    # ------------------------------------------------------------------
    # Initialization (happens on host)
    # ------------------------------------------------------------------
    def init_state(self,
                   seed_points: np.ndarray,
                   seed_logdens: np.ndarray,
                   init_cov: np.ndarray) -> SamplerState:
        """Build the initial :class:`SamplerState` from seed points.

        Parameters
        ----------
        seed_points : (n_seed, ndim)
            Initial mixture centers, e.g. the top LHS points. ``n_seed <= n_proc_max``.
        seed_logdens : (n_seed,)
        init_cov : (ndim, ndim) or (n_seed, ndim, ndim)
            Initial proposal covariance(s).
        """
        P, N, D, H = self.P, self.N, self.D, self.H
        n_seed = seed_points.shape[0]
        if n_seed > P:
            raise ValueError(f"n_seed={n_seed} exceeds n_proc_max={P}")

        seed_points = np.asarray(seed_points, dtype=np.float64)
        seed_logdens = np.asarray(seed_logdens, dtype=np.float64)
        init_cov = np.asarray(init_cov, dtype=np.float64)
        if init_cov.ndim == 2:
            init_cov = np.broadcast_to(init_cov, (n_seed, D, D))

        # Buffers per process, padded out to P with dead rows. Dead rows hold zeros for
        # points/wdeno and -inf for logdens; they are excluded via the `alive` mask.
        points = np.zeros((P, N, D))
        logdens = np.full((P, N), -np.inf)
        wdeno = np.zeros((P, N))
        inv_cov_hist = np.broadcast_to(np.eye(D), (P, H, D, D)).copy()
        norm_hist = np.ones((P, H))
        now_cov = np.broadcast_to(np.eye(D), (P, D, D)).copy()
        now_inv_cov = np.broadcast_to(np.eye(D), (P, D, D)).copy()
        now_norm = np.ones(P)
        now_mean = np.zeros((P, D))
        init_inv_cov = np.broadcast_to(np.eye(D), (P, D, D)).copy()
        init_norm = np.ones(P)
        max_logden = np.full(P, -np.inf)
        alive = np.zeros(P, dtype=bool)

        for j in range(n_seed):
            cov = init_cov[j]
            inv = np.linalg.inv(cov)
            det = np.linalg.det(cov)
            norm = 1.0 / np.sqrt((2 * np.pi) ** D * det)

            points[j, 0] = seed_points[j]
            logdens[j, 0] = seed_logdens[j]
            now_mean[j] = seed_points[j]
            now_cov[j] = cov
            now_inv_cov[j] = inv
            now_norm[j] = norm
            init_inv_cov[j] = inv
            init_norm[j] = norm
            inv_cov_hist[j, 0] = inv
            norm_hist[j, 0] = norm
            # Regularized initial denominator (matches the original seeding convention).
            wdeno[j, 0] = norm * np.exp(-D / 2.0)
            max_logden[j] = seed_logdens[j]
            alive[j] = True

        loglike_norm = float(np.max(seed_logdens))

        state = SamplerState(
            points=jnp.asarray(points),
            logdens=jnp.asarray(logdens),
            wdeno=jnp.asarray(wdeno),
            inv_cov_hist=jnp.asarray(inv_cov_hist),
            norm_hist=jnp.asarray(norm_hist),
            now_cov=jnp.asarray(now_cov),
            now_inv_cov=jnp.asarray(now_inv_cov),
            now_norm=jnp.asarray(now_norm),
            now_mean=jnp.asarray(now_mean),
            init_inv_cov=jnp.asarray(init_inv_cov),
            init_norm=jnp.asarray(init_norm),
            max_logden=jnp.asarray(max_logden),
            alive=jnp.asarray(alive),
            nf=jnp.asarray(1),   # default int (int64 under x64) -> matches slice index dtype
            it=jnp.asarray(0),
            loglike_norm=jnp.asarray(loglike_norm),
            key=jax.random.PRNGKey(self.cfg.seed),
        )
        self.state = state
        return state

    # ------------------------------------------------------------------
    # Per-process numerical primitives. These are batched later
    # ------------------------------------------------------------------
    def _importance_weights(self, logdens_j, wdeno_j, nf, lognorm):
        """Normalized importance weights for one process over filled slots [0, nf)."""
        N = self.N
        valid = jnp.arange(N) < nf
        w = jnp.exp(logdens_j - lognorm)
        wdeno_safe = jnp.where(wdeno_j > 0, wdeno_j, 1.0)
        w = jnp.where(wdeno_j > 0, w / wdeno_safe, w)
        w = jnp.where(valid & (w > 0), w, 0.0)
        s = jnp.sum(w)
        # If all weights vanish, fall back to a uniform distribution over valid slots.
        uniform = valid / jnp.maximum(jnp.sum(valid), 1)
        return jnp.where(s > 0, w / jnp.where(s > 0, s, 1.0), uniform)

    def _weighting_update_proc(self, 
                               points_j, 
                               wdeno_j, 
                               inv_cov_hist_j, 
                               norm_hist_j,
                               now_inv_j, 
                               now_norm_j, 
                               nf):
        """Incremental sliding-window denominator update for one process (3 terms).

        Maintains ``wdeno`` over the window ``[ind1, nf)`` of fixed length ``A`` using
        ``dynamic_slice``/``dynamic_update_slice`` and validity masks, so it is shape-
        static while remaining O(alpha) per iteration. The most-recent proposal lives at
        index ``nf - 1`` (it was appended on the previous iteration).
        """
        A, D, ccount = self.A, self.D, self.cfg.gamma
        ind1 = jnp.maximum(nf - A, 0)
        abs_idx = ind1 + jnp.arange(A)
        valid = abs_idx < nf
        is_new = abs_idx == (nf - 1)
        is_old = valid & (~is_new)

        # Extract the sliding denominator window [ind1, ind1 + A) with a static-shape
        # dynamic_slice (the offset is traced, the length is not).
        pts_win = jax.lax.dynamic_slice(points_j, (ind1, 0), (A, D))      # (A, D)
        wdeno_win = jax.lax.dynamic_slice(wdeno_j, (ind1,), (A,))         # (A,)
        hist_idx = abs_idx // ccount
        inv_win = inv_cov_hist_j[hist_idx]                               # (A, D, D)
        norm_win = norm_hist_j[hist_idx]                                 # (A,)
        x_new = points_j[nf - 1]                                         # (D,)

        # Term 1 (+): the new proposal contributes to every window point (single, now cov).
        d1 = pts_win - x_new
        m1 = jnp.einsum('ad,de,ae->a', d1, now_inv_j, d1)
        add1 = now_norm_j * jnp.exp(-0.5 * m1) * valid
        wdeno_win = wdeno_win + add1

        # Term 2 (+): the new point picks up all *old* proposals (cov history per mean).
        d2 = x_new[None, :] - pts_win
        m2 = jnp.einsum('ad,ade,ae->a', d2, inv_win, d2)
        add2 = jnp.sum(norm_win * jnp.exp(-0.5 * m2) * is_old)
        pos_new = (nf - 1) - ind1
        wdeno_win = wdeno_win.at[pos_new].add(add2)

        # Term 3 (-): subtract the proposal that just fell out of the window (index ind1-1).
        have_exp = ind1 > 0
        e = jnp.maximum(ind1 - 1, 0)
        x_exp = points_j[e]
        inv_e = inv_cov_hist_j[e // ccount]
        norm_e = norm_hist_j[e // ccount]
        d3 = pts_win - x_exp
        m3 = jnp.einsum('ad,de,ae->a', d3, inv_e, d3)
        sub = norm_e * jnp.exp(-0.5 * m3) * is_old
        wdeno_win = wdeno_win - jnp.where(have_exp, sub, 0.0)

        return jax.lax.dynamic_update_slice(wdeno_j, wdeno_win, (ind1,))

    def _cov_update_proc(self, key, points_j, probs_j, mean_j, init_inv_j, init_norm_j, nf):
        """Recompute the proposal covariance for one process (called every gamma steps).

        Weighted covariance -> optional Beta boundary correction (optax) -> OAS shrinkage
        -> jitter -> validity guard. Returns ``(cov, inv_cov, norm)``.
        """
        D = self.D
        cfg = self.cfg
        cov = jax_weighted_cov(points_j, aweights=probs_j)               # (D, D)
        # TODO: why does covariance not remain symmetric?
        cov = 0.5 * (cov + cov.T)                                        # symmetrize

        # Boundary fraction via MC, decides whether the Beta correction is worth applying.
        z = jax.random.multivariate_normal(key, jnp.zeros(D), cov, (cfg.integral_num,))
        shifted = z + mean_j
        oob_frac = jnp.mean(jnp.any((shifted < 0) | (shifted > 1), axis=1))
        use_b = jnp.logical_and(cfg.boundary_limiting and cfg.use_beta,
                                oob_frac >= cfg.use_beta_threshold)

        sign, logdet = jnp.linalg.slogdet(cov)
        cov_inv = jnp.linalg.inv(cov + jnp.eye(D) * 1e-12)
        diff = points_j - mean_j
        Adiff = jnp.einsum('jk,ik->ji', cov_inv, diff)
        beta = find_max_beta(diff, Adiff, probs_j, 1.0, logdet, z, mean_j, D,
                             n_steps=cfg.beta_steps, lr=cfg.beta_lr)
        cov_beta = cov / beta[:, None] / beta[None, :]
        cov_corr = jnp.where(use_b, cov_beta, cov)
        n_eff = jnp.sum(probs_j > 0)
        cov_shr, _ = oracle_approximating_shrinkage(cov_corr, n_eff)
        cov_final = jnp.where(use_b, cov_shr, cov_corr)
        cov_final = cov_final + jnp.eye(D) * cfg.cov_jitter
        cov_final = 0.5 * (cov_final + cov_final.T)

        sign2, logdet2 = jnp.linalg.slogdet(cov_final)
        bad = (sign2 <= 0) | (logdet2 < cfg.min_log_det_cov) | (~jnp.isfinite(logdet2))
        inv_final = jnp.linalg.inv(cov_final + jnp.eye(D) * 1e-12)
        norm_final = jnp.exp(-0.5 * logdet2) / jnp.sqrt((2 * jnp.pi) ** D)

        inv_out = jnp.where(bad, init_inv_j, inv_final)
        norm_out = jnp.where(bad, init_norm_j, norm_final)
        cov_out = jnp.where(bad, jnp.linalg.inv(init_inv_j), cov_final)
        return cov_out, inv_out, norm_out

    def _generate_proc(self, key, points_j, probs_j, now_cov_j, nf):
        """rejection sampling of one new point for one process. There's a limit included

        Draws ``trail_size`` mixture-center indices and Gaussian offsets at once, then
        takes the first in-bounds candidate (falling back to candidate 0). Does not depend on data
        shapes so can be jitted.
        """
        N, D, T = self.N, self.D, self.cfg.trail_size
        kc, ko = jax.random.split(key)

        valid = jnp.arange(N) < nf
        logits = jnp.where(valid & (probs_j > 0), jnp.log(jnp.where(probs_j > 0, probs_j, 1.0)), -jnp.inf)
        center_idx = jax.random.categorical(kc, logits, shape=(T,))      # (T,)
        centers = points_j[center_idx]                                   # (T, D)
        offsets = jax.random.multivariate_normal(ko, jnp.zeros(D), now_cov_j, (T,))
        candidates = centers + offsets                                   # (T, D)

        if self.cfg.boundary_limiting:
            in_bounds = jnp.all((candidates >= 0) & (candidates <= 1), axis=1)
            first = jnp.argmax(in_bounds)                                # 0 if none valid
            sel = candidates[first]
        else:
            sel = candidates[0]
        return sel

    # ------------------------------------------------------------------
    # The jitted single-iteration step
    # ------------------------------------------------------------------
    def _build_step(self):
        cfg = self.cfg
        gamma = cfg.gamma
        ccount = gamma
        log_density_func = self.log_density_func

        # vmapped per-process primitives
        imp_w = jax.vmap(self._importance_weights, in_axes=(0, 0, None, None))
        wt_upd = jax.vmap(self._weighting_update_proc, in_axes=(0, 0, 0, 0, 0, 0, None))
        wmean = jax.vmap(lambda pts, w: jnp.sum(w[:, None] * pts, axis=0), in_axes=(0, 0))
        cov_upd = jax.vmap(self._cov_update_proc, in_axes=(0, 0, 0, 0, 0, 0, None))
        gen = jax.vmap(self._generate_proc, in_axes=(0, 0, 0, 0, None))

        def step(state: SamplerState, _):
            key, k_cov, k_gen = jax.random.split(state.key, 3)
            nf = state.nf

            # Sliding-window denominator update 
            wdeno = jax.lax.cond(
                nf > 1,   # In the very first step this is bypassed 
                lambda: wt_upd(state.points, state.wdeno, state.inv_cov_hist,
                               state.norm_hist, state.now_inv_cov, state.now_norm, nf),
                lambda: state.wdeno,
            )

            # Importance weights and the per-step weighted mean.
            probs = imp_w(state.logdens, wdeno, nf, state.loglike_norm)   # (P, N)
            now_mean = wmean(state.points, probs)                         # (P, D)

            # Covariance update every gamma iterations (history slot nf // gamma).
            do_cov = ((state.it + 1) % gamma) == 0
            proc_keys = jax.random.split(k_cov, self.P)

            def with_cov():
                cov, inv, norm = cov_upd(proc_keys, state.points, probs, now_mean,
                                         state.init_inv_cov, state.init_norm, nf)
                slot = nf // ccount
                inv_hist = state.inv_cov_hist.at[:, slot].set(inv)
                norm_hist = state.norm_hist.at[:, slot].set(norm)
                return cov, inv, norm, inv_hist, norm_hist

            def keep_cov():
                return (state.now_cov, state.now_inv_cov, state.now_norm,
                        state.inv_cov_hist, state.norm_hist)

            now_cov, now_inv, now_norm, inv_hist, norm_hist = jax.lax.cond(
                do_cov, with_cov, keep_cov)

            # Generate one new point per process using rejection smapling with limit
            gen_keys = jax.random.split(k_gen, self.P)
            sel = gen(gen_keys, state.points, probs, now_cov, nf)         # (P, D)
            sel_logdens = log_density_func(sel)                          # (P,)

            # Append the new point at index nf (functional update, no in-placemutation).
            points = state.points.at[:, nf, :].set(sel)
            logdens = state.logdens.at[:, nf].set(sel_logdens)
            max_logden = jnp.maximum(state.max_logden, sel_logdens)

            # Dynamic overflow-guard normalizer from live processes only.
            alive_max = jnp.max(jnp.where(state.alive, max_logden, -jnp.inf))
            loglike_norm = jnp.maximum(state.loglike_norm, alive_max)

            new_state = state._replace(
                points=points, 
                logdens=logdens, 
                wdeno=wdeno,
                inv_cov_hist=inv_hist, 
                norm_hist=norm_hist,
                now_cov=now_cov, 
                now_inv_cov=now_inv, 
                now_norm=now_norm,
                now_mean=now_mean,
                max_logden=max_logden,
                nf=nf + 1, 
                it=state.it + 1,
                loglike_norm=loglike_norm, 
                key=key,
            )
            return new_state, None

        return jax.jit(lambda state, length: jax.lax.scan(step, state, None, length=length)[0],
                       static_argnums=(1,))

    # ------------------------------------------------------------------
    # Merging of processes happens on the host side
    # ------------------------------------------------------------------
    def _merge_host(self, state: SamplerState) -> SamplerState:
        """Distance-based merge: deactivate redundant processes via the ``alive`` mask.

        The quadratic Mahalanobis matrix is computed in JAX. The greedy assignment runs on
        host. Survivors keep their data in place, so indexing remains unaffected. 
        Merged processes are simply marked as dead. maybe not the cleanest option but otherwise 
        we lose advantage of compiled code.
        """
        means = np.asarray(state.now_mean)
        inv_covs = np.asarray(state.now_inv_cov)
        alive = np.asarray(state.alive)
        if alive.sum() <= 1:
            return state

        dist = np.asarray(pairwise_mahalanobis_matrix(jnp.asarray(means), jnp.asarray(inv_covs)))
        clusters = get_cluster_indices_from_matrix(
            dist, 
            dist=self.cfg.merge_dist, 
            alive=alive,
            priority=np.asarray(state.max_logden),
        )
        new_alive = np.zeros_like(alive)
        for c in clusters:
            survivor = c[0]            # highest-priority member (see clustering helper)
            if alive[survivor]:
                new_alive[survivor] = True
        return state._replace(alive=jnp.asarray(new_alive))

    # ------------------------------------------------------------------
    # Evidence (host) and sample extraction
    # ------------------------------------------------------------------
    def log_evidence(self, state: Optional[SamplerState] = None) -> float:
        """Importance-sampling log-evidence estimate.

        ``logZ = c - log(Nsum) + log(wsum * n_alive * alpha)``.

        The ``n_alive * alpha`` factor mirrors the original estimator: ``wdeno`` is the
        *unnormalized* mixture density (a sum over the ``alpha``-proposal window, i.e.
        ``alpha`` times a normalized mixture), and each of the ``n_alive`` processes
        estimates its own mode's contribution to Z, so the per-process sums combine with
        that prefactor. The last half of the chain is used (converged samples).
        """
        state = state if state is not None else self.state
        nf = int(state.nf)
        c = float(state.loglike_norm)
        alive = np.asarray(state.alive)
        logdens = np.asarray(state.logdens)
        wdeno = np.asarray(state.wdeno)
        n_alive = int(alive.sum())

        lo = max(nf // 2, 1)              # last half (drop burn-in)
        hi = max(nf - 1, lo + 1)          # drop the not-yet-weighted final point
        wsum = 0.0
        nsum = 0
        for j in np.where(alive)[0]:
            d = wdeno[j, lo:hi]
            ld = logdens[j, lo:hi]
            good = (d > 0) & np.isfinite(ld)
            wsum += np.sum(np.exp(ld[good] - c) / d[good])
            nsum += int(good.sum())
        if wsum <= 0 or nsum == 0:
            return float("nan")
        return c - np.log(nsum) + np.log(wsum * n_alive * self.A)

    def get_samples_with_weights(self, state: Optional[SamplerState] = None
                                 ) -> Tuple[np.ndarray, np.ndarray]:
        """Return concatenated (samples, importance-weights) over all live processes."""
        state = state if state is not None else self.state
        nf = int(state.nf)
        c = float(state.loglike_norm)
        alive = np.asarray(state.alive)
        points = np.asarray(state.points)
        logdens = np.asarray(state.logdens)
        wdeno = np.asarray(state.wdeno)

        all_pts, all_w = [], []
        for j in np.where(alive)[0]:
            d = wdeno[j, :nf]
            ld = logdens[j, :nf]
            good = (d > 0) & np.isfinite(ld)
            pts = points[j, :nf][good]
            w = np.exp(ld[good] - c) / d[good]
            if self.prior_transform is not None:
                pts = np.asarray(self.prior_transform(pts))
            all_pts.append(pts)
            all_w.append(w)
        if not all_pts:
            return np.empty((0, self.D)), np.empty((0,))
        return np.concatenate(all_pts), np.concatenate(all_w)

    # ------------------------------------------------------------------
    # Driver function: jitted blocks + host merging in between
    # ------------------------------------------------------------------
    def run(self,
            num_iterations: int,
            merge_every: Optional[int] = None,
            verbose: bool = True) -> SamplerState:
        """Run the sampler.

        The iteration budget is split into blocks of ``merge_every`` steps. Each block runs
        as a single jitted ``lax.scan`` on the GPU.  Between blocks we do
        clustering/merging on the host and optional logging. ``merge_every`` defaults to
        ``gamma`` so merging is aligned with covariance updates.
        """
        if self.state is None:
            raise RuntimeError("Call init_state(...) before run(...)")
        if num_iterations <= 0:
            raise ValueError("num_iterations must be positive")
        merge_every = merge_every or self.cfg.gamma

        # The buffer must hold the seed point plus one point per iteration.
        if int(self.state.nf) + num_iterations > self.N:
            raise ValueError(
                f"n_max={self.N} too small for nf={int(self.state.nf)} + "
                f"{num_iterations} iterations")

        state = self.state
        remaining = num_iterations
        while remaining > 0:
            block = int(min(merge_every, remaining))
            state = self._step(state, block)
            state = self._merge_host(state)
            remaining -= block
            if verbose:
                logZ = self.log_evidence(state)
                n_alive = int(np.asarray(state.alive).sum())
                logger.info("iter %d | n_proc %d | logZ %.5f",
                            int(state.it), n_alive, logZ)
                print(f"iter {int(state.it)} | n_proc {n_alive} | logZ {logZ:.5f}")

        self.state = state
        return state
