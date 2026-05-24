"""
epoq_sparse_ed.py — sparse Krylov ED for V_3 ≥ 6 systems where dense
eigh / expm fail by RAM or wall-clock budget.

Provides:
  compute_C_t_hutchinson(H_sparse, n0_sparse, beta, times, n_random, seed)
    → (C, sigma_C, raw_diagnostics)

Math
====
Target: C(t) = Tr[ρ_β · O_t] / Tr[ρ_β]
where ρ_β = e^{-βH}, O_t = U(t)† · n_0 · U(t) · n_0,  U(t) = e^{-iHt}.

By cyclic invariance of the trace:
    Tr[ρ_β · O_t] = Tr[e^{-βH/2} · O_t · e^{-βH/2}]
                  = 𝔼[w† O_t w]    for w = e^{-βH/2} v, v ~ CN(0, 1_dim)
    Tr[ρ_β]       = Tr[e^{-βH/2} · e^{-βH/2}]
                  = 𝔼[w† w]

per random v:
    w = e^{-βH/2} v       (one Krylov expm_multiply on H_sparse, Euclidean)
    n0_w = n_0 · w        (one diagonal apply)
    per t:
      a = U(t) · w
      c = U(t) · n0_w     (a, c sharing the same Krylov subspace via 2-col RHS)
      d = n_0 · c
      ⟨a | d⟩ = a† d    ←   per-sample numerator at time t
    ⟨w | w⟩            ←   per-sample denominator

The symmetric placement (e^{-βH/2} on both sides) is the lower-variance
form; it's standard for stochastic thermal traces.

Stochastic vectors
==================
Complex Gaussian: v_i = (a_i + i b_i) / √2 with a_i, b_i ~ N(0, 1).
Then 𝔼[|v_i|²] = 1, 𝔼[v_i v_j*] = δ_ij, 𝔼[v_i v_j] = 0 — the standard
conditions for an unbiased Hutchinson estimator over any A:
    Tr[A] = 𝔼[v† A v].

Error bars
==========
Per-sample numerator/denominator: SEM = s / √N where s is the
sample standard deviation.  For the ratio C = N̄ / D̄ we use the
delta-method propagation:
    Var[C] ≈ Var[N̄]/D̄² + N̄²·Var[D̄]/D̄⁴ − 2·N̄·Cov[N̄,D̄]/D̄³

The covariance term matters because per-sample N and D both come
from the same w, so they're correlated.

Cost
====
Per v: 1 expm_multiply (Euclidean, real β·H) + 2 expm_multiply
(Minkowski, imaginary i·t·H) per nonzero t, with the 2 Minkowski
calls sharing a Krylov basis via expm_multiply's batched-RHS interface.
At V_3=6 dim 8192, ~7 sec for 64 samples × 3 times.
At V_3=8 3D dim 1M (NQ=20), ~tens of minutes.  Affordable.
"""
from __future__ import annotations
import numpy as np
from scipy.sparse.linalg import expm_multiply


def _complex_gaussian(rng, dim):
    """Complex Gaussian unit-variance random vector."""
    return ((rng.standard_normal(dim) + 1j * rng.standard_normal(dim))
            / np.sqrt(2)).astype(complex)


def compute_C_t_hutchinson(
    H_sparse,
    n0_sparse,
    beta: float,
    times,
    n_random: int = 64,
    seed: int = 0,
    progress: bool = True,
    progress_every: int = 8,
):
    """C(t) = Re Tr[ρ_β·O_t] / Tr[ρ_β]  via sparse Krylov + Hutchinson.

    Args:
        H_sparse:   sparse Hermitian H, scipy CSR (dim, dim) complex.
        n0_sparse:  sparse diagonal n_0 operator (or general sparse Hermitian).
        beta:       inverse temperature.
        times:      iterable of Minkowski times t.
        n_random:   number of stochastic samples.
        seed:       RNG seed.
        progress:   print per-sample timing.
        progress_every:  print every k samples.

    Returns:
        C:        {t: float}  point estimate
        sigma_C:  {t: float}  delta-method ±1σ error bar
        raw:      diagnostic dict
                  - per_sample_num[t] = array of length n_random
                  - per_sample_denom  = array of length n_random
                  - mean_num[t], sem_num[t], mean_denom, sem_denom
                  - cov_num_denom[t]
    """
    import time as _time
    rng = np.random.default_rng(seed)
    dim = H_sparse.shape[0]
    times_list = list(times)

    per_num = {t: np.zeros(n_random, dtype=float) for t in times_list}
    per_denom = np.zeros(n_random, dtype=float)

    t_loop_start = _time.time()
    for k in range(n_random):
        t0 = _time.time()
        v = _complex_gaussian(rng, dim)
        w = expm_multiply(-(beta * 0.5) * H_sparse, v)
        per_denom[k] = float(np.vdot(w, w).real)

        n0_w = n0_sparse @ w

        for t in times_list:
            if t == 0.0:
                # U(0) = I → a = w, c = n0_w, d = n_0 · n0_w = n0_w (projector)
                num_k = float(np.vdot(w, n0_w).real)
            else:
                B = np.column_stack([w, n0_w])
                AB = expm_multiply(-1j * t * H_sparse, B)
                a = AB[:, 0]
                c = AB[:, 1]
                d = n0_sparse @ c
                num_k = float(np.vdot(a, d).real)
            per_num[t][k] = num_k

        if progress and ((k + 1) % progress_every == 0 or k == n_random - 1):
            elapsed = _time.time() - t_loop_start
            per_sample = elapsed / (k + 1)
            remaining = per_sample * (n_random - k - 1)
            print(f"    Hutchinson {k+1:>3d}/{n_random}  "
                  f"({_time.time()-t0:.1f}s/sample, ETA {remaining:.0f}s)",
                  flush=True)

    mean_denom = float(per_denom.mean())
    sem_denom = float(per_denom.std(ddof=1) / np.sqrt(n_random))

    C = {}
    sigma_C = {}
    means_num = {}
    sems_num = {}
    covs_num_denom = {}
    for t in times_list:
        nm = per_num[t]
        means_num[t] = float(nm.mean())
        sems_num[t] = float(nm.std(ddof=1) / np.sqrt(n_random))
        # SEM of the covariance estimator equivalently
        cov_nd = float(np.cov(nm, per_denom, ddof=1)[0, 1] / n_random)
        covs_num_denom[t] = cov_nd

        N = means_num[t]; D = mean_denom
        var_C = (sems_num[t] ** 2 / D ** 2
                 + N ** 2 * sem_denom ** 2 / D ** 4
                 - 2.0 * N * cov_nd / D ** 3)
        C[t] = N / D
        sigma_C[t] = float(np.sqrt(max(var_C, 0.0)))

    raw = {
        'per_sample_num': per_num,
        'per_sample_denom': per_denom,
        'mean_num': means_num,
        'sem_num': sems_num,
        'mean_denom': mean_denom,
        'sem_denom': sem_denom,
        'cov_num_denom': covs_num_denom,
    }
    return C, sigma_C, raw
