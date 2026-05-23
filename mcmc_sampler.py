"""
Generic MCMC sampler for the EρOQ corner-state distribution
P(i, j) ∝ |⟨j| e^{-βH} |i⟩|

Metropolis-within-Gibbs:
  - Outer Metropolis step: propose i' by k-bit flip on i, accept with prob
    min(1, |ψ_{i'}[j]| / |ψ_i[j]|), where ψ_i = e^{-βH}|i⟩ via Trotter.
  - Inner Gibbs step: redraw j from categorical P(j|i) ∝ |ψ_i[j]| (exact).
  - Minimum caching: only the current ψ_i (recomputed on each Metropolis
    proposal accept; not recomputed on j-Gibbs steps).

Autocorrelation diagnostics: integrated autocorrelation time τ_int via the
standard windowed-sum estimator (Sokal); split-R̂ across chains; effective
sample size.

Sign tracking: sample by |G[j,i]| (Henry's choice); sign captured separately.

Designed to be NQ-agnostic; works for both Method 2 (8-qubit penalized H)
and Method 3 (4-qubit Z-C-reduced H_prime) at 2×2, and will scale to 2×4
without code change.
"""
from __future__ import annotations
import time
import numpy as np
from typing import List, Tuple, Callable, Dict
from dataclasses import dataclass, field

Term = Tuple[complex, List[Tuple[int, str]]]


# ---------------------------------------------------------------------------
# Generic Pauli-term application (q0 = LSB convention, matches cs/qs modules)
# ---------------------------------------------------------------------------
def make_pauli_machinery(NQ: int):
    """Build cached arrays for fast Pauli-term application at given NQ."""
    DIM = 1 << NQ
    ALL = np.arange(DIM, dtype=np.int64)
    # Vectorized popcount: bit-by-bit sum (cheap for small NQ)
    POP = np.zeros(DIM, dtype=np.int64)
    for q in range(NQ):
        POP += (ALL >> q) & 1
    return DIM, ALL, POP


def apply_pauli_term(state, term, ALL, POP):
    coef, factors = term
    if not factors:
        return coef * state
    X_mask = Z_mask = 0
    n_y = 0
    for q, ax in factors:
        if ax == 'X':
            X_mask |= 1 << q
        elif ax == 'Z':
            Z_mask |= 1 << q
        elif ax == 'Y':
            X_mask |= 1 << q
            Z_mask |= 1 << q
            n_y += 1
    phase_y = (-1j) ** n_y
    targets = ALL ^ X_mask
    z_phase = (1 - 2 * (POP[ALL & Z_mask] & 1)).astype(np.complex128)
    out = np.zeros_like(state)
    out[targets] = coef * phase_y * z_phase * state
    return out


def apply_exp_pauli_term(state, term, tau, imag, ALL, POP):
    coef, factors = term
    if not factors:
        return state * (np.exp(-tau * coef.real) if imag else np.exp(-1j * tau * coef.real))
    a = tau * coef.real
    Pstate = apply_pauli_term(state, (1.0+0j, factors), ALL, POP)
    if imag:
        return np.cosh(a) * state - np.sinh(a) * Pstate
    else:
        return np.cos(a) * state - 1j * np.sin(a) * Pstate


def evolve_state(state, terms, total_time, n_steps, imag, ALL, POP):
    out = state
    dt = total_time / n_steps
    for _ in range(n_steps):
        for term in terms:
            out = apply_exp_pauli_term(out, term, dt, imag, ALL, POP)
    return out


# ---------------------------------------------------------------------------
# Single MCMC chain
# ---------------------------------------------------------------------------
@dataclass
class ChainResult:
    pairs: np.ndarray         # shape (N, 2): (i, j)
    signs: np.ndarray         # shape (N,)
    accept_rate: float
    n_warmup: int
    n_samples: int


def run_chain(
    H_terms: List[Term],
    NQ: int,
    beta: float,
    n_trotter_imag: int,
    n_outer: int,
    n_warmup: int,
    k_bits: int,    # retained for API compat but unused — heat-bath on single bit per step
    seed: int,
    record_psi_norm: bool = False,
) -> ChainResult:
    """Run a single heat-bath-within-Gibbs chain.

    Per outer step:
      1. Pick a random bit b ∈ {0, ..., NQ-1}.
      2. Heat-bath update on bit b: compute ψ_{i with b cleared} and
         ψ_{i with b set}, sample bit b's new value from
         P(bit=v | rest of i, j) ∝ |ψ_{i_v}[j]|.
         (One Trotter run per step, since one of the two psi's is cached.)
      3. Gibbs j-sample: redraw j from |ψ_{i_new}[j]|.

    Returns ChainResult with the post-warmup (i, j, sign) samples.
    `accept_rate` here is the "bit-switch rate" — fraction of steps where
    bit b actually changed value (= true mixing rate, lower-bounded by
    heat-bath's exactness).
    """
    DIM, ALL, POP = make_pauli_machinery(NQ)
    rng = np.random.default_rng(seed)

    def compute_psi(i: int):
        e_i = np.zeros(DIM, dtype=np.complex128)
        e_i[i] = 1.0
        return evolve_state(e_i, H_terms, beta, n_trotter_imag, imag=True,
                             ALL=ALL, POP=POP)

    def sample_j_from_psi(psi: np.ndarray) -> tuple[int, int]:
        abs_psi = np.abs(psi)
        Z_psi = abs_psi.sum()
        p = abs_psi / Z_psi
        p = p / p.sum()
        j = int(rng.choice(DIM, p=p))
        s = 1 if psi[j].real >= 0 else -1
        return j, s

    i = int(rng.integers(DIM))
    psi = compute_psi(i)
    j, sgn = sample_j_from_psi(psi)

    pairs_buf = np.empty((n_outer, 2), dtype=np.int64)
    signs_buf = np.empty(n_outer, dtype=np.int8)
    n_switch_post = 0
    n_post = 0

    for step in range(n_warmup + n_outer):
        # --- joint (bit_b, j) heat-bath SWEEP over all bits ---
        # For each bit b, marginalize over j: P(bit b = v) ∝ ||ψ_{i with b=v}||_1.
        # This removes the lock-in where j (concentrated near current i) makes
        # every bit flip on i look like a "far" move.
        bit_order = rng.permutation(NQ)
        for b in bit_order:
            b = int(b)
            i_other = i ^ (1 << b)
            psi_other = compute_psi(i_other)
            w_cur = np.abs(psi).sum()
            w_other = np.abs(psi_other).sum()
            total = w_cur + w_other
            if total <= 0:
                continue
            p_other = w_other / total
            if rng.random() < p_other:
                i = i_other
                psi = psi_other
                if step >= n_warmup:
                    n_switch_post += 1
            if step >= n_warmup:
                n_post += 1
            # j-Gibbs after each bit update — now j conditional on new i
            j, sgn = sample_j_from_psi(psi)

        if step >= n_warmup:
            k = step - n_warmup
            pairs_buf[k] = (i, j)
            signs_buf[k] = sgn

    switch_rate = n_switch_post / max(n_post, 1)
    return ChainResult(pairs=pairs_buf, signs=signs_buf,
                       accept_rate=switch_rate, n_warmup=n_warmup,
                       n_samples=n_outer)


# ---------------------------------------------------------------------------
# Parallel chains + diagnostics
# ---------------------------------------------------------------------------
@dataclass
class MCMCResult:
    pairs: np.ndarray        # concatenated, shape (N_chains × N_per_chain, 2)
    signs: np.ndarray
    chain_pairs: List[np.ndarray]
    chain_signs: List[np.ndarray]
    accept_rate: float       # avg across chains
    n_chains: int
    n_per_chain: int
    walltime: float


def run_mcmc(
    H_terms: List[Term],
    NQ: int,
    beta: float,
    n_trotter_imag: int,
    n_outer: int,
    n_warmup: int = 200,
    k_bits: int = 1,
    n_chains: int = 4,
    base_seed: int = 2026,
) -> MCMCResult:
    """Run n_chains independent Metropolis-within-Gibbs chains sequentially.

    (Sequential is simpler and at 2×2 / 2×4 scale fast enough; multiprocessing
    can be added if a chain takes minutes.)
    """
    t0 = time.time()
    chain_pairs, chain_signs = [], []
    accs = []
    for c in range(n_chains):
        result = run_chain(H_terms, NQ, beta, n_trotter_imag,
                           n_outer, n_warmup, k_bits, seed=base_seed + c)
        chain_pairs.append(result.pairs)
        chain_signs.append(result.signs)
        accs.append(result.accept_rate)
    pairs = np.concatenate(chain_pairs, axis=0)
    signs = np.concatenate(chain_signs, axis=0)
    return MCMCResult(
        pairs=pairs, signs=signs,
        chain_pairs=chain_pairs, chain_signs=chain_signs,
        accept_rate=float(np.mean(accs)),
        n_chains=n_chains, n_per_chain=n_outer,
        walltime=time.time() - t0,
    )


# ---------------------------------------------------------------------------
# Autocorrelation diagnostics
# ---------------------------------------------------------------------------
def autocorr(x: np.ndarray, max_lag: int = None) -> np.ndarray:
    """Sample autocorrelation function (un-biased estimator, normalized)."""
    x = x - x.mean()
    n = len(x)
    if max_lag is None:
        max_lag = n // 4
    # FFT-based, fast
    fft_len = 1 << (2 * n - 1).bit_length()
    fx = np.fft.fft(x, fft_len)
    acf = np.fft.ifft(fx * fx.conj()).real[:max_lag]
    acf /= acf[0] if acf[0] != 0 else 1
    return acf


def tau_int_sokal(x: np.ndarray, c_window: float = 5.0) -> float:
    """Integrated autocorrelation time τ_int via Sokal's automatic windowing.

    Returns the smallest M such that M ≥ c × τ(M), where
        τ(M) = 1 + 2 Σ_{t=1..M} ρ(t).
    """
    rho = autocorr(x, max_lag=min(len(x) // 4, 5000))
    tau = 1.0
    for M in range(1, len(rho)):
        tau += 2 * rho[M]
        if M >= c_window * tau:
            return tau
    return tau


def split_rhat(chains: List[np.ndarray]) -> float:
    """Split-R̂ across chains for a scalar observable trace."""
    splits = []
    for c in chains:
        n = len(c)
        splits.append(c[: n // 2])
        splits.append(c[n // 2:])
    splits = np.array(splits)
    m, n = splits.shape
    chain_means = splits.mean(axis=1)
    grand_mean = chain_means.mean()
    B = n * np.var(chain_means, ddof=1)
    W = np.mean(np.var(splits, axis=1, ddof=1))
    if W <= 0:
        return float('nan')
    var_hat = ((n - 1) / n) * W + B / n
    return float(np.sqrt(var_hat / W))


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def _self_test():
    """Validate MCMC against enumeration at 2x2.

    Run MCMC on the 8-qubit 2x2 Z2+staggered H and compare empirical
    P(i, j) histogram and 1-marginal P(i) to the exact distribution.
    """
    import sys
    sys.path.insert(0, '/home/hlamm/Desktop/QC/logdet/m1_toy')
    import epoq_classical_sampler as cs

    beta = 0.5
    n_trotter_imag = 50
    NQ = 8

    # Exact distribution
    H_terms = cs.build_pauli_terms()
    out_enum = cs.sample_corner_pairs(beta, 100, np.random.default_rng(0),
                                       n_trotter=n_trotter_imag)
    G = out_enum['G']
    abs_G = np.abs(G)
    P_exact = abs_G / abs_G.sum()
    P_i_exact = abs_G.sum(axis=0) / abs_G.sum()    # marginal P(i) = sum_j |G[j,i]| / Z_abs

    # MCMC — heat-bath sweep is NQ× more expensive per outer step than single bit,
    # so reduce n_outer × n_warmup accordingly
    mcmc = run_mcmc(H_terms, NQ=NQ, beta=beta, n_trotter_imag=n_trotter_imag,
                    n_outer=500, n_warmup=100, k_bits=1, n_chains=4)
    pairs = mcmc.pairs
    N = len(pairs)
    P_i_mcmc = np.bincount(pairs[:, 0], minlength=256) / N

    print(f"MCMC self-test on 2×2:")
    print(f"  N_samples (post-warmup, all chains) = {N}")
    print(f"  Avg acceptance: {mcmc.accept_rate:.3f}")
    print(f"  Wall-clock: {mcmc.walltime:.1f}s")

    # Compare marginals
    l1_marginal = np.abs(P_i_mcmc - P_i_exact).sum()
    print(f"  L1(P_i mcmc - P_i exact) = {l1_marginal:.4f}")
    print(f"  Expected ~sqrt(256/N) ≈ {np.sqrt(256/N):.4f} for uniform-ish target")

    # ESS diagnostic on a scalar observable: i (just take i mod something)
    # Use the marginal bit4(i) as a scalar
    scalar_obs = ((pairs[:, 0] >> 4) & 1).astype(float)
    chain_obs = [((p[:, 0] >> 4) & 1).astype(float) for p in mcmc.chain_pairs]
    tau_int = tau_int_sokal(scalar_obs)
    n_eff = N / (2 * tau_int + 1)
    rhat = split_rhat(chain_obs)
    print(f"  τ_int(bit4(i)) = {tau_int:.2f}")
    print(f"  N_eff = {n_eff:.0f}")
    print(f"  split-R̂ = {rhat:.4f}  (should be < 1.05)")

    # Pair-level chi^2 test (sparse comparison)
    pair_count = np.zeros((256, 256))
    for k in range(N):
        pair_count[pairs[k, 1], pairs[k, 0]] += 1
    pair_freq = pair_count / N
    l1_pair = np.abs(pair_freq - P_exact).sum()
    print(f"  L1(P(i,j) mcmc - P(i,j) exact) = {l1_pair:.4f}")

    return mcmc


if __name__ == '__main__':
    _self_test()
