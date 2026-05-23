"""
M_act_3: direct corner-state extraction.

For each sampled gauge config U, compute the boundary-to-boundary propagator
G^bdy[U] and the Fock-state-pair Slater determinant weights
⟨Ψ_j|e^{-βh[U]}|Ψ_i⟩ for all (Ψ_i, Ψ_j) pairs with equal particle number.

For free fermions on a background gauge config:
    ⟨Ψ_j|e^{-βh[U]}|Ψ_i⟩ = det[G^bdy[U][b_α, a_β]]_{α,β=1..n}
where {a_β}, {b_α} are the occupied sites in Ψ_i, Ψ_j respectively.

Structural sanity: Σ_Ψ ⟨Ψ|e^{-βh}|Ψ⟩ = det(I + G^bdy) (grand-canonical partition fn).
"""

from __future__ import annotations
import numpy as np
from itertools import combinations
from typing import Iterator

from action_z2_staggered import LatticeGeometry, Z2GaugeConfig, compute_G_bdy


def bitcount(n: int) -> int:
    return bin(n).count('1')


def occupied_sites(psi: int, V3: int) -> tuple:
    """Return ordered tuple of occupied site indices for bit-string ψ (LSB = site 0)."""
    return tuple(k for k in range(V3) if (psi >> k) & 1)


def fock_pair_weights(G_bdy: np.ndarray) -> dict:
    """Compute |⟨Ψ_j|e^{-βh}|Ψ_i⟩| for all Fock pairs with equal particle number.

    Returns a dict: keys are (Ψ_i, Ψ_j) bit-string tuples, values are
    (amplitude, |amplitude|).
    """
    V3 = G_bdy.shape[0]
    D = 1 << V3  # 2^V3 Fock states
    weights = {}
    for psi_i in range(D):
        n_i = bitcount(psi_i)
        sites_i = occupied_sites(psi_i, V3)
        for psi_j in range(D):
            n_j = bitcount(psi_j)
            if n_j != n_i:
                continue
            sites_j = occupied_sites(psi_j, V3)
            if n_i == 0:
                amp = 1.0 + 0j
            else:
                # Slater-det amplitude: det[G_bdy[b, a]] over occupied sites
                sub = G_bdy[np.ix_(sites_j, sites_i)]
                amp = np.linalg.det(sub)
            weights[(psi_i, psi_j)] = (amp, abs(amp))
    return weights


def fock_pair_weights_array(G_bdy: np.ndarray):
    """Same as fock_pair_weights but returns parallel arrays.

    Returns: pairs (N×2 int), amps (N complex), |amps| (N float).
    """
    weights = fock_pair_weights(G_bdy)
    pairs = np.array(list(weights.keys()), dtype=np.int64)
    amps = np.array([w[0] for w in weights.values()], dtype=complex)
    abs_amps = np.array([w[1] for w in weights.values()], dtype=float)
    return pairs, amps, abs_amps


def sample_corner_pairs_categorical(G_bdy: np.ndarray, N_samples: int,
                                    rng: np.random.Generator) -> tuple:
    """Sample (Ψ_i, Ψ_j) pairs categorically from |⟨Ψ_j|e^{-βh}|Ψ_i⟩|.

    Returns: pairs (N_samples × 2), signs (N_samples).
    """
    pairs_all, amps_all, abs_all = fock_pair_weights_array(G_bdy)
    total = abs_all.sum()
    if total <= 0:
        raise RuntimeError("Total |Slater det| weight is zero.")
    prob = abs_all / total
    prob = prob / prob.sum()
    idx = rng.choice(len(pairs_all), size=N_samples, p=prob)
    pairs = pairs_all[idx]
    signs = np.sign(amps_all[idx].real).astype(int)
    signs[signs == 0] = 1
    return pairs, signs, total


def partition_function_consistency_check(G_bdy: np.ndarray) -> tuple[float, float]:
    """Verify Σ_Ψ ⟨Ψ|e^{-βh}|Ψ⟩ == det(I + G^bdy).

    Returns (direct_sum, det_formula) — should agree.
    """
    V3 = G_bdy.shape[0]
    # Direct sum over diagonal Fock states
    direct_sum = 0j
    for psi in range(1 << V3):
        sites = occupied_sites(psi, V3)
        if not sites:
            direct_sum += 1.0
        else:
            sub = G_bdy[np.ix_(sites, sites)]
            direct_sum += np.linalg.det(sub)
    det_formula = np.linalg.det(np.eye(V3) + G_bdy)
    return complex(direct_sum), complex(det_formula)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def _self_test():
    """Sanity check on a random Z₂ config."""
    geom = LatticeGeometry(Lx=2, Ly=2, N_E=4, m=0.5)
    rng = np.random.default_rng(42)
    U = Z2GaugeConfig.random(geom, rng)
    G_bdy = compute_G_bdy(geom, U)
    print(f"G^bdy shape: {G_bdy.shape}, |G^bdy| ∈ "
          f"[{np.abs(G_bdy).min():.4f}, {np.abs(G_bdy).max():.4f}]")

    pairs, amps, abs_amps = fock_pair_weights_array(G_bdy)
    print(f"\nNumber of (Ψ_i, Ψ_j) pairs with N_i = N_j: {len(pairs)} "
          f"(expected: 1 + 16 + 36 + 16 + 1 = 70)")
    assert len(pairs) == 70

    # Distribution by particle number
    print(f"\nBreakdown by particle number:")
    for n in range(5):
        idx = [k for k in range(len(pairs))
               if bitcount(int(pairs[k, 0])) == n]
        if idx:
            tot_weight = abs_amps[idx].sum()
            print(f"  N={n}: {len(idx)} pairs, total |weight| = {tot_weight:.4f}")

    # Partition-function consistency
    direct_sum, det_formula = partition_function_consistency_check(G_bdy)
    print(f"\nPartition function check:")
    print(f"  Σ_Ψ ⟨Ψ|e^(−βh)|Ψ⟩      = {direct_sum:.6f}")
    print(f"  det(I + G^bdy)         = {det_formula:.6f}")
    print(f"  |difference|           = {abs(direct_sum - det_formula):.3e}")
    assert abs(direct_sum - det_formula) < 1e-9, "Partition function consistency failed!"

    # Sample a few corner pairs
    pairs_samp, signs_samp, total = sample_corner_pairs_categorical(
        G_bdy, N_samples=1000, rng=rng)
    print(f"\nSampled 1000 corner pairs; total |Slater| weight = {total:.4f}")
    print(f"  Negative-sign fraction: {np.mean(signs_samp < 0):.3f}")
    print(f"  Unique Ψ_i values sampled: {len(np.unique(pairs_samp[:, 0]))}")
    print(f"  Unique Ψ_j values sampled: {len(np.unique(pairs_samp[:, 1]))}")

    # Sanity: marginal P(Ψ_i = ψ) for ψ with N=0 should be small (only one such pair)
    n0_count = np.sum(pairs_samp[:, 0] == 0)
    print(f"  Samples with Ψ_i = vacuum (0): {n0_count}/1000")


if __name__ == '__main__':
    _self_test()
