"""
action_corner_direct_v3: corner-state extraction using the forward-time
Suzuki-Trotter W matrix.

Drop-in replacement for `action_corner_direct_v2.py`.  Only difference:
the weight matrix `W[Ψ_top, Ψ_bot] = ⟨Ψ_top|T̂_F^{N_E-1}|Ψ_bot⟩` now comes
from `transfer_matrix_kbc_trotterized.compute_combined_weight_trotter`
(Trotter T̂_F = expm(-a_τ · H_KS)) rather than `transfer_matrix_kbc.
compute_combined_weight` (Cauchy-Binet over the central-difference M-K
transfer matrix).

Why: Phase 8 showed the M-K central-diff T̂_F has H_lat ≠ H_QC at a_τ=1.
Phase 9 showed forward-time Trotter T̂_F matches H_QC exactly per gauge
eigenstate.  For the methodology paper, we want the corner-state weights
to be consistent with the staggered KS Hamiltonian observable that the
QC side simulates.

Interface (identical to v2):
  - fock_pair_weights(geom, U, a_tau, m_obs) -> dict
  - fock_pair_weights_array(geom, U, a_tau, m_obs) -> (pairs, amps, abs_amps)
  - sample_corner_pairs_categorical(geom, U, N_samples, rng, a_tau, m_obs)
       -> (pairs, signs, total)
  - sample_corner_pairs_stratified(geom, U, n_per_sector, rng, a_tau, m_obs)
       -> (pairs, signs, sector_w, total)
  - partition_function_consistency_check(geom, U, a_tau, m_obs)
       -> (Σ W, ?det reference)

Convention (matches v2): psi as int, bit k = mode k (= spatial site index
in fock_basis_labels lex order with tuple position a → bit V_3-1-a of psi).
"""
from __future__ import annotations
import numpy as np
from itertools import product as _product

from action_z2_staggered import LatticeGeometry, Z2GaugeConfig
from transfer_matrix_kbc_trotterized import compute_combined_weight_trotter


# Encoding helpers — same as v2.
def bitcount(n: int) -> int:
    return bin(n).count('1')


def occupied_sites(psi: int, V3: int) -> tuple:
    return tuple(k for k in range(V3) if (psi >> k) & 1)


def tuple_to_psi(n: tuple[int, ...]) -> int:
    """fock_basis_labels tuple (n_0, n_1, ..., n_{V_3-1}) → int psi with bit k = n_k."""
    return sum(int(nk) << k for k, nk in enumerate(n))


def psi_to_tuple(psi: int, V3: int) -> tuple[int, ...]:
    return tuple((psi >> k) & 1 for k in range(V3))


def _fock_index_to_psi_map(V3: int) -> dict:
    """fock_basis_labels enumeration index → psi int."""
    out = {}
    for i, n in enumerate(_product((0, 1), repeat=V3)):
        out[i] = tuple_to_psi(n)
    return out


# ---------------------------------------------------------------------------
# Trotter-W corner extraction
# ---------------------------------------------------------------------------
def _build_W_trotter(geom: LatticeGeometry, U: Z2GaugeConfig,
                     a_tau: float, m_obs: float,
                     g_hop: float = 0.5,
                     K_E: float = 0.0, K_M: float = 0.0,
                     order: int = 1):
    """Wrap compute_combined_weight_trotter with the conventions used by
    action MC + corner-state sampling:
      - K_E = K_M = 0 inside the Trotter weight (plaquette factors handled
        by the gauge MC measure e^{-S_g}; including them here would double-
        count).
      - m_obs is the Hamiltonian mass (NOT the action's a_τ-scaled mass).
      - a_tau is the temporal lattice spacing.
      - order=2 selects the palindrome (Hermitian PD) W construction; see
        feedback_W_matrix_nonhermitian memory note.
    """
    W, info = compute_combined_weight_trotter(
        geom, U, a_tau=a_tau,
        K_E=K_E, K_M=K_M, g_hop=g_hop, m_obs=m_obs, order=order,
    )
    return W, info


def fock_pair_weights(geom: LatticeGeometry, U: Z2GaugeConfig,
                      a_tau: float = 1.0, m_obs: float = None,
                      g_hop: float = 0.5, order: int = 1) -> dict:
    """Build dict {(psi_i, psi_j): (amp, |amp|)} for all Fock pairs at same N.

    amp = W[psi_i, psi_j] from compute_combined_weight_trotter (with given order).
    psi_i = "top" (t=0 corner state), psi_j = "bot" (t=N_E−1 corner state).
    """
    V3 = geom.V_3
    if m_obs is None:
        m_obs = geom.m            # default to action mass (may be wrong; pass m_obs explicitly)
    W, _ = _build_W_trotter(geom, U, a_tau=a_tau, m_obs=m_obs, g_hop=g_hop, order=order)
    idx_to_psi = _fock_index_to_psi_map(V3)

    weights = {}
    dim = 2 ** V3
    for it in range(dim):
        psi_i = idx_to_psi[it]
        for ib in range(dim):
            psi_j = idx_to_psi[ib]
            # Particle-number conservation: H_KS conserves N → W is
            # block-diagonal in N.  Skip cross-sector pairs.
            if bitcount(psi_i) != bitcount(psi_j):
                continue
            amp = complex(W[it, ib])
            if abs(amp) < 1e-15:
                continue
            weights[(psi_i, psi_j)] = (amp, abs(amp))
    return weights


def fock_pair_weights_array(geom: LatticeGeometry, U: Z2GaugeConfig,
                            a_tau: float = 1.0, m_obs: float = None,
                            g_hop: float = 0.5, order: int = 1):
    """Same as fock_pair_weights but returns parallel arrays.
    Returns (pairs Nx2 int, amps N complex, abs_amps N float).
    """
    weights = fock_pair_weights(geom, U, a_tau=a_tau, m_obs=m_obs, g_hop=g_hop,
                                 order=order)
    if not weights:
        return (np.zeros((0, 2), dtype=np.int64),
                np.zeros(0, dtype=complex),
                np.zeros(0, dtype=float))
    pairs = np.array(list(weights.keys()), dtype=np.int64)
    amps = np.array([w[0] for w in weights.values()], dtype=complex)
    abs_amps = np.array([w[1] for w in weights.values()], dtype=float)
    return pairs, amps, abs_amps


def sample_corner_pairs_categorical(geom: LatticeGeometry, U: Z2GaugeConfig,
                                    N_samples: int, rng: np.random.Generator,
                                    a_tau: float = 1.0, m_obs: float = None,
                                    g_hop: float = 0.5):
    """Sample (psi_i, psi_j) pairs categorically from |W[psi_i, psi_j]|.

    signs[k] = Re(W[pair_k]) / |W[pair_k]|  ∈ [-1, 1]
    total    = Σ_pairs |W|  (used for partition-function reweighting)
    """
    pairs_all, amps_all, abs_all = fock_pair_weights_array(
        geom, U, a_tau=a_tau, m_obs=m_obs, g_hop=g_hop)
    total = abs_all.sum()
    if total <= 0:
        raise RuntimeError("Total |W| weight is zero — degenerate gauge config.")
    prob = abs_all / total
    prob = prob / prob.sum()
    idx = rng.choice(len(pairs_all), size=N_samples, p=prob)
    pairs = pairs_all[idx]
    sampled_amps = amps_all[idx]
    sampled_abs = abs_all[idx]
    signs = np.zeros(N_samples, dtype=float)
    nonzero = sampled_abs > 0
    signs[nonzero] = sampled_amps[nonzero].real / sampled_abs[nonzero]
    return pairs, signs, total


def sample_corner_pairs_stratified(geom: LatticeGeometry, U: Z2GaugeConfig,
                                   n_per_sector: int, rng: np.random.Generator,
                                   a_tau: float = 1.0, m_obs: float = None,
                                   g_hop: float = 0.5, order: int = 1):
    """Stratified sampling: n_per_sector samples from each particle-number sector.

    |W|-proportional sampling concentrates on vacuum (where n_0=0), starving
    the observable numerator.  Stratifying ensures coverage of N≥1 sectors.
    Per-sample importance weight: w_sample = sign · sector_total / n_per_sector
    so that Σ samples w_sample ≈ Σ_pairs W.

    order=2 selects the Hermitian-PD palindrome W construction.
    """
    pairs_all, amps_all, abs_all = fock_pair_weights_array(
        geom, U, a_tau=a_tau, m_obs=m_obs, g_hop=g_hop, order=order)
    V3 = geom.V_3
    total = abs_all.sum()

    sector_idx = {n: [] for n in range(V3 + 1)}
    for k in range(len(pairs_all)):
        n = bitcount(int(pairs_all[k, 0]))
        sector_idx[n].append(k)

    pairs_out = []
    signs_out = []
    sector_w_out = []
    for n in range(V3 + 1):
        idxs = sector_idx[n]
        if not idxs:
            continue
        abs_in_sec = abs_all[idxs]
        amps_in_sec = amps_all[idxs]
        sec_total = abs_in_sec.sum()
        if sec_total <= 0:
            continue
        prob = abs_in_sec / sec_total
        prob = prob / prob.sum()
        sel = rng.choice(len(idxs), size=n_per_sector, p=prob)
        for ii in sel:
            k = idxs[ii]
            pair = pairs_all[k]
            amp = amps_in_sec[ii]
            ab = abs_in_sec[ii]
            sign = float(amp.real / ab) if ab > 0 else 1.0
            pairs_out.append(pair)
            signs_out.append(sign)
            sector_w_out.append(sec_total / n_per_sector)

    return (np.array(pairs_out, dtype=np.int64),
            np.array(signs_out, dtype=float),
            np.array(sector_w_out, dtype=float),
            float(total))


def partition_function_consistency_check(geom: LatticeGeometry,
                                         U: Z2GaugeConfig,
                                         a_tau: float = 1.0,
                                         m_obs: float = None,
                                         g_hop: float = 0.5):
    """Σ_pairs W[psi_i, psi_j] vs Tr[T̂_F^{N-1}] direct.  Should agree exactly.

    NOTE: this differs from v2's check.  v2 checked Σ W = det M (Cauchy-Binet
    identity).  Here W comes from T̂_F^{N-1} directly, so Σ W = Tr (sum of
    diagonal entries + off-diagonal cross-particle-number pairs which vanish)
    = Σ_Ψ ⟨Ψ|T̂_F^{N-1}|Ψ⟩ = thermal partition function on fermion sector.
    """
    _, amps, _ = fock_pair_weights_array(
        geom, U, a_tau=a_tau, m_obs=m_obs, g_hop=g_hop)
    direct_sum = complex(np.sum(amps))
    # Reference: full W trace from compute_combined_weight_trotter
    W, _ = _build_W_trotter(geom, U, a_tau=a_tau,
                            m_obs=(m_obs if m_obs is not None else geom.m),
                            g_hop=g_hop)
    W_trace = complex(np.trace(W))
    return direct_sum, W_trace


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def _self_test():
    geom = LatticeGeometry(Lx=2, Ly=2, N_E=5, m=0.5)
    rng = np.random.default_rng(42)
    U = Z2GaugeConfig.random(geom, rng)

    print("=" * 72)
    print("action_corner_direct_v3 self-test (Trotter W corner sampler)")
    print("=" * 72)
    pairs, amps, abs_amps = fock_pair_weights_array(geom, U, a_tau=1.0,
                                                     m_obs=0.5, g_hop=0.5)
    print(f"  V_3={geom.V_3}, N_E={geom.N_E}, random Z_2 gauge")
    print(f"  # (psi_i, psi_j) pairs at same N: {len(pairs)}  (expected: 70 for V_3=4)")
    assert len(pairs) <= 70, f"Got {len(pairs)} pairs, expected ≤70"

    by_n = {}
    for k in range(len(pairs)):
        n = bitcount(int(pairs[k, 0]))
        by_n.setdefault(n, []).append(k)
    print("  Breakdown by N (particle number):")
    for n in sorted(by_n):
        tot = abs_amps[by_n[n]].sum()
        print(f"    N={n}: {len(by_n[n])} pairs, Σ|W| = {tot:.4f}")

    # Partition function consistency: Σ W (same-N pairs) vs Tr[T̂_F^{N-1}]
    direct_sum, W_trace = partition_function_consistency_check(
        geom, U, a_tau=1.0, m_obs=0.5, g_hop=0.5)
    print(f"\n  Σ_pairs (same N) W = {direct_sum:+.6e}")
    print(f"  Tr W  (all 256 diag) = {W_trace:+.6e}")
    print(f"  These should agree (off-diagonal cross-N entries are 0 since "
          f"T̂_F conserves N)")

    # Sample
    pairs_samp, signs_samp, total = sample_corner_pairs_categorical(
        geom, U, N_samples=1000, rng=rng, a_tau=1.0, m_obs=0.5)
    print(f"\n  Sampled 1000 corner pairs. Total |W| = {total:.4f}")
    print(f"  Negative-sign fraction: {np.mean(signs_samp < 0):.3f}")
    print(f"  Unique psi_i: {len(np.unique(pairs_samp[:, 0]))}, "
          f"unique psi_j: {len(np.unique(pairs_samp[:, 1]))}")

    # Stratified
    pairs_str, signs_str, sw_str, total_str = sample_corner_pairs_stratified(
        geom, U, n_per_sector=100, rng=rng, a_tau=1.0, m_obs=0.5)
    print(f"\n  Stratified sampling: {len(pairs_str)} samples "
          f"(across {len(set([bitcount(int(p[0])) for p in pairs_str]))} N sectors)")


if __name__ == "__main__":
    _self_test()
