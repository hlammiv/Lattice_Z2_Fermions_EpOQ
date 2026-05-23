"""
M_act_3 v2: P2 corner-state extraction (REPLACES action_corner_direct.py).

The original `action_corner_direct.py` used the boundary propagator G^bdy
(Lagrangian object) in the Slater-det weights — this gave a ~5%-per-step
systematic plus a wrong-sign C(0) at the toy scale.  This v2 uses the exact
combined-weight matrix W from `transfer_matrix_kbc.py` (Phase 1), which
satisfies Σ_pairs W = det M_OBC[U] to machine precision via the full
Cauchy-Binet 70-term sum.

Interface
---------
Drop-in replacement for `action_corner_direct.py`:
  - fock_pair_weights(geom, U) -> dict
  - fock_pair_weights_array(geom, U) -> (pairs, amps, abs_amps)
  - sample_corner_pairs_categorical(geom, U, N_samples, rng) -> (pairs, signs, total)
  - partition_function_consistency_check(geom, U) -> (direct_sum, det_M)

Convention
----------
The pipeline uses:
   pairs[k] = (psi_i, psi_j),  psi_i = t=0 (top) state, psi_j = t=N_E-1 (bot)
   psi as int, bit k = mode k (= spatial site index)
This module converts between int-encoded Fock states (psi) and tuple labels
used by `compute_combined_weight` via tuple_to_psi / psi_to_tuple helpers.
"""
from __future__ import annotations
import numpy as np
from itertools import product as _product

from action_z2_staggered import LatticeGeometry, Z2GaugeConfig, build_dirac_matrix
from transfer_matrix_kbc import compute_combined_weight


# ----------------------------------------------------------------------------
# Encoding helpers
# ----------------------------------------------------------------------------
def bitcount(n: int) -> int:
    return bin(n).count('1')


def occupied_sites(psi: int, V3: int) -> tuple:
    """Return ordered tuple of occupied site indices for bit-string psi (LSB = site 0)."""
    return tuple(k for k in range(V3) if (psi >> k) & 1)


def tuple_to_psi(n: tuple[int, ...]) -> int:
    """Convert occupation tuple (n_0, n_1, ..., n_{V_3-1}) → int psi with bit k = n_k."""
    return sum(int(nk) << k for k, nk in enumerate(n))


def psi_to_tuple(psi: int, V3: int) -> tuple[int, ...]:
    return tuple((psi >> k) & 1 for k in range(V3))


def _fock_index_to_psi_map(V3: int) -> dict:
    """Map fock_basis_labels enumeration index ↔ psi int."""
    out = {}
    for i, n in enumerate(_product((0, 1), repeat=V3)):
        out[i] = tuple_to_psi(n)
    return out


# ----------------------------------------------------------------------------
# Phase 2: combined-weight–based corner extraction
# ----------------------------------------------------------------------------
def fock_pair_weights(geom: LatticeGeometry, U: Z2GaugeConfig) -> dict:
    """Build dict {(psi_i, psi_j): (amp, |amp|)} for all Fock pairs at same N.

    amp = W[psi_i, psi_j] from Phase 1 (combined weight = K_BC · matrix_element).
    Σ_{pairs} amp = det M_OBC[U] exact.

    psi_i (= "top", t=0)  psi_j (= "bot", t=N_E-1) — same as old M_act_3.
    """
    V3 = geom.V_3
    W, _ = compute_combined_weight(geom, U)
    idx_to_psi = _fock_index_to_psi_map(V3)

    weights = {}
    dim = 2 ** V3
    for it in range(dim):
        psi_i = idx_to_psi[it]
        for ib in range(dim):
            psi_j = idx_to_psi[ib]
            # charge conservation: only same-N pairs have nonzero weight
            if bitcount(psi_i) != bitcount(psi_j):
                continue
            amp = complex(W[it, ib])
            weights[(psi_i, psi_j)] = (amp, abs(amp))
    return weights


def fock_pair_weights_array(geom: LatticeGeometry, U: Z2GaugeConfig):
    """Same as fock_pair_weights but returns parallel arrays.

    Returns: pairs (N×2 int), amps (N complex), abs_amps (N float).
    """
    weights = fock_pair_weights(geom, U)
    pairs = np.array(list(weights.keys()), dtype=np.int64)
    amps = np.array([w[0] for w in weights.values()], dtype=complex)
    abs_amps = np.array([w[1] for w in weights.values()], dtype=float)
    return pairs, amps, abs_amps


def sample_corner_pairs_stratified(geom: LatticeGeometry, U: Z2GaugeConfig,
                                   n_per_sector: int, rng: np.random.Generator):
    """Stratified sampling: n_per_sector samples from EACH particle-number sector.

    |W|-proportional sampling concentrates on vacuum (where n_0=0), starving
    the observable numerator.  Stratifying ensures coverage of N≥1 sectors.

    Per sample, the importance weight is:
        w_sample = sign(W) · sector_total / n_per_sector
    so that  Σ_samples (sign · sector_total / n_per_sector) ≈ Σ_pairs W.

    Returns:
      pairs        (N_total × 2 int)
      signs        (N_total float, = Re(W) / |W|)
      sector_w     (N_total float, = sector_total_|W| / n_per_sector)
      total        (float, Σ |W| over all sectors — for diagnostics)
    """
    pairs_all, amps_all, abs_all = fock_pair_weights_array(geom, U)
    V3 = geom.V_3
    total = abs_all.sum()

    # Bucket pairs by particle number (of psi_i, which == that of psi_j)
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


def sample_corner_pairs_categorical(geom: LatticeGeometry, U: Z2GaugeConfig,
                                    N_samples: int, rng: np.random.Generator):
    """Sample (psi_i, psi_j) pairs categorically from |W[psi_i, psi_j]|.

    Returns:
      pairs    (N_samples × 2 int)
      signs    (N_samples float, = Re(W[pair]) / |W[pair]| ∈ [-1, 1]).
                For purely-real W this reduces to ±1. For complex W it is
                the real-projection of the phase factor — required for the
                unbiased estimator  det M = total · ⟨signs⟩.
      total    (float, Σ |W|)
    """
    pairs_all, amps_all, abs_all = fock_pair_weights_array(geom, U)
    total = abs_all.sum()
    if total <= 0:
        raise RuntimeError("Total |W| weight is zero.")
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


def partition_function_consistency_check(geom: LatticeGeometry,
                                         U: Z2GaugeConfig) -> tuple[complex, float]:
    """Verify Σ_pairs W[psi_i, psi_j] == det M_OBC[U]."""
    _, amps, _ = fock_pair_weights_array(geom, U)
    direct_sum = complex(np.sum(amps))
    det_M_true = float(np.linalg.det(build_dirac_matrix(geom, U)))
    return direct_sum, det_M_true


# ----------------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------------
def _self_test():
    geom = LatticeGeometry(Lx=2, Ly=2, N_E=4, m=0.5)
    rng = np.random.default_rng(42)
    U = Z2GaugeConfig.random(geom, rng)

    print("=" * 72)
    print("M_act_3 v2 self-test (P2 Slater-det MC sampler)")
    print("=" * 72)
    pairs, amps, abs_amps = fock_pair_weights_array(geom, U)
    print(f"  # (Ψ_i, Ψ_j) pairs at same N: {len(pairs)} (expected: 70 for V_3=4)")
    assert len(pairs) == 70

    # Distribution by particle number
    print(f"\n  Breakdown by particle number:")
    for n in range(geom.V_3 + 1):
        idx = [k for k in range(len(pairs)) if bitcount(int(pairs[k, 0])) == n]
        if idx:
            tot = abs_amps[idx].sum()
            real_tot = float(np.sum(amps[idx].real))
            print(f"    N={n}: {len(idx)} pairs, Σ|amp| = {tot:+.6f}, "
                  f"Σ(amp).real = {real_tot:+.6f}")

    # Partition-function consistency
    direct_sum, det_M_true = partition_function_consistency_check(geom, U)
    print(f"\n  Σ W (over all pairs)        = {direct_sum:+.10f}")
    print(f"  det M_OBC (true)             = {det_M_true:+.10f}")
    print(f"  |error|                       = {abs(direct_sum - det_M_true):.2e}")
    assert abs(direct_sum - det_M_true) < 1e-10, "Phase 1 consistency broken!"

    # Sample 1000 pairs
    pairs_samp, signs_samp, total = sample_corner_pairs_categorical(
        geom, U, N_samples=1000, rng=rng)
    print(f"\n  Sampled 1000 pairs:  total |W| = {total:+.4f}")
    print(f"    sign(amp) = -1 fraction: {np.mean(signs_samp < 0):.3f}")
    print(f"    unique psi_i values:     {len(np.unique(pairs_samp[:, 0]))}")
    print(f"    unique psi_j values:     {len(np.unique(pairs_samp[:, 1]))}")
    diag_count = int(np.sum(pairs_samp[:, 0] == pairs_samp[:, 1]))
    print(f"    diagonal (psi_i==psi_j): {diag_count}/1000")


if __name__ == '__main__':
    _self_test()
