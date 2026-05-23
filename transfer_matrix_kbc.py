"""
transfer_matrix_kbc.py — Phase 1 of the P2 pipeline.

For each gauge config U, produce the combined_weight matrix W[Ψ_top, Ψ_bot]
in the ORIGINAL Fock basis (2^{V_3} × 2^{V_3}), such that

    det M_OBC[U]  =  Σ_{Ψ_top, Ψ_bot at same N}  W[U][Ψ_top, Ψ_bot]

W is block-diagonal in particle-number sectors (charge conservation), with
off-diagonals within each sector for Z₂ random U.  Used as importance
weights for sampling (Ψ_top, Ψ_bot) in the P2 corner-state pipeline.

Construction.  Use the exact Cauchy-Binet identity (Agent B):

    det M = (∏_τ det F_τ) · Σ_{S ⊂ [2V_3], |S|=V_3}
                            (∏_{k∈S} μ_k) · det(φ_+^S) · det(c^S)

where μ_k, evec, coeffs come from the block-companion C_tot
diagonalization.  Each CB subset S of size V_3 maps to a UNIQUE Fock pair
(Ψ_top, Ψ_bot) at the same particle number, via the bijection:

  S has 2V_3 = (V_3 particle-sector + V_3 antiparticle-sector) eigvals.
  For each mode k ∈ [V_3]:
    plus_idx[k] ∈ S  AND  minus_idx[k] ∈ S    →  "doubled" mode at k
        → Ψ_top has mode k occupied, Ψ_bot does NOT
    plus_idx[k] ∉ S  AND  minus_idx[k] ∉ S    →  "missing" mode at k
        → Ψ_top does NOT have mode k, Ψ_bot has it
    Only plus_idx[k] in S                      →  "single particle" (vacuum)
        → both Ψ_top and Ψ_bot vacuum at k
    Only minus_idx[k] in S                     →  "single antiparticle" (occ)
        → both Ψ_top and Ψ_bot have mode k occupied

Charge conservation (|Ψ_top|=|Ψ_bot|) is automatic: # doubled = # missing
because the V_3 constraint forces 2·#doubled + #single = V_3 = (#doubled)
+ (#missing) + (#single) ⇒ #doubled = #missing.

The bijection is 1-to-1 between C(2V_3, V_3) CB subsets and Fock pairs at
same N.  C(2V_3, V_3) = 1²+...+C(V_3, V_3)² (the number of size-N
pairs summed over N=0..V_3) which equals C(2V_3, V_3) by Vandermonde.

For V_3 = 4: 70 CB subsets ↔ 70 Fock pairs (1+16+36+16+1).
"""
from __future__ import annotations
import sys
import numpy as np
from itertools import product as _product, combinations as _combos

sys.path.insert(0, '/home/hlamm/Desktop/QC/logdet/m1_toy')
from action_z2_staggered import (
    LatticeGeometry, Z2GaugeConfig,
)
# Legacy: build_dirac_matrix (central-diff) removed 2026-05-20.
from transfer_matrix_baseline import (
    build_B_slice, build_F_slice, block_companion_X,
)
from transfer_matrix_bogoliubov import T_F_single_from_companion
from transfer_matrix_boundary import (
    block_companion_product, fock_basis_labels,
)


def fock_state_to_occ(n: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(i for i, ni in enumerate(n) if ni == 1)


def particle_number(n: tuple[int, ...]) -> int:
    return sum(n)


def cb_subset_to_fock_pair(S: tuple[int, ...],
                           plus_idx: list[int],
                           minus_idx: list[int],
                           V3: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Map CB subset S (size V_3 indices into [2V_3]) to a Fock pair (n_top, n_bot).

    Per mode k:
       both plus_idx[k] and minus_idx[k] in S   → doubled (n_top=1, n_bot=0)
       neither in S                              → missing (n_top=0, n_bot=1)
       only plus_idx[k]                          → vacuum spectator (0, 0)
       only minus_idx[k]                         → occupied spectator (1, 1)
    """
    S_set = set(S)
    n_top = [0] * V3
    n_bot = [0] * V3
    for k in range(V3):
        in_plus = plus_idx[k] in S_set
        in_minus = minus_idx[k] in S_set
        if in_plus and in_minus:
            n_top[k] = 1; n_bot[k] = 0
        elif not in_plus and not in_minus:
            n_top[k] = 0; n_bot[k] = 1
        elif in_minus:
            n_top[k] = 1; n_bot[k] = 1
        else:
            n_top[k] = 0; n_bot[k] = 0
    return tuple(n_top), tuple(n_bot)


def compute_combined_weight(geom: LatticeGeometry,
                            U: Z2GaugeConfig) -> tuple[np.ndarray, dict]:
    """Returns (W, info) where W is 2^{V_3} × 2^{V_3} complex matrix in
    original Fock basis (lex-ordered by `fock_basis_labels`), satisfying
    Σ W = det M[U] to machine precision.

    Construction via full Cauchy-Binet sum over C(2V_3, V_3) subsets and
    the subset → Fock-pair bijection (see module docstring).
    """
    V3 = geom.V_3
    dim = 2 ** V3
    basis = fock_basis_labels(V3)
    fock_index = {n: i for i, n in enumerate(basis)}

    # Block-companion C_tot eigendecomposition (Agent B)
    C_tot, v_init, prod_F = block_companion_product(geom, U)
    eigC, evec = np.linalg.eig(C_tot)
    if np.linalg.cond(evec) > 1e10:
        # degenerate eigvals — perturb minimally to break degeneracy
        rng = np.random.default_rng(12345)
        perturb = 1e-10 * (rng.standard_normal(C_tot.shape)
                           + 1j * rng.standard_normal(C_tot.shape))
        eigC, evec = np.linalg.eig(C_tot + perturb)
    coeffs = np.linalg.solve(evec, v_init)
    evec_upper = evec[:V3, :]

    # Identify particle / antiparticle sectors of C_tot by magnitude
    abs_eig = np.abs(eigC)
    order = np.argsort(-abs_eig)
    plus_idx = list(order[:V3])
    minus_idx = list(order[V3:])

    # Cauchy-Binet sum: enumerate all C(2V_3, V_3) subsets, map each to a
    # unique Fock pair, accumulate into W.
    W = np.zeros((dim, dim), dtype=complex)
    cb_sum = complex(0.0)
    n_subsets = 0
    for S in _combos(range(2 * V3), V3):
        S_list = list(S)
        mu_prod = complex(np.prod([eigC[k] for k in S_list]))
        det_phi = complex(np.linalg.det(evec_upper[:, S_list]))
        det_c = complex(np.linalg.det(coeffs[S_list, :]))
        cb_term = prod_F * mu_prod * det_phi * det_c
        cb_sum += cb_term

        n_top, n_bot = cb_subset_to_fock_pair(S, plus_idx, minus_idx, V3)
        it = fock_index[n_top]
        ib = fock_index[n_bot]
        W[it, ib] += cb_term
        n_subsets += 1

    det_M_true = float(np.linalg.det(build_dirac_matrix(geom, U)))
    det_M_via_W = complex(np.sum(W))

    info = {
        'V_bog_eigvecs_upper': evec_upper[:, plus_idx],
        'plus_idx': plus_idx,
        'minus_idx': minus_idx,
        'prod_F': prod_F,
        'cb_sum_total': cb_sum,
        'det_M_true': det_M_true,
        'det_M_via_W': det_M_via_W,
        'err': abs(det_M_via_W - det_M_true),
        'n_subsets': n_subsets,
    }
    return W, info


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------
def test_v3_1():
    print("=" * 72)
    print("TEST 1: V_3 = 1 free, m=0.5, N_E=4")
    print("=" * 72)
    geom = LatticeGeometry(Lx=1, Ly=1, N_E=4, m=0.5)
    U = Z2GaugeConfig.trivial(geom)
    W, info = compute_combined_weight(geom, U)
    print(f"  # CB subsets enumerated     = {info['n_subsets']}")
    print(f"  W_orig matrix:")
    print(f"    {np.real_if_close(W).round(8)}")
    print(f"  det M (true)               = {info['det_M_true']:+.10f}")
    print(f"  det M via Σ W              = {info['det_M_via_W']:+.10f}")
    print(f"  |err|                       = {info['err']:.2e}")


def test_v3_4_free():
    print()
    print("=" * 72)
    print("TEST 2: V_3 = 4 free, 2×2×4, m=0.5 (trivial U)")
    print("=" * 72)
    geom = LatticeGeometry(Lx=2, Ly=2, N_E=4, m=0.5)
    U = Z2GaugeConfig.trivial(geom)
    W, info = compute_combined_weight(geom, U)
    n_offdiag = 0
    max_offdiag = 0.0
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            if i != j and abs(W[i, j]) > 1e-10:
                n_offdiag += 1
                max_offdiag = max(max_offdiag, abs(W[i, j]))
    print(f"  # CB subsets enumerated     = {info['n_subsets']}")
    print(f"  W_orig: # nonzero off-diag = {n_offdiag}")
    print(f"          max |off-diag|     = {max_offdiag:.4e}")
    print(f"  det M (true)               = {info['det_M_true']:+.10f}")
    print(f"  det M via Σ W              = {info['det_M_via_W']:+.10f}")
    print(f"  |err|                       = {info['err']:.2e}")


def test_z2_random(n_trials: int = 8, seed: int = 2026):
    print()
    print("=" * 72)
    print(f"TEST 3: Z_2 random ({n_trials} trials)")
    print("=" * 72)
    geom = LatticeGeometry(Lx=2, Ly=2, N_E=4, m=0.5)
    rng = np.random.default_rng(seed)
    max_err = 0.0
    for trial in range(n_trials):
        U = Z2GaugeConfig.random(geom, rng)
        W, info = compute_combined_weight(geom, U)
        max_err = max(max_err, info['err'])
        n_offdiag = sum(1 for i in range(W.shape[0]) for j in range(W.shape[1])
                        if i != j and abs(W[i, j]) > 1e-10)
        max_offdiag = max((abs(W[i, j]) for i in range(W.shape[0])
                           for j in range(W.shape[1]) if i != j),
                          default=0.0)
        print(f"  trial {trial}: det M = {info['det_M_true']:+.6f}, "
              f"Σ W = {info['det_M_via_W'].real:+.6f}, err = {info['err']:.2e}, "
              f"# off-diag = {n_offdiag}, max |off-diag| = {max_offdiag:.3e}")
    print(f"  max err across {n_trials} trials = {max_err:.2e}")


if __name__ == "__main__":
    test_v3_1()
    test_v3_4_free()
    test_z2_random()
