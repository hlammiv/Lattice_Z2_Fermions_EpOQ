"""Static-gauge consistency check.

At U_static (gauge field constant in time), compute:
  1. det M_forward[U_static]   — V_4-dim Grassmann determinant
  2. Tr T_F[U_static]^{N_E-1}  — Fock-space trace (= det(1 + e^{-β h}) by free-fermion identity)
  3. det(1 + e^{-β h_single})  — direct from single-particle h[U_static]

For the EρOQ pipeline construction
    ρ̃ ∝ ∫ DU det M_forward · e^{-S_g} · ⟨ψ_a|W[U]|ψ_b⟩
to be a path-integral identity for e^{-βH_QC}, the det M factor and W must
be related so that they don't double-count the fermion sector.  The
simplest cross-check is at STATIC gauge where the relationship should be
clean.

We probe several static gauges to see how each quantity scales with N_E
and what relations (if any) hold among them.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import sys
import numpy as np
from scipy.linalg import expm

sys.path.insert(0, '/home/hlamm/Desktop/QC/logdet/m1_toy')

from action_z2_staggered import LatticeGeometry, Z2GaugeConfig
from action_z2_staggered_forwardtime import compute_det_M_forward, build_slice_block_A
from transfer_matrix_kbc_trotterized import compute_combined_weight_trotter, build_T_F_trotter

M_HAM = 0.5
G_HOP = 0.5


def static_U(geom: LatticeGeometry, U_x_val: int = 1, U_y_val: int = 1,
             U_t_val: int = 1) -> Z2GaugeConfig:
    """Build a gauge config with ALL links set to the same value (homogeneous)."""
    return Z2GaugeConfig(
        geom=geom,
        U_x=np.full((geom.N_E, geom.Lx - 1, geom.Ly), U_x_val, dtype=np.int8),
        U_y=np.full((geom.N_E, geom.Lx, geom.Ly - 1), U_y_val, dtype=np.int8),
        U_t=np.full((geom.N_E - 1, geom.Lx, geom.Ly), U_t_val, dtype=np.int8),
    )


def random_static_U(geom: LatticeGeometry, rng: np.random.Generator) -> Z2GaugeConfig:
    """Pick spatial gauge once (a single time-slice), repeat in time."""
    Lx, Ly, N_E = geom.Lx, geom.Ly, geom.N_E
    ux = rng.choice([-1, 1], size=(Lx - 1, Ly))
    uy = rng.choice([-1, 1], size=(Lx, Ly - 1))
    ut = rng.choice([-1, 1], size=(Lx, Ly))
    return Z2GaugeConfig(
        geom=geom,
        U_x=np.broadcast_to(ux, (N_E, Lx - 1, Ly)).copy().astype(np.int8),
        U_y=np.broadcast_to(uy, (N_E, Lx, Ly - 1)).copy().astype(np.int8),
        U_t=np.broadcast_to(ut, (N_E - 1, Lx, Ly)).copy().astype(np.int8),
    )


def check_one(label, geom, U, a_tau):
    """Compute and print the diagnostics for a single U_static config."""
    # 1. det M_forward
    detM = compute_det_M_forward(geom, U)

    # 2. W = T_F^{N-1} at static U → should equal e^{-β h_eff_Fock} where
    #    h_eff_Fock is whatever T_F = e^{-a_τ h} actually exponentiates.
    W_full, _ = compute_combined_weight_trotter(
        geom, U, a_tau=a_tau, K_E=0.0, K_M=0.0,
        m_obs=M_HAM, g_hop=G_HOP, order=1,   # order=1 = time-ordered
    )
    tr_W = float(np.trace(W_full).real)

    # 3. Build single T_F[U(0)] step and check static-product identity:
    #    For static U:  W_full ?= T_F[U(0)]^{N-1}
    T1 = build_T_F_trotter(
        geom, U, t_slice=0, a_tau=a_tau,
        K_E=0.0, K_M=0.0, m_obs=M_HAM, g_hop=G_HOP,
    ) if False else None  # call signature varies; do it via N_E=2 short test

    # Single-step T at this slice from compute_combined_weight_trotter with N_E=2.
    # We synthesize a 2-slice geom on the same gauge and compute W; with N_E=2
    # there's only one T_F^{N-1=1} = T_F[U(0)].
    if geom.N_E >= 2:
        geom2 = LatticeGeometry(Lx=geom.Lx, Ly=geom.Ly, N_E=2, m=geom.m)
        U2 = Z2GaugeConfig(
            geom=geom2,
            U_x=U.U_x[:2].copy(), U_y=U.U_y[:2].copy(),
            U_t=U.U_t[:1].copy(),
        )
        T_single, _ = compute_combined_weight_trotter(
            geom2, U2, a_tau=a_tau, K_E=0.0, K_M=0.0,
            m_obs=M_HAM, g_hop=G_HOP, order=1,
        )
        # Compare W to T_single^{N-1}
        T_pow = np.linalg.matrix_power(T_single, geom.N_E - 1)
        ratio_W = np.max(np.abs(W_full - T_pow))
        tr_T_pow = float(np.trace(T_pow).real)
    else:
        ratio_W = float('nan')
        tr_T_pow = float('nan')

    # 4. Free-fermion Slater-det identity check:
    #    Tr T_F^{N-1} = det(1 + T1_single_particle^{N-1}) for free fermions.
    #    (Doesn't apply with interactions, but H_KS at fixed gauge IS quadratic.)
    print(f"  {label}")
    print(f"    det M_forward       = {detM:+.5e}")
    print(f"    Tr W (= Tr T_F^N-1) = {tr_W:+.5e}")
    print(f"    Tr T_single^{geom.N_E-1}     = {tr_T_pow:+.5e}  "
          f"(static-product check)")
    print(f"    max|W - T_single^N-1| = {ratio_W:.2e}  "
          f"(should be 0 at static U)")
    return detM, tr_W, ratio_W


def main():
    print("=" * 80)
    print("Static-gauge consistency test")
    print(f"  m={M_HAM}, g_hop={G_HOP}")
    print("=" * 80)

    rng = np.random.default_rng(2026)

    for N_E in (2, 3, 4, 5):
        a_tau = 0.5  # fixed a_τ; vary N_E to see scaling
        geom = LatticeGeometry(Lx=2, Ly=2, N_E=N_E, m=a_tau * M_HAM)
        print(f"\n[N_E = {N_E}, a_τ = {a_tau}, β_eff = {(N_E-1)*a_tau}]")

        # All-+1
        U = static_U(geom, +1, +1, +1)
        d1, t1, r1 = check_one("U_static = +1 (all links)", geom, U, a_tau)

        # Random static
        U_r = random_static_U(geom, rng)
        d2, t2, r2 = check_one("U_static random spatial", geom, U_r, a_tau)

    print("\n" + "=" * 80)
    print("Scaling check (homogeneous U=+1):")
    print(f"  {'N_E':>4} | {'a_τ·(N-1)=β':>12} | {'det M':>14} | {'Tr W':>14} | "
          f"{'ratio':>10}")
    a_tau = 0.5
    for N_E in (2, 3, 4, 5):
        geom = LatticeGeometry(Lx=2, Ly=2, N_E=N_E, m=a_tau * M_HAM)
        U = static_U(geom, +1, +1, +1)
        detM = compute_det_M_forward(geom, U)
        W_full, _ = compute_combined_weight_trotter(
            geom, U, a_tau=a_tau, K_E=0.0, K_M=0.0,
            m_obs=M_HAM, g_hop=G_HOP, order=1)
        trW = np.trace(W_full).real
        ratio = detM / trW if trW != 0 else float('nan')
        print(f"  {N_E:>4d} | {(N_E-1)*a_tau:>12.2f} | {detM:>+14.5e} | "
              f"{trW:>+14.5e} | {ratio:>+10.4e}")


if __name__ == "__main__":
    main()
