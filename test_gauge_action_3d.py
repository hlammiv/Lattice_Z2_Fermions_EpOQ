"""Phase 40 step 2 test: gauge_action_kernel_3d.

Checks:
  1. Trivial 3D 2x2x2 config: S_g matches the expected plaquette count.
  2. Random 3D config: action equals a pure-Python reference (independent
     of the numba kernel).
  3. 2D backward compat: gauge_action at Lz=1 unchanged.
  4. "z-trivial extension" consistency: a 3D config that's identical
     across z layers with U_z = +1 has S_g = (z-independent kernel
     sum), checked against an explicit closed form.
"""
from __future__ import annotations
import numpy as np
import sys

sys.path.insert(0, '/home/hlamm/Desktop/QC/logdet/m1_toy')

from action_z2_staggered import (
    LatticeGeometry, Z2GaugeConfig,
    count_links_and_plaquettes, gauge_action,
)


def python_gauge_action_3d(geom, U, K_E=1.0, K_M=1.0):
    """Pure-Python reference for the 3D action — independent of numba kernel."""
    Lx, Ly, Lz, N_E = geom.Lx, geom.Ly, geom.Lz, geom.N_E
    total_E = 0.0; total_M = 0.0
    Ux, Uy, Uz, Ut = U.U_x, U.U_y, U.U_z, U.U_t
    # Magnetic — xy, xz, yz
    for t in range(N_E):
        for z in range(Lz):
            for x in range(Lx - 1):
                for y in range(Ly - 1):
                    total_M += (Ux[t, x, y, z] * Uy[t, x+1, y, z]
                                * Ux[t, x, y+1, z] * Uy[t, x, y, z])
        for y in range(Ly):
            for x in range(Lx - 1):
                for z in range(Lz - 1):
                    total_M += (Ux[t, x, y, z] * Uz[t, x+1, y, z]
                                * Ux[t, x, y, z+1] * Uz[t, x, y, z])
        for x in range(Lx):
            for y in range(Ly - 1):
                for z in range(Lz - 1):
                    total_M += (Uy[t, x, y, z] * Uz[t, x, y+1, z]
                                * Uy[t, x, y, z+1] * Uz[t, x, y, z])
    # Electric — xτ, yτ, zτ
    for t in range(N_E - 1):
        for z in range(Lz):
            for x in range(Lx - 1):
                for y in range(Ly):
                    total_E += (Ux[t, x, y, z] * Ut[t, x+1, y, z]
                                * Ux[t+1, x, y, z] * Ut[t, x, y, z])
            for x in range(Lx):
                for y in range(Ly - 1):
                    total_E += (Uy[t, x, y, z] * Ut[t, x, y+1, z]
                                * Uy[t+1, x, y, z] * Ut[t, x, y, z])
        for x in range(Lx):
            for y in range(Ly):
                for z in range(Lz - 1):
                    total_E += (Uz[t, x, y, z] * Ut[t, x, y, z+1]
                                * Uz[t+1, x, y, z] * Ut[t, x, y, z])
    return -K_E * float(total_E) - K_M * float(total_M)


def main():
    print("=" * 70)
    print("Phase 40 step 2: gauge_action_kernel_3d")
    print("=" * 70)

    # --- 1. Trivial 2x2x2, K_E=K_M=1 ---
    geom = LatticeGeometry(Lx=2, Ly=2, Lz=2, N_E=5, m=0.5)
    counts = count_links_and_plaquettes(geom)
    U_triv = Z2GaugeConfig.trivial(geom)
    S_triv = gauge_action(geom, U_triv, K_E=1.0, K_M=1.0)
    # Expected: every plaquette product = 1; S_g = -K_M*magnetic - K_E*electric
    mag_count = counts['xy_plaq'] + counts['xz_plaq'] + counts['yz_plaq']
    elec_count = counts['xt_plaq'] + counts['yt_plaq'] + counts['zt_plaq']
    expected = -1.0 * elec_count - 1.0 * mag_count
    print(f"\nTrivial 2x2x2 N_E=5: S_g = {S_triv}, expected = {expected}.")
    assert S_triv == expected, f"{S_triv} != {expected}"
    print(f"  Magnetic plaqs = {mag_count} (xy={counts['xy_plaq']} "
          f"+ xz={counts['xz_plaq']} + yz={counts['yz_plaq']})")
    print(f"  Electric plaqs = {elec_count} (xt={counts['xt_plaq']} "
          f"+ yt={counts['yt_plaq']} + zt={counts['zt_plaq']})")
    print(f"  PASS")

    # --- 2. Random 3D 2x2x2: kernel matches pure-Python reference ---
    rng = np.random.default_rng(2026)
    U_rand = Z2GaugeConfig.random(geom, rng)
    S_kernel = gauge_action(geom, U_rand, K_E=1.0, K_M=0.5)
    S_python = python_gauge_action_3d(geom, U_rand, K_E=1.0, K_M=0.5)
    diff = abs(S_kernel - S_python)
    print(f"\nRandom 3D 2x2x2: kernel = {S_kernel:+.6f}, "
          f"python ref = {S_python:+.6f}, |Δ| = {diff:.3e}.")
    assert diff < 1e-10, f"{diff} too big"
    print(f"  PASS")

    # --- 3. 2D backward compat: gauge_action at Lz=1 unchanged ---
    geom_2d = LatticeGeometry(Lx=2, Ly=2, N_E=5, m=0.5)
    U_2d_triv = Z2GaugeConfig.trivial(geom_2d)
    S_2d_triv = gauge_action(geom_2d, U_2d_triv, K_E=1.0, K_M=1.0)
    counts_2d = count_links_and_plaquettes(geom_2d)
    expected_2d = -(counts_2d['xt_plaq'] + counts_2d['yt_plaq']) - counts_2d['xy_plaq']
    print(f"\n2D backward compat trivial: S_g = {S_2d_triv}, expected = {expected_2d}.")
    assert S_2d_triv == expected_2d
    # Random 2D
    rng2 = np.random.default_rng(42)
    U_2d_rand = Z2GaugeConfig.random(geom_2d, rng2)
    S_2d_rand = gauge_action(geom_2d, U_2d_rand, K_E=1.0, K_M=0.5)
    print(f"  Random 2D config gives S_g = {S_2d_rand:.4f} (smoke check only).  PASS")

    # --- 4. "z-trivial extension" consistency ---
    # Build a 3D 2x2x2 config where each z layer is the SAME 2D config,
    # U_z = +1 everywhere, and U_t is also z-independent.  Then:
    #   S_g_3D = K_E (xτ + yτ + zτ_contribs) + K_M (xy + xz + yz_contribs)
    # with z-independence + U_z=1, all the xz/yz/zτ plaquettes evaluate
    # to +1 (since u1=u3 in those plaq products), giving constant counts.
    geom_ext = LatticeGeometry(Lx=2, Ly=2, Lz=2, N_E=4, m=0.5)
    geom_2d_match = LatticeGeometry(Lx=2, Ly=2, N_E=4, m=0.5)
    rng3 = np.random.default_rng(7)
    U_2d_base = Z2GaugeConfig.random(geom_2d_match, rng3)

    # 3D embedding: replicate the 2D config across z, set U_z = +1.
    Lz_ext = geom_ext.Lz
    U_x_3d = np.broadcast_to(U_2d_base.U_x[..., None],
                              U_2d_base.U_x.shape + (Lz_ext,)).copy()
    U_y_3d = np.broadcast_to(U_2d_base.U_y[..., None],
                              U_2d_base.U_y.shape + (Lz_ext,)).copy()
    U_t_3d = np.broadcast_to(U_2d_base.U_t[..., None],
                              U_2d_base.U_t.shape + (Lz_ext,)).copy()
    U_z_3d = np.ones((geom_ext.N_E, geom_ext.Lx, geom_ext.Ly, Lz_ext - 1),
                     dtype=int)
    U_3d_ext = Z2GaugeConfig(geom=geom_ext,
                             U_x=U_x_3d, U_y=U_y_3d, U_z=U_z_3d, U_t=U_t_3d)

    S_3d_ext = gauge_action(geom_ext, U_3d_ext, K_E=1.0, K_M=0.5)
    S_2d_base = gauge_action(geom_2d_match, U_2d_base, K_E=1.0, K_M=0.5)

    # Closed form:
    #   M_3D = Σ_z (M_2D for that z layer) + (xz + yz constants)
    #   With z-independent U_x and U_y, EVERY xz plaq = U_x(z)·1·U_x(z+1)·1 = 1
    #   With z-independent U_y, EVERY yz plaq = U_y(z)·1·U_y(z+1)·1 = 1
    #   So M_3D = Lz · (xy 2D sum) + (xz_count + yz_count) · 1
    #
    #   E_3D = Σ_z (E_2D for that z layer) + (zτ constants)
    #   zτ plaq = U_z(t)·U_t(z)·U_z(t+1)·U_t(z) = 1·1·1·1 = 1 = +1
    #   So E_3D = Lz · (xτ_2D + yτ_2D) + zτ_count · 1
    #
    #   With K_E and K_M:
    #   S_3D = Lz·(S_2D_pieces) + extra constants from xz, yz, zτ
    sums_2d = count_links_and_plaquettes(geom_2d_match)
    sums_ext = count_links_and_plaquettes(geom_ext)
    # 2D "magnetic" = xy.  In our action S_2D = -K_E·(xτ+yτ) - K_M·xy.
    # We don't know the raw plaquette sums for the random U_2d_base apart
    # from S_2d_base.  But we can compute S_2D_E and S_2D_M separately:
    S_2D_M_only = gauge_action(geom_2d_match, U_2d_base, K_E=0.0, K_M=1.0)
    S_2D_E_only = gauge_action(geom_2d_match, U_2d_base, K_E=1.0, K_M=0.0)
    xy_2d_sum = -S_2D_M_only
    et_2d_sum = -S_2D_E_only

    expected_M_3D = Lz_ext * xy_2d_sum + (sums_ext['xz_plaq']
                                          + sums_ext['yz_plaq'])
    expected_E_3D = Lz_ext * et_2d_sum + sums_ext['zt_plaq']
    expected_S_3D = -1.0 * expected_E_3D - 0.5 * expected_M_3D

    diff_ext = abs(S_3d_ext - expected_S_3D)
    print(f"\nz-trivial extension consistency check:")
    print(f"  S_2d_base = {S_2d_base:+.4f}, S_3d_ext = {S_3d_ext:+.4f}, "
          f"expected = {expected_S_3D:+.4f}, |Δ| = {diff_ext:.3e}.")
    assert diff_ext < 1e-10
    print(f"  PASS")

    print("\n" + "=" * 70)
    print("Phase 40 step 2 — ALL PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()
