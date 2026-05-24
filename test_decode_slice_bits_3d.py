"""Phase 40 step 4 test: gauge_qc_bits_general + slab's decode_slice_bits
round-trip at 2D Lz=1 and 3D Lz=2.

The decode logic lives inside accumulate_C_direct_slab; reproduce it here
(this test is the spec).  Random Z2GaugeConfig → encode each slice via
gauge_qc_bits_general → decode via the (Lx, Ly, Lz)-aware decoder →
verify recovered slice arrays == originals.
"""
from __future__ import annotations
import sys
import numpy as np

sys.path.insert(0, '/home/hlamm/Desktop/QC/logdet/m1_toy')

from action_z2_staggered import LatticeGeometry, Z2GaugeConfig
from action_minkowski_stitch import gauge_qc_bits_general, qc_layout_counts


def decode_slice_bits_2d(slice_bits, geom):
    Lx, Ly = geom.Lx, geom.Ly
    n_y = Lx * (Ly - 1)
    U_y_slice = np.ones((Lx, Ly - 1), dtype=int)
    U_x_slice = np.ones((Lx - 1, Ly), dtype=int)
    for x in range(Lx):
        for y in range(Ly - 1):
            if (slice_bits >> (x * (Ly - 1) + y)) & 1:
                U_y_slice[x, y] = -1
    for x in range(Lx - 1):
        for y in range(Ly):
            if (slice_bits >> (n_y + x * Ly + y)) & 1:
                U_x_slice[x, y] = -1
    return U_x_slice, U_y_slice, None


def decode_slice_bits_3d(slice_bits, geom):
    Lx, Ly, Lz = geom.Lx, geom.Ly, geom.Lz
    n_y = Lx * (Ly - 1) * Lz
    n_x = (Lx - 1) * Ly * Lz
    U_y_slice = np.ones((Lx, Ly - 1, Lz), dtype=int)
    U_x_slice = np.ones((Lx - 1, Ly, Lz), dtype=int)
    U_z_slice = np.ones((Lx, Ly, Lz - 1), dtype=int)
    for x in range(Lx):
        for y in range(Ly - 1):
            for z in range(Lz):
                pos = (x * (Ly - 1) + y) * Lz + z
                if (slice_bits >> pos) & 1:
                    U_y_slice[x, y, z] = -1
    for x in range(Lx - 1):
        for y in range(Ly):
            for z in range(Lz):
                pos = n_y + (x * Ly + y) * Lz + z
                if (slice_bits >> pos) & 1:
                    U_x_slice[x, y, z] = -1
    for x in range(Lx):
        for y in range(Ly):
            for z in range(Lz - 1):
                pos = n_y + n_x + (x * Ly + y) * (Lz - 1) + z
                if (slice_bits >> pos) & 1:
                    U_z_slice[x, y, z] = -1
    return U_x_slice, U_y_slice, U_z_slice


def main():
    print("=" * 70)
    print("Phase 40 step 4: gauge_qc_bits_general ↔ decode_slice_bits round trip")
    print("=" * 70)

    # --- 2D 2×2, N_E=4: round-trip every slice of a random config ---
    print("\n2D 2×2:")
    geom_2d = LatticeGeometry(Lx=2, Ly=2, N_E=4, m=0.5)
    rng = np.random.default_rng(2026)
    U_2d = Z2GaugeConfig.random(geom_2d, rng)
    n_g_2d, _ = qc_layout_counts(geom_2d)
    print(f"  n_gauge = {n_g_2d}, slice space = 2^{n_g_2d} = {1 << n_g_2d}")
    for t in range(geom_2d.N_E):
        sb = gauge_qc_bits_general(U_2d, t, geom_2d)
        Ux, Uy, Uz = decode_slice_bits_2d(sb, geom_2d)
        assert np.array_equal(Ux, U_2d.U_x[t])
        assert np.array_equal(Uy, U_2d.U_y[t])
        assert Uz is None
    print(f"  All {geom_2d.N_E} slices round-trip exactly.  PASS")

    # --- 2D 2×3, N_E=3 ---
    print("\n2D 2×3:")
    geom_2x3 = LatticeGeometry(Lx=2, Ly=3, N_E=3, m=0.5)
    U_2x3 = Z2GaugeConfig.random(geom_2x3, rng)
    n_g_2x3, _ = qc_layout_counts(geom_2x3)
    print(f"  n_gauge = {n_g_2x3}, slice space = 2^{n_g_2x3} = {1 << n_g_2x3}")
    for t in range(geom_2x3.N_E):
        sb = gauge_qc_bits_general(U_2x3, t, geom_2x3)
        Ux, Uy, Uz = decode_slice_bits_2d(sb, geom_2x3)
        assert np.array_equal(Ux, U_2x3.U_x[t])
        assert np.array_equal(Uy, U_2x3.U_y[t])
        assert Uz is None
    print(f"  All {geom_2x3.N_E} slices round-trip.  PASS")

    # --- 3D 2×2×2, N_E=3 ---
    print("\n3D 2×2×2:")
    geom_3d = LatticeGeometry(Lx=2, Ly=2, Lz=2, N_E=3, m=0.5)
    U_3d = Z2GaugeConfig.random(geom_3d, rng)
    n_g_3d, _ = qc_layout_counts(geom_3d)
    print(f"  n_gauge = {n_g_3d}, slice space = 2^{n_g_3d} = {1 << n_g_3d}")
    for t in range(geom_3d.N_E):
        sb = gauge_qc_bits_general(U_3d, t, geom_3d)
        Ux, Uy, Uz = decode_slice_bits_3d(sb, geom_3d)
        assert np.array_equal(Ux, U_3d.U_x[t]), (
            f"x mismatch at t={t}: got {Ux}, expected {U_3d.U_x[t]}")
        assert np.array_equal(Uy, U_3d.U_y[t]), (
            f"y mismatch at t={t}")
        assert np.array_equal(Uz, U_3d.U_z[t]), (
            f"z mismatch at t={t}: got {Uz}, expected {U_3d.U_z[t]}")
    print(f"  All {geom_3d.N_E} slices round-trip including U_z.  PASS")

    # --- 3D 2×3×2, N_E=2 (a non-square 3D case) ---
    print("\n3D 2×3×2:")
    geom_2x3x2 = LatticeGeometry(Lx=2, Ly=3, Lz=2, N_E=2, m=0.5)
    U_2x3x2 = Z2GaugeConfig.random(geom_2x3x2, rng)
    n_g_223, _ = qc_layout_counts(geom_2x3x2)
    print(f"  n_gauge = {n_g_223}, slice space = 2^{n_g_223} = {1 << n_g_223}")
    for t in range(geom_2x3x2.N_E):
        sb = gauge_qc_bits_general(U_2x3x2, t, geom_2x3x2)
        Ux, Uy, Uz = decode_slice_bits_3d(sb, geom_2x3x2)
        assert np.array_equal(Ux, U_2x3x2.U_x[t])
        assert np.array_equal(Uy, U_2x3x2.U_y[t])
        assert np.array_equal(Uz, U_2x3x2.U_z[t])
    print(f"  All {geom_2x3x2.N_E} slices round-trip.  PASS")

    print("\n" + "=" * 70)
    print("Phase 40 step 4 — ALL PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()
