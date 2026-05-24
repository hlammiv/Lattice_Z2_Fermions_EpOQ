"""Phase 40 step 1 test: 3D geometry data classes.

Verifies:
  - LatticeGeometry(Lx, Ly, N_E, m) (no Lz) → 2D default, V_3=Lx*Ly, eta unchanged.
  - LatticeGeometry(Lx, Ly, Lz=1, N_E, m) → identical to default (back-compat).
  - LatticeGeometry(Lx, Ly, Lz=2, N_E, m) → V_3=Lx*Ly*Lz, site_idx, eta(μ=3).
  - Z2GaugeConfig: 3-axis arrays at Lz=1; 4-axis + U_z at Lz>1.
  - gauge_action raises at Lz>1 (deferred).
  - count_links_and_plaquettes gives the known 3D 2×2×2 link/plaquette counts.
"""
from __future__ import annotations
import numpy as np
import sys

sys.path.insert(0, '/home/hlamm/Desktop/QC/logdet/m1_toy')

from action_z2_staggered import (
    LatticeGeometry, Z2GaugeConfig,
    count_links_and_plaquettes, gauge_action,
)


def main():
    print("=" * 70)
    print("Phase 40 step 1: 3D geometry data classes")
    print("=" * 70)

    # --- 1. 2D default (Lz omitted) ---
    g2d_default = LatticeGeometry(Lx=2, Ly=2, N_E=5, m=0.5)
    assert g2d_default.Lz == 1, "Default Lz should be 1"
    assert g2d_default.V_3 == 4
    assert g2d_default.site_idx(1, 1, 1) == (1 * 2 + 1) * 2 + 1   # = 7
    print(f"\n2D default: V_3={g2d_default.V_3}, "
          f"site_idx(1,1,1)={g2d_default.site_idx(1,1,1)}.  PASS")

    # --- 2. 2D explicit (Lz=1) — must match default ---
    g2d_explicit = LatticeGeometry(Lx=2, Ly=2, N_E=5, m=0.5, Lz=1)
    assert g2d_explicit.V_3 == g2d_default.V_3
    assert g2d_explicit.site_idx(1, 1, 1) == g2d_default.site_idx(1, 1, 1)
    for mu in (0, 1, 2):
        for t in range(3):
            for x in range(2):
                for y in range(2):
                    assert (g2d_explicit.eta(mu, t, x, y)
                            == g2d_default.eta(mu, t, x, y))
    print(f"2D explicit (Lz=1) matches default for V_3, site_idx, eta.  PASS")

    # --- 3. 3D 2×2×2 ---
    g3d = LatticeGeometry(Lx=2, Ly=2, Lz=2, N_E=5, m=0.5)
    assert g3d.V_3 == 8, f"V_3 should be 8 at 2×2×2, got {g3d.V_3}"
    # site_idx: indexing through (t, x, y, z) with z varying fastest
    expected = [(0, 0, 0, 0, 0),
                (0, 0, 0, 1, 1),
                (0, 0, 1, 0, 2),
                (0, 0, 1, 1, 3),
                (0, 1, 0, 0, 4),
                (0, 1, 0, 1, 5),
                (0, 1, 1, 0, 6),
                (0, 1, 1, 1, 7),
                (1, 0, 0, 0, 8)]
    for (t, x, y, z, expect) in expected:
        got = g3d.site_idx(t, x, y, z)
        assert got == expect, f"site_idx{(t,x,y,z)} = {got} ≠ {expect}"
    print(f"3D site_idx ordering (t outer, z inner) verified for 2×2×2.  PASS")

    # eta(μ=3, z): η_z = (-1)^{t+x+y}
    cases = [(0, 0, 0, 0, 1),
             (1, 0, 0, 0, -1),
             (0, 1, 0, 0, -1),
             (0, 0, 1, 0, -1),
             (1, 1, 1, 0, -1),
             (0, 1, 1, 0, 1)]
    for (t, x, y, z, expect) in cases:
        got = g3d.eta(3, t, x, y, z)
        assert got == expect, f"eta(3, {t},{x},{y},{z}) = {got} ≠ {expect}"
    print(f"3D η_z = (-1)^(t+x+y) verified for several cases.  PASS")

    # --- 4. Z2GaugeConfig at Lz=1 vs Lz>1 ---
    rng = np.random.default_rng(0)
    U_2d = Z2GaugeConfig.random(g2d_default, rng)
    assert U_2d.U_x.shape == (5, 1, 2), U_2d.U_x.shape
    assert U_2d.U_y.shape == (5, 2, 1), U_2d.U_y.shape
    assert U_2d.U_t.shape == (4, 2, 2), U_2d.U_t.shape
    assert U_2d.U_z is None
    print(f"\n2D Z2GaugeConfig.random shapes:  "
          f"U_x={U_2d.U_x.shape}, U_y={U_2d.U_y.shape}, "
          f"U_t={U_2d.U_t.shape}, U_z=None.  PASS")

    U_3d = Z2GaugeConfig.random(g3d, rng)
    Lx, Ly, Lz, N_E = 2, 2, 2, 5
    assert U_3d.U_x.shape == (N_E, Lx - 1, Ly, Lz)   # (5, 1, 2, 2)
    assert U_3d.U_y.shape == (N_E, Lx, Ly - 1, Lz)   # (5, 2, 1, 2)
    assert U_3d.U_z.shape == (N_E, Lx, Ly, Lz - 1)   # (5, 2, 2, 1)
    assert U_3d.U_t.shape == (N_E - 1, Lx, Ly, Lz)   # (4, 2, 2, 2)
    print(f"3D Z2GaugeConfig.random shapes:  "
          f"U_x={U_3d.U_x.shape}, U_y={U_3d.U_y.shape}, "
          f"U_z={U_3d.U_z.shape}, U_t={U_3d.U_t.shape}.  PASS")

    U_3d_triv = Z2GaugeConfig.trivial(g3d)
    assert (U_3d_triv.U_x == 1).all()
    assert (U_3d_triv.U_y == 1).all()
    assert (U_3d_triv.U_z == 1).all()
    assert (U_3d_triv.U_t == 1).all()
    print(f"3D Z2GaugeConfig.trivial: all links = +1.  PASS")

    # --- 5. gauge_action raises at Lz>1 (defer to step 2) ---
    try:
        gauge_action(g3d, U_3d, K=1.0)
    except NotImplementedError as e:
        assert "3D" in str(e)
        print(f"\ngauge_action(3D) → NotImplementedError as guarded.  PASS")
    else:
        raise AssertionError("gauge_action(3D) should have raised NotImplementedError")

    # 2D gauge_action still works
    S_2d = gauge_action(g2d_default, U_2d, K=1.0)
    print(f"gauge_action(2D Lz=1) = {S_2d:.4f} (still works, no regression).  PASS")

    # --- 6. count_links_and_plaquettes at 3D 2×2×2 ---
    counts = count_links_and_plaquettes(g3d)
    # Expected for Lx=2, Ly=2, Lz=2, N_E=5:
    #   x_links = N_E * (Lx-1) * Ly * Lz  = 5 * 1 * 2 * 2 = 20
    #   y_links = N_E * Lx * (Ly-1) * Lz  = 5 * 2 * 1 * 2 = 20
    #   z_links = N_E * Lx * Ly * (Lz-1)  = 5 * 2 * 2 * 1 = 20
    #   t_links = (N_E-1) * Lx * Ly * Lz  = 4 * 2 * 2 * 2 = 32
    #   xy_plaq = N_E * (Lx-1) * (Ly-1) * Lz  = 5 * 1 * 1 * 2 = 10
    #   xz_plaq = N_E * (Lx-1) * Ly * (Lz-1)  = 5 * 1 * 2 * 1 = 10
    #   yz_plaq = N_E * Lx * (Ly-1) * (Lz-1)  = 5 * 2 * 1 * 1 = 10
    #   xt_plaq = (N_E-1) * (Lx-1) * Ly * Lz  = 4 * 1 * 2 * 2 = 16
    #   yt_plaq = (N_E-1) * Lx * (Ly-1) * Lz  = 4 * 2 * 1 * 2 = 16
    #   zt_plaq = (N_E-1) * Lx * Ly * (Lz-1)  = 4 * 2 * 2 * 1 = 16
    expected = {
        'x_links': 20, 'y_links': 20, 'z_links': 20, 't_links': 32,
        'total_links': 92,
        'xy_plaq': 10, 'xz_plaq': 10, 'yz_plaq': 10,
        'xt_plaq': 16, 'yt_plaq': 16, 'zt_plaq': 16,
        'total_plaq': 78,
    }
    for k, v in expected.items():
        assert counts[k] == v, f"{k}: got {counts[k]}, expected {v}"
    print(f"\ncount_links_and_plaquettes(2×2×2): all counts match expected.")
    print(f"  links: x={counts['x_links']}, y={counts['y_links']}, "
          f"z={counts['z_links']}, t={counts['t_links']} "
          f"(total {counts['total_links']})")
    print(f"  plaqs: xy={counts['xy_plaq']}, xz={counts['xz_plaq']}, "
          f"yz={counts['yz_plaq']}, xt={counts['xt_plaq']}, "
          f"yt={counts['yt_plaq']}, zt={counts['zt_plaq']} "
          f"(total {counts['total_plaq']}).  PASS")

    # --- 7. count_links_and_plaquettes(2D 2x2) matches old behavior ---
    counts_2d = count_links_and_plaquettes(g2d_default)
    # Lx=2 Ly=2 N_E=5 Lz=1: x_links=5*1*2*1=10, y=5*2*1*1=10, z=0, t=4*2*2*1=16
    # xy_plaq=5*1*1*1=5, xt=4*1*2*1=8, yt=4*2*1*1=8, others 0
    assert counts_2d['x_links'] == 10
    assert counts_2d['y_links'] == 10
    assert counts_2d['z_links'] == 0
    assert counts_2d['t_links'] == 16
    assert counts_2d['total_links'] == 36
    assert counts_2d['xy_plaq'] == 5
    assert counts_2d['xt_plaq'] == 8
    assert counts_2d['yt_plaq'] == 8
    assert counts_2d['xz_plaq'] == 0
    assert counts_2d['yz_plaq'] == 0
    assert counts_2d['zt_plaq'] == 0
    assert counts_2d['total_plaq'] == 21
    print(f"\ncount_links_and_plaquettes(2D 2×2): preserves 2D counts "
          f"(total links {counts_2d['total_links']}, "
          f"total plaqs {counts_2d['total_plaq']}).  PASS")

    print("\n" + "=" * 70)
    print("Phase 40 step 1 — ALL PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()
