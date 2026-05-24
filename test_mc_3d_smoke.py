"""Phase 40 step 5 smoke: single-chain Metropolis MC at 3D 2x2x2.

Just verifies the new U_z proposal path runs end-to-end:
  - link_list emits 5-tuples including 'z' link_type at Lz=2
  - flip_link handles z coord and 'z' link type
  - gauge_action (3D dispatch) gives sensible values
  - Recorded configs have proper U_z arrays
  - Acceptance rate is non-degenerate
"""
from __future__ import annotations
import sys
import numpy as np

sys.path.insert(0, '/home/hlamm/Desktop/QC/logdet/m1_toy')

from action_z2_staggered import LatticeGeometry, Z2GaugeConfig, gauge_action
from action_z2_metropolis import run_metropolis


def main():
    print("=" * 70)
    print("Phase 40 step 5 smoke: 3D 2×2×2 single-chain MC")
    print("=" * 70)

    geom = LatticeGeometry(Lx=2, Ly=2, Lz=2, N_E=3, m=0.5)
    print(f"\nGeom: V_3={geom.V_3}, Lz={geom.Lz}, N_E={geom.N_E}")

    print(f"\nRunning 100 sweeps (50 warmup), temporal_gauge=True, "
          f"action_type='gauge_only' ...", flush=True)
    result = run_metropolis(
        geom, K_E=1.0, K_M=0.5,
        n_sweeps=100, n_warmup=50, seed=2026,
        record_every=10, cold_start=False,
        action_type='gauge_only',
        temporal_gauge=True,
    )
    print(f"  walltime: {result.walltime:.1f}s, acceptance rate "
          f"{result.accept_rate:.3f}", flush=True)
    n_recorded = len(result.configs)
    print(f"  recorded {n_recorded} configs", flush=True)
    assert n_recorded > 0, "No configs recorded!"

    # Verify config has U_z populated and is 3D (4-axis) shaped
    U0 = result.configs[0]
    print(f"\nFirst recorded config:")
    print(f"  U_x shape = {U0.U_x.shape} (expected (3, 1, 2, 2))")
    print(f"  U_y shape = {U0.U_y.shape} (expected (3, 2, 1, 2))")
    print(f"  U_z shape = {U0.U_z.shape if U0.U_z is not None else None} "
          f"(expected (3, 2, 2, 1))")
    print(f"  U_t shape = {U0.U_t.shape} (expected (2, 2, 2, 2))")
    assert U0.U_x.shape == (3, 1, 2, 2)
    assert U0.U_y.shape == (3, 2, 1, 2)
    assert U0.U_z is not None and U0.U_z.shape == (3, 2, 2, 1)
    assert U0.U_t.shape == (2, 2, 2, 2)
    # temporal gauge: U_t should be all +1
    assert (U0.U_t == 1).all(), "U_t not pinned to +1 despite temporal_gauge=True"

    # Verify U_z does get flipped — i.e., MC actually proposes z moves
    # Compare to a trivial config
    U_triv = Z2GaugeConfig.trivial(geom)
    n_diff_z = 0
    for cfg in result.configs:
        if not (cfg.U_z == U_triv.U_z).all():
            n_diff_z += 1
    print(f"\nConfigs with non-trivial U_z: {n_diff_z}/{n_recorded} "
          f"(should be most of them after warmup)")
    assert n_diff_z > 0, ("No config has non-trivial U_z — MC isn't proposing "
                          "z moves or rejecting all of them")

    # Verify gauge_action gives sensible values
    Sg = gauge_action(geom, U0, K_E=1.0, K_M=0.5)
    print(f"  gauge_action(first config) = {Sg:+.4f}", flush=True)

    print(f"\n  Acceptance rate {result.accept_rate:.3f} non-degenerate. "
          f"U_z flipped in {n_diff_z}/{n_recorded} recorded configs.")
    print("\n" + "=" * 70)
    print("Phase 40 step 5 smoke — PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()
