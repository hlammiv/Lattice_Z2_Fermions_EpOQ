"""
verify_lattice_thermal_ref.py — exact "lattice-Hamiltonian thermal" reference.

For a free fermion many-body T̂_F^n with single-particle operator T_F_single:
    ⟨n̂_i⟩ = [(I + T_F_single)^{-1} · T_F_single]_{i,i}
This is the lattice Hamiltonian thermal expectation, where T̂_F^n is treated
as e^{−β H_lat_eff} from our staggered Euclidean action.

The QC `physical_sector_reference` uses z2_setup's Hamiltonian, which is
a DIFFERENT lattice Hamiltonian from what our action gives. The matching
(EρOQ paper footnote 1) is non-trivial. This script provides the action's
own lattice prediction — what our pipeline at t=0 should reproduce.
"""
from __future__ import annotations
import sys
import numpy as np
sys.path.insert(0, '/home/hlamm/Desktop/QC/logdet/m1_toy')

from action_z2_staggered import LatticeGeometry, Z2GaugeConfig, build_dirac_matrix
from action_z2_metropolis import run_metropolis
from transfer_matrix_bogoliubov import T_F_single_from_companion


def lattice_thermal_n0(geom: LatticeGeometry, U: Z2GaugeConfig) -> complex:
    """⟨n̂_0⟩ from the action's cumulative T̂_F^single (= T̂_F^{N_E−1})."""
    T = T_F_single_from_companion(geom, U)  # V_3 × V_3 complex
    V3 = T.shape[0]
    I = np.eye(V3, dtype=complex)
    # Slater-det formula:  ⟨n̂_i⟩ = [T (I+T)^{-1}]_{ii}
    rho_single = np.linalg.solve(I + T, T)
    return rho_single[0, 0]


def main():
    for N_E in [4, 5]:
        print("=" * 72)
        print(f"N_E = {N_E}  (T̂_F^{{N_E-1}} = T̂_F^{N_E - 1})")
        print("=" * 72)
        geom = LatticeGeometry(Lx=2, Ly=2, N_E=N_E, m=0.5)

        # Trivial U baseline (free case)
        U_triv = Z2GaugeConfig.trivial(geom)
        n0_triv = lattice_thermal_n0(geom, U_triv)
        print(f"  trivial U:    ⟨n̂_0⟩ = {n0_triv.real:+.6f} (imag {n0_triv.imag:+.1e})")

        # Gauge MC average
        mc = run_metropolis(geom, K=1.0, n_sweeps=50, n_warmup=100, seed=2026)
        print(f"  gauge MC: {len(mc.configs)} configs, accept rate {mc.accept_rate:.3f}")
        n0_per_U = []
        for U in mc.configs:
            n0 = lattice_thermal_n0(geom, U)
            n0_per_U.append(n0.real)
        n0_avg = np.mean(n0_per_U)
        n0_err = np.std(n0_per_U) / np.sqrt(len(n0_per_U))
        print(f"  gauge-averaged ⟨n̂_0⟩ = {n0_avg:+.6f} ± {n0_err:.4f}")
        print(f"  per-U range: [{np.min(n0_per_U):+.4f}, {np.max(n0_per_U):+.4f}]")
        # Imag parts for diagnostics
        print(f"  per-U |Im|: max = {np.max([abs(lattice_thermal_n0(geom, U).imag) for U in mc.configs]):.4e}")

    # Compare with physical_sector_reference for context
    print()
    print("=" * 72)
    print("Reference: physical_sector_reference (β=4, z2_setup Hamiltonian)")
    print("=" * 72)
    from action_endtoend_pipeline import physical_sector_reference
    ref = physical_sector_reference(beta=4.0, times=[0.0])
    print(f"  physical_sector_reference(β=4) at t=0:  {ref[0.0]:+.6f}")


if __name__ == "__main__":
    main()
