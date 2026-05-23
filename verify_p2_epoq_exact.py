"""
verify_p2_epoq_exact.py — EρOQ-faithful exact reference.

The EρOQ-faithful observable is

    ⟨n_0(0) n_0(t)⟩_EρOQ  =  Tr[ρ_E · n_0(0) n_0(t)]  /  Tr[ρ_E]

where ρ_E is the Euclidean density matrix prepared by the OBC lattice
(NOT the thermal-Tr-with-PBC operator).  In the corner-state representation:

    ρ_E_matrix_element = E_U[ gauge_weight(U) · W_U[Ψ_top, Ψ_bot] ]
                        · |gauge_top(U), Ψ_top⟩⟨gauge_bot(U), Ψ_bot|

The trace closure happens in the 8-qubit Hilbert space (gauge × fermion):

    Tr[ρ_E O] = Σ_U gauge_weight(U) · Σ_pairs W_U · ⟨bot|O|top⟩      (numerator)
    Tr[ρ_E]   = Σ_U gauge_weight(U) · Σ_{(g,Ψ) with g_top=g_bot, Ψ_top=Ψ_bot}
                                       W_U[Ψ,Ψ]                       (denominator)

The standard physical_sector_reference uses PBC-time thermal Tr, which is
a DIFFERENT observable.  This script computes the EρOQ-faithful version
exactly (enumerate all 70 Fock pairs per gauge config, exact QC matrix
elements), to compare against the P2 MC pipeline.

Identical formula as the MC pipeline; difference is enumeration vs sampling.
"""
from __future__ import annotations
import sys
import time
import numpy as np

sys.path.insert(0, '/home/hlamm/Desktop/QC/logdet/m1_toy')
from action_z2_staggered import LatticeGeometry, Z2GaugeConfig, build_dirac_matrix
from action_z2_metropolis import run_metropolis
from action_minkowski_stitch import (
    matrix_element_minkowski, qc_basis_index, static_observable_diagonal,
)
from action_corner_direct_v2 import (
    fock_pair_weights_array, bitcount,
)
from action_endtoend_pipeline import physical_sector_reference


def run_epoq_exact(geom: LatticeGeometry, K: float, times: list[float],
                  n_gauge: int = 20, n_warmup_gauge: int = 50,
                  seed: int = 2026, n_trotter: int = 200) -> dict:
    """EρOQ-faithful exact reference: same formula as pipeline, full
    enumeration of Fock pairs per gauge config (no MC on Fock side).

    Returns:
       per-time C(t) (numerator / denominator)
       per-time numerator, denominator
    """
    rng = np.random.default_rng(seed)

    print(f"Phase 1: gauge MC (n_gauge={n_gauge})...")
    t0 = time.time()
    mc = run_metropolis(geom, K=K, n_sweeps=n_gauge,
                       n_warmup=n_warmup_gauge, seed=seed)
    print(f"  done in {time.time() - t0:.1f}s, accept rate {mc.accept_rate:.3f}")

    print(f"Phase 2+3: enumerate Fock pairs + exact QC matrix elements...")
    t0 = time.time()
    num = {t: 0.0 for t in times}
    denom = 0.0
    n_eval = 0
    for U_idx, U in enumerate(mc.configs):
        if U_idx % 5 == 0:
            print(f"  gauge {U_idx}/{len(mc.configs)}  "
                  f"(t={time.time() - t0:.1f}s, n_eval={n_eval})...",
                  flush=True)
        pairs, amps, _ = fock_pair_weights_array(geom, U)
        for k in range(len(pairs)):
            psi_i = int(pairs[k, 0])      # = Ψ_top (t=0)
            psi_j = int(pairs[k, 1])      # = Ψ_bot (t=N-1)
            w = complex(amps[k])
            # Hamiltonian basis indices
            bits_top = qc_basis_index(geom, U, 0, psi_i)
            bits_bot = qc_basis_index(geom, U, geom.N_E - 1, psi_j)
            is_diag = (bits_top == bits_bot)
            n0_j = (psi_j >> 0) & 1
            for t in times:
                m_el = matrix_element_minkowski(
                    geom, U, psi_i, psi_j, t=t, n_trotter=n_trotter)
                num[t] += (w * n0_j * m_el).real
                n_eval += 1
            if is_diag:
                denom += w.real
    t_total = time.time() - t0
    print(f"  done in {t_total:.1f}s, total mat_el calls = {n_eval}")

    # Final ratio
    out = {}
    for t in times:
        out[t] = num[t] / denom if abs(denom) > 1e-12 else float('nan')

    return {
        'C': out, 'num': num, 'denom': denom,
        'walltime_qc': t_total,
        'gauge_configs': mc.configs,
    }


def main():
    # N_E = 5: T̂_F^{N_E-1} = T̂_F^4 = (T̂_F^2)^2 positive-definite (per M-K).
    # β_eff = (N_E-1)·a = 4 still matches the reference at β=4.
    geom = LatticeGeometry(Lx=2, Ly=2, N_E=5, m=0.5)
    K = 1.0
    times = [0.0, 0.5, 1.0, 2.0]

    print("=" * 72)
    print("EρOQ-faithful exact reference (OBC, N_E=5 → T̂_F^4 positive-def)")
    print("=" * 72)
    result = run_epoq_exact(geom, K=K, times=times,
                            n_gauge=6, n_warmup_gauge=50, seed=2026,
                            n_trotter=200)

    print("\n" + "=" * 72)
    print("Comparison with PBC Tr-reference at β = N_E - 1 = 4")
    print("=" * 72)
    pbc_ref = physical_sector_reference(beta=float(geom.N_E - 1), times=times)
    print(f"\n  {'t':>5} | {'EρOQ exact':>14} | {'PBC Tr-ref':>14} | "
          f"{'difference':>12}")
    for t in times:
        e = result['C'][t]
        p = pbc_ref[t]
        d = e - p
        print(f"  {t:>5.2f} | {e:>14.6f} | {p:>14.6f} | {d:>+12.6f}")

    print(f"\n  denominator = {result['denom']:+.6f}")
    print(f"  numerator at t=0 = {result['num'][0.0]:+.6f}")
    print(f"\n(EρOQ exact and PBC Tr-ref should DIFFER — EρOQ uses OBC-Euclidean")
    print(f" matrix elements, while PBC-Tr uses thermal trace.  The MC pipeline")
    print(f" should agree with EρOQ exact, NOT with PBC Tr-ref.)")


if __name__ == "__main__":
    main()
