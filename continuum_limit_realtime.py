"""
continuum_limit_realtime.py — full real-time pipeline continuum-limit study.

Sweep a_τ ∈ {1, 0.5, 0.25, 0.125} at β = 4 fixed (N_E ∈ {5, 9, 17, 33}).
For each, run the v3 pipeline (gauge MC + Trotter W corner sampling + Minkowski
matrix elements) and report ⟨n_0(t) · n_0(0)⟩ at t ∈ {0, 0.5, 1, 2} with
bootstrap error bars.

Reference: Gauss-physical staggered-KS Hamiltonian at β=4 (`staggered_ks_reference`).
Pipeline should converge to reference as a_τ → 0 (no explicit P_Gauss needed;
gauge invariance + continuum limit is the standard lattice QCD path).

Coupling scaling at each a_τ:
  K_M_action = a_τ · g_M_Ham = a_τ · 0.5      (diagonal Trotter)
  K_E_action = -(1/2) log tanh(a_τ · g_E_Ham) (Suzuki for transverse Ising)
  m_action   = a_τ · m_Ham                    (Trotter mass)
  m_obs / g_hop / observable: Hamiltonian values (a_τ enters via expm inside T̂_F)
"""
from __future__ import annotations
import os
import pickle
import sys
import time
import numpy as np

sys.path.insert(0, '/home/hlamm/Desktop/QC/logdet/m1_toy')

from action_z2_staggered import LatticeGeometry
from action_endtoend_pipeline_v3 import run_action_p3_pipeline


CHECKPOINT_DIR = "/tmp/continuum_realtime_ckpt"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# Hamiltonian-side targets (z2_setup defaults)
G_E_HAM = 1.0
G_M_HAM = 0.5
M_HAM = 0.5
G_HOP = 0.5
BETA = 4.0


def gauss_physical_reference(times):
    """Gauss-law-physical staggered-KS thermal ⟨n_0(t)·n_0(0)⟩ at β = BETA.

    Uses `physical_sector_reference` from `action_endtoend_pipeline.py`,
    which now (post-Phase-16) uses the staggered Hamiltonian built by
    `epoq_classical_sampler.build_pauli_terms` (with K-S spatial η and
    Jordan-Wigner strings).  This is the right reference for the staggered
    everywhere pipeline.
    """
    from action_endtoend_pipeline import physical_sector_reference
    return physical_sector_reference(beta=BETA, times=list(times))


def run_at_a_tau(a_tau, times, n_gauge, n_per_sector, n_chains, obs_n_workers,
                 seed=2026):
    """Run v3 pipeline at one (a_τ, β=4) point."""
    K_M = a_tau * G_M_HAM
    K_E = -0.5 * np.log(np.tanh(a_tau * G_E_HAM))
    m_action = a_tau * M_HAM
    N_E = int(round(BETA / a_tau)) + 1
    beta_eff = (N_E - 1) * a_tau

    print(f"\n{'=' * 78}")
    print(f"a_τ = {a_tau:.4f}, N_E = {N_E}, β_eff = {beta_eff:.4f}")
    print(f"  Action: m={m_action:.4f}, K_E={K_E:.4f}, K_M={K_M:.4f}")
    print(f"  Observable: m_obs={M_HAM}, g_hop={G_HOP}")
    print(f"  Stats: n_gauge={n_gauge}, n_per_sector={n_per_sector}, "
          f"n_chains={n_chains}, obs_workers={obs_n_workers}")
    print(f"{'=' * 78}")

    geom = LatticeGeometry(Lx=2, Ly=2, N_E=N_E, m=m_action)

    t0 = time.time()
    result = run_action_p3_pipeline(
        geom, K_E=K_E, K_M=K_M, times=times,
        n_gauge=n_gauge, n_per_sector=n_per_sector,
        n_warmup_gauge=max(500, n_gauge // 4),
        seed=seed, n_trotter_real=200,
        n_chains=n_chains, action_type='forward', plaq_flip_every=5,
        a_tau=a_tau, m_obs=M_HAM, g_hop=G_HOP,
        obs_n_workers=obs_n_workers,
    )
    result['a_tau'] = a_tau
    result['N_E'] = N_E
    result['beta_eff'] = beta_eff
    result['K_E'] = K_E
    result['K_M'] = K_M
    result['m_action'] = m_action
    result['total_walltime'] = time.time() - t0
    return result


def main():
    # a_τ=1.0 is sign-problem dominated (⟨sign⟩≈-0.4); skip it.  Smaller values
    # have ⟨sign⟩≥0.79 (a_τ=0.5), 0.95 (0.25), 0.99 (0.125), giving usable stats.
    # Extending to 0.0625 and 0.03125 to test continuum convergence after the
    # Phase 18 bug fix.
    a_tau_list = [0.5, 0.25, 0.125, 0.0625, 0.03125]
    times = [0.0, 0.5, 1.0, 2.0]
    n_gauge = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    n_per_sector = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    n_chains = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    obs_n_workers = int(sys.argv[4]) if len(sys.argv) > 4 else 8

    print(f"Reference Hamiltonian: g_E={G_E_HAM}, g_M={G_M_HAM}, m={M_HAM}, "
          f"g_hop={G_HOP}, β={BETA}")
    ref = gauss_physical_reference(times)
    print(f"\nGauss-physical reference ⟨n_0(t)·n_0(0)⟩:")
    for t in times:
        print(f"  C({t}) = {ref[t]:+.5f}")
    print()

    results = []
    for a_tau in a_tau_list:
        ckpt = os.path.join(CHECKPOINT_DIR, f"a_tau_{a_tau:.4f}.pkl")
        if os.path.exists(ckpt):
            print(f"\n[checkpoint] loading {ckpt}", flush=True)
            with open(ckpt, 'rb') as f:
                r = pickle.load(f)
        else:
            r = run_at_a_tau(a_tau, times, n_gauge, n_per_sector,
                              n_chains, obs_n_workers)
            with open(ckpt, 'wb') as f:
                pickle.dump(r, f)
            print(f"[checkpoint] saved {ckpt}", flush=True)
        results.append(r)
        print(f"\n  Walltime: {r['total_walltime']:.0f}s "
              f"(MC: {r['walltime']['mc']:.0f}s, "
              f"obs: {r['walltime']['obs_total']:.0f}s)")
        for t in times:
            m, e = r['C'][t], r['err'][t]
            ref_t = ref[t]
            sigma_off = (m - ref_t) / e if e > 0 else float('inf')
            print(f"  C({t:.1f}) = {m:+.5f} ± {e:.5f}  "
                  f"(ref = {ref_t:+.5f}, Δ/σ = {sigma_off:+.1f})")

    # Summary table
    print(f"\n{'=' * 78}")
    print("CONTINUUM-LIMIT SUMMARY — full real-time pipeline, β=4")
    print(f"{'=' * 78}")
    print(f"{'a_τ':>8} {'N_E':>5} {'n_smp':>8} | "
          + " ".join(f"{'C(' + f'{t:.1f}' + ')':>16}" for t in times))
    for r in results:
        line = f"{r['a_tau']:>8.4f} {r['N_E']:>5d} {r['n_samples']:>8d} |"
        for t in times:
            line += f" {r['C'][t]:>+8.4f}±{r['err'][t]:.4f}"
        print(line)
    print(f"\n{'ref':>8} {'∞':>5} {'-':>8} |"
          + " ".join(f" {ref[t]:>+8.4f}{'-':>7}" for t in times))


if __name__ == "__main__":
    main()
