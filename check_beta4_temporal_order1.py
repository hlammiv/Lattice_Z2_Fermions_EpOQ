"""β=4, m=2 production verification with the full Phase 37/38 fix:
  - W_ORDER = 1 (time-ordered Lie, NOT palindrome)
  - temporal_gauge = True (U_t ≡ +1, sample only U_x, U_y)
  - ρ̃ → (ρ̃ + ρ̃†)/2 Hermitization (applied implicitly via the
    half-and-half sum of Tr[ρ̃·O_t] + Tr[ρ̃†·O_t] per config)

Phase 39 refactor (2026-05-24):
  Replaces the inline 256×256 ρ̃ assembly with direct accumulation of
  C(t) numerator/denominator scalars via
  epoq_pipeline_direct_sampling.accumulate_C_direct.  Identity vs the
  prior path was verified to machine precision in
  check_v3_4_direct_vs_rho.py — see commit history.

  Diagnostic outputs (‖Δρ‖_F, min_eig, #neg eigs) from the original
  Phase 38 verification are dropped; they're recorded in
  project_phase38_complete_fix_verified for reference.

If both fixes together give clean Trotter convergence to ED, this is
the methodology-paper-ready production setting.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import sys, time
import numpy as np
from scipy.linalg import expm

sys.path.insert(0, '/home/hlamm/Desktop/QC/logdet/m1_toy')

import epoq_classical_sampler as cs
cs.M_MASS = 2.0
import epoq_quantum_simulator as qs
qs.M_MASS = 2.0

from action_z2_staggered import LatticeGeometry
from action_z2_metropolis_pt import run_metropolis_pt
from epoq_pipeline_direct_sampling import accumulate_C_direct

M_HAM, G_E_HAM, G_M_HAM, G_HOP = 2.0, 1.0, 0.5, 0.5
BETA = 4.0
W_ORDER = 1   # the path-integral-correct ordering

A_TAUS = [0.5, 0.25]
N_CHAINS = 4
N_SWEEPS = 30_000
N_WARMUP = 3_000
TIMES = [0.0, 0.5, 1.0, 2.0]


def build_K_E_ladder(K_E_target, n_replicas=6):
    K_E_low = max(0.05, K_E_target - 0.6)
    return list(np.linspace(K_E_low, K_E_target, n_replicas))


def build_H_QC():
    P2 = {'I': np.eye(2, dtype=complex),
          'X': np.array([[0, 1], [1, 0]], dtype=complex),
          'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
          'Z': np.diag([1, -1]).astype(complex)}
    def pauli(factors, nq=8):
        by_q = {q: P2['I'] for q in range(nq)}
        for q, ax in factors:
            by_q[q] = P2[ax]
        r = by_q[nq - 1]
        for q in range(nq - 2, -1, -1):
            r = np.kron(r, by_q[q])
        return r
    H = sum(c * pauli(f) for c, f in cs.build_pauli_terms())
    return (H + H.conj().T) / 2


def run_one(a_tau, O_total, C_ED):
    N_E = int(round(BETA / a_tau)) + 1
    K_E = -0.5 * np.log(np.tanh(a_tau * G_E_HAM))
    K_M = a_tau * G_M_HAM
    m_action = a_tau * M_HAM
    geom = LatticeGeometry(Lx=2, Ly=2, N_E=N_E, m=m_action)
    K_E_ladder = build_K_E_ladder(K_E)
    print(f"\n  a_τ={a_tau}, N_E={N_E}, K_E={K_E:.4f}, K_M={K_M:.4f}")
    print(f"  temporal gauge: U_t = +1 fixed, only U_x/U_y sampled\n", flush=True)

    C_chains = {t: [] for t in TIMES}
    t_total0 = time.time()
    for ci in range(N_CHAINS):
        seed = 2026 + ci * 11111
        t0 = time.time()
        pt_results, _ = run_metropolis_pt(
            geom, K_E_ladder=K_E_ladder, K_M=K_M,
            n_sweeps=N_SWEEPS, n_warmup=N_WARMUP, swap_every=5,
            seed=seed, action_type='gauge_only', n_workers=6,
            temporal_gauge=True)
        configs = pt_results[-1].configs

        C, _, _, _ = accumulate_C_direct(
            geom, configs, a_tau=a_tau, times=TIMES,
            O_total_dict=O_total, m_obs=M_HAM, g_hop=G_HOP,
            w_order=W_ORDER,
        )
        for t in TIMES:
            C_chains[t].append(C[t])
        twall = time.time() - t0
        print(f"    chain {ci+1}/{N_CHAINS}, walltime={twall:.0f}s, "
              + "  ".join(f"C({t})={C_chains[t][-1]:+.5f}" for t in TIMES),
              flush=True)

    total_wall = time.time() - t_total0
    print(f"\n  Cross-chain at a_τ={a_tau} (walltime {total_wall:.0f}s):")
    results = {}
    for t in TIMES:
        vals = np.array(C_chains[t])
        mean = vals.mean()
        sem = vals.std(ddof=1) / np.sqrt(N_CHAINS)
        gap = mean - C_ED[t]
        results[t] = (mean, sem, gap)
        print(f"  t={t}: C_lat={mean:+.5f}±{sem:.5f}, C_ED={C_ED[t]:+.5f}, "
              f"gap={gap:+.5f}")
    return results


def main():
    print(f"β=4 m=2 PRODUCTION VERIFICATION  (direct-sampling refactor)")
    print(f"  W_ORDER={W_ORDER} (Lie time-ordered), temporal_gauge=True")
    print(f"  {N_CHAINS} chains × {N_SWEEPS} sweeps, 6 PT replicas each", flush=True)

    H = build_H_QC()

    Z = np.diag([1.0, -1.0]).astype(complex)
    n0_minus = np.array([[1.0]], dtype=complex)
    for k in reversed(range(8)):
        m = Z if k == 4 else np.eye(2, dtype=complex)
        n0_minus = np.kron(n0_minus, m)
    n0_op = 0.5 * np.eye(256, dtype=complex) - 0.5 * n0_minus

    UOU = {t: expm(-1j * H * t).conj().T @ n0_op @ expm(-1j * H * t) for t in TIMES}
    O_total = {t: UOU[t] @ n0_op for t in TIMES}

    rho_ED = expm(-BETA * H)
    rho_ED_n = rho_ED / np.trace(rho_ED).real
    C_ED = {t: np.real(np.trace(rho_ED_n @ UOU[t] @ n0_op)) for t in TIMES}
    print(f"\nED reference: " + "  ".join(f"C({t})={C_ED[t]:+.5f}" for t in TIMES))

    results = {}
    for a_tau in A_TAUS:
        results[a_tau] = run_one(a_tau, O_total, C_ED)

    print("\n" + "=" * 80)
    print("β=4 m=2 — Production (ORDER=1 + temporal gauge) vs prior palindrome")
    print("=" * 80)
    PALINDROME_PRIOR = {
        0.5:  {0.0: +0.00065, 0.5: +0.00481, 1.0: -0.02573, 2.0: -0.01479},
        0.25: {0.0: -0.00573, 0.5: +0.00668, 1.0: -0.02220, 2.0: -0.01541},
    }
    for a_tau in A_TAUS:
        print(f"\na_τ={a_tau}")
        print(f"  {'t':>5} | {'new gap':>12} | {'palindrome gap':>15} | "
              f"{'|new|/|old|':>13}")
        for t in TIMES:
            gap_new = results[a_tau][t][2]
            gap_old = PALINDROME_PRIOR[a_tau][t]
            ratio = abs(gap_new) / abs(gap_old) if gap_old != 0 else float('inf')
            print(f"  {t:>5.2f} | {gap_new:>+12.5f} | {gap_old:>+15.5f} | "
                  f"{ratio:>13.3f}")


if __name__ == "__main__":
    main()
