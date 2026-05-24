"""Identity-level comparison of direct C(t) accumulator vs dense ρ̃ assembly.

The two methods should agree to machine precision (~10⁻¹²) on the SAME
config set — this is an algebraic identity, NOT a statistical
agreement.  MC error doesn't enter because both paths process the
identical sequence of (U, W_psi, g_top, g_bot) data.

Test config: β=4, m=2, a_τ=0.5, 1 chain × 30k sweeps in temporal gauge,
W_ORDER=1.  Matches check_beta4_temporal_order1.py production settings
at a single (chain, a_τ) corner of its grid.

What we check:
  1. Tr[ρ_H]_direct        == Tr[ρ_H]_dense        (denominator)
  2. Tr[ρ_H · O_t]_direct  == Tr[ρ_H · O_t]_dense  (numerator per t)
  3. C(t)_direct           == C(t)_dense           (final observable)
Each comparison reports max abs diff, expected ~10⁻¹² for the trace
quantities and ~10⁻¹⁴ for the normalized ratio.
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
from epoq_pipeline_direct_sampling import accumulate_C_direct, assemble_rho_dense


M_HAM, G_E_HAM, G_M_HAM, G_HOP = 2.0, 1.0, 0.5, 0.5
BETA = 4.0
W_ORDER = 1
A_TAU = 0.5
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


def main():
    print("=" * 78)
    print("IDENTITY CHECK: direct-sampling C(t) vs dense ρ̃ assembly")
    print(f"  β={BETA}, m={M_HAM}, a_τ={A_TAU}, W_ORDER={W_ORDER}, "
          f"1 chain × {N_SWEEPS} sweeps, temporal gauge")
    print("=" * 78, flush=True)

    H = build_H_QC()
    Z = np.diag([1.0, -1.0]).astype(complex)
    n0_minus = np.array([[1.0]], dtype=complex)
    for k in reversed(range(8)):
        m = Z if k == 4 else np.eye(2, dtype=complex)
        n0_minus = np.kron(n0_minus, m)
    n0_op = 0.5 * np.eye(256, dtype=complex) - 0.5 * n0_minus
    UOU = {t: expm(-1j * H * t).conj().T @ n0_op @ expm(-1j * H * t) for t in TIMES}
    O_total = {t: UOU[t] @ n0_op for t in TIMES}

    N_E = int(round(BETA / A_TAU)) + 1
    K_E = -0.5 * np.log(np.tanh(A_TAU * G_E_HAM))
    K_M = A_TAU * G_M_HAM
    m_action = A_TAU * M_HAM
    geom = LatticeGeometry(Lx=2, Ly=2, N_E=N_E, m=m_action)
    K_E_ladder = build_K_E_ladder(K_E)
    print(f"\nGeometry: V_3=4, N_E={N_E}, K_E={K_E:.4f}, K_M={K_M:.4f}")
    print(f"Sampling {N_SWEEPS} sweeps (warmup {N_WARMUP})...", flush=True)

    t0 = time.time()
    pt_results, _ = run_metropolis_pt(
        geom, K_E_ladder=K_E_ladder, K_M=K_M,
        n_sweeps=N_SWEEPS, n_warmup=N_WARMUP, swap_every=5,
        seed=20260524, action_type='gauge_only', n_workers=6,
        temporal_gauge=True,
    )
    configs = pt_results[-1].configs
    n_configs = len(configs)
    print(f"  MC done in {time.time()-t0:.0f}s.  Got {n_configs} configs from the top replica.",
          flush=True)

    print("\nRunning direct-sampling accumulator ...", flush=True)
    t0 = time.time()
    C_direct, denom_direct, num_direct, n_done = accumulate_C_direct(
        geom, configs, a_tau=A_TAU, times=TIMES, O_total_dict=O_total,
        m_obs=M_HAM, g_hop=G_HOP, w_order=W_ORDER,
    )
    t_direct = time.time() - t0
    print(f"  Direct done in {t_direct:.0f}s on {n_done} configs.", flush=True)

    print("\nRunning dense ρ̃ reference assembly ...", flush=True)
    t0 = time.time()
    rho = assemble_rho_dense(
        geom, configs, a_tau=A_TAU, m_obs=M_HAM, g_hop=G_HOP, w_order=W_ORDER,
    )
    t_dense = time.time() - t0
    print(f"  Dense done in {t_dense:.0f}s.", flush=True)
    rho_H = (rho + rho.conj().T) / 2
    tr_rho_H = np.trace(rho_H).real
    num_dense = {t: np.trace(rho_H @ O_total[t]) for t in TIMES}
    C_dense = {t: num_dense[t].real / tr_rho_H for t in TIMES}

    # Pass thresholds:
    # Traces sum 30k contributions of order ~1e6, so float64 roundoff
    # bound is N·eps ~ 30k · 2.2e-16 = 7e-12 relative.  We allow 1e-10
    # relative for safety.  C(t) is a normalized ratio of order 1e-2;
    # the final precision is closer to absolute eps, so we use 1e-12
    # absolute (well above the ~1e-16 absolute floor seen empirically).
    REL_TOL_TRACE = 1e-10
    ABS_TOL_C     = 1e-12

    print("\n" + "=" * 78)
    print(f"IDENTITY DIFFS  (rel tol {REL_TOL_TRACE:g} on traces, "
          f"abs tol {ABS_TOL_C:g} on C(t))")
    print("=" * 78)
    print(f"\nDenominator Tr[ρ_H]:")
    print(f"  direct = {denom_direct:+.15e}")
    print(f"  dense  = {tr_rho_H:+.15e}")
    d_denom = abs(denom_direct - tr_rho_H)
    rel_denom = d_denom / max(abs(tr_rho_H), 1e-30)
    print(f"  |Δ|    = {d_denom:.3e}   |Δ|/|val| = {rel_denom:.3e}   "
          f"({'PASS' if rel_denom < REL_TOL_TRACE else 'FAIL'})")

    print(f"\nNumerator Tr[ρ_H · O_t]:")
    print(f"  {'t':>5} | {'direct (Re)':>18} | {'dense (Re)':>18} | "
          f"{'|Δ Re|':>10} | {'|Δ|/|val|':>10}")
    max_num_rel = 0.0
    for t in TIMES:
        n_d = num_direct[t]; n_e = num_dense[t]
        d_re = abs(n_d.real - n_e.real)
        rel_re = d_re / max(abs(n_e.real), 1e-30)
        max_num_rel = max(max_num_rel, rel_re)
        print(f"  {t:>5.2f} | {n_d.real:+.10e} | {n_e.real:+.10e} | "
              f"{d_re:.3e} | {rel_re:.3e}")
    print(f"  max |Δ|/|val|: {max_num_rel:.3e}  "
          f"({'PASS' if max_num_rel < REL_TOL_TRACE else 'FAIL'})")

    print(f"\nFinal C(t):")
    print(f"  {'t':>5} | {'direct':>15} | {'dense':>15} | {'|Δ|':>10}")
    max_C = 0.0
    for t in TIMES:
        d = abs(C_direct[t] - C_dense[t])
        max_C = max(max_C, d)
        print(f"  {t:>5.2f} | {C_direct[t]:+.10e} | {C_dense[t]:+.10e} | {d:.3e}")
    print(f"  max |Δ|: {max_C:.3e}  ({'PASS' if max_C < ABS_TOL_C else 'FAIL'})")

    print(f"\nTiming: direct {t_direct:.0f}s vs dense {t_dense:.0f}s "
          f"on {n_configs} configs.")

    all_pass = (rel_denom < REL_TOL_TRACE) and (max_num_rel < REL_TOL_TRACE) \
               and (max_C < ABS_TOL_C)
    print("\n" + ("ALL PASS — identity verified at machine precision"
                  if all_pass else "FAILURE(S) ABOVE"))
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
