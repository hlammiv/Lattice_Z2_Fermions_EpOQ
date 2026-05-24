"""First V_3=6 scaling test: 2×3 OBC, V_3=6 (13 qubits, dim 8192).

Goal: validate the V_3=4-to-V_3=6 extension end-to-end.  The check is:

  1. Build H_QC at 2×3 via build_pauli_terms_general (13 qubits, dim 8192).
  2. Compute ED reference C(t) = ⟨n_0(t) n_0(0)⟩_β by diagonalizing H once
     (eigh) and reusing the spectrum.  Cheaper than four expm calls.
  3. Run gauge-only MC at 2×3 in temporal gauge with W_ORDER=1.
  4. Feed configs into accumulate_C_direct (now general); compare to ED.

Settings (smaller than β=4 V_3=4 production to keep first run under ~20 min):
  β = 2, m = 0.5, a_τ = 0.5  →  N_E = 5
  1 chain × 5000 sweeps
  6 PT replicas
  times: 0.0, 0.5, 1.0
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "4"   # ED is the bottleneck; allow threads

import sys, time
import numpy as np

sys.path.insert(0, '/home/hlamm/Desktop/QC/logdet/m1_toy')

import epoq_classical_sampler as cs
cs.M_MASS = 0.5
import epoq_quantum_simulator as qs
qs.M_MASS = 0.5

from action_z2_staggered import LatticeGeometry
from action_z2_metropolis_pt import run_metropolis_pt
from action_minkowski_stitch import qc_layout_counts
from epoq_pipeline_direct_sampling import accumulate_C_direct


# --- Geometry and couplings ---
LX, LY = 2, 3
M_HAM, G_E_HAM, G_M_HAM, G_HOP = 0.5, 1.0, 0.5, 0.5
BETA = 2.0
W_ORDER = 1
A_TAU = 0.5
N_SWEEPS = 5_000
N_WARMUP = 1_000
N_CHAINS = 1
TIMES = [0.0, 0.5, 1.0]


P2 = {
    'I': np.eye(2, dtype=complex),
    'X': np.array([[0, 1], [1, 0]], dtype=complex),
    'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
    'Z': np.diag([1, -1]).astype(complex),
}


def pauli_string(factors, nq):
    by_q = {q: P2['I'] for q in range(nq)}
    for q, ax in factors:
        by_q[q] = P2[ax]
    r = by_q[nq - 1]
    for q in range(nq - 2, -1, -1):
        r = np.kron(r, by_q[q])
    return r


def build_K_E_ladder(K_E_target, n_replicas=6):
    K_E_low = max(0.05, K_E_target - 0.6)
    return list(np.linspace(K_E_low, K_E_target, n_replicas))


def main():
    print("=" * 78)
    print(f"V_3=6 first scaling test  (Lx={LX}, Ly={LY}, OBC)")
    print(f"  β={BETA}, m={M_HAM}, a_τ={A_TAU}, W_ORDER={W_ORDER}, "
          f"temporal gauge, {N_CHAINS} chain × {N_SWEEPS} sweeps")
    print("=" * 78, flush=True)

    geom_pauli = LatticeGeometry(Lx=LX, Ly=LY, N_E=1, m=M_HAM)
    n_gauge, n_matter = qc_layout_counts(geom_pauli)
    NQ = n_gauge + n_matter
    DIM = 1 << NQ
    print(f"\nQubits: n_gauge={n_gauge}, n_matter={n_matter}, NQ={NQ}, "
          f"DIM=2^{NQ}={DIM}")

    # --- Build H_QC ---
    t0 = time.time()
    terms = cs.build_pauli_terms_general(geom_pauli,
                                          g_e=G_E_HAM, g_m=G_M_HAM,
                                          g_hop=G_HOP, m_mass=M_HAM)
    print(f"\nBuilding H_QC ({len(terms)} Pauli terms, dim {DIM}^2)...",
          flush=True)
    H = sum(c * pauli_string(f, NQ) for c, f in terms)
    H = (H + H.conj().T) / 2
    print(f"  H built in {time.time()-t0:.1f}s.  "
          f"max|H − H†|={np.max(np.abs(H - H.conj().T)):.2e}", flush=True)

    # --- Diagonalize H once for ED ---
    print(f"\nDiagonalizing H (dim {DIM})...", flush=True)
    t0 = time.time()
    eigs, V = np.linalg.eigh(H)
    print(f"  eigh done in {time.time()-t0:.1f}s.  "
          f"E_0={eigs[0]:+.6f}, E_max={eigs[-1]:+.6f}", flush=True)

    # n_0 operator: (I - Z_{q_m(0,0)})/2 where q_m(0,0) = n_gauge
    q_n0 = n_gauge
    n0_op = 0.5 * np.eye(DIM, dtype=complex) - 0.5 * pauli_string(
        [(q_n0, "Z")], NQ)

    # --- ED reference C(t) ---
    print(f"\nED reference (via eigenbasis)...", flush=True)
    t0 = time.time()
    w_beta = np.exp(-BETA * eigs)
    Z_beta = w_beta.sum()
    # rho_ED in eigenbasis is diag(w_beta) / Z_beta; in computational basis:
    #   ρ = V·diag(w_beta)·V† / Z_beta
    # We don't need the full rho explicitly — we evaluate C(t) in eigenbasis.
    n0_eig = V.conj().T @ n0_op @ V
    O_total_eig = {}
    C_ED = {}
    for t in TIMES:
        # U(t) in eigenbasis is diag(e^{-iλt})
        phase = np.exp(-1j * eigs * t)
        UOU_eig = (phase.conj()[:, None] * n0_eig) * phase[None, :]   # U† n0 U
        # The actually-needed operator O_t = UOU(t) · n_0_op in COMPUTATIONAL
        # basis lives in V·O_eig·V†.  For accumulate_C_direct we need this
        # in the original basis, so transform back:
        O_eig = UOU_eig @ n0_eig    # in eigenbasis
        O_total_eig[t] = O_eig
        # C_ED(t) = (1/Z) Σ_n w_beta_n · ⟨n|U_t† n0 U_t · n0|n⟩
        #         = (1/Z) Σ_n w_beta_n · O_eig[n,n]
        C_ED[t] = float((w_beta * np.diag(O_eig).real).sum() / Z_beta)
    print(f"  ED done in {time.time()-t0:.1f}s. "
          + "  ".join(f"C_ED({t})={C_ED[t]:+.5f}" for t in TIMES), flush=True)

    # --- Build O_total in computational basis (needed by accumulate_C_direct) ---
    print(f"\nTransforming O_t to computational basis (4 × dim³ matmul, "
          f"may take ~minute)...", flush=True)
    t0 = time.time()
    O_total = {t: V @ O_total_eig[t] @ V.conj().T for t in TIMES}
    print(f"  done in {time.time()-t0:.1f}s.", flush=True)

    # --- Run MC at V_3=6 ---
    N_E = int(round(BETA / A_TAU)) + 1
    K_E = -0.5 * np.log(np.tanh(A_TAU * G_E_HAM))
    K_M = A_TAU * G_M_HAM
    m_action = A_TAU * M_HAM
    geom_mc = LatticeGeometry(Lx=LX, Ly=LY, N_E=N_E, m=m_action)
    print(f"\nMC geometry: Lx={LX}, Ly={LY}, N_E={N_E}, "
          f"K_E={K_E:.4f}, K_M={K_M:.4f}, m_action={m_action:.4f}")

    K_E_ladder = build_K_E_ladder(K_E)
    C_chains = {t: [] for t in TIMES}
    t_total0 = time.time()
    for ci in range(N_CHAINS):
        seed = 20260524 + ci * 9999
        t0 = time.time()
        print(f"  chain {ci+1}/{N_CHAINS}: MC...", flush=True)
        pt_results, _ = run_metropolis_pt(
            geom_mc, K_E_ladder=K_E_ladder, K_M=K_M,
            n_sweeps=N_SWEEPS, n_warmup=N_WARMUP, swap_every=5,
            seed=seed, action_type='gauge_only', n_workers=6,
            temporal_gauge=True,
        )
        configs = pt_results[-1].configs
        t_mc = time.time() - t0
        print(f"    MC: {t_mc:.0f}s, {len(configs)} configs from top replica",
              flush=True)

        t0 = time.time()
        C, denom, num, n_done = accumulate_C_direct(
            geom_mc, configs, a_tau=A_TAU, times=TIMES, O_total_dict=O_total,
            m_obs=M_HAM, g_hop=G_HOP, w_order=W_ORDER,
        )
        t_acc = time.time() - t0
        for t in TIMES:
            C_chains[t].append(C[t])
        print(f"    accumulate: {t_acc:.0f}s, "
              f"Tr[ρ_H]={denom:+.3e}, "
              + "  ".join(f"C({t})={C[t]:+.5f}" for t in TIMES),
              flush=True)

    print(f"\nMC total: {time.time()-t_total0:.0f}s")
    print(f"\n{'t':>5} | {'C_lat':>15} | {'C_ED':>15} | {'gap':>15}")
    for t in TIMES:
        vals = np.array(C_chains[t])
        if N_CHAINS > 1:
            mean = vals.mean(); sem = vals.std(ddof=1) / np.sqrt(N_CHAINS)
            tag = f"{mean:+.5f}±{sem:.5f}"
        else:
            tag = f"{vals[0]:+.5f}"
        gap = vals.mean() - C_ED[t]
        print(f"  {t:>5.2f} | {tag:>15} | {C_ED[t]:+.5f} | {gap:+.5f}")

    print(f"\nDone.  Smoke check: V_3=6 pipeline runs end-to-end at "
          f"a_τ={A_TAU}.\n"
          f"Gaps include Trotter error (a_τ=0.5 is coarse) AND MC stat error "
          f"(only {N_SWEEPS} sweeps, {N_CHAINS} chain).")


if __name__ == "__main__":
    main()
