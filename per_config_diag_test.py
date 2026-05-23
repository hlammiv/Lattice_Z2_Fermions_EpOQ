"""Test: at each MC gauge config, directly compute the W-diagonal contribution
to C(0) and aggregate across the ensemble with full sign reweighting. This
bypasses the v3 corner sampler entirely — the diag part of C(0) at each U is
just Σ_psi W_U[psi,psi] · n_0(psi) / Σ_psi W_U[psi,psi].

We compute THREE different aggregations:
  (A) PIPELINE-STYLE: ⟨1{gauge_top=gauge_bot} · sign(det) · Σ W · n_0⟩
                       / ⟨1{gauge_top=gauge_bot} · sign(det) · Σ W⟩
      — should match the pipeline's C(0) (~+0.0007)
  (B) UNRESTRICTED:   ⟨sign(det) · Σ W · n_0⟩ / ⟨sign(det) · Σ W⟩
      — should match unprojected ED (+0.0135) if OBC Lagrangian gives PBC trace
  (C) NO sign reweighting: ⟨Σ W · n_0⟩ / ⟨Σ W⟩ — sanity check

If (B) ≈ ED, the diag-only restriction in the pipeline is the problem.
If (B) ≠ ED, the issue is structural (OBC vs PBC, or something else).
"""
from __future__ import annotations
import sys, time
import numpy as np

sys.path.insert(0, '/home/hlamm/Desktop/QC/logdet/m1_toy')
import epoq_quantum_simulator; epoq_quantum_simulator.M_MASS = 2.0
import epoq_classical_sampler; epoq_classical_sampler.M_MASS = 2.0

from action_z2_staggered import LatticeGeometry
from action_z2_metropolis_pt import run_metropolis_pt, electric_plaquette_sum
from transfer_matrix_kbc_trotterized import compute_combined_weight_trotter
from action_minkowski_stitch import gauge_qc_bits_from_slice

M_HAM, G_E_HAM, G_M_HAM, BETA = 2.0, 1.0, 0.5, 4.0
A_TAU = 0.125
N_WARMUP, N_SWEEPS, SWAP_EVERY = 5000, 2000, 5   # shorter run; enough for per-config stats
SEED = 2026


def build_K_E_ladder(K_E_target, n_replicas=8):
    K_E_low = max(0.30, K_E_target - 0.7)
    if K_E_low >= K_E_target - 0.10:
        K_E_low = K_E_target - 0.10
    return list(np.linspace(K_E_low, K_E_target, n_replicas))


def main():
    print("=" * 80, flush=True)
    print(f"Per-config diag C(0) test @ a_τ={A_TAU}, m={M_HAM}, β={BETA}")
    print(f"  PT-exact one instance, warm={N_WARMUP} swp={N_SWEEPS}")
    print("=" * 80, flush=True)

    # ED reference
    from scipy.linalg import expm
    P2 = {'I': np.eye(2, dtype=complex), 'X':np.array([[0,1],[1,0]], dtype=complex),
          'Y':np.array([[0,-1j],[1j,0]], dtype=complex),
          'Z':np.diag([1,-1]).astype(complex)}
    def pauli(factors, nq=8):
        by_q = {q: P2['I'] for q in range(nq)}
        for q, ax in factors: by_q[q] = P2[ax]
        r = by_q[nq-1]
        for q in range(nq-2, -1, -1): r = np.kron(r, by_q[q])
        return r
    H = sum(c * pauli(f) for c, f in epoq_classical_sampler.build_pauli_terms())
    H = (H + H.conj().T) / 2
    n0 = 0.5 * np.eye(256, dtype=complex) - 0.5 * pauli([(4, 'Z')])
    rho = expm(-BETA * H); Z = np.trace(rho).real
    ED_n0 = (np.trace(rho @ n0) / Z).real
    print(f"\nED unprojected ⟨n_0⟩ = {ED_n0:+.6f}", flush=True)

    # PT MC
    K_E_target = -0.5 * np.log(np.tanh(A_TAU * G_E_HAM))
    K_M = A_TAU * G_M_HAM
    N_E = int(round(BETA / A_TAU)) + 1
    m_action = A_TAU * M_HAM
    geom = LatticeGeometry(Lx=2, Ly=2, N_E=N_E, m=m_action)
    K_E_ladder = build_K_E_ladder(K_E_target)
    print(f"PT MC: N_E={N_E}, K_E_target={K_E_target:.4f}, ladder=[...{K_E_ladder[0]:.2f}..{K_E_ladder[-1]:.2f}]", flush=True)

    t0 = time.time()
    pt_results, sw_stats = run_metropolis_pt(
        geom, K_E_ladder=K_E_ladder, K_M=K_M,
        n_sweeps=N_SWEEPS, n_warmup=N_WARMUP, swap_every=SWAP_EVERY,
        seed=SEED, action_type='forward')
    t_mc = time.time() - t0
    print(f"PT MC walltime: {t_mc:.0f}s", flush=True)

    top = pt_results[-1]
    configs = top.configs
    sign_history = np.array(top.sign_history, dtype=int)
    print(f"top replica: {len(configs)} configs, ⟨sgn⟩={sign_history.mean():+.3f}",
          flush=True)

    # For each config: build W and compute Σ W[psi,psi], Σ W[psi,psi]·n_0(psi),
    # and gauge_match indicator.
    print(f"Computing per-config W diagonal sums...", flush=True)
    t0 = time.time()
    V3 = geom.V_3
    n_configs = len(configs)
    sum_W_diag = np.zeros(n_configs)
    sum_W_diag_n0 = np.zeros(n_configs)
    gauge_match = np.zeros(n_configs, dtype=int)
    for i, U in enumerate(configs):
        W, _ = compute_combined_weight_trotter(
            geom, U, a_tau=A_TAU, K_E=0.0, K_M=0.0, m_obs=M_HAM, g_hop=0.5)
        diagW = np.real(np.diag(W))
        # In v3/W convention, site (0,0) occupation = bit (V_3-1) of lex index
        n0_by_state = np.array([(s >> (V3 - 1)) & 1 for s in range(W.shape[0])],
                                dtype=float)
        sum_W_diag[i] = diagW.sum()
        sum_W_diag_n0[i] = (diagW * n0_by_state).sum()
        gtop = gauge_qc_bits_from_slice(U, geom.N_E - 1)
        gbot = gauge_qc_bits_from_slice(U, 0)
        gauge_match[i] = 1 if gtop == gbot else 0
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{n_configs} configs, {time.time()-t0:.1f}s", flush=True)
    t_perconf = time.time() - t0
    print(f"Per-config build: {t_perconf:.0f}s", flush=True)

    print(f"\nGauge-match fraction across MC: {gauge_match.mean():.3f}",
          flush=True)

    # Aggregation
    print()
    print("=" * 80)
    print("Three aggregations of ⟨n_0⟩ over the MC ensemble:")
    print("=" * 80)

    # (A) Pipeline-style: with g_match restriction and sign reweighting
    num_A = (gauge_match * sign_history * sum_W_diag_n0).sum()
    den_A = (gauge_match * sign_history * sum_W_diag).sum()
    A_val = num_A / den_A if abs(den_A) > 1e-30 else float('nan')
    print(f"  (A) ⟨g_match · sgn · ΣW·n_0⟩ / ⟨g_match · sgn · ΣW⟩ = {A_val:+.6f}")
    print(f"      (pipeline-style; n_match={gauge_match.sum()})")

    # (B) Unrestricted but sign-reweighted
    num_B = (sign_history * sum_W_diag_n0).sum()
    den_B = (sign_history * sum_W_diag).sum()
    B_val = num_B / den_B if abs(den_B) > 1e-30 else float('nan')
    print(f"  (B) ⟨sgn · ΣW·n_0⟩ / ⟨sgn · ΣW⟩                   = {B_val:+.6f}")
    print(f"      (unrestricted; matches PBC trace via OBC ensemble if valid)")

    # (C) No sign reweighting
    num_C = sum_W_diag_n0.sum()
    den_C = sum_W_diag.sum()
    C_val = num_C / den_C if abs(den_C) > 1e-30 else float('nan')
    print(f"  (C) ⟨ΣW·n_0⟩ / ⟨ΣW⟩                              = {C_val:+.6f}")
    print(f"      (sanity check; no sign reweight)")

    print(f"\nED unprojected reference:                                {ED_n0:+.6f}")
    print(f"\nGap (A) - ED = {A_val - ED_n0:+.6f}")
    print(f"Gap (B) - ED = {B_val - ED_n0:+.6f}")
    print(f"Gap (C) - ED = {C_val - ED_n0:+.6f}")


if __name__ == "__main__":
    main()
