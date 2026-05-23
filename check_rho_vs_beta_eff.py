"""Compare the EMPIRICAL lattice density matrix ρ̃_lat (built from MC at finite a_τ)
directly to ED ρ_QC(β_eff) = exp(−β_eff · H_QC) at various β_eff. Find:
  - The best-fit β_eff that minimizes ||ρ̃_lat/Tr − ρ_QC/Tr||_F (Frobenius norm).
  - Whether the lattice ρ̃ at our finite a_τ "looks like" ED at some shifted β,
    or whether the structure is incompatible with ANY β shift.

For each MC gauge sample U at a_τ=0.5, m=2:
  - Compute W_U[ψ_top, ψ_bot] (16×16 fermion block at fixed gauge, palindrome).
  - Lift to a 256×256 contribution: ρ̃_lat[bits_top, bits_bot] += w(U) · W_U
    where bits_top = (gauge_top << 4) | ψ_top  etc.
After MC, ρ̃_lat is the empirical density matrix.
"""
from __future__ import annotations
import sys, time
import numpy as np
from scipy.linalg import expm

sys.path.insert(0, '/home/hlamm/Desktop/QC/logdet/m1_toy')
import epoq_quantum_simulator; epoq_quantum_simulator.M_MASS = 2.0
import epoq_classical_sampler; epoq_classical_sampler.M_MASS = 2.0

from action_z2_staggered import LatticeGeometry
from action_z2_metropolis_pt import run_metropolis_pt
from transfer_matrix_kbc_trotterized import compute_combined_weight_trotter
from action_minkowski_stitch import gauge_qc_bits_from_slice
from action_corner_direct_v3 import _fock_index_to_psi_map

M_HAM, G_E_HAM, G_M_HAM, BETA = 2.0, 1.0, 0.5, 4.0
A_TAU = 0.5
N_WARMUP, N_SWEEPS, SWAP_EVERY = 5000, 4000, 5
N_INSTANCES = 4
W_ORDER = 2


def build_K_E_ladder(K_E_target, n_replicas=8):
    K_E_low = max(0.10, K_E_target - 0.7)
    if K_E_low >= K_E_target - 0.05:
        K_E_low = K_E_target - 0.05
    return list(np.linspace(K_E_low, K_E_target, n_replicas))


def accumulate_rho_per_instance(seed):
    """Run one PT-forward instance; build empirical 256×256 ρ̃_lat."""
    K_E_target = -0.5 * np.log(np.tanh(A_TAU * G_E_HAM))
    K_M = A_TAU * G_M_HAM
    N_E = int(round(BETA / A_TAU)) + 1
    m_action = A_TAU * M_HAM
    geom = LatticeGeometry(Lx=2, Ly=2, N_E=N_E, m=m_action)
    K_E_ladder = build_K_E_ladder(K_E_target)
    t0 = time.time()
    pt_results, _ = run_metropolis_pt(
        geom, K_E_ladder=K_E_ladder, K_M=K_M,
        n_sweeps=N_SWEEPS, n_warmup=N_WARMUP, swap_every=SWAP_EVERY,
        seed=seed, action_type='forward')
    top = pt_results[-1]
    configs = top.configs
    signs = np.array(top.sign_history, dtype=int)

    # Accumulate 256×256 ρ̃
    rho = np.zeros((256, 256), dtype=complex)
    V3 = geom.V_3
    idx_to_psi = _fock_index_to_psi_map(V3)

    for U_idx, U in enumerate(configs):
        # Sign-reweight: in MC, samples are weighted by |det M|, observable
        # uses sign(det M).  Net effective weight per sample is sign(det M).
        w = float(signs[U_idx])

        # Compute palindrome W on fermion sector at this gauge config.
        W_full, _ = compute_combined_weight_trotter(
            geom, U, a_tau=A_TAU, K_E=0.0, K_M=0.0,
            m_obs=M_HAM, g_hop=0.5, order=W_ORDER)

        # Map lex-index -> psi
        # W_full is indexed by lex indices [0..15] × [0..15]
        # Translate to psi indices using idx_to_psi
        # Construct 16×16 W matrix in psi basis
        W_psi = np.zeros((16, 16), dtype=complex)
        for lex_i in range(16):
            for lex_j in range(16):
                psi_i = idx_to_psi[lex_i]
                psi_j = idx_to_psi[lex_j]
                W_psi[psi_i, psi_j] = W_full[lex_i, lex_j]

        # Get boundary gauge bits
        g_top = gauge_qc_bits_from_slice(U, geom.N_E - 1)
        g_bot = gauge_qc_bits_from_slice(U, 0)

        # Block contribution: ρ̃[g_top·16+ψ_top, g_bot·16+ψ_bot] += w · W_psi[ψ_top, ψ_bot]
        # 8q convention: bits 0-3 = gauge, bits 4-7 = fermion
        # so 8q index = (psi << 4) | gauge_bits
        for psi_top in range(16):
            for psi_bot in range(16):
                bit_a = (psi_top << 4) | g_top
                bit_b = (psi_bot << 4) | g_bot
                rho[bit_a, bit_b] += w * W_psi[psi_top, psi_bot]
    return rho, time.time() - t0


def main():
    print("=" * 80, flush=True)
    print(f"ρ̃_lat vs ρ_QC(β_eff) comparison @ a_τ={A_TAU}, m={M_HAM}, β_bare={BETA}")
    print(f"  Palindrome W (order=2), {N_INSTANCES} indep PT-forward, "
          f"warm={N_WARMUP} swp={N_SWEEPS}")
    print("=" * 80, flush=True)

    # Build empirical ρ̃_lat
    rho_lat = np.zeros((256, 256), dtype=complex)
    for i in range(N_INSTANCES):
        seed = 2026 + i * 11111
        print(f"\n--- instance {i+1}/{N_INSTANCES} (seed={seed}) ---", flush=True)
        rho_i, twall = accumulate_rho_per_instance(seed)
        rho_lat += rho_i
        print(f"    walltime={twall:.0f}s, Tr(ρ_i)={np.trace(rho_i).real:.3e}", flush=True)
    rho_lat /= N_INSTANCES   # average

    # Diagnostics BEFORE Hermitization
    print(f"\nBEFORE Hermitization:", flush=True)
    print(f"  Tr(ρ̃_lat) = {np.trace(rho_lat).real:.3e}, "
          f"max |Im| = {np.abs(rho_lat.imag).max():.3e}", flush=True)
    print(f"  ||ρ̃ - ρ̃†||_F = {np.linalg.norm(rho_lat - rho_lat.conj().T):.3e}",
          flush=True)

    # Hermitize per EρOQ eq (2.3): ρ̃ → (ρ̃ + ρ̃†)/2
    rho_lat = (rho_lat + rho_lat.conj().T) / 2

    print(f"\nAFTER Hermitization:", flush=True)
    print(f"  Tr(ρ̃_lat) = {np.trace(rho_lat).real:.3e}, "
          f"max |Im| = {np.abs(rho_lat.imag).max():.3e}", flush=True)
    print(f"  ||ρ̃ - ρ̃†||_F = {np.linalg.norm(rho_lat - rho_lat.conj().T):.3e}",
          flush=True)

    # Build H_QC for ED
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

    # Normalize ρ̃_lat
    norm_lat = np.trace(rho_lat).real
    rho_lat_normed = rho_lat / norm_lat

    # Scan β_eff and compare
    print()
    print("=" * 70)
    print("Comparing ρ̃_lat_normalized to ρ_QC(β_eff)_normalized:")
    print("=" * 70)
    print(f"{'β_eff':>8} | {'||Δρ||_F':>10} | {'||Δdiag||':>10} | {'||Δoffdiag||':>12} | "
          f"{'Tr(ρ_QC)':>12}")
    print("-" * 70)
    betas = np.linspace(0.5, 8.0, 31)
    best_beta = None; best_dist = float('inf')
    distances = []
    for beta_eff in betas:
        rho_ED = expm(-beta_eff * H)
        rho_ED_normed = rho_ED / np.trace(rho_ED).real
        diff = rho_lat_normed - rho_ED_normed
        d_total = np.linalg.norm(diff)
        d_diag = np.linalg.norm(np.diag(diff))
        d_offdiag = np.linalg.norm(diff - np.diag(np.diag(diff)))
        distances.append(d_total)
        if d_total < best_dist:
            best_dist = d_total; best_beta = beta_eff
        if abs(beta_eff - round(beta_eff*2)/2) < 1e-9:  # print every 0.5
            print(f"{beta_eff:>8.3f} | {d_total:>10.4f} | {d_diag:>10.4f} | "
                  f"{d_offdiag:>12.4f} | {np.trace(rho_ED).real:>12.3e}")

    print(f"\nBEST FIT: β_eff = {best_beta:.3f}  (bare β = {BETA})  "
          f"||Δρ||_F = {best_dist:.4f}", flush=True)
    print(f"vs β=4 (bare): ||Δρ||_F = "
          f"{distances[np.argmin(np.abs(betas - BETA))]:.4f}")

    # Print a few key matrix elements: how do they compare at best β_eff?
    print("\n  Some structural diagnostics:")
    print(f"  Tr(ρ̃_lat)        = {np.trace(rho_lat).real:.3e}")
    print(f"  Tr(ρ̃_lat normed) = {np.trace(rho_lat_normed).real:.4f} (should be ~1)")
    eigs_lat = np.real(np.linalg.eigvals(rho_lat_normed))
    eigs_lat.sort()
    print(f"  ρ̃_lat eigvals (real): min={eigs_lat.min():+.4e}, max={eigs_lat.max():+.4e}, "
          f"# negative = {(eigs_lat < -1e-6).sum()}/256")

    rho_ED_best = expm(-best_beta * H) / np.trace(expm(-best_beta * H)).real
    eigs_ED = np.real(np.linalg.eigvals(rho_ED_best))
    eigs_ED.sort()
    print(f"  ρ_QC(β_eff) eigvals (real): min={eigs_ED.min():+.4e}, max={eigs_ED.max():+.4e}")


if __name__ == "__main__":
    main()
