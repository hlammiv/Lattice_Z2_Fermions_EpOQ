"""Pure-gauge cross-check (Phase 34): g_hop=0, m=0 → [H_g, H_F]=0 trivially.

If MC ⟨σ_plaq⟩_β matches ED at all a_τ values, the gauge Suzuki mapping is
correct and the β=4 systematic error in the fermion case is the
gauge-fermion commutator.  If MC ⟨σ_plaq⟩ doesn't match ED, there's a
separate gauge-sector issue we need to address first.

Hamiltonian: H_pure_gauge = −g_E · Σ_l X_l − g_M · Z_0 Z_1 Z_2 Z_3 (acts on 4-link Hilbert space, dim 16).
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import sys, time
import numpy as np
from scipy.linalg import expm

sys.path.insert(0, '/home/hlamm/Desktop/QC/logdet/m1_toy')

from action_z2_staggered import LatticeGeometry
from action_z2_metropolis_pt import run_metropolis_pt

G_E = 1.0
G_M = 0.5
BETA = 4.0
A_TAUS = [0.5, 0.333333333, 0.25, 0.2]
N_CHAINS = 4
N_SWEEPS = 50_000
N_WARMUP = 5_000


def build_K_E_ladder(K_E_target, n_replicas=6):
    K_E_low = max(0.05, K_E_target - 0.6)
    return list(np.linspace(K_E_low, K_E_target, n_replicas))


def ED_pure_gauge(beta):
    """4-link pure gauge Hamiltonian: −g_E Σ X_l − g_M Z_0Z_1Z_2Z_3.
    Returns ⟨σ_plaq⟩_β = ⟨Z_0Z_1Z_2Z_3⟩_β.
    """
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.diag([1, -1]).astype(complex)
    def kron4(a0, a1, a2, a3):
        return np.kron(np.kron(np.kron(a3, a2), a1), a0)  # q0 = LSB
    H = np.zeros((16, 16), dtype=complex)
    for q in range(4):
        ops = [I, I, I, I]; ops[q] = X
        H -= G_E * kron4(*ops)
    H -= G_M * kron4(Z, Z, Z, Z)
    H = (H + H.conj().T) / 2

    rho = expm(-beta * H)
    Z_part = np.trace(rho).real
    plaq_op = kron4(Z, Z, Z, Z)
    avg_plaq = (np.trace(rho @ plaq_op) / Z_part).real
    return avg_plaq, Z_part


def avg_plaquette_from_configs(configs, geom):
    """Average σ_plaq over all spatial slices and configs.  For 2x2 OBC, 1 plaquette per slice."""
    total = 0.0
    count = 0
    for U in configs:
        for t in range(geom.N_E):
            # The spatial plaquette: U_x[t,0,0]·U_y[t,1,0]·U_x[t,0,1]·U_y[t,0,0]
            u1 = U.U_x[t, 0, 0]
            u2 = U.U_y[t, 1, 0]
            u3 = U.U_x[t, 0, 1]
            u4 = U.U_y[t, 0, 0]
            total += float(u1 * u2 * u3 * u4)
            count += 1
    return total / count


def main():
    print("=" * 80, flush=True)
    print(f"Phase 34: Pure-gauge cross-check, β={BETA}")
    print(f"  g_E={G_E}, g_M={G_M}, m=g_hop=0")
    print("=" * 80, flush=True)

    plaq_ED, Z_ED = ED_pure_gauge(BETA)
    print(f"\nED reference @ β={BETA}: ⟨σ_plaq⟩ = {plaq_ED:+.6f}, Z = {Z_ED:.4e}")

    print(f"\n{'a_τ':>6} | {'N_E':>4} | {'K_E':>8} | {'K_M':>8} | "
          f"{'⟨plaq⟩_MC':>14} | {'σ_chains':>10} | {'gap':>10}")
    print("-" * 90)
    for a_tau in A_TAUS:
        N_E = int(round(BETA / a_tau)) + 1
        K_E = -0.5 * np.log(np.tanh(a_tau * G_E))
        K_M = a_tau * G_M
        geom = LatticeGeometry(Lx=2, Ly=2, N_E=N_E, m=0.0)  # m=0
        K_E_ladder = build_K_E_ladder(K_E)

        plaq_chains = []
        for ci in range(N_CHAINS):
            seed = 2026 + ci * 11111
            pt_results, _ = run_metropolis_pt(
                geom, K_E_ladder=K_E_ladder, K_M=K_M,
                n_sweeps=N_SWEEPS, n_warmup=N_WARMUP, swap_every=5,
                seed=seed, action_type='gauge_only', n_workers=6)
            top = pt_results[-1]
            plaq_chains.append(avg_plaquette_from_configs(top.configs, geom))
        plaq_mean = np.mean(plaq_chains)
        plaq_sem = np.std(plaq_chains, ddof=1) / np.sqrt(N_CHAINS) if N_CHAINS > 1 else 0
        gap = plaq_mean - plaq_ED
        print(f"  {a_tau:>4.3f} | {N_E:>4d} | {K_E:>8.4f} | {K_M:>8.4f} | "
              f"{plaq_mean:>+12.6f} | {plaq_sem:>10.6f} | {gap:>+10.6f}", flush=True)


if __name__ == "__main__":
    main()
