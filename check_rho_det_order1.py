"""Quick variant: rerun deterministic test with W ORDER=1 (Lie-Trotter,
time-ordered product) instead of palindrome.  If gap at a_τ=0.25 shrinks
back below a_τ=0.5, palindrome is the artifact and the real time-ordered
W converges normally.

Re-uses build_rho_deterministic from check_rho_deterministic but passes
order=1 through.  Since order is fixed inside per_config_contribution,
we monkey-patch W_ORDER for this script.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import sys, time
import numpy as np
from scipy.linalg import expm

sys.path.insert(0, '/home/hlamm/Desktop/QC/logdet/m1_toy')

import check_rho_deterministic as cd
cd.W_ORDER = 1   # Lie-Trotter (time-ordered, non-Hermitian per-config)

import epoq_classical_sampler as cs


def main():
    print("=" * 80, flush=True)
    print("DETERMINISTIC PATH-INTEGRAL TEST — ORDER=1 (Lie-Trotter)", flush=True)
    print("=" * 80, flush=True)

    H = cd.build_H_QC_dense()
    rho_ED = expm(-cd.BETA * H)
    print(f"Tr e^{{-βH}} = {np.trace(rho_ED).real:.6f}\n", flush=True)

    print(f"[a_τ=0.5, N_E=2]", flush=True)
    rho_a050 = cd.build_rho_deterministic(N_E=2, a_tau=0.5, n_workers=6)
    cd.report(rho_a050, rho_ED, H, label="order=1, a_τ=0.5 (N_E=2)")

    print(f"\n[a_τ=0.25, N_E=3]", flush=True)
    rho_a025 = cd.build_rho_deterministic(N_E=3, a_tau=0.25, n_workers=6)
    cd.report(rho_a025, rho_ED, H, label="order=1, a_τ=0.25 (N_E=3)")

    print("\n" + "=" * 80)
    rl1 = rho_a050 / np.trace(rho_a050).real
    rl2 = rho_a025 / np.trace(rho_a025).real
    re  = rho_ED / np.trace(rho_ED).real
    d1 = np.linalg.norm(rl1 - re)
    d2 = np.linalg.norm(rl2 - re)
    print(f"  order=1 ‖Δρ‖(a_τ=0.50) = {d1:.5f}")
    print(f"  order=1 ‖Δρ‖(a_τ=0.25) = {d2:.5f}")
    print(f"  ratio: {d2/d1:.3f}  (expect ~0.25 if O(a_τ²))")


if __name__ == "__main__":
    main()
