"""
verify_p2_vs_cb70.py — Phase 3 of the P2 pipeline.

Verify that the P2 Slater-det Monte Carlo sampler (action_corner_direct_v2)
reproduces the exact Cauchy–Binet partition-function and Euclidean static
observable values within MC error bars.

The "exact reference" is the full 70-term enumeration (CB-70) for V_3=4.
The "P2 MC" is the categorical sampler from action_corner_direct_v2.

For each test gauge config U:
  1. Compute Σ_pairs W = det M exactly (enumeration).
  2. Compute exact static Euclidean ⟨n̂_0(τ=0)⟩ via Σ_pairs n_0(Ψ_top) · W
     / Σ_pairs W.
  3. Run P2 MC at N_samples ∈ {500, 2000, 10000}.
  4. Compute MC estimators with bootstrap error bars.
  5. Verify: |MC − exact| < (a few) · MC_err.

The P2 sampler should be UNBIASED — MC mean → exact as N → ∞.
"""
from __future__ import annotations
import sys
import numpy as np

sys.path.insert(0, '/home/hlamm/Desktop/QC/logdet/m1_toy')
from action_z2_staggered import LatticeGeometry, Z2GaugeConfig, build_dirac_matrix
from action_corner_direct_v2 import (
    fock_pair_weights_array, sample_corner_pairs_categorical, bitcount,
)


def n_0_observable(psi: int) -> int:
    """Number-operator observable at site 0 (matter occupation): bit 0 of psi."""
    return psi & 1


def exact_observables(geom: LatticeGeometry, U: Z2GaugeConfig) -> dict:
    """Compute exact static observables via full pair enumeration."""
    pairs, amps, abs_amps = fock_pair_weights_array(geom, U)
    det_M_via_W = complex(np.sum(amps))
    det_M_true = float(np.linalg.det(build_dirac_matrix(geom, U)))

    # ⟨n_0⟩ = Σ_pairs n_0(psi_i) · W[psi_i, psi_j] / Σ_pairs W
    numerator = complex(np.sum([n_0_observable(int(p[0])) * a
                                for p, a in zip(pairs, amps)]))
    n0_exact = float((numerator / det_M_via_W).real)

    return {
        'det_M_via_W': det_M_via_W,
        'det_M_true': det_M_true,
        'err_det_M': abs(det_M_via_W - det_M_true),
        'n_0_exact': n0_exact,
        'numerator': numerator,
    }


def mc_estimator(geom: LatticeGeometry, U: Z2GaugeConfig,
                 N_samples: int, rng: np.random.Generator,
                 n_bootstrap: int = 200) -> dict:
    """Run P2 MC, compute ratio estimator for ⟨n_0⟩ with bootstrap error."""
    pairs, signs, total = sample_corner_pairs_categorical(geom, U, N_samples, rng)
    # Per-sample observable values
    sign_arr = signs.astype(float)
    n0_arr = np.array([n_0_observable(int(p)) for p in pairs[:, 0]], dtype=float)

    # det M estimator: det M ≈ total · ⟨sign⟩
    det_M_est = total * sign_arr.mean()

    # ⟨n_0⟩ ratio estimator: ⟨sign · n_0⟩ / ⟨sign⟩
    if abs(sign_arr.mean()) < 1e-12:
        n0_est = float('nan')
    else:
        n0_est = (sign_arr * n0_arr).mean() / sign_arr.mean()

    # Bootstrap
    rng_b = np.random.default_rng(31337)
    det_M_boots = np.zeros(n_bootstrap)
    n0_boots = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng_b.integers(0, N_samples, N_samples)
        sm = sign_arr[idx].mean()
        det_M_boots[b] = total * sm
        if abs(sm) > 1e-12:
            n0_boots[b] = (sign_arr[idx] * n0_arr[idx]).mean() / sm
        else:
            n0_boots[b] = float('nan')

    return {
        'N_samples': N_samples,
        'total': total,
        'det_M_est': det_M_est,
        'det_M_err': float(np.std(det_M_boots)),
        'n_0_est': n0_est,
        'n_0_err': float(np.nanstd(n0_boots)),
        'sign_mean': float(sign_arr.mean()),
    }


def run_test_config(geom: LatticeGeometry, U: Z2GaugeConfig, label: str = ""):
    print(f"\n--- {label} ---")
    ex = exact_observables(geom, U)
    print(f"  Exact:  det M = {ex['det_M_true']:+.6f},  "
          f"⟨n_0⟩ = {ex['n_0_exact']:+.6f}")
    print(f"  (consistency: |Σ W − det M| = {ex['err_det_M']:.2e})")

    rng = np.random.default_rng(2026)
    for N in [500, 2000, 10000]:
        mc = mc_estimator(geom, U, N, rng)
        z_det = (mc['det_M_est'] - ex['det_M_true']) / max(mc['det_M_err'], 1e-30)
        z_n0 = (mc['n_0_est'] - ex['n_0_exact']) / max(mc['n_0_err'], 1e-30)
        print(f"  N={N:>5}:  det M = {mc['det_M_est']:+.6f} ± {mc['det_M_err']:.4f}  "
              f"(z={z_det:+.2f})  "
              f"⟨n_0⟩ = {mc['n_0_est']:+.6f} ± {mc['n_0_err']:.4f}  "
              f"(z={z_n0:+.2f})  ⟨sign⟩={mc['sign_mean']:+.3f}")


def main():
    print("=" * 72)
    print("PHASE 3: P2 MC vs CB-70 exact reference")
    print("=" * 72)

    geom = LatticeGeometry(Lx=2, Ly=2, N_E=4, m=0.5)

    # Test 1: trivial gauge config
    U_triv = Z2GaugeConfig.trivial(geom)
    run_test_config(geom, U_triv, "trivial U (free V_3=4)")

    # Test 2-4: random Z₂ configs
    rng = np.random.default_rng(2026)
    for trial in range(3):
        U = Z2GaugeConfig.random(geom, rng)
        run_test_config(geom, U, f"random Z₂ config (seed=2026, trial={trial})")

    print()
    print("=" * 72)
    print("Phase 3 result: P2 MC should give |z| ≲ 2-3 for det M and ⟨n_0⟩")
    print("at N=10000.  Bias = O(1/√N), no systematic.")
    print("=" * 72)


if __name__ == "__main__":
    main()
