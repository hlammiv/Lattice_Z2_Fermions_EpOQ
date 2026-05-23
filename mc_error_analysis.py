"""
mc_error_analysis.py — autocorrelation-aware error bars for MC observables.

Tools:
  - blocked_jackknife: jackknife error bar with block size b, where each
    block contains b consecutive MC samples.  As b grows, σ_jack stabilizes
    when b > 2·τ_int (integrated autocorrelation time).
  - autocorr_time: estimate integrated autocorrelation time τ_int from the
    plateau of σ_jack(b).
  - ratio_jackknife: jackknife for ratio estimators ⟨A⟩/⟨B⟩ (which is what
    sign-reweighted observables look like).

For sign-reweighted ratio:  ⟨O⟩ = ⟨sign · num⟩ / ⟨sign · den⟩
  Pass arrays: a = sign * num,  b = sign * den.
  Jackknife handles the ratio bias automatically.

Recommended usage in scripts:
  numer_per_config = [...]   # length N
  denom_per_config = [...]
  mean, sigma, tau_int = blocked_ratio_jackknife(numer_per_config,
                                                  denom_per_config)
  print(f"⟨O⟩ = {mean:+.5f} ± {sigma:.5f}  (τ_int = {tau_int:.1f})")
"""
from __future__ import annotations
import numpy as np


def blocked_ratio_jackknife(numer: np.ndarray, denom: np.ndarray,
                            block_sizes: list[int] = None
                            ) -> tuple[float, float, float, dict]:
    """Blocked jackknife for ratio estimator R = ⟨numer⟩ / ⟨denom⟩.

    For each block size b, splits the N samples into M = N // b consecutive
    blocks.  Jackknife estimator: for each block i, compute R_i = mean(numer
    excluding block i) / mean(denom excluding block i).  σ_jack(b) =
    √[(M-1)/M · Σ_i (R_i - R̄)²].

    σ_jack(b) is a function of b: small b underestimates (assumes
    independence), large b is noisy.  The plateau gives the true σ; the
    ratio τ_int(b) = σ_jack(b)² / σ_jack(1)² ÷ 2 estimates the integrated
    autocorrelation time.

    Args:
      numer: array (length N) of per-sample numerator contributions.
      denom: array (length N) of per-sample denominator contributions.
      block_sizes: list of block sizes to try.  Default: doubling from 1
                   to N/8.

    Returns:
      (mean, sigma, tau_int_estimate, diagnostics_dict)
    """
    numer = np.asarray(numer, dtype=np.float64)
    denom = np.asarray(denom, dtype=np.float64)
    N = len(numer)
    if N != len(denom):
        raise ValueError("numer and denom must have same length")
    if N < 4:
        # Can't do jackknife with too few samples
        d = float(np.sum(denom))
        if abs(d) < 1e-14:
            return float('nan'), float('nan'), float('nan'), {'N': N}
        return float(np.sum(numer) / d), float('nan'), float('nan'), {'N': N}

    if block_sizes is None:
        block_sizes = []
        b = 1
        while b <= max(1, N // 8):
            block_sizes.append(b)
            b *= 2

    # Cumulative sums for fast block-removal jackknife
    cum_n = np.concatenate([[0.0], np.cumsum(numer)])
    cum_d = np.concatenate([[0.0], np.cumsum(denom)])
    total_n = cum_n[-1]
    total_d = cum_d[-1]

    if abs(total_d) < 1e-14:
        return float('nan'), float('nan'), float('nan'), {'N': N, 'msg': 'zero denom'}

    mean_full = float(total_n / total_d)

    sigma_by_b = {}
    for b in block_sizes:
        M = N // b
        if M < 2:
            continue
        # Truncate to multiple-of-b
        N_eff = M * b
        # Compute jackknife R_i for each block
        R_i = np.empty(M)
        for i in range(M):
            start = i * b
            end = start + b
            sum_n_excl = total_n - (cum_n[end] - cum_n[start])
            sum_d_excl = total_d - (cum_d[end] - cum_d[start])
            if abs(sum_d_excl) < 1e-14:
                R_i[i] = float('nan')
            else:
                R_i[i] = sum_n_excl / sum_d_excl
        # σ_jack(b) = √[(M-1)/M · Σ_i (R_i - R̄)²]
        if np.any(~np.isfinite(R_i)):
            sigma_by_b[b] = float('nan')
        else:
            R_bar = np.mean(R_i)
            sigma_by_b[b] = float(np.sqrt((M - 1) / M * np.sum((R_i - R_bar) ** 2)))

    # Pick the σ from the largest block size with valid σ
    valid_bs = [b for b in sigma_by_b if np.isfinite(sigma_by_b[b])]
    if not valid_bs:
        return mean_full, float('nan'), float('nan'), {'N': N, 'sigma_by_b': sigma_by_b}

    # σ "plateau": take median of last 3 valid block sizes (robust to noise)
    last_few = sorted(valid_bs)[-3:]
    sigma_plateau = float(np.median([sigma_by_b[b] for b in last_few]))

    # τ_int ≈ (1/2) · (σ_plateau² / σ_b=1²) - 1/2
    sigma_1 = sigma_by_b.get(1, sigma_by_b[valid_bs[0]])
    if sigma_1 > 0:
        tau_int = max(0.5, 0.5 * (sigma_plateau ** 2 / sigma_1 ** 2 - 1) + 0.5)
    else:
        tau_int = float('nan')

    return mean_full, sigma_plateau, tau_int, {
        'N': N,
        'sigma_by_b': sigma_by_b,
        'sigma_naive_b1': sigma_1,
        'sigma_plateau_b': last_few,
    }


def format_estimate(mean: float, sigma: float, tau_int: float,
                    label: str = '⟨O⟩') -> str:
    if not np.isfinite(sigma):
        return f"{label} = {mean:+.5f}  (σ N/A, τ_int N/A)"
    return f"{label} = {mean:+.5f} ± {sigma:.5f}  (τ_int ≈ {tau_int:.1f})"


def _self_test():
    """Smoke test with toy correlated data."""
    rng = np.random.default_rng(2026)
    N = 10000
    # Mock: O = sin(x)/cos(x) ~ tan(x), with autocorrelated x
    x = np.zeros(N)
    sigma_x = 0.1
    for i in range(1, N):
        x[i] = 0.99 * x[i - 1] + sigma_x * rng.standard_normal()
    # Estimator: ⟨sin(x)⟩ / ⟨cos(x)⟩
    numer = np.sin(x)
    denom = np.cos(x)
    mean, sig, tau, info = blocked_ratio_jackknife(numer, denom)
    print(f"Self-test (AR(1) ρ=0.99, σ=0.1):")
    print(f"  N={N}, mean={mean:+.5f}")
    print(f"  σ_naive(b=1) = {info['sigma_naive_b1']:.5f}")
    print(f"  σ_plateau    = {sig:.5f}")
    print(f"  τ_int ≈ {tau:.1f}")
    print(f"  σ_plateau / σ_naive = {sig/info['sigma_naive_b1']:.1f}× "
          f"(expected ~√(2τ+1) = {np.sqrt(2*tau+1):.1f}×)")
    print(f"  σ by block:")
    for b, s in sorted(info['sigma_by_b'].items()):
        print(f"    b={b:5d}: σ={s:.6f}")


if __name__ == "__main__":
    _self_test()
