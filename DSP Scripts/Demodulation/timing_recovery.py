"""Symbol timing recovery for 8VSB demodulation.

Uses a Gardner timing error detector (TED) with a Farrow-structure
cubic interpolator and a proportional-integral loop filter to achieve
optimal symbol sampling at exactly 1 sample/symbol.
"""

import numpy as np

from osmium.utils.constants import ATSC_SYMBOL_RATE


class TimingRecovery:
    """Gardner TED with interpolation for symbol synchronization.

    Input: real samples at ~2 samples/symbol (from matched filter).
    Output: real samples at exactly 1 sample/symbol (10.762 Msym/s).

    Parameters
    ----------
    samples_per_symbol : float
        Nominal input samples per symbol. Default 2.0.
    loop_bw : float
        Normalized loop bandwidth (0 to 1). Default 0.01.
    damping : float
        Loop damping factor. Default 0.707 (critically damped).
    """

    def __init__(
        self,
        samples_per_symbol: float = 2.0,
        loop_bw: float = 0.01,
        damping: float = 0.707,
    ):
        self.sps = samples_per_symbol
        self.loop_bw = loop_bw
        self.damping = damping

        # Compute PI loop filter gains from loop_bw and damping
        # Using Gardner's formulation
        denom = 1.0 + 2.0 * damping * loop_bw + loop_bw ** 2
        self.alpha = 4.0 * damping * loop_bw / denom        # proportional
        self.beta = 4.0 * loop_bw ** 2 / denom               # integral

        # Internal state
        self._mu = 0.0          # fractional delay (0 to 1)
        self._omega = samples_per_symbol  # samples per symbol estimate
        self._omega_mid = samples_per_symbol
        self._omega_lim = 0.01 * samples_per_symbol  # max omega deviation

        # Previous samples for interpolation
        self._prev_samples = np.zeros(4, dtype=np.float64)

    def process(self, samples: np.ndarray) -> np.ndarray:
        """Perform timing recovery on a block of filtered samples.

        Parameters
        ----------
        samples : np.ndarray
            Real-valued matched-filtered samples at ~2 sps.

        Returns
        -------
        np.ndarray
            Optimally sampled symbols at 1 sample/symbol.
        """
        out = []
        n = len(samples)
        idx = 0

        # Prepend previous tail samples for continuity
        extended = np.concatenate([self._prev_samples, samples])
        idx = len(self._prev_samples)
        n_ext = len(extended)

        mu = self._mu
        omega = self._omega

        while idx + 2 < n_ext:
            # Integer sample index and fractional delay
            k = int(idx)
            mu_frac = mu

            if k + 1 >= n_ext:
                break

            # Cubic (Farrow) interpolation using 4 neighboring samples
            k0 = max(0, k - 1)
            k1 = k
            k2 = min(k + 1, n_ext - 1)
            k3 = min(k + 2, n_ext - 1)

            d0 = extended[k0]
            d1 = extended[k1]
            d2 = extended[k2]
            d3 = extended[k3]

            # Cubic interpolation coefficients
            s_interp = _cubic_interpolate(d0, d1, d2, d3, mu_frac)
            out.append(s_interp)

            # Gardner TED: needs the current symbol, the mid-point, and previous symbol
            if len(out) >= 2:
                # Current symbol sample
                sym_curr = out[-1]
                sym_prev = out[-2]

                # Mid-point sample (between previous and current symbol)
                mid_idx = idx - omega / 2.0
                mid_k = int(mid_idx)
                mid_mu = mid_idx - mid_k

                mid_k0 = max(0, mid_k - 1)
                mid_k1 = mid_k
                mid_k2 = min(mid_k + 1, n_ext - 1)
                mid_k3 = min(mid_k + 2, n_ext - 1)

                mid_sample = _cubic_interpolate(
                    extended[mid_k0], extended[mid_k1],
                    extended[mid_k2], extended[mid_k3],
                    mid_mu,
                )

                # Gardner TED: e = mid * (prev - curr)
                error = mid_sample * (sym_prev - sym_curr)

                # Loop filter (PI controller)
                omega = omega + self.beta * error
                mu_next = mu + omega + self.alpha * error

                # Clamp omega to prevent divergence
                omega = np.clip(
                    omega,
                    self._omega_mid - self._omega_lim,
                    self._omega_mid + self._omega_lim,
                )
            else:
                mu_next = mu + omega

            # Advance the index
            idx_advance = int(mu_next)
            mu = mu_next - idx_advance
            idx += idx_advance

        # Save tail for next block
        tail_len = min(4, len(samples))
        self._prev_samples = samples[-tail_len:].copy()
        self._mu = mu
        self._omega = omega

        return np.array(out, dtype=np.float64)

    def reset(self):
        self._mu = 0.0
        self._omega = self.sps
        self._prev_samples = np.zeros(4, dtype=np.float64)


def _cubic_interpolate(
    y0: float, y1: float, y2: float, y3: float, mu: float
) -> float:
    """Cubic (Hermite) interpolation between y1 and y2.

    mu = 0 returns y1, mu = 1 returns y2.
    """
    a0 = -0.5 * y0 + 1.5 * y1 - 1.5 * y2 + 0.5 * y3
    a1 = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3
    a2 = -0.5 * y0 + 0.5 * y2
    a3 = y1
    return ((a0 * mu + a1) * mu + a2) * mu + a3
