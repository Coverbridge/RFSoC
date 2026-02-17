"""Root Raised Cosine matched filter for 8VSB pulse shaping.

The ATSC transmitter uses an RRC filter with rolloff alpha=0.1152.
The receiver matched filter is the same RRC, and the cascade produces
a raised cosine pulse with zero ISI at the symbol sampling instants.
"""

import numpy as np
from scipy.signal import lfilter

from osmium.utils.constants import ATSC_RRC_ALPHA, ATSC_SYMBOL_RATE


def design_rrc_filter(
    num_taps: int = 65,
    alpha: float = ATSC_RRC_ALPHA,
    samples_per_symbol: int = 2,
) -> np.ndarray:
    """Design a Root Raised Cosine FIR filter.

    Parameters
    ----------
    num_taps : int
        Number of filter taps (should be odd). Default 65.
    alpha : float
        Rolloff factor. Default 0.1152 per ATSC A/53.
    samples_per_symbol : int
        Oversampling ratio (samples per symbol). Default 2.

    Returns
    -------
    np.ndarray
        FIR filter coefficients, normalized to unit energy.
    """
    if num_taps % 2 == 0:
        num_taps += 1

    T = 1.0  # symbol period (normalized)
    Ts = T / samples_per_symbol  # sample period

    t = np.arange(-(num_taps - 1) / 2, (num_taps - 1) / 2 + 1) * Ts
    h = np.zeros(num_taps, dtype=np.float64)

    for i in range(num_taps):
        ti = t[i]

        if abs(ti) < 1e-10:
            # t = 0
            h[i] = (1.0 / T) * (1.0 + alpha * (4.0 / np.pi - 1.0))
        elif abs(abs(ti) - T / (4.0 * alpha)) < 1e-10:
            # t = ±T/(4*alpha): special case to avoid division by zero
            h[i] = (alpha / (T * np.sqrt(2.0))) * (
                (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * alpha))
                + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * alpha))
            )
        else:
            # General case
            num = np.sin(np.pi * ti / T * (1 - alpha)) + \
                  4.0 * alpha * ti / T * np.cos(np.pi * ti / T * (1 + alpha))
            den = np.pi * ti / T * (1 - (4.0 * alpha * ti / T) ** 2)
            h[i] = (1.0 / T) * num / den

    # Normalize to unit energy
    h /= np.sqrt(np.sum(h ** 2))

    return h


class MatchedFilter:
    """RRC matched filter for ATSC 8VSB.

    Parameters
    ----------
    num_taps : int
        Number of filter taps. Default 65.
    alpha : float
        RRC rolloff. Default 0.1152.
    samples_per_symbol : int
        Input oversampling ratio. Default 2 (at ~21.524 MSPS).
    """

    def __init__(
        self,
        num_taps: int = 65,
        alpha: float = ATSC_RRC_ALPHA,
        samples_per_symbol: int = 2,
    ):
        self.coeffs = design_rrc_filter(num_taps, alpha, samples_per_symbol)
        self.num_taps = len(self.coeffs)
        self.samples_per_symbol = samples_per_symbol

        # Filter state for continuous processing
        self._zi = np.zeros(self.num_taps - 1, dtype=np.float64)

    def process(self, samples: np.ndarray) -> np.ndarray:
        """Apply the RRC matched filter.

        Parameters
        ----------
        samples : np.ndarray
            Real-valued input samples (from FPLL output) at 2 sps.

        Returns
        -------
        np.ndarray
            Filtered samples at the same rate. Zero ISI at optimal
            sampling instants (every 2 samples if timing is perfect).
        """
        out, self._zi = lfilter(self.coeffs, 1.0, samples, zi=self._zi)
        return out

    def reset(self):
        self._zi = np.zeros(self.num_taps - 1, dtype=np.float64)
