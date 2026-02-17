"""DC offset removal for 8VSB demodulation."""

import numpy as np


class DCBlocker:
    """Single-pole IIR DC blocker.

    Removes DC offset using a high-pass filter:
        y[n] = x[n] - x_avg
        x_avg = (1 - alpha) * x_avg + alpha * x[n]

    Parameters
    ----------
    alpha : float
        Tracking rate. Smaller = slower tracking, deeper null at DC.
        Default 1e-4.
    """

    def __init__(self, alpha: float = 1e-4):
        self.alpha = alpha
        self._avg = 0.0 + 0.0j

    def process(self, samples: np.ndarray) -> np.ndarray:
        """Remove DC offset from a block of samples.

        Parameters
        ----------
        samples : np.ndarray
            Input samples (complex or real).

        Returns
        -------
        np.ndarray
            DC-blocked samples.
        """
        out = np.empty_like(samples)
        avg = self._avg

        for i in range(len(samples)):
            avg = (1.0 - self.alpha) * avg + self.alpha * samples[i]
            out[i] = samples[i] - avg

        self._avg = avg
        return out

    def process_block(self, samples: np.ndarray) -> np.ndarray:
        """Faster block-based DC removal (subtract block mean)."""
        block_mean = np.mean(samples)
        self._avg = block_mean
        return samples - block_mean

    def reset(self):
        self._avg = 0.0 + 0.0j
