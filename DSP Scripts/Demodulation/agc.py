"""Automatic Gain Control for 8VSB demodulation."""

import numpy as np


class AGC:
    """Simple feedback AGC that normalizes signal power to a target level.

    Parameters
    ----------
    alpha : float
        Adaptation rate. Smaller = slower, more stable. Default 1e-4.
    target_power : float
        Target output power level (mean |x|^2). Default 1.0.
    initial_gain : float
        Initial gain value. Default 1.0.
    """

    def __init__(
        self,
        alpha: float = 1e-4,
        target_power: float = 1.0,
        initial_gain: float = 1.0,
    ):
        self.alpha = alpha
        self.target_power = target_power
        self.gain = initial_gain

    def process(self, samples: np.ndarray) -> np.ndarray:
        """Apply AGC to a block of samples.

        Parameters
        ----------
        samples : np.ndarray
            Input samples (complex or real).

        Returns
        -------
        np.ndarray
            Gain-adjusted samples.
        """
        out = np.empty_like(samples)
        gain = self.gain

        for i in range(len(samples)):
            out[i] = samples[i] * gain
            power = np.real(out[i] * np.conj(out[i]))
            error = self.target_power - power
            gain += self.alpha * error

            # Prevent gain from going negative or exploding
            gain = max(gain, 1e-6)
            gain = min(gain, 1e6)

        self.gain = gain
        return out

    def process_block(self, samples: np.ndarray) -> np.ndarray:
        """Faster block-based AGC using per-block gain update.

        Less accurate than sample-by-sample but much faster for prototyping.
        """
        block_power = np.mean(np.abs(samples) ** 2)
        if block_power > 0:
            self.gain = np.sqrt(self.target_power / block_power)
        return samples * self.gain

    def reset(self, initial_gain: float = 1.0):
        self.gain = initial_gain
