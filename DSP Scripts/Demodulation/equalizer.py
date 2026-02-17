"""Adaptive LMS equalizer for ATSC 8VSB channel correction.

Compensates for multipath distortion, group delay, and frequency response
impairments using an LMS (Least Mean Squares) adaptive FIR filter.
Supports training mode (using field sync PN sequences) and decision-directed
mode for steady-state operation.
"""

import numpy as np

from osmium.utils.constants import (
    VSB8_LEVELS,
    ATSC_SEGMENT_LENGTH,
    ATSC_SEGMENT_SYNC_LENGTH,
    ATSC_SEGMENT_DATA_LENGTH,
)


def _hard_decision(symbol: float) -> float:
    """Slice to nearest 8VSB level."""
    idx = np.argmin(np.abs(VSB8_LEVELS - symbol))
    return VSB8_LEVELS[idx]


class AdaptiveEqualizer:
    """LMS adaptive FIR equalizer for 8VSB.

    Parameters
    ----------
    num_taps : int
        Number of equalizer taps. Default 64.
    pre_taps : int
        Number of anti-causal (pre-cursor) taps. Default 16.
    step_size : float
        LMS step size (mu). Default 5e-5 (from gr-dtv).
    """

    def __init__(
        self,
        num_taps: int = 64,
        pre_taps: int = 16,
        step_size: float = 5e-5,
    ):
        self.num_taps = num_taps
        self.pre_taps = pre_taps
        self.post_taps = num_taps - pre_taps
        self.step_size = step_size

        # Filter weights: initialize as unit impulse at the pre_taps position
        self.weights = np.zeros(num_taps, dtype=np.float64)
        self.weights[pre_taps] = 1.0

        # Input buffer (delay line)
        self._buffer = np.zeros(num_taps, dtype=np.float64)
        self._buf_idx = 0

        # Training state
        self._training = True
        self._converged = False

    def _push_sample(self, sample: float):
        """Push a sample into the circular delay line."""
        self._buffer[self._buf_idx] = sample
        self._buf_idx = (self._buf_idx + 1) % self.num_taps

    def _compute_output(self) -> float:
        """Compute the equalizer output (dot product of weights and buffer)."""
        # Arrange buffer in correct order (oldest to newest)
        idx = self._buf_idx
        ordered = np.roll(self._buffer, -idx)
        return np.dot(self.weights, ordered)

    def _update_weights(self, error: float):
        """LMS weight update: w += mu * error * x."""
        idx = self._buf_idx
        ordered = np.roll(self._buffer, -idx)
        self.weights += self.step_size * error * ordered

    def process(
        self,
        symbols: np.ndarray,
        training_ref: np.ndarray = None,
    ) -> np.ndarray:
        """Equalize a block of symbols.

        Parameters
        ----------
        symbols : np.ndarray
            Input symbols at 1 sample/symbol.
        training_ref : np.ndarray, optional
            Known reference symbols for training mode. If provided, used
            as the desired signal for LMS error computation. Length must
            match symbols. Use None for decision-directed mode.

        Returns
        -------
        np.ndarray
            Equalized symbols.
        """
        n = len(symbols)
        out = np.empty(n, dtype=np.float64)

        for i in range(n):
            self._push_sample(symbols[i])
            y = self._compute_output()
            out[i] = y

            # Compute error
            if training_ref is not None and i < len(training_ref):
                # Training mode: error = desired - actual
                error = training_ref[i] - y
            else:
                # Decision-directed: error = nearest_level - actual
                decision = _hard_decision(y)
                error = decision - y

            self._update_weights(error)

        return out

    def process_segment(
        self,
        segment: np.ndarray,
        is_field_sync: bool = False,
        field_sync_ref: np.ndarray = None,
    ) -> np.ndarray:
        """Process one ATSC segment (832 symbols).

        For data segments, uses decision-directed mode.
        For field sync segments, uses training mode with known PN sequences.

        Parameters
        ----------
        segment : np.ndarray
            832 symbols of one ATSC segment.
        is_field_sync : bool
            True if this is a field sync segment.
        field_sync_ref : np.ndarray, optional
            Known 832-symbol reference for field sync training.

        Returns
        -------
        np.ndarray
            Equalized segment.
        """
        if is_field_sync and field_sync_ref is not None:
            return self.process(segment, training_ref=field_sync_ref)
        else:
            return self.process(segment)

    @property
    def tap_energy(self) -> float:
        """Total energy in the equalizer taps (useful for monitoring)."""
        return float(np.sum(self.weights ** 2))

    def reset(self):
        self.weights = np.zeros(self.num_taps, dtype=np.float64)
        self.weights[self.pre_taps] = 1.0
        self._buffer = np.zeros(self.num_taps, dtype=np.float64)
        self._buf_idx = 0
