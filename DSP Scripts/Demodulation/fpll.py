"""Frequency/Phase Locked Loop for ATSC carrier recovery.

Based on GNU Radio's atsc_fpll_impl.cc. The ATSC pilot tone at +309.441 kHz
above the lower band edge provides the carrier reference. The FPLL locks to
this pilot and produces real-valued (I-channel) baseband output.
"""

import numpy as np

from osmium.utils.constants import ATSC_PILOT_FREQ, TARGET_BASEBAND_RATE


class FPLL:
    """ATSC Frequency/Phase Locked Loop for carrier recovery.

    Parameters
    ----------
    sample_rate : float
        Input sample rate in Hz. Default ~21.524 MSPS.
    alpha : float
        Phase tracking gain. Default 0.01 (from gr-dtv).
    beta : float
        Frequency tracking gain. Default alpha^2/4 (from gr-dtv).
    initial_freq_offset : float
        Initial NCO frequency offset in Hz. The ATSC pilot should be near
        DC after baseband conversion, so this should be small. Default 0.
    """

    def __init__(
        self,
        sample_rate: float = TARGET_BASEBAND_RATE,
        alpha: float = 0.01,
        beta: float = None,
        initial_freq_offset: float = 0.0,
    ):
        self.sample_rate = sample_rate
        self.alpha = alpha
        self.beta = beta if beta is not None else (alpha ** 2) / 4.0

        # NCO state: phase in radians, frequency in radians/sample
        self._phase = 0.0
        self._freq = 2.0 * np.pi * initial_freq_offset / sample_rate

        # Lock detector state
        self._lock_alpha = 0.001
        self._lock_accum = 0.0

    def process(self, samples: np.ndarray) -> np.ndarray:
        """Perform carrier recovery on complex baseband samples.

        The FPLL mixes the input with a local NCO to remove residual carrier
        offset, then uses the pilot tone (which should appear near DC in the
        I channel) for phase/frequency error estimation.

        Parameters
        ----------
        samples : np.ndarray
            Complex baseband samples (I+jQ) at ~21.524 MSPS.

        Returns
        -------
        np.ndarray
            Real-valued carrier-recovered samples (I channel only).
            The 8VSB information is entirely in the real (I) component.
        """
        n = len(samples)
        out = np.empty(n, dtype=np.float64)
        phase = self._phase
        freq = self._freq
        alpha = self.alpha
        beta = self.beta

        for i in range(n):
            # Mix input with NCO: multiply by e^{-j*phase}
            nco = np.exp(-1j * phase)
            mixed = samples[i] * nco

            # Output is the real part (I channel contains the 8VSB data)
            out[i] = mixed.real

            # Phase error: use atan2 of the mixed signal
            # For ATSC, the pilot should produce a positive DC component in I.
            # Phase error is the angle of the mixed signal relative to the
            # real axis. We want the Q component to be zero.
            error = np.arctan2(mixed.imag, mixed.real)

            # Update NCO frequency and phase (PI controller)
            freq += beta * error
            phase += freq + alpha * error

            # Wrap phase to [-pi, pi] periodically to prevent float overflow
            if phase > np.pi:
                phase -= 2.0 * np.pi
            elif phase < -np.pi:
                phase += 2.0 * np.pi

            # Lock detector: running average of cos(error)
            # When locked, error ≈ 0 so cos(error) ≈ 1
            self._lock_accum += self._lock_alpha * (
                np.cos(error) - self._lock_accum
            )

        self._phase = phase
        self._freq = freq
        return out

    @property
    def is_locked(self) -> bool:
        """True if the loop appears to be locked (lock indicator > 0.8)."""
        return self._lock_accum > 0.8

    @property
    def lock_indicator(self) -> float:
        """Lock quality indicator: 0.0 = unlocked, 1.0 = perfect lock."""
        return float(self._lock_accum)

    @property
    def freq_offset_hz(self) -> float:
        """Current estimated frequency offset in Hz."""
        return self._freq * self.sample_rate / (2.0 * np.pi)

    def reset(self):
        self._phase = 0.0
        self._freq = 0.0
        self._lock_accum = 0.0
