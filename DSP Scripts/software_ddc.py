## DDC for RFSoC 4x2

import numpy as np
from scipy.signal import firwin, lfilter, decimate, resample_poly
from math import gcd

from osmium.utils.constants import (
    DDC_OUTPUT_RATE,
    TARGET_BASEBAND_RATE,
    ATSC_CHANNEL_BW,
    ATSC_SYMBOL_RATE,
)


class SoftwareDDC:

    def __init__(
        self,
        input_rate: float = DDC_OUTPUT_RATE,
        output_rate: float = TARGET_BASEBAND_RATE,
        guard_bw: float = 0.1,
    ):
        self.input_rate = input_rate
        self.output_rate = output_rate
        self.guard_bw = guard_bw

        # Compute rational resampling ratio: output_rate / input_rate = up / down
        # Use integer approximation to avoid floating-point drift
        self._compute_stages()

    def _compute_stages(self):

        # Find rational approximation
        # 614.4 MSPS -> 21.524... MSPS
        # Ratio = 21524475.52 / 614400000 = 684 * 4500000 / (286 * 2 * 614400000)
        # Simplify: use integer ratio with sufficient precision
        ratio = self.output_rate / self.input_rate

        # Scale to integers: multiply both by a large number and reduce
        # For better precision, express rates in kHz
        input_khz = round(self.input_rate / 1000)
        output_khz = round(self.output_rate / 1000)
        g = gcd(input_khz, output_khz)
        self.up = output_khz // g
        self.down = input_khz // g

        # Anti-alias filter: cutoff at half the output bandwidth
        # ATSC signal bandwidth is ~5.38 MHz (symbol_rate / 2 * (1 + alpha))
        # with alpha=0.1152, that's 5.381 * 1.1152 ≈ 6 MHz
        # At the output rate, Nyquist is output_rate/2 ≈ 10.76 MHz
        # Cutoff normalized to output Nyquist: 5.38 / 10.76 ≈ 0.5
        signal_bw = ATSC_SYMBOL_RATE / 2.0 * (1.0 + 0.1152)
        nyquist_out = self.output_rate / 2.0
        self.cutoff_norm = min(signal_bw * (1.0 + self.guard_bw) / nyquist_out, 0.95)


        # decimates by at most 10x (for filter stability)
        total_decim = self.down / self.up
        self._stages = []
        remaining = total_decim

        # Stage design: decimate in factors ≤ 10
        for factor in [8, 4, 2]:
            while remaining >= factor * 0.99:
                self._stages.append(factor)
                remaining /= factor
        if remaining > 1.01:
            # Remaining fractional part handled by resample_poly
            self._use_resample_poly = True
        else:
            self._use_resample_poly = False

    def process(self, samples: np.ndarray) -> np.ndarray:
        """Decimate input samples to the target baseband rate.

        Parameters:
        samples : np.ndarray
            Complex input samples at the DDC output rate.
        np.ndarray
            Complex baseband samples at ~21.524 MSPS.
        """
        # Primary approach: use resample_poly for the full rational conversion
        # This is simpler and more accurate than multi-stage integer decimation
        out = resample_poly(samples, self.up, self.down)
        return out.astype(np.complex64)

    def process_staged(self, samples: np.ndarray) -> np.ndarray:

        x = samples
        current_rate = self.input_rate

        for stage_decim in self._stages:
            # Design anti-alias filter for this stage
            cutoff = 0.8 / stage_decim  # normalized to Nyquist before decimation
            n_taps = max(31, stage_decim * 10 + 1)
            h = firwin(n_taps, cutoff)

            # Apply filter and decimate
            x = lfilter(h, 1.0, x)[::stage_decim]
            current_rate /= stage_decim

        # If there's a remaining fractional resampling needed
        if self._use_resample_poly:
            remaining_ratio = self.output_rate / current_rate
            up = round(remaining_ratio * 1000)
            down = 1000
            g = gcd(up, down)
            x = resample_poly(x, up // g, down // g)

        return x.astype(np.complex64)

    @property
    def actual_output_rate(self) -> float:
        #The actual output sample rate after rational resampling.
        return self.input_rate * self.up / self.down

    def __repr__(self):
        return (
            f"SoftwareDDC(input={self.input_rate/1e6:.1f} MSPS, "
            f"output={self.actual_output_rate/1e6:.4f} MSPS, "
            f"ratio={self.up}/{self.down})"
        )
