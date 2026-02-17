"""Full 8VSB demodulation chain orchestrator.

Wires together all demod blocks in the correct order and manages
the data flow from complex baseband IQ through to MPEG-TS packets.
"""

import numpy as np

from osmium.demod.agc import AGC
from osmium.demod.dc_blocker import DCBlocker
from osmium.demod.fpll import FPLL
from osmium.demod.matched_filter import MatchedFilter
from osmium.demod.timing_recovery import TimingRecovery
from osmium.demod.equalizer import AdaptiveEqualizer
from osmium.demod.field_sync import FieldSyncDetector
from osmium.demod.trellis_decoder import TrellisDecoder
from osmium.demod.deinterleaver import ByteDeinterleaver
from osmium.demod.reed_solomon import ReedSolomonDecoder
from osmium.demod.derandomizer import Derandomizer
from osmium.utils.constants import (
    TARGET_BASEBAND_RATE,
    ATSC_SEGMENT_DATA_LENGTH,
    RS_N,
    RS_K,
    TS_PACKET_SIZE,
)


class DemodChain:
    """Complete ATSC 8VSB demodulation chain.

    Processes complex baseband IQ samples (at ~21.524 MSPS, 2 sps) through
    the full chain and outputs MPEG-TS packets.

    Parameters
    ----------
    sample_rate : float
        Input sample rate. Default ~21.524 MSPS.
    fast_mode : bool
        If True, use block-based AGC/DC removal for speed. Default False.
    """

    def __init__(
        self,
        sample_rate: float = TARGET_BASEBAND_RATE,
        fast_mode: bool = False,
    ):
        self.sample_rate = sample_rate
        self.fast_mode = fast_mode

        # Instantiate all blocks
        self.agc = AGC(alpha=1e-4, target_power=1.0)
        self.dc_blocker = DCBlocker(alpha=1e-4)
        self.fpll = FPLL(sample_rate=sample_rate)
        self.matched_filter = MatchedFilter(num_taps=65, samples_per_symbol=2)
        self.timing_recovery = TimingRecovery(samples_per_symbol=2.0)
        self.equalizer = AdaptiveEqualizer(num_taps=64, pre_taps=16)
        self.field_sync = FieldSyncDetector()
        self.trellis_decoder = TrellisDecoder()
        self.deinterleaver = ByteDeinterleaver()
        self.rs_decoder = ReedSolomonDecoder()
        self.derandomizer = Derandomizer()

        # Internal accumulation buffers
        self._rs_buffer = np.array([], dtype=np.uint8)
        self._ts_packets = []

    def process(self, iq_samples: np.ndarray) -> list:
        """Process a block of complex baseband IQ samples.

        Parameters
        ----------
        iq_samples : np.ndarray
            Complex baseband samples at ~21.524 MSPS (2 sps).

        Returns
        -------
        list of np.ndarray
            Complete 188-byte MPEG-TS packets extracted from this block.
        """
        # --- Front-end ---
        if self.fast_mode:
            x = self.agc.process_block(iq_samples)
            x = self.dc_blocker.process_block(x)
        else:
            x = self.agc.process(iq_samples)
            x = self.dc_blocker.process(x)

        # --- Carrier recovery ---
        x_real = self.fpll.process(x)

        # --- Matched filter ---
        x_filt = self.matched_filter.process(x_real)

        # --- Timing recovery (2 sps -> 1 sps) ---
        symbols = self.timing_recovery.process(x_filt)

        if len(symbols) == 0:
            return []

        # --- Equalization ---
        eq_symbols = self.equalizer.process(symbols)

        # --- Field sync detection + framing ---
        segments = self.field_sync.process(eq_symbols)

        ts_packets = []

        for seg_info in segments:
            if seg_info['is_field_sync']:
                # Field sync segment: used for equalizer training, not data
                self.derandomizer.reset_field()
                continue

            # --- Trellis decode ---
            decoded_bytes = self.trellis_decoder.decode_segment(seg_info['data'])

            if len(decoded_bytes) == 0:
                continue

            # --- Byte deinterleave ---
            deinterleaved = self.deinterleaver.process(decoded_bytes)

            # Accumulate bytes for RS decoding
            self._rs_buffer = np.concatenate([self._rs_buffer, deinterleaved])

            # --- RS decode when we have a complete codeword ---
            while len(self._rs_buffer) >= RS_N:
                codeword = self._rs_buffer[:RS_N]
                self._rs_buffer = self._rs_buffer[RS_N:]

                data, errors = self.rs_decoder.decode(codeword)
                if data is None:
                    continue  # uncorrectable, skip

                # --- Derandomize ---
                ts_packet = self.derandomizer.process_packet(data)

                # Validate sync byte
                if ts_packet[0] == 0x47:
                    ts_packets.append(ts_packet)

        return ts_packets

    @property
    def status(self) -> dict:
        """Return current demodulator status."""
        return {
            'fpll_locked': self.fpll.is_locked,
            'fpll_freq_offset_hz': self.fpll.freq_offset_hz,
            'fpll_lock_indicator': self.fpll.lock_indicator,
            'synced': self.field_sync.is_synced,
            'agc_gain': self.agc.gain,
            'eq_tap_energy': self.equalizer.tap_energy,
            'rs_total_codewords': self.rs_decoder.total_codewords,
            'rs_corrected': self.rs_decoder.corrected_codewords,
            'rs_uncorrectable': self.rs_decoder.uncorrectable_codewords,
            'rs_error_rate': self.rs_decoder.error_rate,
        }

    def reset(self):
        self.agc.reset()
        self.dc_blocker.reset()
        self.fpll.reset()
        self.matched_filter.reset()
        self.timing_recovery.reset()
        self.equalizer.reset()
        self.field_sync.reset()
        self.trellis_decoder.reset()
        self.deinterleaver.reset()
        self.rs_decoder.reset_stats()
        self.derandomizer.reset_field()
        self._rs_buffer = np.array([], dtype=np.uint8)
        self._ts_packets = []
