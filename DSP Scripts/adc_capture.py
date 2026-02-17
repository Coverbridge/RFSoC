"""RFSoC 4x2 ADC capture via xrfdc driver and PYNQ DMA.

Configures the RF-ADC tile for direct RF sampling of UHF ATSC signals,
performs DMA burst captures into PS DDR4, and provides save/load for
offline processing.
"""

import numpy as np
from pathlib import Path

from osmium.utils.constants import (
    ADC_SAMPLE_RATE,
    ADC_HW_DECIMATION,
    DDC_OUTPUT_RATE,
    channel_center_freq,
)

# xrfdc constants (only available on the RFSoC board)
try:
    import xrfdc
    _XRFDC_AVAILABLE = True
except ImportError:
    _XRFDC_AVAILABLE = False

try:
    import pynq
    _PYNQ_AVAILABLE = True
except ImportError:
    _PYNQ_AVAILABLE = False


class ADCCapture:
    """Configure RFSoC ADC tiles and perform DMA burst captures.

    Parameters
    ----------
    overlay : pynq.Overlay, optional
        Pre-loaded PYNQ overlay. If None, loads the base overlay.
    tile_id : int
        ADC tile index (0-3). Default 0.
    block_id : int
        ADC block index within the tile. Default 0.
    """

    def __init__(self, overlay=None, tile_id: int = 0, block_id: int = 0):
        self.tile_id = tile_id
        self.block_id = block_id
        self._overlay = overlay
        self._tile = None
        self._block = None
        self._dma = None

        if _PYNQ_AVAILABLE and overlay is not None:
            self._init_hardware(overlay)

    def _init_hardware(self, overlay):
        """Initialize hardware references from the overlay."""
        self._overlay = overlay

        # Access the RF data converter block
        rf = overlay.usp_rf_data_converter
        self._tile = rf.adc_tiles[self.tile_id]
        self._block = self._tile.blocks[self.block_id]

        # DMA engine for burst capture
        if hasattr(overlay, 'axi_dma_0'):
            self._dma = overlay.axi_dma_0
        elif hasattr(overlay, 'dma'):
            self._dma = overlay.dma

    def configure_tile(self, channel: int = None, center_freq_hz: float = None):
        """Configure the ADC tile for ATSC reception.

        Parameters
        ----------
        channel : int, optional
            ATSC UHF channel number (14-51, excluding 37).
        center_freq_hz : float, optional
            Explicit center frequency in Hz. Overrides channel if both given.
        """
        if not _XRFDC_AVAILABLE:
            raise RuntimeError("xrfdc not available — run on the RFSoC board.")

        if center_freq_hz is None:
            if channel is None:
                raise ValueError("Provide either channel or center_freq_hz.")
            center_freq_hz = channel_center_freq(channel)

        nco_freq_mhz = -center_freq_hz / 1e6  # negative for downconversion

        # Nyquist zone 1: 0 to fs/2 (0-2457.6 MHz at 4915.2 MSPS)
        self._block.NyquistZone = 1

        # Configure the fine mixer for real-to-complex DDC
        self._block.MixerSettings = {
            'CoarseMixFreq': xrfdc.COARSE_MIX_BYPASS,
            'EventSource': xrfdc.EVNT_SRC_TILE,
            'FineMixerScale': xrfdc.MIXER_SCALE_1P0,
            'Freq': nco_freq_mhz,
            'MixerMode': xrfdc.MIXER_MODE_R2C,
            'MixerType': xrfdc.MIXER_TYPE_FINE,
            'PhaseOffset': 0.0,
        }
        self._block.UpdateEvent(xrfdc.EVENT_MIXER)

        self._configured_freq = center_freq_hz
        return {
            'center_freq_hz': center_freq_hz,
            'nco_freq_mhz': nco_freq_mhz,
            'adc_sample_rate': ADC_SAMPLE_RATE,
            'hw_decimation': ADC_HW_DECIMATION,
            'ddc_output_rate': DDC_OUTPUT_RATE,
        }

    def capture_burst(self, num_samples: int = 2**20) -> np.ndarray:
        """Capture a burst of IQ samples via DMA.

        Parameters
        ----------
        num_samples : int
            Number of complex samples to capture. Default 1M samples
            (~1.6 ms at 614.4 MSPS).

        Returns
        -------
        np.ndarray
            Complex64 array of shape (num_samples,) with I+jQ samples.
        """
        if self._dma is None:
            raise RuntimeError("DMA not available. Load an overlay first.")

        # Allocate contiguous buffer for DMA
        # Each complex sample = 2x int16 (I, Q) = 4 bytes
        buf = pynq.allocate(shape=(num_samples * 2,), dtype=np.int16)

        # Start DMA transfer
        self._dma.recvchannel.transfer(buf)
        self._dma.recvchannel.wait()

        # Convert interleaved int16 I/Q to complex64
        iq = buf.astype(np.float32).view(np.complex64)
        buf.freebuffer()

        return iq

    @staticmethod
    def save_to_file(samples: np.ndarray, path: str, metadata: dict = None):
        """Save captured IQ samples to a .npz file.

        Parameters
        ----------
        samples : np.ndarray
            Complex IQ samples.
        path : str
            Output file path (.npz).
        metadata : dict, optional
            Additional metadata (center_freq, sample_rate, etc.).
        """
        save_dict = {'samples': samples}
        if metadata is not None:
            for k, v in metadata.items():
                save_dict[f'meta_{k}'] = np.array(v)
        np.savez_compressed(path, **save_dict)

    @staticmethod
    def load_from_file(path: str) -> tuple:
        """Load IQ samples from a .npz file.

        Returns
        -------
        samples : np.ndarray
            Complex IQ samples.
        metadata : dict
            Any saved metadata.
        """
        data = np.load(path, allow_pickle=False)
        samples = data['samples']
        metadata = {}
        for key in data.files:
            if key.startswith('meta_'):
                metadata[key[5:]] = data[key].item()
        return samples, metadata

    @staticmethod
    def load_raw_iq(path: str, dtype: str = 'complex64') -> np.ndarray:
        """Load raw IQ from a binary file (GNU Radio .fc32 / .sc16 format).

        Parameters
        ----------
        path : str
            Path to raw binary IQ file.
        dtype : str
            'complex64' for float32 IQ (.fc32),
            'int16' for interleaved int16 IQ (.sc16).
        """
        if dtype == 'complex64':
            return np.fromfile(path, dtype=np.complex64)
        elif dtype == 'int16':
            raw = np.fromfile(path, dtype=np.int16)
            return (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")
