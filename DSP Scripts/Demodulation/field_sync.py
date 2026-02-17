"""ATSC segment and field sync detection.

Detects the 4-symbol segment sync pattern at the start of each 832-symbol
segment, and the PN511+PN63 field sync pattern to identify field boundaries.
Provides framing information for downstream processing.
"""

import numpy as np

from osmium.utils.constants import (
    SEGMENT_SYNC_PATTERN,
    ATSC_SEGMENT_LENGTH,
    ATSC_SEGMENT_SYNC_LENGTH,
    ATSC_SEGMENTS_PER_FIELD,
    PN511,
    PN63,
    FIELD_SYNC_PN511_START,
    FIELD_SYNC_PN63_1_START,
    VSB8_LEVELS,
)


class FieldSyncDetector:
    """Detects ATSC segment sync and field sync patterns.

    Operates on equalized 8-level symbols at 1 sample/symbol.
    Outputs framed segments with field boundary markers.

    Parameters
    ----------
    seg_sync_threshold : float
        Correlation threshold for segment sync detection (normalized).
        Default 0.7.
    field_sync_threshold : float
        Correlation threshold for field sync PN511 detection (normalized).
        Default 0.7.
    """

    def __init__(
        self,
        seg_sync_threshold: float = 0.7,
        field_sync_threshold: float = 0.7,
    ):
        self.seg_sync_threshold = seg_sync_threshold
        self.field_sync_threshold = field_sync_threshold

        # State
        self._synced = False
        self._segment_counter = 0          # counts segments within a field
        self._field_number = 0             # 0 = even, 1 = odd
        self._symbol_buffer = np.array([], dtype=np.float64)
        self._expected_next_sync = 0       # expected position of next seg sync
        self._sync_miss_count = 0

        # Normalized segment sync template (scaled to ±5)
        self._seg_sync_ref = SEGMENT_SYNC_PATTERN / np.linalg.norm(SEGMENT_SYNC_PATTERN)

        # PN511 reference (scaled to ±5 for correlation, mapped from ±1)
        self._pn511_ref = PN511.astype(np.float64) * 5.0
        self._pn511_norm = self._pn511_ref / np.linalg.norm(self._pn511_ref)

    def _correlate_seg_sync(self, symbols: np.ndarray, pos: int) -> float:
        """Compute normalized correlation with segment sync at position pos."""
        if pos + ATSC_SEGMENT_SYNC_LENGTH > len(symbols):
            return 0.0
        candidate = symbols[pos:pos + ATSC_SEGMENT_SYNC_LENGTH]
        norm = np.linalg.norm(candidate)
        if norm < 1e-10:
            return 0.0
        return float(np.dot(candidate / norm, self._seg_sync_ref))

    def _correlate_field_sync(self, segment: np.ndarray) -> tuple:
        """Check if a segment is a field sync by correlating with PN511.

        Returns
        -------
        is_field_sync : bool
        field_parity : int
            0 for even field, 1 for odd field (determined by PN63 inversion).
        correlation : float
        """
        if len(segment) < ATSC_SEGMENT_LENGTH:
            return False, 0, 0.0

        # Extract the PN511 portion (symbols 4..514)
        pn_section = segment[FIELD_SYNC_PN511_START:FIELD_SYNC_PN511_START + 511]
        norm = np.linalg.norm(pn_section)
        if norm < 1e-10:
            return False, 0, 0.0

        corr = np.dot(pn_section / norm, self._pn511_norm)

        if abs(corr) > self.field_sync_threshold:
            # Determine field parity from the middle PN63 segment
            # In field 1, the middle PN63 is inverted relative to field 0
            pn63_section = segment[FIELD_SYNC_PN63_1_START:FIELD_SYNC_PN63_1_START + 63]
            pn63_ref = PN63.astype(np.float64) * 5.0
            pn63_norm = np.linalg.norm(pn63_ref)
            pn63_corr = np.dot(pn63_section, pn63_ref) / (
                np.linalg.norm(pn63_section) * pn63_norm + 1e-10
            )
            field_parity = 0 if pn63_corr > 0 else 1
            return True, field_parity, abs(corr)

        return False, 0, abs(corr)

    def find_initial_sync(self, symbols: np.ndarray) -> int:
        """Search for the first segment sync position in a symbol stream.

        Uses sliding correlation to find the segment sync pattern.

        Returns
        -------
        int
            Index of the first segment sync, or -1 if not found.
        """
        best_pos = -1
        best_corr = 0.0

        for pos in range(len(symbols) - ATSC_SEGMENT_LENGTH):
            corr = self._correlate_seg_sync(symbols, pos)
            if corr > self.seg_sync_threshold and corr > best_corr:
                # Verify periodicity: check if sync appears 832 symbols later
                next_corr = self._correlate_seg_sync(
                    symbols, pos + ATSC_SEGMENT_LENGTH
                )
                if next_corr > self.seg_sync_threshold:
                    best_pos = pos
                    best_corr = corr
                    break  # Found a confirmed sync with periodicity

        return best_pos

    def process(self, symbols: np.ndarray) -> list:
        """Process a stream of symbols and output framed segments.

        Parameters
        ----------
        symbols : np.ndarray
            Equalized symbols at 1 sample/symbol.

        Returns
        -------
        list of dict
            Each dict contains:
            - 'data': np.ndarray of 832 symbols
            - 'is_field_sync': bool
            - 'field_number': int (0 or 1)
            - 'segment_in_field': int (0..312)
        """
        # Append new symbols to buffer
        self._symbol_buffer = np.concatenate([self._symbol_buffer, symbols])
        segments = []

        if not self._synced:
            sync_pos = self.find_initial_sync(self._symbol_buffer)
            if sync_pos < 0:
                # Keep only the last segment's worth of symbols for overlap
                if len(self._symbol_buffer) > ATSC_SEGMENT_LENGTH * 2:
                    self._symbol_buffer = self._symbol_buffer[-ATSC_SEGMENT_LENGTH:]
                return segments
            self._symbol_buffer = self._symbol_buffer[sync_pos:]
            self._synced = True
            self._expected_next_sync = 0

        # Extract complete segments
        while len(self._symbol_buffer) >= ATSC_SEGMENT_LENGTH:
            segment = self._symbol_buffer[:ATSC_SEGMENT_LENGTH]

            # Verify segment sync at the expected position
            corr = self._correlate_seg_sync(segment, 0)
            if corr < self.seg_sync_threshold * 0.5:
                # Sync lost — try to re-acquire
                self._sync_miss_count += 1
                if self._sync_miss_count > 5:
                    self._synced = False
                    self._sync_miss_count = 0
                    return segments
            else:
                self._sync_miss_count = 0

            # Check for field sync
            is_fs, field_parity, fs_corr = self._correlate_field_sync(segment)

            if is_fs:
                self._field_number = field_parity
                self._segment_counter = 0

            segments.append({
                'data': segment.copy(),
                'is_field_sync': is_fs,
                'field_number': self._field_number,
                'segment_in_field': self._segment_counter,
            })

            self._segment_counter += 1
            if self._segment_counter >= ATSC_SEGMENTS_PER_FIELD:
                self._segment_counter = 0
                self._field_number = 1 - self._field_number

            self._symbol_buffer = self._symbol_buffer[ATSC_SEGMENT_LENGTH:]

        return segments

    @property
    def is_synced(self) -> bool:
        return self._synced

    def reset(self):
        self._synced = False
        self._segment_counter = 0
        self._field_number = 0
        self._symbol_buffer = np.array([], dtype=np.float64)
        self._sync_miss_count = 0
