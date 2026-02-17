"""Reed-Solomon (207, 187, t=10) decoder for ATSC.

ATSC uses RS(207, 187) over GF(2^8) with primitive polynomial
p(x) = x^8 + x^4 + x^3 + x^2 + 1 (0x11D) and FCR=0.
Corrects up to t=10 byte errors per 207-byte codeword.

Uses the 'reedsolo' library for the actual Galois field arithmetic
and Berlekamp-Massey / Forney decoding.
"""

import numpy as np

from osmium.utils.constants import RS_N, RS_K, RS_T, RS_PARITY

try:
    from reedsolo import RSCodec, ReedSolomonError
    _REEDSOLO_AVAILABLE = True
except ImportError:
    _REEDSOLO_AVAILABLE = False


class ReedSolomonDecoder:
    """ATSC RS(207,187,t=10) decoder.

    Parameters
    ----------
    n : int
        Codeword length. Default 207.
    k : int
        Data length. Default 187.
    """

    def __init__(self, n: int = RS_N, k: int = RS_K):
        self.n = n
        self.k = k
        self.t = (n - k) // 2  # error correction capability

        if not _REEDSOLO_AVAILABLE:
            raise ImportError(
                "reedsolo package required for RS decoding. "
                "Install with: pip install reedsolo"
            )

        # ATSC uses a shortened RS code: RS(207,187) is derived from
        # RS(255,235) by shortening 48 bytes. reedsolo handles this
        # with nsym = n-k = 20 parity symbols.
        self._codec = RSCodec(nsym=RS_PARITY, nsize=255, fcr=0, prim=0x11D)

        # Statistics
        self.total_codewords = 0
        self.corrected_codewords = 0
        self.uncorrectable_codewords = 0
        self.total_errors_corrected = 0

    def decode(self, codeword: np.ndarray) -> tuple:
        """Decode a single RS(207,187) codeword.

        Parameters
        ----------
        codeword : np.ndarray
            207-byte codeword (187 data + 20 parity), uint8.

        Returns
        -------
        data : np.ndarray
            187-byte corrected data, or None if uncorrectable.
        errors : int
            Number of byte errors corrected, or -1 if uncorrectable.
        """
        self.total_codewords += 1

        # reedsolo expects the codeword to be 255 bytes for a full RS code.
        # For shortened codes, we pad with zeros at the front.
        pad_len = 255 - self.n  # 48 bytes of padding
        padded = bytes(pad_len) + bytes(codeword)

        try:
            decoded_msg, decoded_msgecc, errata_pos = self._codec.decode(padded)
            num_errors = len(errata_pos)

            if num_errors > 0:
                self.corrected_codewords += 1
                self.total_errors_corrected += num_errors

            # Extract the 187 data bytes (skip the padding)
            data = np.frombuffer(bytes(decoded_msg), dtype=np.uint8)
            # reedsolo returns the message without padding
            # We need the last 187 bytes of the decoded message
            if len(data) > self.k:
                data = data[-self.k:]

            return data, num_errors

        except ReedSolomonError:
            self.uncorrectable_codewords += 1
            return None, -1

    def decode_stream(self, data: np.ndarray) -> list:
        """Decode a stream of bytes, extracting RS codewords.

        Parameters
        ----------
        data : np.ndarray
            Stream of bytes. Must be a multiple of 207 bytes.

        Returns
        -------
        list of tuple
            List of (decoded_data, error_count) for each codeword.
        """
        results = []
        n_codewords = len(data) // self.n

        for i in range(n_codewords):
            codeword = data[i * self.n:(i + 1) * self.n]
            decoded, errors = self.decode(codeword)
            results.append((decoded, errors))

        return results

    @property
    def error_rate(self) -> float:
        """Fraction of codewords with errors (correctable + uncorrectable)."""
        if self.total_codewords == 0:
            return 0.0
        return (self.corrected_codewords + self.uncorrectable_codewords) / \
            self.total_codewords

    @property
    def uncorrectable_rate(self) -> float:
        """Fraction of uncorrectable codewords."""
        if self.total_codewords == 0:
            return 0.0
        return self.uncorrectable_codewords / self.total_codewords

    def reset_stats(self):
        self.total_codewords = 0
        self.corrected_codewords = 0
        self.uncorrectable_codewords = 0
        self.total_errors_corrected = 0
