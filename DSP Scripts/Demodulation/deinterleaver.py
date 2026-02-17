"""Convolutional byte deinterleaver for ATSC.

The ATSC transmitter uses a convolutional interleaver with B=52 branches
and M=4 bytes depth per branch. The deinterleaver is the inverse operation.

Each branch i (0..51) has a FIFO delay of (51-i)*M = (51-i)*4 bytes.
Branch 0 has maximum delay (204 bytes), branch 51 has zero delay.
Bytes are assigned to branches in round-robin order.
"""

import numpy as np

from osmium.utils.constants import INTERLEAVER_B, INTERLEAVER_M


class ByteDeinterleaver:
    """ATSC convolutional byte deinterleaver.

    Parameters
    ----------
    B : int
        Number of branches. Default 52.
    M : int
        Delay depth per branch. Default 4 bytes.
    """

    def __init__(self, B: int = INTERLEAVER_B, M: int = INTERLEAVER_M):
        self.B = B
        self.M = M

        # Create delay line FIFOs for each branch
        # Branch i has delay (B-1-i)*M bytes
        self._fifos = []
        for i in range(B):
            delay = (B - 1 - i) * M
            self._fifos.append(np.zeros(delay, dtype=np.uint8) if delay > 0 else None)

        self._fifo_ptrs = [0] * B
        self._branch_counter = 0

    def process_byte(self, byte_in: int) -> int:
        """Process one byte through the deinterleaver.

        Parameters
        ----------
        byte_in : int
            Input byte value (0-255).

        Returns
        -------
        int
            Output byte value (0-255), delayed by the appropriate amount.
        """
        branch = self._branch_counter % self.B

        if self._fifos[branch] is None:
            # Zero-delay branch: pass through
            out = byte_in
        else:
            fifo = self._fifos[branch]
            ptr = self._fifo_ptrs[branch]

            # Read the oldest value and replace with new
            out = int(fifo[ptr])
            fifo[ptr] = byte_in
            self._fifo_ptrs[branch] = (ptr + 1) % len(fifo)

        self._branch_counter += 1
        return out

    def process(self, data: np.ndarray) -> np.ndarray:
        """Process a block of bytes through the deinterleaver.

        Parameters
        ----------
        data : np.ndarray
            Input bytes (uint8).

        Returns
        -------
        np.ndarray
            Deinterleaved bytes (uint8).
        """
        out = np.empty_like(data)
        for i in range(len(data)):
            out[i] = self.process_byte(int(data[i]))
        return out

    @property
    def total_delay(self) -> int:
        """Total delay through the deinterleaver in bytes."""
        return sum((self.B - 1 - i) * self.M for i in range(self.B))

    def reset(self):
        for i in range(self.B):
            delay = (self.B - 1 - i) * self.M
            if delay > 0:
                self._fifos[i] = np.zeros(delay, dtype=np.uint8)
            self._fifo_ptrs[i] = 0
        self._branch_counter = 0
