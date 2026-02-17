"""12-phase trellis decoder for ATSC 8VSB.

ATSC uses a 12-way interleaved trellis-coded modulation (TCM) scheme.
Each of the 12 phases has an independent rate-2/3 trellis encoder with
a 4-state convolutional code mapping 2 input bits to 3 output bits
(selecting one of 8 VSB levels).

The decoder runs 12 independent Viterbi decoders, one per trellis phase.
Symbols within each segment are assigned to phases in round-robin order.

Reference: ATSC A/53 Part 2, Section 5.3 (Trellis Coding)
           GNU Radio gr-dtv: atsc_viterbi_decoder_impl.cc
"""

import numpy as np

from osmium.utils.constants import (
    TRELLIS_PHASES,
    TRELLIS_STATES,
    VITERBI_TRACEBACK_DEPTH,
    ATSC_SEGMENT_LENGTH,
    ATSC_SEGMENT_SYNC_LENGTH,
    ATSC_SEGMENT_DATA_LENGTH,
    VSB8_LEVELS,
)

# ---------------------------------------------------------------------------
# ATSC Trellis Code Definition
# ---------------------------------------------------------------------------
# The ATSC trellis encoder is a rate-2/3 code with 4 states.
# Input: 2 bits (X2, X1)
# State: 2 bits (D2, D1) -- D1 is the delay element in the encoder
# Output: 3 bits (Z2, Z1, Z0)
#   Z2 = X2 (uncoded, MSB)
#   Z1 = X1 (precoded)
#   Z0 = D1 XOR X1 (coded bit from convolutional encoder)
#
# The 3-bit output Z2Z1Z0 maps to one of 8 levels: {-7,-5,-3,-1,+1,+3,+5,+7}
# Mapping: Z2Z1Z0 in binary (0..7) -> level = 2*val - 7
#   000 -> -7, 001 -> -5, 010 -> -3, 011 -> -1,
#   100 -> +1, 101 -> +3, 110 -> +5, 111 -> +7

# Trellis transitions: for each state and input, what is the next state
# and output symbol.
# State = value of D1 (1 bit, but we track the full encoder state as 2-bit
# for the precoder+convolutional pair)
#
# Simplified model:
# The encoder has 1 delay element for the convolutional code: D1
# State = D1 (0 or 1), 2 states effectively used for the Viterbi
# But the precoder adds another state bit. Per A/53, the full trellis
# has 4 states when considering the precoder.
#
# For Viterbi decoding, we model 4 states:
#   state = (precoder_state, encoder_state) = (P, D1), 2 bits

# Build the trellis lookup tables
NUM_STATES = 4
NUM_SYMBOLS = 8  # 8VSB levels


def _build_trellis():
    """Build trellis transition and output tables.

    Returns
    -------
    next_state : ndarray, shape (NUM_STATES, 4)
        next_state[s, input] = next state for input (2-bit: X2,X1)
    output_symbol : ndarray, shape (NUM_STATES, 4)
        output_symbol[s, input] = 8VSB output level index (0-7)
    """
    next_state = np.zeros((NUM_STATES, 4), dtype=np.int32)
    output_symbol = np.zeros((NUM_STATES, 4), dtype=np.int32)

    for state in range(NUM_STATES):
        p = (state >> 1) & 1   # precoder state
        d1 = state & 1         # convolutional encoder state

        for inp in range(4):
            x2 = (inp >> 1) & 1  # uncoded bit (MSB)
            x1 = inp & 1         # input to precoder

            # Precoder: X1_precoded = X1 XOR P
            x1_precoded = x1 ^ p

            # Convolutional encoder output
            z0 = d1 ^ x1_precoded   # coded bit
            z1 = x1_precoded         # systematic
            z2 = x2                  # uncoded

            # Next state
            new_p = x1_precoded      # precoder delay = current precoded input
            new_d1 = x1_precoded     # encoder delay = current input

            next_s = (new_p << 1) | new_d1
            symbol_idx = (z2 << 2) | (z1 << 1) | z0

            next_state[state, inp] = next_s
            output_symbol[state, inp] = symbol_idx

    return next_state, output_symbol


NEXT_STATE, OUTPUT_SYMBOL = _build_trellis()

# Precompute: for each state, list all possible (prev_state, input) pairs
# that lead to it. This is needed for the ACS step.
PREV_STATES = {}  # prev_states[next_s] = list of (prev_s, input, symbol_idx)
for ns in range(NUM_STATES):
    PREV_STATES[ns] = []
for s in range(NUM_STATES):
    for inp in range(4):
        ns = NEXT_STATE[s, inp]
        sym_idx = OUTPUT_SYMBOL[s, inp]
        PREV_STATES[ns].append((s, inp, sym_idx))


class SingleViterbi:
    """Viterbi decoder for one trellis phase (4-state, rate 2/3).

    Parameters
    ----------
    traceback_depth : int
        Number of symbols to trace back. Default 32.
    """

    def __init__(self, traceback_depth: int = VITERBI_TRACEBACK_DEPTH):
        self.traceback_depth = traceback_depth
        self.reset()

    def reset(self):
        self._path_metric = np.zeros(NUM_STATES, dtype=np.float64)
        self._path_metric[1:] = 1e9  # start in state 0

        # Traceback memory: store the surviving predecessor state
        # and decoded input at each time step
        self._traceback_states = np.zeros(
            (self.traceback_depth, NUM_STATES), dtype=np.int32
        )
        self._traceback_inputs = np.zeros(
            (self.traceback_depth, NUM_STATES), dtype=np.int32
        )
        self._tb_ptr = 0
        self._count = 0

    def decode_symbol(self, received_symbol: float) -> int:
        """Decode one received 8VSB symbol.

        Parameters
        ----------
        received_symbol : float
            Soft (analog) received symbol value.

        Returns
        -------
        int
            Decoded 2-bit data value (0..3), or -1 if traceback not yet full.
        """
        new_metric = np.full(NUM_STATES, 1e9, dtype=np.float64)
        new_prev = np.zeros(NUM_STATES, dtype=np.int32)
        new_input = np.zeros(NUM_STATES, dtype=np.int32)

        for ns in range(NUM_STATES):
            for prev_s, inp, sym_idx in PREV_STATES[ns]:
                # Branch metric: squared Euclidean distance between
                # received symbol and expected symbol level
                expected_level = VSB8_LEVELS[sym_idx]
                branch_metric = (received_symbol - expected_level) ** 2

                candidate_metric = self._path_metric[prev_s] + branch_metric

                if candidate_metric < new_metric[ns]:
                    new_metric[ns] = candidate_metric
                    new_prev[ns] = prev_s
                    new_input[ns] = inp

        # Store traceback info
        tb_idx = self._tb_ptr % self.traceback_depth
        self._traceback_states[tb_idx] = new_prev
        self._traceback_inputs[tb_idx] = new_input
        self._tb_ptr += 1
        self._count += 1

        # Update path metrics (normalize to prevent overflow)
        self._path_metric = new_metric - np.min(new_metric)

        # Perform traceback if we have enough history
        if self._count >= self.traceback_depth:
            return self._traceback()
        return -1

    def _traceback(self) -> int:
        """Trace back through the trellis to find the decoded data."""
        # Start from the state with the best metric
        state = int(np.argmin(self._path_metric))

        # Trace back through the stored path
        tb_end = (self._tb_ptr - 1) % self.traceback_depth
        tb_start = (self._tb_ptr - self.traceback_depth) % self.traceback_depth

        decoded_input = 0
        for i in range(self.traceback_depth):
            idx = (tb_end - i) % self.traceback_depth
            prev_state = self._traceback_states[idx][state]
            inp = self._traceback_inputs[idx][state]

            if i == self.traceback_depth - 1:
                decoded_input = inp

            state = prev_state

        return decoded_input  # 2-bit decoded data

    def flush(self) -> list:
        """Flush remaining symbols from the traceback buffer.

        Returns decoded values for the remaining symbols in the buffer.
        """
        results = []
        # Force decode remaining by feeding dummy symbols
        for _ in range(self.traceback_depth):
            val = self.decode_symbol(0.0)
            if val >= 0:
                results.append(val)
        return results


class TrellisDecoder:
    """12-phase ATSC trellis decoder.

    Demultiplexes the 12 interleaved trellis phases, runs an independent
    Viterbi decoder for each phase, and reassembles the decoded bytes.

    Parameters
    ----------
    traceback_depth : int
        Viterbi traceback depth per phase. Default 32.
    """

    def __init__(self, traceback_depth: int = VITERBI_TRACEBACK_DEPTH):
        self.decoders = [
            SingleViterbi(traceback_depth) for _ in range(TRELLIS_PHASES)
        ]
        self.traceback_depth = traceback_depth

    def decode_segment(self, segment: np.ndarray) -> np.ndarray:
        """Decode one ATSC data segment (832 symbols) to bytes.

        Parameters
        ----------
        segment : np.ndarray
            832 equalized symbols (4 sync + 828 data).

        Returns
        -------
        np.ndarray
            Decoded bytes from this segment. Each Viterbi output is 2 bits;
            828 data symbols / 12 phases = 69 symbols per phase.
            69 symbols * 2 bits = 138 bits per phase.
            12 phases * 138 bits = 1656 bits = 207 bytes per segment
            (matches one RS codeword).
        """
        # Strip segment sync (first 4 symbols)
        data_symbols = segment[ATSC_SEGMENT_SYNC_LENGTH:]
        assert len(data_symbols) == ATSC_SEGMENT_DATA_LENGTH  # 828

        # Demux symbols to 12 phases (round-robin assignment)
        decoded_dibits = []
        for i, sym in enumerate(data_symbols):
            phase = i % TRELLIS_PHASES
            dibit = self.decoders[phase].decode_symbol(sym)
            if dibit >= 0:
                decoded_dibits.append(dibit)

        # Pack dibits (2-bit values) into bytes
        # 4 dibits = 1 byte (MSB first)
        bytes_out = []
        for i in range(0, len(decoded_dibits) - 3, 4):
            byte_val = (
                (decoded_dibits[i] << 6)
                | (decoded_dibits[i + 1] << 4)
                | (decoded_dibits[i + 2] << 2)
                | decoded_dibits[i + 3]
            )
            bytes_out.append(byte_val)

        return np.array(bytes_out, dtype=np.uint8)

    def reset(self):
        for dec in self.decoders:
            dec.reset()
