/**
 * @file agc_dc.cpp
 * @brief Combined AGC + DC Blocker implementation.
 *
 * Translates:
 *   osmium/demod/dc_blocker.py - DCBlocker.process() lines 37-44
 *   osmium/demod/agc.py        - AGC.process() lines 42-53
 *
 * Algorithm (per sample):
 *   1. DC blocker: IIR moving average subtraction
 *      avg_i = (1-alpha)*avg_i + alpha*x_i
 *      avg_q = (1-alpha)*avg_q + alpha*x_q
 *      x_dc = x - avg
 *
 *   2. AGC: feedback gain control
 *      y = x_dc * gain
 *      power = y_i^2 + y_q^2
 *      error = target_power - power
 *      gain += agc_alpha * error
 *      gain = clamp(gain, GAIN_MIN, GAIN_MAX)
 */

#include "agc_dc.h"

/* Gain clamp limits (from agc.py lines 52-53) */
static const gain_t GAIN_MIN = 0.000001;  /* 1e-6 */
static const gain_t GAIN_MAX = 1000000.0; /* 1e6  */

void agc_dc_top(
    hls::stream<complex_sample_t>& in_stream,
    hls::stream<complex_sample_t>& out_stream,
    loop_coeff_t dc_alpha,
    loop_coeff_t agc_alpha,
    gain_t       target_power,
    gain_t&      current_gain
) {
#pragma HLS INTERFACE axis port=in_stream
#pragma HLS INTERFACE axis port=out_stream
#pragma HLS INTERFACE s_axilite port=dc_alpha      bundle=ctrl
#pragma HLS INTERFACE s_axilite port=agc_alpha     bundle=ctrl
#pragma HLS INTERFACE s_axilite port=target_power  bundle=ctrl
#pragma HLS INTERFACE s_axilite port=current_gain  bundle=ctrl
#pragma HLS INTERFACE s_axilite port=return        bundle=ctrl

    /* Persistent state across function calls (matches Python class vars) */
    static sample_t dc_avg_i = 0;   /* dc_blocker.py: self._avg (real part) */
    static sample_t dc_avg_q = 0;   /* dc_blocker.py: self._avg (imag part) */
    static gain_t   gain     = 1;   /* agc.py: self.gain */

    /* Precompute (1 - dc_alpha) to avoid subtraction in the loop */
    loop_coeff_t one_minus_dc_alpha = (loop_coeff_t)1.0 - dc_alpha;

    /* Free-running processing loop.
     * Terminates when the input stream is empty (non-blocking read).
     * In hardware, this loop runs continuously as long as data arrives. */
    agc_dc_loop:
    while (true) {
#pragma HLS PIPELINE II=1
        /* II=1: one complex sample per clock cycle.
         * At 250 MHz this supports up to 250 MSPS -- 11.6x margin
         * over the 21.5 MSPS requirement. The multiply-accumulate
         * operations map to DSP48E2 slices (ap_fixed<18,4> * ap_fixed<32,16>). */

        complex_sample_t in_sample;
        if (!in_stream.read_nb(in_sample))
            break;

        /* ---- DC Blocker (dc_blocker.py lines 40-42) ---- */
        /* IIR: avg = (1-alpha)*avg + alpha*x */
        dc_avg_i = one_minus_dc_alpha * dc_avg_i + dc_alpha * in_sample.i;
        dc_avg_q = one_minus_dc_alpha * dc_avg_q + dc_alpha * in_sample.q;

        sample_t dc_out_i = in_sample.i - dc_avg_i;
        sample_t dc_out_q = in_sample.q - dc_avg_q;

        /* ---- AGC (agc.py lines 46-53) ---- */
        /* Apply gain */
        sample_t agc_out_i = dc_out_i * gain;
        sample_t agc_out_q = dc_out_q * gain;

        /* Compute instantaneous power: |y|^2 = i^2 + q^2 */
        /* Replaces: np.real(out[i] * np.conj(out[i])) */
        gain_t power = (gain_t)(agc_out_i * agc_out_i)
                     + (gain_t)(agc_out_q * agc_out_q);

        /* Error and gain update */
        gain_t error = target_power - power;
        gain += agc_alpha * error;

        /* Clamp gain (agc.py lines 52-53) */
        if (gain < GAIN_MIN)
            gain = GAIN_MIN;
        if (gain > GAIN_MAX)
            gain = GAIN_MAX;

        /* Output normalized sample */
        complex_sample_t out_sample;
        out_sample.i = agc_out_i;
        out_sample.q = agc_out_q;
        out_stream.write(out_sample);
    }

    /* Export current gain for PS monitoring via AXI-Lite readback
     * (matches demod_chain.py status['agc_gain']) */
    current_gain = gain;
}
