/**
 * @file agc_dc.h
 * @brief Combined AGC + DC Blocker for ATSC 8VSB receiver.
 *
 * Translates osmium/demod/dc_blocker.py and osmium/demod/agc.py into a
 * single pipelined HLS block. The DC blocker runs first to remove any
 * offset, then the AGC normalizes the output power.
 *
 * AXI4-Stream: complex sample in -> complex sample out
 * AXI-Lite: runtime-tunable parameters + gain readback
 *
 * Data rate: ~21.524 MSPS complex. At 250 MHz dsp_clk, II=1 is easily
 * achievable (~11.6 cycles/sample margin).
 */

#ifndef AGC_DC_H
#define AGC_DC_H

#include "../common/types.h"
#include <hls_stream.h>

/**
 * Top-level function synthesized to IP.
 *
 * @param in_stream     AXI4-Stream complex input (~21.524 MSPS)
 * @param out_stream    AXI4-Stream complex output (power-normalized)
 * @param dc_alpha      DC blocker tracking rate (Python default: 1e-4)
 * @param agc_alpha     AGC adaptation rate (Python default: 1e-4)
 * @param target_power  AGC target output power (Python default: 1.0)
 * @param current_gain  Readback: current AGC gain (for PS monitoring)
 */
void agc_dc_top(
    hls::stream<complex_sample_t>& in_stream,
    hls::stream<complex_sample_t>& out_stream,
    loop_coeff_t dc_alpha,
    loop_coeff_t agc_alpha,
    gain_t       target_power,
    gain_t&      current_gain
);

#endif /* AGC_DC_H */
