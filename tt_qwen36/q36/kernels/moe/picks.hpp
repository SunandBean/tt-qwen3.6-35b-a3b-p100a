// Build the control row (common.hpp layout) from the per-row picks of route_rows.cpp, identically on every
// core (entries numbered by first appearance, row-major). picks: [32 rows][32 words], 9 x (expert, weight bits).
#pragma once
#include <cstdint>

inline void build_ctl(volatile uint32_t* ctl, volatile uint32_t* picks, uint32_t T) {
    constexpr uint32_t CTL_SLOTS_ = 2, CTL_ROWS_ = 258;
    uint16_t entry[257];
    for (uint32_t e = 0; e < 257; e++) entry[e] = 0xFFFF;
    uint32_t count = 0;
    for (uint32_t r = 0; r < T; r++) {
        for (uint32_t i = 0; i < 10; i++) {
            const uint32_t e = i < 9 ? picks[r * 32 + 2 * i] : 0xFFFFFFFFu;
            if (e > 256) {
                ctl[CTL_ROWS_ + r * 20 + 2 * i] = 0xFFFFFFFFu;
                ctl[CTL_ROWS_ + r * 20 + 2 * i + 1] = 0;
                continue;
            }
            if (entry[e] == 0xFFFF) {
                entry[e] = count;
                ctl[CTL_SLOTS_ + count] = e;
                count++;
            }
            ctl[CTL_ROWS_ + r * 20 + 2 * i] = entry[e];
            ctl[CTL_ROWS_ + r * 20 + 2 * i + 1] = picks[r * 32 + 2 * i + 1];
        }
    }
    ctl[0] = count;
    ctl[1] = T;
}
