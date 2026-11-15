// On-card MoE routing for Qwen3.6 (256 experts, top 8), one data-movement core, no host round trip.
// In:  router logits, fp32 TILE [1,1,32,256] (8 tiles; rows >= T are ignored).
// Out: the uint32 control row of tt_qwen36/q36/kernels/moe (S2 layout, see common.hpp there):
//      [0] count of distinct experts over the T rows, [1] = T, [2 + j] = expert id of entry j,
//      [258 + r*20 + 2i], [+1] = (entry j, fp32 bits of the weight) of row r's i-th pick (i < 8; 8, 9 unused).
// Weights = softmax over the 8 selected logits (= HF: softmax over all 256, top 8, renormalised).
// SHARED (compile-time arg 1): logit column 256 is the shared-expert gate; every row also gets entry "expert 256"
// (the shared expert, stored after the routed ones) with weight sigmoid(gate) as its 9th pick.
// The RISC-V data-movement cores have no FPU, so the softmax runs in fixed point, as in Lottolabs' topk_routing.cpp
// (Apache-2.0, routing_math.hpp; exp table and polynomial taken from there), here from fp32 logits: Q16 logit
// differences, exp as Q31, one 64-bit reciprocal, result rounded once to fp32 (rel. error ~2e-6).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t CTL_WORDS = 1024, CTL_SLOTS = 2, CTL_ROWS = 258, NE = 256, TOPK = 8;
constexpr uint32_t cb_lg = 0, cb_ctl = 1;

namespace {
// fp32 -> signed Q16 (|x| < 2^14)
inline int32_t f32_to_q16(uint32_t bits) {
    const int32_t e = (bits >> 23) & 0xff;
    if (e == 0) return 0;
    const int32_t m = 0x800000 | (bits & 0x7fffff);       // value = m * 2^(e - 150); Q16: m * 2^(e - 134)
    const int32_t sh = e - 134;
    int32_t mag;
    if (sh >= 0) mag = m << (sh > 6 ? 6 : sh);            // clamp: |x| < 2^14
    else mag = sh <= -24 ? 0 : (m >> -sh);
    return (bits & 0x80000000u) ? -mag : mag;
}
// exp(-d) in Q31 for d >= 0 in Q16 (table of 2^(-i/32) and a quadratic: Lottolabs routing_math.hpp, Apache-2.0)
inline uint32_t exp_neg_q31(uint32_t d) {
    static constexpr uint32_t pow2_neg[33] = {
        2147483648u, 2101467502u, 2056437387u, 2012372174u, 1969251188u, 1927054196u, 1885761398u,
        1845353420u, 1805811301u, 1767116489u, 1729250827u, 1692196547u, 1655936265u, 1620452965u,
        1585730000u, 1551751076u, 1518500250u, 1485961921u, 1454120821u, 1422962010u, 1392470869u,
        1362633090u, 1333434672u, 1304861917u, 1276901417u, 1249540052u, 1222764986u, 1196563654u,
        1170923762u, 1145833280u, 1121280436u, 1097253708u, 1073741824u};
    const uint64_t t = (static_cast<uint64_t>(d) * 1549082005u) >> 30;   // d * log2(e), Q16
    const uint64_t whole = t >> 16;
    if (whole >= 31) return 0;
    const uint32_t frac = static_cast<uint32_t>(t) & 0xffff;
    const uint32_t a = static_cast<uint32_t>((static_cast<uint64_t>(frac & 0x7ff) * 1488522236u) >> 16);
    const uint32_t a2_half = static_cast<uint32_t>((static_cast<uint64_t>(a) * a) >> 32);
    const uint32_t poly = 0x80000000u - a + a2_half;
    const uint32_t v = static_cast<uint32_t>((static_cast<uint64_t>(pow2_neg[frac >> 11]) * poly) >> 31);
    return v >> whole;
}
// unsigned Q62 (value < 2) -> fp32, round to nearest even
inline uint32_t q62_to_f32(uint64_t q) {
    if (q == 0) return 0;
    uint32_t top = 63;
    while (!((q >> top) & 1u)) --top;
    uint32_t e = top + 127 - 62;
    uint32_t m;
    if (top >= 24) {
        const uint32_t drop = top - 23;
        m = static_cast<uint32_t>(q >> drop);
        const uint64_t rest = q & ((uint64_t{1} << drop) - 1), half = uint64_t{1} << (drop - 1);
        if (rest > half || (rest == half && (m & 1u))) {
            if (++m == 0x1000000u) { m = 0x800000u; ++e; }
        }
    } else {
        m = static_cast<uint32_t>(q << (23 - top));
    }
    return (e << 23) | (m & 0x7fffffu);
}
inline uint32_t order_key(uint32_t b) { return (b & 0x80000000u) ? ~b : (b | 0x80000000u); }
}  // namespace

void kernel_main() {
    constexpr uint32_t T = get_compile_time_arg_val(0);
    constexpr uint32_t SHARED = get_compile_time_arg_val(1);
    constexpr uint32_t NT = NE / 32 + (SHARED ? 1 : 0);         // logit tiles
    constexpr auto l_a = TensorAccessorArgs<2>();
    constexpr auto c_a = TensorAccessorArgs<l_a.next_compile_time_args_offset()>();
    const uint32_t l_addr = get_arg_val<uint32_t>(0);
    const uint32_t c_addr = get_arg_val<uint32_t>(1);
    const uint32_t tb = get_tile_size(cb_lg);                 // fp32 tile, 4096 B
    const auto l_acc = TensorAccessor(l_a, l_addr, tb);
    const auto c_acc = TensorAccessor(c_a, c_addr, CTL_WORDS * 4);
    Noc noc;
    CircularBuffer lcb(cb_lg), ccb(cb_ctl);
    lcb.reserve_back(NT);
    for (uint32_t t = 0; t < NT; t++) noc.async_read(l_acc, lcb, tb, {.page_id = t}, {.offset_bytes = t * tb});
    ccb.reserve_back(1);
    volatile uint32_t* ctl = reinterpret_cast<volatile uint32_t*>(ccb.get_write_ptr());
    for (uint32_t i = 0; i < CTL_WORDS; i++) ctl[i] = 0;
    for (uint32_t r = 0; r < T; r++)
        for (uint32_t i = 0; i < 10; i++) ctl[CTL_ROWS + r * 20 + 2 * i] = 0xFFFFFFFFu;
    uint16_t entry[NE + 1];
    for (uint32_t e = 0; e <= NE; e++) entry[e] = 0xFFFF;
    noc.async_read_barrier();
    const uint32_t base = lcb.get_write_ptr();
    uint32_t count = 0;
    for (uint32_t r = 0; r < T; r++) {
        // row r of tile t: face (r/16)*2 + (c/16), 16 fp32 per face row
        const uint32_t roff = (r / 16) * 2048 + (r % 16) * 64;
        uint32_t bk[TOPK], bi[TOPK], bb[TOPK], n = 0, thr = 0;
        for (uint32_t t = 0; t < NE / 32; t++) {
            for (uint32_t half = 0; half < 2; half++) {
                volatile uint32_t* v = reinterpret_cast<volatile uint32_t*>(base + t * tb + roff + half * 1024);
                for (uint32_t c = 0; c < 16; c++) {
                    const uint32_t bits = v[c];
                    const uint32_t key = order_key(bits);
                    if (n == TOPK && key <= thr) continue;      // ties keep the lower expert id (as torch.topk)
                    uint32_t pos = n < TOPK ? n++ : TOPK - 1;
                    while (pos > 0 && bk[pos - 1] < key) {
                        bk[pos] = bk[pos - 1]; bi[pos] = bi[pos - 1]; bb[pos] = bb[pos - 1]; --pos;
                    }
                    bk[pos] = key; bi[pos] = t * 32 + half * 16 + c; bb[pos] = bits;
                    thr = bk[TOPK - 1];
                }
            }
        }
        const int32_t mx = f32_to_q16(bb[0]);
        uint32_t raw[TOPK];
        uint64_t tot = 0;
        for (uint32_t k = 0; k < TOPK; k++) {
            raw[k] = exp_neg_q31(static_cast<uint32_t>(mx - f32_to_q16(bb[k])));
            tot += raw[k];
        }
        const uint64_t rcp = (uint64_t{1} << 62) / tot;
        for (uint32_t k = 0; k < TOPK; k++) {
            const uint32_t e = bi[k];
            if (entry[e] == 0xFFFF) { entry[e] = count; ctl[CTL_SLOTS + count] = e; count++; }
            ctl[CTL_ROWS + r * 20 + 2 * k] = entry[e];
            ctl[CTL_ROWS + r * 20 + 2 * k + 1] = q62_to_f32(static_cast<uint64_t>(raw[k]) * rcp);
        }
        if constexpr (SHARED) {           // sigmoid(g) = 1/(1+e^-g); with e = exp(-|g|): g >= 0 -> 1/(1+e), else e/(1+e)
            const uint32_t gb = *reinterpret_cast<volatile uint32_t*>(base + (NE / 32) * tb + roff);   // column 256
            const int32_t g = f32_to_q16(gb);
            const uint32_t e = exp_neg_q31(static_cast<uint32_t>(g < 0 ? -g : g));
            const uint64_t inv = (uint64_t{1} << 62) / (uint64_t{0x80000000u} + e);
            const uint64_t num = g >= 0 ? uint64_t{0x80000000u} : e;
            if (entry[NE] == 0xFFFF) { entry[NE] = count; ctl[CTL_SLOTS + count] = NE; count++; }
            ctl[CTL_ROWS + r * 20 + 2 * TOPK] = entry[NE];
            ctl[CTL_ROWS + r * 20 + 2 * TOPK + 1] = q62_to_f32(num * inv);
        }
    }
    ctl[0] = count;
    ctl[1] = T;
    noc.async_write(use<CircularBuffer::AddrSelector::WRITE_PTR>(ccb), c_acc, CTL_WORDS * 4, {.offset_bytes = 0},
                    {.page_id = 0});
    noc.async_write_barrier();
}
