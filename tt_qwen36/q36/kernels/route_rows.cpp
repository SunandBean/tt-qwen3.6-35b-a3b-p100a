// MoE routing, one row per core (data movement only; the RISC-V cores have no FPU, so fixed point as in
// route.cpp / Lottolabs' routing_math.hpp, Apache-2.0).
// In:  router logits fp32 TILE [1,1,32,256(+32)] (column 256 = shared-expert gate when SHARED).
// Out: picks uint32 row-major [1,1,32,32]: row r = 9 x (expert id, fp32 weight bits): the 8 routed picks (top 8,
//      softmax over the 8) then the shared expert (id 256, weight sigmoid(gate)) when SHARED; unused = 0xFFFFFFFF.
// The MoE readers (kernels/moe, PICKS mode) build their control rows from these picks, identically on every core.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t NE = 256, TOPK = 8;
constexpr uint32_t cb_row = 0, cb_out = 1;

namespace {
inline int32_t f32_to_q16(uint32_t bits) {
    const int32_t e = (bits >> 23) & 0xff;
    if (e == 0) return 0;
    const int32_t m = 0x800000 | (bits & 0x7fffff);
    const int32_t sh = e - 134;
    int32_t mag;
    if (sh >= 0) mag = m << (sh > 6 ? 6 : sh);
    else mag = sh <= -24 ? 0 : (m >> -sh);
    return (bits & 0x80000000u) ? -mag : mag;
}
inline uint32_t exp_neg_q31(uint32_t d) {
    static constexpr uint32_t pow2_neg[33] = {
        2147483648u, 2101467502u, 2056437387u, 2012372174u, 1969251188u, 1927054196u, 1885761398u,
        1845353420u, 1805811301u, 1767116489u, 1729250827u, 1692196547u, 1655936265u, 1620452965u,
        1585730000u, 1551751076u, 1518500250u, 1485961921u, 1454120821u, 1422962010u, 1392470869u,
        1362633090u, 1333434672u, 1304861917u, 1276901417u, 1249540052u, 1222764986u, 1196563654u,
        1170923762u, 1145833280u, 1121280436u, 1097253708u, 1073741824u};
    const uint64_t t = (static_cast<uint64_t>(d) * 1549082005u) >> 30;
    const uint64_t whole = t >> 16;
    if (whole >= 31) return 0;
    const uint32_t frac = static_cast<uint32_t>(t) & 0xffff;
    const uint32_t a = static_cast<uint32_t>((static_cast<uint64_t>(frac & 0x7ff) * 1488522236u) >> 16);
    const uint32_t a2_half = static_cast<uint32_t>((static_cast<uint64_t>(a) * a) >> 32);
    const uint32_t poly = 0x80000000u - a + a2_half;
    const uint32_t v = static_cast<uint32_t>((static_cast<uint64_t>(pow2_neg[frac >> 11]) * poly) >> 31);
    return v >> whole;
}
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
    constexpr uint32_t SHARED = get_compile_time_arg_val(0);
    constexpr uint32_t NT = NE / 32 + (SHARED ? 1 : 0);
    constexpr auto l_a = TensorAccessorArgs<1>();
    constexpr auto p_a = TensorAccessorArgs<l_a.next_compile_time_args_offset()>();
    constexpr auto n_a = TensorAccessorArgs<p_a.next_compile_time_args_offset()>();
    const uint32_t r = get_arg_val<uint32_t>(0);
    const uint32_t l_addr = get_arg_val<uint32_t>(1);
    const uint32_t p_addr = get_arg_val<uint32_t>(2);
    const uint32_t n_addr = get_arg_val<uint32_t>(3);         // uint32 [8]: word 0 = valid rows of this pass
    const uint32_t tb = 4096;                                     // fp32 tile
    const auto l_acc = TensorAccessor(l_a, l_addr, tb);
    const auto p_acc = TensorAccessor(p_a, p_addr, 128);
    Noc noc;
    CircularBuffer rcb(cb_row), ocb(cb_out);
    rcb.reserve_back(1);
    ocb.reserve_back(1);
    // row r of every logit tile: two 64-byte face-row segments (columns 0..15 and 16..31)
    const uint32_t roff = (r / 16) * 2048 + (r % 16) * 64;
    const uint32_t buf = rcb.get_write_ptr();
    for (uint32_t t = 0; t < NT; t++) {
        const uint64_t src = get_noc_addr(t, l_acc);
        noc_async_read(src + roff, buf + t * 128, 64);
        noc_async_read(src + roff + 1024, buf + t * 128 + 64, 64);
    }
    volatile uint32_t* out = reinterpret_cast<volatile uint32_t*>(ocb.get_write_ptr());
    for (uint32_t i = 0; i < 32; i++) out[i] = (i % 2 == 0) ? 0xFFFFFFFFu : 0u;
    const auto n_acc = TensorAccessor(n_a, n_addr, 32);
    noc_async_read(get_noc_addr(0, n_acc), buf + 9 * 128, 32);
    noc_async_read_barrier();
    if (r >= reinterpret_cast<volatile uint32_t*>(buf + 9 * 128)[0]) {   // padding row: no experts at all
        noc_async_write(ocb.get_write_ptr(), get_noc_addr(r, p_acc), 128);
        noc_async_write_barrier();
        return;
    }
    volatile uint32_t* v = reinterpret_cast<volatile uint32_t*>(buf);      // 32 values per tile, in column order
    uint32_t bk[TOPK], bi[TOPK], bb[TOPK], n = 0, thr = 0;
    for (uint32_t e = 0; e < NE; e++) {
        const uint32_t bits = v[e];
        const uint32_t key = order_key(bits);
        if (n == TOPK && key <= thr) continue;                   // ties keep the lower expert id
        uint32_t pos = n < TOPK ? n++ : TOPK - 1;
        while (pos > 0 && bk[pos - 1] < key) {
            bk[pos] = bk[pos - 1]; bi[pos] = bi[pos - 1]; bb[pos] = bb[pos - 1]; --pos;
        }
        bk[pos] = key; bi[pos] = e; bb[pos] = bits;
        thr = bk[TOPK - 1];
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
        out[2 * k] = bi[k];
        out[2 * k + 1] = q62_to_f32(static_cast<uint64_t>(raw[k]) * rcp);
    }
    if constexpr (SHARED) {
        const int32_t g = f32_to_q16(v[NE]);
        const uint32_t e = exp_neg_q31(static_cast<uint32_t>(g < 0 ? -g : g));
        const uint64_t inv = (uint64_t{1} << 62) / (uint64_t{0x80000000u} + e);
        const uint64_t num = g >= 0 ? uint64_t{0x80000000u} : e;
        out[2 * TOPK] = NE;
        out[2 * TOPK + 1] = q62_to_f32(num * inv);
    }
    noc_async_write(ocb.get_write_ptr(), get_noc_addr(r, p_acc), 128);
    noc_async_write_barrier();
}
