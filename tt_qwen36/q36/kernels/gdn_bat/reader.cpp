// Batched decode: GDN core for B users x 1 token (row b = user b), one value head per core.
// Reads: consts (tril, stril, ones, eye), q/k/v head tiles of conv [1,1,32,8192] (q/k head = h/2), ab [1,1,32,64] +
// alpha/beta one-hot tiles, z, norm weight, the B states S_b [B,32,128,128] (16 tiles each), and per user b a
// selector tile SEL_b (every row one-hot at column b) and a row mask RM_b (column 0 = 1 at row b).
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_q = 0, cb_k = 1, cb_v = 2, cb_S = 5, cb_c = 6, cb_z = 7, cb_nw = 8;
constexpr uint32_t cb_ab = 31, cb_oh = 32, cb_sel = 38, cb_rm = 39, cb_Sx = 47;

void kernel_main() {
    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr auto cv_a = TensorAccessorArgs<1>();
    constexpr auto s_a = TensorAccessorArgs<cv_a.next_compile_time_args_offset()>();
    constexpr auto c_a = TensorAccessorArgs<s_a.next_compile_time_args_offset()>();
    constexpr auto z_a = TensorAccessorArgs<c_a.next_compile_time_args_offset()>();
    constexpr auto n_a = TensorAccessorArgs<z_a.next_compile_time_args_offset()>();
    constexpr auto ab_a = TensorAccessorArgs<n_a.next_compile_time_args_offset()>();
    const uint32_t h = get_arg_val<uint32_t>(0);
    const uint32_t cv_addr = get_arg_val<uint32_t>(1), s_addr = get_arg_val<uint32_t>(2);
    const uint32_t c_addr = get_arg_val<uint32_t>(3), z_addr = get_arg_val<uint32_t>(4);
    const uint32_t n_addr = get_arg_val<uint32_t>(5), ab_addr = get_arg_val<uint32_t>(6);
    const uint32_t z_off = get_arg_val<uint32_t>(7), ab_off = get_arg_val<uint32_t>(8);
    const uint32_t tb = get_tile_size(cb_q);
    const auto cv_acc = TensorAccessor(cv_a, cv_addr, tb);
    const auto s_acc = TensorAccessor(s_a, s_addr, get_tile_size(cb_S));
    const auto c_acc = TensorAccessor(c_a, c_addr, get_tile_size(cb_c));
    const auto z_acc = TensorAccessor(z_a, z_addr, get_tile_size(cb_z));
    const auto n_acc = TensorAccessor(n_a, n_addr, get_tile_size(cb_nw));
    const auto ab_acc = TensorAccessor(ab_a, ab_addr, get_tile_size(cb_ab));
    Noc noc;
    auto read_into = [&](const auto& acc, uint32_t cb_id, uint32_t base, uint32_t n, uint32_t tsz) {
        CircularBuffer cb(cb_id);
        cb.reserve_back(n);
        for (uint32_t t = 0; t < n; t++) noc.async_read(acc, cb, tsz, {.page_id = base + t}, {.offset_bytes = t * tsz});
        noc.async_read_barrier();
        cb.push_back(n);
    };
    auto onehot = [&](uint32_t cb_id, uint32_t n, auto&& fill) {     // n generated fp32 tiles
        CircularBuffer cb(cb_id);
        cb.reserve_back(n);
        volatile uint32_t* p = reinterpret_cast<volatile uint32_t*>(cb.get_write_ptr());
        for (uint32_t q = 0; q < 1024 * n; q++) p[q] = 0;
        for (uint32_t t = 0; t < n; t++) fill(p + 1024 * t, t);
        cb.push_back(n);
    };
    // element (r, c) of an fp32 tile: face (r/16)*2 + c/16, offset (r%16)*16 + c%16
    auto at = [](uint32_t r, uint32_t c) { return ((r / 16) * 2 + c / 16) * 256 + (r % 16) * 16 + (c % 16); };
    const uint32_t hq = h / 2;
    read_into(c_acc, cb_c, 0, 4, get_tile_size(cb_c));
    read_into(cv_acc, cb_q, hq * 4, 4, tb);
    read_into(cv_acc, cb_k, 64 + hq * 4, 4, tb);
    read_into(ab_acc, cb_ab, ab_off, 2, get_tile_size(cb_ab));
    onehot(cb_oh, 2, [&](volatile uint32_t* p, uint32_t t) {       // tile 0: column h (alpha), tile 1: 32 + h (beta)
        p[at(t == 0 ? h % 32 : (32 + h) % 32, 0)] = 0x3f800000u;
    });
    read_into(cv_acc, cb_v, 128 + h * 4, 4, tb);
    for (uint32_t b = 0; b < B; b++) read_into(s_acc, cb_S, (b * 32 + h) * 16, 16, get_tile_size(cb_S));
    for (uint32_t b = 0; b < B; b++) read_into(s_acc, cb_Sx, (b * 32 + h) * 16, 16, get_tile_size(cb_Sx));
    read_into(z_acc, cb_z, z_off + h * 4, 4, get_tile_size(cb_z));
    read_into(n_acc, cb_nw, 0, 4, get_tile_size(cb_nw));
    onehot(cb_sel, B, [&](volatile uint32_t* p, uint32_t b) {      // SEL_b: every row one-hot at column b
        for (uint32_t r = 0; r < 32; r++) p[at(r, b)] = 0x3f800000u;
    });
    onehot(cb_rm, B, [&](volatile uint32_t* p, uint32_t b) {       // RM_b: column 0 = 1 at row b
        p[at(b, 0)] = 0x3f800000u;
    });
}
