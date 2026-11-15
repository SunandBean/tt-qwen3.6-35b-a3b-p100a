// BFP8 variant of the KV-cache row writer (tt_qwen36/kernels/kvw/writer.cpp). The cache is bfloat8_b
// (1088-byte tiles: 64 shared exponents, one per 16-value face row, then 1024 sign|7-bit codes; value = m/64 * 2^(e-127),
// layout checked against ttnn 2026-11-13). The new row arrives as bf16 and is packed here with integer ops only (no
// FPU on these cores): shared exponent = max exponent of the 16 values, code = round(|v| / 2^(E-127) * 64), as ttnn.
// Original description: KV-cache row writer (data movement only). Core (cache q, head b, column tile c) patches the T new rows of head b
// into the bf16 TILE cache [B, 1, S, W]: for each row r at position p = pos[2r] (pos: the int32 cur_pos row of the
// verify pass), read cache tile (b, p/32, c), overwrite row p%32 with src[r, b, 32c .. 32c+31], write it back.
// src is row-major bf16 [1, T, B, W] (page = one (r, b) row of W values). Up to 2 caches (K and V) per launch.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_t = 0, cb_row = 1, cb_pos = 2;
// KVBLK = cache in the standard paged layout [S/32, B, 32, W] (block-major); else [B, 1, S, W]
#ifdef KVBLK
#define PAGE_OF(trow) (((trow) * B + b) * WT + c)
#else
#define PAGE_OF(trow) ((b * ST + (trow)) * WT + c)
#endif

void kernel_main() {
    constexpr uint32_t T = get_compile_time_arg_val(0);
    constexpr uint32_t B = get_compile_time_arg_val(1);       // heads in the cache (dim 0)
    constexpr uint32_t WT = get_compile_time_arg_val(2);      // tiles per row (W / 32)
    constexpr uint32_t ST = get_compile_time_arg_val(3);      // tile rows per head (S / 32)
    constexpr uint32_t POS_STRIDE = get_compile_time_arg_val(4);  // pos index of row r = r * POS_STRIDE
    constexpr auto c0_a = TensorAccessorArgs<5>();
    constexpr auto s0_a = TensorAccessorArgs<c0_a.next_compile_time_args_offset()>();
    constexpr auto c1_a = TensorAccessorArgs<s0_a.next_compile_time_args_offset()>();
    constexpr auto s1_a = TensorAccessorArgs<c1_a.next_compile_time_args_offset()>();
    constexpr auto p_a = TensorAccessorArgs<s1_a.next_compile_time_args_offset()>();
    const uint32_t q = get_arg_val<uint32_t>(0);          // which cache (0 or 1)
    const uint32_t b = get_arg_val<uint32_t>(1);
    const uint32_t c = get_arg_val<uint32_t>(2);
    const uint32_t c0 = get_arg_val<uint32_t>(3), s0 = get_arg_val<uint32_t>(4);
    const uint32_t c1 = get_arg_val<uint32_t>(5), s1 = get_arg_val<uint32_t>(6);
    const uint32_t p_addr = get_arg_val<uint32_t>(7);
    const uint32_t p_bytes = get_arg_val<uint32_t>(8);
    const uint32_t tb = get_tile_size(cb_t);              // bf16 tile
    const uint32_t rowb = WT * 32 * 2;                    // one src row in bytes
    const auto ca0 = TensorAccessor(c0_a, c0, tb);
    const auto sa0 = TensorAccessor(s0_a, s0, rowb);
    const auto ca1 = TensorAccessor(c1_a, c1, tb);
    const auto sa1 = TensorAccessor(s1_a, s1, rowb);
    const auto pa = TensorAccessor(p_a, p_addr, p_bytes);
    Noc noc;
    CircularBuffer pcb(cb_pos), tcb(cb_t), rcb(cb_row);
    pcb.reserve_back(1);
    noc.async_read(pa, pcb, p_bytes, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    volatile int32_t* pos = reinterpret_cast<volatile int32_t*>(pcb.get_write_ptr());
    tcb.reserve_back(1);
    rcb.reserve_back(1);
    volatile uint8_t* tile = reinterpret_cast<volatile uint8_t*>(tcb.get_write_ptr());
    volatile uint16_t* row = reinterpret_cast<volatile uint16_t*>(rcb.get_write_ptr());
    int32_t cur = -1;
    for (uint32_t r = 0; r < T; r++) {
        const int32_t p = pos[r * POS_STRIDE];
        const int32_t tr = p / 32;
        const uint32_t page = PAGE_OF(tr);
        if (tr != cur) {
            if (cur >= 0) {                                // flush the previous tile
                const uint32_t prev = PAGE_OF(cur);
                if (q == 0) noc.async_write(use<CircularBuffer::AddrSelector::WRITE_PTR>(tcb), ca0, tb, {.offset_bytes = 0}, {.page_id = prev});
                else noc.async_write(use<CircularBuffer::AddrSelector::WRITE_PTR>(tcb), ca1, tb, {.offset_bytes = 0}, {.page_id = prev});
                noc.async_write_barrier();
            }
            if (q == 0) noc.async_read(ca0, tcb, tb, {.page_id = page}, {.offset_bytes = 0});
            else noc.async_read(ca1, tcb, tb, {.page_id = page}, {.offset_bytes = 0});
            noc.async_read_barrier();
            cur = tr;
        }
        if (q == 0) noc.async_read(sa0, rcb, rowb, {.page_id = r * B + b}, {.offset_bytes = 0});
        else noc.async_read(sa1, rcb, rowb, {.page_id = r * B + b}, {.offset_bytes = 0});
        noc.async_read_barrier();
        const uint32_t rr = p % 32;
        const uint32_t face = (rr / 16) * 2;
        for (uint32_t half = 0; half < 2; half++) {        // face (rr/16)*2 + half, face row rr%16
            const uint32_t fr = (face + half) * 16 + (rr % 16);
            volatile uint16_t* src = row + c * 32 + half * 16;
            uint32_t emax = 0;
            for (uint32_t i = 0; i < 16; i++) {
                const uint32_t e = (src[i] >> 7) & 0xFF;
                emax = e > emax ? e : emax;
            }
            tile[fr] = static_cast<uint8_t>(emax);
            for (uint32_t i = 0; i < 16; i++) {
                const uint32_t v = src[i];
                const uint32_t e = (v >> 7) & 0xFF;
                uint32_t q = 0;
                if (e != 0) {
                    const uint32_t m8 = 0x80 | (v & 0x7F);       // 1.mmmmmmm as an 8-bit integer
                    const uint32_t sh = emax - e + 1;              // code = m8 / 2^sh, rounded to nearest
                    q = sh >= 9 ? 0 : (m8 + (1u << (sh - 1))) >> sh;
                    q = q > 127 ? 127 : q;
                }
                tile[64 + fr * 16 + i] = static_cast<uint8_t>(q | ((v & 0x8000) ? 0x80 : 0));
            }
        }
    }
    if (cur >= 0) {
        const uint32_t prev = PAGE_OF(cur);
        if (q == 0) noc.async_write(use<CircularBuffer::AddrSelector::WRITE_PTR>(tcb), ca0, tb, {.offset_bytes = 0}, {.page_id = prev});
        else noc.async_write(use<CircularBuffer::AddrSelector::WRITE_PTR>(tcb), ca1, tb, {.offset_bytes = 0}, {.page_id = prev});
        noc.async_write_barrier();
    }
}
