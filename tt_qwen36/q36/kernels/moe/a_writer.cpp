// Program A writer: H tile of work item (j, c) -> Hbuf page j*20 + c ([1, 256, 32, 640] bf16).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t FT = 16;   // Qwen3.6
constexpr uint32_t KT = 64, GU_NT = 32, CTL_SLOTS = 2;
#ifndef A_CHUNK
#define A_CHUNK 8
#endif
constexpr uint32_t CHUNK = A_CHUNK;
constexpr uint32_t cb_u = 2, cb_ctl = 4, cb_h = 16;

void kernel_main() {
    constexpr auto h_a = TensorAccessorArgs<0>();
    const uint32_t core = get_arg_val<uint32_t>(0);
    const uint32_t ncores = get_arg_val<uint32_t>(1);
    const uint32_t h_addr = get_arg_val<uint32_t>(2);
    const uint32_t tb_h = get_tile_size(cb_h);
    const auto h_acc = TensorAccessor(h_a, h_addr, tb_h);
    Noc noc;
    CircularBuffer ccb(cb_ctl);
    ccb.wait_front(1);
    volatile uint32_t* ctl = reinterpret_cast<volatile uint32_t*>(ccb.get_read_ptr());
    const uint32_t count = ctl[0];
    CircularBuffer hcb(cb_h);
#ifdef DUALNOC
    // this RISC (other NoC) streams the up-projection tiles while the reader streams the gate tiles
    constexpr auto gu_a = TensorAccessorArgs<h_a.next_compile_time_args_offset()>();
    const uint32_t gu_addr = get_arg_val<uint32_t>(3);
    CircularBuffer ucb(cb_u);
    const uint32_t tb_g = get_tile_size(cb_u);
    const auto gu_acc = TensorAccessor(gu_a, gu_addr, tb_g);
#endif
    for (uint32_t i = core; i < count * FT; i += ncores) {
#ifdef DUALNOC
        {
            const uint32_t j = i / FT, c = i % FT;
            const uint32_t base = ctl[CTL_SLOTS + j] * (KT * GU_NT);
            for (uint32_t k0 = 0; k0 < KT; k0 += CHUNK) {
                ucb.reserve_back(CHUNK);
                for (uint32_t t = 0; t < CHUNK; t++) {
                    noc.async_read(gu_acc, ucb, tb_g, {.page_id = base + (k0 + t) * GU_NT + c + FT},
                                   {.offset_bytes = t * tb_g});
                }
                noc.async_read_barrier();
                ucb.push_back(CHUNK);
            }
        }
#endif
        hcb.wait_front(1);
        auto src = use<CircularBuffer::AddrSelector::READ_PTR>(hcb);
        noc.async_write(src, h_acc, tb_h, {.offset_bytes = 0}, {.page_id = i});
        noc.async_write_barrier();
        hcb.pop_front(1);
    }
}
