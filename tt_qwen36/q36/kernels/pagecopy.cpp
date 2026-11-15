// Copy a contiguous range of pages between two DRAM tensors (data movement only). Core i copies pages
// [start + i*per, start + min((i+1)*per, n)) from src page (s0 + k) to dst page (d0 + k). Used for KV rows < b of the
// prefix-cache / session slots (tile pages of one KV head are contiguous in [B,1,S,256] TILE layout).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t PAGE = get_compile_time_arg_val(0);
    constexpr auto s_a = TensorAccessorArgs<1>();
    constexpr auto d_a = TensorAccessorArgs<s_a.next_compile_time_args_offset()>();
    const uint32_t s_addr = get_arg_val<uint32_t>(0), d_addr = get_arg_val<uint32_t>(1);
    const uint32_t s0 = get_arg_val<uint32_t>(2), d0 = get_arg_val<uint32_t>(3);
    const uint32_t k0 = get_arg_val<uint32_t>(4), k1 = get_arg_val<uint32_t>(5);
    const auto sa = TensorAccessor(s_a, s_addr, PAGE);
    const auto da = TensorAccessor(d_a, d_addr, PAGE);
    Noc noc;
    CircularBuffer cb(0);
    cb.reserve_back(8);
    const uint32_t buf = cb.get_write_ptr();
    for (uint32_t k = k0; k < k1; k += 8) {
        const uint32_t n = (k1 - k) < 8 ? (k1 - k) : 8;
        for (uint32_t i = 0; i < n; i++) noc_async_read(get_noc_addr(s0 + k + i, sa), buf + i * PAGE, PAGE);
        noc_async_read_barrier();
        for (uint32_t i = 0; i < n; i++) noc_async_write(buf + i * PAGE, get_noc_addr(d0 + k + i, da), PAGE);
        noc_async_write_barrier();
    }
}
