// Batched decode GDN core: writes y (4 tiles of head h into [1,1,32,4096]) and the B updated states.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_y = 16, cb_out = 40;

void kernel_main() {
    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr auto y_a = TensorAccessorArgs<1>();
    constexpr auto s_a = TensorAccessorArgs<y_a.next_compile_time_args_offset()>();
    const uint32_t h = get_arg_val<uint32_t>(0);
    const uint32_t y_addr = get_arg_val<uint32_t>(1), s_addr = get_arg_val<uint32_t>(2);
    const auto y_acc = TensorAccessor(y_a, y_addr, get_tile_size(cb_y));
    const auto s_acc = TensorAccessor(s_a, s_addr, get_tile_size(cb_out));
    Noc noc;
    CircularBuffer ycb(cb_y), ocb(cb_out);
    for (uint32_t c = 0; c < 4; c++) {
        ycb.wait_front(1);
        noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(ycb), y_acc, get_tile_size(cb_y),
                        {.offset_bytes = 0}, {.page_id = h * 4 + c});
        noc.async_write_barrier();
        ycb.pop_front(1);
    }
    for (uint32_t b = 0; b < B; b++) {
        for (uint32_t t = 0; t < 16; t++) {
            ocb.wait_front(1);
            noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(ocb), s_acc, get_tile_size(cb_out),
                            {.offset_bytes = 0}, {.page_id = (b * 32 + h) * 16 + t});
            noc.async_write_barrier();
            ocb.pop_front(1);
        }
    }
}
