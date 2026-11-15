"""MoE for Qwen3.6 entirely on the card, three tt-metal programs per layer and pass, no host round trip:
  route (tt_qwen36/q36/kernels/route.cpp): fp32 router logits [32, 256] -> top 8 per row, softmax weights, the
        distinct experts of the pass -> the uint32 control row (one data-movement core);
  A, B  (tt_qwen36/q36/kernels/moe/, the kernels of tt_qwen36/kmoe.py with H = 2048, F = 512): gate/up and down
        projections of the ACTIVE experts only (all 256 experts of every layer are resident in BFP4).
Weights in the layout of the Lottolabs checkpoint: gu [1, 256, 2048, 1024] (gate | up), dn [1, 256, 512, 2048].
"""
from __future__ import annotations

import os

import torch
import ttnn

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")
CTL_WORDS, CTL_SLOTS, CTL_ROWS, MAX_ACTIVE = 1024, 2, 258, 256
H, F, NE = 2048, 512, 256
PAGE = {ttnn.bfloat16: 2048, ttnn.float32: 4096, ttnn.bfloat4_b: 576, ttnn.bfloat8_b: 1088, ttnn.uint32: 4096}


_HBUF = {}


def _cb(idx, dt, n, grid, page=None):
    page = page or PAGE[dt]
    return ttnn.CBDescriptor(total_size=page * n, core_ranges=grid,
                             format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dt,
                                                                         page_size=page)])


class Q36Moe:
    def __init__(self, dev, nc_a: int = 64, nq: int = 2, fidelity=ttnn.MathFidelity.HiFi2, chunk: int = 8, kc: int = 4,
                 hdt=None, picks_T: int = 0, dualnoc=None, hl1=None):
        self.dev = dev
        self.picks_T = picks_T     # > 0: route one row per core (route_rows.cpp); the MoE readers build ctl from picks
        self.hdt = hdt or (ttnn.bfloat8_b if os.environ.get("P100A_Q36_HBFP8", "0") == "1" else ttnn.bfloat16)
        self.chunk, self.kc = chunk, kc            # weight tiles per reader push (A), H tiles per push (B)
        self.defs_a, self.defs_b = [("A_CHUNK", str(chunk))], [("B_KC", str(kc))]
        for d in os.environ.get("P100A_Q36_MOE_DBG", "").split(","):     # timing experiments only (wrong results)
            if d:
                self.defs_a.append((d, "1"))
        if dualnoc if dualnoc is not None else os.environ.get("P100A_Q36_DUALNOC", "1") == "1":
            self.defs_a.append(("DUALNOC", "1"))
        if picks_T:
            self.defs_a.append(("PICKS", str(picks_T)))
            self.defs_b.append(("PICKS", str(picks_T)))
            self.cores_r = [ttnn.CoreCoord(10 - (i % 4), 9 - i // 4) for i in range(picks_T)]   # top-right corner
            self.grid_rr = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in self.cores_r])
        g = dev.compute_with_storage_grid_size()
        self.nc_a = min(nc_a, g.x * g.y)
        self.nq = nq
        self.nc_b = (H // 32) // nq
        self.cores_a = [ttnn.CoreCoord(i % g.x, i // g.x) for i in range(self.nc_a)]
        self.cores_b = [ttnn.CoreCoord(i % g.x, i // g.x) for i in range(self.nc_b)]
        self.grid_a = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in self.cores_a])
        self.grid_b = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in self.cores_b])
        last = ttnn.CoreCoord(g.x - 1, g.y - 1)               # routing core: away from the busy low cores
        self.grid_r = ttnn.CoreRangeSet([ttnn.CoreRange(last, last)])
        self.core_r = last
        self.fid = fidelity
        # H tiles in L1 (interleaved over the cores' SRAM, <= 8 MB): program B's re-reads stay off DRAM
        self.hl1 = hl1 if hl1 is not None else os.environ.get("P100A_Q36_HL1", "1") == "1"
        key = (id(dev), self.hl1, str(self.hdt))       # one H buffer per device (the passes run one after another)
        if key not in _HBUF:
            _HBUF[key] = ttnn.from_torch(torch.zeros(1, MAX_ACTIVE, 32, F), dtype=self.hdt, layout=ttnn.TILE_LAYOUT,
                                         device=dev, memory_config=ttnn.L1_MEMORY_CONFIG if self.hl1 else
                                         ttnn.DRAM_MEMORY_CONFIG)
        self.hbuf = _HBUF[key]

    def new_ctl(self):
        """the control row (route) or, in picks mode, the per-row picks [1,1,32,32] (route_rows)"""
        if self.picks_T:
            return ttnn.from_torch(torch.full((1, 1, 32, 32), -1, dtype=torch.int32), dtype=ttnn.uint32,
                                   layout=ttnn.ROW_MAJOR_LAYOUT, device=self.dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.from_torch(torch.zeros(1, 1, 1, CTL_WORDS, dtype=torch.int32), dtype=ttnn.uint32,
                               layout=ttnn.ROW_MAJOR_LAYOUT, device=self.dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def new_nvalid(self):
        """uint32 [1,1,1,8]: word 0 = rows of the pass that are real tokens (the host writes it per pass)"""
        return ttnn.from_torch(torch.full((1, 1, 1, 8), self.picks_T, dtype=torch.int32), dtype=ttnn.uint32,
                               layout=ttnn.ROW_MAJOR_LAYOUT, device=self.dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def route_rows(self, logits, picks, nvalid):
        T = self.picks_T
        shared = int(logits.shape[-1] > NE)
        ct = [shared] + ttnn.TensorAccessorArgs(logits).get_compile_time_args() + \
            ttnn.TensorAccessorArgs(picks).get_compile_time_args() + ttnn.TensorAccessorArgs(nvalid).get_compile_time_args()
        rt = ttnn.RuntimeArgs()
        for r, c in enumerate(self.cores_r):
            rt[c.x][c.y] = [r, logits.buffer_address(), picks.buffer_address(), nvalid.buffer_address()]
        g = self.grid_rr
        cbs = [_cb(0, ttnn.float32, 1, g, page=128 * (NE // 32 + shared) + 128), _cb(1, ttnn.uint32, 1, g, page=128)]
        k = [ttnn.KernelDescriptor(kernel_source=f"{KDIR}/route_rows.cpp", core_ranges=g, compile_time_args=ct,
                                   runtime_args=rt, config=ttnn.ReaderConfigDescriptor())]
        return ttnn.generic_op([logits, picks, nvalid], ttnn.ProgramDescriptor(kernels=k, semaphores=[], cbs=cbs))

    # ---- routing: logits fp32 TILE [1,1,32,256] -> ctl
    def route(self, logits, ctl, T: int):
        """logits [1,1,32,256] or [1,1,32,288] (column 256 = shared-expert gate logit: shared expert = entry 256)"""
        ins, outs = [logits], [ctl]
        shared = int(logits.shape[-1] > NE)
        ct = [T, shared] + ttnn.TensorAccessorArgs(logits).get_compile_time_args() + \
            ttnn.TensorAccessorArgs(ctl).get_compile_time_args()
        rt = ttnn.RuntimeArgs()
        c = self.core_r
        rt[c.x][c.y] = [logits.buffer_address(), ctl.buffer_address()]
        gr = self.grid_r
        cbs = [_cb(0, ttnn.float32, NE // 32 + shared, gr), _cb(1, ttnn.uint32, 1, gr)]
        k = [ttnn.KernelDescriptor(kernel_source=f"{KDIR}/route.cpp", core_ranges=gr, compile_time_args=ct,
                                   runtime_args=rt, config=ttnn.ReaderConfigDescriptor())]
        return ttnn.generic_op(ins + outs, ttnn.ProgramDescriptor(kernels=k, semaphores=[], cbs=cbs))

    def _ctl_page(self):
        return 4096 + (32 * 128 if self.picks_T else 0)

    # ---- experts: x bf16 TILE [1,1,32,2048] (rows >= T zero), ctl from route(), y fp32 TILE [1,1,32,2048]
    def experts(self, x, gu, dn, ctl, y, parts: str = "AB"):
        D = f"{KDIR}/moe"
        ins, outs = [x, gu, ctl], [self.hbuf]
        r_ct = []
        for t in ins:
            r_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        w_ct = ttnn.TensorAccessorArgs(self.hbuf).get_compile_time_args() + \
            ttnn.TensorAccessorArgs(gu).get_compile_time_args()
        r_rt, w_rt, c_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for i, c in enumerate(self.cores_a):
            r_rt[c.x][c.y] = [i, self.nc_a, x.buffer_address(), gu.buffer_address(), ctl.buffer_address()]
            w_rt[c.x][c.y] = [i, self.nc_a, self.hbuf.buffer_address(), gu.buffer_address()]
            c_rt[c.x][c.y] = [i, self.nc_a]
        ga = self.grid_a
        cbs = [_cb(0, ttnn.bfloat16, H // 32, ga), _cb(1, gu.dtype, 2 * self.chunk, ga), _cb(2, gu.dtype, 2 * self.chunk, ga),
               _cb(3, ttnn.float32, 2, ga), _cb(4, ttnn.uint32, 1, ga, page=self._ctl_page()), _cb(5, ttnn.float32, 1, ga),
               _cb(6, ttnn.float32, 1, ga), _cb(7, ttnn.float32, 1, ga), _cb(8, ttnn.float32, 1, ga),
               _cb(16, self.hdt, 2, ga)]
        ka = [ttnn.KernelDescriptor(kernel_source=f"{D}/a_reader.cpp", core_ranges=ga, compile_time_args=r_ct,
                                    runtime_args=r_rt, config=ttnn.ReaderConfigDescriptor(), defines=self.defs_a),
              ttnn.KernelDescriptor(kernel_source=f"{D}/a_writer.cpp", core_ranges=ga, compile_time_args=w_ct,
                                    runtime_args=w_rt, config=ttnn.WriterConfigDescriptor(), defines=self.defs_a),
              ttnn.KernelDescriptor(kernel_source=f"{D}/a_compute.cpp", core_ranges=ga, compile_time_args=[],
                                    runtime_args=c_rt, defines=self.defs_a,
                                    config=ttnn.ComputeConfigDescriptor(math_fidelity=self.fid, fp32_dest_acc_en=True))]
        if "A" in parts:
            ttnn.generic_op(ins + outs, ttnn.ProgramDescriptor(kernels=ka, semaphores=[], cbs=cbs))
        if "B" not in parts:
            return y
        nq = self.nq
        ins, outs = [self.hbuf, dn, ctl], [y]
        r_ct = [nq]
        for t in ins:
            r_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        w_ct = [nq] + ttnn.TensorAccessorArgs(y).get_compile_time_args()
        r_rt, w_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for i, c in enumerate(self.cores_b):
            r_rt[c.x][c.y] = [i * nq, self.hbuf.buffer_address(), dn.buffer_address(), ctl.buffer_address()]
            w_rt[c.x][c.y] = [i * nq, y.buffer_address()]
        gb = self.grid_b
        cbs = [_cb(0, self.hdt, 2 * self.kc, gb), _cb(1, dn.dtype, 2 * self.kc * nq, gb),
               _cb(4, ttnn.uint32, 1, gb, page=self._ctl_page()),
               _cb(16, y.dtype, nq, gb)]
        kb = [ttnn.KernelDescriptor(kernel_source=f"{D}/b_reader.cpp", core_ranges=gb, compile_time_args=r_ct,
                                    runtime_args=r_rt, config=ttnn.ReaderConfigDescriptor(), defines=self.defs_b),
              ttnn.KernelDescriptor(kernel_source=f"{D}/b_writer.cpp", core_ranges=gb, compile_time_args=w_ct,
                                    runtime_args=w_rt, config=ttnn.WriterConfigDescriptor()),
              ttnn.KernelDescriptor(kernel_source=f"{D}/b_compute.cpp", core_ranges=gb, compile_time_args=[nq],
                                    runtime_args=[], defines=self.defs_b,
                                    config=ttnn.ComputeConfigDescriptor(math_fidelity=self.fid, fp32_dest_acc_en=True))]
        return ttnn.generic_op(ins + outs, ttnn.ProgramDescriptor(kernels=kb, semaphores=[], cbs=cbs))


def ctl_decode(ctl_words, T):
    """host check helper: ctl (uint32 list) -> per row [(expert, weight)]"""
    import struct
    c = list(ctl_words)
    out = []
    for r in range(T):
        row = []
        for i in range(10):
            j = c[CTL_ROWS + r * 20 + 2 * i]
            if j == 0xFFFFFFFF:
                continue
            w = struct.unpack("<f", struct.pack("<I", c[CTL_ROWS + r * 20 + 2 * i + 1]))[0]
            row.append((c[CTL_SLOTS + j], w))
        out.append(row)
    return out


class KvWrite8:
    """BFP8 KV-cache row writer (kernels/kvw8/writer.cpp): tt_qwen36/kkv.KvWrite for bfloat8_b caches [B,1,S,W]
    (blk=True: [S/32,B,32,W]). KvWriteBlk (below): the bf16 writer for the block layout."""
    KERNEL, PAGE, DT = "kvw8/writer.cpp", 1088, ttnn.bfloat8_b

    def __init__(self, dev, T: int, B: int, W: int, S: int, ncaches: int = 2, blk: bool = False):
        from ..kkv import KvWrite
        self.base = KvWrite(dev, T, B, W, S, ncaches)
        self.defines = [("KVBLK", "1")] if blk else []

    def __call__(self, caches: list, srcs: list, cur):
        k = self.base
        c0, s0 = caches[0], srcs[0]
        c1, s1 = (caches[1], srcs[1]) if k.nc > 1 else (c0, s0)
        ins = [c0, s0, c1, s1, cur]
        ct = [k.T, k.B, k.WT, k.ST, 2]
        for t in ins:
            ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        pbytes = 8 * k.T
        rt = ttnn.RuntimeArgs()
        for (q, b, c), cc in zip(k.work, k.cores):
            rt[cc.x][cc.y] = [q, b, c, c0.buffer_address(), s0.buffer_address(), c1.buffer_address(),
                              s1.buffer_address(), cur.buffer_address(), pbytes]
        rowb = k.WT * 64
        cbs = [ttnn.CBDescriptor(total_size=self.PAGE, core_ranges=k.grid, format_descriptors=[
                   ttnn.CBFormatDescriptor(buffer_index=0, data_format=self.DT, page_size=self.PAGE)]),
               ttnn.CBDescriptor(total_size=rowb, core_ranges=k.grid, format_descriptors=[
                   ttnn.CBFormatDescriptor(buffer_index=1, data_format=ttnn.bfloat16, page_size=rowb)]),
               ttnn.CBDescriptor(total_size=max(64, pbytes), core_ranges=k.grid, format_descriptors=[
                   ttnn.CBFormatDescriptor(buffer_index=2, data_format=ttnn.int32, page_size=max(64, pbytes))])]
        kd = ttnn.KernelDescriptor(kernel_source=f"{KDIR}/{self.KERNEL}", core_ranges=k.grid, compile_time_args=ct,
                                   runtime_args=rt, config=ttnn.ReaderConfigDescriptor(), defines=self.defines)
        return ttnn.generic_op(ins, ttnn.ProgramDescriptor(kernels=[kd], semaphores=[], cbs=cbs))


class KvWriteBlk(KvWrite8):
    """bf16 KV writer for the block layout [S/32, B, 32, W] (kernels/kvw/writer.cpp with KVBLK)"""
    KERNEL, PAGE, DT = "kvw/writer.cpp", 2048, ttnn.bfloat16

    def __init__(self, dev, T: int, B: int, W: int, S: int, ncaches: int = 2):
        super().__init__(dev, T, B, W, S, ncaches, blk=True)


class PageCopy:
    """copy KV rows [0, rows) of each head between [B,1,S,W] TILE tensors of different S (kernels/pagecopy.cpp)"""

    def __init__(self, dev, ncores: int = 32):
        g = dev.compute_with_storage_grid_size()
        self.cores = [ttnn.CoreCoord(i % g.x, i // g.x) for i in range(ncores)]
        self.grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in self.cores])

    def raw(self, src, dst, s0: int, d0: int, n: int):
        """pages [s0, s0+n) of src -> [d0, d0+n) of dst"""
        page = PAGE[src.dtype]
        per = -(-n // len(self.cores))
        rt = ttnn.RuntimeArgs()
        for i, c in enumerate(self.cores):
            rt[c.x][c.y] = [src.buffer_address(), dst.buffer_address(), s0, d0, min(n, i * per), min(n, (i + 1) * per)]
        ct = [page] + ttnn.TensorAccessorArgs(src).get_compile_time_args() + \
            ttnn.TensorAccessorArgs(dst).get_compile_time_args()
        cbs = [ttnn.CBDescriptor(total_size=8 * page, core_ranges=self.grid, format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=0, data_format=src.dtype, page_size=page)])]
        k = ttnn.KernelDescriptor(kernel_source=f"{KDIR}/pagecopy.cpp", core_ranges=self.grid,
                                  compile_time_args=ct, runtime_args=rt, config=ttnn.ReaderConfigDescriptor())
        ttnn.generic_op([src, dst], ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=cbs))

    def __call__(self, src, dst, rows: int):
        if src.shape[2] == 32 and src.shape[1] == 2 and len(src.shape) == 4 and src.shape[0] != 2:
            # block layout [S/32, 2, 32, W]: rows < `rows` = the first ceil(rows/32) blocks, contiguous pages
            n = -(-rows // 32) * 2 * (src.shape[3] // 32)
            return self.raw(src, dst, 0, 0, n)
        B, S_s, S_d, W = src.shape[0], src.shape[2], dst.shape[2], src.shape[3]
        wt = W // 32
        page = PAGE[src.dtype]
        n = -(-rows // 32) * wt                                  # pages per head
        per = -(-n // len(self.cores))
        for b in range(B):
            rt = ttnn.RuntimeArgs()
            for i, c in enumerate(self.cores):
                k0, k1 = min(n, i * per), min(n, (i + 1) * per)
                rt[c.x][c.y] = [src.buffer_address(), dst.buffer_address(), b * (S_s // 32) * wt,
                                b * (S_d // 32) * wt, k0, k1]
            ct = [page] + ttnn.TensorAccessorArgs(src).get_compile_time_args() + \
                ttnn.TensorAccessorArgs(dst).get_compile_time_args()
            cbs = [ttnn.CBDescriptor(total_size=8 * page, core_ranges=self.grid, format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=0, data_format=src.dtype, page_size=page)])]
            k = ttnn.KernelDescriptor(kernel_source=f"{KDIR}/pagecopy.cpp", core_ranges=self.grid,
                                      compile_time_args=ct, runtime_args=rt, config=ttnn.ReaderConfigDescriptor())
            ttnn.generic_op([src, dst], ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=cbs))
