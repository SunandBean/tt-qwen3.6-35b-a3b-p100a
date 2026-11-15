"""The step-4 fused GDN programs (tt_qwen36/kgdn.py) at Qwen3.6 shapes: 32 value heads (one per core), q/k
head = h / 2 (repeat_interleave), conv over 8192 channels (256 tiles), ab = [alpha 32 | beta 32], silu output gate.
Core kernels: tt_qwen36/q36/kernels/gdn_core (a copy with those changes); conv and commit reuse tt_qwen36/kernels/gdn_conv
and gdn_commit unchanged (they are generic in the channel / head count).
"""
from __future__ import annotations

import os

import torch
import ttnn

from .. import kgdn as base

NH = 32
KDIR36 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels", "gdn_core")


class GdnCore36(base.GdnCore):
    def __init__(self, dev, T: int, eps: float = 1e-6, eps_l2: float = 1e-6):
        super().__init__(dev, T, eps, eps_l2)
        g = dev.compute_with_storage_grid_size()
        self.cores = [ttnn.CoreCoord(h % g.x, h // g.x) for h in range(NH)]
        self.grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in self.cores])

    def __call__(self, conv, S0, z, nw, y, kn, U, ab, dt, a, gout, z_off: int = 0, ab_off: int = 0):
        """ABG mode only: conv [1,1,32,8192], S0 [1,32,128,128], z [1,1,32,4096], nw [1,1,1|32,128], ab [1,1,32,64]
        (fp32 TILE); dt / a: 32 floats; outputs y [1,1,32,4096], kn / U [1,32,32,128], gout [1,32,32,32]."""
        dummy = self.consts
        ins = [conv, dummy, dummy, S0, self.consts, z, nw, ab]
        outs = [y, kn, U, gout]
        fmt = {0: conv.dtype, 1: conv.dtype, 2: conv.dtype, 5: S0.dtype, 7: z.dtype, 8: nw.dtype,
               16: y.dtype, 17: kn.dtype, 18: U.dtype}
        r_ct, w_ct = [], []
        for t in ins:
            r_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        r_ct.append(1)
        for t in outs:
            w_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        w_ct.append(1)
        r_rt, w_rt, c_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for h, c in enumerate(self.cores):
            r_rt[c.x][c.y] = [h] + [t.buffer_address() for t in ins] + [z_off, ab_off]
            w_rt[c.x][c.y] = [h] + [t.buffer_address() for t in outs]
            c_rt[c.x][c.y] = [h // 32, (NH + h) // 32, base.f2u(dt[h]), base.f2u(a[h])]
        cbs = self._cbs(fmt)
        extra = {30: (ttnn.bfloat16, 1), 31: (ttnn.float32, 2), 32: (ttnn.float32, 2), 33: (ttnn.float32, 1),
                 35: (ttnn.float32, 2), 36: (ttnn.float32, 1), 37: (ttnn.float32, 1)}
        for idx, (dtp, n) in extra.items():
            page = 4096 if dtp == ttnn.float32 else 2048
            cbs.append(ttnn.CBDescriptor(total_size=page * n, core_ranges=self.grid, format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dtp, page_size=page)]))
        ccfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
        modes = [ttnn.UnpackToDestMode.Default] * 64
        for i in (31, 35, 36, 37):
            modes[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
        ccfg.unpack_to_dest_mode = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode(modes)
        kernels = [
            ttnn.KernelDescriptor(kernel_source=f"{KDIR36}/reader.cpp", core_ranges=self.grid,
                                  compile_time_args=r_ct, runtime_args=r_rt, config=ttnn.ReaderConfigDescriptor()),
            ttnn.KernelDescriptor(kernel_source=f"{KDIR36}/writer.cpp", core_ranges=self.grid,
                                  compile_time_args=w_ct, runtime_args=w_rt, config=ttnn.WriterConfigDescriptor()),
            ttnn.KernelDescriptor(kernel_source=f"{KDIR36}/compute.cpp", core_ranges=self.grid,
                                  compile_time_args=self.ct_args + [1, self.stage], runtime_args=c_rt, config=ccfg,
                                  defines=[("INP_FLOAT32", "1")]),
        ]
        return ttnn.generic_op(ins + outs, ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))


class GdnCommit36(base.GdnCommit):
    def __init__(self, dev, debug: int = 0):
        super().__init__(dev, debug)
        g = dev.compute_with_storage_grid_size()
        self.cores = [ttnn.CoreCoord(h % g.x, h // g.x) for h in range(NH)]
        self.grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in self.cores])


class GdnConv36(base.GdnConv):
    def __init__(self, dev, ncores: int = 64):
        super().__init__(dev, ncores)
        g = dev.compute_with_storage_grid_size()
        nt = 256                                               # 8192 channels
        per = -(-nt // ncores)
        self.work = [(j0, min(per, nt - j0)) for j0 in range(0, nt, per)]
        self.cores = [ttnn.CoreCoord(i % g.x, i // g.x) for i in range(len(self.work))]
        self.grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in self.cores])


def reference(conv, gam, beta, S0, z, nw, T, eps=1e-6, eps_l2=1e-6):
    """float64, Qwen3.6 shapes: conv [32,8192], gam/beta [32,32], S0 [32,128,128], z [32,4096], nw [128]"""
    import math
    d = torch.float64
    hd = conv.to(d).reshape(32, 64, 128).permute(1, 0, 2)
    q = hd[:16].repeat_interleave(2, 0)
    k = hd[16:32].repeat_interleave(2, 0)
    v = hd[32:64]
    q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + eps_l2) / math.sqrt(128)
    k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + eps_l2)
    g = gam.to(d).T[:, :, None]
    b = beta.to(d).T[:, :, None]
    tril = torch.tril(torch.ones(32, 32, dtype=d))
    D = torch.exp((g - g.transpose(1, 2)) * tril) * tril
    A = (k @ k.transpose(1, 2)) * D * torch.tril(torch.ones(32, 32, dtype=d), -1) * b
    eG = torch.exp(g)
    S = S0.to(d)
    VB = (v - (k @ S) * eG) * b
    U = VB.clone()
    for _ in range(T - 1):
        U = VB - A @ U
    O = (q @ S) * eG + ((q @ k.transpose(1, 2)) * D) @ U
    O = O / torch.sqrt((O * O).mean(-1, keepdim=True) + eps) * nw.to(d)
    zz = z.to(d).reshape(32, 32, 128).permute(1, 0, 2)
    y = (O * torch.nn.functional.silu(zz)).permute(1, 0, 2).reshape(32, 4096)
    return y, k, U


if __name__ == "__main__":       # python -m tt_qwen36.q36.kgdn [T]
    import tt_qwen36.wallclock  # noqa: F401
    import sys
    import time
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    dev = ttnn.open_device(device_id=0, l1_small_size=24576, trace_region_size=64 * 1024 * 1024)
    try:
        torch.manual_seed(0)
        rows = torch.arange(32)[:, None] < T
        f = lambda t, shape: ttnn.from_torch(t.reshape(shape).float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                                             device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        conv = torch.randn(32, 8192) * rows
        dtv = (torch.randn(32) * 0.5).tolist()
        av = (-torch.rand(32) * 2 - 0.1).tolist()
        abm = torch.zeros(32, 64)
        abm[:T] = torch.randn(T, 64)
        g_step = torch.nn.functional.softplus(abm[:, :32].double() + torch.tensor(dtv, dtype=torch.float64)) * \
            torch.tensor(av, dtype=torch.float64)
        gam = torch.cumsum(g_step, 0)
        beta = torch.sigmoid(abm[:, 32:].double())
        S0 = torch.randn(32, 128, 128) * 0.05
        z = torch.randn(32, 4096)
        nw = 1 + 0.1 * torch.randn(128)
        ts = [f(conv, (1, 1, 32, 8192)), f(S0, (1, 32, 128, 128)), f(z, (1, 1, 32, 4096)), f(nw, (1, 1, 1, 128))]
        y_t, kn_t, U_t = f(torch.zeros(32, 4096), (1, 1, 32, 4096)), f(torch.zeros(32, 32, 128), (1, 32, 32, 128)), \
            f(torch.zeros(32, 32, 128), (1, 32, 32, 128))
        gout = f(torch.zeros(32, 32, 32), (1, 32, 32, 32))
        ab_t = f(abm, (1, 1, 32, 64))
        k = GdnCore36(dev, T)
        k(*ts, y_t, kn_t, U_t, ab_t, dtv, av, gout)
        yr, kr, Ur = reference(conv, gam.float(), beta.float(), S0, z, nw, T)
        yd = ttnn.to_torch(y_t).reshape(32, 4096)
        Ud = ttnn.to_torch(U_t).reshape(32, 32, 128)
        kd = ttnn.to_torch(kn_t).reshape(32, 32, 128)
        gd = ttnn.to_torch(gout).double().reshape(32, 32, 32)[:, :, 0].T
        print(f"T={T}: y PCC {base.pcc(yd[:T], yr[:T]):.6f} rel {float((yd[:T].double() - yr[:T]).norm() / yr[:T].norm()):.2e}"
              f" | U PCC {base.pcc(Ud[:, :T], Ur[:, :T]):.6f} | kn PCC {base.pcc(kd[:, :T], kr[:, :T]):.6f}"
              f" | gamma max rel {float(((gd[:T] - gam[:T]).abs() / gam[:T].abs()).max()):.2e}", flush=True)
        kc = GdnCommit36(dev)
        for n in (1, T):
            S0c = torch.randn(32, 128, 128)
            S_dev = f(S0c, (1, 32, 128, 128))
            kc.write_sel(n)
            kc(S_dev, kn_t, U_t, gout)
            g = gam.double().T
            dcol = torch.exp(g[:, n - 1:n] - g) * (torch.arange(32) < n)
            ref = torch.exp(g[:, n - 1])[:, None, None] * S0c.double() + kd.double().transpose(1, 2) @ (Ud.double() * dcol[:, :, None])
            got = ttnn.to_torch(S_dev).double().reshape(32, 128, 128)
            print(f"commit n={n}: max rel err {float(((got - ref).abs() / ref.abs().clamp_min(1e-3)).max()):.2e} PCC {base.pcc(got, ref):.8f}",
                  flush=True)
        kv = GdnConv36(dev)
        qkv = torch.randn(32, 8192)
        qkv[T:] = 0
        hist = torch.zeros(32, 8192)
        hist[:3] = torch.randn(3, 8192).bfloat16().float()
        wts = [torch.randn(8192) * 0.3 for _ in range(4)]
        conv_o, E_o = f(torch.zeros(32, 8192), (1, 1, 32, 8192)), f(torch.zeros(32, 8192), (1, 1, 32, 8192))
        q_t, h_t, w_t = f(qkv, (1, 1, 32, 8192)), f(hist, (1, 1, 32, 8192)), [f(w, (1, 1, 1, 8192)) for w in wts]
        kv(q_t, h_t, w_t, conv_o, E_o)
        Eref = hist.double().clone()
        Eref[3:] += qkv.double()[:29]
        cref = torch.zeros(32, 8192, dtype=torch.float64)
        for kk in range(4):
            sh = torch.zeros(32, 8192, dtype=torch.float64)
            sh[:32 - kk] = Eref[kk:]
            cref += sh * wts[kk].double()
        cref = torch.nn.functional.silu(cref)
        cd = ttnn.to_torch(conv_o).double().reshape(32, 8192)
        print(f"conv: PCC {base.pcc(cd[:T], cref[:T]):.7f}", flush=True)

        def timed(fn, reps=10):
            fn()
            ttnn.synchronize_device(dev)
            tid = ttnn.begin_trace_capture(dev, cq_id=0)
            for _ in range(reps):
                fn()
            ttnn.end_trace_capture(dev, tid, cq_id=0)
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
            t0 = time.perf_counter()
            for _ in range(3):
                ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(dev)
            ttnn.release_trace(dev, tid)
            return (time.perf_counter() - t0) / (3 * reps) * 1e3
        print(f"core {timed(lambda: k(*ts, y_t, kn_t, U_t, ab_t, dtv, av, gout)):.3f} ms | "
              f"commit {timed(lambda: kc(S_dev, kn_t, U_t, gout)):.3f} ms | "
              f"conv {timed(lambda: kv(q_t, h_t, w_t, conv_o, E_o)):.3f} ms",
              flush=True)
    finally:
        ttnn.close_device(dev)


KDIRB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels", "gdn_bat")


class GdnBatch:
    """GDN core for B users x 1 token (row b = user b), states S [B,32,128,128] updated in place.
    Kernels tt_qwen36/q36/kernels/gdn_bat (one value head per core, loop over the users)."""
    EXACT = (31, 35, 36, 37, 43, 46, 47, 48)

    def __init__(self, dev, B: int, eps: float = 1e-6, eps_l2: float = 1e-6):
        import math
        self.dev, self.B = dev, B
        g = dev.compute_with_storage_grid_size()
        self.cores = [ttnn.CoreCoord(h % g.x, h // g.x) for h in range(NH)]
        self.grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in self.cores])
        self.ct = [B, base.f2u(eps_l2), base.f2u(1 / math.sqrt(128)), base.f2u(1.0), base.f2u(1 / 128), base.f2u(eps)]
        c = torch.zeros(1, 1, 32, 128)
        c[0, 0, :, 0:32] = torch.tril(torch.ones(32, 32))
        c[0, 0, :, 32:64] = torch.tril(torch.ones(32, 32), -1)
        c[0, 0, :, 64:96] = 1.0
        c[0, 0, :, 96:128] = torch.eye(32)
        self.consts = ttnn.from_torch(c, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev,
                                      memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def __call__(self, conv, S, z, nw, y, ab, dt, a, z_off=0, ab_off=0):
        B = self.B
        ins = [conv, S, self.consts, z, nw, ab]
        r_ct = [B]
        for t in ins:
            r_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        w_ct = [B] + ttnn.TensorAccessorArgs(y).get_compile_time_args() + ttnn.TensorAccessorArgs(S).get_compile_time_args()
        r_rt, w_rt, c_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for h, c in enumerate(self.cores):
            r_rt[c.x][c.y] = [h] + [t.buffer_address() for t in ins] + [z_off, ab_off]
            w_rt[c.x][c.y] = [h, y.buffer_address(), S.buffer_address()]
            c_rt[c.x][c.y] = [h // 32, (NH + h) // 32, base.f2u(dt[h]), base.f2u(a[h])]
        sizes = {0: 4, 1: 4, 2: 4, 4: 1, 5: 16 * B, 6: 4, 7: 4, 8: 4, 9: 4, 10: 1, 11: 4, 12: 4, 13: 1, 16: 4,
                 19: 1, 21: 4, 22: 4, 25: 4, 26: 1, 27: 4, 28: 4, 29: 4, 30: 1, 31: 2, 32: 2, 33: 1, 35: 2, 36: 1, 37: 1,
                 38: B, 39: B, 40: 2, 41: 4, 42: 4, 43: 1, 44: 4 * B, 45: 4, 46: 1, 47: 16 * B, 48: 1, 49: 4 * B}
        cbs = []
        for idx, n in sizes.items():
            dt_ = ttnn.bfloat16 if idx == 30 else ttnn.float32
            page = 2048 if idx == 30 else 4096
            cbs.append(ttnn.CBDescriptor(total_size=page * n, core_ranges=self.grid, format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dt_, page_size=page)]))
        ccfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
        modes = [ttnn.UnpackToDestMode.Default] * 64
        for i in self.EXACT:
            modes[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
        ccfg.unpack_to_dest_mode = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode(modes)
        ks = [ttnn.KernelDescriptor(kernel_source=f"{KDIRB}/reader.cpp", core_ranges=self.grid, compile_time_args=r_ct,
                                    runtime_args=r_rt, config=ttnn.ReaderConfigDescriptor()),
              ttnn.KernelDescriptor(kernel_source=f"{KDIRB}/writer.cpp", core_ranges=self.grid, compile_time_args=w_ct,
                                    runtime_args=w_rt, config=ttnn.WriterConfigDescriptor()),
              ttnn.KernelDescriptor(kernel_source=f"{KDIRB}/compute.cpp", core_ranges=self.grid,
                                    compile_time_args=self.ct, runtime_args=c_rt, config=ccfg,
                                    defines=[("INP_FLOAT32", "1")])]
        return ttnn.generic_op(ins + [y], ttnn.ProgramDescriptor(kernels=ks, semaphores=[], cbs=cbs))


def batch_reference(conv, ab, S0, z, nw, dt, a, B, eps=1e-6):
    """float64: rows b < B are independent users, each one recurrent step from its own state S0[b] [32,128,128]"""
    import math
    d = torch.float64
    hd = conv.to(d).reshape(32, 64, 128)
    ys, Ss = torch.zeros(32, 4096, dtype=d), []
    dt_, a_ = torch.tensor(dt, dtype=d), torch.tensor(a, dtype=d)
    for b in range(B):
        q = hd[b, :16].repeat_interleave(2, 0)
        k = hd[b, 16:32].repeat_interleave(2, 0)
        v = hd[b, 32:64]
        q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) / math.sqrt(128)
        k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        g = torch.nn.functional.softplus(ab[b, :32].to(d) + dt_) * a_
        beta = torch.sigmoid(ab[b, 32:64].to(d))
        S = S0[b].to(d) * torch.exp(g)[:, None, None]
        kv = (S * k[:, :, None]).sum(1)
        dlt = (v - kv) * beta[:, None]
        S = S + k[:, :, None] * dlt[:, None, :]
        o = (S * q[:, :, None]).sum(1)
        o = o / torch.sqrt((o * o).mean(-1, keepdim=True) + eps) * nw.to(d)
        o = o * torch.nn.functional.silu(z[b].to(d).reshape(32, 128))
        ys[b] = o.reshape(4096)
        Ss.append(S)
    return ys, torch.stack(Ss)


def test_batch():                  # python -c "from tt_qwen36.q36.kgdn import test_batch; test_batch()"
    import time
    dev = ttnn.open_device(device_id=0, l1_small_size=24576, trace_region_size=64 * 1024 * 1024)
    try:
        f = lambda t, shape: ttnn.from_torch(t.reshape(shape).float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                                             device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for B in (1, 2, 4):
            torch.manual_seed(B)
            conv = torch.zeros(32, 8192)
            conv[:B] = torch.randn(B, 8192)
            ab = torch.zeros(32, 64)
            ab[:B] = torch.randn(B, 64)
            S0 = torch.randn(B, 32, 128, 128) * 0.05
            z = torch.randn(32, 4096)
            nw = 1 + 0.1 * torch.randn(128)
            dtv = (torch.randn(32) * 0.5).tolist()
            av = (-torch.rand(32) * 2 - 0.1).tolist()
            S_t = f(S0, (B, 32, 128, 128))
            y_t = f(torch.zeros(32, 4096), (1, 1, 32, 4096))
            k = GdnBatch(dev, B)
            args = (f(conv, (1, 1, 32, 8192)), S_t, f(z, (1, 1, 32, 4096)), f(nw, (1, 1, 1, 128)), y_t,
                    f(ab, (1, 1, 32, 64)), dtv, av)
            k(*args)
            yr, Sr = batch_reference(conv, ab, S0, z, nw, dtv, av, B)
            yd = ttnn.to_torch(y_t).double().reshape(32, 4096)[:B]
            Sd = ttnn.to_torch(S_t).double().reshape(B, 32, 128, 128)
            print(f"B={B}: y PCC {base.pcc(yd, yr[:B]):.6f} rel {float((yd - yr[:B]).norm() / yr[:B].norm()):.2e} | "
                  f"S PCC {base.pcc(Sd, Sr):.8f} max abs {float((Sd - Sr).abs().max()):.2e}", flush=True)
            k(*args)
            ttnn.synchronize_device(dev)
            tid = ttnn.begin_trace_capture(dev, cq_id=0)
            for _ in range(10):
                k(*args)
            ttnn.end_trace_capture(dev, tid, cq_id=0)
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
            t0 = time.perf_counter()
            for _ in range(3):
                ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(dev)
            ttnn.release_trace(dev, tid)
            print(f"   {(time.perf_counter() - t0) / 30 * 1e3:.3f} ms per call", flush=True)
    finally:
        ttnn.close_device(dev)
