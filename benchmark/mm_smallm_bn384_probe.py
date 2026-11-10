"""Probe a 384-column multifield TLE tile for M=64 workloads."""
import argparse
import importlib

import torch
import torch_musa  # noqa: F401
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.tools.tensor_descriptor import TensorDescriptor
from flag_gems.runtime import torch_device_fn
from triton.experimental.tle.language import gpu as tg

mm = importlib.import_module("flag_gems.runtime.backend._mthreads.ops.mm")

@triton.jit
def producer(writer, ad, bd, no, kt: tl.constexpr):
    for i in range(kt):
        s = writer.acquire(i)
        tg.copy(ad, s.a, [64, 64], [0, i * 64])
        tg.copy(bd, s.b, [64, 384], [i * 64, no])
        writer.commit(i)

@triton.jit
def consumer(reader, c, no, scm, scn, m, n, kt: tl.constexpr):
    acc0 = tl.zeros((64, 128), dtype=tl.float32)
    acc1 = tl.zeros((64, 128), dtype=tl.float32)
    acc2 = tl.zeros((64, 128), dtype=tl.float32)
    for i in range(kt):
        s = reader.wait(i)
        acc0 = tg.wgmma(s.slot.a, s.slot.b.slice(0, 128, dim=1), acc0)
        acc1 = tg.wgmma(s.slot.a, s.slot.b.slice(128, 128, dim=1), acc1)
        acc2 = tg.wgmma(s.slot.a, s.slot.b.slice(256, 128, dim=1), acc2)
        acc0 = tg.wgmma_wait(0, acc0)
        acc1 = tg.wgmma_wait(0, acc1)
        acc2 = tg.wgmma_wait(0, acc2)
        reader.release(i)
    om = tl.arange(0, 64); on = tl.arange(0, 128); maskm = (om < m)[:, None]
    for j, acc in [(0, acc0), (128, acc1), (256, acc2)]:
        cols = no + j + on
        ptr = c + om[:, None] * scm + cols[None, :] * scn
        tl.store(ptr, acc.to(c.dtype.element_ty), mask=maskm & (cols < n)[None, :])

@triton.jit
def kernel(ad, bd, c, m, n, scm, scn, kt: tl.constexpr, slots: tl.constexpr):
    no = (tl.program_id(0) * 384).to(tl.int32)
    a = tg.alloc([slots, 64, 64], dtype=ad.dtype, layout=None, scope=tg.smem, nv_mma_shared_layout=True)
    b = tg.alloc([slots, 64, 384], dtype=bd.dtype, layout=None, scope=tg.smem, nv_mma_shared_layout=True)
    p = tle.pipe(capacity=slots, scope="cta", name="smallm384", a=a, b=b)
    tg.warp_specialize([(consumer, (p.reader(), c, no, scm, scn, m, n, kt)),
                        (producer, (p.writer(), ad, bd, no, kt))], [4], [24])

def run(a, b, out, slots):
    m, k = a.shape; n = b.shape[1]
    ad = TensorDescriptor.from_tensor(a, [64, 64]); bd = TensorDescriptor.from_tensor(b, [64, 384])
    with torch_device_fn.device(a.device):
        kernel[(triton.cdiv(n, 384),)](ad, bd, out, m, n, out.stride(0), out.stride(1), triton.cdiv(k, 64), slots, num_warps=4, enable_backend_opt=True)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--m", type=int, default=64); ap.add_argument("--n", type=int, default=9216); ap.add_argument("--k", type=int, default=2048); ap.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16"); ap.add_argument("--slots", type=int, default=2); ap.add_argument("--warmup", type=int, default=20); ap.add_argument("--rep", type=int, default=100); args = ap.parse_args()
    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    a = torch.randn((args.m, args.k), device="musa", dtype=dt); b = torch.randn((args.k, args.n), device="musa", dtype=dt); out = torch.empty((args.m, args.n), device="musa", dtype=dt); ref = torch.mm(a, b)
    run(a, b, out, args.slots); torch.musa.synchronize(); torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
    t = triton.testing.do_bench(lambda: torch.mm(a, b, out=out), warmup=args.warmup, rep=args.rep); x = triton.testing.do_bench(lambda: run(a, b, out, args.slots), warmup=args.warmup, rep=args.rep)
    print(f"RESULT shape={args.m}x{args.n}x{args.k} dtype={args.dtype} bn=384 slots={args.slots} torch_ms={t:.6f} mm_ms={x:.6f} ratio={t/x:.4f}", flush=True)

if __name__ == "__main__": main()
