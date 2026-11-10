"""Probe a one-consumer/128-column multifield TLE GEMM for small M."""
import argparse
import importlib
import torch
import torch_musa  # noqa: F401
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
from flag_gems.runtime import torch_device_fn
from triton.experimental.tle.language import gpu as tg
import triton.experimental.tle.language as tle

mm = importlib.import_module("flag_gems.runtime.backend._mthreads.ops.mm")

@triton.jit
def prod(writer, ad, bd, no, kt: tl.constexpr, bm: tl.constexpr, bk: tl.constexpr):
    for i in range(kt):
        s = writer.acquire(i)
        tg.copy(ad, s.a, [bm, bk], [0, i * bk])
        tg.copy(bd, s.b, [bk, 128], [i * bk, no])
        writer.commit(i)

@triton.jit
def cons(reader, c, no, scm, scn, m, n, kt: tl.constexpr, bm: tl.constexpr):
    acc = tl.zeros((bm, 128), dtype=tl.float32)
    for i in range(kt):
        s = reader.wait(i)
        acc = tg.wgmma(s.slot.a, s.slot.b, acc)
        acc = tg.wgmma_wait(0, acc)
        reader.release(i)
    om = tl.arange(0, bm)
    on = no + tl.arange(0, 128)
    ptr = c + om[:, None] * scm + on[None, :] * scn
    tl.store(ptr, acc.to(c.dtype.element_ty), mask=(om < m)[:, None] & (on < n)[None, :])

@triton.jit
def kernel(ad, bd, c, m, n, scm, scn, kt: tl.constexpr, bm: tl.constexpr, bk: tl.constexpr, slots: tl.constexpr):
    no = (tl.program_id(0) * 128).to(tl.int32)
    asmem = tg.alloc([slots, bm, bk], dtype=ad.dtype, layout=None, scope=tg.smem, nv_mma_shared_layout=True)
    bsmem = tg.alloc([slots, bk, 128], dtype=bd.dtype, layout=None, scope=tg.smem, nv_mma_shared_layout=True)
    p = tle.pipe(capacity=slots, scope="cta", name="smallm128", a=asmem, b=bsmem)
    tg.warp_specialize([(cons, (p.reader(), c, no, scm, scn, m, n, kt, bm)),
                        (prod, (p.writer(), ad, bd, no, kt, bm, bk))], [4], [24])

def run(a, b, o, bm, bk, slots):
    m, k = a.shape; n = b.shape[1]
    ad = TensorDescriptor.from_tensor(a, [bm, bk]); bd = TensorDescriptor.from_tensor(b, [bk, 128])
    with torch_device_fn.device(a.device):
        kernel[(triton.cdiv(n, 128),)](ad, bd, o, m, n, o.stride(0), o.stride(1), triton.cdiv(k, bk), bm, bk, slots, num_warps=4, enable_backend_opt=True)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--m", type=int, default=4); ap.add_argument("--n", type=int, default=12288); ap.add_argument("--k", type=int, default=2048); ap.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16"); ap.add_argument("--bm", type=int, default=16); ap.add_argument("--bk", type=int, default=64); ap.add_argument("--slots", type=int, default=2); ap.add_argument("--warmup", type=int, default=30); ap.add_argument("--rep", type=int, default=150); args = ap.parse_args()
    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    a = torch.randn((args.m, args.k), device="musa", dtype=dt); b = torch.randn((args.k, args.n), device="musa", dtype=dt); o = torch.empty((args.m, args.n), device="musa", dtype=dt)
    ref = torch.mm(a, b); run(a, b, o, args.bm, args.bk, args.slots); torch.musa.synchronize(); torch.testing.assert_close(o, ref, atol=3e-2, rtol=3e-2)
    t = triton.testing.do_bench(lambda: torch.mm(a, b, out=o), warmup=args.warmup, rep=args.rep); x = triton.testing.do_bench(lambda: run(a, b, o, args.bm, args.bk, args.slots), warmup=args.warmup, rep=args.rep)
    print(f"RESULT shape={args.m}x{args.n}x{args.k} dtype={args.dtype} bm={args.bm} bn=128 bk={args.bk} slots={args.slots} torch_ms={t:.6f} mm_ms={x:.6f} ratio={t/x:.4f}", flush=True)

if __name__ == "__main__": main()
