"""Probe a fused single-pipe 320x256 split-M schedule (no dispatch changes)."""

import argparse
import importlib

import torch
import torch_musa  # noqa: F401
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from flag_gems.runtime import torch_device_fn

tle = importlib.import_module("triton.experimental.tle.language")
mm = importlib.import_module("flag_gems.runtime.backend._mthreads.ops.mm")


@triton.jit
def _producer(writer, a_desc, b_desc, m_offset, n_offset,
              k_tiles: tl.constexpr, block_k: tl.constexpr):
    for k_iter in range(k_tiles):
        k_offset = k_iter * block_k
        slot = writer.acquire(k_iter)
        # TLE requires the copy shape to equal the allocated buffer shape;
        # consumers use only the first 320 rows of this 512-row payload.
        tle.gpu.copy(a_desc, slot.a, [512, block_k], [m_offset, k_offset])
        tle.gpu.copy(b_desc, slot.b, [block_k, 256], [k_offset, n_offset])
        writer.commit(k_iter)


@triton.jit
def _consumer(reader, c_ptr, m_offset, n_offset, stride_cm, stride_cn,
              m, n, k_tiles: tl.constexpr, row_offset: tl.constexpr,
              block_m: tl.constexpr):
    acc = tl.zeros((block_m, 256), dtype=tl.float32)
    for k_iter in range(k_tiles):
        ready = reader.wait(k_iter)
        a_tile = ready.slot.a.slice(row_offset, block_m, dim=0)
        acc = tle.gpu.wgmma(a_tile, ready.slot.b, acc)
        acc = tle.gpu.wgmma_wait(0, acc)
        reader.release(k_iter)
    rm = m_offset + row_offset + tl.arange(0, block_m)
    rn = n_offset + tl.arange(0, 256)
    ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    mask = (rm < m)[:, None] & (rn < n)[None, :]
    tl.store(ptrs, acc.to(c_ptr.dtype.element_ty), mask=mask)


@triton.jit
def splitm_fused_single_pipe_kernel(a_desc, b_desc, c_ptr, m, n,
                                    stride_cm, stride_cn,
                                    k_tiles: tl.constexpr,
                                    block_k: tl.constexpr,
                                    slots: tl.constexpr,
                                    block_m_bottom: tl.constexpr):
    pid = tl.program_id(0)
    grid_n = tl.cdiv(n, 256)
    m_offset = ((pid // grid_n) * (256 + block_m_bottom)).to(tl.int32)
    n_offset = ((pid % grid_n) * 256).to(tl.int32)
    a_smem = tle.gpu.alloc([slots, 512, block_k], dtype=a_desc.dtype,
                            layout=None, scope=tle.gpu.smem,
                            nv_mma_shared_layout=True)
    b_smem = tle.gpu.alloc([slots, block_k, 256], dtype=b_desc.dtype,
                            layout=None, scope=tle.gpu.smem,
                            nv_mma_shared_layout=True)
    pipe = tle.pipe(capacity=slots, scope="cta", name="splitm_fused_probe",
                    a=a_smem, b=b_smem)
    tle.gpu.warp_specialize(
        [
            (_consumer, (pipe.reader(), c_ptr, m_offset, n_offset,
                         stride_cm, stride_cn, m, n, k_tiles, 0, 256)),
            (_consumer, (pipe.reader(), c_ptr, m_offset, n_offset,
                         stride_cm, stride_cn, m, n, k_tiles, 256,
                         block_m_bottom)),
            (_producer, (pipe.writer(), a_desc, b_desc, m_offset, n_offset,
                         k_tiles, block_k)),
        ],
        [4, 4],
        [168, 24],
    )


def run(a, b, out, slots, block_m_bottom):
    m, k = a.shape
    n = b.shape[1]
    # Descriptor dimensions must be powers of two; the producer copies the
    # logical top+bottom rows from the 512-row descriptor.
    block_k = run.block_k
    da = TensorDescriptor.from_tensor(a, [512, block_k])
    db = TensorDescriptor.from_tensor(b, [block_k, 256])
    grid = (triton.cdiv(m, 256 + block_m_bottom) * triton.cdiv(n, 256),)
    with torch_device_fn.device(a.device):
        splitm_fused_single_pipe_kernel[grid](
            da, db, out, m, n, out.stride(0), out.stride(1),
            triton.cdiv(k, block_k), block_k, slots, block_m_bottom,
            num_warps=16, enable_backend_opt=True,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", nargs=3, type=int, action="append", required=True)
    ap.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    ap.add_argument("--slots", type=int, default=3)
    ap.add_argument("--block-m-bottom", type=int, default=64)
    ap.add_argument("--block-k", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--rep", type=int, default=30)
    args = ap.parse_args()
    run.block_k = args.block_k
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    for m, n, k in args.shape:
        if k % 64:
            raise ValueError("split-M fused probe requires K divisible by 64")
        a = torch.randn((m, k), device="musa", dtype=dtype)
        b = torch.randn((k, n), device="musa", dtype=dtype)
        ref = torch.mm(a, b)
        out_fused = torch.empty_like(ref)
        out_current = torch.empty_like(ref)
        run(a, b, out_fused, args.slots, args.block_m_bottom)
        mm.mm_tle_split_m_pipe(a, b, out_current, m, n, k)
        torch.musa.synchronize()
        torch.testing.assert_close(out_fused, ref, atol=3e-2, rtol=3e-2)
        torch.testing.assert_close(out_current, ref, atol=3e-2, rtol=3e-2)
        torch_fn = lambda: torch.mm(a, b, out=out_current)
        fused_fn = lambda: run(a, b, out_fused, args.slots,
                               args.block_m_bottom)
        current_fn = lambda: mm.mm_tle_split_m_pipe(a, b, out_current, m, n, k)
        torch_ms = triton.testing.do_bench(torch_fn, warmup=args.warmup, rep=args.rep)
        fused_ms = triton.testing.do_bench(fused_fn, warmup=args.warmup, rep=args.rep)
        current_ms = triton.testing.do_bench(current_fn, warmup=args.warmup, rep=args.rep)
        print(
            f"RESULT shape={m}x{n}x{k} dtype={args.dtype} slots={args.slots} "
            f"torch_ms={torch_ms:.6f} fused_ms={fused_ms:.6f} "
            f"fused_ratio={torch_ms/fused_ms:.4f} current_ms={current_ms:.6f} "
            f"current_ratio={torch_ms/current_ms:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
