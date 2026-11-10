"""Probe BN320 with one exact three-field A/B pipe.

The original probe padded the 320-column output to a 512-column B payload,
which transferred 33 percent more data than production BN320.  This variant
keeps A, B-main and B-tail as separate fields while sharing one pipe token.
"""
import argparse
import importlib
import statistics

import torch
import torch_musa  # noqa: F401
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from flag_gems.runtime import torch_device_fn

tle = importlib.import_module("triton.experimental.tle.language")
mm = importlib.import_module("flag_gems.runtime.backend._mthreads.ops.mm")


@triton.jit
def _producer(writer, a_desc, b_main_desc, b_tail_desc, m_offset, n_offset,
              BLOCK_K: tl.constexpr, k_tiles: tl.constexpr):
    for k_iter in range(k_tiles):
        k_offset = k_iter * BLOCK_K
        slot = writer.acquire(k_iter)
        tle.gpu.copy(a_desc, slot.a, [256, BLOCK_K], [m_offset, k_offset])
        tle.gpu.copy(
            b_main_desc, slot.b_main, [BLOCK_K, 256], [k_offset, n_offset]
        )
        tle.gpu.copy(
            b_tail_desc, slot.b_tail, [BLOCK_K, 64], [k_offset, n_offset + 256]
        )
        writer.commit(k_iter)


@triton.jit
def _consumer(reader, to_top, to_bottom, c_ptr, m_offset, n_offset,
              consumer_id: tl.constexpr, row_offset: tl.constexpr,
              block_m: tl.constexpr, stride_cm, stride_cn, m, n,
              k_tiles: tl.constexpr):
    acc_main = tl.zeros((block_m, 256), dtype=tl.float32)
    acc_tail = tl.zeros((block_m, 64), dtype=tl.float32)
    for k_iter in range(k_tiles):
        ready = reader.wait(k_iter)
        if consumer_id == 0:
            tle.gpu.barrier_wait(to_top, phaseIdx=(k_iter + 1) & 1)
        else:
            tle.gpu.barrier_wait(to_bottom, phaseIdx=k_iter & 1)
        a_tile = ready.slot.a.slice(row_offset, block_m, dim=0)
        b_main = ready.slot.b_main
        b_tail = ready.slot.b_tail
        acc_main = tle.gpu.wgmma(a_tile, b_main, acc_main)
        acc_tail = tle.gpu.wgmma(a_tile, b_tail, acc_tail)
        if consumer_id == 0:
            tle.gpu.barrier_arrive(to_bottom, phaseIdx=k_iter & 1)
        else:
            tle.gpu.barrier_arrive(to_top, phaseIdx=k_iter & 1)
        acc_main = tle.gpu.wgmma_wait(0, acc_main)
        acc_tail = tle.gpu.wgmma_wait(0, acc_tail)
        reader.release(k_iter)

    rm = m_offset + row_offset + tl.arange(0, block_m)
    rn = n_offset + tl.arange(0, 256)
    ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(ptrs, acc_main.to(c_ptr.dtype.element_ty),
             mask=(rm < m)[:, None] & (rn < n)[None, :])
    rn_tail = n_offset + 256 + tl.arange(0, 64)
    ptrs_tail = c_ptr + rm[:, None] * stride_cm + rn_tail[None, :] * stride_cn
    tl.store(ptrs_tail, acc_tail.to(c_ptr.dtype.element_ty),
             mask=(rm < m)[:, None] & (rn_tail < n)[None, :])


@triton.jit
def bn320_onepipe_kernel(a_desc, b_main_desc, b_tail_desc, c_ptr, m, n,
                         stride_cm, stride_cn,
                         grid_m: tl.constexpr, grid_n: tl.constexpr,
                         k_tiles: tl.constexpr, BLOCK_K: tl.constexpr,
                         slots: tl.constexpr):
    pid = tl.program_id(0)
    group_m: tl.constexpr = min(grid_m, 2)
    group_width: tl.constexpr = group_m * grid_n
    group_id = pid // group_width
    first_m = group_id * group_m
    actual_group_m = min(grid_m - first_m, group_m)
    pid_in_group = pid % group_width
    pid_m = first_m + pid_in_group % actual_group_m
    pid_n = pid_in_group // actual_group_m
    m_bottom: tl.constexpr = 64 if grid_m > 0 else 64
    m_offset = (pid_m * 192).to(tl.int32)
    n_offset = (pid_n * 320).to(tl.int32)
    a_smem = tle.gpu.alloc([slots, 256, BLOCK_K], dtype=a_desc.dtype,
                            layout=None, scope=tle.gpu.smem,
                            nv_mma_shared_layout=True)
    b_main_smem = tle.gpu.alloc([slots, BLOCK_K, 256], dtype=b_main_desc.dtype,
                                 layout=None, scope=tle.gpu.smem,
                                 nv_mma_shared_layout=True)
    b_tail_smem = tle.gpu.alloc([slots, BLOCK_K, 64], dtype=b_tail_desc.dtype,
                                 layout=None, scope=tle.gpu.smem,
                                 nv_mma_shared_layout=True)
    pipe = tle.pipe(capacity=slots, scope="cta", name="bn320_onepipe_exact",
                    a=a_smem, b_main=b_main_smem, b_tail=b_tail_smem)
    to_top = tle.gpu.alloc_barrier(arrive_count=8, init=tle.gpu.PENDING)
    to_bottom = tle.gpu.alloc_barrier(arrive_count=8, init=tle.gpu.PENDING)
    tle.gpu.warp_specialize(
        [
            (_consumer, (pipe.reader(), to_top, to_bottom, c_ptr, m_offset,
                         n_offset, 0, 0, 128, stride_cm, stride_cn, m, n,
                         k_tiles)),
            (_consumer, (pipe.reader(), to_top, to_bottom, c_ptr, m_offset,
                         n_offset, 1, 128, 64, stride_cm, stride_cn, m, n,
                         k_tiles)),
            (_producer, (pipe.writer(), a_desc, b_main_desc, b_tail_desc,
                         m_offset, n_offset, BLOCK_K,
                         k_tiles)),
        ],
        [8, 4], [168, 24],
    )


def run(a, b, out, slots, block_k):
    m, k = a.shape
    n = b.shape[1]
    da = TensorDescriptor.from_tensor(a, [256, block_k])
    db_main = TensorDescriptor.from_tensor(b, [block_k, 256])
    db_tail = TensorDescriptor.from_tensor(b, [block_k, 64])
    grid_m = (m + 191) // 192
    grid_n = (n + 319) // 320
    with torch_device_fn.device(a.device):
        bn320_onepipe_kernel[(grid_m * grid_n,)](
            da, db_main, db_tail, out, m, n, out.stride(0), out.stride(1), grid_m,
            grid_n, (k + block_k - 1) // block_k, block_k, slots,
            num_warps=8, enable_backend_opt=True)


def bench(fn, warmup, rep):
    for _ in range(warmup): fn()
    torch.musa.synchronize()
    vals = []
    for _ in range(5):
        s, e = torch.musa.Event(enable_timing=True), torch.musa.Event(enable_timing=True)
        s.record()
        for _ in range(rep): fn()
        e.record(); torch.musa.synchronize()
        vals.append(s.elapsed_time(e) / rep)
    return statistics.median(vals), vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, required=True); ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--k", type=int, default=2048); ap.add_argument("--dtype", choices=("bf16", "fp16"), required=True)
    ap.add_argument("--slots", type=int, default=2); ap.add_argument("--warmup", type=int, default=20); ap.add_argument("--rep", type=int, default=100)
    ap.add_argument("--block-k", type=int, choices=(32, 64), default=64)
    args = ap.parse_args(); dt = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    m, n, k = args.m, args.n, args.k
    a = torch.randn((m, k), device="musa", dtype=dt); b = torch.randn((k, n), device="musa", dtype=dt)
    out = torch.empty((m, n), device="musa", dtype=dt); ref = torch.mm(a, b)
    fn = lambda: run(a, b, out, args.slots, args.block_k)
    fn(); torch.musa.synchronize(); torch.testing.assert_close(out, ref, atol=3e-2, rtol=3e-2)
    torch_ms, ts = bench(lambda: torch.mm(a, b, out=out), args.warmup, args.rep)
    probe_ms, ps = bench(fn, args.warmup, args.rep)
    print(f"RESULT shape={m}x{n}x{k} dtype={args.dtype} slots={args.slots} block_k={args.block_k} torch_ms={torch_ms:.6f} onepipe_ms={probe_ms:.6f} ratio={torch_ms/probe_ms:.4f} torch_samples={ts} onepipe_samples={ps}", flush=True)


if __name__ == "__main__": main()
