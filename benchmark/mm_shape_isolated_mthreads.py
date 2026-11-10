"""Run the MTT MM shape corpus with process isolation.

The MUSA runtime can report an asynchronous launch error at a later API call.
Running one shape in one child process prevents an OOB/timeout from poisoning
the measurements for every following shape.  The child uses the existing
``mm_shape_batch_mthreads.py`` benchmark, so dispatch and timing semantics stay
identical to the normal FlagGems benchmark.
"""

from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import re
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / "benchmark" / "mm_shape_batch_mthreads.py"


def read_shape_records(path: str):
    """Read ``mm.shapes`` as ``(M, N, K, count)`` records.

    The source YAML is intentionally parsed without depending on PyYAML.  The
    shape section is a regular five-value block: batch, M, N, K, count.
    """
    lines = Path(path).read_text().splitlines()
    in_mm = False
    records = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line == "mm:":
            in_mm = True
            i += 1
            continue
        if in_mm and re.match(r"^[A-Za-z0-9_]+:", line):
            break
        if in_mm and line.startswith("  - -"):
            values = []
            for j in range(i, min(i + 5, len(lines))):
                match = re.search(r"-\s*(-?\d+)\s*$", lines[j])
                if not match:
                    break
                values.append(int(match.group(1)))
            if len(values) == 5:
                _, m, n, k, count = values
                records.append((m, n, k, count))
                i += 5
                continue
        i += 1
    if not records:
        raise ValueError(f"no mm shapes found in {path}")
    # Preserve the corpus order while removing duplicate shape rows. Counts
    # are summed because the YAML may contain the same shape more than once.
    merged = {}
    order = []
    for m, n, k, count in records:
        key = (m, n, k)
        if key not in merged:
            merged[key] = 0
            order.append(key)
        merged[key] += count
    return [(m, n, k, merged[(m, n, k)]) for m, n, k in order]


def _run_child(shape, dtype, args, device, workdir):
    m, n, k, _ = shape
    shape_text = f"{m}x{n}x{k}"
    result_path = workdir / f"{m}_{n}_{k}_{dtype}.tsv"
    cache = workdir / f"cache_{m}_{n}_{k}_{dtype}"
    env = os.environ.copy()
    env["MUSA_VISIBLE_DEVICES"] = str(device)
    env["TRITON_CACHE_DIR"] = str(cache)
    # AABS can rewrite BLOCK_* values after autotune metadata is selected.
    # Disable it so every row measures the exact FlagGems Triton candidate.
    env["FLAGTREE_AABS"] = "0"
    command = [
        sys.executable,
        str(BATCH),
        "--shapes",
        args.shapes,
        "--shape",
        shape_text,
        "--dtype",
        dtype,
        "--warmup",
        str(args.warmup),
        "--rep",
        str(args.rep),
        "--out",
        str(result_path),
    ]
    try:
        completed = subprocess.run(
            command,
            env=env,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=args.timeout,
            check=False,
        )
        output = completed.stdout or ""
        rows = []
        if result_path.exists():
            with result_path.open() as stream:
                rows = list(csv.DictReader(stream, delimiter="\t"))
        if rows:
            for row in rows:
                row["child_rc"] = str(completed.returncode)
                row["child_log"] = output[-1000:].replace("\n", " ")
            return rows
        return [{
            "dtype": dtype,
            "M": str(m),
            "N": str(n),
            "K": str(k),
            "torch_ms": "nan",
            "triton_ms": "nan",
            "ratio": "nan",
            "dispatch": "unknown",
            "status": f"child_rc={completed.returncode}: {output[-500:]}".replace(
                "\n", " "
            ),
            "child_rc": str(completed.returncode),
            "child_log": output[-1000:].replace("\n", " "),
        }]
    except subprocess.TimeoutExpired as exc:
        return [{
            "dtype": dtype,
            "M": str(m),
            "N": str(n),
            "K": str(k),
            "torch_ms": "nan",
            "triton_ms": "nan",
            "ratio": "nan",
            "dispatch": "unknown",
            "status": f"TimeoutExpired({args.timeout}s)",
            "child_rc": "timeout",
            "child_log": str(exc)[-1000:].replace("\n", " "),
        }]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--devices", default="0")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rep", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    shapes = read_shape_records(args.shapes)
    if args.limit:
        shapes = shapes[: args.limit]
    devices = [x.strip() for x in args.devices.split(",") if x.strip()]
    if not devices:
        parser.error("--devices must contain at least one device")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mm-isolated-", dir=out_path.parent) as tmp:
        workdir = Path(tmp)
        jobs = []
        for index, shape in enumerate(shapes):
            jobs.append((index, shape))

        results = []
        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            futures = {
                pool.submit(
                    _run_child, shape, "both", args,
                    devices[index % len(devices)], workdir
                ): (index, shape)
                for index, shape in jobs
            }
            for future in as_completed(futures):
                index, shape = futures[future]
                try:
                    rows = future.result()
                except Exception as exc:  # keep the corpus scan running
                    rows = {
                        "dtype": "both",
                        "M": str(shape[0]),
                        "N": str(shape[1]),
                        "K": str(shape[2]),
                        "torch_ms": "nan",
                        "triton_ms": "nan",
                        "ratio": "nan",
                        "dispatch": "unknown",
                        "status": f"parent:{type(exc).__name__}:{exc}",
                        "child_rc": "parent",
                        "child_log": "",
                    }
                results.append((index, shape, rows))
                print(f"finished {index + 1}/{len(jobs)} {shape[:3]}", flush=True)

    # The batch child emits one row per dtype.  A failed dtype is retried in a
    # fresh process, which handles asynchronous MUSA errors that poison the
    # sibling dtype in the first child.
    normalized = []
    for index, shape, rows in sorted(results):
        if isinstance(rows, dict):
            rows = [rows]
        for row in rows:
            if row.get("status") != "ok":
                dtype = row.get("dtype")
                status = row.get("status", "")
                retryable = not any(
                    marker in status
                    for marker in ("TimeoutExpired", "PTXASError", "no registers")
                )
                if dtype in ("bf16", "fp16") and retryable:
                    retry = _run_child(shape, dtype, args, devices[index % len(devices)], Path(args.out).parent)
                    # _run_child consistently returns a list of rows; a
                    # single-dtype retry therefore still needs selecting its
                    # matching row rather than treating the list as a dict.
                    retry_rows = retry if isinstance(retry, list) else [retry]
                    retry_row = next(
                        (candidate for candidate in retry_rows
                         if candidate.get("dtype") == dtype),
                        None,
                    )
                    if retry_row is not None and retry_row.get("status") == "ok":
                        row = retry_row
            row["shape_index"] = str(index)
            row["count"] = str(shape[3])
            normalized.append(row)

    fields = [
        "shape_index", "count", "dtype", "M", "N", "K", "torch_ms",
        "triton_ms", "ratio", "dispatch", "status", "child_rc", "child_log",
    ]
    with out_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(normalized)
    print(f"wrote {len(normalized)} rows to {out_path}", flush=True)


if __name__ == "__main__":
    main()
