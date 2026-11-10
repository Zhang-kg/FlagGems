# MThreads legacy source snapshot

This branch preserves source files from the pre-migration worktree:

- host: `zhiyuan_mthreads`
- path: `/data/zhiyuan_kernels/.work/flaggems_pr_baddbmm_20260925`
- source HEAD: `39cdcbb768116b022662fe743c109bb6efd976ef`
- snapshot date: 2026-11-10

It extends `backup/legacy-mm-experiments-20261110` with the remaining
MThreads baddbmm implementation, its helper kernels, the v14 companion
reference, and the original MM source backups.

The snapshot intentionally excludes Git metadata, Python and Triton caches,
compiled binaries, core dumps, AppleDouble files, large model inputs, and
benchmark logs. The source worktree was not modified while creating it.
