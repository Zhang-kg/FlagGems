# mthreads Triton MM 最终评估

## 范围与方法

- 设备：`zhiyuan-mthreads`，MTT S5000；8 卡并行。
- FlagTree AABS：明确关闭，设置 `FLAGTREE_AABS=0`。AABS 会在运行时改写 block 参数，因此不关闭会使候选配置和测量结果不可复现。
- 内核：FlagGems `src/flag_gems/runtime/backend/_mthreads/ops/mm.py`，默认 expanded config/autotune，并保留已验证的 GEMV、split-N、split-M、split384、persistent 分流。
- 工具：`benchmark/mm_shape_isolated_mthreads.py`，每个 shape 使用独立 Python 进程和独立 Triton cache；`warmup=1`、`rep=3`、`timeout=120s`。隔离是必要的，因为 MUSA 异步 OOB/launch timeout 会污染同一进程的后续调用。
- 比例定义：`ratio = torch_ms / triton_ms`，`0.95` 表示达到 Torch 基线的 95%。简单平均、中位数只在 `status=ok` 的成功 shape 上计算；加权平均按 YAML `count` 对成功 shape 加权。失败 shape 不伪造性能比例，单独统计。

两份 YAML 合并重复项后的调用量分别为：35B `71,902`，397B `105,170`。调用量只用于成功 shape 的加权平均；失败 shape 没有可用 ratio。

## 全量结果

| 模型 YAML | dtype | unique shape | 成功 | 失败 | 达到 95% | 达标率（全部） | 达标率（成功） | 简单平均 | count 加权平均 | 中位数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-35B-A3B-p1024d1024 | BF16 | 491 | 232 | 259 | 77 | 15.7% | 33.2% | 0.9238 | 0.9731 | 0.8612 |
| Qwen3.5-35B-A3B-p1024d1024 | FP16 | 491 | 233 | 258 | 57 | 11.6% | 24.5% | 0.8257 | 0.8746 | 0.8228 |
| Qwen3.5-397B-A17B-p1024d1024 | BF16 | 418 | 147 | 271 | 86 | 20.6% | 58.5% | 1.0097 | 1.0245 | 1.1290 |
| Qwen3.5-397B-A17B-p1024d1024 | FP16 | 418 | 147 | 271 | 73 | 17.5% | 49.7% | 0.8571 | 0.8786 | 0.9489 |

35B 的 259 个 BF16 失败中有 258 个 allocator `PTXASError`、1 个异步 MUSA unknown error；FP16 的 258 个全部为 allocator `PTXASError`。397B 每个 dtype 有 232 个 allocator `PTXASError`，另有 39 个 shape 在双 dtype 子进程中触发 120 秒 timeout，因此按 dtype 计为 232+39=271 个失败。397B 的 `both` 行不会被计入成功平均值。

## 低性能形状与原因

### 1. 小 M、N=256/64 的 generic 路径

35B 最低 BF16 为 `M=2,N=64,K=2048`（0.224x），最低 FP16 为 `M=4,N=12288,K=2048`（0.199x，split-N）。397B 最低 BF16/FP16 均为 `M=2,N=4096,K=128`（约 0.204/0.209x）。小 M 时 Torch 的 muBLASLt 直接选择专用小矩阵或 GEMV/轻量 kernel；当前 Triton generic 仍承担 TLE pipe 创建、barrier/wait、shared-memory 搬运和 kernel launch 固定成本，固定成本超过矩阵乘本身。应优先增加真正的 SQMMA 小 M kernel 或更细粒度 GEMV/row-reduction 分流，而不是继续扩大大 tile 表。

### 2. N=4096/2560 等中等列宽的 allocator 失败

35B 中 `512x512x4096`、`496x2560x4096` 等 shape 的 expanded/TLE 候选稳定触发 `no registers from class available to allocate`；固定 BM/BN/BK、单 slot 仍能复现。397B 也有大量 `N=4096` shape 在相同 allocator 阶段失败。Torch 使用 muBLASLt/muDNN 专用 persistent TCE kernel；Triton 版本的 rolling SQMMA、TME descriptor、barrier 状态和多阶段 pipeline 产生更长 SSA live range。FlagTree 当前 MTGPU stripped LLVM allocator 没有 spill/interval splitting，因此无法把高压候选降级为可执行代码。这个问题不是 autotune 搜索空间不足。

### 3. N=64/窄 N 的同步和 occupancy 开销

已验证的 `64x12288x2048` BF16/FP16 仅约 0.641/0.662x；`16384x1024x2048` 通过 muBLASLt 风格 `split384` 分流后达到约 0.994/0.984x（高重复复测 1.020/0.991x）。后者若走通用 WS/multifield 只有约 0.55--0.86x。Torch 对窄 N 选择 `768_384x256` 一类 persistent split kernel；当前 split384 是针对该拓扑的精确补偿，但其它窄 N 仍受 wait-all、pipe/barrier 和 CTA 分区开销限制。

### 4. Torch 与 Triton 的 kernel 拓扑差异

远端 muBLASLt/muDNN 字符串和 profiler 显示 Torch 会按 shape 选择 `128x64`、`256x128`、`384x256`、`512x256`、`768x384/320` 等 TCE kernel，并支持 `64x96`、`96x128`、`128x160` 等非二次幂 tile 及 3x2/5x2/3x3 cluster。当前 Triton LinearLayout/BlockedEncoding/SQMMA selector 主要表达二次幂 tile，且 TLE pipe 使用保守 wait-all ABI；因此即使 expanded config 候选不爆炸，也无法表达 Torch 的完整拓扑。`512x256` 多 reader 探针还会 timeout，说明直接照搬 muDNN payload 会触发当前 pipe/barrier 生命周期限制。

## FlagTree/mthreads 编译器问题与修复状态

| 原问题 | 修复 | 验证与当前状态 |
|---|---|---|
| `PipelineExpander` 只识别直接 `arith.constant`，把 `2*STAGES` 等 constexpr bound 当 dynamic，生成错误/保守 pipeline。 | 增加受检查的递归常量求值（add/sub/mul/div/rem/min/max、index cast），处理除零、负 step、span 溢出并 fail-closed。 | 远端增量构建成功；constant-arithmetic loop-bound 回归通过，已修复。 |
| `LowerSqmma` 错误删除循环内 wait；异步累加器可能在下一迭代读取未提交结果。 | 对 loop-carried、下一次 SQMMA 消费的 wait 保守保留；仅对循环外同一基本块且 dominance/post-dominance 成立的链允许消除。 | SQMMA 回归约 140 passed/4 skipped，split-N correctness 通过；已修复但会牺牲部分性能。 |
| `LowerPipe` 只按线性邻接匹配 pipe completion，sibling `scf.if` 下可能错误复用 phase。 | 引入 region/dominance/post-dominance 约束和 path-aware matcher；无法证明安全时 fail-closed。 | pipe/WS 回归通过；sibling-if 仍不自动合并，属于有意保守限制。 |
| allocator 失败以裸 LLVM 错误退出，难以被 FlagGems 隔离。 | mthreads backend 将寄存器分配 abort 转换为 `PTXASError`；未知 LLVM 错误仍抛 RuntimeError；增加 llc 选项去重和架构归一化。 | benchmark 可按 shape 捕获 allocator 失败并继续；根本 spill 能力仍缺失。对 `-internal-opt-prera-interval-split`、`-internal-opt-postra-interval-split`、`-enable-deferred-spilling`、`-mtgpu-control-flow-mode=1` 等隐藏选项做过 A/B，仍会触发寄存器分配失败，说明当前精简 `llc` 的 allocator 不是简单开关即可补齐。 |
| TLE wait/release、grouped completion、warp-specialize 前置 barrier 生命周期不完整。 | 收紧 grouped completion 条件、修复 branch-local wait；撤回会导致 WS hang 的“保留所有前置 syncthreads”试验。 | 多数已验证路径稳定；WS/大 payload 仍可能 timeout/OOB，未宣称完全解决。 |
| 多 fragment SQMMA carrier 使用长 LLVM vector，寄存器 live range/allocator 压力过高；`scf.if`/`scf.for` cloning 还可能产生非法 terminator。 | 将 carrier 改为 literal struct（每 fragment 一个 field），用 `extractvalue/insertvalue` 替代长 vector；region cloning 显式复用/更新唯一 `scf.yield`，并恢复正确 insertion point；`LowerSqmma` 只对严格 loop-carried、branch-local 模式放行。 | 最新 targeted compile 已进入 carrier lowering，SQMMA 回归约 139--140 passed、4 skipped（剩余失败为既有测试 typo/assertion，而非 terminator 崩溃）。该能力仍只覆盖窄模式；带复杂嵌套 region 或额外 use 的 carrier 继续 fail-closed，尚未接入 MM 生产 dispatch。 |
| `scf.if` 分支直接 yield 异步 SQMMA tensor 时，lowering 可能在删除 TLE op 后留下悬空 SSA，或让未完成结果逃出 region。 | 在 branch terminator 前物化 branch-local `sqmma_wait`；对 loop-carried 路径做显式 Pack -> native carrier loop -> wait -> Unpack，并拒绝无法证明的额外 use。 | `test_tle_sqmma.py` 的 direct-loop/if-carried 回归已解除 xfail 并通过；复杂 pipe + sibling-if 仍不放行，避免 correctness 回归。 |
| TME landing/descriptor 推导直接使用逻辑 shape，staged `MemDesc` 的物理 rank/shape 不一致时会生成错误 swizzle 或 layout。 | SQMMA/TME lowering 改用 `getMemDescPhysicalShape` 和 matrix-view 约束推导 landing order；staged index 的类型按去除 stage 维度重新构造。 | 定向 TLE/SQMMA 编译回归通过；非 direct MM 曾出现 layout correctness 回归，后续通过保守 wait 保留和生产 guard 隔离，不能把该改动表述为所有 MM layout 已完全解决。 |
| 编译器前端/缓存错误曾泄露 `StopIteration`、复用错误的旧 kernel cache，导致真实后端问题与 Python 诊断混淆。 | 为 mthreads frontend 参数/返回值不一致增加带源码位置的 `CompilationError`；benchmark 为每个 shape/dtype 使用独立进程和独立 Triton cache。 | 前端单测和隔离 benchmark 已通过；这改善了错误定位和可重复性，但不改变 allocator 或硬件执行性能。 |
| 编译器无法表达 muDNN 的非二次幂 tile、cluster 和 24-warp 拓扑。 | 已通过库头文件、符号和机器码确认缺口；尚未在本轮实现完整非二次幂 layout/SQMMA lowering。 | 这是当前 Triton 层面无法绕过的主要性能上限，需要 compiler/ABI 扩展。 |

### 已修复项的判定

本表中的“已修复”只表示对应 IR/编译器回归不再触发原错误，并不等于所有 MM 形状都能获得性能收益。例如，保留 loop-carried wait 修复了异步累加器的正确性，但由于当前 SQMMA 只有 wait-all ABI，仍会增加同步开销；carrier 的 struct/region 修复消除了 terminator 崩溃，但复杂 pipe 生命周期仍被 capability guard 拒绝。对于 allocator，错误分类和候选隔离已经修复，spill/interval splitting 本身没有修复，也没有通过隐藏 `llc` 选项验证出可替代实现。

## 结论与后续边界

本轮在 `FLAGTREE_AABS=0` 下完成了两份 YAML 的隔离全量测量并停止继续扩展算子。397B 的成功 BF16 shape 平均已超过 Torch，FP16 成功 shape 平均约 0.857x；35B 受 allocator 失败和小 M 固定开销影响更大。要把“全部 shape >=95%”作为可交付目标，首要依赖是 MTGPU LLVM allocator 的 spill/interval splitting，以及可表达非二次幂 tile、cluster、异步 wait token 的编译器/ABI；仅在当前 Triton API 上扩大 autotune 表无法解决这些结构性限制。

说明：表格中的全量比例对应最终隔离扫描时的生产 dispatch。之后验证的 small-M（`M<=16`、`N` 256 对齐）路径已在远端完成 correctness，并在代表性 shape 上达到或超过 Torch，但尚未重新纳入两份 YAML 的全量统计；SQMMA carrier 修复同样只完成定向回归，未进入生产 MM guard。因此不能把这些后续局部收益外推到上表的全量达标率。
