# MegaMoe 关键路径 tracing

本目录提供 Ascend 950 MegaMoe MTE A8W8 wave 与 A8W4 specialized 路径的诊断态关键路径打点、`moe_test` 适配器、
严格解码和中文 HTML。默认打点粒度已经从“一批专家一个长框”细化为“本物理核
实际参与的专家”，并把同一 wave 内相邻专家区间之间的间隙单独画出。它只支持
这两条已审计路径，不是通用 MegaMoe profiler。

## 基线与适用路径

- 远端默认主线：`upstream/master`（`upstream/HEAD -> upstream/master`）。
- 建立独立 worktree 时记录的基线：
  `4b3d39658e1d4f5d1f5b60acedfe2df8201e08b2`。
- 最终集成并验证的最新主线：
  `27455755e`（分支中的主线 merge commit 为 `b25081aa4`）。
- 分支：`codex/mega-moe-critical-path-trace-20260803`。
- 独立 worktree：
  `/home/mace/workspace/ops-transformer_megaMoeOpt_trace_20260803`。
- `moe_test` 审计快照：commit
  `90319bbd05993d2fbe7674d693395d5520732553`；`mega_moe.py` SHA-256
  `35f600ca443782fd8f210a33af879af651a9dea011f32afa51191ae2bf9ee43e`；
  `graph.py` SHA-256
  `f050d695c5d4bde708e71f501ee587268c05ccbfef934b61f95dcd0b7e1fa13f`。

当前 `mega_moe_apt.cpp` 仍按通信模式和 dtype 选择实现：MTE+A8W8 进入
`MegaMoeWave`，MTE 非 A8W8 进入 `MegaMoe`，URMA 进入 `MegaMoeLayered`。
适配器会用设备 header 的 path/config hash、dtype key、shape、核数和量化配置
确认实际运行命中了所审计路径。0717 分支只作为历史案例；本实现没有复用旧行号、
旧时间点或旧输出布局。

## 两级诊断宏与 ABI

```cpp
#define MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE 1
#define MEGA_MOE_ENABLE_FINE_GRAIN_CRITICAL_PATH_TRACE 1
```

- 两个宏默认都是 1：启用细粒度 protocol v7，schema
  `0x3eb1c6b3`，flags `0x3ff`。
- 仅将 `MEGA_MOE_ENABLE_FINE_GRAIN_CRITICAL_PATH_TRACE=0`：保留原来的一批专家/
  wave 粒度、时间点、桥接和 wire layout，输出 protocol v5，schema
  `0xd32d8a49`，flags `0xff`。
- 将 `MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE=0`：所有 tracing 设备逻辑消失，恢复
  生产行为；细粒度宏值不再产生作用。

v7 不再预留固定四槽。每条 record 的 word 数由运行时每卡 routed expert 数决定：
`record_words = 2 + 4 * moeExpertPerRank`。word 0/1 是 process begin/work end；
随后每个 expert index 固定占四个 word。AIC 将四 word 解释为 GMM1 begin/end、
GMM2 begin/end，AIV0 使用前两个 word 表示 SwiGLU，AIV1 在每个 wave 首 expert
index 写 Dispatch 与 Combine。未参与该专家的核保持 0，不再有 captured count、
truncation meta 或 `--allow-truncated` 需求。wave ID 由 header 中专家数、
`experts_per_wave` 和主线当前 `GetWaveExpertCount` 小尾波合并规则推导。

热路径不能使用动态 LocalTensor/栈数组，因此每个 role/core 先把
`[expertIndex, begin, end]` 写入 `WorkspaceInfo::workspaceSize` 后的 50 MiB tiling
保留区；有效工作结束后再按 expert index 整理进 y。设备按实际专家数检查 y 与
workspace 两种容量，adapter 还会在第一次算子执行之前做同公式的 Python 预检。

运行器会根据 raw header 自动选择 v7 细粒度规范或 v5 粗粒度规范。因此旧的 v5
包仍可用当前 host 脚本解析，不再出现“runtime v5、expected v7”的误报；但要获得
新的逐专家打点和间隙，必须重新编译并安装默认 v7 kernel。其他 version/schema
仍然 fail closed。

A8W4 specialized 路径使用独立的 protocol v1、schema `0x34dddbe0`、path id 2。
每条 record 为 `2 + 6 * moeExpertPerRank` words：AIC 记录 GMM1/GMM2，AIV0 记录两段
权重 prologue，AIV1 记录 Dispatch、SwiGLU 和 Combine。所有阶段 end 均带有直接或复用的完成依赖证明；
prologue 为诊断态增加 `MTE3_S`，SwiGLU 复用算子内部已有的 `MTE3_S`。

## 当前依赖与时间点

稳态因果链是：AIV1 Dispatch 发布输入，AIC GMM1 消费，AIV0 SwiGLU+MX quant
发布，AIC GMM2 直接写 GM 并发布 ready，AIV1 在全部 AIC ready fan-in 后
Combine；下一 wave 的 Dispatch 可以与当前 wave 的计算重叠。

| 时间点 | 文件与函数/符号 | 什么时候打、为什么打及主要扰动 |
|---|---|---|
| `process_begin_S` | `mega_moe_wave.h` / `MegaMoeWave::Process` | 各角色尚未发出工作时建立本 AI Block 的参考轴；一次 Scalar 时钟读取。 |
| `work_end_S` | `mega_moe_wave.h` / `MegaMoeWave::Process` | 角色本地有效工作和正常输出已结束、tracing rendezvous 尚未开始；把诊断落盘开销排除在外。 |
| `dispatch_begin_S` | `mega_moe_wave.h` / `DispatchExpert` | 当前 wave 首个专家即将执行 `SendCntCal`；每活跃 wave 一次条件检查和时钟。 |
| `dispatch_end_S` | `mega_moe_wave.h` / `DispatchExpert` | 当前 wave 最后一个专家的 token/scale/meta 写入和发布完成；增加一次 MTE3→Scalar bridge。 |
| `gmm1_expert_begin_S` | `mega_moe_impl.h` / `Detail::AicComputeGeneric` | 当前专家 readiness/ping-pong 等待已完成，首个 GMM1 MMAD 即将发出；每个被捕获专家增加一次 FIX→Scalar causal bridge 和时钟。 |
| `gmm1_expert_end_S` | `mega_moe_wave.h` / `ProcessGmm1Wave` | 当前专家本核 MMAD/NotifyVector 已全部发出；故意不增加逐专家 FIX drain，避免把可重叠尾部串行化。它是低扰动 issue 边界。 |
| `swiglu_expert_begin_S` | `block_epilogue_swiglu_mx_quant.h` / `BlockEpilogueSwigluMxQuant::operator()` | 当前专家首个非空 epilogue 的上游依赖已进入 Vector，SwiGLU 即将开始；每个被捕获专家一次 V→Scalar bridge 和时钟。 |
| `swiglu_expert_end_S` | `mega_moe_wave.h` / `ProcessGmm1Wave` | 当前专家全部 epilogue tile 返回，data/scale GM 写回完成且 ready 已发布；复用每 tile 现有 MTE3_S，每专家只增加一次条件检查和时钟。 |
| `gmm2_expert_begin_S` | `mega_moe_impl.h` / `Detail::AicComputeGeneric` | 当前专家的 SwiGLU-ready 已被 Scalar 观察，首个 GMM2 MMAD 即将发出；无新增跨流水 bridge。 |
| `gmm2_expert_end_S` | `mega_moe_wave.h` / `ProcessGmm2Wave` | 当前专家 FIX 写回完成且 ready 已发布；复用 `PublishGmm2Ready` 中现有 FIX_S，只增加时钟。 |
| `combine_begin_S` | `mega_moe_wave.h` / `ProcessCombineExperts` | 当前 wave 的完整 AIC ready fan-in 已观察，首个非空 Combine 即将开始；复用既有依赖。 |
| `combine_end_S` | `mega_moe_wave.h` / `ProcessCombineExperts` | ring drain 后远端 MTE3 send 完成；增加一次 MTE2→Scalar bridge和时钟。 |

完整的文件、函数/符号、唯一语义锚点、解析行号、边界等级、原因、性能关系、
依赖证明和潜在扰动见：

- v7 机器规范：[`trace-spec.json`](trace-spec.json)
- v7 计划/实现审计：[`audit-plan/`](audit-plan/) 与
  [`audit-implemented/`](audit-implemented/)
- v5 回退规范：[`trace-spec-coarse-v5.json`](trace-spec-coarse-v5.json)
- v5 计划/实现审计：[`audit-plan-coarse-v5/`](audit-plan-coarse-v5/) 与
  [`audit-implemented-coarse-v5/`](audit-implemented-coarse-v5/)
- A8W4 机器规范：[`trace-spec-a8w4.json`](trace-spec-a8w4.json)
- A8W4 实现审计：[`audit-implemented-a8w4/`](audit-implemented-a8w4/)

## HTML 如何表示“空等”

同一物理 AI Block 的 AIC、AIV0、AIV1 使用同一时钟域，HTML 将三者放在一组，
并以该组 AIC 的 `process_begin_S` 独立归零。不同 AI Block 可能存在时间戳固定
偏移，禁止横向比较不同组的绝对位置；rank 之间也不得相减。

细粒度 HTML 对 GMM1、SwiGLU、GMM2 分别画逐专家区间，并在同一 wave 的相邻
专家之间画橙色“推断间隙”：

- SwiGLU 间隙是前一专家实际 output/ready 完成到下一专家 causal begin，主要
  包含 `WaitForCube`/输入就绪等待和少量调度。
- GMM2 间隙是前一专家 FIX+ready 完成到下一专家 causal begin，可直接观察它
  是否如预期接近连续输出。
- GMM1 end 是 issue 边界，所以其间隙还包含前一专家的 FIX 尾部。

这些橙色区间是两个已证明边界之差，不是硬件“wait 指令自身耗时”。它可能同时
包含流水尾部、真实依赖等待和少量 Scalar 调度；HTML tooltip 和中文说明会明确
这一点。这样不会再把整批专家之间的等待全部塞进一个 83us 的 SwiGLU 长框，
但也不会把推断间隙伪称为纯空等。

## `y` 载体证明

Tracing 只在诊断宏打开时复用最终 BF16 `y`。热路径时间戳先写入正常 workspace
布局末尾的诊断保留区，不与 y 的正常生命周期冲突。第一次混合 AIC/AIV
rendezvous 保证所有正常 Unpermute MTE3 写回完成，此后 kernel 不再读取正常 `y`；
各角色才按小端 `uint32_t` 整理动态 record。第二次 rendezvous 等待所有 record，
AIC0 再写 header、执行 `PIPE_ALL`，最后发布 magic。

- dtype/shape：当前路径输出 BF16 `[bs, h]`，容量是
  `bs * h * sizeof(OutputType)`；适配器要求实际输出 dtype/shape 精确匹配。
- v7 布局：`30 + 3 * aicNum * (2 + 4 * moeExpertPerRank)` 个 word，不含固定
  专家上限。
- 用户 case `bs=24, h=6144, aicNum=36, moeExpertPerRank=8` 时，record 为
  34 words，最终 trace 需要 `14808` bytes；BF16 y 有 `294912` bytes。
- 同一 case 的热路径 workspace scratch 为
  `3 * 36 * 2 * 3 * 8 * 4 = 20736` bytes，远小于 tiling 固定追加且尚未被
  `WorkspaceInfo` 使用的 50 MiB reserve。合法最大 1024 专家也小于该 reserve。
- 36 个 AIC 的 v5 固定布局需要 7896 bytes。
- adapter 在算子执行前按当前脚本里的 `bs/h/aic_core_num/local_moe_expert_num/
  out_dtype` 做保守容量预检。实际记录数和 config hash 使用设备 header 中由 tiling
  写出的 AIC 数；脚本核数较大只会多预留空间，小于运行时核数则失败关闭。设备仍用
  64 位中间量独立检查 y 与 scratch，host 解码时再次核对 header、record 数和 raw 长度。
- 诊断态会覆盖正常输出，适配器在最终 NPU synchronize 后按原始字节保存最后
  一轮 `y`，并跳过精度比较。业务不得消费 tracing-on 的 `y`。
- tracing 全关时不复用 `y`；预处理 token 等价检查以最新集成主线为基线。

## 构建、运行与 case 修改

默认细粒度诊断包：

```bash
bash build.sh --pkg --soc=ascend950 --ops=mega_moe -j8
```

保留原 v5 批专家粒度：

```bash
bash build.sh --pkg --soc=ascend950 --ops=mega_moe \
  --ops-compile-options '-DMEGA_MOE_ENABLE_FINE_GRAIN_CRITICAL_PATH_TRACE=0' -j8
```

生产包必须显式关闭 tracing 并重新构建：

```bash
bash build.sh --pkg --soc=ascend950 --ops=mega_moe \
  --ops-compile-options '-DMEGA_MOE_ENABLE_CRITICAL_PATH_TRACE=0' -j8
```

运行 A8W4 采集时显式选择 profile；这会切换 Python 容量预检，并由设备 header 再确认实际命中的协议：

```bash
python3 mc2/mega_moe/tools/critical_path_trace/run_critical_path_trace.py \
  --trace-profile a8w4 \
  --artifact-dir /tmp/mega-moe-a8w4-trace \
  --set weight_dtype=4
```

安装本次 run 包并确认 Python 环境加载到它后，在两个源码仓之外建立运行目录：

```bash
mkdir -p /path/to/mega-moe-trace-workspace
cd /path/to/mega-moe-trace-workspace
python3 /path/to/ops-transformer/mc2/mega_moe/tools/critical_path_trace/run_critical_path_trace.py \
  --test-script /path/to/moe_test/mega_moe/mega_moe.py
```

命令会直接执行。默认运行 100 轮，仅保存既有最终同步之后的第 100 轮 `y`；轮间
不增加 D2H 或 host synchronize。可用 `--iterations N` 改轮数。它选择的是预热
后的最后一轮，不声称是 100 轮中的最优轮。

不传 `--artifact-dir` 时，产物写入命令执行目录下的
`artifacts/YYYYMMDD_HHMMSS/`。run/check 模式拒绝将 artifact 写进
ops-transformer 源码仓，避免旧 `rankN.raw` 与本次 rank 集合混入；指定一个新的
外部目录也可以。适配器和运行子进程的 cwd、Torch extension cache、golden cache
都重定向到本次 artifact，两个源码 checkout 不被改写。

case shape 可以直接改：建议复制一份 `moe_test`，直接编辑其中顶部的 `bs`、`h`、
`n`、`k`、专家数等，再加 `--allow-compatible-test-script`。该选项只放宽已人工
复核过的脚本 hash/commit，唯一 AST/文本锚点仍然严格检查；适配器仍强制
`is_performance=0` 和 `loop_acc_time=iterations`。少量临时参数也可用重复的
`--set NAME=VALUE`。

每个 rank 生成 `rankN.raw`、`rankN.runtime.json`、`rankN.normalized.json`、
contract/order 检查 JSON 和 `rankN.html`，并生成总览 `index.html`。已有产物可
只解析：

```bash
python3 /path/to/ops-transformer/mc2/mega_moe/tools/critical_path_trace/run_critical_path_trace.py \
  --decode-only --artifact-dir /path/to/trace-artifacts
```

## 已完成与仍需服务器完成的验证

本 worktree 已完成：v7/v5 机器规范校验、两套计划/实现锚点审计、Python 语法、
host 单测、当前 `moe_test` 适配检查、8 专家 v7/v5 合成 raw 的
contract/decode/order/中文 HTML 链路、动态容量 fail-closed、trace-off token 等价，以及 Ascend
950 默认细粒度、v5 回退、tracing 全关三种 opkernel 构建（每种 8 个变体）。
结构化结果见 [`validation-report.json`](validation-report.json)。

当前环境没有 NPU，因而没有声称完成真实多 rank 执行、真实 raw 捕获、实际设备
时间戳顺序/边界核验、真实 tracing 扰动测量或 trace-off 数值精度回归。必须在
目标服务器运行上述命令，确认所有 rank 的 contract/order 通过，并人工查看
SwiGLU/GMM1/GMM2 专家槽与橙色间隙后，硬件运行验证才算完成。
