# MegaMoe 关键路径 tracing

本目录提供 Ascend 950 `MegaMoeWave` 的诊断态关键路径打点、输出协议、当前
`moe_test` 适配器、严格解码与人工可读 HTML。它只支持当前 arch35 MTE A8W8
路径；不是通用 MegaMoe profiler。

## 审计基线与适用路径

- 远端默认主线：`upstream/master`（`upstream/HEAD` 指向 `refs/heads/master`）。
- 实施基线：`4b3d39658e1d4f5d1f5b60acedfe2df8201e08b2`。
- 分支：`codex/mega-moe-critical-path-trace-20260803`。
- 独立 worktree：`/home/mace/workspace/ops-transformer_megaMoeOpt_trace_20260803`。
- 运行脚本审计快照：`moe_test` commit
  `90319bbd05993d2fbe7674d693395d5520732553`；`mega_moe.py` SHA-256
  `35f600ca443782fd8f210a33af879af651a9dea011f32afa51191ae2bf9ee43e`，
  `graph.py` SHA-256
  `f050d695c5d4bde708e71f501ee587268c05ccbfef934b61f95dcd0b7e1fa13f`。

当前 `mega_moe_apt.cpp` 根据通信模式和 dtype 选择实现：MTE+A8W8 进入
`MegaMoeWave`，MTE 非 A8W8 进入 `MegaMoe`，URMA 进入 `MegaMoeLayered`。
被审计脚本的默认配置是 `graph_type=0, quant_mode=4, comm_quant_mode=0,
bs=1, h=4096, n=1024, k=6, local_moe_expert_num=4,
shared_expert_num=0, world_size=2`，输入、SwiGLU 与权重均为 E4M3，因此命中
MTE A8W8 `MegaMoeWave`。设备 header 的 path/config hash、dtype key、shape、
核数和量化配置不一致时，解析器会拒绝结果。

0717 分支仅用于识别历史上出现过的边界和输出布局错误；本实现的符号、语义
锚点、时间点、协议和脚本插入位置全部由上述主线及当前 `moe_test` 重新推导，
没有沿用旧行号或旧 ABI。

## 当前并行依赖与时间点

稳态波次的因果链是：AIV1 Dispatch(B/B+1) 发布输入，AIC GMM1(B) 消费，
AIV0 SwiGLU+MX quant(B) 发布，AIC GMM2(B) 直接写 GM 并发布 ready，AIV1 在
全部 AIC ready fan-in 后 Combine(B)；同时下一波 Dispatch 可与当前波的
GMM1/SwiGLU/GMM2/Combine 重叠。因此时间点落在实际生产/观察边界，而不是
外层循环或函数返回的近似位置。

| 时间点 | 角色与符号 | 语义、证明关系与主要扰动 |
|---|---|---|
| `process_begin_S` | 全角色，`MegaMoeWave::Process` | 进入当前实现且尚未发出工作；给出同一 AI Block 内的公共参考轴。一次 Scalar 时钟读取。 |
| `work_end_S` | 全角色，`MegaMoeWave::Process` | 角色本地有效工作已完成，且在诊断 rendezvous 前；区分真实尾部与落盘开销。一次时钟读取。 |
| `dispatch_begin_S` | AIV1，`DispatchExpert` | 当前波首个专家即将执行 `SendCntCal`；覆盖真实预取窗口。每个活跃波一次条件检查和时钟。 |
| `dispatch_end_S` | AIV1，`DispatchExpert` | 末个专家的 token/scale/meta GM 写入与发布已完成；可与 GMM1 ready 观察比较。增加一次 MTE3→Scalar bridge 和时钟。 |
| `gmm1_begin_S` | AIC，`Detail::AicComputeGeneric` | Dispatch readiness 与 ping-pong 依赖已满足，首个 MMAD 即将发出；把等待从 GMM1 计算中分离。增加一次 FIX→Scalar bridge 和时钟。 |
| `gmm1_end_S` | AIC，`ProcessRoutedExpertWaves` | 本核当前波全部 GMM1 FIX 工作实际完成。增加一次 FIX→Scalar bridge 和时钟。 |
| `swiglu_begin_S` | AIV0，`BlockEpilogueSwigluMxQuant::operator()` | 公共 epilogue 入口，GMM1/输入依赖已进入 Vector，SwiGLU 即将开始。增加一次 Vector→Scalar bridge 和时钟。 |
| `swiglu_end_S` | AIV0，`ProcessRoutedExpertWaves` | 本核最后一个 epilogue 的 GM 输出完成；证明与 GMM2/Dispatch/Combine 的重叠。复用现有 MTE3_S，仅增加时钟。 |
| `gmm2_begin_S` | AIC，`Detail::AicComputeGeneric` | 已观察到 SwiGLU ready，首个 GMM2 MMAD 即将发出；发布到观察的等待不计入 GMM2。条件检查和时钟。 |
| `gmm2_end_S` | AIC，`ProcessRoutedExpertWaves` | 本核 GMM2 FIX 写入完成且最终 ready 已发布；与 AIV1 的观察时刻分开。复用现有 FIX_S，仅增加时钟。 |
| `combine_begin_S` | AIV1，`ProcessCombineExperts` | `WaitGmm2Ready` 已观察完整 AIC fan-in，首个非空 Combine 即将开始。复用现有 MTE2_V/V_S，仅增加时钟。 |
| `combine_end_S` | AIV1，`ProcessCombineExperts` | ring drain 后所有远端 MTE3 send 实际完成。增加一次 MTE2→Scalar bridge 和时钟。 |

完整的文件、函数/符号、稳定语义锚点、解析出的行号、边界等级、打点原因、
依赖证明、性能关系、潜在扰动及上下文片段见
[`audit-implemented/trace_point_audit.md`](audit-implemented/trace_point_audit.md)。机器可读
规范是 [`trace-spec.json`](trace-spec.json)，计划态与实现态 audit 分别在
`audit-plan/` 和 `audit-implemented/`。所有六个 span 都由成对边界组成；没有
用缺失值补造跨度。

时间戳是 `AscendC::GetSystemCycle()` 的低 32 位 cycle。同一物理 AI Block
（相同 paired `blockIdx`）内的 AIC/AIV0/AIV1 共用时钟域，可以直接对齐；不同
AI Block 可能存在时钟偏移，不得比较原始绝对时间戳，rank 之间同样不得相减。
HTML 按 AI Block 分组，每组固定排列 AIC、AIV0、AIV1，并以本组 AIC 的
`process_begin_S` 独立归零。实际采样为零时重映射为 1，零专用于“该核/该波无
工作”。顺序检查采用 32 位环绕差，并拒绝大于等于半周期的歧义区间。

## `y` 载体证明

Tracing 仅在编译宏 `MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE=1` 时存在，当前诊断
分支默认值为 1。诊断态在最终 `Unpermute` 的 MTE3_V 完成后执行混合 AIC/AIV rendezvous，
此时正常 BF16 `y` 已不再被 kernel 消费，才把 `y` 重用为小端 `uint32` 原始
记录。每个角色写唯一 role-major record；第二次混合 rendezvous 后 AIC0 写
header 字段、`PIPE_ALL` 排序，并最后发布 magic。

- dtype/shape：当前路径输出是 BF16 `[bs, h]`，载体字节数为
  `bs * h * sizeof(OutputType)`；runner 要求 `torch.bfloat16` 和精确 shape。
- 对齐：以 `y` 的 GM 基地址按 `uint32` 写入，长度按 `floor(bytes / 4)` 计。
- 布局：30 个 header word；每个物理 AIC、AIV0、AIV1 各一个 18-word record；
  需求为 `(30 + 3 * aicNum * 18) * 4` bytes。
- 默认例：`aicNum=36` 时需求 `7896` bytes；`[1,4096]` BF16 提供 `8192`
  bytes，余量 `296` bytes。
- 容量：设备使用 64 位中间量先检查容量；不足时不采样、不同步、不写 record/
  header/magic。解析器同时核对 raw 长度、header `carrier_words`、record 数量与
  runtime metadata，不能把未写入的正常 `y` 误判成 trace。
- 生命周期：诊断态会覆盖用户输出，所以 runner 在最终 NPU synchronize 后、
  mask/精度比较前按原始字节保存 `y`，并明确跳过精度比较。正常业务不得消费
  tracing 打开的 `y`。
- 捕获容量：最多四个 routed-expert wave；超出时 header 置 `truncated=1`，
  默认解析失败，只有人工确认接受部分分析时才使用 `--allow-truncated`。

## 服务器构建、运行与解析

从干净构建目录生成 tracing 算子包：

```bash
bash build.sh --pkg --soc=ascend950 --ops=mega_moe -j8
```

不传该宏即使用默认 tracing-on；也可显式传
`--ops-compile-options '-DMEGA_MOE_ENABLE_CRITICAL_PATH_TRACE=1'`。

按仓库安装指南安装 `build_out/` 下生成的 run 包；若使用自定义安装路径，还需
source 对应 vendor 的 `bin/set_env.bash`。确认当前 Python 环境加载的是这次
安装的 tracing 包后，在源码仓外创建独立运行目录，并通过绝对路径调用适配器：

```bash
mkdir -p /path/to/mega-moe-trace-workspace
cd /path/to/mega-moe-trace-workspace
python3 /path/to/ops-transformer/mc2/mega_moe/tools/critical_path_trace/run_critical_path_trace.py \
  --test-script /path/to/moe_test/mega_moe/mega_moe.py
```

不传 `--artifact-dir` 时，产物默认写入当前工作目录的
`artifacts/YYYYMMDD_HHMMSS/`，而不是写回 `ops-transformer` 源码仓。run/check
模式会拒绝任何位于源码仓内的 artifact 路径；`--decode-only` 仍可读取仓内已有
fixture。适配器自身和子进程都禁用 Python bytecode 落盘，运行子进程的 cwd 也
设为本次 artifact 目录；`moe_test` 的 Torch extension cache 和 golden cache
也重定向到 artifact 下的 `runtime-cache/`，两个源码 checkout 都只读。

适配器不会修改 `moe_test` checkout：它验证 commit、`mega_moe.py`/`graph.py`
哈希和唯一 AST/文本锚点，在 artifact 目录生成可直接运行的临时脚本，并强制
`is_performance=0`。默认连续执行 100 轮 MegaMoe，不在轮间执行 host
同步或 D2H；第 100 轮完成后的既有最终同步之后，只保存最后一轮的 `y`。可用
`--iterations N` 调整执行轮数。runtime metadata 明确记录执行轮数、采集轮次和
`last_iteration_only` 策略；它表示预热稳定后的最后一轮，不声称是 N 轮最优值。
每个 rank 输出 `rankN.raw` 和 `rankN.runtime.json`，随后严格校验
header/config/hash/容量/截断/时间顺序，生成 `rankN.normalized.json`、检查 JSON、
`rankN.html` 及总览 `index.html`。HTML 的打点位置、打点时机、打点原因、依赖
证明和潜在扰动均使用中文说明。

若服务器上的脚本不是上述精确快照，先人工复核当前 ABI 和插入锚点，再显式加
`--allow-compatible-test-script`；锚点不唯一或缺失仍会失败。可用重复的
`--set NAME=VALUE` 修改白名单内的 case 参数。已有 raw 与 metadata 可用：

```bash
python3 mc2/mega_moe/tools/critical_path_trace/run_critical_path_trace.py \
  --decode-only --artifact-dir /path/to/trace-artifacts
```

关闭 tracing 的生产包必须显式传 0 重新构建并安装：

```bash
bash build.sh --pkg --soc=ascend950 --ops=mega_moe \
  --ops-compile-options '-DMEGA_MOE_ENABLE_CRITICAL_PATH_TRACE=0' -j8
```

当前诊断分支默认宏为 1；显式置 0 后，修改的三个生产 device 文件在剥离
trace-only 预处理块后与主线基线 token 完全相同。不要把默认构建的 tracing 包
留作生产包。

## 本地验证与边界

在本 worktree 已完成以下真实检查：机器规范校验、计划/实现位置 audit、Python
语法检查、5 个 host 单测、当前 `moe_test` fail-closed 适配检查、合成 raw 的
完整 decode/order/render 链路、trace-off token 等价检查，以及 Ascend 950
MegaMoe 的默认开启和显式关闭两组 opkernel 构建（每组 8 个变体）。合成 HTML
入口是 [`artifacts/synthetic/index.html`](artifacts/synthetic/index.html)，其中
明确标记为 host-only synthetic，不是设备结果。结构化结果见
[`validation-report.json`](validation-report.json)。

当前环境没有 `npu-smi`，Python 也没有 `torch`，所以**没有**声称完成真实 NPU
执行、真实多 rank raw 捕获、真实设备时间戳顺序/边界验证或 trace-off 数值精度
回归。必须在目标服务器按上节执行，只有每 rank contract/order 均通过且生成的
HTML 经人工核验后，设备运行验证才算完成；静态检查和合成 fixture 不能替代它。
