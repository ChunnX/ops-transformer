# MegaMoe 关键路径 tracing（全 AI Block / 稀疏任务版）

本目录面向当前 token-group Wave 实现，提供设备打点、`moe_test` 运行适配、严格
协议校验和中文 HTML。业务基线是
`d60c90ed7310181b6398d83a3033ae6b2f08c6ed`（PR10302 + wave/interleave/sendIndex 移植分支）；
0803 打点分支仅用于理解历史语义，所有
文件、调用路径、输出 ABI 和解析规则均按当前源码重新审计。

## 这版解决什么问题

协议为 v13，schema ID 为 `0xdae1e878`，默认诊断开关为：

```cpp
#define MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE 1
```

v8 只记录物理 AI Block 0，而且按整段 Wave 画色块。若一个 Wave 中先后调用两个
Dispatch 或两个 SwiGLU helper，两次调用之间的 cursor 推进、依赖等待也会落入同一
个色块，容易把空档误读成工作耗时。

v13 在全 AI Block 稀疏任务记录基础上进一步改为：

- 记录全部物理 AI Block；每组固定按 `AIC / AIV0 / AIV1` 排列；
- 每个 AI Block 用本组 AIC 的 `process_begin_S` 独立归零；组内三角色可直接比较，
  不比较不同 AI Block 的绝对时间戳；
- 只在真正进入 production helper 时追加稀疏任务段；同一 Wave 内相邻 helper
  之间的空档保持白色；
- 每段携带 `wave` 和 `expert`，HTML 悬浮框可直接核验来源；
- 无 owned tile/token 的核不产生对应 GMM/SwiGLU/Combine 色块。
- Dispatch 前固定记录输入 UB 初始化、量化、compact route index/count、workspace reset、单卡同步、跨 rank
  同步和专家 buffer 初始化；
- GMM1/GMM2 的结束点均经过 tracing-only `FIX_S`，显示实际 FIX 完成而不是 helper 发出。
- AIC 的 GMM1 空洞额外分成 Scalar 调度/无 owner 跳过、Dispatch-ready
  轮询、SwiGLU ping-pong 回压三类非 MMAD 区间；连续无 owner 专家聚合记录，
  不逐专家写 trace。
- AIV1 将两个连续有效 Dispatch task 之间的 count 查询、rolling owner 规划、Wave
  cursor 推进和相邻预取调用衔接聚合为一段；最后一个 task 后的尾段丢弃，不跨入
  GMM2/Combine。

| 角色 | 稀疏任务段 | begin | end |
|---|---|---|---|
| AIC | GMM1 problem | 首个 owned tile 的 ready/wait 返回后 | 本 problem 的 FIX 经 FIX_S 完成后 |
| AIC | GMM1 Scalar 调度 | 开始扫描/准备下一个 problem | 找到 owned tile 或当前 Wave 扫描结束 |
| AIC | GMM1 Dispatch 等待 | owned tile 即将轮询 ready | ready 值已观察到 |
| AIC | GMM1 SwiGLU 回压 | 生产路径即将 `WaitForVector` | ping-pong 槽位已归还 |
| AIC | GMM2 problem | 首个 owned tile 观察到 SwiGLU-ready 后 | 本 problem 的 FIX 经 FIX_S 完成后 |
| AIV0 | SwiGLU/MX problem | 首个 owned epilogue tile ready 后 | helper 返回 |
| AIV1 | 有效 Dispatch expert task | 本核获得非空连续行区间后、`DispatchWaveExpertRows` 前 | helper 自带 copy-ring drain 和 ready 发布后 |
| AIV1 | Dispatch 间 Scalar 调度 | 前一个有效 Dispatch 的 trace 写入后 | 下一有效专家已完成 count/owner 规划且即将建立 Dispatch task |
| AIV0/AIV1 | Combine 256-token group task | 当前 expert-local group 的 GMM2 N tile 全部 ready 后 | 本核在该 group 的远端 send 经 MTE3_S 桥接到 Scalar 后 |

GMM1/GMM2 为了量化真实工作与 problem 间空洞，在每个本核活跃 problem 末尾显式
增加 `FIX_S`。它会抑制诊断态跨 problem 的 FIX 尾部重叠，因此 tracing-on 的算子总
耗时不能当作生产性能；关闭 tracing 后没有额外同步。Dispatch 复用生产 helper 已有
的 MTE3 完成语义；Combine 在每个本核活跃 256-token group 末尾显式增加 `MTE3_S`。
因此 tracing-on 的 Combine group 之间会被刻意串行化，用来看分项时间，不用来代表生产态端到端性能。完整的打点原因、
依赖证明和潜在扰动见：

- [`trace-spec.json`](trace-spec.json)：机器可读 trace spec；
- [`audit-plan/`](audit-plan/)：设计位置说明；
- [`audit-implemented/`](audit-implemented/)：从实际 marker 反查生成的实现审计。

## 输出 ABI 与容量

Tracing 仅在诊断态复用最终 BF16 `y[bs,h]` 的字节。正常 Unpermute 完成后，所有
角色先同步保护 `y` 生命周期；AIC0 再汇总独占 workspace scratch 中的计数、规划
紧凑 offset，各角色并行复制自己的 payload；最后由 AIC0 发布 header，magic 最后
写入。因此 tracing-on 的 `y` 是诊断产物，不能再用于精度比较或业务消费。

每个角色/核 record 是 26 个 `uint32`：

```text
[process_begin, work_end,
 input_buffer_init_begin, input_buffer_init_end,
 input_quantize_begin, input_quantize_end,
 route_index_begin, route_index_end,
 workspace_reset_begin, workspace_reset_end,
 local_prepare_sync_begin, local_prepare_sync_end,
 cross_rank_sync_begin, cross_rank_sync_end,
 expert_buffer_init_begin, expert_buffer_init_end,
 stage0_offset, stage0_count,
 stage1_offset, stage1_count,
 stage2_offset, stage2_count,
 stage3_offset, stage3_count,
 stage4_offset, stage4_count]
```

每个稀疏 entry 是：

```text
[wave_index, expert_index, begin_cycle, end_cycle]
```

Header `flags` 的 bit13 记录构建时的
`MEGA_MOE_A8W8_DISPATCH_GROUP_READY`：0 表示专家段完成后批量发布，1 表示逐
256-token group 发布。脚本会校验该位，防止将两种构建的结果混在一起。

令 `A=aic_core_num`、`E=local_moe_expert_num`、`P=ep_world_size`：

```text
max_rows        = bs × P × min(top_k, E)
max_m_groups    = min(max_rows,E) + floor((max_rows-min(max_rows,E))/256)
segment_need    = max(E, max_m_groups)
record_count    = 3A
descriptor_y    = 4 × (30 + record_count × 26)
scratch_record  = 22 + 5 × 4 × segment_capacity  // 固定点、计数和五个 stage
```

Host 在运行前证明 descriptor 和 50 MiB 预留 scratch 足够；设备结束时再用实际稀疏
计数检查紧凑 payload 是否装得下 `y`。任何 scratch/y 超限都会设置 `truncated`，
默认解析器失败关闭，不会把部分采集当完整结果。

## 运行

先构建并安装 tracing-on 包，然后在源码仓之外执行：

```bash
mkdir -p /path/to/mega-moe-trace-workspace
cd /path/to/mega-moe-trace-workspace
python3 /path/to/ops-transformer/mc2/mega_moe/tools/critical_path_trace/run_critical_path_trace.py \
  --test-script /path/to/moe_test/mega_moe/mega_moe.py \
  --dispatch-group-ready 0
```

`--dispatch-group-ready` 必须与已安装 kernel 的编译宏一致；逐 group 版构建后改为
`--dispatch-group-ready 1`。

命令会直接运行 `moe_test`。默认连续执行 100 轮，只在脚本原有最终同步后保存最后
一轮 `y`，轮间不新增 D2H/synchronize；可用 `--iterations N` 修改。

不传 `--artifact-dir` 时，输出在当前执行目录的
`artifacts/YYYYMMDD_HHMMSS/`。run/check 模式拒绝写进 ops-transformer 源码仓，
以免运行缓存污染仓库或旧 `rankN.raw` 混入。每个 rank 生成 raw、runtime metadata、
协议/顺序检查 JSON、normalized JSON 和 HTML，并生成总览 `index.html`。

case 参数可直接修改 `moe_test/mega_moe.py`。若不是审计快照，先复核调用 ABI 和
适配锚点，再显式加 `--allow-compatible-test-script`；临时参数也可重复使用
`--set NAME=VALUE`。适配器始终强制 `is_performance=0`，并把 `loop_acc_time` 设为
`--iterations`。

只解析已有产物：

```bash
python3 /path/to/run_critical_path_trace.py \
  --decode-only --artifact-dir /path/to/trace-artifacts
```

若报告 version/schema/config hash 不一致，说明安装的 kernel、解析脚本或 raw 不是
同一 ABI，必须换成匹配构建和本次新目录，不能修改 expected 常量绕过。

## 压缩 normalized trace 便于传输

完整 normalized JSON 包含大量重复字段和缩进。分析时可生成默认 gzip 压缩的最小
版本：

```bash
python3 /path/to/pack_normalized_trace.py \
  /path/to/trace-artifacts/rank0.normalized.json
```

默认输出同目录的 `rank0.compact.json.gz`。它保留协议/schema、shape、Wave 大小、
角色核数、32 位时钟信息、每个核的固定前处理
`[阶段类型, begin, end]` 以及稀疏任务
`[任务类型, wave, expert, begin, end]`；删除 HTML 文案、审计证据、文件哈希、label、
counter 等分析不需要的重复信息。不能只保留裸时间戳，否则无法判断时间戳属于哪个
核、Wave、专家，也无法正确处理时钟回绕。

若希望不删除任何 normalized 字段、只做无损 minify+gzip，可加 `--lossless`；若需
查看精简后的明文 JSON，可加 `--plain`。

## 关闭 tracing 与验证边界

生产包以 `MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE=0` 构建。关闭后，预处理 token 与
业务基线完全一致：无诊断参数、scratch、时钟、额外同步或 `y` 覆盖；可运行
`verify_trace_off.py` 复核。

仓内可验证源码审计、协议、容量、raw 解码、时间顺序/边界、中文 HTML，以及
tracing-on 编译。真实多 rank NPU 执行、设备时间戳和实际扰动必须在目标服务器
完成；host 合成 fixture 通过不等于设备运行成功。
