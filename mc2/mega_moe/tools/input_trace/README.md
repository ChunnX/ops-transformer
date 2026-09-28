# MegaMoe 输入准备与共享阶段 trace

基线：`f592e9f36`。共享直接使用输入时，AIV0 做 reset，AIV1 做 pack；AIC 提前搬入共享输入。
采集范围：输入准备、共享 GMM1/激活、MoE 入口和 count 表检查。继续执行完整算子。

## 构建

修改 `op_kernel/arch35/common/mega_moe_input_trace_config.h` 中 `MEGA_MOE_INPUT_TRACE`：

- `0`：关闭（默认）。不预留 trace workspace，不写记录，不新增同步。
- `1`：记录 Scalar 调用/下发区间，以及原代码已有的 Scalar 完成边界。先用这一档。
- `2`：在模式 1 基础上，在共享阶段函数返回后额外执行本核 `PIPE_ALL`，记录完成上界。
  这会减少 Scalar 超前下发，可能改变共享与 MoE 的重叠，只用于辅助验证。

**Host tiling 与 kernel 必须使用同一配置重新构建**；涉及工作区大小计算的 torch extension 也应重编。
不要只给 device 编译命令加宏。WSL 构建使用 `-j 8`。
启用 trace 时，即使 `MEGA_MOE_PROFILE_INPUT=1`，也会执行完整算子。
只接入 arch35 MTE Wave（A8W8/A8W4/A4W4），URMA 不生成此 trace。

这是诊断构建：正常计算结束后会用 trace 覆盖 `y`，关闭精度检查；不要用于下游模型计算。
恢复正式性能/精度测试时将开关设回 0 并重编。

## 存储与扰动

不新增 UB。MTE workspace 末尾预留每个逻辑 core 三条独占 record：

```
scratchBytes = aicNum * 3 * (8 + 32 * 9) * 4
```

28 core 为 99,456 bytes，36 core 为 127,872 bytes。
输出需要额外容纳 16-word header。bs=72、H=5120、BF16 输出足以保存全部 32 个事件/record。
小输出按实际容量减少 eventCapacity；容量不足时置 truncated，解析工具报告截断。
若连一个事件/record 都放不下，不写 trace，解析器拒绝缺少 magic 的输出。不会扩大业务输出 shape。

每个事件写 9 个 uint32 到 GM，有 Scalar 与 GM 写开销。
mode 1 不增加阶段内同步；结束发布增加本核 PIPE_ALL 和两次全角色 SyncAll，发布开销在 Process end 之后。
所有角色完成正常输出后再复制记录，AIC0 最后发布 header/magic。
输出的 Process 时长包含打点开销和末尾本核排空；不能替代无打点端到端性能。

## 原始数据导出

每个 rank 选定同一轮，在算子完成后导出第一个输出 tensor `y` 的原始字节：

```python
from pathlib import Path
import torch

out_dir = Path(trace_dir)  # 显式选择目录，建议按 build/case/iteration 分层
out_dir.mkdir(parents=True, exist_ok=True)
y.detach().contiguous().cpu().view(torch.uint8).numpy().tofile(out_dir / f"rank{rank}.raw")
```

不要先转 float/int32。使用新输出分配并确认执行成功，避免把失败运行后残留的旧 trace 当成本轮结果。
保留原测试脚本的 rank 间协调；本工具不修改外部测试脚本。

## 解析

```bash
python3 trace_tool.py /path/to/case/iteration --output-dir /path/to/report
python3 trace_tool.py /path/to/rank0.raw --output-dir /path/to/report-rank0
python3 trace_tool.py /path/to/rank0.raw /path/to/rank1.raw --output-dir /path/to/report
```

默认显示 cycles；确认本机 SYS_CNT 频率后，用 `--cycles-per-us <数值>` 换算。不要套用处理器主频。
每个 rank 输出 `trace.json`、`events.csv`、`timeline.svg`、自包含 `index.html`。
多 rank 有汇总入口，各 rank 保持各自时钟轴，不强行对齐。

## 如何读图

1. 比较基线和优化版同一输入、路由与相同打点模式的多轮采集。
2. `SharedGmm1ActivationSubmit` 是调用区间，包含内部等待，结束时异步任务可能仍未完成。
   AIC 执行 GMM1；AIV0 在 Generic 执行激活，在 A8W4 执行权重转换；AIV1 仅在 A8W4 执行激活。
3. `SharedCompletionFence` 只在 mode 2 出现，结束是本核流水完成上界。其耗时不是纯 reset 等待。
4. `MoeEntry` 是 Scalar 即将调用 MoE 阶段的时刻，不是实际 GMM 首次搬入。
5. `CountTableCheckWait` 返回已有 V_S 和 `pending.GetValue()==0` 保证快照就绪；区间包含搬运、VF、检查及退避。
6. `CountTablePrepareComplete` 还包含 cumsum、GM 持久化及 countTableReady 发布。
7. `MoeCountReadyWait` 是 AIC/AIV0 读取 countTableReady 的现有 Scalar 轮询，返回说明元数据已发布，
   不表示对应 token 数据已 dispatch 完成。
8. `InputResetScalarWait` 复用已有 MTE3_S 快照，包含 reset 及快照前本核 MTE3 工作。
   `InputPrepareReturn` 不能当作 count DMA 完成。
9. `*Submit` 一律只代表 Scalar 组织/下发区间。未拆分的等待仍在区间里，不显示成纯计算。

先看共享调用返回是否提前、MoE 入口是否提前、count 表等待是否增加。
若需要证明共享流水真正完成的时间，另采 mode 2 并核对打点扰动；不要用 mode 1 的返回点推断完成。

## 协议

小端 uint32。header 16 words：magic `0x4D495452`、version 1、headerWords、recordWords、eventWords、
recordCount、coreCount、eventCapacity、bs、h、worldSize、traceMode、sharedMode、moeMode、reserved×2。
record 按 AIC / AIV0 / AIV1 分组，每组按 core 排列。
record header 8 words：role、core、eventCount、truncated、processBegin、processEnd、reserved×2。
event 9 words：eventType、iteration、unit、rangeStart、rangeCount、workCount、parentEvent、begin、end。
本轮 iteration/unit 为 0；范围按事件对应 token、expert 或 count slot。
记录 32-bit SYS_CNT 低位，解析器支持一次回绕；单次记录跨度必须小于 2^31 cycles。
版本不匹配直接拒绝。

## 验证

```bash
python3 -m unittest discover -s . -p 'test_*.py' -v
```

合成测试覆盖布局、越界、截断、父等待区间、cycle 回绕、单/多 rank 输出及报告 JavaScript 语法。
首次上机需使用小用例验证事件因果关系，再跑实际对照。合成测试不能证明设备同步语义或实际性能。
