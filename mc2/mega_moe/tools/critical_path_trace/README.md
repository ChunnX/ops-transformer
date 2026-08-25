# MegaMoe A4W4 Wave 关键路径打点

这是一套用于 A4W4 Wave 性能分析的轻量诊断工具。协议 v4 将打点下沉到真正的 tile/chunk 计算调用，记录
Dispatch、GMM1、Activation、GMM2 prologue、GMM2 和 Combine 的精确计算区间。

本轮打点主要用于分析小 BS 下的三类问题：Wave 是否欠载、AIC/AIV0/AIV1 的阶段是否失衡，以及后级阶段是否因
等待 Dispatch、GMM1、Activation 或 GMM2 出现空泡。等待事件与计算事件分离，因此计算块不再包含地址更新、
调度和依赖等待。

## 1. 构建与运行

主流程和 Torch Extension 独立编译，因此分别保留一个同名手动开关。两者默认都关闭。

主流程在 `op_kernel/arch35/common/mega_moe_trace_config.h` 中修改：

```cpp
#define MEGA_MOE_ENABLE_A4W4_WAVE_TRACE 1
```

这一处同时控制 kernel、host InferShape 和 tiling。Torch Extension 在 `torch_extension/mega_moe.py`
中修改同名布尔量：

```python
MEGA_MOE_ENABLE_A4W4_WAVE_TRACE = True
```

该布尔量只控制 Torch Meta 输出 shape。C++ Extension 独立编译，还需要在
`torch_extension/csrc/mega_moe.cpp` 中手动修改同名宏：

```cpp
#define MEGA_MOE_ENABLE_A4W4_WAVE_TRACE 1
```

生成诊断包时三个开关必须同时打开：

- kernel 编译打点协议及 trace-only 完成同步；
- host 允许诊断输出的首维大于 `bs`；
- Torch Extension 为 `y` 分配至少 1 MiB，Meta 路径使用相同 shape。

正常构建中，kernel 不读 cycle、不增加 trace-only 同步、不写 trace；InferShape 和 tiling 严格要求
`y=[bs,h]`，Torch Extension 也只分配正常输出。修改 Extension 开关后需要重启 Python 进程并重新编译
Extension。

当前 A4W4 调度允许在专家内部按 token 切分 Wave，报告中的 `wave`、`expert`、`tokenStartInExpert` 和
`tokenCount` 可还原每个阶段实际处理的切片范围。做性能对比时应保持用例和 tiling 配置一致，只替换诊断包。

启用后，kernel 在正常计算完成后把 trace 覆盖到输出 `y`，因此该包只用于性能诊断，不能继续用 `y`
做精度比较。运行正常用例前应将三处开关都恢复为关闭，重新构建并重启 Python 进程。

运行目标用例后，在 Python 测试代码中将 `y` 的原始字节保存下来。需要保留 BF16 的原始位模式，不能先转换为
FP32：

```python
y.detach().cpu().view(torch.uint8).numpy().tofile("rank0.raw")
```

如果测试框架在算子返回后立即检查精度，应在精度检查之前保存；诊断包的输出本来就会因为承载 trace 而不同。

### 适配 `moe_test/mega_moe/mega_moe.py`

该脚本通过多个子进程分别运行各 rank。可在全局配置区增加：

```python
critical_path_trace_enable = True
critical_path_trace_output_dir = os.path.join(script_dir, "critical_path_trace_output")
```

再增加落盘函数：

```python
def dump_critical_path_trace(output_tensor, rank, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, f"rank{rank}.raw")
    output_bytes = output_tensor.detach().contiguous().cpu().view(torch.uint8).numpy()
    output_bytes.tofile(raw_path)
    print(f"[TRACE] rank {rank}: raw trace saved to {raw_path}", flush=True)
```

在 `run_cascade_npu()` 中找到 MegaMoe 运行结束后的同步：

```python
torch.npu.synchronize()
print(f'rank {rank} epid ... npu finished!')
```

紧接着、在 `if kernel_timing_enable:` 和 `is_mask` 对 `y` 切片之前加入：

```python
if critical_path_trace_enable:
    dump_critical_path_trace(
        y,
        rank,
        os.path.join(critical_path_trace_output_dir, current_case_name),
    )
```

每个 rank 会得到独立文件：

```text
critical_path_trace_output/<case-name>/rank0.raw
critical_path_trace_output/<case-name>/rank1.raw
...
```

这里应当先 `.cpu()`，再将 CPU tensor `view(torch.uint8)`，从而保留 BF16/FP16 输出的原始字节并避免 NumPy
直接处理 BF16 的兼容性问题。该位置已经执行过 `torch.npu.synchronize()`，所以落盘时 kernel 写回已完成。

## 2. 解析

推荐直接传入一个用例目录，工具会自动发现其中全部 `rank*.raw`，不需要逐 rank 执行：

```bash
python3 mc2/mega_moe/tools/critical_path_trace/trace_tool.py \
  critical_path_trace_output/<case-name>
```

默认生成：

```text
critical_path_trace_output/<case-name>/trace-report/
├── index.html              # 多 rank 汇总入口
├── run-summary.json
├── rank0/
│   ├── index.html          # rank0 交互报告
│   ├── trace.json
│   ├── events.csv
│   └── timeline.svg
├── rank1/
│   └── ...
└── ...
```

通常只需打开根目录下的 `trace-report/index.html`，再从汇总表进入各 rank 报告。

工具仍兼容单文件解析：

```bash
python3 mc2/mega_moe/tools/critical_path_trace/trace_tool.py rank0.raw
```

默认输出到原始文件旁边的 `rank0-trace/`。可显式指定路径：

```bash
python3 mc2/mega_moe/tools/critical_path_trace/trace_tool.py rank0.raw \
  --output-dir /tmp/mega-moe-trace/run-001
```

生成文件：

- `trace.json`：header、各角色/core 及完整事件元数据；
- `events.csv`：便于排序、聚合和计算每个 Wave 的耗时；
- `timeline.svg`：AIC/AIV0/AIV1 的阶段与等待区间，鼠标悬停可查看 Wave、专家和 token 范围。
- `index.html`：自包含交互报告，可选择单个或全部 AI Block，并按 Wave 筛选。全部 lane 使用统一的
  `SYS_CNT` cycle 原点；每个 role/core 只占一条泳道，依赖等待以全高灰色区间覆盖在对应阶段上。鼠标悬停
  或点击事件块可查看 begin/end cycle、父阶段和对应的代码打点边界。当前平台按 `1000 cycles = 1 μs` 换算。

交互报告还提供泳道概览。筛选、排序和前 N 限制均以 Block 为单位，同一 Block 的 AIC/AIV0/AIV1 始终绑定
显示；可按是否存在等待筛选，并按 Block 的 Process 总时长、事件覆盖时长、累计等待、等待占比、最后打点位置
或尾部空洞排序。时间线中每个 Block 的三条泳道使用同一背景框，左侧只展示角色及 Process/Wait 简写。点击图钉可将
整个 Block 或单条泳道固定到顶部对比区；固定项仍跟随 Wave 筛选，但不受 AI Block、等待筛选、排序和 Top-N 限制影响。概览表先显示
pin 只影响固定对比区，下方普通 Block 列表仍按原筛选和排序显示，不会将已固定 Block 置顶或高亮。概览表先显示 Block 汇总，再显示三个
角色的明细；点击任意一行可定位到对应 Block。

也可以一次显式传入多个文件：

```bash
python3 mc2/mega_moe/tools/critical_path_trace/trace_tool.py rank0.raw rank1.raw rank2.raw
```

输出路径完全由 `--output-dir` 控制；未指定时，单文件报告写入输入文件旁的 `<rank>-trace/`，目录或多文件报告
写入 `trace-report/`，不会固定写入 `/tmp`。

## 3. 记录语义

每个物理 AI Block 对应三条记录：`AIC core N`、`AIV0 core N`、`AIV1 core N`。时间戳来自
`AscendC::GetSystemCycle()`；同一份 rank trace 中的全部 AI Block 使用统一 cycle 坐标，跨 rank 不合并时间轴。

每个事件保存：

```text
stage, wave, expert, tokenStartInExpert, tokenCount, mGroupCount, parentStage, beginCycle, endCycle
```

协议 v4 将每次依赖等待记录为独立事件：

```text
Wait Dispatch：GMM1 AIC 等待 Dispatch 输入
Wait GMM1：Activation AIV0 通过 PIPE_V 等待 GMM1 tile 或 Cube ping-pong 数据
Wait Activation：GMM2 AIC 等待 Activation 输出
Wait GMM2：Combine AIV1 等待 GMM2 completion counter
```

wait 事件通过 `parentStage` 表示所阻塞的计算类型，但在时间轴上是计算开始前的相邻独立区间。协议 v4 的
`activeCycles` 就是 tile/chunk 计算事件本身的时长，不再从外层阶段时长中扣除等待。工具仍兼容协议 v1-v3。

GMM/Activation 事件按本核拥有的 tile 记录，Combine 按本核拥有的 chunk 记录；坐标仍映射为 Wave 内专家 token
范围。Dispatch 只记录本核实际发送的行，因此没有任务的核不会产生伪计算块。

结束时间需要表示异步流水的真实完成：

- AIC GMM1/GMM2 复用实现内部的计算完成或同步发布边界；
- AIV1 Dispatch/Combine 和 AIV0 Activation 复用各阶段已有的完成发布边界；
- generic AIV0 的 C/V 等待在 begin 前和 wait 后各增加一次 trace-only `V_S`，使 Scalar 时间戳对应
  PIPE_V 真正进入并退出等待的边界；
- AIV0 GMM2 prologue 与 AIV1 Combine 在外层额外执行一次 `MTE3_S`，再读取 Scalar 时间戳。

## 4. 当前有意保留的限制

- 每条 core 记录最多 256 个事件；独立 wait 事件也占用容量，超出后在记录中设置 `truncated=1`；
- generic C/V 等待的两次 `V_S` 会减少 Scalar 对 Vector 的超前下发，只用于定位等待区间，带打点包的绝对耗时
  不能作为正式性能结果；
- trace 最终借用扩容后的输出 `y` 承载，当前诊断构建提供 1 MiB；容量不足时，应结合报告中的
  `event_capacity` 和 `truncated` 判断数据是否完整；
- 只支持 A4W4 token-wave，不兼容旧 A8W8/coarse trace 协议；
- 暂不自动修改外部测试脚本，也不绑定某个固定的 `moe_test` checkout；
- 打点会增加 Scalar 写 GM 和 AIV0 完成同步开销，适合判断流水结构，不应把绝对耗时当成无打点性能。

后续精简或扩充协议时，只需同步修改 `mega_moe_trace.h` 和 `trace_tool.py`，不再维护旧系统中的 spec、audit、
contract hash 和多 profile 分支。
