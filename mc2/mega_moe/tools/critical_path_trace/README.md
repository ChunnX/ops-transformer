# MegaMoe A4W4 Wave 关键路径打点

这是一套用于 A4W4 Wave 性能分析的轻量诊断工具。它只记录 Wave 切片粒度的
Dispatch、GMM1、prologue、Activation、GMM2 和 Combine，不记录逐 tile 数据。

本轮打点主要用于分析小 BS 下的三类问题：Wave 是否欠载、AIC/AIV0/AIV1 的阶段是否失衡，以及后级阶段是否因
等待 Dispatch、GMM1、Activation 或 GMM2 出现空泡。事件保持在 Wave 内专家切片粒度，避免逐 tile 记录量和
Scalar 写 GM 开销掩盖原始流水特征。

## 1. 构建与运行

诊断分支默认启用：

```cpp
MEGA_MOE_ENABLE_A4W4_WAVE_TRACE=1
```

当前 A4W4 调度允许在专家内部按 token 切分 Wave，报告中的 `wave`、`expert`、`tokenStartInExpert` 和
`tokenCount` 可还原每个阶段实际处理的切片范围。做性能对比时应保持用例和 tiling 配置一致，只替换诊断包。

正常代码合入前应将其关闭或删除打点。启用后，kernel 在正常计算完成后把 trace 覆盖到输出 `y`，因此该包只用于
性能诊断，不能继续用 `y` 做精度比较。

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

协议 v3 将每次依赖等待记录为独立事件：

```text
Wait Dispatch：GMM1 AIC 等待 Dispatch 输入
Wait GMM1：Activation AIV0 通过 PIPE_V 等待 GMM1 tile 或 Cube ping-pong 数据
Wait Activation：GMM2 AIC 等待 Activation 输出
Wait GMM2：Combine AIV1 等待 GMM2 completion counter
```

wait 事件和父阶段使用相同的 wave、expert、token slice 元数据，字段 `parentStage` 表示依赖关系。报告按真实
begin/end cycle 将 wait 事件覆盖在对应 role/core 的同一泳道，并用落在父阶段内部的等待区间计算：

```text
activeCycles = stageCycles - sum(waitIntervalCycles)
```

因此父阶段块与 wait 块会在时间轴上重叠，这是为了直接表达“该阶段的这一段正在等待”，不是重复执行。
Dispatch 内部的 buffer/MTE 握手暂未拆分。`activeCycles` 仍包含地址准备、调度和普通完成同步，并非纯
MMAD/Vector 指令耗时。工具继续兼容协议 v1 的 8-word event 和协议 v2 的累计等待格式。

GMM/Activation/Combine 事件按 Wave 内的专家 token 切片记录。Dispatch 仍以完整专家执行，因此一个跨多个 Wave
的专家只在首次 Dispatch 时产生一次 Dispatch 事件，该事件标记其开始服务的 Wave。

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
- trace 最终借用输出 `y` 承载，实际单核容量会根据 `y` 的字节数动态缩小。极小 BS 若无法容纳协议头和全部
  core 记录，解析器会报告 magic 不匹配；容量不足但仍可写入时，应结合报告中的 `event_capacity` 和
  `truncated` 判断数据是否完整；
- 只支持 A4W4 token-wave，不兼容旧 A8W8/coarse trace 协议；
- 暂不自动修改外部测试脚本，也不绑定某个固定的 `moe_test` checkout；
- 打点会增加 Scalar 写 GM 和 AIV0 完成同步开销，适合判断流水结构，不应把绝对耗时当成无打点性能。

后续精简或扩充协议时，只需同步修改 `mega_moe_trace.h` 和 `trace_tool.py`，不再维护旧系统中的 spec、audit、
contract hash 和多 profile 分支。
