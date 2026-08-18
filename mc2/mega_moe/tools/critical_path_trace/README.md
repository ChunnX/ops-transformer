# MegaMoe A8W4 token-wave 关键路径打点

这是一套用于当前 A8W4 token-wave 性能分析的轻量诊断工具。它只记录 Wave 切片粒度的
Dispatch、GMM1、prologue、Activation、GMM2 和 Combine，不记录逐 tile 数据。

## 1. 构建与运行

诊断分支默认启用：

```cpp
MEGA_MOE_ENABLE_A8W4_WAVE_TRACE=1
```

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
  `SYS_CNT` cycle 原点；每个 role/core 只占一条泳道，显式轮询等待以窄色条覆盖在对应阶段区间底部。鼠标悬停
  或点击事件块可查看 begin/end cycle、父阶段和对应的代码打点边界。当前平台按 `1000 cycles = 1 μs` 换算。

交互报告还提供泳道概览，可按角色和是否存在等待筛选，并按 Process 总时长、事件覆盖时长、累计等待、等待占比、
最后打点位置或尾部空洞排序。时间线标签和概览表会显示 Process begin/end、最后事件、最后结束 cycle、累计
Stage/Wait 和截断状态；点击概览表中的一行可定位到对应 Block/角色。

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

协议 v3 将每次显式轮询记录为独立事件：

```text
Wait Dispatch：GMM1 AIC 等待 Dispatch 输入
Wait GMM1：Activation AIV1 等待 GMM1 tile
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

- AIC GMM1/GMM2 复用实现内部已有的 `FIX_S` 发布边界；
- AIV1 Dispatch、Activation、Combine 复用各阶段已有的 MTE-to-Scalar 完成边界；
- AIV0 prologue 在外层额外执行一次 `MTE3_S`，再读取 Scalar 时间戳。

## 4. 当前有意保留的限制

- 每条 core 记录最多 256 个事件；独立 wait 事件也占用容量，超出后在记录中设置 `truncated=1`；
- 只支持 A8W4 token-wave，不兼容旧 A8W8/coarse trace 协议；
- 暂不自动修改外部测试脚本，也不绑定某个固定的 `moe_test` checkout；
- 打点会增加 Scalar 写 GM 和 AIV0 完成同步开销，适合判断流水结构，不应把绝对耗时当成无打点性能。

后续精简或扩充协议时，只需同步修改 `mega_moe_trace.h` 和 `trace_tool.py`，不再维护旧系统中的 spec、audit、
contract hash 和多 profile 分支。
