# MegaMoe A8W8 Wave 关键路径打点

该诊断构建记录 AIC、AIV0、AIV1 在各个 wave 上的 Dispatch、GMM1、Activation、GMM2、Combine，
并把显式轮询等待保存为父阶段内部的独立时间区间。

协议和报告结构参考 `ascendc-agent-toolkit` 的 `instrument-ascendc-critical-path` skill，
基准版本为 `b14f3bd`；解析器基于其中已验证的 MegaMoe v3 工具适配。

## 注意

- `MEGA_MOE_ENABLE_WAVE_TRACE=1` 时，算子正常计算完成后会用 trace 覆盖输出 `y`，不能检查精度。
- 事件先写入业务 workspace 末尾已有的 50 MiB 保留区；只有 kernel 结束时才复制到 `y`。
- 每条 record 最多 512 个事件，并同时受实际 `y` 容量限制；超出时只设置 `truncated`，不会越界。
- 打点不会在业务阶段新增 `X_S`；最终发布 trace 时会执行全角色同步。
- `active = stage - 已记录的显式 wait`，仍包含地址计算、调度和未拆分同步，不等同于纯计算时间。

## 导出 raw

在 MegaMoe 调用完成且 `torch.npu.synchronize()` 之后、任何精度检查或 `y` 切片之前保存原始字节：

```python
def dump_critical_path_trace(y, rank, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    raw_path = os.path.join(output_dir, f"rank{rank}.raw")
    y.detach().contiguous().cpu().view(torch.uint8).numpy().tofile(raw_path)
    print(f"[TRACE] rank {rank}: {raw_path}", flush=True)
```

不要把 BF16/FP16 输出转换成 FP32 后再落盘。

## 生成报告

目录中存在 `rank0.raw`、`rank1.raw` 等文件时执行：

```bash
python3 mc2/mega_moe/tools/critical_path_trace/trace_tool.py <raw-directory>
```

默认生成 `<raw-directory>/trace-report/index.html`，并为每个 rank 生成：

- `index.html`：交互时间线；
- `events.csv`：逐事件数据；
- `trace.json`：完整协议数据；
- `timeline.svg`：静态时间线。

当前报告按已验证环境的 `1000 cycles = 1 us` 展示。不同平台需先确认 SYS_CNT 频率。

## Dispatch 分项

rank-interleave trace 按 AIV1 core/wave 累计以下五个互不重叠的指标：

- `PrepareDispatchCounts (aggregate)`：从 cumsum 准备当前专家各 source rank 的 token 计数；
- `RankRoundPlan (aggregate)`：规划本 core 的 compact `{rank, topkIndex}` 队列，不含嵌套的 mask 处理；
- `MaskLoadAndGather (aggregate)`：mask GM→UB、`CreateVecIndex`、`GatherMask` 及结果对 Scalar 可见；
- `TransferPipeline (aggregate)`：token/scale/meta 的 MTE2/MTE3 ring 搬运完整流程；
- `ReadyPublish (aggregate)`：数据可见同步以及 GMM1 tile-ready counter 发布。

`MaskLoadAndGather` 还会细分为五个二级累计项：`Mask DataCopy issue`、
`CreateVecIndex issue`、`Wait mask/index ready`、`GatherMask issue` 和
`Wait GatherMask ready`。三个 `issue` 只表示异步指令下发区间；实际完成延迟分别由既有
`MTE2_V`、`V_S` 同步等待承接。该细分没有增加新的同步屏障。

`GatherMask call count` 另行记录每个 AIV1 core/wave 实际调用 `GatherMask` 的总次数。
它只用于判断重复扫描规模，不是 cycle 数据，也不会参与 Dispatch 分项的耗时归类。
可用 `GatherMask issue cycles / GatherMask call count` 估算平均单次 API 下发成本。

这些事件是一个 wave 内多个短区间的累计值，不对应一段连续时间轴位置，因此不会绘制为时间线色块；
它们会出现在 HTML 的 Dispatch 分项、wave/event 表格、CSV 和 JSON 中。`Other/control` 是 Dispatch
父事件减去五个分项后的剩余周期，主要包含 wave 游标规划、地址设置、条件检查及计时本身的固定开销。
容量不足而回退到旧的 row-range Dispatch 路径时，回退路径暂不拆分，时间会归入 `Other/control`。

## 当前等待覆盖

- AIC GMM1 等 Dispatch：`WaitForGmm1InputReady`；
- AIV0 Activation 等 GMM1：`WaitGmm1TileStatus`，仅 prefetch 路径；
- AIC GMM2 等 Activation：`WaitForGmm2InputReady`；
- AIV1 Combine 等 GMM2：`WaitWaveGmm2GroupReady`。

Dispatch 内部 buffer/MTE 握手、非 prefetch Activation 的硬件事件等待目前未拆分。
