# Arch22 Wave 阶段打点

性能分支 `perf/gmma2av-a5-wave-phases` 在固定 buffer 0 的 Wave 版本上适配原打点提交 `313db239b`。按 tile 跨卡发布的快速穿刺未合入，当前跨卡仍按完整 expert 发布。入口及退出同步协议保持不变。

`GMMA2AV_MTE_TIMER=1` 在 host tiling 时选择打点 specialization。此模式需要提供 dummy shared MM 输入，用 `mm_y` 存储记录并跳过 shared MM；只适用于无真实 shared MM 的诊断。关闭该环境变量时选择 NullTimer specialization，热循环的记录操作在编译期消除。环境变量切换使用独立测试进程，避免复用先前的 tiling。

沿用 V3 缓冲布局和逐 lane 完成标记、溢出检测、cache-line flush。Wave 事件语义有扩展，应使用本次采样目录中的配套 decoder：

| 事件 | 含义 |
| --- | --- |
| Pipeline / ResetSync / CountTable / CountReady / BuildDestOffsets | 入口清理、count 发布、跨卡及本地初始化同步 |
| AicEntryWait（新增 42/43） | AIC 等待初始化完成 |
| AicMm | 单个 owned M tile 的全部 N tile 发射，payload 为 expert index |
| AicNotify | 已有 FIX_S 等待、wave progress 标量写入与 dcci |
| ComputeWaveWait（22/23） | Stage 等待配对 AIC 的 wave sequence，payload 为 sequence |
| StageCopy / StageProgress | workspace→window 搬运（含既有 drain）及本地 expert 完成发布 |
| WaitStage / Publish | Comm 等待本地 producer 后向对端发布 expert-ready |
| PeerWait / BuildSrcOffsets / Pull | 等待对端 ready、构建源偏移、window→输出搬运（含既有 drain） |
| RecvDone / AivExitWait / AicExitWait | 三阶段跨卡退出握手、最终本地同步 |

保留 V3 的计数槽，当前采集 Stage/Pull 搬运块数、字节数、drain 次数及 AIC MM tile 数；未埋点的计数槽为 0，不应用其推断该操作未发生。

这些是标量控制流区间，不等于异步引擎 busy time。AicMm 与 AicNotify 顺序发生，但不同 lane 的阶段会重叠。BuildSrcOffsets 嵌套于 Pull，不能重复加总；跨卡系统计数器不保证同一时间原点。退出等待包含其他角色或卡的未完成工作，不能直接视为同步实现的纯开销。

本次 8 卡、3 模型、128/1024/8192 token 的脚本、原始 CSV、逐 lane 二进制记录、分析与构建归档位于 `/root/workspace/Q2/GMMA2Av/wave_profile_20260915`。性能按逐卡最后 5 次求均值，打点阶段取最后一次调用；同时保留原包和新包关闭打点的性能对照，以量化扰动。
