# Jetty 与发送对端数拓扑批跑

## 运行

先重新生成、编译并部署算子包、Torch 扩展和静态图内核。此次新增 `peers_per_jetty` 算子属性及 tiling 字段，服务器上的安装产物必须与源码一致。

两台服务器各运行一次：

```bash
python bandwidth_test.py
```

`config.py` 默认 `RunConfig.sweep_enabled=True`。所有服务器使用相同 shape、核数、协议、batch、Drain 阈值、master IP 和初始端口；非云环境的 `server_index` 分别设为 0、1。
默认 `shared_jetty_comm_level=1` 为全 UBC_CTP。`batch_tokens`、`drain_threshold`、图模式和采样设置沿用配置。

协调进程为每组拓扑启动新的测试进程和通信 context。组内复用通信域，顺序执行全部 `ShapeConfig.test_configs`。
默认 16 卡、18 个尺寸对应 12 组拓扑、216 个用例。各组使用从 `master_port` 开始的连续 12 个端口，避免跨组复用旧 rendezvous。
两台服务器都必须启动；一台失败时，另一台可能等待后端超时。

设置 `sweep_enabled=False` 可恢复原有单拓扑测试；此时 `mode`、`shared_jetty_num`、`peers_per_jetty` 直接来自配置。
扫描器会覆盖模式、Jetty 数、对端数和输出路径，强制开启带宽批跑并关闭精度测试；不会改变配置文件中的 BS/H 用例。

## 矩阵

| 实验 | 模式 | Jetty 数 J | 每个 Jetty 实际发送对端数 P |
| --- | --- | --- | --- |
| jetty_scan | 3 | 1、2、4、8、16、32 | W−1 |
| peer_scan | 3 | 固定 15 | 1、2、4、8……，补上 W−1 并去重 |
| mode4_baseline | 4 | W−1 | 1 |

例如 16 卡的 P 为 1、2、4、8、15；32 卡的 P 为 1、2、4、8、16、31。
所有组使用共享 URMA；选定 AIV 数必须覆盖最大 Jetty 数，且所有 Jetty 数均不超过当前 56 上限。超出能力时启动器报错，不会静默跳过实验。

## 轮转规则和计时口径

mode 3 中，本卡 rank 为 r、Jetty 下标 j、第 k 个对端（k 从 0 到 P−1）的选择为：

```text
delta = 1 + (j + floor(k × (W−1) / P)) % (W−1)
peer  = (r + delta) % W
```

16 卡时，卡 0 的 Jetty 0 在 P=2 时发送卡 1、8；Jetty 1 发送卡 2、9。
P=4 时 Jetty 0 发送卡 1、4、8、12。每个 Jetty 的集合无重复且跳过本卡。
接收端用相同 delta 反算来源 `(r + W − delta) % W`，仅等待这些来源的当前 Jetty flag。
每个选中对端仍收到完整 BS 个 token；数据允许覆盖，flag 独立。

这改变了旧 mode 3 按 rank 升序发送的顺序，所有对端实验也使用新的轮转顺序。与旧日志比较时需考虑发送顺序变化。
建链仍覆盖全部远端，P 控制流量对端数，而非建链通道数。
16 卡下固定 J=15、P=1 与 mode 4 有相同的每 rank 发送量和一对一目的卡覆盖，但 kernel 分支及 Jetty 对应顺序不同。其他卡数下 J=15 不等于 mode 4 的 W−1。

```text
mode 3 每 rank payload 字节 = J × P × BS × H × 2
mode 4 每 rank payload 字节 = (W−1) × BS × H × 2
修正耗时 = profiler 算子设备耗时 − 2 μs
平均带宽 = 每 rank payload 字节 / 修正平均耗时
最大带宽 = 每 rank payload 字节 / 修正最小耗时
```

报告只保存修正耗时，不额外记录原始耗时列。原始 profiler 产物保留在各组目录。
非正修正耗时对应的带宽为 N/A（CSV 空值），不会钳制成极小正数；可有效计算的平均带宽仍保留。
带宽沿用现有 GB/s 字段名，但实际除以 2^30，数值单位为 GiB/s。flag/count 不计 payload 字节，但计入队列预算。
结果统计的是当前服务器各 rank 的样本，不是全服务器聚合吞吐。

## 结果文件

```text
Bandwidth_Sweep_Results/
  server_0/<运行时间>/
    manifest.json
    results.md
    results.csv
    01_jetty_scan_j1_p15/
      results.json
      Profiling_Result_config_1/...
      ...
    ...
  server_1/<运行时间>/...
```

每组实验照常打印尺寸、批量、字节数、带宽与修正耗时，启动行标注实验组、J、P。
`results.md` 按三组实验分表；`results.csv` 使用 UTF-8 BOM，可直接导入 Excel。
每行包含拓扑、服务器/采样 rank、协议层、图模式、AIV、dtype、BS/H、批量与字节数、Drain 阈值、2 μs 扣除值及修正 Mean/Min/Max 耗时和 Mean/Max 带宽。
Markdown 头部记录共同配置，CSV 每行记录配置，方便筛选和制作 PPT 图表。

每完成一个拓扑就更新总表；后续失败时保留已有数据，`manifest.json` 标记失败位置和已完成组数。全部成功才标记 complete。
两台服务器结果目录分别保留，合并 Excel 时用 `server_index` 和 `sampled_ranks` 区分，不应直接把本机平均带宽相加。

## CPU 检查

```bash
python -m unittest discover -s tests -p test_bandwidth_sweep.py
```

测试覆盖矩阵生成、真实子进程配置继承、模拟完整批跑和 CSV/Markdown 输出、失败保留结果。这些检查不依赖 CANN/NPU，也不替代设备编译和多卡验证。

`ParallelConfig.aiv_num` 默认 -1，由 tiling 读取平台实际 AIV 数；None 也按 -1 处理。扫描器仅在显式指定正数时按该核数预检，自动模式由 tiling 校验 Jetty 数是否可用。报告中的 aiv_num 为请求值，-1 不代表实际核数。
