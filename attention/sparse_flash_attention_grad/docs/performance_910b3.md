# Ascend910B3 确定性 SparseFlashAttentionGrad 优化验证

本次在现有注册接口中增加受保护的私有执行路径，保留原公开接口、支持集合和 legacy 实现。目标条件不匹配时使用原路径。

## 变更

- 在已验收的 query 分波次调度中恢复计算流水，按批搬运随机索引及 Gather/Scatter 数据。
- 在 L0C 中合并 dK 与 dV 的 FP32 计算，减少中间结果的 Fixpipe 输出及读取；保留确定性累加顺序。
- 使用已有显式缓冲区初始化，避免满 UB 配置下隐式临时空间不足。
- 对目标路径应用逐索引右下因果掩码；范围预检查不通过时回退原实现。同一 query 中重复索引仍保持原子累加语义。

## 验证场景

Ascend910B3（20 AIC / 40 AIV），CANN 9.1.0，默认 DVFS，FP16、BSND、确定性注册接口 `aclnnSparseFlashAttentionGradV2`。

| 参数 | 值 |
|---|---|
| B / S1 / S2 | 1 / 1024 / 131072 |
| N1 / N2 | 64 / 1 |
| D / ROPE D | 512 / 64 |
| topk | 2048 |
| sparse_mode / sparse_block_size | 3 / 1 |
| scale | 1 |
| 可选输入 | query_rope、key_rope 提供；sinks、value、actual_seq_lengths_query、actual_seq_lengths_kv 为空 |
| indices | mt19937 seed=20260912；每 query 2048 个无重复、未排序索引，范围 [0, S2) |

冻结输入使用 q/k/rope/out=0、d_out=1、softmax_max=0，softmax_sum 为各 query 实际因果有效索引数。另以非零输入及重复索引补充检查计算和确定性行为。

[冻结基线记录](../../../sparse_flash_attention_grad_baseline.csv) 是调优开始时保存的历史快照，包含当时本机证据路径和 37.04101 ms 历史测量；不是自包含测试程序。下表使用后续同环境、同集合、同预热协议的正式可比基线，未改写该历史快照。

## 性能

msprof application / PipeUtilization，每次目标调用前执行同一 256 MiB FP16 预热，共 10 组预热—目标相邻调用。调用映射完整，取 10 次目标 kernel 耗时中位数，不包含预热时间。

| 实现 | 中位数 | 样本范围 |
|---|---:|---:|
| 原始基线 | 36.95532 ms | 36.90540–37.09954 ms |
| 保留实现 | 27.21855 ms | 27.20228–27.26990 ms |

耗时下降 26.35%，加速比 1.3577。20 ms 调优目标尚未达到。核间切 S2 / dQ reduce 方案实际测试后出现回退，未纳入此提交；后续未超过波动门限的候选也未纳入。

## 正确性与范围

- 冻结用例四输出通过 FP16 MERE < 2^-10、MARE < 10×2^-10；20 次连续调用及最终三次重复测试逐字节一致。
- 非零及重复索引补充用例各重复三次，输出保持确定性。部分补充信号存在已登记的基线精度残差，保留实现未恶化，不将其描述为全部严格精度通过。
- 原工程真实重新构建、官方工作区安装及注册回归通过；注册三次输出与保留实现逐字节一致，TTK 精度及确定性检查通过。
- 实际命中 TilingKey 111111、私有标记 21；原工程生成并加载的目标设备内核与已验收内核 SHA256 一致：`34bcc3a3c0a7febeeb7f8ef75dea2f733d05b800c4f64c2c6376370c3bf717e4`。
- 原始 AddConfig 支持集合保持 ascend910b / ascend910_93 / ascend950。只有 Ascend910B3 动态验证；其他型号和未涉及路径仅通过源码隔离、原实现保留及回退检查，不宣称已上板通过。
