# A2 SparseFlashAttention Split-KV decode

## 适配范围

本优化仅针对 Ascend 910B / A2（arch22）的普通 SparseFlashAttention，
不包含 KvQuantSparseFlashAttention / V2、indexer 或模型接入改动。

| 项目 | 分核路径 |
| --- | --- |
| 数据类型 | 同类型 FP16 或 BF16 query、KV、attention output；FP32 max/sum |
| 布局 | query TND，KV PA_BSND |
| Head / 维度 | 单 KV head，G ∈ {1, 2, 4, 8, 16, 32, 64, 128}，value dim=512；可选 RoPE dim=64 |
| 稀疏参数 | sparse mode=0，sparse block size=1，TopK 256–2048，KV 长度不小于 TopK |
| Batch / LSE | 单 batch、多 batch；支持返回 softmax max/sum |
| 其他条件 | 无 sinks；可用核数和索引块数足够分成至少两份 |

其他既有合法配置保持原计算路径。索引行含连续尾部 `-1` padding，或实际 KV 长度
小于 TopK 时，该 query 使用原流水，并按有效前缀裁剪计算长度；同一次调用可以包含
分核行和回退行。全空行的 output、max、sum 均写零。有效前缀裁剪同时覆盖非分核调用。
本改动同时使能 A2 TND/PA_BSND 调用的 softmax max/sum 输出。

## 算法

1. 令总 query 数为 `T`，按 `min(16, floor(AIC/T), ceil(TopK/s2BaseSize))`
   选择不超过该值的最大 2 的幂作为分区数 `P`。A2 inner block 为 512，
   每个分区至少计算一个 inner block。
2. 每个核处理一个 `(query, KV partition)`，复用原 gather、QK、softmax、PV
   流水，将局部归一化 FP16/BF16 输出 `O_i` 和 FP32 统计量 `m_i`、`s_i`
   写入用户 workspace。根据 TND 累积 query 长度定位所属 batch。
3. 核间同步后，由首分区在同一个 kernel 内完成稳定 softmax 归并：

```text
M   = max_i(m_i)
w_i = exp(m_i - M) * s_i
S   = sum_i(w_i)
O   = sum_i(w_i * O_i) / S
```

请求 LSE 时，写回 FP32 `M` 和 `S`，对应 `LSE = M + log(S)`。
归并每次处理至多 16 个 head，流水排空后复用 32 KiB float 与 16 KiB 输出缓冲区；
小 G 的统计量按 32B 对齐存放，GM 按实际 G 精确读写。
不新增 kernel launch，不新增公共开关。

分核条件编码进 tiling key，使用编译期 `IS_FD` 特化，非分核实例不携带分核运行时分支。
原非 TND `FLASH_DECODE` 工作区和输出语义保持独立，不与新 TND 分核路径共用。

## 验证

2026-10-08：host UT 35/35 通过，包括原生 8 个 G、FP16/BF16、可选 RoPE、LSE、
多 batch、TopK 512–2048 的分核选择、24 核全占用时的 FD key、mode3/block2 回退、
不支持 G/类型的公共拒绝，以及既有 shape/dtype 推导测试。

```bash
bash build.sh --ophost_test --soc=ascend910b --ops=sparse_flash_attention -j8
bash build.sh --pkg --soc=ascend910b --ops=sparse_flash_attention \
    --vendor_name=sfa_a2_verify -j8
```

将单算子包安装到隔离 vendor 并加载其环境后，执行：

```bash
python -m pytest -q attention/sparse_flash_attention/tests/pytest/test_sfa_split_kv_arch22.py
```

Ascend910B2 设备回归 **906 passed，0 skipped，0 failure/error**（127.98s）：原生 G、两种 dtype、可选 RoPE、LSE 开关、单/多 batch、
完整/尾 padding/全空/混合索引、eager 与 NPU graph、3 个随机 seed、512 边界附近、
TopK128 非分核、短实际 KV 和空 query batch。
每行输出对照独立 FP32 softmax reference，relative L2 阈值为 BF16 0.02、FP16 0.005。
LSE 分别检查 max、sum 和 `max + log(sum)`；全空行严格检查 output/max/sum 为零。

完整单算子包以 CANN 9.1.0 编译，设备环境为 CANN 9.2.0、Python 3.11.10、
PyTorch 2.10.0 / torch_npu 2.10.0.post7.dev20260918、Ascend910B2。
master 基线为 `b2d40bd0d3c75356e2eb9af85d7313fe29e302f8`，完整索引数值回归 64/64 通过。
单算子包 `cann-ops-transformer-sfa_a2_general_20261008_linux-aarch64.run` 的 SHA256 为
`6cf3be4829e2752f329ab8042047672a50b97ffd5173ff9d0d268324383d42d9`，本地与设备端一致。

## 独立单算子性能

同一 Ascend910B2 设备，对比上述 master 与本实现；TND/PA_BSND、BF16、RoPE64、
单 KV head、dim512、TopK2048、KV4096、mode0/block1、LSE=false。
每图 32 次调用，每轮 101 个 NPU Event 样本，baseline/candidate 交替测量 3 轮，
取三轮中位数。每个计时配置先逐行校验独立 FP32 reference。
Q/B 表示每 batch 的 query 数，下降比例为 `(baseline-with)/baseline`。

| G | B | Q/B | master (µs) | Split-KV (µs) | 下降比例 |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 1 | 85.382 | 37.266 | 56.4% |
| 8 | 1 | 1 | 85.083 | 38.362 | 54.9% |
| 8 | 1 | 4 | 87.455 | 47.161 | 46.1% |
| 8 | 2 | 2 | 87.728 | 46.199 | 47.3% |
| 8 | 4 | 4 | 90.907 | 91.851 | -1.0% |
| 8 | 8 | 8 | 211.686 | 209.759 | 0.9% |
| 32 | 1 | 1 | 86.505 | 40.914 | 52.7% |
| 64 | 1 | 1 | 90.878 | 45.911 | 49.5% |
| 128 | 1 | 1 | 102.639 | 57.767 | 43.7% |

总计对比 31 个配置，包括上述 G、不同 B/Q、FP16 和无 RoPE。
本实现另测 G8/G128 × FP16/BF16 × 有/无 RoPE 的 B1/Q1 LSE 开关成本，
8 个配置的时延变化范围为 -0.6% 至 +0.5%。
