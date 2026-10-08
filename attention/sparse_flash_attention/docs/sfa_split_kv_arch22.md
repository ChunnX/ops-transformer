# A2 SparseFlashAttention Split-KV decode

## 适配范围

本优化仅针对 Ascend 910B / A2（arch22）的普通 SparseFlashAttention，
不包含 A5、KvQuantSparseFlashAttention / V2、indexer 或模型接入改动。

| 项目 | 分核路径 |
| --- | --- |
| 数据类型 | BF16 query、KV、attention output；FP32 max/sum |
| 布局 | query TND，KV PA_BSND |
| Head / 维度 | 单 KV head，G=8 或 64，value dim=512，RoPE dim=64 |
| 稀疏参数 | sparse mode=0，sparse block size=1，TopK 256–2048，KV 长度不小于 TopK |
| Batch / LSE | 单 batch、多 batch；支持返回 softmax max/sum |
| 其他条件 | 无 sinks；可用核数和索引块数足够分成至少两份 |

其他既有合法配置保持原计算路径。索引行含 `-1` padding，或实际 KV 长度
小于 TopK 时，该 query 使用原路径；同一次调用可以包含分核行和回退行。
本改动同时使能 A2 TND/PA_BSND 调用的 softmax max/sum 输出。

## 算法

1. 令总 query 数为 `T`，按 `min(16, floor(AIC/T), ceil(TopK/s2BaseSize))`
   选择不超过该值的最大 2 的幂作为分区数 `P`。A2 inner block 为 512，
   每个分区至少计算一个 inner block。
2. 每个核处理一个 `(query, KV partition)`，复用原 gather、QK、softmax、PV
   流水，将局部归一化 BF16 输出 `O_i` 和 FP32 统计量 `m_i`、`s_i`
   写入用户 workspace。根据 TND 累积 query 长度定位所属 batch。
3. 核间同步后，由首分区在同一个 kernel 内完成稳定 softmax 归并：

```text
M   = max_i(m_i)
w_i = exp(m_i - M) * s_i
S   = sum_i(w_i)
O   = sum_i(w_i * O_i) / S
```

请求 LSE 时，写回 FP32 `M` 和 `S`，对应 `LSE = M + log(S)`。
G64 的归并每次处理至多 16 个 head，流水排空后复用 UB scratch；
不新增 kernel launch，不新增公共开关。

## 验证

2026-10-08 独立分支基于 master `45227fd67`，代码提交 `74db99f57`：host UT 35/35 通过，
包括 G8/G64、LSE 开关、多 batch、TopK 512–2048 的分核数选择，
以及既有 shape/dtype 推导测试。

```bash
bash build.sh --ophost_test --soc=ascend910b --ops=sparse_flash_attention -j8
bash build.sh --pkg --soc=ascend910b --ops=sparse_flash_attention \
    --vendor_name=sfa_a2_verify -j8
```

将单算子包安装到隔离 vendor 并加载其环境后，执行：

```bash
python -m pytest -q attention/sparse_flash_attention/tests/pytest/test_sfa_split_kv_arch22.py
```

设备测试覆盖 G8/G64、LSE 开关、单/多 batch、完整索引和 padding、eager 与
NPU graph。完整索引输出对照独立 FP32 softmax reference，relative L2 阈值
为 0.02；LSE 通过 `max + log(sum)` 与 reference 比较。Padding 用例仅做
有限输出和回退 smoke 检查。

完整单算子包以 CANN 9.1.0 编译成功，安装到隔离 vendor 后，Ascend910B2
设备回归 **256 passed，0 skipped，0 failure/error**。设备环境为 CANN 9.2.0、
Python 3.11.10、PyTorch 2.10.0 / torch_npu 2.10.0.post7.dev20260918。
未替换系统 CANN，也未修改其他运行服务。

包文件 `cann-ops-transformer-sfa_a2_only_20261008_linux-aarch64.run` 的 SHA256 为
`2c168fb56891a6b18c3fd2955273f9006eba7c27ae2e0457504b4bf3a93aa4e5`，
已确认本地与设备端一致。仓库 pre-commit 检查通过。

尚未在该分支重测性能。
原综合优化的 A2 128K/256K AISBench 命中量化 C8 算子，不作为本普通 SFA
改动的端到端性能数据。
