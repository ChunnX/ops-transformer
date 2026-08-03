# MsaIndexScore 测试说明

## 1. 端到端精度自验证（当前主用例）

`examples/test_aclnn_msa_index_score.cpp` 是一个自包含的 aclnn 调用程序，内置 CPU golden，
运行时会依次跑完整个用例矩阵并逐例给出 PASS/FAIL。

```bash
# 1) 编译自定义算子包
bash build.sh --pkg --soc=ascend910b --ops=msa_index_score -j32

# 2) 安装到独立路径（避免污染共享 CANN 安装）
./build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/tmp/msa_opp

# 3) 上板跑精度
export ASCEND_CUSTOM_OPP_PATH=/tmp/msa_opp/vendors/custom_transformer
bash build.sh --run_example msa_index_score eager cust --vendor_name=custom
```

## 2. 用例矩阵

对齐设计文档 `AR/A2A3_MSA/CatlassMsaIndexScore_A2_Design.md` §10.4 的黄金用例矩阵。

| 用例 | 场景 | 覆盖的缺陷类别 |
|------|------|---------------|
| `L0-prefill-aligned` | `q_len=32`，`start_loc=224`，`kv_len=256` | 主路径、128 对齐快通道 |
| `L1-prefill-unaligned` | `start_loc % 128 != 0`，多 batch varlen | 边界 block 的 causal mask、cumsum 推进 |
| `L1-prefill-multi-mtile` | 每请求行数 > `MSA_ROW_TILE_M` | M-tile 切分、跨 tile 的 `q_pos` 阶梯 |
| `L1-decode-lq1` | 6 个请求 `q_len=1`，`kv_len ∈ {1,4,128,129,512,900,4096}` | 长度截断、单 partial block、小 M |
| `L1-decode-speculative` | `q_len ∈ {2,4}` | 投机解码下 head-minor 打包的 `q_pos` 推导 |
| `L1-long-seq-multi-stile` | `kv_len=4096` → 32 个 block → 多个 S-tile | workspace 三级轮转、cross-core flag 计数配对 |
| `L1-bf16` | bf16 输入 | dtype 分支（TilingKey 0） |
| `L1-no-scale` | 不传 `scale` | 可选输入缺省分支 |
| `L1-head-dim-64` | `head_dim=64 < L1TileShape::K` | BlockMmad 的 K 尾块路径 |
| `L1-heads-16` | `num_q_heads=16` | M-tile 跨 token 边界的行映射 |
| `L2-tiny-kv` | `kv_len ∈ {1,3}`，`q_len=1` | 极小值边界、无效尾填充 |

所有用例的 `block_table` 都被刻意打乱，用于验证 paged 间接寻址。

## 3. Python 参考实现

`tests/golden/msa_index_score_golden.py` 提供与 kernel 数值路径一致的 numpy 参考实现
（fp32 累加 → 逐 block max → 乘 scale）与双标准比对函数，可用于接入仓上 pytest / ATK 框架。

```python
from msa_index_score_golden import msa_index_score_golden, compare

golden = msa_index_score_golden(query, key_cache, block_table, cu_seq_q, kv_len, start_loc,
                                block_size=128, scale=scale)
ok, msg = compare(actual, golden)
```

## 4. 判定标准

- 填充位（不可见 block）必须两侧同为 `-inf`。
- 有效位按 `atol=1e-3, rtol=1e-3, error_ratio <= 1e-3` 判定。
- 当前实测最大绝对误差量级为 `1e-6`。

## 5. 待补充

- `tests/ut/op_host/` 的 infershape / tiling gtest 用例。
- `tests/st/` 的 ATK json 用例。
