# TurboQuantAttentionScore

## 产品支持情况

| 产品 | 支持 |
| --- | --- |
| Atlas A2 训练/推理系列（ascend910b） | 是 |

使用 Ascend C 在旋转空间中直接计算 TurboQuant 编码 Key 的 attention score。
输入 query 已完成旋转和 QJL 投影；本算子不执行旋转、softmax、V 聚合或 MLA。

## 参数说明

记 query token 数为 \(T\)，query head 数为 \(H_q\)，KV head 数为 \(H_k\)，物理块数为 \(N\)，batch 为 \(B\)，页表宽度为 \(P\)。

| 参数 | 形状 | 类型 |
| --- | --- | --- |
| query_rotated | `[T,Hq,128]` | bfloat16 |
| query_qjl | `[T,Hq,128]` | bfloat16 |
| key_cache_idx | `[N,128,Hk,16*mse_bits]` | uint8 |
| key_cache_qjl | `[N,128,Hk,16]` | uint8 |
| key_cache_norm | `[N,128,Hk]` | bfloat16 / float16 |
| key_cache_gamma | `[N,128,Hk]` | 与 norm 相同 |
| block_table | `[B,P]` | int32 |
| seq_lens | `[B]` | int32（额外兼容 int64） |
| attn_scores | `[T,Hq,max(seq_lens)]` | float32 |
| mse_bits（属性） | 2、3、4，默认 3 | int |

所有 Tensor 使用 ND 格式。输入通过生成的 aclnn 接口自动连续化。

## 约束和语义

- `head_dim=qjl_dim=block_size=128`，`Hq` 必须是 `Hk` 的正整数倍。
- `B=1` 时全部 query token 共享该序列，支持 decode 和 prefill。
- `B>1` 时 `T=B`，第 t 个 query 对应第 t 个请求。接口没有多序列 prefill 的 query 边界信息，因此不接受此类布局。
- 长度范围为 `[0,128*P]`；超出有效长度的输出为负无穷。所有长度为零时输出最后一维为零。
- 有效位置的物理块编号须在 `[0,N)`，无效位置的页表项不会被读取。内核对越界物理块防护为负无穷，调用者仍须提供合法映射。
- 量化索引按低位优先连续打包，3 bit 编码可跨字节；QJL bit 0/1 对应 -1/+1。
- centroid 使用附件 golden.py 中的 FP32 常量。BF16 query/norm/gamma 先转 FP32，点积和最终加法均用 FP32。
- 不增加 `1/sqrt(head_dim)` 缩放，不施加 causal mask。

## 编译和调用

在 ops-transformer 仓库根目录执行：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash build.sh --pkg --experimental --soc=ascend910b \
  --ops=turbo_quant_attention_score --vendor_name=turboquant -j8
```

安装 `build_out` 目录中的 `.run` 包后，加载安装目录下的环境脚本：

```bash
bash build_out/cann-ops-transformer-turboquant_linux-aarch64.run \
  --quiet --install-path=/path/to/install
source /path/to/install/vendors/turboquant_transformer/bin/set_env.bash
```

aclnn 两阶段接口、输出分配和数据依赖见 [接口文档](docs/aclnnTurboQuantAttentionScore.md)。
自测步骤见 [tests/README.md](tests/README.md)，设计见 [设计文档](docs/design.md)。


## 对齐 prefill 流水

单序列对齐 prefill 可使用 Vector 准备与 Cube 计算重叠的路径；其他形状保留原 Vector 实现。
该路径使用 FP16 双分量系数、FP32 累加和有界临时工作区，并对不适合矩阵乘的运行时输入回退。
实现条件、精度边界和工作区公式见 [设计说明](docs/design.md#对齐-prefillvector-准备与-cube-分阶段流水2026-10-02)。

安装自定义包并配置 `TURBOQUANT_OPAPI` 后，可执行：

```bash
python tests/test_prefill_regression.py --output prefill_regression.json
python tests/test_golden_nonfinite.py
```

测试使用原有混合误差阈值；完整附件用例仍需额外校验，不能仅凭随机回归声明任务精度验收完成。
