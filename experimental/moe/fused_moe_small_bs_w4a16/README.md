# 算子名称：FusedMoeSmallBsW4a16（实验性质）

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品（ascend910b）      | 是       |
| 其他产品                                                     | 未验证   |

> 本算子为**实验性质**（移植自 MoE-Ascend-Opt/moe_ascend_npu 的 `fused_moe_small_bs_w4a16` 核函数），
> 目录组织遵循仓内 GE 注册算子规范，但 Python 绑定采用目录内自包含 JIT 扩展，
> 不侵入 `torch_extension/cann_ops_transformer` 共享包。

## 功能说明

- **算子功能**：面向 decode 阶段小 batch 场景的 W4A16 融合 MoE 前向计算。单个 kernel 依次完成
  `W13` 分组反量化 GEMV（广播 `x`）→ `SwiGLU` 激活 → `W2` 分组反量化 GEMV → 按 `topk_weights`
  加权累加。权重为对称量化 int4（group_size=32，无 zero-point），中间结果全部落在 workspace，
  相对逐算子拆解的官方 `npu_fused_experts` 路径省掉中间结果的显存往返。

- **计算公式**：记 batch 内第 `t=(b,k)` 个 token 路由到专家 `e = expert_ids[b, k]`：

$$
y[b] = \sum_{k=0}^{K-1} topk\_weights[b, k] \cdot GEMV_{W2}^{e}\left( SwiGLU\left( GEMV_{W13}^{e}(x[b]) \right) \right)
$$

其中 $GEMV_{W}^{e}(v)[n] = \sum_{g} scale^{e}[g, n] \cdot \sum_{j \in group\ g} v[j] \cdot w^{e}[j, n]$，
group 大小为 32，kernel 内按 group 以 FP16 累加、乘 scale 后以 FP32 跨 group 累加（BF16 输入先转 FP16 计算）。

## 目录结构

```
fused_moe_small_bs_w4a16/
├── CMakeLists.txt
├── op_host/                                   # host 侧注册
│   ├── fused_moe_small_bs_w4a16_def.cpp       # OpDef（7 输入 1 输出，fp16/bf16 两组 dtype）
│   ├── fused_moe_small_bs_w4a16_infershape.cpp
│   └── fused_moe_small_bs_w4a16_tiling.cpp    # shape 校验、workspace 计算、tilingKey 分发
├── op_kernel/
│   ├── fused_moe_small_bs_w4a16.cpp           # 模板入口（schMode 分发 fp16/bf16）
│   ├── fused_moe_small_bs_w4a16_kernel.h      # 核心计算逻辑（ZeroOut/W13 GEMV/SwiGLU/W2 GEMV/Cast）
│   ├── fused_moe_small_bs_w4a16_tiling_data.h
│   └── fused_moe_small_bs_w4a16_tiling_key.h  # ASCENDC_TPL 模板声明
├── examples/
│   └── test_aclnn_fused_moe_small_bs_w4a16.cpp  # aclnn C++ 验证（自带 CPU 参考实现）
└── tests/                                     # Python 测试（自包含 torch 绑定）
    ├── fused_moe_small_bs_w4a16_binding.cpp   # torch<->aclnn 桥接（JIT 编译）
    ├── _binding.py                            # JIT 加载 + torch.library 注册（支持 NPUGraph 捕获）
    ├── _helpers.py                            # int4 打包 + 纯 torch 参考实现
    ├── test_fused_moe_small_bs_w4a16.py       # 正确性测试（含 NPUGraph 用例）
    └── benchmark_fused_moe_small_bs_w4a16.py  # 性能测试（对标官方 npu_fused_experts 路径）
```

## 参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | shape |
| :--- | :--- | :--- | :--- | :--- |
| x | 输入 | 输入隐状态 | FLOAT16、BFLOAT16 | `[B, H]` |
| w13_weight | 输入 | W13 打包权重（8 个 int4 打包为一个 int32） | INT32 | `[E, H, 2*I/8]` |
| w13_scales | 输入 | W13 分组量化 scale（对称量化） | 与 x 一致 | `[E, H/32, 2*I]` |
| w2_weight | 输入 | W2 打包权重 | INT32 | `[E, I, H/8]` |
| w2_scales | 输入 | W2 分组量化 scale | 与 x 一致 | `[E, I/32, H]` |
| expert_ids | 输入 | 每个 token 路由到的专家 id（越界 id 贡献为 0） | INT32 | `[B, K]` |
| topk_weights | 输入 | 每个 token 的路由权重 | FLOAT32 | `[B, K]` |
| y | 输出 | MoE 加权累加后的隐状态 | 与 x 一致 | `[B, H]` |

其中 `B`=BatchSize，`K`=TopK，`E`=NumExperts，`H`=HiddenSize，`I`=InterSize。

## 权重打包约定

`w13_weight`/`w2_weight` 的 int4 打包方式：每个 int32 沿 N 方向打包 8 个 int4，
第 `j` 个值位于第 `4*j`~`4*j+3` 位（低位在前），存储为二进制补码（`value & 0xF`），
与 kernel 侧 `int4b_t` 的解释一致。打包实现参考 `tests/_helpers.py` 的 `pack_int4`。

## 约束说明

- `H`（HiddenSize）与 `I`（InterSize）必须为 128 的倍数（kernel 以 128 为 K 方向加载块、32 为量化 group）。
- 面向小 batch（GEMV 路线）；大 batch 场景建议使用官方 `npu_fused_experts` 路径。
- 仅支持 ascend910b（Atlas A2）；tiling 中 `SetBlockDim` 取 AIV 核数。

## 编译与安装

```bash
# 编译算子包（experimental 目录需显式打开 --experimental）
bash build.sh --pkg --experimental --soc=ascend910b --ops=fused_moe_small_bs_w4a16 -j32

# 安装（默认装到 ${ASCEND_HOME_PATH}/opp/vendors/custom_transformer）
./build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet
```

## 功能验证（C++ aclnn example）

example 自带完整 CPU 参考实现（int4 解包 + SwiGLU + topk 加权），
覆盖 fp16/bf16 × 3 组 shape（含 K 方向多 block 原子加场景）：

```bash
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH}
bash build.sh --run_example fused_moe_small_bs_w4a16 eager cust --vendor_name=custom
```

预期输出：6 个用例全部 `TEST PASSED`（实测最大绝对误差 < 4e-3）。

## Python 测试（自包含绑定，支持 NPUGraph）

`tests/` 下的绑定通过 `torch.utils.cpp_extension.load` JIT 编译，
直接调用 `libcust_opapi.so` 的 aclnn 两段式接口，并注册 `torch.library`
（schema + Meta + PrivateUse1），因此算子可被 `torch.npu.NPUGraph`（aclgraph）捕获，
满足 vLLM decode 路径的 graph 化要求：

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0   # 按实际 CANN 路径
cd experimental/moe/fused_moe_small_bs_w4a16/tests
python3 test_fused_moe_small_bs_w4a16.py               # 正确性 + NPUGraph 捕获验证
python3 benchmark_fused_moe_small_bs_w4a16.py          # 性能对比（--json-out 可导出）
```

正确性实测（E=16, H=2048, I=768, K=8，对照纯 torch FP32 参考）：
fp16 max_diff≈0，bf16 max_diff≈1e-4；NPUGraph replay 与 eager 输出 diff = 0。

性能实测（910B3，对标官方 `npu_fused_experts` W4A16 antiquant 路径，
E=128, H=2048, I=768, K=8，两条路径各捕获 16 个路由块进一个 NPUGraph，event 计时）：

| BS | Ref(us) | Custom(us) | Ref GB/s | Custom GB/s | Speedup |
| -: | ------: | ---------: | -------: | ----------: | ------: |
| 1  |  202.09 |      41.51 |   105.07 |      511.59 |   4.87x |
| 2  |  269.92 |      58.28 |   157.33 |      728.68 |   4.63x |
| 3  |  412.76 |      76.51 |   154.33 |      832.62 |   5.40x |
| 4  |  521.48 |      93.39 |   162.87 |      909.42 |   5.58x |
| 5  |  620.44 |     110.57 |   171.12 |      960.19 |   5.61x |
| 6  |  735.32 |     136.25 |   173.26 |      935.09 |   5.40x |
| 7  |  843.11 |     154.38 |   176.29 |      962.81 |   5.46x |
| 8  |  944.75 |     172.22 |   179.80 |      986.33 |   5.49x |

## 备注

- 本算子为实验性质；若后续转正，可按 `torch_extension` 开发规范接入
  `cann_ops_transformer` 共享包（aclnn wrapper + OpBuilder + torchair GE Converter）。
- 源 kernel 中的 `grouped_gemv_w4a16_moe`、`batch_gemm_w4a16_small_bs`、
  `fused_moe_w4a16_cached` 等入口未纳入本次移植，如需可从源仓库补充。
