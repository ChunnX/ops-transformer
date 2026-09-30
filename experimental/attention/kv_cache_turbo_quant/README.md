# KvCacheTurboQuant

## 产品支持情况

| 产品                                                     | 是否支持 |
| -------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                   |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                  |    ×     |
| <term>Atlas 推理系列产品</term>                          |    ×     |
| <term>Atlas 训练系列产品</term>                          |    ×     |

## 功能说明

- 算子功能：对标准 MHA/GQA 的 KV cache 做**在线 TurboQuant 编码（Encode）**。每条 `head_dim = 128`
  的 KV 向量依次做：L2 范数、归一化到单位球面、正交旋转、Lloyd-Max 主量化（2/3/4 bit）、
  残差范数与 QJL 1-bit 随机投影符号。输入 BF16，输出 3 个编码张量 + 2 个标量张量。

- 计算公式（与任务包 `golden.py` 一致；`H` 为正交旋转矩阵，`S` 为高斯投影矩阵，`c` 为码本）：

  $$
  \begin{aligned}
  n &= \lVert x \rVert_2, & u &= \begin{cases} x / \max(n, 10^{-30}), & n > 0 \\ 0, & n = 0 \end{cases} \\
  y &= H u, & \mathrm{idx} &= \#\{k : y > \mathrm{mid}_k\} \\
  \hat y &= c[\mathrm{idx}], & r &= y - \hat y \\
  \rho &= \lVert r \rVert_2, & \gamma &= n \cdot \rho \\
  \tilde r &= \begin{cases} r / \max(\rho, 10^{-30}), & \rho > 0 \\ 0, & \rho = 0 \end{cases}, &
  \mathrm{qjl} &= \mathbb{1}\!\left[S \tilde r \ge 0\right]
  \end{aligned}
  $$

  量化判定为**严格大于**（落在中点边界上的值归左侧码字）；`idx` 与 `qjl` 按 8 通道一组、
  **低位在前（LSB-first）**打包为 `uint8`；`n` 与 `gamma` 以 BF16（ties-to-even）写出。

- 存储：`head_dim = 128`、`mse_bits = 3` 时每 head 为 48 B（主码）+ 16 B（QJL）+ 2 B（norm）+ 2 B（gamma）
  = **68 B**，相对 BF16 的 256 B 压缩约 **3.76×**。

## 参数说明

| 参数名          | 输入/输出/属性 | 描述                                | 数据类型 | 数据格式 |
| --------------- | -------------- | ----------------------------------- | -------- | -------- |
| kv_vectors      | 输入           | 待量化的 K 或 V 向量                | BF16     | ND，`[T, H_kv, 128]` |
| rotation_matrix | 输入           | 正交旋转矩阵 `H`，推理期固定        | FP32 / BF16（接口扩展） | ND，`[128, 128]` |
| qjl_matrix      | 输入           | 残差高斯投影矩阵 `S`，推理期固定    | FP32 / BF16（接口扩展） | ND，`[128, 128]` |
| mse_bits        | 属性（可选）   | 主量化位宽，默认 3，支持 2/3/4      | INT      | -        |
| quant_idx       | 输出           | 主量化码字（bit-packed）            | UINT8    | ND，`[T, H_kv, 16 * mse_bits]` |
| quant_qjl       | 输出           | 残差 QJL 符号（bit-packed）         | UINT8    | ND，`[T, H_kv, 16]` |
| quant_norm      | 输出           | `\lVert x \rVert_2`                 | BF16 / FP16（接口扩展） | ND，`[T, H_kv]` |
| quant_gamma     | 输出           | 原尺度残差范数 `n · \lVert r \rVert_2` | BF16 / FP16（接口扩展） | ND，`[T, H_kv]` |

## 约束说明

- 仅支持标准 MHA/GQA 的 KV cache；**不支持 MLA**（`cKV` / `kR` 不能拼成普通 head 后套用本接口）。
- `head_dim` 固定 128；`qjl_dim` 固定 128；`num_kv_heads` 支持 4 ~ 32。
- `mse_bits` 仅支持 2 / 3 / 4，默认 3；不支持 3.5-bit outlier 混合模式。
- 旋转/投影矩阵在一次推理内不变，由调用方传入。
- **dtype 口径**（按任务书 §2.3 与官方 2026-09-30 口径）：正式用例固定 kv=BF16、矩阵=FP32、
  metadata=BF16；**矩阵 BF16 与 metadata FP16 属于接口扩展，通过功能回归验证，不纳入性能门槛**。
  矩阵为 bf16 时算子内部**上转 fp32**（bf16 值在 fp32 中精确，与参考实现 `golden.py` 的上转
  一致）；metadata 为 fp16 时两个标量按 fp32 值 RNE 舍入到 fp16。矩阵 bf16/fp32 必须与
  qjl 一致（不允许混用）；metadata 两个输出必须同为 bf16 或同为 fp16。
- 空 tensor 不支持；非连续输入在算子定义中声明 `AutoContiguous`（官方口径：非连续输入不在
  本次直接 ACLNN 验收范围，此声明为超集）。

## 调用说明

算子执行接口为[两段式接口](https://gitcode.com/cann/opbase)，先调用
`aclnnKvCacheTurboQuantGetWorkspaceSize` 获取 workspace 大小与执行器，再调用
`aclnnKvCacheTurboQuant` 执行。接口细节见 [docs/aclnnKvCacheTurboQuant.md](./docs/aclnnKvCacheTurboQuant.md)。

| 调用方式 | 样例 | 说明 |
| -------- | ---- | ---- |
| 执行样例 | [examples/test_aclnn_kv_cache_turbo_quant.cpp](./examples/test_aclnn_kv_cache_turbo_quant.cpp) | 从裸二进制读输入、调 aclnn 算子、写回四个输出并打印计时；编译脚本 [examples/build_and_run.sh](./examples/build_and_run.sh) |
| 自测（精度/性能） | [tests/README.md](./tests/README.md) | 5 个自测 case 的逐字节对拍、V3 双口径精度与计时，一条命令可复现 |

## 性能与精度摘要

- 性能（Atlas A2、910B3、CANN 9.1.0；warmup 3 + repeat 10 取中位数，计时含 workspace 申请）：
  5 个自测 case 相对任务书 PyTorch eager 基线 **11.4× ~ 30.8×**，全部 ≥10×；
  `gqa_prefill_t2048`（最紧的一档）330.5 µs vs 门槛 385.5 µs（余量 12–14%）。
- 精度（任务包 `golden.py` 逐字节对拍，5 case × bits 2/3/4）：`quant_idx` / `quant_norm`
  **逐位差异 0**；`quant_gamma` 仅 3-bit 单点 1 ULP；`quant_qjl` 仅 4-bit 单点 1/262144
  （均为浮点归约顺序导致的 1 ULP 级差异）。
- 统计质量（decode 双口径，详见自测报告）：3-bit 重构 relMSE(MSE-only) ≈ 0.034 < 0.05；
  内积 p95（归一化、含 QJL）≈ 0.040 ~ 0.062 < 0.1。

## 目录结构

```
kv_cache_turbo_quant/
├── CMakeLists.txt
├── README.md
├── docs/aclnnKvCacheTurboQuant.md     # aclnn 接口说明
├── examples/                          # aclnn 调用样例与编译脚本
├── op_host/                           # 算子定义、shape/type 推导、tiling
├── op_kernel/                         # Ascend C kernel（MIX_AIC_1_2，单 kernel 融合两段 GEMM 与向量阶段）
└── tests/                             # 自测用例：对拍脚本、V3 精度、case.json/op.json（任务包原样）
```
