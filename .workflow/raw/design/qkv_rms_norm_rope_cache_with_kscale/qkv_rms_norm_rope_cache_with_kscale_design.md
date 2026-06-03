# QkvRmsNormRopeCacheWithKScale Design Document

本文记录新算子 `QkvRmsNormRopeCacheWithKScale` 的接口、数据规格、计算语义、tiling、kernel 分层和开发落地方案。当前版本已经覆盖首期 arch35 交付的主体计算链路，并保留仍需确认的边界语义和参数组织方式。

## 1. 算子概述

`QkvRmsNormRopeCacheWithKScale` 是一个面向 Transformer decode/prefill 相关流程的融合算子。设计目标是在一个 kernel 框架内组织：

- AIV 前处理
- AIC matmul 或矩阵类计算
- AIV 后处理
- controller 分块调度

整体框架计划参考 `ops-transformer/gmm/grouped_matmul_finalize_routing/op_kernel/arch35/weight_quant_basic_block` 的分层方式：

```text
controller
  - 负责 token/head/dim 等维度的分块调度
  - 负责多核任务分发
  - 负责输入输出 GM 地址推进或 offset 计算

vcv/basic_block
  - 负责串联 AIV 前处理、AIC 计算、AIV 后处理
  - 负责 AIV/AIC 跨核同步

vec_compute
  - 负责 AIV 侧前处理和后处理

cube_compute
  - 负责 AIC 侧矩阵类计算
```

当前交付范围限定为 **arch35 平台**。实现上只需要面向 Ascend950/arch35 注册、编译和测试，不交付 arch34/910B/910_93 等平台的 kernel 分支和配置。

## 2. 接口定义

### 2.1 必选输入

| 输入 | dtype | shape | 说明 |
|------|-------|-------|------|
| `qkv` | `bf16` | `[T, Nq + Nk + Nv, D]` | Q/K/V 融合输入 |
| `q_gamma` | `float32` | `[D]` | Q 分支 RMSNorm gamma |
| `k_gamma` | `float32` | `[D]` | K 分支 RMSNorm gamma |
| `cos_sin` | `float32` | `[MaxSeqLen, D]` | RoPE 所需 cos/sin 数据；`MaxSeqLen` 语义为最大 batch 内 token 数 |
| `slot_mapping` | `int32` | `[T]` | token 到 cache slot 的映射 |
| `k_cache` | `float8_e4m3fn` | `[BlockNum, Nk, BlockSize, D]` | K cache，输入输出复用空间 |
| `v_cache` | `float8_e4m3fn` | `[BlockNum, Nv, BlockSize, D]` | V cache，输入输出复用空间 |
| `k_scale_cache` | `float32` | `[BlockNum, Nk, BlockSize, 1]` | K scale cache，输入输出复用空间 |
| `cu_seq_len` | `int32` | `[Batch + 1]` | batch 序列前缀和信息，采用 cu-seqlens 风格；`cu_seq_len[0] = 0`，`cu_seq_len[Batch] = T` |

> 约定：本文中所有 `float8_e4m3` 均按工程 dtype `float8_e4m3fn` 理解，包括 `q_out`、`k_cache`、`v_cache` 以及中间 FP8 quant 结果。

### 2.2 可选输入

| 输入 | dtype | shape | 说明 |
|------|-------|-------|------|
| `rotation` | `bf16` | `[D, D]` | 旋转矩阵，Cube matmul 右矩阵；接口形态为可选输入，但当前场景必须存在 |
| `v_scale` | `float32` | `[Nv]` | V 分支 per-head 量化乘子；接口形态为可选输入，但当前场景必须存在 |

### 2.3 输出

| 输出 | dtype | shape | 说明 |
|------|-------|-------|------|
| `q_out` | `float8_e4m3fn` | `[T, Nq, D]` | Q 分支输出；用户语义中的 `float8_e4m3` 对应工程 dtype `float8_e4m3fn` |
| `q_scale` | `float32` | `[T, Nq]` | Q 分支量化 scale |
| `k_cache` | `float8_e4m3fn` | `[BlockNum, Nk, BlockSize, D]` | K cache 原地更新输出 |
| `v_cache` | `float8_e4m3fn` | `[BlockNum, Nv, BlockSize, D]` | V cache 原地更新输出 |
| `k_scale_cache` | `float32` | `[BlockNum, Nk, BlockSize, 1]` | K scale cache 原地更新输出 |

### 2.4 属性

| 属性 | dtype | 默认值 | 说明 |
|------|-------|--------|------|
| `head_nums` | `ListInt` | 必选 | head 数配置，长度必须为 3，依次为 `[Nq, Nk, Nv]`。host tiling 需要将三者写入 tiling data 并传递给 kernel。 |
| `epsilon` | `float32` | `1e-6` | RMSNorm 防止除零的小量，由 host tiling 写入 kernel tiling data。 |

## 3. 交付范围与目录结构

### 3.1 建议落盘目录

本算子属于 attention 前处理/cache 更新类融合算子，建议落在 attention 目录下：

```text
ops-transformer/attention/qkv_rms_norm_rope_cache_with_kscale/
```

参考对象：

- `attention/scatter_pa_kv_cache`：cache 写回、slot/block 映射、arch35 专用 kernel 子目录。
- `attention/nsa_compress_with_cache`：带 cache 更新的 attention 类算子，包含 `op_host/op_api` 形式的 ACLNN 交付件。
- `attention/mla_prolog_v3`：attention prolog 类融合算子，包含独立 `op_api`、`op_graph`、`op_host`、`op_kernel`、`tests`。
- `gmm/grouped_matmul_finalize_routing`：本算子 kernel 框架参考对象，尤其是 arch35 下 controller/basic_block/vec_compute/cube_compute 分层。

建议目录结构：

```text
qkv_rms_norm_rope_cache_with_kscale/
  CMakeLists.txt
  README.md
  docs/
    aclnnQkvRmsNormRopeCacheWithKScale.md
    QkvRmsNormRopeCacheWithKScale算子设计介绍.md
  examples/
    test_aclnn_qkv_rms_norm_rope_cache_with_kscale.cpp
    arch35/
      test_aclnn_qkv_rms_norm_rope_cache_with_kscale.cpp
  op_api/ 或 op_host/op_api/
    aclnn_qkv_rms_norm_rope_cache_with_kscale.h
    aclnn_qkv_rms_norm_rope_cache_with_kscale.cpp
    qkv_rms_norm_rope_cache_with_kscale.h
    qkv_rms_norm_rope_cache_with_kscale.cpp
  op_graph/
    fallback_qkv_rms_norm_rope_cache_with_kscale.cpp
    fallback_qkv_rms_norm_rope_cache_with_kscale.h
  op_host/
    CMakeLists.txt
    qkv_rms_norm_rope_cache_with_kscale_def.cpp
    qkv_rms_norm_rope_cache_with_kscale_infershape.cpp
    qkv_rms_norm_rope_cache_with_kscale_tiling.h
    qkv_rms_norm_rope_cache_with_kscale_tiling.cpp
    qkv_rms_norm_rope_cache_with_kscale_tiling_register.cpp
    op_tiling/
      arch35/
        qkv_rms_norm_rope_cache_with_kscale_tiling.h
        qkv_rms_norm_rope_cache_with_kscale_tiling.cpp
  op_kernel/
    qkv_rms_norm_rope_cache_with_kscale.cpp
    qkv_rms_norm_rope_cache_with_kscale_apt.cpp
    qkv_rms_norm_rope_cache_with_kscale.h
    arch35/
      common/
        qkv_rms_norm_rope_cache_with_kscale_common.h
        qkv_rms_norm_rope_cache_with_kscale_tiling_data.h
        qkv_rms_norm_rope_cache_with_kscale_tiling_key.h
      basic_block/
        qkv_rrc_controller.h
        qkv_rrc_vcv_basic_block.h
        qkv_rrc_vec_compute.h
        qkv_rrc_cube_compute.h
        qkv_rrc_cube_compute_tools.h
        qkv_rrc_vf.h
  tests/
    CMakeLists.txt
    assets/
      inputs.py
      golden.py
    ut/
      CMakeLists.txt
      op_host/
        test_qkv_rms_norm_rope_cache_with_kscale_infershape.cpp
        test_qkv_rms_norm_rope_cache_with_kscale_tiling.cpp
      op_kernel/
        test_qkv_rms_norm_rope_cache_with_kscale.cpp
        qkv_rms_norm_rope_cache_with_kscale_tiling_def.h
      op_api/
        test_aclnn_qkv_rms_norm_rope_cache_with_kscale.cpp
```

`op_api` 的位置需在开发时按目标子系统风格最终确认：

- 如果按 `mla_prolog_v3` 和 `grouped_matmul_finalize_routing`，使用根目录 `op_api/`。
- 如果按 `nsa_compress_with_cache`，使用 `op_host/op_api/`。

为了便于对外暴露 ACLNN 接口，设计上必须交付 ACLNN API 文件；目录位置不影响 API 内容，但会影响 CMake 接入方式。

### 3.2 必须交付件

#### 3.2.1 根目录交付件

| 文件 | 必要性 | 说明 |
|------|--------|------|
| `CMakeLists.txt` | 必须 | 将 `op_host`、`op_kernel`、`op_api`、`tests` 等子目录纳入构建。 |
| `README.md` | 必须 | 简要描述算子功能、支持平台、编译和运行示例。 |
| `docs/aclnnQkvRmsNormRopeCacheWithKScale.md` | 必须 | ACLNN 接口文档，描述输入输出、shape、dtype、约束和调用方式。 |
| `docs/QkvRmsNormRopeCacheWithKScale算子设计介绍.md` | 建议 | 面向维护者的设计说明，可由本文档收敛后同步。 |

#### 3.2.2 API 交付件

ACLNN 层至少需要交付：

```text
aclnn_qkv_rms_norm_rope_cache_with_kscale.h
aclnn_qkv_rms_norm_rope_cache_with_kscale.cpp
qkv_rms_norm_rope_cache_with_kscale.h
qkv_rms_norm_rope_cache_with_kscale.cpp
```

职责：

1. 暴露 `aclnnQkvRmsNormRopeCacheWithKScaleGetWorkspaceSize`。
2. 暴露 `aclnnQkvRmsNormRopeCacheWithKScale` 执行接口。
3. 完成 TensorList/Tensor 参数解析和空指针检查。
4. 完成 dtype、维度、可选输入存在性和输入输出复用关系校验。
5. 调用 `l0op` 或 `op_host` 注册路径生成 executor。
6. 在只支持 arch35 的前提下，对非 ascend950/arch35 平台返回明确不支持错误。

接口参数顺序建议与用户语义一致：

```cpp
aclnnStatus aclnnQkvRmsNormRopeCacheWithKScaleGetWorkspaceSize(
    const aclTensor *qkv,
    const aclTensor *qGamma,
    const aclTensor *kGamma,
    const aclTensor *cosSin,
    const aclTensor *slotMapping,
    const aclTensor *kCache,
    const aclTensor *vCache,
    const aclTensor *kScaleCache,
    const aclTensor *cuSeqLen,
    const aclTensor *optionalRotation,
    const aclTensor *optionalVScale,
    const aclIntArray *headNums,
    double epsilon,
    aclTensor *qOut,
    aclTensor *qScale,
    aclTensor *kCacheOut,
    aclTensor *vCacheOut,
    aclTensor *kScaleCacheOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);
```

执行接口：

```cpp
aclnnStatus aclnnQkvRmsNormRopeCacheWithKScale(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

注意：

1. `kCacheOut/vCacheOut/kScaleCacheOut` 与对应输入共享物理空间。API checker 必须允许输入输出 alias，并明确禁止其他未声明 alias。
2. `optionalRotation` 是 ACLNN 接口中的可选输入形参名，对应算子语义输入 `rotation`。当前 arch35 交付场景必须校验 `optionalRotation != nullptr`，否则返回参数错误或不支持。
3. `optionalVScale` 是 ACLNN 接口中的可选输入形参名，对应算子语义输入 `v_scale`。当前 arch35 交付场景必须校验 `optionalVScale != nullptr`，否则返回参数错误或不支持。
4. `headNums` 是必选 ListInt 属性，ACLNN 侧建议用 `const aclIntArray *headNums` 表达，必须校验非空、长度为 3，且三个值依次为正整数 `Nq`、`Nk`、`Nv`。

#### 3.2.3 op_graph 交付件

若该算子需要图模式 fallback 或 proto 注册，交付：

```text
op_graph/fallback_qkv_rms_norm_rope_cache_with_kscale.cpp
op_graph/fallback_qkv_rms_norm_rope_cache_with_kscale.h
```

职责：

1. 注册图模式 fallback。
2. 对不支持平台或不支持 shape 走 fallback/报错策略。
3. 与 `op_host` 的 op type 名称保持一致。

如果首期只交付 ACLNN 直调路径，可以将 `op_graph` 标为可选，但 `op_host` 的 `def/infershape/tiling` 仍然必须存在。

#### 3.2.4 op_host 交付件

必须交付：

```text
op_host/CMakeLists.txt
op_host/qkv_rms_norm_rope_cache_with_kscale_def.cpp
op_host/qkv_rms_norm_rope_cache_with_kscale_infershape.cpp
op_host/qkv_rms_norm_rope_cache_with_kscale_tiling.h
op_host/qkv_rms_norm_rope_cache_with_kscale_tiling.cpp
op_host/qkv_rms_norm_rope_cache_with_kscale_tiling_register.cpp
op_host/op_tiling/arch35/qkv_rms_norm_rope_cache_with_kscale_tiling.h
op_host/op_tiling/arch35/qkv_rms_norm_rope_cache_with_kscale_tiling.cpp
```

职责：

1. `def.cpp`
   - 注册 op type：`QkvRmsNormRopeCacheWithKScale`。
   - 定义输入、可选输入、输出、属性。
   - 标记 `k_cache`、`v_cache`、`k_scale_cache` 与输出的 in-place/alias 语义。
   - 不新增 `op_host/config` 配置文件；按 `grouped_matmul_finalize_routing` 的 arch35 路径，通过 op_host 代码和 tiling key 选择 arch35 kernel。

2. `infershape.cpp`
   - 从 `qkv` 推导 `T`、融合 head 总数 `Hqkv`、`D`。
   - 从 cache 推导 `BlockNum`、cache head 维、`BlockSize`、`D`。
   - 从必选属性 `head_nums` 读取 `Nq/Nk/Nv`，并用它校验 `Hqkv` 和 cache head 维。
   - 推导输出：

     ```text
     q_out: [T, Nq, D]
     q_scale: [T, Nq]
     k_cache_out: same as k_cache
     v_cache_out: same as v_cache
     k_scale_cache_out: same as k_scale_cache
     ```

   - 校验 `qkv.D == q_gamma.D == k_gamma.D == cache.D`。
   - 校验 `D` 为偶数；RoPE 当前按 `D/2` 分割低半区和高半区。
   - 从必选属性 `head_nums` 读取 `Nq`、`Nk`、`Nv`，校验 `head_nums` 长度为 3，且三个值均大于 0。
   - 校验 `qkv.shape[1] == Nq + Nk + Nv`。
   - 校验 `k_cache.shape[1] == Nk`、`v_cache.shape[1] == Nv`、`k_scale_cache.shape[1] == Nk`。
   - 校验 `cos_sin.shape[1] == D`，`cos_sin.shape[0]` 作为 `MaxSeqLen` 使用。
   - 校验 `slot_mapping.shape == [T]`。
   - 校验 `cu_seq_len` 为一维 `[Batch + 1]`，并由此推导 `Batch = cu_seq_len.shape[0] - 1`。
   - `cu_seq_len` 固定采用前缀和输入：`cu_seq_len[0] = 0`，`cu_seq_len[b + 1] - cu_seq_len[b]` 表示第 `b` 个 batch 的有效 token 数，期望 `cu_seq_len[Batch] == T`。
   - 如果 host 侧能够读取 `cu_seq_len` 值，则建议校验 `cu_seq_len[0] == 0`、`cu_seq_len` 单调不降、`cu_seq_len[Batch] == T`；如果 tiling 阶段不可读，则 kernel 按合法输入假设处理。
   - `MaxSeqLen` 的实际语义是最大 batch 内长度，即 `MaxSeqLen = max(cu_seq_len[b + 1] - cu_seq_len[b])`。host 无法可靠校验 `cos_sin.shape[0] == MaxSeqLen`，因此只做 `cos_sin.shape[1] == D` 等静态校验。
   - ACLNN 层校验 `optionalRotation != nullptr`，op_host 语义上对应 `rotation.shape == [D, D]`、`rotation.dtype == bf16`。
   - ACLNN 层校验 `optionalVScale != nullptr`，op_host 语义上对应 `v_scale.shape == [Nv]`、`v_scale.dtype == float32`。
   - 校验其他输入输出 dtype 与接口定义一致，并校验 in-place 输出 alias 关系。

3. `tiling.h/.cpp`
   - 获取平台信息，只接受 arch35/ascend950。
   - 解析输入 shape、dtype、format 和可选输入。
   - 计算 kernel 所需 tiling data。
   - 计算 workspace size。
   - 生成 tiling key。
   - 保存 tiling data。

4. `tiling_register.cpp`
   - 注册 tiling class 和 arch35 tiling key。

5. `op_tiling/arch35/*`
   - 放置 arch35 专用 tiling/checker 实现。
   - 首期不交付 `op_host/config/ascend950/*.json` 或 `*_simplified_key.ini`。
   - 参考 `grouped_matmul_finalize_routing` 的 arch35 方案，不通过新增 config 文件完成平台分发。

#### 3.2.5 op_kernel 交付件

必须交付：

```text
op_kernel/qkv_rms_norm_rope_cache_with_kscale.cpp
op_kernel/qkv_rms_norm_rope_cache_with_kscale_apt.cpp
op_kernel/qkv_rms_norm_rope_cache_with_kscale.h
op_kernel/arch35/common/qkv_rms_norm_rope_cache_with_kscale_common.h
op_kernel/arch35/common/qkv_rms_norm_rope_cache_with_kscale_tiling_data.h
op_kernel/arch35/common/qkv_rms_norm_rope_cache_with_kscale_tiling_key.h
op_kernel/arch35/basic_block/qkv_rrc_controller.h
op_kernel/arch35/basic_block/qkv_rrc_vcv_basic_block.h
op_kernel/arch35/basic_block/qkv_rrc_vec_compute.h
op_kernel/arch35/basic_block/qkv_rrc_cube_compute.h
op_kernel/arch35/basic_block/qkv_rrc_cube_compute_tools.h
op_kernel/arch35/basic_block/qkv_rrc_vf.h
```

职责：

1. `qkv_rms_norm_rope_cache_with_kscale.cpp`
   - kernel 编译入口。
   - include arch35 入口或统一入口。
   - 注册 kernel symbol。

2. `qkv_rms_norm_rope_cache_with_kscale_apt.cpp`
   - arch35 APT 入口。
   - 参考 `grouped_matmul_finalize_routing_apt.cpp`，根据 dtype/tiling key 分发模板实例。
   - 设置 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)` 或最终确认的 AIC/AIV 组合模式。
   - `REGISTER_TILING_DEFAULT` 并读取 arch35 tiling data。

3. `tiling_data.h`
   - 定义 host 到 kernel 的 packed tiling data。
   - 初始字段建议包括：

     ```cpp
     struct QkvRmsNormRopeCacheWithKScaleTilingData {
         uint32_t coreNum;
         uint32_t aicNum;
         uint32_t aivNum;
         uint32_t tSize;
         uint32_t nQ;  // from head_nums[0]
         uint32_t nK;  // from head_nums[1]
         uint32_t nV;  // from head_nums[2]
         uint32_t headDim;
         uint32_t batch;
         uint32_t maxSeqLen;
         uint32_t blockNum;
         uint32_t blockSize;

         uint32_t qkvStrideT;
         uint32_t qkvStrideH;
         uint32_t cacheStrideBlock;
         uint32_t cacheStrideHead;
         uint32_t cacheStrideToken;

         uint32_t tokenTile;
         uint32_t tokenTilePerAiv;
         uint32_t rowTile;
         uint32_t rowTileAligned;
         uint32_t qkHeadTile;
         uint32_t vHeadTile;
         uint32_t dimTile;

         uint32_t flags;
         float epsilon;
     };
     ```

     首期 `D=128` fast path 固定 `tokenTile` 为偶数，`qkHeadTile = Nq + Nk`，`vHeadTile = Nv`，`dimTile = 128`，并通过 `flags` 标记 rotation L1 resident、V cache 在 `A_READY` 后提前写回、L0C double buffer 等策略。`rowTile` 表示 full tile 的 Q/K row 容量，`M_tile` 表示当前运行时 tile 的实际 Cube M 值，两者在 tail tile 中可能不同。UB/L1 总量、buffer 数、reserve 子区容量和 Cube M 上限 256 均为 kernel 静态常量；host tiling 使用这些常量计算 `tokenTile` 和校验资源上界，但不写入 tiling data。

4. `tiling_key.h`
   - 定义 arch35-only 模板参数。
   - `head_nums` 不进入 tiling key；host tiling 将其解析为 `nQ/nK/nV` 三个字段写入 tiling data。
   - `cos_sin` layout 固定为 `[MaxSeqLen, D]`，`cu_seq_len` shape 固定为 `[Batch + 1]` 前缀和输入。
   - `cu_seq_len` 不再需要 `SEQ_LEN_MODE` 进入 tiling key；kernel 按固定前缀和语义解析。
   - dtype 首期固定，尽量不放入 tiling key，避免实例膨胀：

     ```text
     qkv      fixed bf16
     gamma    fixed fp32
     q_out    fixed fp8_e4m3fn
     cache    fixed fp8_e4m3fn
     scale    fixed fp32
     ```

5. `controller.h`
   - 管理 token/head/dim 分块。
   - 管理多核任务分发。
   - 管理 `slot_mapping` 到 cache 地址的映射。
   - 管理双缓冲 process id。
   - 调用 basic block。

6. `vcv_basic_block.h`
   - 参考 `GMMFRWeightQuantVcvBasicBlock`。
   - 串联 AIV 前处理、AIC matmul/矩阵计算、AIV 后处理。
   - 管理 cross-core flag。

7. `vec_compute.h`
   - 实现 AIV 前处理/后处理。
   - 预留 Q/K RMSNorm、RoPE、FP8 quant、cache 写回等逻辑。

8. `cube_compute.h`
   - 实现 AIC rotation matmul。
   - 负责 `rotation[D,D]` 的 ND2NZ 搬运、L1 常驻或分块复用、L0A/L0B/L0C 数据流、MMAD 和 Fixpipe。
   - 输出从 L0C 通过 Fixpipe 搬到 UB，layout 固定为 ND `[M_tile, D]`。

9. `vf.h`
   - 放 micro API 小函数。
   - 适合放 RMSNorm reduce、RoPE pair compute、FP8 pack/scale 等 AIV 内核片段。

#### 3.2.6 tests 交付件

必须交付 host 和 kernel 单测，建议同时交付 ACLNN 样例。

```text
tests/assets/inputs.py
tests/assets/golden.py
tests/ut/op_host/test_qkv_rms_norm_rope_cache_with_kscale_infershape.cpp
tests/ut/op_host/test_qkv_rms_norm_rope_cache_with_kscale_tiling.cpp
tests/ut/op_kernel/test_qkv_rms_norm_rope_cache_with_kscale.cpp
tests/ut/op_kernel/qkv_rms_norm_rope_cache_with_kscale_tiling_def.h
tests/ut/op_api/test_aclnn_qkv_rms_norm_rope_cache_with_kscale.cpp
examples/test_aclnn_qkv_rms_norm_rope_cache_with_kscale.cpp
```

测试覆盖范围：

1. 基础 shape：
   - `T = 1` decode。
   - `T > 1` batch/prefill 风格。
   - `Nq != Nk` 正常路径。
   - `Nk != Nv` 拒绝路径；首期 fast path 只支持 `Nv == Nk`。
   - `D = 128` 典型路径。

2. 可选输入形态下的强制存在校验：
   - 缺失 `rotation` 时 checker 报错。
   - 缺失 `v_scale` 时 checker 报错。
   - `rotation` 和 `v_scale` 都存在时进入正常 kernel 路径。
   - 缺失 `head_nums`、`head_nums` 长度不是 3 或任一 head 数非正时报错。
   - `qkv.shape[1] != sum(head_nums)` 报错。

3. cache 映射：
   - `slot_mapping` 连续。
   - `slot_mapping` 跨 block。
   - `slot_mapping` 非连续。
   - 当前不覆盖负数或越界值，按输入合法处理。

4. RoPE 位置映射：
   - `Batch = 1` 时，`s(t) = t`。
   - `Batch > 1` 且 batch length 不等长时，验证 `t` 通过 `cu_seq_len` 前缀和映射到正确的 batch 内位置。
   - 覆盖 `cu_seq_len = [0, len_0, len_0 + len_1, ..., T]` 的正常路径。
   - 若 host 能读取 `cu_seq_len` 值，补充 `cu_seq_len[0] != 0`、非单调、`cu_seq_len[Batch] != T` 的报错用例；若不能读取，则该项不作为强制测试。
   - `cos_sin.shape[1] != D` 报错。

5. in-place 输出：
   - `k_cache_out` 与 `k_cache` alias。
   - `v_cache_out` 与 `v_cache` alias。
   - `k_scale_cache_out` 与 `k_scale_cache` alias。

6. 平台限制：
   - ascend950/arch35 可 tiling。
   - 非 arch35 返回不支持。

### 3.3 明确不交付或暂不交付

首期不交付：

1. 非 arch35 kernel。
2. `ascend910b`、`ascend910_93` config。
3. 非固定 dtype 组合。
4. 训练反向算子。
5. 动态 dtype 自动分发。
6. 多种 cache format 的自动适配，除非后续计算逻辑明确需要。

## 4. Shape 符号

| 符号 | 含义 |
|------|------|
| `T` | 当前处理 token 数 |
| `Nq` | Q head 数 |
| `Nk` | K head 数 |
| `Nv` | V head 数 |
| `D` | 每个 head 的 hidden dimension |
| `Batch` | 本次输入包含的 batch 数，满足 `cu_seq_len.shape[0] = Batch + 1` |
| `MaxSeqLen` | `cos_sin` 的第一维，语义上等于最大 batch 内 token 数 |
| `BlockNum` | KV cache block 数 |
| `BlockSize` | KV cache 单 block 内 token 数 |
| `K` | `rotation` 维度，已确认 `K = D` |

## 5. 已确认约束与待确认项

### 5.1 已确认

1. `k_cache`、`v_cache`、`k_scale_cache` 是输入输出复用空间。
2. `qkv` 的 shape 为 `[T, Nq + Nk + Nv, D]`，Q/K/V 沿 head 维拼接。
3. Q 分支和 K 分支分别有独立 RMSNorm gamma：`q_gamma`、`k_gamma`。
4. `cos_sin` shape 固定为 `[MaxSeqLen, D]`，前 `D/2` 为 cos，后 `D/2` 为 sin。
5. 首期只交付 arch35 平台。
6. `v_scale` shape 固定为 `[Nv]`。
7. `head_nums` 是必选 ListInt 属性，长度固定为 3，依次给出 `Nq`、`Nk`、`Nv`。host tiling 必须将三个值写入 tiling data 的 `nQ/nK/nV` 字段，kernel 不再从 tensor shape 推导 head 数。
8. `qkv.shape[1] == Nq + Nk + Nv`，其中 `Nq/Nk/Nv` 来自 `head_nums`。
9. 首期 fast path 只支持 `Nv == Nk`；接口仍保留独立 `Nv` 字段，`Nv != Nk` 由 checker/tiling 拒绝。
10. `rotation` shape 为 `[D, D]`，也即原始符号中的 `K = D`。
11. `rotation` 当前必须存在；虽然接口形态为可选输入，但本交付场景不支持缺省。
12. `rotation` 不直接参与 RoPE，计算顺序为 `RoPE -> rotation matmul`，即 `y @ rotation`。
13. RoPE 永远作用于完整 D 轴，不支持 `rotaryDim < D`。
14. `float8_e4m3` 对应工程 dtype `float8_e4m3fn`。
15. RoPE 自身计算过程使用 FP32；进入 Cube 前需要 cast 成 BF16。
16. `rotation` 可以来自 Hadamard 场景，但 kernel 和 golden 不假设 Hadamard 结构；首期只按外部传入的 `[D,D]` dense bf16 矩阵计算。
17. `fp8_max` 当前近似取 `448`。
18. `round_to_e4m3fn_rint` 需要在本算子的 VF 工具中实现。
19. `v_scale` 当前必须存在；虽然接口形态为可选输入，但本交付场景不支持缺省。
20. `slot_mapping` shape 固定为 `[T]`，当前按合法输入处理，不考虑负数或越界。
21. `slot_mapping` 到 cache 的映射公式固定为 `block_id = slot_mapping[t] / BlockSize`、`block_offset = slot_mapping[t] % BlockSize`。
22. `rotation` 按普通外部输入矩阵处理，不考虑 FWHT 优化。
23. `rotation` layout 为 ND `[D, D]`，进入 Cube 前采用 ND2NZ 搬运。
24. Fixpipe 从 L0C 搬出到 UB 的输出 layout 为 ND `[M_tile, D]`。
25. `cu_seq_len` shape 固定为 `[Batch + 1]`，采用前缀和输入，不再通过 tiling key 支持多种 `cu_seq_len` 组织方式。
26. `cu_seq_len[0] = 0`，`cu_seq_len[b + 1] - cu_seq_len[b]` 表示第 `b` 个 batch 的有效 token 数，期望 `cu_seq_len[Batch] == T`。若 host 侧无法读取 `cu_seq_len` 值，则不做有效值校验，kernel 按合法输入处理。
27. RoPE 的位置索引由 token 的 batch 内偏移决定。对全局 token 下标 `t`，根据 `cu_seq_len` 推断所属 batch `b(t)` 和 batch 内 token 下标 `s(t)`，然后读取 `cos_sin[s(t), :]`。
28. `D` 必须为偶数，因为 RoPE 公式按 `D/2` 拆分低半区和高半区。
29. Q/K 动态量化对齐 `ops-nn/quant/dynamic_quant` 在 arch35 上输出 FP8 E4M3FN 的实现：写出的 scale 为 `max_abs / 448`，量化计算不对 `scale == 0` 额外加跳过或保护分支，而是按 `Div -> Cast fp8` 执行。

### 5.2 待确认

当前无阻塞首期实现的待确认项。

## 6. AIV 前处理逻辑

AIV 前处理主要包含四个阶段：

1. 从 `qkv` 中搬入 Q/K/V，其中 Q/K 进入 RMSNorm + RoPE。
2. 对 Q/K 执行 RMSNorm。
3. 对 RMSNorm 后的 Q/K 执行 RoPE。
4. 暂存 V，供 `A_READY` 后的 V 提前后处理执行 `v * v_scale` 量化并写入 `v_cache`。

当前已明确 RMSNorm、RoPE 和 `rotation` matmul 的初版语义。`rotation` 不参与 RoPE 公式本身，而是在 RoPE 之后执行 `y @ rotation`。

### 6.1 Q/K 提取

输入 `qkv` 的逻辑 shape 为：

```text
qkv: [T, Nq + Nk + Nv, D]
```

其中 head 维按如下方式切分：

```text
Q: qkv[:, 0 : Nq, :]
K: qkv[:, Nq : Nq + Nk, :]
V: qkv[:, Nq + Nk : Nq + Nk + Nv, :]
```

更精确地说，`nq/nk/nv` 分别是 Q/K/V 分支内部的局部 head 下标，`h_qkv` 才是 `qkv` 第二维的融合 head 下标：

```text
h_qkv = nq,              0 <= nq < Nq
Q[t, nq, d] = qkv[t, h_qkv, d]

h_qkv = Nq + nk,         0 <= nk < Nk
K[t, nk, d] = qkv[t, h_qkv, d]

h_qkv = Nq + Nk + nv,    0 <= nv < Nv
V[t, nv, d] = qkv[t, h_qkv, d]
```

前处理阶段从 `qkv` 搬入 Q/K/V。Q/K 形成逻辑视图：

```text
x_qk: [T, Nq + Nk, D]
```

逻辑映射为：

```text
if 0 <= h_qk < Nq:
    h_qkv = h_qk
    x_qk[t, h_qk, d] = qkv[t, h_qkv, d]
else:
    nk = h_qk - Nq
    h_qkv = Nq + nk
    x_qk[t, h_qk, d] = qkv[t, h_qkv, d]

h_qkv = Nq + Nk + nv
v_in[t, nv, d] = qkv[t, h_qkv, d]
```

这里 `x_qk` 和 `v_in` 都是逻辑视图，不要求一定在 GM 或 UB 中形成连续中间张量。首期实现中，V 分支的 cache 写回属于后处理语义，但不依赖 Cube 输出；AIV 在完成 Q/K preprocess 并发送 `A_READY` 后即可提前执行 V quant 和 `v_cache` 写回，用完即可释放输入 UB buffer。

### 6.2 RMSNorm 数学语义

对提取出的 Q/K 矩阵，在 `D` 方向计算平方和。为避免公式和实现伪代码混淆，先用数学对象描述 RMSNorm。定义索引集合：

```text
T_set = {0, 1, ..., T - 1}
```

```text
H_qk_set = {0, 1, ..., Nq + Nk - 1}
```

```text
D_set = {0, 1, ..., D - 1}
```

定义 Q/K 拼接后的 FP32 数学输入 `X`，其数学 shape 为：

```text
X shape: [T, Nq + Nk, D]
```

```text
X[t,h,j] =
  fp32(Q[t,h,j])       if 0 <= h < Nq
  fp32(K[t,h-Nq,j])    if Nq <= h < Nq + Nk
```

上述公式的下标范围为：

```text
t in T_set, h in H_qk_set, j in D_set
```

平方和张量 `S` 的数学 shape 为：

```text
S shape: [T, Nq + Nk]
```

`S` 定义为：

```text
S[t,h] = sum over j in [0, D): X[t,h,j] * X[t,h,j]
```

工程实现中的 `sum_sq` 是数学张量 `S` 增加一个尾部 singleton 维后的存储形式，即 `sum_sq` 的逻辑 shape 为 `[T, Nq + Nk, 1]`，并满足 `sum_sq[t,h,0]` 对应 `S_{t,h}`。

输出逻辑 shape 为：

```text
sum_sq: [T, Nq + Nk, 1]
```

根据 `sum_sq` 计算 RMS：

```text
R[t,h] = sqrt(S[t,h] / D + epsilon)
```

工程实现中的 `rms` 逻辑 shape 为 `[T, Nq + Nk, 1]`，`rms[t,h,0]` 对应 `R_{t,h}`。

其中 `epsilon` 是算子属性，dtype 为 `float32`，默认值为 `1e-6`。`op_host` 在 `GetShapeAttrsInfo` 或等价阶段读取 attr，并在 `DoOpTiling` 中保存到 tiling data。

归一化和 gamma 缩放：

先定义两组 gamma 数学向量，它们分别由工程输入 `q_gamma` 和 `k_gamma` 承载：

```text
GammaQ shape: [D]
GammaK shape: [D]
```

再定义拼接到 Q/K head 维后的 gamma 张量 `G`：

```text
G[h,j] =
  GammaQ[j]    if 0 <= h < Nq
  GammaK[j]    if Nq <= h < Nq + Nk
```

```text
Y[t,h,j] = X[t,h,j] / R[t,h] * G[h,j]
```

工程实现中的 `y_norm[t,h,j]` 对应 `Y_{t,h,j}`。

输出逻辑 shape：

```text
y_norm: [T, Nq + Nk, D]
```

其中：

- `q_gamma` 只用于 Q head，覆盖 `h in [0, Nq)`。
- `k_gamma` 只用于 K head，覆盖 `h in [Nq, Nq + Nk)`。
- `x_qk` 输入 dtype 为 `bf16`，RMSNorm 的累加、除法、sqrt 和 gamma 乘法按 `float32` 语义描述。
- `y_norm` 是 RoPE 的输入，不一定需要写回 GM；首选实现是在 UB 内完成 RMSNorm 后直接进入 RoPE。

### 6.3 AIV 实现拆分建议

RMSNorm 可以在 `qkv_rrc_vec_compute.h` 中拆成以下步骤：

```text
CopyQKTileGmToUb
  - 根据 t/head/d tile 从 qkv 搬入 Q/K 到 UB

CopyVTileGmToUb
  - 根据 t/head/d tile 从 qkv 搬入 V 到 UB
  - V 不参与 RMSNorm/RoPE，暂存到后处理阶段使用

ComputeSquareSum
  - 将 bf16 转 float32
  - 在 D 方向做 sum(x * x)
  - 每个 [t, h] 得到一个 sum_sq

ComputeRms
  - rms = sqrt(sum_sq / D + epsilon)
  - 可保存 reciprocal_rms = 1 / rms，减少后续除法

ApplyGamma
  - Q head 使用 q_gamma
  - K head 使用 k_gamma
  - y_norm = x * reciprocal_rms * gamma

ApplyRope
  - 对 y_norm 执行 RoPE
  - 使用 6.4/6.5 中的 half-split RoPE 公式
```

如果 `D` 较大，单个 `[t, h, D]` 无法一次放入 UB，则需要在 controller 或 vec compute 中引入 `dimTile`：

```text
第一遍：按 dimTile 累加 sum_sq
第二遍：重新读取或复用 x tile，执行 x * reciprocal_rms * gamma
```

如果 `D` 可完整放入 UB，则优先采用单 head 完整加载，减少 GM 读次数：

```text
Copy [D] -> Compute sum_sq -> Compute reciprocal_rms -> Apply gamma -> Apply RoPE
```

### 6.4 RoPE 计算语义

RMSNorm 后的 Q/K 会进入 RoPE：

```text
rope_in[t, h, d] = y_norm[t, h, d]
rope_out[t, h, d] = Rope(rope_in[t, h, :], cos_sin[s(t), :])
```

其中 `t` 是 `qkv` 的全局 token 下标，范围为 `0 <= t < T`。`cu_seq_len` 的 shape 为 `[Batch + 1]`，固定采用前缀和输入，也就是常见 cu-seqlens 风格：

```text
cu_seq_len[0] = 0
cu_seq_len[b + 1] = len_0 + len_1 + ... + len_b
cu_seq_len[Batch] = T
```

其中第 `b` 个 batch 的有效 token 数为：

```text
len_b = cu_seq_len[b + 1] - cu_seq_len[b]
```

为了统一公式，后文记：

```text
prefix[b] = cu_seq_len[b], 0 <= b <= Batch
```

对任意全局 token 下标 `t`，其所属 batch `b(t)` 和 batch 内位置 `s(t)` 定义为：

```text
b(t) = max { b | prefix[b] <= t }
     = the unique b such that prefix[b] <= t < prefix[b + 1]

s(t) = t - prefix[b(t)]
```

也就是说，`qkv` 的 T 轴按 batch 顺序拼接。kernel 在处理全局 token `t` 时，需要先根据 `cu_seq_len` 前缀和判断它属于第几个 batch，再得到它在该 batch 内是第几个 token。RoPE 不使用全局下标 `t` 直接索引 `cos_sin`，而是使用 batch 内位置 `s(t)`：

```text
cos_sin_row(t) = cos_sin[s(t), :]
```

`cos_sin` 的 shape 为 `[MaxSeqLen, D]`，其中 `MaxSeqLen` 的实际语义是最大 batch 内 token 数：

```text
MaxSeqLen = max_{0 <= b < Batch} (prefix[b + 1] - prefix[b])
```

由于 `cu_seq_len` 是运行期输入，host 侧无法稳定校验 `cos_sin.shape[0] == MaxSeqLen`，因此首期只要求静态校验 `cos_sin.shape[1] == D`，并按合法输入假设 `0 <= s(t) < MaxSeqLen`。

当前按以下原始方案描述 RoPE。设：

```text
half = D / 2
pos = s(t)
cos_sin[pos, 0 : half]      = cos[pos, 0 : half]
cos_sin[pos, half : D]      = sin[pos, 0 : half]
```

用户给出的原始公式以 `D = 128` 为例，其中 `64 = D / 2`：

```text
0 <= i < D / 2:
    rope_out_i = y_i * cos_i - y_(i + 64) * sin_i

D / 2 <= i < D:
    rope_out_i = y_i * cos_i + y_(i - 64) * sin_i
```

由于 `cos_sin` 的前 `D / 2` 个元素是 `cos`、后 `D / 2` 个元素是 `sin`，第二段公式中的 `cos_i`、`sin_i` 在实现上应理解为对前半段 cos/sin 的折返索引。泛化到任意偶数 `D` 后，可执行定义为：

```text
0 <= i < half:
    pos = s(t)
    c = cos_sin[pos, i]
    sinv = cos_sin[pos, half + i]
    rope_out[t, h, i] = y_norm[t, h, i] * c - y_norm[t, h, i + half] * sinv

half <= i < D:
    j = i - half
    pos = s(t)
    c = cos_sin[pos, j]
    sinv = cos_sin[pos, half + j]
    rope_out[t, h, i] = y_norm[t, h, i] * c + y_norm[t, h, j] * sinv
```

等价地，对每个 pair index `j in [0, half)`，一次处理低半区和高半区两个元素：

```text
x_low  = y_norm[t, h, j]
x_high = y_norm[t, h, j + half]
pos    = s(t)
c      = cos_sin[pos, j]
sinv   = cos_sin[pos, half + j]

out_low  = x_low  * c - x_high * sinv
out_high = x_high * c + x_low  * sinv

rope_out[t, h, j]        = out_low
rope_out[t, h, j + half] = out_high
```

这个 pair 形式是向量化实现的首选公式，因为它避免按 `i` 分支判断，并且能保证 `x_low/x_high` 在覆盖写回前都已经读出。

### 6.5 RoPE 向量化公式

AIV 向量化时，可以把 `D` 维拆成两个长度为 `half` 的向量：

```text
Y0 = y_norm[t, h, 0 : half]
Y1 = y_norm[t, h, half : D]
pos = s(t)
C  = cos_sin[pos, 0 : half]
S  = cos_sin[pos, half : D]
```

向量化 RoPE 公式：

```text
O0 = Y0 * C - Y1 * S
O1 = Y1 * C + Y0 * S
```

输出拼接：

```text
rope_out[t, h, 0 : half] = O0
rope_out[t, h, half : D] = O1
```

对于 AIV MicroAPI 或 AscendC vector 实现，推荐的寄存器/UB 计算顺序为：

```text
load y_low
load y_high
load cos
load sin

tmp0 = y_low  * cos
tmp1 = y_high * sin
out_low = tmp0 - tmp1

tmp2 = y_high * cos
tmp3 = y_low  * sin
out_high = tmp2 + tmp3

store out_low
store out_high
```

如果使用 FMA 能力，可以进一步写成：

```text
out_low  = fma(-y_high, sin, y_low  * cos)
out_high = fma( y_low,  sin, y_high * cos)
```

实现注意事项：

1. 如果 RoPE 输出覆盖 `y_norm` 原 UB 地址，必须先同时加载 `y_low` 和 `y_high`，再写回 `out_low/out_high`，避免 `out_low` 覆盖后影响 `out_high` 计算。
2. 当前公式默认 RoPE 作用于完整 `D`，因此要求 `D` 为偶数。
3. 当前典型 `D = 128`，offset 为 64；实现仍应使用 tiling data 中的 `headDim / 2`，避免将 64 写死到通用路径中。
4. RoPE 读取 `cos_sin[s(t), :]`，其中 `s(t)` 由 `cu_seq_len` 前缀和推导得到。
5. Q 和 K 当前使用同一套 RoPE 公式。

## 7. AIC Cube Rotation Matmul

本阶段对应用户描述中的 hardama/Cube 矩阵乘部分。它接收 RoPE 输出，将其搬运到 L1，并与 `rotation` 矩阵做 Cube matmul。matmul 结果从 L0C 搬出到 UB，作为后续 quant 的输入。

`rotation` 在接口形态上是可选输入，但当前交付场景必须存在。ACLNN 形参名为 `optionalRotation`，API checker 需要校验 `optionalRotation != nullptr`。本阶段不设计缺省 bypass 路径。

### 7.1 数学语义

`rotation` 是 Cube matmul 的右矩阵：

```text
rotation: [D, D]
```

RoPE 输出逻辑 shape：

```text
rope_out: [T, Nq + Nk, D]
```

当前已确认 `K = D`，不支持将 D 方向拆成多个 `K` block。每个 `[t, h]` 对应一个长度为 `D` 的 row vector，并右乘 `rotation[D, D]`：

```text
rot_out[t, h, j] =
    sum_{k=0}^{D-1} rope_out[t, h, k] * rotation[k, j]

0 <= j < D
```

输出逻辑 shape：

```text
rot_out: [T, Nq + Nk, D]
```

等价矩阵视角：

```text
A_rot: [T * (Nq + Nk), D]
B_rot: [D, D]
C_rot: [T * (Nq + Nk), D]

C_rot = A_rot * B_rot
```

其中 `A_rot` 来自 RoPE 输出，`B_rot` 是 `rotation`。

### 7.2 dtype 与精度

当前已知：

```text
rotation dtype = bf16
```

RoPE 输出来自 RMSNorm + RoPE，按数学语义是 `float32`。Cube matmul 输入类型需要在实现时收敛，首选方案如下：

```text
A_rot: bf16 in L1 NZ format
B_rot: bf16 in L1 NZ format, converted from external ND [D, D] rotation by ND2NZ copy
L0C: float32 accumulation
Fixpipe output to UB: float32 ND [M_tile, D]
```

也就是说，AIV 侧在完成 FP32 RoPE 后，将 `float32` 的 RoPE 结果 cast 到 `bf16`，并通过 VF 相关指令转换为 NZ 布局，再搬运到 L1 给 Cube 使用。`rotation` 作为外部输入，GM layout 为 ND `[D, D]`，由 AIC 侧使用 ND2NZ 搬运到 L1。这样二者以 BF16 NZ 形式进入 Cube，Cube 输出在 L0C 上累加为 `float32`，再通过 Fixpipe 以 ND `[M_tile, D]` 搬到 UB 做后续 quant。

如果后续确认 Cube 支持更高精度输入或有特定硬件格式要求，则以硬件 API 为准更新本节。

### 7.3 RoPE 输出转 NZ

Cube 左矩阵 `A_rot` 来自 AIV 侧 RoPE 输出。为了让 AIC 直接进行矩阵乘，AIV 需要在写入 L1 前完成格式转换：

```text
RoPE output ND layout in UB:
    [M_tile, D]

VF transform:
    ND/BF16 -> NZ/BF16

L1 layout:
    A_rot_l1_nz
```

其中：

```text
M_tile = 本次 matmul 处理的 row 数
row = flatten(t, h)
```

建议 AIV 的 `vec_compute` 中新增阶段：

```text
CastRopeOutToBf16
  - 将 RoPE float32 输出 cast 为 bf16

TransformRopeNdToNz
  - 使用 VF 相关指令将 [M_tile, D] 转为 Cube 期望的 NZ layout
  - 处理 M/K 尾块 padding

CopyRopeNzUbToL1
  - 将 NZ/BF16 左矩阵写入 L1
  - Set AIV->AIC flag
```

注意事项：

1. `K = D` 是首期固定约束。
2. `D` 需要满足 Cube matmul 的 K/N 方向对齐要求；若不满足，需要在 NZ 转换或 rotation 搬运时 padding 到硬件要求的对齐长度。
3. `M_tile` 可能不是 16 对齐，NZ 转换和 L1 padding 必须处理尾块。
4. 如果 RoPE 后续 quant 也需要原始 float32 结果，则不能只保留 bf16/NZ 副本；当前设计假设 rotation matmul 输出才进入 quant。
5. 如果硬件 ND2NZ 搬运接口可以直接将 UB/GM ND 数据搬到 L1 NZ，则优先使用直接 ND2NZ 路径；否则在 VF 中完成重排后再写 L1。

### 7.4 Rotation 矩阵搬运与 L1 常驻

`rotation[D, D]` 是 Cube 右矩阵，由 AIC 侧负责搬运。它可以与 AIV 前处理并行：

```text
AIV:
    RMSNorm -> RoPE -> cast/NZ -> write A_rot to L1

AIC:
    preload/copy rotation to L1
    wait A_rot ready
    launch matmul
```

由于 `rotation` 对不同 token/head 复用，建议尝试常驻 L1：

```text
rotation_bytes = D * D * sizeof(bf16)
```

如果 `rotation_bytes` 在 L1 预算内，且不挤占 A_rot double buffer、输出 buffer、同步 buffer，则采用 resident 策略：

```text
Init 阶段或首个 tile 前：
    Copy rotation GM -> L1 once
    Set rotationReady flag

后续所有 tile:
    复用 rotationL1
```

如果无法常驻，则采用双缓冲搬运：

```text
for each tile or group of tiles:
    Wait rotation buffer free
    Copy rotation GM -> rotationL1[db]
    Launch matmul with rotationL1[db]
```

优先级建议：

1. `D` 较小且 `D*D*sizeof(bf16)` 可控时，优先 L1 常驻。
2. 当前 `K = D`，同一个 `rotation` 会被全部 token/head 复用，常驻收益主要取决于 `T * (Nq + Nk)` 的复用次数。
3. 如果 L1 空间紧张，优先保证 A_rot 的 double buffer 和 matmul pipeline，再退化为 rotation 分块/重复搬运。

### 7.5 Cube matmul 数据流

Cube 侧主要流程建议放在 `qkv_rrc_cube_compute.h`：

```text
Init
  - 分配 A_rot L1 buffer
  - 分配 rotation L1 buffer
  - 分配 L0A/L0B/L0C
  - 如果可以常驻，预留 rotation resident 区域

UpdateGlobalAddr
  - 绑定 rotation GM 地址
  - 绑定后续输出/中间 buffer 地址

PreloadRotation
  - AIC 将 rotation 从 GM ND [D, D] 通过 ND2NZ 搬到 L1
  - 可与 AIV RMSNorm/RoPE 并行

WaitAivRopeNzReady
  - 等待 AIV 完成 A_rot NZ 写入 L1

LaunchRotationMatmul
  - A_rot_l1_nz [M_tile, D] x rotation_l1 [D, D]
  - 累加到 L0C

FixL0CToUb
  - 将 L0C 上的 [M_tile, D] 搬出到 UB
  - 输出 dtype 优先为 float32
  - 输出 layout 固定为 ND [M_tile, D]
  - Set AIC->AIV flag，通知后续 quant 可以读取
```

从矩阵乘角度，一个 tile 的核心参数为：

```text
m = M_tile
n = D
k = D
```

首期区分 `rowTile` 和 `M_tile`：

```text
Hqk = Nq + Nk
Nv  = Nk
Htot = Nq + Nk + Nv = Nq + 2 * Nk

rowTile = tokenTile * Hqk
M_tile  = actualTokenTile * Hqk
```

这里 `Nv = Nk` 是首期 fast path 的显式支持约束；host 仍从 `head_nums=[Nq,Nk,Nv]` 读取三者，并在进入当前 tiling 前校验 `Nv == Nk`。

`rowTile` 是 host tiling 选出的 full tile Q/K row 容量，用于 L1 A buffer、UB output buffer 等容量分配。`M_tile` 是 kernel 运行时当前 tile 实际送入 Cube 的矩阵 M 参数，非 tail tile 中 `M_tile == rowTile`，tail tile 中 `M_tile <= rowTile`。

首期 fast path 采用 kernel 静态 UB/L1 layout，而不是按 shape 动态重分配 buffer。以下数值是 host tiling 计算资源上界时使用的固定常量，不写入 tiling data：

```text
UB reserve            = 40 KiB
UB input one buffer   = 40 KiB
UB output one buffer  = 64 KiB

L1 rotation resident  = 32 KiB
L1 A one buffer       = 64 KiB
L1 A buffer num       = 4
```

其中 UB 容量均表示单 AIV 本地容量。64 KiB UB output one buffer 除了承接 Cube fix 输出，也可以在生命周期不重叠时复用为 RMS/RoPE fp32 scratch。

40 KiB reserve 是单 AIV 本地 scratch。首期 reserve 中需要承载 `gamma_ub`、`cos_sin_ub[2]`、`slot_mapping_ub[2]`、`v_scale_ub`、`v_out_ub[2]`。其中 `cos_sin`、`slot_mapping` 和 `v_out` 均开 double buffer；`v_scale` 读入 UB 后复用，不开 DB。具体子区容量、offset 和对齐规则由 kernel 设计静态定义，不由 tiling 分配，也不写入 tiling data。

Q/K dynamic quant 的 FP8 staging 不占 reserve；它复用 `output DB pool` 中当前 `qk_after_cube_ub` 所在 buffer，在按 row 或 chunk 消费完 FP32 C 后原地复用或直接写 GM/cache。其他标量数组不在 UB 中常驻；Q/K row scale 计算后直接写 `q_scale` 或 `k_scale_cache`。前处理阶段的中间临时量不在 reserve 中静态分配；大块 Q/K fp32 中间量仍按生命周期复用 output DB pool。

静态 UB layout 直接反推每个 AIV 的 token 上限：

```text
Ta = tokenTile / 2

Ta <= floor(40 KiB / (Htot * 128 * sizeof(bf16)))  = floor(160 / Htot)
Ta <= floor(64 KiB / (Hqk  * 128 * sizeof(fp32)))  = floor(128 / Hqk)
```

因此 host tiling 使用：

```text
TaMax = min(8, floor(128 / Hqk), floor(160 / Htot))
tokenTile = 2 * TaMax
```

若 `TaMax < 1`，该 shape 不进入当前 fast path。host tiling 同时按 full tile 容量直接保证：

```text
rowTile <= 256
```

由于任意运行时 `M_tile <= rowTile`，该约束可以保证所有 tile 的 Cube M 都不超过 256。该约束用于在 256 KiB L0C 上保留 C buffer double buffer。`D=128` 且 Cube 输出为 FP32 时，单个 C buffer 预算为：

```text
align_up(M_tile, 16) * 128 * sizeof(float32) <= 128 KiB
```

首期不在 AIC 内部继续切 `M_sub`，也不关闭 L0C double buffer；`rowTile > 256` 的方案仅作为后续扩展。

### 7.6 AIV/AIC 流水与同步

该阶段可复用 `grouped_matmul_finalize_routing` 的 AIV/AIC 协同方式。

推荐双缓冲流水：

```text
Tile i:
    AIV 计算 Q/K RMSNorm/RoPE，并写 A_rot_l1_nz[buf]
    AIV Set AIV_AIC_A_READY[buf]
    AIV 对 V 做 v * v_scale 量化并写 v_cache

    AIC Wait AIV_AIC_A_READY[buf]
    AIC 使用 rotationL1 和 A_rot_l1_nz[buf] 做 matmul
    AIC Fix L0C -> UB output[buf]
    AIC Set AIC_AIV_C_READY[buf]

    AIV Wait AIC_AIV_C_READY[buf]
    AIV 对 UB output[buf] 做 Q/K dynamic quant 和写回
```

同时，AIC 可在 Tile 0 前或 Tile 0 期间搬运 `rotation`：

```text
AIC PreloadRotation
AIV RMSNorm/RoPE Tile 0
```

同步点建议：

| 同步方向 | 含义 |
|----------|------|
| AIC -> AIV | rotation resident buffer 初始化完成，可选；若 AIV 不依赖 rotation 可不需要。 |
| AIV -> AIC | A_rot NZ 已写入 L1，Cube 可读。 |
| AIC -> AIV | L0C 已 fix 到 UB，quant 可读。 |
| AIV -> AIC | quant 已消费 UB output，相关 buffer 可复用。 |

### 7.7 Controller 分块影响

引入 rotation matmul 后，tile 内部使用 Q/K 的 flatten row 维度：

```text
row = t * (Nq + Nk) + h
```

反解：

```text
h = row % (Nq + Nk)
t = row / (Nq + Nk)
```

首期 controller 不按 `row` 做跨核分发，而是按连续 token tile 分发：

```text
Hqk = Nq + Nk
tokenTile % 2 == 0

for tileStart in token tiles:
    actualTokenTile = min(tokenTile, T - tileStart)
    rowTile = tokenTile * Hqk
    M_tile = actualTokenTile * Hqk
    process tokens [tileStart, tileStart + actualTokenTile)
```

这里 `rowTile` 是 full tile 容量，通常由 tiling data 传入；`M_tile` 是本次循环的实际 Cube M，tail 时会小于 `rowTile`。

两个 AIV 对主 tile 做严格 token 平分：

```text
tokenTilePerAiv = tokenTile / 2
AIV0 tokens: [tileStart, tileStart + tokenTilePerAiv)
AIV1 tokens: [tileStart + tokenTilePerAiv, tileStart + tokenTile)
```

最后一个 tail tile 允许 `actualTokenTile < tokenTile`，tail 内部可用 `ceil/floor` 分配。首期不为奇数 tail 引入 Q/K flatten row 均衡切分。

由于 `K = D`，每个 `[t, h]` 只对应一个 rotation matmul row。首期按 token 内完整 Q/K heads 处理，随后将完整 D 向量转 NZ 并交给 Cube。按 Q/K flatten row 在两个 AIV 之间均衡切分会解耦 Q/K 与 V 的生命周期，属于后续扩展。

首期也不按 `coreGroupNum` 重调 token tile 大小。若 `ceil(T / tokenTile) % coreGroupNum != 0`，最后一个 core wave 可能只有少数核组仍有 token tile，需要接受该尾波不均衡。后续可以评估 tail two-phase rebalance 或 uniform smaller token tile，但前者需要 large/small 两套 tile 调度参数，后者依赖 profiling 才能确定性能下限且不能保证消除最后一个尾块，因此均不进入首版。

RoPE 的 `cos_sin` 行索引依赖 `cu_seq_len`，因此 controller 或 vec compute 在展开 `row -> (t,h)` 后，还需要得到 `t -> s(t)` 的映射。`cu_seq_len` 已固定为前缀和输入，kernel 直接把 `cu_seq_len` 视为逻辑前缀和 `prefix` 使用。首期可采用以下任一实现方式：

1. 在处理 token tile 前，顺序扫描 `cu_seq_len` 维护当前 batch、当前 batch 的起始全局 token 下标 `cu_seq_len[b]`，从而为 tile 内每个 `t` 计算 `s(t) = t - cu_seq_len[b]`。
2. 如果 `Batch` 较小，可在 UB/临时标量中维护 `prefix`，对每个 `t` 查找满足 `prefix[b] <= t < prefix[b + 1]` 的 `b`。
3. 若 tile 按连续 `t` 切分，则推荐让 tile 不跨或少跨 batch 边界；跨 batch 时只需要在 `t == prefix[b + 1]` 处更新 `b` 和 `prefix[b]`。

该映射只影响 AIV 前处理读取 `cos_sin[s(t), :]`，不改变 Cube flatten row 组织，也不改变后处理根据 `slot_mapping[t]` 写 cache 的规则。

### 7.8 Rotation Dense 处理策略

`rotation` 可以来自 Hadamard 场景，但它作为外部输入矩阵传入。当前交付不考虑 FWHT 优化，也不在 kernel 内生成、归一化或重排 Hadamard。首期统一按普通 dense `rotation[D, D]` 走 Cube matmul，确保外部传入的任意 `[D,D]` bf16 矩阵都按输入值精确生效。

#### 7.8.1 Dense Cube 基线方案

首期最稳妥方案仍按普通 dense matmul 实现：

```text
C = Y @ H
Y: [M_tile, D]
H: [D, D]
C: [M_tile, D]
```

优点：

1. 直接复用 Cube matmul 框架。
2. 对任意外部输入 `rotation[D,D]` 都正确。
3. 实现风险最低。

缺点：

1. 计算复杂度是 `O(M_tile * D * D)`。
2. 需要搬运或常驻 `D * D` 的 rotation 矩阵。
3. 如果 `rotation` 来自 Hadamard 场景，其 `+1/-1` 结构不会被利用。

#### 7.8.2 FWHT 后续优化方向

当前不启用本方案。后续如果业务场景固定要求 `rotation` 是标准 Sylvester Hadamard，或者可以表示为标准 Hadamard 加简单行列置换/符号翻转，才可以考虑用 butterfly 计算：

```text
for step = 1; step < D; step *= 2:
    for base in range(0, D, 2 * step):
        for j in range(0, step):
            a = x[base + j]
            b = x[base + j + step]
            x[base + j]        = a + b
            x[base + j + step] = a - b
```

复杂度：

```text
O(M_tile * D * log2(D))
```

潜在优化收益：

1. 不需要读取完整 `rotation[D,D]`。
2. 不需要 Cube matmul。
3. 不需要 RoPE 输出 FP32 -> BF16 后交给 Cube；可以在 AIV/UB 中保持 FP32 做 Hadamard，减少精度损失。
4. 对常见 `D = 128`，dense 每 row 约 `128 * 128 = 16384` 次乘加；FWHT 每 row 约 `128 * 7` 级 butterfly，计算量大幅下降。

代价和约束：

1. 要求 `D` 是 2 的幂。
2. 必须确认 Hadamard 的具体排列、归一化系数、行列符号翻转。
3. FWHT 更适合 AIV 做加减 butterfly，会改变当前“Cube 执行 rotation matmul”的流水结构。
4. 如果目标是保持 AIC 利用率或复用已有 Cube 框架，FWHT 需要作为独立优化路径评估。
5. 因为当前 `rotation` 是外部输入，若未确认其标准形式，FWHT 不能替代 dense `y @ rotation`。

#### 7.8.3 折中优化

在不改变首期 Cube 框架的前提下，可以做较低风险优化：

1. `rotation` 常驻 L1，避免重复 GM 搬运。
2. 对外部输入的 `rotation` 只做一次 ND2NZ 搬运并复用，避免每个 tile 重复读取 GM。
3. 如果 Cube 不支持利用 `+1/-1` 减少乘法，则即使矩阵是 Hadamard，dense Cube 路径也只能优化搬运，不能降低 MMAD 计算量。

结论：首期固定采用 dense Cube 路径；当 `rotation` 来自 Hadamard 场景时，首期最直接的优化仍只是 **rotation 常驻 L1** 和 **避免重复搬运**。FWHT 只作为未来在矩阵形式充分受控时的优化方向。

### 7.9 已确认约束

1. `D` 的典型值为 `128`，满足当前 Cube matmul 对齐要求。
2. `rotation` 完全来自外部输入，可以来自 Hadamard 场景，但 kernel 不生成、不归一化、不假设标准排列。测试 golden 必须直接使用外部传入的 `rotation[D,D]` 参与 `y @ rotation`，首期测试数据中的 `rotation` value 可以先使用随机值。

## 8. AIV 后处理与写回

后处理阶段负责：

1. AIV 在发送 `A_READY` 后对 V 矩阵做 FP8 量化，并根据 `slot_mapping` 提前写入 `v_cache`。
2. AIV 后处理阶段对 Cube 输出的 Q/K 矩阵做 D 方向动态 scale FP8 量化。
3. 将 K 量化结果和 K scale 根据 `slot_mapping` 写入 `k_cache` 和 `k_scale_cache`。
4. 将 Q 量化结果和 Q scale 写入 `q_out` 和 `q_scale`。

所有量化 round 均采用 **RINT** 原则。

### 8.1 后处理输入状态

进入 Q/K 后处理时，数据状态为：

```text
qk_after_cube: [T, Nq + Nk, D]
```

其中：

- `qk_after_cube` 是 Cube rotation matmul 从 L0C 搬出到 UB 后的 Q/K 矩阵。
- `qk_after_cube` 的 head 维仍按 `[Q, K]` 排列：

  ```text
  Q: qk_after_cube[:, 0 : Nq, :]
  K: qk_after_cube[:, Nq : Nq + Nk, :]
  ```

V 不等待 Cube 输出。首期在 AIV 前处理阶段读取 V；V cache 写回属于后处理语义，但 AIV 在发送 `A_READY` 后即可执行 `v * v_scale` 量化并写入 `v_cache`，以释放输入 UB buffer。V 的提取来源：

```text
h_qkv = Nq + Nk + nv
v_in[t, nv, d] = qkv[t, h_qkv, d]
```

### 8.2 V 分支量化

对 V 矩阵执行量化：

```text
v_out[t, nv, d] = round_to_e4m3fn_rint(v_in[t, nv, d] * v_scale[nv])
```

`v_scale` 已确认对 V head 生效：

```text
v_scale: [Nv]
```

V 量化输出 dtype：

```text
v_out: float8_e4m3fn
```

实现注意事项：

1. V 输入来自 `bf16` qkv，量化前按 `float32` 计算描述。
2. `round_to_e4m3fn_rint` 表示按 RINT 舍入到 FP8 E4M3FN。
3. `v_scale` 为输入，不产生额外 V scale 输出。
4. `v_scale` 是量化乘子，不是除数；因此 `v_scale == 0` 合法，结果按公式量化为 0。

### 8.3 V cache 写回

对 T 轴索引得到 token 下标 `it`：

```text
slot = slot_mapping[it]
block_num_idx = slot / BlockSize
block_size_idx = slot % BlockSize
```

将 V 量化结果写入 `v_cache`：

```text
v_cache[block_num_idx, nv, block_size_idx, d] = v_out[it, nv, d]
```

逻辑循环：

```text
for it in [0, T):
    slot = slot_mapping[it]
    block_num_idx = slot / BlockSize
    block_size_idx = slot % BlockSize
    for nv in [0, Nv):
        for d in [0, D):
            v_cache[block_num_idx, nv, block_size_idx, d] = v_out[it, nv, d]
```

`v_cache` 是输入输出复用空间，写回必须允许 input/output alias。

### 8.4 Q/K 动态 scale 量化

定义 Cube 输出对应的数学张量 `C`：

```text
C shape: [T, Nq + Nk, D]
```

下标范围为：

```text
t in T_set, h in H_qk_set, j in D_set
```

对 `C` 在 D 方向求绝对值最大值，得到数学张量 `A`：

```text
A[t,h] = max over j in [0, D): abs(C[t,h,j])
```

根据 FP8 E4M3FN 最大有限值计算 scale。当前 `fp8_max` 近似取 `448`。定义数学张量 `Scale`：

```text
fp8_max ~= 448
```

```text
Scale[t,h] = A[t,h] / fp8_max
```

量化前的归一化中间结果定义为：

```text
Z[t,h,j] = C[t,h,j] / Scale[t,h]
```

最终 FP8 E4M3FN 量化结果定义为：

```text
Q_fp8[t,h,j] = cast_e4m3fn_sat_rint(Z[t,h,j])
```

数学符号与工程张量的对应关系：

| 数学符号 | 工程张量或中间量 |
|----------|------------------|
| `C` | `qk_after_cube` |
| `A` | `max_abs` |
| `Scale` | `scale_out` |
| `Q^fp8` | `qk_quant` |

输出逻辑 shape：

```text
qk_quant: [T, Nq + Nk, D]
scale_out:[T, Nq + Nk]
```

其中：

- `h in [0, Nq)` 对应 Q。
- `h in [Nq, Nq + Nk)` 对应 K。
- 所有 round 都使用 RINT。
- `fp8_max` 当前按近似值 `448` 设计。如果工程内已有统一常量或工具函数，后续可以替换为工程定义。
- `scale_out` 是对外写出的 scale。若某个 row 的绝对值最大值为 0，则该 row 的 scale 也为 0。首期不在本算子中额外定义零行保护语义，直接对齐 arch35 `dynamic_quant` 的 FP8 E4M3FN 量化路径。

`ops-nn/quant/dynamic_quant` 的 arch35 FP8 E4M3FN 路径可参考以下实现点：

1. `dynamic_quant_apt.cpp` 在 common full-load、large-shape、MOE full-load、MOE large-shape 路径中均以 `isSymmetrical = true` 实例化。
2. `dynamic_quant_regbase_full_load.h` 中 `SetMaxValue` 对 `fp8_e4m3fn_t` 设置 `maxValue = 1.0 / FP8_E4M3FN_MAX_VALUE`，`ComputeScaleVF` 计算 `scale = reduce_max(abs(x)) * maxValue`，`ComputeYVF` 直接执行 `Div(input, scale)`，再用 `RoundMode::CAST_RINT`、`SatMode::SAT` cast 到 FP8。
3. `dynamic_quant_regbase_large_shape_db.h` 中同样设置 `scaleMaxValue = 1.0 / FP8_E4M3FN_MAX_VALUE`，先把 `abs(x)` 乘 `scaleMaxValue` 后 reduce 得到 scale，随后 `ComputeYVF` 直接执行 `Div(input, scale)`，再 cast 到 FP8。

因此本文档不能写成“scale 为 0 时 kernel 不执行”或“强制输出全 0”。若后续希望对 `max_abs == 0` 加保护分支，那是相对于 arch35 `dynamic_quant` 的显式语义偏离，需要单独确认并在 golden 中体现。

### 8.5 K cache 与 K scale 写回

K 分支对应：

```text
h = Nq + nk
0 <= nk < Nk
```

复用 T 轴的 `slot_mapping`：

```text
slot = slot_mapping[it]
block_num_idx = slot / BlockSize
block_size_idx = slot % BlockSize
```

写入 K cache：

```text
k_cache[block_num_idx, nk, block_size_idx, d] = qk_quant[it, Nq + nk, d]
```

写入 K scale cache：

```text
k_scale_cache[block_num_idx, nk, block_size_idx, 0] = scale_out[it, Nq + nk]
```

逻辑循环：

```text
for it in [0, T):
    slot = slot_mapping[it]
    block_num_idx = slot / BlockSize
    block_size_idx = slot % BlockSize
    for nk in [0, Nk):
        k_scale_cache[block_num_idx, nk, block_size_idx, 0] = scale_out[it, Nq + nk]
        for d in [0, D):
            k_cache[block_num_idx, nk, block_size_idx, d] = qk_quant[it, Nq + nk, d]
```

`k_cache` 和 `k_scale_cache` 都是输入输出复用空间，写回必须允许 input/output alias。

### 8.6 Q 输出写回

Q 分支对应：

```text
h = nq
0 <= nq < Nq
```

写入普通输出 `q_out` 和 `q_scale`：

```text
q_out[it, nq, d] = qk_quant[it, nq, d]
q_scale[it, nq]  = scale_out[it, nq]
```

逻辑循环：

```text
for it in [0, T):
    for nq in [0, Nq):
        q_scale[it, nq] = scale_out[it, nq]
        for d in [0, D):
            q_out[it, nq, d] = qk_quant[it, nq, d]
```

### 8.7 后处理实现建议

V 分支建议拆成读取暂存和提前后处理两步：前处理函数读取 qkv 中的 V；AIV 发送 `A_READY` 后立即执行 `v * v_scale` 量化并写回 `v_cache`。Q/K 后处理建议放在 `qkv_rrc_vec_compute.h`，由 AIV 在收到 AIC 的 `C_READY` flag 后执行。

建议拆成以下函数：

```text
QuantVAndWriteCache
  - 在 A_READY 后消费暂存的 V
  - 按 v * v_scale 做 RINT FP8 E4M3FN 量化
  - 根据 slot_mapping 写 v_cache

ComputeQKAbsMax
  - 对 qk_after_cube 在 D 方向求 max_abs

ComputeQKScale
  - scale_out = max_abs / fp8_max

QuantQK
  - qk_div = qk_after_cube / scale_out
  - qk_quant = cast_to_e4m3fn_sat_rint(qk_div)

WriteKCacheAndScale
  - 根据 slot_mapping 写 k_cache 和 k_scale_cache

WriteQOutAndScale
  - 顺序写 q_out 和 q_scale
```

首期 UB 不要求同时保存完整 V tile 到后处理阶段。controller 按 token tile 分块：

```text
tile_qk: [T_tile, Nq + Nk, D]
tile_v:  [T_tile, Nv, D]
```

其中 Q/K 的动态 scale 需要完整 D 方向参与，因此单个 quant tile 必须覆盖完整 `D`，或采用两遍 D tile：

```text
第一遍：D tile reduce max_abs
第二遍：重新读取/复用 D tile 执行 quant 和写回
```

首选实现仍是让单个 `[it, head, D]` 完整进入 UB，避免二次读取和跨 tile reduce。

### 8.8 后处理边界语义

1. V 分支使用 `v * v_scale[nv]` 后 RINT 转 FP8 E4M3FN，`v_scale == 0` 合法。
2. Q/K 分支使用 `max_abs / 448` 作为对外 scale。若 `max_abs == 0`，对外 scale 写 `0`，量化阶段仍按 arch35 `dynamic_quant` 的 FP8 E4M3FN 路径执行 `Div -> Cast fp8`，不额外增加跳过分支。

## 9. 当前未决项汇总

截至当前版本，主体计算链路、AIV/AIC/controller 分工、cache 写回和 arch35 交付件已形成首期实现方案，当前无阻塞首期实现的未决项。
