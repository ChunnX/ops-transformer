# 交付件纵览设计

## 1. 目标

本文描述 `QkvRmsNormRopeCacheWithKScale` 首期 arch35 交付需要新增或修改的文件集合，以及每类文件的职责边界。具体计算公式见 `01_operator_proto_and_formula.md`，kernel 框架见 `05_kernel_design.md`。

## 2. 建议目录

算子属于 attention 前处理和 KV cache 更新融合算子，建议落在 attention 目录：

```text
attention/qkv_rms_norm_rope_cache_with_kscale/
```

建议结构：

```text
attention/qkv_rms_norm_rope_cache_with_kscale/
  op_api/
  op_host/
  op_kernel/
    arch35/
      qkv_rms_norm_rope_cache_with_kscale_apt.cpp
      qkv_rms_norm_rope_cache_with_kscale_tiling_data.h
      qkv_rms_norm_rope_cache_with_kscale_tiling_key.h
      basic_block/
  tests/
  examples/
  docs/
```

如果仓内 attention 类算子使用其他固定目录规范，应优先遵循现有规范；本文只定义交付件类型和职责。

## 3. API 交付件

建议新增：

```text
op_api/aclnn_qkv_rms_norm_rope_cache_with_kscale.h
op_api/aclnn_qkv_rms_norm_rope_cache_with_kscale.cpp
```

API 职责：

1. 暴露 ACLNN workspace size/executor 接口。
2. 保持 `rotation`、`v_scale` 为 optional 形式；ACLNN 形参命名使用 `optionalRotation`、`optionalVScale`。
3. 在当前场景中调用 checker 强制 `optionalRotation != nullptr`、`optionalVScale != nullptr`。
4. 传递必选 `head_nums` ListInt 属性，ACLNN 侧建议用 `const aclIntArray *headNums`。
5. 首期 checker/tiling 只接受 `head_nums[2] == head_nums[1]`，即 `Nv == Nk`。
6. 传递 `epsilon` 属性。
7. 处理 in-place 输出 tensor：`k_cache`、`v_cache`、`k_scale_cache`。

建议 API 形态参考：

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
    uint64_t *workspaceSize,
    aclOpExecutor **executor);
```

实际签名需要按仓内 ACLNN 规范调整。

## 4. op_host 交付件

建议新增：

```text
op_host/qkv_rms_norm_rope_cache_with_kscale_def.cpp
op_host/qkv_rms_norm_rope_cache_with_kscale_infershape.cpp
op_host/qkv_rms_norm_rope_cache_with_kscale_tiling.cpp
op_host/op_tiling/arch35/qkv_rms_norm_rope_cache_with_kscale_tiling.h
op_host/op_tiling/arch35/qkv_rms_norm_rope_cache_with_kscale_tiling.cpp
```

职责：

1. 定义输入输出、dtype、format、attr，其中 `head_nums` 为 REQUIRED `ListInt`。
2. infer shape：输出 `q_out [T,Nq,D]`、`q_scale [T,Nq]`，cache 输出与输入同 shape。
3. checker：shape/dtype/optional-but-required 约束。
4. tiling：先只提供 kernel 必需元信息，具体 tiling 策略待定。
5. 不交付 `op_host/config` 配置文件；参考 `grouped_matmul_finalize_routing` 的 arch35 方案，通过 op_host 代码路径和 tiling key 约束到 arch35。

Checker 关键项：

```text
qkv dtype bf16, rank 3
q_gamma/k_gamma dtype fp32, shape [D]
cos_sin dtype fp32, shape [MaxSeqLen,D]
slot_mapping dtype int32, shape [T]
cu_seq_len dtype int32, shape [Batch + 1]
head_nums required ListInt, length 3, values [Nq,Nk,Nv], all positive
Nv == Nk in first arch35 fast path
qkv.shape[1] == Nq + Nk + Nv
k_cache.shape[1] == Nk, v_cache.shape[1] == Nv, k_scale_cache.shape[1] == Nk
rotation dtype bf16, shape [D,D], non-null
v_scale dtype fp32, shape [Nv], non-null
k_cache/v_cache/q_out dtype float8_e4m3fn
k_scale_cache/q_scale dtype fp32
```

## 5. op_kernel 交付件

建议新增：

```text
op_kernel/arch35/qkv_rms_norm_rope_cache_with_kscale_apt.cpp
op_kernel/arch35/qkv_rms_norm_rope_cache_with_kscale_tiling_data.h
op_kernel/arch35/qkv_rms_norm_rope_cache_with_kscale_tiling_key.h
op_kernel/arch35/basic_block/qkv_rrc_controller.h
op_kernel/arch35/basic_block/qkv_rrc_vcv_basic_block.h
op_kernel/arch35/basic_block/qkv_rrc_vec_compute.h
op_kernel/arch35/basic_block/qkv_rrc_cube_compute.h
op_kernel/arch35/basic_block/qkv_rrc_vf.h
```

kernel 框架先参考 `grouped_matmul_finalize_routing` 的 arch35 `weight_quant_basic_block`：

```text
apt.cpp
  -> REGISTER_TILING_DEFAULT
  -> GET_TILING_DATA_WITH_STRUCT
  -> KERNEL_TYPE_MIX_AIC_1_2
  -> Controller.Init(...)
  -> Controller.Process()

controller
  -> 解析 tiling 和 GM 地址
  -> 分发 basic block

vcv_basic_block
  -> AIV 前处理
  -> AIC Cube matmul
  -> AIV 后处理

vec_compute
  -> RMSNorm/RoPE/V quant/QK quant/cache write
  -> V cache write is a postprocess action, but may run immediately after A_READY because it does not depend on Cube output

cube_compute
  -> rotation ND2NZ/L1 resident/matmul/Fixpipe
```

## 6. 测试交付件

建议新增：

```text
tests/ut/op_host/test_qkv_rms_norm_rope_cache_with_kscale_infershape.cpp
tests/ut/op_host/test_qkv_rms_norm_rope_cache_with_kscale_tiling.cpp
tests/st/test_qkv_rms_norm_rope_cache_with_kscale.py
examples/test_aclnn_qkv_rms_norm_rope_cache_with_kscale.cpp
```

测试职责：

1. op_host UT 覆盖 checker、infer shape、tiling data 基础字段。
2. kernel ST 覆盖端到端 golden。
3. ACLNN example 覆盖 workspace/executor 调用链。
4. 必须覆盖缺失 `rotation` 和缺失 `v_scale` 报错。

## 7. 文档交付件

建议新增：

```text
docs/aclnnQkvRmsNormRopeCacheWithKScale.md
```

文档包含：

1. API 说明。
2. 输入输出 shape/dtype。
3. `rotation`、`v_scale` optional-but-required 说明。
4. `cu_seq_len` 前缀和语义。
5. `head_nums=[Nq,Nk,Nv]` 必选属性说明。
6. `epsilon` 属性说明。
7. 支持平台 arch35。

## 8. 不交付范围

首期不交付：

1. arch34/910B/910_93 kernel。
2. `rotation == nullptr` fallback。
3. `v_scale == nullptr` fallback。
4. FWHT/Hadamard butterfly 优化。
5. `cu_seq_len` 长度模式或双模式 tiling key。
6. `rotaryDim < D`。
7. `Nv != Nk` 正常 kernel 路径。
