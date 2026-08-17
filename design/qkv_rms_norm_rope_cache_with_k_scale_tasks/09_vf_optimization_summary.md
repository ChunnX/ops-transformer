# VF 优化总结

## 1. 背景

本文总结 `QkvRmsNormRopeCacheWithKScale` 在本轮 kernel 设计和实现中的 VF 优化内容。算子无关的通用 VF 优化规则见：

```text
../vf_optimization_rules.md
```

本文只记录这些规则在当前算子中的具体落地，重点覆盖：

1. Q/K 前处理 VF 的展开与寻址方式调整。
2. Q/K 共享 `rotation` 后的前处理布局约束。
3. V scale、Q/K dynamic quant VF 的 caller-side layout dispatch 与 stride 参数化实现。
4. MicroAPI load/store 接口使用规范。
5. 已完成的构建和 torch_extension golden 验证。

目标文件：

```text
ops-transformer/posembedding/qkv_rms_norm_rope_cache_with_k_scale/op_kernel/arch35/qkv_rms_norm_rope_cache_with_k_scale_vf.h
```

相关调用侧：

```text
ops-transformer/posembedding/qkv_rms_norm_rope_cache_with_k_scale/op_kernel/arch35/qkv_rms_norm_rope_cache_with_k_scale_vec.h
ops-transformer/posembedding/qkv_rms_norm_rope_cache_with_k_scale/op_kernel/arch35/qkv_rms_norm_rope_cache_with_k_scale_cube.h
```

## 2. 设计约束

### 2.1 Q/K 共享 rotation

当前原型收敛为单个外部 `rotation`：

```text
rotation: [D, D]
```

AIC/Cube 侧约束：

1. `rotation` 从 GM 预载到 L1 的 `rotationL1Nz_`。
2. tile loop 前通过一次 MTE1 加载到 L0B，并在 loop 内常驻复用。
3. D=128 BF16 rotation 大小为 `128 * 128 * 2 = 32 KiB`。
4. 当前阶段不调整原 L1 静态分配，但不再执行第二份 rotation 的 ND2NZ 或 MTE1。
5. 不在 tile 内重复加载 rotation。

### 2.2 Q/K 前处理使用一个 compact A buffer

Q/K head 数量可能差异显著，因此不能把 AL1 active A buffer 固定等分成两个 slot。当前设计保持一个 active logical A buffer，并按实际 rows 紧凑切成两个 segment：

```text
activeA = aRotL1Pool_[(aRotL1BufferUseId_ % A_ROT_L1_BUFFER_NUM)
                      * QKV_K_SCALE_A_ROT_L1_LOGICAL_BUFFER_ELEMENTS]

qRows = tile.cubeTokenSize * numQHeads
kRows = tile.cubeTokenSize * numKHeads

Q segment base = activeA
K segment base = activeA[NzMatrixElements(qRows)]
```

其中 `A_ROT_L1_BUFFER_NUM` 由 BasicBlock 根据 AL1 pool 总容量与 logical slot 容量局部推导，不是
`common.h` 中按资源新增的全局常量。

Vec 侧写入和 Cube 侧读取必须使用同一 offset 规则：

```text
K base = NzMatrixElements(tile.cubeTokenSize * qHeadNum)
```

本轮明确不做：

1. 不固定 Q/K 两个等大小 slot。
2. 不新增 Q/K offset helper/accessor。
3. 不新增 `Q_A_READY` / `K_A_READY` 等 CrossCore flag。
4. 不拆分 Q/K 的本地 AL1 event。

## 3. QK VF 展开优化

### 3.1 原始结构

优化前，`QkRmsNormRopeD128SegmentNzVf` 通过多个 helper 分层：

```text
QkRmsNormRopeD128SegmentNzVf
  -> QkRmsNormRopeD128HeadRangeNzVf
    -> QkRmsNormRopeD128RowNzVf
      -> StoreRopeBf16RowToNzByScatter
        -> ScatterRopeBf16HalfToNz
```

这类封装会让循环、寄存器生命周期、load/store 位置和地址计算不够直观，也不利于按具体 tile/head 场景调整性能。

### 3.2 展开后的结构

优化后，Q/K 的 token/head 遍历集中在 `QkRmsNormRopeD128SegmentNzVfImpl` 内，不再保留 range/row/store/scatter 多层遍历 helper。RMSNorm、RoPE cast 和单 row scatter 仍保留为紧凑的 `__simd_callee__` 计算 helper。`QKV_LAYOUT/Q_OUT_LAYOUT` 是 Vec 调用侧的模板常量；调用侧用 `if constexpr` 计算 input/output token/head stride，VF 只按显式 stride 寻址，不走运行时 layout 分支。

展开后的逻辑顺序为：

1. 函数入口加载循环无关数据。
2. token 维循环。
3. head 维循环。
4. 对每个 `[token, head, D=128]` row：
   - BF16 low/high half load。
   - BF16 -> FP32 cast。
   - RMSNorm reduce。
   - gamma multiply。
   - RoPE low/high half 计算。
   - FP32 -> BF16 cast。
   - 通过 NZ scatter index 和 BF16 高位 mask 直接 scatter 到 NZ layout，不再执行 pack。

## 4. 循环变量与地址计算优化

### 4.1 循环变量只能使用 `uint16_t`

`QkRmsNormRopeD128SegmentNzVfImpl` 内的 token/head 循环变量只能使用 `uint16_t`：

```cpp
for (uint16_t tokenIdx = 0U; tokenIdx < tokenSize; ++tokenIdx) {
    ...
    for (uint16_t headIdx = 0U; headIdx < headSize; ++headIdx) {
        ...
    }
}
```

约束：

1. 当前 tiling 下 `tokenSize` 和 `headSize` 都必须落在 `uint16_t` 可表示范围内。
2. VF 热路径显式 `for` 循环不允许使用 `uint32_t`、`int32_t`、`int`、`size_t` 等作为循环变量。
3. 如果未来范围超过 `uint16_t`，应拆分循环、调整 tiling，或拒绝进入当前实现，不能通过更换循环变量类型解决。

注意：`uint16_t` 只用于循环控制；地址计算优先在当前循环层级或具体 load/store/scatter 使用点附近直接表达，避免维护跨迭代传播的地址状态：

```cpp
auto inputAddrReg = MicroAPI::CreateAddrReg<bfloat16_t>(
    tokenIdx, inputTokenStride, headIdx, inputHeadStride);
Load(..., input, inputAddrReg);

rowOutOffset = (tokenIdx * outputTokenStride + headIdx * outputHeadStride) * C0;
Scatter(..., outNz + rowOutOffset, ...);
```

### 4.2 禁止累加式指针更新

优化前存在类似写法：

```cpp
tokenInput += qkvStrideT;
tokenCosSin += QKV_K_SCALE_D128_FLOAT_REPEAT_SIZE * 2U;
tokenOutNz += headSize * QKV_K_SCALE_NZ_C0;
rowOutNz += QKV_K_SCALE_NZ_C0;
```

优化后改为在具体 load/store/scatter 地址中使用 loop id 和 stride 直接计算：

```cpp
const uint32_t tokenCosSinOffset = static_cast<uint32_t>(tokenIdx) * QKV_K_SCALE_D128_FLOAT_REPEAT_SIZE * 2U;

const uint32_t inputOffset = tokenIdx * inputTokenStride + headIdx * inputHeadStride;
const uint32_t rowOutOffset =
    (tokenIdx * outputTokenStride + headIdx * outputHeadStride) * QKV_K_SCALE_NZ_C0;
```

收益：

1. 地址来源更直接，方便审查 Q/K segment 和 token/head 对应关系。
2. 减少循环尾部指针状态维护。
3. 避免后续插入 early continue 或拆分路径时破坏累加状态。

## 5. Load 外提策略

### 5.1 函数级外提

以下数据与 token/head 都无关，外提到函数入口：

```cpp
gammaLow
gammaHigh
nzIndex
mask64
maskFirst
maskHalf
bf16HighBitMask
```

对应 load：

```cpp
MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(gammaLow, gamma);
MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(gammaHigh, gamma + QKV_K_SCALE_D128_FLOAT_REPEAT_SIZE);
MicroAPI::LoadAlign<uint16_t>(nzIndex, nzScatterIndex);
```

说明：

1. `gamma` 对同一次 Q 或 K segment VF 调用固定不变。
2. `nzScatterIndex` 对所有 token/head row 固定不变。
3. mask 构造与具体 row 无关。

### 5.2 token 级外提

`cosValue` 和 `sinValue` 只依赖 token 的 RoPE position，不依赖 head，因此从最内层 head loop 外提到 token loop。该 position 由 `query_start_loc` local 前缀和和 `seq_lens` 历史偏移共同计算，VF hot loop 只消费已经解析好的 `tokenCosSinOffset`：

```cpp
MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(cosValue, cosSin + tokenCosSinOffset);
MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(
    sinValue, cosSin + tokenCosSinOffset + QKV_K_SCALE_D128_HALF_SIZE);
```

收益：

1. 同一个 token 下所有 Q/K heads 复用同一份 cos/sin。
2. 避免每个 head 重复 load RoPE 参数。

### 5.3 row 级保留

以下内容仍在最内层循环中，因为它们依赖具体 token/head row：

1. Q/K input low/high half load。
2. RMSNorm reduce。
3. divisor duplicate。
4. RoPE 结果。
5. BF16 cast 结果直接 scatter。

## 6. MicroAPI 接口替换

### 6.1 不再使用 `MicroAPI::DataCopy`

本轮优化明确要求 VF 中不使用 `MicroAPI::DataCopy`，改用 `Load*` / `Store*` 类接口。

替换内容：

1. QK cos/sin：

```cpp
// before
MicroAPI::DataCopy(cosValue, cosSin + tokenCosSinOffset);
MicroAPI::DataCopy(sinValue, cosSin + tokenCosSinOffset + QKV_K_SCALE_D128_HALF_SIZE);

// after
MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(cosValue, cosSin + tokenCosSinOffset);
MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(
    sinValue, cosSin + tokenCosSinOffset + QKV_K_SCALE_D128_HALF_SIZE);
```

2. V FP8 写回：

```cpp
MicroAPI::StoreAlign<uint8_t, MicroAPI::StoreDist::DIST_PACK4_B32>(...);
```

3. QK dynamic quant FP8 写回：

```cpp
MicroAPI::StoreAlign<uint8_t, MicroAPI::StoreDist::DIST_PACK4_B32>(...);
```

4. QK scale 写回：

```cpp
MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(...);
```

### 6.2 当前 VF 文件扫描约束

当前目标 VF 文件应满足：

```text
rg -n "MicroAPI::DataCopy" qkv_rms_norm_rope_cache_with_k_scale_vf.h
```

无匹配结果。

## 7. 当前 VF 数据流

`QkRmsNormRopeD128SegmentNzVfImpl` 当前核心数据流：

```text
for tokenIdx:uint16_t:
  tokenInputOffset = tokenIdx * inputTokenStride
  tokenCosSinOffset = tokenIdx * 128
  tokenOutOffset = tokenIdx * outputTokenStride * 16

  Load cos/sin once for this token

  for headIdx:uint16_t:
    inputOffset = tokenInputOffset + headIdx * inputHeadStride
    rowOutOffset = tokenOutOffset + headIdx * outputHeadStride * 16

    Load BF16 low/high
    Cast BF16 -> FP32
    RMSNorm reduce
    Apply gamma
    Apply RoPE
    Cast FP32 -> BF16
    Scatter low/high BF16 directly to NZ row
```

其中 `128` 表示 D=128 row 的 BF16 element count，`16` 表示 NZ C0。

### 7.1 V scale VF

本节记录的是 RoPE `[Nv]` head-scale 基线路径。V 分支当前直接调用：

```text
VScaleFp8D128ToNtdVfImpl(..., inputTokenStride, inputHeadStride)
```

Vec 调用侧根据 `QKV_LAYOUT` 计算 input token/head stride，VF 只按 stride 读取 input UB；输出 staging 始终按 NTD-like `[head, token, D]` 存入 `v_out_ub`，便于后续按 cache head stride 搬出到 `v_cache_out`。循环顺序为 head 外层、token 内层，RoPE `v_scale[n]` 按 head 外提到 token 循环外并沿 D 广播。

M-RoPE 新合同为 `[Nv,D]`，不能复用上述 `DIST_BRC_B32` 单标量语义。当前 VF 入口使用
`VScaleFp8D128ToNtdVfImpl<V_SCALE_PER_CHANNEL>` 编译期参数：M-RoPE 每个 head 通过两次
`DIST_NORM` 加载连续的 128 个 FP32 scale，并按 `head * D + d` 与 V row 逐通道相乘；RoPE 仍通过
`DIST_BRC_B32` 加载一个 head scale。输出 staging、cache scatter 和 V-first 时序保持不变。

2026-08-03 单轮 msprof 中，M-RoPE `T=5379,Nq/Nk/Nv=16/2/2` 从 `40.037 us` 增至
`40.920 us`（`+2.21%`）；同轮 7 个未改动的 RoPE case 为 `-0.93%..+0.16%`。该结果记录逐通道
scale load 的实际成本，不外推为多轮统计结论。

### 7.2 Q/K dynamic quant VF

Q/K dynamic quant 当前直接暴露：

```text
QDynamicQuantD128NtdVfImpl
QDynamicQuantD128TndVfImpl
KDynamicQuantD128VfImpl
```

这些入口都在 VF 内部完成：

```text
Load FP32 row
abs/max-reduce
scale = max_abs / FP8_E4M3FN_MAX
Div -> Cast fp8_e4m3fn
Store FP8 row
Store scale
```

RmsNormRopeNz 阶段已经按 `Q_OUT_LAYOUT` 统一 Q/K 内部 row order。Q 分支由 Vec 调用侧按 `Q_OUT_LAYOUT` 选择 NTD/TND 两个入口；K 分支使用一个 `KDynamicQuantD128VfImpl`，由调用侧传入 input/output head/token stride 表达 NTD/TND。Q 的 FP8 staging 与 FP32 输入首地址相同。因为一个 FP32 D row 的空间可以容纳 4 条 FP8 D row，FP8 写回使用 sparse row stride，不会踩踏尚未读取的 FP32 row；最终 `StoreQOutputs` 通过 `srcStride` 适配这种非连续 FP8 staging。K 分支采用同类处理，最终仍按 cache slot/head/token 写 `k_cache_out/k_scale_cache_out`。

## 8. 验证记录

以下是对应实现版本已完成的历史验证记录；文档更新不等价于对当前 worktree 重新执行这些命令。

### 8.1 静态扫描

1. QK helper 已移除：

```text
QkRmsNormRopeD128RowNzVf
QkRmsNormRopeD128HeadRangeNzVf
ScatterRopeBf16HalfToNz
StoreRopeBf16RowToNzByScatter
```

2. `QkRmsNormRopeD128SegmentNzVfImpl`、`VScaleFp8D128ToNtdVfImpl`、`QDynamicQuantD128NtdVfImpl`、`QDynamicQuantD128TndVfImpl`、`KDynamicQuantD128VfImpl` 内显式 token/head loop 使用 `uint16_t`。

3. VF 文件不再包含：

```text
MicroAPI::DataCopy
```

4. `git diff --check` 通过。

### 8.2 package build

命令：

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
bash build.sh --pkg --ops=qkv_rms_norm_rope_cache_with_k_scale --soc=ascend950 -j8
```

结果：通过，生成：

```text
build/cann-ops-transformer-custom_linux-x86_64.run
```

### 8.3 host UT

命令：

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
bash build.sh --ophost_test --ops=qkv_rms_norm_rope_cache_with_k_scale --soc=ascend950 -j8
```

结果：

```text
[  PASSED  ] 12 tests.
```

### 8.4 torch_extension full golden

当前代码对应 run 包安装到 `/usr/local/Ascend` 后，重新全量执行 torch_extension 用例：

```bash
bash /root/.codex/skills/qkv-torch-extension-runner/scripts/run_qkv_torch_extension_cases.sh \
  --log /tmp/qkv_torch_extension_common_constants_full_rerun2.log
```

结果：

```text
[SUMMARY] cases=82 success=82 failed=0
```

## 9. 后续建议

1. 后续若继续扩大 case，应继续覆盖 random dense `rotation`、tail tile、AIV1 empty slice、K rows 非 16 对齐、多 batch `query_start_loc`、`seq_lens` 非零历史偏移、非连续/跨 block `slot_mapping`。
2. layout dispatch 已收敛到 Vec 调用侧模板分支，VF 入口通过专用入口或显式 stride 保持热循环无 runtime layout 分支；后续修改仍需保持：
   - 使用 `Load*` / `Store*`，不使用 `MicroAPI::DataCopy`。
   - 明确 loop id * stride 寻址，不使用循环尾部指针累加。
   - 将循环无关 load 外提。
3. 若未来放宽 D 或 head/tile 约束，需要重新审查 `uint16_t` loop 变量是否仍满足范围要求。
