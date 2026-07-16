# BLASST 稀疏注意力算子设计文档

> 参考论文: [BLASST: Dynamic BLocked Attention Sparsity via Softmax Thresholding](https://arxiv.org/abs/2512.12087)
> 目标架构: arch35 (Ascend950) FIA NoQuant GQA kernel

---

## 1. 算法原理

### 1.1 核心思想

在 FlashAttention online softmax 过程中, 每个 s2 tile 计算完 `local_max` 后, 与 `running_max` 比较:

```
if (local_max[i] - running_max[i] < sparseLambda):
    → bit[i] = 1  (稀疏, 可跳过)
```

- `sparseLambda`: log 域阈值, 用户传入
- **首 s2 tile**: 无 running_max → 全部 non-sparse
- **尾 s2 tile**: Vec2 的 Div+CopyOut 不能跳过 (必须对所有行输出最终 O_final)
- **BMM2_TOUB=true 场景**: BMM2 已在 UB 中完成, 跳过收益为零, 不做 block 级跳过

### 1.2 sparseLambda 参数


| 值 | 含义 |
|:---|------|
| `≤ -999.0f` | 关闭 BLASST (默认, 哨兵值, 向后兼容) |
| `-10` | exp(-10)≈4.5e-5, 极度保守 
| `0` | exp(0)=1.0, 中等 |
| `10` | exp(10)≈22026, 极度激进 |

### 1.3 跳过逻辑

| 阶段 | 粒度 | 条件 | 效果 |
|------|:---:|------|------|
| BMM2 (AIC) | **Block 级** | 全 64 行 mask=1, 首块除外 | 跳过 V 加载+MatMul |
| Vec2 (AIV) | **行级** | bit[i]=1, 首块除外, 尾块不跳 Div+CopyOut | 跳过该行 FlashUpdate 累加 |

---

## 2. SSMask 存储设计

### 2.1 `__ssbuf__` 硬件 scratchpad

独立于 GM/UB/L1 的地址空间, AIC/AIV 共享。已有先例 `flash_attention_score_antiquant_block_cube.h:78-84`：

```cpp
CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(flagID);        // AIV0
CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(16 + flagID);   // AIV1
```

### 2.2 3 槽 ring + 6 flag

```
__ssbuf__ 布局 (per AIC+AIV pair, 24 bytes):

  slot 0 (offset 0):   [AIV0 uint32_t] [AIV1 uint32_t]  ← runInfo.loop%3=0
  slot 1 (offset 8):   [AIV0 uint32_t] [AIV1 uint32_t]  ← runInfo.loop%3=1
  slot 2 (offset 16):  [AIV0 uint32_t] [AIV1 uint32_t]  ← runInfo.loop%3=2

Flag (PIPE_S):
  slot 0: AIV0→14, AIV1→30  (=16+14)
  slot 1: AIV0→15, AIV1→31  (=16+15)
  slot 2: AIV0→16, AIV1→32  (=16+16)
```

```cpp
// fia_kernel_noquant_gqa.h: 新增
static constexpr uint32_t SSBUF_BASE_FLAG = 14;
static constexpr uint32_t SSBUF_OFFSET    = 0;

__aicore__ inline uint32_t ssMaskFlag(uint32_t slot, uint32_t subBlockIdx) {
    return SSBUF_BASE_FLAG + slot + subBlockIdx * 16;
}
```

### 2.3 流水线安全

PRELOAD_N=2, Vec1 写 `slot=runInfo.loop%3`, BMM2/Vec2 读同 slot。`(N+2)%3 ≠ N%3` 永不同槽。

```
Loop 0: Vec1(0) → slot[0], SetFlag(14,30)
Loop 1: Vec1(1) → slot[1], SetFlag(15,31)
Loop 2: Vec1(2) → slot[2], SetFlag(16,32)
         BMM2(0)+Vec2(0) → WaitFlag(14,30) → slot[0] ✓
Loop 3: Vec1(3) → slot[0], SetFlag(14,30)   ← 复用
         BMM2(1)+Vec2(1) → WaitFlag(15,31) → slot[1] ✓
```

### 2.4 AIV0/AIV1 写入

每槽 2 个 `uint32_t`, AIV0 写 `slot*2+0` (rows 0..31), AIV1 写 `slot*2+1` (rows 32..63), 各 Set 各自 flag:

```
Vec1:  AIV0 → ssbuf[slot*2+0]=mask₀, SetFlag(14+slot)
       AIV1 → ssbuf[slot*2+1]=mask₁, SetFlag(30+slot)

BMM2/ WaitFlag(14+slot) + WaitFlag(30+slot)
Vec2: mask64 = (ssbuf[slot*2+1]<<32) | ssbuf[slot*2]
```

---

## 3. 各阶段修改

### 3.1 Host: TilingData 新增

```cpp
// op_host/fused_infer_attention_score_tiling.h
struct FusedInferAttentionSparseParams {
    float    sparseLambda;    // log域阈值
    uint32_t ssBufSize;       // 24 (3槽×2AIV×4B)
};
```

Tiling 从 context 读 `sparse_lambda`, 写入 `fiaSparseParams.sparseLambda`。SSMask 存 `__ssbuf__`, 无需 GM workspace。

### 3.2 ConstInfoX 新增

```cpp
// common/op_kernel/arch35/fia_public_define_arch35.h CommonConstInfo
float sparseLambda;  // ≤-999=关闭
```

`InitConstInfo` 从 `tilingData->fiaSparseParams.sparseLambda` 加载。

### 3.3 Vec1: 稀疏 mask 计算

**位置**: `ProcessVec1Nd` 末尾, DataCopy P→L1 之后

```cpp
// fia_block_vec_noquant_gqa.h
if (ASCEND_IS_AIV && constInfo.sparseLambda > -999.0f && !runInfo.isFirstS2Loop) {
    uint32_t slot  = runInfo.loop % 3;
    uint32_t flag  = ssMaskFlag(slot, constInfo.subBlockIdx);
    uint32_t mask  = ComputeSparseMaskVF(localMaxUb, runningMaxUb,
                                         constInfo.sparseLambda, runInfo.actVecMSize);
    auto ssbuf = reinterpret_cast<__ssbuf__ uint32_t*>(SSBUF_OFFSET);
    ssbuf[slot * 2 + constInfo.subBlockIdx] = mask;
    CrossCoreSetFlag<SYNC_MODE, PIPE_S>(flag);
}
// 首块或关闭: 也 SetFlag(mask=0), 保证 BMM2/Vec2 不 block
```

**新增文件** `vf/vf_sparse_mask.h`:

```cpp
template <typename T>
__simd_vf__ uint32_t ComputeSparseMaskVF(
    __ubuf__ T *localMaxUb, __ubuf__ T *runningMaxUb,
    float sparseLambda, uint16_t m) // m≤32
{
    RegTensor<float> vreg_local, vreg_running, vreg_diff, vreg_threshold;
    RegTensor<uint32_t> vreg_mask;
    MaskReg preg = UpdateMask<float>(m);
    Duplicate(vreg_threshold, sparseLambda);
    LoadAlign(vreg_local, localMaxUb);
    LoadAlign(vreg_running, runningMaxUb);
    Sub(vreg_diff, vreg_local, vreg_running, preg);
    CmpLt(vreg_mask, vreg_diff, vreg_threshold, preg);
    uint32_t resultMask = 0;
    // extract → resultMask
    return resultMask;
}
```

### 3.4 BMM2: Block 级跳过

**位置**: `ComputeMm2` (AIC), 在 MatMul 之前

```cpp
// fia_kernel_noquant_gqa.h
if (constInfo.sparseLambda > -999.0f && !runInfo.isFirstS2Loop) {
    uint32_t slot = runInfo.loop % 3;
    auto ssbuf = reinterpret_cast<__ssbuf__ uint32_t*>(SSBUF_OFFSET);
    CrossCoreWaitFlag<SYNC_MODE, PIPE_S>(ssMaskFlag(slot, 0));  // AIV0
    CrossCoreWaitFlag<SYNC_MODE, PIPE_S>(ssMaskFlag(slot, 1));  // AIV1
    uint64_t mask = ((uint64_t)ssbuf[slot*2+1] << 32) | ssbuf[slot*2];

    if (mask == 0xFFFFFFFFFFFFFFFFull) {  // 全 64 行稀疏
        // 跳过 MatMul, 维护 cross-core sync 链
        bmm2ResGmBuffers.Get().SetCrossCore();  // 或 bmm2Buffers
        return;
    }
}
// 正常 BMM2...
```

BMM2_TOUB=true 时不跳过 (已在 UB 中完成)。

### 3.5 Vec2: 行级跳过

**位置**: `ProcessVec2OnUb` / `ProcessVec2_C2GmUpdateUb` (AIV)

```cpp
// fia_block_vec_noquant_gqa.h
uint32_t slot   = runInfo.loop % 3;
auto ssbuf = reinterpret_cast<__ssbuf__ uint32_t*>(SSBUF_OFFSET);
CrossCoreWaitFlag<SYNC_MODE, PIPE_S>(ssMaskFlag(slot, 0));
CrossCoreWaitFlag<SYNC_MODE, PIPE_S>(ssMaskFlag(slot, 1));
uint32_t myMask = ssbuf[slot * 2 + constInfo.subBlockIdx];  // AIV 只拿自己的 32bit

if (!runInfo.isFirstS2Loop && !runInfo.isLastS2Loop) {
    // 中间块: 稀疏行跳过 FlashUpdate
    FlashUpdateSparse(vec2ResInner, bmm2Ub, vec2ResInner, expUb, myMask, m, d);
} else if (runInfo.isLastS2Loop) {
    // 尾块: 所有行做完整 FlashUpdateLast + Div + CopyOut
    FlashUpdateLastNew(vec2ResInner, bmm2Ub, vec2ResInner, expUb, sumUb, m, d);
    Cast + CopyOut;
}
```

**新增文件** `vf/vf_flashupdate_sparse.h`:

```cpp
template <typename T>
__simd_vf__ void FlashUpdateSparseVF(
    __ubuf__ float *dst, __ubuf__ float *cur, __ubuf__ float *pre,
    __ubuf__ float *expMax, uint32_t mask, uint16_t m, uint16_t d) {
    RegTensor<float> vreg_exp_max, vreg_pre, vreg_cur, vreg_mul, vreg_add;
    MaskReg preg = CreateMask<float, MaskPattern::ALL>();
    for (uint16_t i = 0; i < m; ++i) {
        if (mask & (1u << i)) continue;  // 稀疏行跳过
        LoadAlign<BRC_B32>(vreg_exp_max, expMax + i);
        for (uint16_t j = 0; j < d/64; ++j) {
            LoadAlign(vreg_pre, pre + i*d + j*64);
            LoadAlign(vreg_cur, cur + i*d + j*64);
            Mul(vreg_mul, vreg_exp_max, vreg_pre, preg);
            Add(vreg_add, vreg_mul, vreg_cur, preg);
            StoreAlign(dst + i*d + j*64, vreg_add, preg);
        }
    }
}
```

---

## 4. 执行流程

```
s2 tile 0 (首块): BMM1 → Vec1(NoUpdate, mask=全0) → P→L1 → BMM2 → Vec2(首块init)

s2 tile 1..N-1 (中间):
  BMM1 → Vec1(Update, ComputeSparseMask→__ssbuf__) → P→L1
       → BMM2: WaitFlag → 读mask → 全1? skip : 正常
       → Vec2: WaitFlag → 读mask → 稀疏行skip FlashUpdate

s2 tile N (尾块):
  BMM1 → Vec1(Update, mask) → P→L1 → BMM2(skip?) → Vec2: 全行 FlashUpdateLast + Div + CopyOut
```

---

## 5. 代码修改清单

### 新增

| 文件 | 内容 |
|------|------|
| `common/op_kernel/arch35/vf/vf_sparse_mask.h` | `ComputeSparseMaskVF` |
| `common/op_kernel/arch35/vf/vf_flashupdate_sparse.h` | `FlashUpdateSparseVF` |

### 修改

| 文件 | 内容 |
|------|------|
| `op_api/aclnn_fused_infer_attention_score_v5.h` | +`sparseLambda` 参数 (默认 -999.0f) |
| `op_host/fused_infer_attention_score_tiling.h` | TilingData +`FiaSparseParams` |
| `common/op_host/arch35/fia_tiling_nonquant_gqa.cpp` | FillTiling: sparseLambda, ssBufSize=24 |
| `common/op_kernel/arch35/fia_public_define_arch35.h` | CommonConstInfo +`sparseLambda` |
| `common/op_kernel/arch35/fia_kernel_noquant_gqa.h` | InitConstInfo 加载; +SSBUF 常量+ssMaskFlag(); ComputeMm2 跳过逻辑 |
| `common/op_kernel/arch35/fia_block_vec_noquant_gqa.h` | ProcessVec1Nd 末: SSMask→__ssbuf__; ProcessVec2*: 读 SSMask, 行跳过 |

### 兼容性

`sparseLambda ≤ -999.0f` (默认) → 全路径跳过 → 等价于原 FIA。

---

## 6. 内存

```
Workspace/UB/L1: 无需新增
__ssbuf__:       24B (3槽×2AIV×4B) per pair, 28 pairs × 24B = 672B
```

---

## 7. 用例场景

```
输入: B=2, N2=4, G=2, S1=32, S2=9856, D=128, FP16
      sparseLambda = 5.0
      28 AIC + 56 AIV, s2BaseSize=128

s2 tile 0 (tokens 0..127):
  blk=0 处理 bN2=0, Vec1 NoUpdate, running_max = local_max₀
  SSMask = 全0 (首块) → BMM2 正常 → Vec2 init

s2 tile 1 (tokens 128..255):
  blk=0: Vec1: local_max₁ vs running_max
    假设 64 行中有 24 行 local_max₁ < running_max - 5 (sparseLambda=5)
    → AIV0: mask=0x00FFFFFF (rows 0..23 sparse) → __ssbuf__[slot*2+0]
    → AIV1: mask=0x00000000 (rows 32..63 全活跃)  → __ssbuf__[slot*2+1]
  BMM2: mask64 = 0x0000000000FFFFFF → 不全1 → 正常 MatMul
  Vec2: AIV0 跳过 rows 0..23 FlashUpdate, AIV1 全算

s2 tile 8 (tokens 896..1023):
  假设全部 64 行 sparse (所有 local_max₈ << running_max)
  → mask64 = 0xFFFFFFFFFFFFFFFF
  BMM2: 全1 → 跳过 V 加载 + MatMul! 节省 32KB GM→L1 搬运 + 524K MACs
  Vec2: AIV0+AIV1 全跳 FlashUpdate, O_acc 不变 (正确: 该 tile 贡献≈0)

s2 tile 76 (尾块, tokens 9728..9855):
  BMM2: 全1 → skip (节省同上)
  Vec2: 不做 FlashUpdate, 但仍执行 FlashUpdateLastNew + Div + Cast + CopyOut
    → 尾块约束: 必须对所有行输出最终结果
```

---

## 附录A: 真实输入 Case 代码路径追踪

### A.1 输入数据

来源: `test_res/layer_30_1775227380534.pt`

```
Q: (8192, 16, 128) bf16    → B=1, BSND layout → Sq=8192, Nq=16, D=128
K: (8192, 2, 128)  bf16    → Skv=8192, Nkv=2
V: (8192, 2, 128)  bf16    → GQA group_size = 16/2 = 8
scale = 0.0883 (≈ 1/sqrt(128))
sparse_mode = 0, block_table empty (no PA)
```

### A.2 Tiling 选型

```
S1=8192, S2=8192, D=128, DV=128  arch35 GQA
    ↓ SplitPolicy + GetGqaSplitConfigMap
mBaseSize=128, s2BaseSize=128, dBaseSize=128, dVBaseSize=128
    ↓
CV_RATIO=2 → AIV0/AIV1 each: actVecMSize ≤ 64 rows
```

### A.3 Kernel 入口 → Vec1 VF 调用链

```
FlashAttentionNoQuantGqaKernel::Process()
  └─ FlashAttention() → ExecuteTask()
       └─ ComputeVec1() → ProcessVec1() → ProcessVec1Nd()
            │  INPUT_T=bfloat16_t, T=float, hasUpdate=true (非首s2)
            │  actSingleLoopS2Size=128 → 模板标签 EQ_128
            │
            └─ vf_mul_sel_softmaxflashv2_cast_nz.h:389
               ProcessVec1Vf<bfloat16_t, float, ..., true, 128, 128, EQ_128, ...>
                 │
                 └─ ProcessVec1Update (oriNRange==EQ_128, line 284)
                      │
                      └─ ProcessVec1UpdateImpl128 (line 291)
                           │  vf_basic_block_aligned128_update.h
                           │
                           └─ ProcessVec1UpdateImpl128VF (SIMD __simd_vf__, line 27)
```

### A.4 寄存器级 max merge (稀疏判定注入点)

文件: `vf_basic_block_aligned128_update.h:201-206`

```c++
// ★ line 201: 加载 old running max (每个 AIV ≤64 行, fp32)
LoadAlign(vreg_in_max, inMaxUb);

// line 202: VEC_STORE → VEC_LOAD 内存屏障
LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

// ★ line 203: 加载 local_max (已乘 scale = 1/sqrt(D))
LoadAlign(vreg_input_max, tmpMaxUb2);

// ★★★ line 204: 合并 old_max 与 local_max → new_max
//     此时 vreg_input_max = local_max (scaled)
//           vreg_in_max    = old running max
//           vreg_max_new   = max(local, old)  ← 会被覆盖
Max(vreg_max_new, vreg_input_max, vreg_in_max, preg_all);

// line 205-206: 存回 new_max 到 UB (后续 Vec2/W 使用)
StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
    (__ubuf__ T *&)tmpMaxUb2, vreg_max_new, preg_all);
```

**第 204 行 `Max(vreg_max_new, vreg_input_max, vreg_in_max, preg_all)` 就是稀疏判定的注入点。**

此时在一个 VF SIMD 指令内, 3 个浮点寄存器同时有效:

| 寄存器 | 含义 | 用途 |
|--------|------|------|
| `vreg_input_max` | `local_max` (scaled) | 当前 s2 tile 的逐行最大值 |
| `vreg_in_max` | `old_max` | 全局历史 running max |
| `vreg_max_new` | `max(local, old)` | 合并后的 new max, 即将存回 maxUb |

稀疏判定逻辑:
```
mask[i] = (vreg_input_max[i] - vreg_in_max[i] < sparseLambda) ? 1 : 0
```

### A.5 涉及的 VF Update 变体

`sparseLambda` 参数和 `mask` 输出需要穿透以下 VF 函数:

| 模板标签 | VF 函数 | 文件 |
|:---------|:--------|:-----|
| `EQ_128` | `ProcessVec1UpdateImpl128VF` | `vf_basic_block_aligned128_update.h` |
| `GT_0_AND_LTE_64` | `ProcessVec1UpdateImpl64VF` | `vf_basic_block_unaligned64_update.h` |
| `GT_64_AND_LTE_128` | `ProcessVec1UpdateGeneralImpl128` | `vf_basic_block_unaligned128_update.h` |
| `GT_128_AND_LTE_256` | `ProcessVec1UpdateGeneralImpl256` | `vf_basic_block_unaligned256_update.h` |
| `GT_256_AND_LTE_512` | `ProcessVec1UpdateGeneralImpl512` | `vf_basic_block_unaligned512_update.h` |
| `GT_512_AND_LTE_1024` | `ProcessVec1UpdateGeneralImpl1024` | `vf_basic_block_unaligned1024_update.h` |
| FullQuant256 | `ProcessVec1UpdateGeneralImpl256GqaFullquant` | `vf_basic_block_fullquant_unaligned256_update.h` |
| FullQuant512 | `ProcessVec1UpdateGeneralImpl512GqaFullquant` | `vf_basic_block_fullquant_unaligned512_update.h` |
| DN | `ProcessVec1VfDn` | `vf_mul_sel_softmaxflashv2_cast_nz_dn.h` |
| MX | `ProcessVec1UpdateImpl128VF` (MX variant) | `vf_basic_block_aligned128_update_mx.h` |

共 ~12 个 VF 函数需要新增 `sparseLambda` + `sparseMask` 参数, 并在各自 `Max(vreg_max_new, vreg_input_max, vreg_in_max, ...)` 之后插入 `vreg_input_max - vreg_in_max < sparseLambda` 的 mask 累加逻辑。

# To DO List
## 实现不符合预期
1. Phase 2 未跳过执行
2. sync mode应该为2，估计是因为设置的flagid不对导致的，
3. 
