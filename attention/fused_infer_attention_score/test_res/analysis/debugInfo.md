# BLASST 编译调试记录

> 时间: 2026-07-01 ~ 2026-07-02
> 目标: arch35 FIA NoQuant GQA kernel 新增 BLASST block 级稀疏跳过

---

## 编译问题汇总

### 1. sync mode 参数不合法

**错误信息**:
```
error: the ranges of 1st parameter must be [1, 1], [4, 5]
```

**原因**: `CrossCoreSetFlag<1, PIPE_S>(flag)` 中 sync mode 值 1 在 arch35 上对 PIPE_S 不合法。arch35 的 scratchpad FFTS 协议统一用 mode 4。

**修复**: 全部改为 `<4, PIPE_S>`。

**涉及位置**:
- `fia_block_vec_noquant_gqa.h`: `CrossCoreSetFlag`
- `fia_block_cube_noquant_gqa.h`: `CrossCoreWaitFlag` (2 处)

---

### 2. `if constexpr` 参数非编译期常量

**错误信息**: 无直接报错，但逻辑永远不生效。

**原因**:
```c++
const float sparseLambda = -999.0f;  // 函数参数，默认值
if constexpr (sparseLambda > -999.0f)  // 编译期求值，用默认值，永远 false
```
`sparseLambda` 是函数参数不是模板参数，`if constexpr` 在编译期看到的是默认值 `-999.0f`，运行时传入的 `-7.0f` 无法改变编译期分支。

**修复**: `if constexpr` 改为普通 `if`。

**涉及文件**: `vf_basic_block_aligned128_update.h`

---

### 3. RegTensor 不能直接与标量比较

**错误信息**:
```
error: invalid operands to binary expression ('RegTensor<float>' and 'const float')
```

**原因**: `Reduce<MAX>` 返回类型是 `RegTensor<float>`（向量寄存器），即使结果只有一个有效 lane，也不能直接用 `<` 与标量 float 比较。

**修复**: `StoreAlign` 写到 UB → `__aicore__` 中读回标量 → 比较。

**涉及文件**: `vf_basic_block_aligned128_update.h` (VF 侧 StoreAlign), `ProcessVec1UpdateImpl128` (__aicore__ 侧标量比较)

---

### 4. bisheng SIGSEGV — `for(1)` 包大循环

**错误信息**:
```
bisheng: error: clang frontend command failed with exit code 139 (use -v to see invocation)
```

**原因**:
```c++
for (uint16_t bs = 0; bs < nonBlockSparseU16; ++bs) {
    for (uint16_t i = 0; i < m; ++i) {  // Phase 2, 200+ 行
        ...
    }
}
```
外层 `for(1)` 包了内层 200+ 行的复杂循环体，bisheng 编译器展开嵌套 hardware loop 时崩溃。

**修复**: 去掉外层 `for(1)` 包装，Phase 2 循环无条件运行。block sparse 时在 `__aicore__` 中尾后清零 `tmpExpSumUb`。

**涉及文件**: `vf_basic_block_aligned128_update.h`

---

### 5. AIV 循环中的标量指令

**错误信息**:
```
fatal error: error in backend: Unsupported scalar instruction in AIV loop.
bisheng: error: clang frontend command failed with exit code 70
```

**原因**: VF 中的 `((__ubuf__ float*)tmpExpSumUb)[0]` 标量 UB 读和 `if (sparseLocalMaxUb != nullptr)` 条件分支，AIV 向量后端无法展开为硬件指令。

**修复**: 标量读和条件分支全部移到 `__aicore__` 的 `ProcessVec1UpdateImpl128` 中。VF 只保留纯向量操作（Sub + Reduce + StoreAlign）。

**涉及文件**: `vf_basic_block_aligned128_update.h`

---

### 6. VF 中条件 StoreAlign

**错误信息**:
```
fatal error: error in backend: Do not know how to expand this operator's operand!
bisheng: error: clang frontend command failed with exit code 70
```

**原因**:
```c++
if (sparseLocalMaxUb != nullptr) {
    StoreAlign<float, ...>(sparseLocalMaxUb, vreg_max_tmp, preg_all);
}
```
AIV 后端不认运行时条件存储——它只认无条件的向量 StoreAlign。

**修复**: 去掉 `if`，StoreAlign 无条件执行。同时 `ProcessVec1Nd` 中 `localMaxUb` 改为无条件分配，确保指针始终有效。

**涉及文件**:
- `vf_basic_block_aligned128_update.h`: 去掉 `if (sparseLocalMaxUb != nullptr)`
- `fia_block_vec_noquant_gqa.h`: `localMaxUb` 无条件分配

---

### 7. 源文件缓存未刷新

**错误信息**:
```
fatal error: '/path/to/fused_infer_attention_score_apt.cpp' file not found
```

**原因**: build 目录下的源码拷贝是旧的，修改 `.h` 后未触发 cmake `_src_copy` 重新拷贝。

**修复**: 走完整 `build.sh --opkernel` 重新 cmake + 拷贝 + 编译。

---

## VF/AIV 编译器关键约束（基于本次调试经验）

| 约束 | 说明 |
|------|------|
| `__simd_vf__` 内禁用 `if constexpr` 依赖运行时参数 | 函数参数非编译期常量，永远看默认值 |
| `__simd_vf__` 内禁用运行时条件 `if`（含 `for(1)` 模式）| for(1) 包简单循环可行，包大循环(200+行)编译器崩溃 |
| `__simd_vf__` 内禁用标量 UB 读写 | 只允许 StoreAlign/LoadAlign 等向量操作 |
| `__simd_vf__` 内禁用条件 Store | StoreAlign 必须无条件执行 |
| `RegTensor<float>` 不能直接比较/运算标量 | 必须先 Store→UB，在 `__aicore__` 中读回标量 |
| arch35 PIPE_S sync mode 必须用 4 | mode 1/2 编译报错 |
| 同 core AIV 内 Vec1→Vec2 无需 PIPE_S | pipeline 顺序保证 |

---

## 最终 VF 改动量

`ProcessVec1UpdateImpl128VF`（`__simd_vf__`）仅新增 **3 行**：

```c++
Sub(vreg_diff_for_blasst, vreg_input_max, vreg_in_max, preg_all);
Reduce<MAX>(vreg_max_tmp, vreg_diff_for_blasst, preg_all);
StoreAlign<float>(sparseLocalMaxUb, vreg_max_tmp, preg_all);
```

`ProcessVec1UpdateImpl128`（`__aicore__`）新增 **8 行**：

```c++
if (sparseLocalMaxUb != nullptr && sparseLambda > -999.0f) {
    float maxDiff = sparseLocalMaxUb[0];
    bool blockSparse = (maxDiff < sparseLambda);
    sparseLocalMaxUb[0] = blockSparse ? 1.0f : 0.0f;
    if (blockSparse) { for: tmpExpSumUb[i] = 0; }
}
```

---

## 8. BLASST Vec2 FlashUpdateLastNew: 最后一个 S2 block 为 sparse 时使用陈旧数据

**时间**: 2026-07-07

**错误现象**: NPU output max=122.69（正常 ~3.0），cos_sim=0.248

**根本原因**: `ProcessVec2OnUb`（`fia_block_vec_noquant_gqa.h:730-735`）最后一个 S2 loop 时 `FlashUpdateLastNew` 无条件执行，未检查 `allRowsSparse`。当末尾 block 被 BLASST 判定 sparse：
1. Vec1 Part 2（softmax）跳过 → `expUb` 陈旧
2. BMM2 skip → `mmRes` 陈旧
3. `FlashUpdateLastNew` 用 `staleExp * staleMmRes` 污染输出

**修复**: 末尾 loop 分支加 sparse 检查：
```c++
if (allRowsSparse) {
    LastDivNew(vec2ResUb, vec2ResUb, sumUb, ...);  // 直接归一化
} else {
    FlashUpdateLastNew(...);  // 正常路径
}
```

**修复效果**: output max 从 122.69 降至 2.79（正常范围）

**涉及文件**: `attention/common/op_kernel/arch35/fia_block_vec_noquant_gqa.h`

---

## 9. 新 CANN 9.0 导致 tiling key (fd/reconst) 变化

**时间**: 2026-07-07

**现象**: 新 key=132383490(fd=0,reconst=0) vs 旧 key=36639605506(fd=1,reconst=1)，导致 "not find tilingKey"

**原因**: `CheckFlashDecode()` 条件 `loopTimes < 0.4 * platformInfo_.aicNum` — 新 CANN 9.0 头文件 `platformInfo_.aicNum` 值不同

**修复**: `--tiling-key "132383490"` 过滤编译
