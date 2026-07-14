# MhcPreSinkhorn output_0 精度问题分析

## 1. 问题背景

### 1.1 算子与输出

- **算子**：`MhcPreSinkhorn`
- **问题输出**：`output_0`（即 `y = sum(pre * x, dim=hcMult)`，shape 如 `[B, S, d]`）
- **复现 case**（ATK）：
  - 输入 `x`：`[4, 4, 4, 99328]`，bf16
  - `n = hcMult = 4`
  - `d = 99328`
- **复现命令**（在 `gmm-atk-samples/atk_case/MhcPreSinkhorn` 下）：

```bash
conda activate hey311
source /data/hey/package/0701/ascend-toolkit/set_env.sh
source /data/hey/custom/vendors/custom_transformer/bin/set_env.bash

atk node --backend pyaclnn --devices 2 node --backend cpu task \
  -c all_aclnn_mhc_pre_sinkhorn.json --task accuracy -p MhcPreSinkhorn.py -e 1 --save_data output
```

### 1.2 现象

- **仅 `output_0` 精度失败**，`output_1`（post）、`output_2`（combFrag）等其他输出正常
- 对比 CPU 标杆与 NPU 输出：
  - 整体 match rate 约 **99.97%**
  - 每个 token 在 d 轴**前 22 个 chunk 完全正确**
  - **仅最后一个 tail chunk**（`d ∈ [99264, 99327]`，64 个元素）出错
  - 14/16 个 token 的首个错误位置在 `d = 99296`

### 1.3 涉及代码路径

问题集中在 **Stage2** 的 ProcessY 循环（`mhc_pre_sinkhorn_m_split_core.h` 中 d 轴分块逻辑）：

```cpp
for (int64_t dLoopIdx = 0; dLoopIdx < tilingData->dLoop; dLoopIdx++) {
    int64_t curDFactor = (dLoopIdx == dLoop - 1) ? tailDFactor : dFactor;
    CopyIn(x ...);
    ProcessY(y ...);   // y = sum(pre * x, dim=hcMult)
    CopyOut(y ...);
}
```

---

## 2. 根因分析

### 2.1 直接原因：存在非完整 tail 块

当 `d` 过大、UB 无法一次容纳整个 d 轴时，tiling 将 d 拆成多轮循环：

| 参数 | 修复前 | 说明 |
|------|--------|------|
| `d` | 99328 | — |
| `dFactor` | 4512 | 每轮处理的 d 长度 |
| `dLoop` | 23 | 循环次数 |
| `tailDFactor` | **64** | 最后一轮仅 64 元素（`99328 = 4512×22 + 64`） |

前 22 轮均为完整 chunk（`curDFactor = 4512`），计算正确；**第 23 轮 tail 块**（64 元素）出错。

### 2.2 机理：tail 轮次 UB 布局与 buffer 分配不一致

Stage2 中 x/y 相关 buffer 按 **`RoundUp(dFactor)`**（4512）分配，例如：

```cpp
xQueNum2 = stage2RowFactor * hcMult * RoundUp(dFactor);  // 4 * 4512
```

但 tail 轮次（`curDFactor = 64`）时：

1. **CopyIn** 以 `dstStride=0` 紧凑写入，各行间距为 **64**
2. **ProcessY / CastTwoDim** 按 `RoundUp(curDFactor)=64` 读取 xLocal
3. xQue 为**双 buffer**，上一轮完整 chunk 在 UB 中按 **4512 行距** 排列；tail 轮次只在开头写入 `4×64` 元素，**未覆盖/未对齐到 4512 行距的 buffer 布局**

因此 tail 轮次 ProcessY 可能读到错误数据，导致 `output_0` 在 d 轴末尾出错。post/combFrag 不依赖该 x 读取路径，故不受影响。

### 2.3 为何不是「d > 16000 就全错」

- **d ≤ ~4768**（192KB UB）：`dLoop=1`，整段 d 一次处理，**无 tail**
- **d 更大**：需分块；仅当 **`d % dFactor ≠ 0`** 时出现不满块的 tail
- 99328 案例：仅最后 64 个元素错，整体仍 ~99.97% 正确

---

## 3. 修复方案（仅 Tiling）

### 3.1 思路

在 `CalcMKSplitCoreMembasePart2Tiling` 中，当 `d % dFactor != 0` 时，**向下搜索 32 对齐、能整除 d 且仍满足 UB 的 dFactor**，消除 tail 块。

### 3.2 修改文件

`op_host/op_tiling/mhc_pre_sinkhorn_tiling.cpp`

```cpp
if (dFactor_ > 32) {
    dFactor_ = DownAlign(dFactor_, 32);
}
// Prefer a dFactor that divides d evenly to avoid error-prone partial tail chunks in ProcessY.
if (d_ % dFactor_ != 0) {
    for (int64_t candidate = DownAlign(dFactor_, 32); candidate >= 32; candidate -= 32) {
        if (d_ % candidate != 0) {
            continue;
        }
        int64_t candXSize = rowOnceLoop * hcMult_ * RoundUp(candidate, 16) * 2 * DOUBLE_BUFFER;
        int64_t candYSize = rowOnceLoop * RoundUp(candidate, 16) * 2 * DOUBLE_BUFFER;
        int64_t candXCastSize = rowOnceLoop * hcMult_ * RoundUp(candidate, 8) * sizeof(float);
        int64_t candYCastSize = rowOnceLoop * RoundUp(candidate, 8) * sizeof(float);
        if (candXSize + candYSize + candXCastSize + candYCastSize <= ubRemain) {
            dFactor_ = candidate;
            break;
        }
    }
}
dLoop_ = CeilDiv(d_, dFactor_);
tailDFactor_ = d_ % dFactor_ == 0 ? dFactor_ : d_ % dFactor_;
```

### 3.3 修复效果（d=99328）

| | 修复前 | 修复后 |
|---|--------|--------|
| dFactor | 4512 | **3104** |
| dLoop | 23 | **32** |
| tailDFactor | 64（不满块） | **3104**（整除，无 tail） |
| 验证 | output_0 失败 | **精度通过** |

`99328 = 3104 × 32`，所有 d 轴循环均为完整 chunk，不再进入有问题的 tail 路径。

### 3.4 未采用的 Kernel 方案（已还原）

曾考虑在 kernel 中对 tail 轮次做逐行 CopyIn + UB 行距对齐（`mhc_pre_sinkhorn_base.h`、`mhc_pre_sinkhorn_m_split_core.h`）。在 **d 恒 128 对齐 + tiling 修复** 的前提下，tail 块可被消除，**kernel 改动非必需**，已还原，仅保留 tiling 修改。

---

## 4. 相关结论（设计约束）

### 4.1 n=4 时 d 何时需要分块（192KB UB）

| 条件 | 阈值 |
|------|------|
| 不需分块（dLoop=1） | **d ≤ 4768** |
| 需要分块（dLoop>1） | **d ≥ 4769** |
| d 128 对齐且不分块 | **最大 d = 4736** |

示例：`d=3584` → `dLoop=1`，不会触发本问题。

### 4.2 d 恒 128 对齐时的 tail 保证

- **128 对齐 ≠ 无 tail**（修复前：`99328` 128 对齐，但 `4512` 有余数 64）
- **128 对齐 + tiling 修复**：d 可被 32 整除，搜索最差可 fallback 到 `dFactor=32`，故 **实际可无 tail**
- 在此约束下 **仅 tiling 修改即可**，无需 kernel 兜底

### 4.3 编译与安装

```bash
cd /data/hey/git/ops-transformer
source /data/hey/package/0701/ascend-toolkit/set_env.sh
bash build.sh --pkg --ops=mhc_pre_sinkhorn
./build_out/cann-ops-transformer-custom_linux-aarch64.run --install-path=/data/hey/custom
source /data/hey/custom/vendors/custom_transformer/bin/set_env.bash
```

---

## 5. 总结

| 项目 | 内容 |
|------|------|
| **问题** | 大 d 分块时 tail 块（`tailDFactor < dFactor`）导致 output_0 末尾错误 |
| **根因** | tail 轮次 UB 布局与按 dFactor 分配的 buffer 不一致 |
| **修复** | Tiling 优先选择整除 d 的 dFactor，消除 tail |
| **改动范围** | 仅 `mhc_pre_sinkhorn_tiling.cpp` |
| **验证** | ATK case `d=99328` 精度通过 |
