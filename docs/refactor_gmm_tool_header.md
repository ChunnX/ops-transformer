# GMM Kernel tool.h 头文件去重与重构方案

## 1. 背景与问题分析

当前 `gmm` 模块下存在 **3 个同名头文件 `tool.h`**，分别位于：

| 序号 | 路径 | 主要用途 |
|------|------|----------|
| 1 | `gmm/grouped_matmul/op_kernel/a16w4_msd/tool.h` | A16W4 MSD 场景 kernel 工具函数 |
| 2 | `gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/tool.h` | Arch35 WeightQuant BasicBlock 工具函数 |
| 3 | `gmm/grouped_matmul_finalize_routing/op_kernel/arch35/common/tool.h` | Arch35 WeightQuant FinalizeRouting 工具函数 |

### 1.1 重复命名风险

三个头文件同名，在编译系统中若 `include` 路径配置不当，极易发生 **头文件覆盖（shadowing）**，导致编译错误或在不同编译单元中引入不同实现，引发难以排查的 ODR（One Definition Rule）违例。

### 1.2 代码重复

- **文件 2 与文件 3 几乎 100% 重复**：namespace、常量定义、枚举、模板函数实现完全一致，仅有注释格式差异。
- **文件 1 与文件 2/3 存在大量重复代码**：`CeilAlign`、`DataCopyPad2D`、`GetKBUnit` 等核心工具函数逻辑高度重合，仅有细微差异（如 `CeilDiv` 与 `CeilDiv` 命名、是否包含 `fp4x2` 类型判断等）。

这种重复导致：
- 修改一处工具函数（如修复 `CeilAlign` 溢出检查）需要同步修改多份拷贝，维护成本高。
- 同一逻辑在不同目录下逐渐分化，最终形成“相似但不同”的技术债。

---

## 2. 目标

1. **消除同名头文件**：每个头文件具有全局唯一的文件名。
2. **消除重复代码**：通过公共头文件提取通用逻辑，各业务模块仅保留自身特有的定义。
3. **保持功能等价**：重构后所有 kernel 行为与重构前完全一致。
4. **最小化侵入**：尽量不改业务逻辑代码，仅调整 `#include` 与命名。

---

## 3. 差异分析

### 3.1 文件 1（a16w4_msd） vs 文件 2/3（arch35 weight quant）

| 对比项 | 文件 1（a16w4_msd） | 文件 2/3（arch35） | 兼容性结论 |
|--------|---------------------|---------------------|------------|
| namespace | `GROUPED_MATMUL::A16W4Msd` | `WeightQuantBatchMatmulV2::Arch35` | 需保留各自 namespace，公共函数放在独立公共 namespace |
| `CeilDiv` / `CeilDiv` | `CeilDiv` | `CeilDiv` | 统一为 `CeilDiv`，修改文件 1 的 4 处引用 |
| `CeilAlign` 重载 | 仅模板版本 | 模板 + `uint32_t` 溢出检查版本 | 保留两者，无冲突 |
| `DataCopyPad2D` 4bit 判断 | 仅 `int4b_t` | `int4b_t` + `fp4x2_e2m1_t` + `fp4x2_e1m2_t` | 统一为后者，`if constexpr` 不会编译失败 |
| `GetKBUnit` | 显式 `int8_t`(1024)、`float`(256)、default(512) | `int4b_t/fp4x2`(2048)、`default(1024/sizeof(T))` | 统一为后者，数学结果一致 |
| Arch35 特有常量 | 无 | Buffer、同步、控制参数等大量常量 | 提取到 Arch35 专用公共头文件 |
| `Min` 函数 | 无 | 有 | 公共头文件统一提供 |
| `IsMxA8W4` / `MatmulL1GmType` | 无 | 有 | 提取到 Arch35 专用公共头文件 |

### 3.2 文件 2 与文件 3

- **差异度 ≈ 0%**：所有代码逻辑、常量值、namespace 完全一致。
- **结论**：可直接合并为同一个公共头文件。

---

## 4. 方案设计

采用 **"单一公共头文件提取"** 的策略。

```
┌──────────────────────────────────────────────────────────────────────┐
│                    gmm/common/gmm_kernel_tool.h                      │
│  公共工具层（GmmOpKernel）：CeilAlign, CeilDiv, Min, DataCopyPad2D,   │
│  GetKBUnit                                                           │
│  Arch35 业务层（WeightQuantBatchMatmulV2::Arch35）：QuantType, 常量,  │
│  IsMxA8W4, MatmulL1GmType                                            │
├──────────────────────────────────────────────────────────────────────┤
│  业务层（仅保留各自特有逻辑，无 tool.h）                                │
│  ├── gmm/grouped_matmul/op_kernel/a16w4_msd/...                       │
│  ├── gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/    │
│  └── gmm/grouped_matmul_finalize_routing/op_kernel/arch35/...         │
└──────────────────────────────────────────────────────────────────────┘
```

### 4.1 新建公共头文件：`gmm/common/gmm_kernel_tool.h`

**职责**：存放所有 GMM kernel 可能复用的纯数学/数据搬运工具函数，与具体业务场景解耦。

**内容构成**（基于三份 `tool.h` 的并集与兼容性统一）：

```cpp
#ifndef GMM_COMMON_GMM_KERNEL_TOOL_H
#define GMM_COMMON_GMM_KERNEL_TOOL_H

#include "kernel_operator.h"

using AscendC::DataCopyExtParams;
using AscendC::DataCopyPadExtParams;
using AscendC::GlobalTensor;
using AscendC::int4b_t;
using AscendC::IsSameType;
using AscendC::LocalTensor;
using AscendC::ONE_BLK_SIZE;

namespace GmmOpKernel {

template <typename T>
__aicore__ inline T CeilAlign(T a, T b)
{
    ASCENDC_ASSERT(b != 0, { KERNEL_LOG(KERNEL_ERROR, "Division by zero error!"); });
    return (a + b - 1) / b * b;
}

__aicore__ inline uint32_t CeilAlign(uint32_t a, uint32_t b)
{
    ASCENDC_ASSERT(a <= (std::numeric_limits<uint32_t>::max() - b),
                   { KERNEL_LOG(KERNEL_ERROR, "CeilAlign uint32 over limit."); });
    ASCENDC_ASSERT(b != 0, { KERNEL_LOG(KERNEL_ERROR, "Division by zero error!"); });
    return (a + b - 1) / b * b;
}

template <typename T>
__aicore__ inline T CeilDiv(T a, T b)
{
    ASCENDC_ASSERT(b != 0, { KERNEL_LOG(KERNEL_ERROR, "Division by zero error!"); });
    return (a + b - 1) / b;
}

template <typename T>
__aicore__ inline T Min(T a, T b)
{
    return a < b ? a : b;
}

template <typename T>
__aicore__ inline void DataCopyPad2D(const LocalTensor<T> &dst, const GlobalTensor<T> &src,
                                     uint32_t blockCount, uint32_t blockLen,
                                     uint32_t dstInnerLength, uint32_t srcInnerLength)
{
    DataCopyExtParams params;
    params.blockCount = blockCount;
    params.blockLen = blockLen * sizeof(T);
    params.srcStride = (srcInnerLength - blockLen) * sizeof(T);
    params.dstStride = (dstInnerLength - blockLen) * sizeof(T) / ONE_BLK_SIZE;
    DataCopyPadExtParams<T> padParams;
    if (blockLen % (32 / sizeof(T)) != 0) {
        padParams.isPad = true;
        padParams.rightPadding = CeilAlign(blockLen, static_cast<uint32_t>(32 / sizeof(T))) - blockLen;
        padParams.paddingValue = 0;
    }
    if constexpr (IsSameType<T, int4b_t>::value || IsSameType<T, fp4x2_e2m1_t>::value ||
                  IsSameType<T, fp4x2_e1m2_t>::value) {
        params.blockLen = params.blockLen >> 1;
        params.srcStride = params.srcStride >> 1;
        params.dstStride = params.dstStride >> 1;
        padParams.rightPadding = padParams.rightPadding >> 1;
    }
    DataCopyPad(dst, src, params, padParams);
}

template <typename T>
__aicore__ inline void DataCopyPad2D(const GlobalTensor<T> &dst, const LocalTensor<T> &src,
                                     uint32_t dim1, uint32_t dim0,
                                     uint32_t srcFullDim0, uint32_t dstFullDim0)
{
    DataCopyExtParams params;
    params.blockCount = dim1;
    params.blockLen = dim0 * sizeof(T);
    params.srcStride = CeilDiv((srcFullDim0 - dim0) * sizeof(T), static_cast<uint64_t>(ONE_BLK_SIZE));
    params.dstStride = (dstFullDim0 - dim0) * sizeof(T);
    DataCopyPad(dst, src, params);
}

template <typename T>
__aicore__ constexpr uint32_t GetKBUnit()
{
    if constexpr (IsSameType<T, int4b_t>::value || IsSameType<T, fp4x2_e2m1_t>::value ||
                  IsSameType<T, fp4x2_e1m2_t>::value) {
        return 2048;
    }
    return 1024 / sizeof(T);
}

} // namespace GmmOpKernel
#endif
```

**关键兼容性处理**：
- `CeilDiv` 统一改名为 `CeilDiv`，后续在文件 1 的引用侧同步修改。
- `GetKBUnit` 采用通用公式 `1024 / sizeof(T)`，经推算对 `int8_t`(1024)、`float`(256)、`half`(512) 的结果与文件 1 原实现完全一致。
- `DataCopyPad2D` 统一包含 `fp4x2` 分支，因 `if constexpr` 在类型不匹配时不生成代码，对 A16W4 MSD 无副作用。

### 4.2 A16W4 MSD 侧头文件重命名与重构

**原文件**：`gmm/grouped_matmul/op_kernel/a16w4_msd/tool.h`

**处理方式**：
- **删除**原 `tool.h`。
- 由于该文件原内容经差异合并后，其工具函数已全部被 `gmm/common/gmm_kernel_tool.h` 覆盖，A16W4 MSD 特有的仅有 namespace 差异。
- 因此 A16W4 MSD 各源文件不再需要一个本地 `tool.h`，直接 `#include "gmm/common/gmm_kernel_tool.h"`，并在代码中使用 `GmmOpKernel::CeilDiv`（替代原 `CeilDiv`）等。
- 若未来 A16W4 MSD 需要补充 MSD 特有的工具函数，再新建 `a16w4_msd_utils.h`（或 `a16w4_msd_tool.h`，保证全局唯一即可）。

### 4.3 Arch35 WeightQuant 两侧头文件删除

**删除文件**：
1. `gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/tool.h`
2. `gmm/grouped_matmul_finalize_routing/op_kernel/arch35/common/tool.h`

**引用路径修改**：

| 原引用 | 新引用 |
|--------|--------|
| `gmm/grouped_matmul/op_kernel/a16w4_msd/*` 中的 `#include "tool.h"` | `#include "gmm/common/gmm_kernel_tool.h"` |
| `gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/*` 中的 `#include "tool.h"` | `#include "gmm/common/gmm_kernel_tool.h"` |
| `gmm/grouped_matmul_finalize_routing/op_kernel/arch35/common/basic_block_config.h` 中的 `#include "tool.h"` | `#include "gmm/common/gmm_kernel_tool.h"` |
| `gmm/grouped_matmul_finalize_routing/op_kernel/arch35/weight_quant_basic_block/*` 中的 `#include "../common/tool.h"` | `#include "gmm/common/gmm_kernel_tool.h"` |

由于 `gmm_kernel_tool.h` 内部已通过 `using` 将 `GmmOpKernel` 中的工具函数引入到 `WeightQuantBatchMatmulV2::Arch35` namespace，既有代码中直接写 `CeilAlign`、`DataCopyPad2D` 等无需修改。

---

## 5. 实施步骤

### Step 1：新建公共头文件

```bash
touch gmm/common/gmm_kernel_tool.h
# 写入 4.1 节内容
```

### Step 2：修改 A16W4 MSD 侧引用

修改以下 4 个文件中的 `#include "tool.h"`：

- `gmm/grouped_matmul/op_kernel/a16w4_msd/grouped_matmul_weight_quant_a16w4_msd_controller.h`
- `gmm/grouped_matmul/op_kernel/a16w4_msd/grouped_matmul_weight_quant_a16w4_msd_vector_service.h`
- `gmm/grouped_matmul/op_kernel/a16w4_msd/grouped_matmul_weight_quant_a16w4_msd_basic_block.h`
- `gmm/grouped_matmul/op_kernel/a16w4_msd/grouped_matmul_weight_quant_a16w4_msd_cube_service.h`

修改点：
1. 删除 `#include "tool.h"`。
2. 新增 `#include "gmm/common/gmm_kernel_tool.h"`。
3. 若文件中使用了 `CeilDiv`，替换为 `GmmOpKernel::CeilDiv`。
4. 若文件中使用了 `DataCopyPad2D`、`GetKBUnit` 等，添加 `GmmOpKernel::` 前缀或在文件中添加 `using GmmOpKernel::CeilDiv;` 等。

### Step 3：修改 Arch35 WeightQuant 侧引用

修改以下文件的 `#include`：

**grouped_matmul 侧（10 个文件）**：
- `gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/weight_quant_vec_compute.h`
- `gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/basic_block_config.h`
- `gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/anti_quant_y_vf.h`
- `gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/weight_quant_basic_block.h`
- `gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/weight_quant_cube_compute.h`
- `gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/basic_block_vf_nz.h`
- `gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/basic_api/weight_quant_basic_api_v1.h`
- `gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/weight_quant_vcv_basic_block.h`
- `gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/basic_block_vf_nd.h`

**grouped_matmul_finalize_routing 侧（5 个文件）**：
- `gmm/grouped_matmul_finalize_routing/op_kernel/arch35/common/basic_block_config.h`
- `gmm/grouped_matmul_finalize_routing/op_kernel/arch35/weight_quant_basic_block/gmm_fr_weight_quant_vec_compute.h`
- `gmm/grouped_matmul_finalize_routing/op_kernel/arch35/weight_quant_basic_block/gmm_fr_weight_quant_resplit_controller.h`
- `gmm/grouped_matmul_finalize_routing/op_kernel/arch35/weight_quant_basic_block/gmm_fr_weight_quant_vcv_basic_block.h`
- `gmm/grouped_matmul_finalize_routing/op_kernel/arch35/weight_quant_basic_block/gmm_fr_weight_quant_cube_compute.h`

统一修改：
- 删除 `#include "tool.h"` 或 `#include "../common/tool.h"`。
- 新增 `#include "gmm/common/gmm_kernel_tool.h"`。
- 由于公共头文件内已通过 `using` 引入工具函数，业务代码中现有调用无需改动。

### Step 4：删除旧头文件

```bash
rm gmm/grouped_matmul/op_kernel/a16w4_msd/tool.h
rm gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/tool.h
rm gmm/grouped_matmul_finalize_routing/op_kernel/arch35/common/tool.h
```

### Step 5：编译验证

1. 对 `gmm/grouped_matmul`、`gmm/grouped_matmul_finalize_routing` 进行全量编译。
2. 检查是否有 ODR 违例或头文件找不到报错。
3. 运行对应的 kernel UT/ST，确保精度无回退。

---

## 6. 头文件命名规范（后续约束）

为避免再次出现同名头文件，建议 GMM 模块后续遵循以下命名约定：

1. **禁止在业务子目录下使用 `tool.h`、`common.h`、`utils.h` 等过于泛化的文件名**。
2. 业务目录下的头文件命名应体现业务特征，例如：
   - `a16w4_msd_utils.h`
   - `weight_quant_basic_block_config.h`
3. 跨业务共享的头文件应上提到 `gmm/common/` 或 `gmm/op_kernel/common/`，并以 `gmm_` 前缀命名，例如：
   - `gmm_kernel_tool.h`
   - `gmm_gmm_kernel_tool.h`
4. 头文件保护宏（include guard）必须与文件路径关联，禁止使用简单的 `TOOL_H`、`COMMON_H` 等，推荐格式：
   ```cpp
   #ifndef GMM_COMMON_GMM_KERNEL_UTILS_H
   #define GMM_COMMON_GMM_KERNEL_UTILS_H
   ```

---

## 7. 风险与回滚

| 风险点 | 缓解措施 |
|--------|----------|
| include 路径未覆盖 `gmm/common/` | 检查并确保顶层 `CMakeLists.txt` 或编译脚本已将项目根目录（或 `gmm/` 目录）加入 `-I` 路径 |
| `CeilDiv` 改名遗漏 | 在 Step 2 中通过全局搜索 `CeilDiv(` 确认全部替换；编译失败会立即暴露未改点 |
| `fp4x2_e2m1_t` 在 A16W4 MSD 场景下未定义导致编译失败 | `if constexpr` 不会实例化不匹配的 branch，只要类型名在编译器可见范围内有声明即可通过编译；若确实有环境问题，可在公共头文件中加 `#ifdef` 保护 |
| 头文件删除后其他未识别到的引用 | 全局搜索 `tool.h`，确保三个目录之外无其他文件引用这三个特定路径的 tool.h（目前 grep 显示仅上述 19 个文件引用） |

若编译或测试发现问题，可快速通过 git revert 回滚本次提交，再针对性修复。

---

## 8. 预期收益

1. **消除 3 个同名头文件**，彻底避免头文件 shadowing 与 ODR 风险。
2. **消除约 200+ 行重复代码**（文件 2/3 合计约 350 行，其中 100% 重复；文件 1 与公共层重复约 80 行）。
3. **统一工具函数实现**：后续修复 `CeilAlign` 溢出检查、扩展 `DataCopyPad2D` 支持新数据类型，只需修改一处。
4. **建立公共层规范**：为 GMM 模块后续 kernel 开发提供可复用的基础工具与目录范式。
