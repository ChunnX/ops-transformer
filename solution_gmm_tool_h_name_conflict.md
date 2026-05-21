# GMM 模块 `tool.h` 同名文件冲突解决方案

## 1. 问题概述

GMM 模块下曾存在多个名为 `tool.h` 的头文件，分布在不同算子子目录中，引发同名冲突及头文件保护宏重复定义问题。

### 1.1 涉及文件

| 序号 | 原文件路径 | 头文件保护宏 | 所属命名空间 |
|------|-----------|------------|------------|
| 1 | `gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/tool.h` | `GROUPED_MATMUL_WEIGHT_QUANT_TOOL_H` | `WeightQuantBatchMatmulV2::Arch35` |
| 2 | `gmm/grouped_matmul_finalize_routing/op_kernel/arch35/common/tool.h` | `GROUPED_MATMUL_WEIGHT_QUANT_TOOL_H` | `WeightQuantBatchMatmulV2::Arch35` |
| 3 | `gmm/grouped_matmul/op_kernel/a16w4_msd/tool.h` | `GROUPED_MATMUL_WEIGHT_QUANT_TOOL_H` | `GROUPED_MATMUL::A16W4Msd` |

### 1.2 核心问题

1. **头文件保护宏完全重复**：文件 1、2、3 均使用相同的保护宏 `GROUPED_MATMUL_WEIGHT_QUANT_TOOL_H`。当任意两个文件被同一个编译单元间接包含时，后一个文件的全部内容会被预处理器跳过，可能导致符号缺失或类型不匹配。
2. **代码严重重复**：文件 1 和文件 2 内容高度重合（差异仅在于 `VEC_MAX_ELEM_B32` 常量及格式空格），属于典型的 Copy-Paste 代码，增加维护成本。
3. **同名文件语义混淆**：开发者在不同目录下看到同名 `tool.h`，难以快速判断应该引用哪一个，且 IDE 跳转容易进入错误文件。

---

## 2. 方案决策

| 方案 | 思路 | 优点 | 缺点 | 决策 |
|------|------|------|------|------|
| **A. 提取公共代码** | 将文件 1 与文件 2 的公共内容下沉到 `gmm/common/` 共享目录 | 彻底消除重复代码与同名冲突；后续修改只需维护一处 | 涉及跨算子目录重构，需同步修改 include 路径 | **采用** |
| **B. 重命名文件** | 按模块语义重命名，赋予唯一保护宏 | 改动范围可控，可快速消除冲突 | 代码重复问题仍未解决 | **采用（过渡）** |
| **C. 符号链接** | 保留一份物理文件，其余位置使用软链接 | 无代码重复 | 跨平台兼容性差；版本控制行为不一致 | 不采用 |
| **D. 仅修复保护宏** | 不改文件名，仅修改宏 | 改动最小 | 同名文件继续存在，开发者体验差；代码重复依旧 | 不采用 |

最终采用 **提取公共代码 + 重命名** 的组合策略：通过提取公共头文件消除 Arch35 两个模块的代码重复，通过重命名消除 A16W4 MSD 模块的同名冲突，并辅以命名规范预防复发。

---

## 3. 最终架构设计

### 3.1 目录结构

```
gmm/
├── common/
│   └── op_kernel/
│       └── arch35/
│           └── weight_quant_tool.h          # 公共头文件（Arch35 WeightQuant）
│
├── grouped_matmul/
│   └── op_kernel/
│       ├── a16w4_msd/
│       │   └── a16w4_msd_tool.h             # 独立头文件（A16W4 MSD）
│       └── arch35/
│           └── weight_quant_basic_block/
│               ├── anti_quant_y_vf.h
│               ├── basic_block_config.h
│               ├── basic_block_vf_nd.h
│               ├── basic_block_vf_nz.h
│               ├── weight_quant_basic_api_v1.h
│               ├── weight_quant_basic_block.h
│               ├── weight_quant_cube_compute.h
│               ├── weight_quant_vcv_basic_block.h
│               └── weight_quant_vec_compute.h
│
└── grouped_matmul_finalize_routing/
    └── op_kernel/
        └── arch35/
            ├── common/
            │   └── basic_block_config.h
            └── weight_quant_basic_block/
                ├── gmm_fr_weight_quant_cube_compute.h
                ├── gmm_fr_weight_quant_resplit_controller.h
                ├── gmm_fr_weight_quant_vcv_basic_block.h
                └── gmm_fr_weight_quant_vec_compute.h
```

### 3.2 公共头文件设计

`gmm/common/op_kernel/arch35/weight_quant_tool.h` 为 Arch35 WeightQuant 相关算子提供共享定义，合并了原 `tool.h` 的并集内容（含 `VEC_MAX_ELEM_B32`），命名空间保持 `WeightQuantBatchMatmulV2::Arch35`。

```cpp
#ifndef GMM_COMMON_WEIGHT_QUANT_ARCH35_TOOL_H
#define GMM_COMMON_WEIGHT_QUANT_ARCH35_TOOL_H

#include "kernel_operator.h"
// ... 公共内容（含 VEC_MAX_ELEM_B32）...
namespace WeightQuantBatchMatmulV2::Arch35 {
// ...
}
#endif
```

### 3.3 模块引用方式

Arch35 两侧模块（`grouped_matmul` 与 `grouped_matmul_finalize_routing`）统一通过相对路径直接引用公共头文件：

```cpp
#include "../../../../common/op_kernel/arch35/weight_quant_tool.h"
```

A16W4 MSD 模块保持独立头文件 `a16w4_msd_tool.h`，通过重命名消除了同名冲突：

```cpp
#ifndef GROUPED_MATMUL_A16W4_MSD_TOOL_H
#define GROUPED_MATMUL_A16W4_MSD_TOOL_H
// ...
#endif
```

### 3.4 头文件保护宏设计

| 文件 | 保护宏 |
|------|--------|
| `gmm/common/op_kernel/arch35/weight_quant_tool.h` | `GMM_COMMON_WEIGHT_QUANT_ARCH35_TOOL_H` |
| `gmm/grouped_matmul/op_kernel/a16w4_msd/a16w4_msd_tool.h` | `GROUPED_MATMUL_A16W4_MSD_TOOL_H` |

> **注意**：原 Arch35 两个模块的 `tool.h` 已分别提取至公共头文件并删除，不再存在同名文件及宏重复问题。

---

## 4. 命名规范建议（预防复发）

为避免后续再次出现同名冲突，建议制定模块内头文件命名规范：

```
<module>_<sub_feature>_<arch>_<type>.h

示例：
- weight_quant_arch35_tool.h
- gmm_fr_weight_quant_arch35_tool.h
- a16w4_msd_tool.h
```

禁止在不同算子目录下直接使用无差别的通用名称（如 `tool.h`、`common.h`、`utils.h`）。

---

## 5. 风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| 引用路径遗漏导致编译失败 | 中 | 中 | 全局搜索所有 `#include`；CI 流水线编译验证 |
| 提取公共代码后引入宏/命名空间冲突 | 低 | 中 | 保持原有命名空间不变；确保保护宏唯一 |
| 跨模块 include 路径导致构建系统不兼容 | 低 | 高 | 提前确认 CMake/编译脚本的 include 规则；必要时新增 include 目录 |

---

## 6. 附录：受影响文件列表

```text
gmm/common/op_kernel/arch35/weight_quant_tool.h
gmm/grouped_matmul/op_kernel/a16w4_msd/a16w4_msd_tool.h
gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/anti_quant_y_vf.h
gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/basic_api/weight_quant_basic_api_v1.h
gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/basic_block_config.h
gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/basic_block_vf_nd.h
gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/basic_block_vf_nz.h
gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/weight_quant_basic_block.h
gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/weight_quant_cube_compute.h
gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/weight_quant_vcv_basic_block.h
gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/weight_quant_vec_compute.h
gmm/grouped_matmul_finalize_routing/op_kernel/arch35/common/basic_block_config.h
gmm/grouped_matmul_finalize_routing/op_kernel/arch35/weight_quant_basic_block/gmm_fr_weight_quant_cube_compute.h
gmm/grouped_matmul_finalize_routing/op_kernel/arch35/weight_quant_basic_block/gmm_fr_weight_quant_resplit_controller.h
gmm/grouped_matmul_finalize_routing/op_kernel/arch35/weight_quant_basic_block/gmm_fr_weight_quant_vcv_basic_block.h
gmm/grouped_matmul_finalize_routing/op_kernel/arch35/weight_quant_basic_block/gmm_fr_weight_quant_vec_compute.h
```
