# GMM 模块 `tool.h` 同名文件冲突解决方案

## 1. 问题概述

GMM 模块下存在多个名为 `tool.h` 的头文件，分布在不同算子子目录中，引发同名冲突及头文件保护宏重复定义问题。

### 1.1 涉及文件

| 序号 | 文件路径 | 头文件保护宏 | 所属命名空间 |
|------|---------|------------|------------|
| 1 | `gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/tool.h` | `GROUPED_MATMUL_WEIGHT_QUANT_TOOL_H` | `WeightQuantBatchMatmulV2::Arch35` |
| 2 | `gmm/grouped_matmul_finalize_routing/op_kernel/arch35/common/tool.h` | `GROUPED_MATMUL_WEIGHT_QUANT_TOOL_H` | `WeightQuantBatchMatmulV2::Arch35` |
| 3 | `gmm/grouped_matmul/op_kernel/a16w4_msd/tool.h` | `GROUPED_MATMUL_WEIGHT_QUANT_TOOL_H` | `GROUPED_MATMUL::A16W4Msd` |

### 1.2 核心问题

1. **头文件保护宏完全重复**：文件 1、2、3 均使用相同的保护宏 `GROUPED_MATMUL_WEIGHT_QUANT_TOOL_H`。当任意两个文件被同一个编译单元间接包含时，后一个文件的全部内容会被预处理器跳过，可能导致符号缺失或类型不匹配。
2. **代码严重重复**：文件 1 和文件 2 内容高度重合（差异仅在于 `VEC_MAX_ELEM_B32` 常量及格式空格），属于典型的 Copy-Paste 代码，增加维护成本。
3. **同名文件语义混淆**：开发者在不同目录下看到同名 `tool.h`，难以快速判断应该引用哪一个，且 IDE 跳转容易进入错误文件。

---

## 2. 可选方案对比

| 方案 | 思路 | 优点 | 缺点 | 适用阶段 |
|------|------|------|------|---------|
| **A. 提取公共代码** | 将文件 1 与文件 2 的公共内容下沉到 `gmm/common/` 共享目录，差异部分在各自模块内单独维护 | 彻底消除重复代码与同名冲突；后续修改只需维护一处 | 涉及跨算子目录重构，需同步修改 CMake/include 路径；需梳理两个文件的细微差异（如 `VEC_MAX_ELEM_B32`） | **长期推荐** |
| **B. 重命名文件** | 按模块语义重命名，如 `gmm_fr_arch35_weight_quant_tool.h`、`grouped_matmul_arch35_weight_quant_tool.h` 等，并赋予唯一保护宏 | 改动范围可控，不引入跨模块依赖；可快速消除冲突 | 代码重复问题仍未解决，后续仍需同步修改多份文件 | **短期应急** |
| **C. 符号链接** | 保留一份物理文件，其余位置使用软链接指向它 | 无代码重复 | 跨平台兼容性差（Windows 构建环境可能不支持）；符号链接在版本控制中行为不一致；IDE 索引可能异常 | 不推荐 |
| **D. 仅修复保护宏** | 不改文件名，仅将三处 `GROUPED_MATMUL_WEIGHT_QUANT_TOOL_H` 改为各自唯一的宏 | 改动最小，风险最低 | 同名文件继续存在，开发者体验差；代码重复依旧 | 不推荐 |

---

## 3. 推荐方案：A+B 组合实施

建议采用 **"短期重命名去冲突 + 中期提取公共代码"** 的分阶段策略，兼顾风险与代码健康度。

### 3.1 第一阶段：重命名与修复宏（短期，1~2 天）

**目标**：立即消除同名冲突与宏重复定义风险。

#### 3.1.1 文件 1 & 2（Arch35 WeightQuant）

由于文件 1 和文件 2 位于 Arch35 weight quant 相关目录下，建议统一重命名为具有明确上下文的名称：

- `gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/tool.h`
  → `gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/**weight_quant_arch35_tool.h**`

- `gmm/grouped_matmul_finalize_routing/op_kernel/arch35/common/tool.h`
  → `gmm/grouped_matmul_finalize_routing/op_kernel/arch35/common/**gmm_fr_weight_quant_arch35_tool.h**`

并同步修改头文件保护宏：

```cpp
// weight_quant_arch35_tool.h
#ifndef GROUPED_MATMUL_WEIGHT_QUANT_ARCH35_TOOL_H
#define GROUPED_MATMUL_WEIGHT_QUANT_ARCH35_TOOL_H
// ...
#endif

// gmm_fr_weight_quant_arch35_tool.h
#ifndef GMM_FR_WEIGHT_QUANT_ARCH35_TOOL_H
#define GMM_FR_WEIGHT_QUANT_ARCH35_TOOL_H
// ...
#endif
```

#### 3.1.2 文件 3（A16W4 MSD）

- `gmm/grouped_matmul/op_kernel/a16w4_msd/tool.h`
  → `gmm/grouped_matmul/op_kernel/a16w4_msd/**a16w4_msd_tool.h**`

```cpp
#ifndef GROUPED_MATMUL_A16W4_MSD_TOOL_H
#define GROUPED_MATMUL_A16W4_MSD_TOOL_H
// ...
#endif
```

#### 3.1.3 批量更新引用路径

需修改所有 `#include "tool.h"` 及相对路径引用：

```bash
# 使用 ripgrep 确认引用点（示例）
rg '#include.*tool\.h' gmm/ --type cpp --type h
```

主要涉及的文件列表（需根据实际情况补充）：

- `gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/*.h`
- `gmm/grouped_matmul/op_kernel/a16w4_msd/*.h`
- `gmm/grouped_matmul_finalize_routing/op_kernel/arch35/weight_quant_basic_block/*.h`
- `gmm/grouped_matmul_finalize_routing/op_kernel/arch35/common/*.h`

### 3.2 第二阶段：提取公共代码（中期，1 周内）

**目标**：消除文件 1 与文件 2 之间的重复代码。

#### 3.2.1 差异分析

当前文件 1 与文件 2 的主要差异：

| 差异项 | 文件 1（grouped_matmul） | 文件 2（gmm_fr） | 处理建议 |
|--------|------------------------|------------------|---------|
| `VEC_MAX_ELEM_B32` | ❌ 缺失 | ✅ 存在 | 该常量属于通用向量计算工具，应下沉到公共头文件 |
| 常量/函数内容 | 其余 99% 相同 | 其余 99% 相同 | 提取到公共头文件 |

> **结论**：两个文件本质上应是同一份代码，差异项 `VEC_MAX_ELEM_B32` 可合并到公共版本中。

#### 3.2.2 新建公共目录与文件

建议新建共享目录（若不存在）：

```
gmm/
└── common/
    └── op_kernel/
        └── arch35/
            └── weight_quant_tool.h      <-- 新增公共头文件
```

公共头文件内容：取文件 1 与文件 2 的并集（保留 `VEC_MAX_ELEM_B32`），命名空间保持 `WeightQuantBatchMatmulV2::Arch35`。

```cpp
// gmm/common/op_kernel/arch35/weight_quant_tool.h
#ifndef GMM_COMMON_WEIGHT_QUANT_ARCH35_TOOL_H
#define GMM_COMMON_WEIGHT_QUANT_ARCH35_TOOL_H

#include "kernel_operator.h"
// ... 公共内容（含 VEC_MAX_ELEM_B32）...
namespace WeightQuantBatchMatmulV2::Arch35 {
// ...
}
#endif
```

#### 3.2.3 删除重复文件并更新引用

- 删除 `gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/weight_quant_arch35_tool.h`
- 删除 `gmm/grouped_matmul_finalize_routing/op_kernel/arch35/common/gmm_fr_weight_quant_arch35_tool.h`

在两侧模块中统一改为：

```cpp
#include "gmm/common/op_kernel/arch35/weight_quant_tool.h"
```

> **注意**：若构建系统使用基于目录的 include path（如 `-I${PROJECT_ROOT}`），则上述绝对风格路径可直接使用；若使用相对路径，需在 CMake 中新增 `gmm/common/op_kernel/arch35` 的 include 目录，或保持相对路径引用 `../../common/op_kernel/arch35/weight_quant_tool.h`。建议优先采用统一 include prefix 风格，提升可读性。

### 3.3 命名规范建议（预防复发）

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

## 4. 实施检查清单（Checklist）

- [ ] 完成文件重命名（第一阶段）
- [ ] 所有头文件保护宏已唯一化，不再重复
- [ ] 所有 `#include` 引用路径已更新，编译通过
- [ ] 新增公共目录 `gmm/common/op_kernel/arch35/`（第二阶段）
- [ ] 公共 `weight_quant_tool.h` 已合并 `VEC_MAX_ELEM_B32` 差异
- [ ] 删除旧重复文件，确认无残留引用
- [ ] 运行 GMM 相关单元测试/算子测试，验证功能无损
- [ ] 代码审查（CR）通过

---

## 5. 风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| 引用路径遗漏导致编译失败 | 中 | 中 | 全局搜索所有 `#include`；CI 流水线编译验证 |
| 提取公共代码后引入宏/命名空间冲突 | 低 | 中 | 保持原有命名空间不变；确保保护宏唯一 |
| 删除旧文件后 git 历史追溯困难 | 低 | 低 | 分两次 commit（先重命名，再提取），保留 git mv 记录 |
| 跨模块 include 路径导致构建系统不兼容 | 低 | 高 | 提前确认 CMake/编译脚本的 include 规则；必要时新增 include 目录 |

---

## 6. 附录：相关引用速查

```text
gmm/grouped_matmul/op_kernel/a16w4_msd/grouped_matmul_weight_quant_a16w4_msd_controller.h
gmm/grouped_matmul/op_kernel/a16w4_msd/grouped_matmul_weight_quant_a16w4_msd_vector_service.h
gmm/grouped_matmul/op_kernel/a16w4_msd/grouped_matmul_weight_quant_a16w4_msd_basic_block.h
gmm/grouped_matmul/op_kernel/a16w4_msd/grouped_matmul_weight_quant_a16w4_msd_cube_service.h
gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/weight_quant_vec_compute.h
gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/basic_block_config.h
gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/weight_quant_vcv_basic_block.h
gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/anti_quant_y_vf.h
gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/weight_quant_basic_block.h
gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/weight_quant_cube_compute.h
gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/basic_block_vf_nd.h
gmm/grouped_matmul_finalize_routing/op_kernel/arch35/common/basic_block_config.h
gmm/grouped_matmul_finalize_routing/op_kernel/arch35/weight_quant_basic_block/gmm_fr_weight_quant_vec_compute.h
gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/basic_api/weight_quant_basic_api_v1.h
gmm/grouped_matmul_finalize_routing/op_kernel/arch35/weight_quant_basic_block/gmm_fr_weight_quant_resplit_controller.h
gmm/grouped_matmul_finalize_routing/op_kernel/arch35/weight_quant_basic_block/gmm_fr_weight_quant_vcv_basic_block.h
gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/basic_block_vf_nz.h
gmm/grouped_matmul_finalize_routing/op_kernel/arch35/weight_quant_basic_block/gmm_fr_weight_quant_cube_compute.h
```
