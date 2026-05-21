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

## 2. 方案决策

| 方案 | 思路 | 优点 | 缺点 | 决策 |
|------|------|------|------|------|
| **A. 提取公共代码** | 将文件 1 与文件 2 的公共内容下沉到 `gmm/common/` 共享目录 | 彻底消除重复代码与同名冲突；后续修改只需维护一处 | 涉及跨算子目录重构，需同步修改 include 路径 | **✅ 采用（第二阶段）** |
| **B. 重命名文件** | 按模块语义重命名，赋予唯一保护宏 | 改动范围可控，可快速消除冲突 | 代码重复问题仍未解决 | **✅ 采用（第一阶段）** |
| **C. 符号链接** | 保留一份物理文件，其余位置使用软链接 | 无代码重复 | 跨平台兼容性差；版本控制行为不一致 | 不采用 |
| **D. 仅修复保护宏** | 不改文件名，仅修改宏 | 改动最小 | 同名文件继续存在，开发者体验差；代码重复依旧 | 不采用 |

最终采用 **A + B 组合方案**：先重命名消除冲突，再提取公共代码消除重复。

---

## 3. 实施记录

### 3.1 第一次提交：提取公共代码并修复宏

**提交信息**：`fix(gmm): resolve tool.h name conflict in arch35 weight quant kernels`

**完成内容**：

1. **提取公共头文件**
   - 新建 `gmm/common/op_kernel/arch35/weight_quant_tool.h`
   - 合并原文件 1 与文件 2 的并集，保留 `VEC_MAX_ELEM_B32` 差异项
   - 命名空间保持 `WeightQuantBatchMatmulV2::Arch35`
   - 保护宏定义为 `GMM_COMMON_WEIGHT_QUANT_ARCH35_TOOL_H`

2. **重命名 Arch35 文件并添加 Wrapper**
   - `gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/tool.h`
     → `weight_quant_arch35_tool.h`（仅包含 `#include "../../../../common/op_kernel/arch35/weight_quant_tool.h"`）
   - `gmm/grouped_matmul_finalize_routing/op_kernel/arch35/common/tool.h`
     → `gmm_fr_weight_quant_arch35_tool.h`（仅包含 `#include "../../../../common/op_kernel/arch35/weight_quant_tool.h"`）

3. **重命名 A16W4 MSD 文件**
   - `gmm/grouped_matmul/op_kernel/a16w4_msd/tool.h`
     → `a16w4_msd_tool.h`
   - 保护宏更新为 `GROUPED_MATMUL_A16W4_MSD_TOOL_H`

4. **批量更新引用路径**
   - 更新所有 `#include "tool.h"` 为对应的新文件名
   - 涉及文件见第 6 节附录

### 3.2 第二次提交：删除 Wrapper 头文件，扁平化 Include 链

**提交信息**：`fix(gmm): remove arch35 wrapper headers and flatten includes`

**完成内容**：

1. **删除中间 Wrapper 头文件**
   - 删除 `gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/weight_quant_arch35_tool.h`
   - 删除 `gmm/grouped_matmul_finalize_routing/op_kernel/arch35/common/gmm_fr_weight_quant_arch35_tool.h`

2. **扁平化 Include 路径**
   - 将两侧模块中共 14 个源文件的 `#include` 直接指向公共头文件：
   ```cpp
   #include "../../../../common/op_kernel/arch35/weight_quant_tool.h"
   ```

3. **确认无残留引用**
   - 全局搜索确认代码库中已无对上述两个 wrapper 头文件的引用

> **注意**：当前实现采用相对路径引用。后续若构建系统支持基于目录的 include path（如 `-I${PROJECT_ROOT}`），可再统一为绝对路径 `gmm/common/op_kernel/arch35/weight_quant_tool.h` 以提升可读性。

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

- [x] 完成文件重命名（第一阶段）
- [x] 所有头文件保护宏已唯一化，不再重复
- [x] 所有 `#include` 引用路径已更新，编译通过
- [x] 新增公共目录 `gmm/common/op_kernel/arch35/`（第二阶段）
- [x] 公共 `weight_quant_tool.h` 已合并 `VEC_MAX_ELEM_B32` 差异
- [x] 删除旧重复文件 `weight_quant_arch35_tool.h`、`gmm_fr_weight_quant_arch35_tool.h`，确认无残留引用
- [ ] 运行 GMM 相关单元测试/算子测试，验证功能无损
- [ ] 代码审查（CR）通过

---

## 5. 风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| 引用路径遗漏导致编译失败 | 中 | 中 | 全局搜索所有 `#include`；CI 流水线编译验证 |
| 提取公共代码后引入宏/命名空间冲突 | 低 | 中 | 保持原有命名空间不变；确保保护宏唯一 |
| 删除旧文件后 git 历史追溯困难 | 低 | 低 | 分两次 commit（先重命名+提取，再删除 wrapper），保留 git mv 记录 |
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
