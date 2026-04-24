# 复杂算子 op_host 完整适配指南

**定位**：复杂算子迁移的**自包含**指南，覆盖从复杂度判定到编译调试的全流程。
复杂算子定义见下方第0节。简单算子请直接使用 `conversion_rules.md`。

以下通用规则仍需按 `conversion_rules.md` 执行：第1节（头文件处理）、第2节（tiling.cpp必需头文件三件套）、第3节（非目标平台清理）、第7节（日志宏定义）、第8节（op_kernel不需要include tiling.h）。
复杂算子特有的所有额外规则均在本文件内。

---

## 0. 复杂度判断标准

满足以下**任意一条**即为"复杂算子"：

1. op_host/ 的 `#include` 指向**同级其他算子目录**（如 `#include "../mat_mul_v3/xxx.h"` 或 `#include "matmul/mat_mul_v3/..."`)
2. op_host/ 存在 `common/` 子目录且含有 tiling 相关头文件
3. Tiling 类继承链超过1层（如 `class XxxTiling : public YyyBaseTiling : public TilingBaseClass`）
4. CompileInfo 含 **cube 硬件参数**（l1Size, l0ASize, l0BSize, l0CSize, l2Size, btSize 等）或**平台特性探测**（supportL0c2out, npuArch 等），且被 tiling 继承链深度使用
5. 包含 `ASCENDC_TPL_*` 或 `GenSimplifiedKey`
6. 引用框架内部分头文件（`op_cache_tiling.h`、`runtime_kb_api.h`）

**注意**：仅使用 `TilingParse<XxxCompileInfo>` 注册本身**不构成**复杂算子。许多简单算子（如 ForeachAbs、GeGluV2）也使用 TilingParse，但 CompileInfo 为空结构体或仅含基础参数（coreNum, ubSize），可直接按 `conversion_rules.md` 第4节处理。

通过 `is_complex` 命令可自动判定：

```bash
python .claude/skills/migrate-op/scripts/migrate_helper.py is_complex <src_dir>
```

---

## 1. 依赖发现与文件收集

### 1.1 自动发现

```bash
python scripts/migrate_helper.py is_complex <src_dir>
python scripts/migrate_helper.py discover <src_dir>
```

`is_complex` 输出 JSON，包含 `is_complex: bool` 和匹配的原因列表。
`discover` 输出直接依赖目录及文件列表、框架内部分依赖列表。

### 1.2 手动扫描方法

若自动工具不精确，手动在 `op_host/` 所有 `.cpp/.h` 中搜索：

```bash
grep -rn '#include' <src_dir>/op_host/ | grep -E '\.\./|matmul/' | grep -v arch35
```

关注以下模式：
- `#include "matmul/mat_mul_v3/..."` — 跨算子依赖
- `#include "matmul/common/..."` — 共享工具依赖
- `#include "../xxx/..."` — 同级目录依赖
- `#include "op_cache_tiling.h"` — 框架内部分依赖

### 1.3 典型依赖链（matmul 系列）

```
batch_mat_mul_v3
  ├── matmul/mat_mul_v3  (基础 matmul tiling + CompileInfo)
  │   ├── op_host/op_tiling/matmul_v3_base_tiling.h/.cpp
  │   ├── op_host/op_tiling/matmul_v3_common.h
  │   ├── op_host/op_tiling/matmul_v3_compile_info.h
  │   ├── op_host/op_tiling/matmul_v3_tiling.h/.cpp
  │   ├── op_host/op_tiling/matmul_v3_tuning.h/.cpp
  │   ├── op_host/op_tiling/matmul_v3_l2_cache.h/.cpp
  │   ├── op_host/op_tiling/matmul_v3_simplifiedkey.h
  │   ├── op_host/op_tiling/matmul_v3_platform_common.h
  │   ├── op_kernel/mat_mul_v3_tiling_data.h
  │   └── op_kernel/mat_mul_v3_tiling_key.h
  └── matmul/common
      └── op_host/math_util.h
```

### 1.4 文件分类

| 类别 | 处理方式 | 示例 |
|:---|:---|:---|
| 必须 copy | 保留原逻辑 | `matmul_v3_base_tiling.h/.cpp`, `matmul_v3_compile_info.h` |
| 必须 copy (kernel) | 保留 tiling 数据定义 | `mat_mul_v3_tiling_data.h`, `*_tiling_key.h` |
| 框架内部 (stub) | 删除 include + 删除调用 | `op_cache_tiling.h`, `runtime_kb_api.h` |
| 删除 (非目标平台) | 不 copy | `arch35/` 子目录, `config/` 目录 |
| 删除 (测试/文档) | 不 copy | `tests/`, `docs/`, `examples/` |

---

## 2. CompileInfo 处理决策树

本节仅适用于**复杂算子**。简单算子的 CompileInfo 处理见 `conversion_rules.md` 第4节。

```
CompileInfo 使用方式？
├── 仅 totalCoreNum, ubSize, aivCoreNum 等基础参数
│   └── 非复杂算子：按 conversion_rules.md 第4节处理（替换或保留均可）
│
├── 含 cube 硬件参数（l1Size, l0ASize, l0BSize, l0CSize, l2Size, btSize 等）
│   或含平台特性探测（supportL0c2out, supportL12BtBf16, npuArch 等）
│   且被 tiling 继承链深度使用
│   └── 复杂算子：保留 CompileInfo + TilingParse（见下方规则）
│
└── 含 bank/cache 相关字段（由 TilingPrepareForOpCache 填充）
    └── 保留 CompileInfo，但 stub OpCache 调用（见第3节）
```

### CompileInfo 保留规则

1. **保留** CompileInfo 结构体定义，不做任何字段删减
2. **保留** TilingParse 注册：`.TilingParse<XxxCompileInfo>(TilingPrepareForXxx)`
3. **改造** TilingPrepare 函数：
   - 保留 `fe::PlatFormInfos` → `PlatformAscendC` 的硬件参数填充路径
   - 保留 `GetPlatformRes("AICoreintrinsicDtypeMap", ...)` 等平台特性查询
   - **删除** `TilingPrepareForOpCache(context)` 调用（见第3节）
   - **删除** `socVersionType` 相关条件分支（如 310p 判断）
4. **保留** GenSimplifiedKey 注册和函数实现

---

## 3. 框架内部分依赖 Stub 策略

**判断条件**：源码 include 了以下不在 CANN SDK 中的头文件：

| 头文件 | 实际位置 | 问题 |
|:---|:---|:---|
| `op_cache_tiling.h` | `libophost_comm_legacy.so` | 链接失败 |
| `runtime_kb_api.h` | `libophost_comm_legacy.so` | 链接失败 |
| `op_cache_def_tiling.h` | 被 `op_cache_tiling.h` 引用 | 链接失败 |

**Stub 策略**：
1. **删除** 所有 `#include "op_cache_tiling.h"` 和 `#include "runtime_kb_api.h"` 行
2. **删除** `using Ops::NN::TilingPrepareForOpCache` 等 using 声明
3. **注释掉** `TilingPrepareForOpCache(context)` 调用
4. **注释掉** `QueryBank(...)` 和 `GenTiling(...)` 调用
5. 删除后不需要任何替代代码 — 这些功能是 AOE 调优/知识库缓存，单算子工程不依赖

### 改造示例

```cpp
// 替换前（TilingPrepareForBatchMatMulV3 末尾）
    if(!TilingPrepareForOpCache(context)) {
        OP_LOGE(context->GetNodeName(), "TilingPrepareForOpCache fail");
        return ge::GRAPH_FAILED;
    }

// 替换后
    // TilingPrepareForOpCache removed: framework-internal, not available in single-op build
```

```cpp
// 替换前（tiling.cpp 中）
#include "op_host/tiling_templates_registry.h"
#include "op_cache_tiling.h"
#include "runtime_kb_api.h"
using Ops::NN::TilingPrepareForOpCache;

// 替换后
#include "op_host/tiling_templates_registry.h"
// REMOVED: op_cache_tiling.h, runtime_kb_api.h (framework-internal, stubbed)
```

---

## 4. TilingBaseClass 继承链适配

复杂算子常见继承链：
```
TilingBaseClass (common/tiling_base.h)
  └── MatmulV3BaseTiling (matmul_v3/)
      └── BatchMatmulV3BaseTiling (batch_mat_mul_v3/)
```

**适配规则**：

1. `TilingBaseClass` 已由项目 `libs/common_include/` 提供（参考 RmsNormQuant 的 `common/tiling_base.h`），需复制到 `answer/0/op_host/common/tiling_base.h`
2. 中间层 Tiling 类（`MatmulV3BaseTiling`）从依赖算子目录复制到 `answer/0/op_host/matmul_v3/`
3. 仅对最底层的具体 Tiling 类和 _def.cpp 做平台清理（删除 310p 分支等）
4. **不在继承链中间层替换 CompileInfo** — 基类的 `compileInfo_` 成员被整个链路使用
5. 保持所有类的 `#include` 指向本地改写后的路径

### common/ 目录所需文件

参考 `tasks/level2/Norm/RmsNormQuant/answer/0/op_host/common/`，提供：

| 文件 | 用途 |
|:---|:---|
| `tiling_base.h` | TilingBaseClass 基类 + AiCoreParams 结构 |
| `tiling_templates_registry.h` | REGISTER_TILING_TEMPLATE 宏 |
| `error_util.h` | CUBE_INNER_ERR_REPORT, OP_TILING_CHECK, OPS_CHECK_NULL_WITH_CONTEXT 等宏 |
| `log.h` | OP_LOGD/E/I/W 宏 |

---

## 5. 跨算子 Include 路径改写

**判断条件**：源码包含指向其他算子目录的 include，如：
- `#include "matmul/mat_mul_v3/op_host/op_tiling/matmul_v3_base_tiling.h"`
- `#include "matmul/common/op_host/math_util.h"`
- `#include "../../op_kernel/mat_mul_v3_tiling_data.h"`

**规则**：

1. 所有依赖文件复制到 `answer/0/op_host/` 下的对应子目录
2. 按下表改写 include 路径：

| 源码路径 | 目标路径 |
|:---|:---|
| `"matmul/mat_mul_v3/op_host/op_tiling/xxx.h"` | `"matmul_v3/xxx.h"` |
| `"matmul/mat_mul_v3/op_kernel/xxx.h"` | `"matmul_v3_kernel/xxx.h"` |
| `"../../op_kernel/xxx.h"` | `"matmul_v3_kernel/xxx.h"` |
| `"matmul/common/op_host/xxx.h"` | `"common/xxx.h"` |
| `"matmul/common/op_kernel/xxx.h"` | `"common/xxx.h"` |

3. `common/` 目录复用已有的 `log.h`、`error_util.h`（参考 `tasks/level2/Norm/RmsNormQuant/answer/0/op_host/common/`）
4. `op_host/tiling_base.h`、`op_host/tiling_templates_registry.h` 等使用项目已提供的版本

`extract_complex` 命令自动执行路径改写。

### 目标目录结构

```
answer/0/op_host/
  common/                          # 共享工具（复用已有文件）
    tiling_base.h
    tiling_templates_registry.h
    error_util.h
    log.h
  matmul_v3/                       # mat_mul_v3 依赖文件
    matmul_v3_base_tiling.h/.cpp
    matmul_v3_compile_info.h
    matmul_v3_common.h
    matmul_v3_tiling.h/.cpp
    matmul_v3_tuning.h
    matmul_v3_l2_cache.h/.cpp
    matmul_v3_simplifiedkey.h
    matmul_v3_platform_common.h
  matmul_v3_kernel/                # mat_mul_v3 kernel tiling 数据
    mat_mul_v3_tiling_data.h
    mat_mul_v3_tiling_key.h
  op_tiling/                       # 本算子 tiling（保持原目录结构）
    batch_mat_mul_v3_tiling.h/.cpp
    batch_mat_mul_v3_base_tiling.h/.cpp
  batch_mat_mul_v3_def.cpp
  CMakeLists.txt                   # 使用 GLOB_RECURSE 模板
```

---

## 6. _def.cpp 适配

复杂算子 _def.cpp 的典型改造：

```cpp
// 删除这些框架专用标志
.ExtendCfgInfo("softsync.flag", "true")         // 删除
.ExtendCfgInfo("opFile.value", "batch_mat_mul_v3_apt")  // 删除
.DynamicCompileStaticFlag(true)                  // 删除
.DynamicFormatFlag(false)                        // 删除
.DynamicRankSupportFlag(true)                    // 删除
.DynamicShapeSupportFlag(true)                   // 删除
.NeedCheckSupportFlag(false)                     // 删除

// 仅保留 ascend910b/ascend910_93 平台配置
this->AICore().AddConfig("ascend910b", aicConfig);
this->AICore().AddConfig("ascend910_93", aicConfig);
// 删除 ascend310p, ascend950, kirinx90, kirin9030 等配置

// 删除 GetKirinCoreConfig() 等非目标平台辅助函数
```

保留：
- 输入/输出/属性的 DataType、Format 定义
- `OP_ADD(Xxx)` 宏
- TilingParse / GenSimplifiedKey 注册（如果在独立文件中）

---

## 7. CMakeLists.txt 适配

复杂算子 op_host 需要使用 `GLOB_RECURSE` 自动收集子目录文件。
完整模板见 `templates/op_host_CMakeLists.txt`，由 `extract_complex` 命令自动放置。

如需手动调整，核心要点：
- `file(GLOB_RECURSE ...)` 收集所有子目录的 `.cpp/.h`
- `opbuild(OPS_SRC ${DEF_CPP} OUT_DIR ${ASCEND_AUTOGEN_PATH})` 生成 proto 文件
- 同时构建 `cust_op_proto` 和 `cust_optiling` 两个 target

参考示例：`tasks/level2/Norm/RmsNormQuant/answer/0/op_host/CMakeLists.txt`。

---

## 8. 模板化 Tiling Key 处理

**判断条件**：源码使用 `ASCENDC_TPL_ARGS_DECL`、`ASCENDC_TPL_SEL` 等宏，或 `.GenSimplifiedKey()` 注册。

**规则**：
1. `ASCENDC_TPL_*` 宏来自 `tiling/tiling_api.h`（CANN SDK 提供），**保持不变**
2. `GenSimplifiedKey` 函数使用的 API 全部在 CANN SDK 中，**保持不变**
3. `ascendc/host_api/tiling/template_argument.h`（`ASCENDC_TPL_*` 宏定义）在 SDK 中，**保持不变**
4. `_def.cpp` 中的 `.ExtendCfgInfo(...)` 等**删除**（框架专用配置）
5. `config/` 目录（binary JSON + simplified_key.ini）**不复制**（msopgen 不需要）

---

## 9. 调试检查清单

| 错误信息 | 原因 | 修复 |
|:---|:---|:---|
| `undefined reference to TilingPrepareForOpCache` | 未 stub op_cache_tiling | 删除 include 和调用 |
| `undefined reference to QueryBank` | 未 stub runtime_kb_api | 删除 include 和调用 |
| `undefined reference to GenTiling` | 同上 | 删除 include 和调用 |
| `MatmulV3CompileInfo has no member xxx` | CompileInfo 字段被误删 | 恢复该字段 |
| `No such file: ../mat_mul_v3/xxx.h` | Include 路径未改写 | 按第5节规则改写 |
| `No such file: matmul/common/xxx.h` | 依赖文件未复制 | 按第1节复制到 common/ |
| `TilingBaseClass is ambiguous` | common/ 有重复定义 | 统一使用一份 tiling_base.h |
| `ops::CeilDiv is not a member of Ops::Base` | 缺少 math_util.h | 复制到 common/ 并改写 include |
| `op_cache_def_tiling.h not found` | 间接依赖未处理 | 删除 include |
| `LOG_CPP redefined` | log.h 重复定义 | 合并到一份 log.h |
| `cannot find -lcust_opapi` | cust_opapi 未构建或路径不对 | 见下方 cust_opapi 构建规则 |
| `No SOURCES given to target: cust_opapi` | GLOB 在源目录找不到 ACLNN 文件 | ACLNN 文件在 autogen 目录，需用 `${ASCEND_AUTOGEN_PATH}/aclnn_*.cpp` |
| `Verification failed: opapi=False` | cust_opapi 缺少 install 规则 | 添加 `install(TARGETS cust_opapi ...)` 规则 |

### cust_opapi 构建规则（复杂算子）

复杂算子的 cust_opapi 必须从 **autogen 目录**的 ACLNN 文件构建，而非从源目录。验证器要求库文件在 `op_api/lib/` 路径下，需要 install 规则。

```cmake
# 从 autogen 目录获取 ACLNN 文件（opbuild 生成）
file(GLOB aclnn_src ${ASCEND_AUTOGEN_PATH}/aclnn_*.cpp)
file(GLOB aclnn_inc ${ASCEND_AUTOGEN_PATH}/aclnn_*.h)
if(aclnn_src)
    add_library(cust_opapi SHARED ${aclnn_src})
else()
    add_library(cust_opapi SHARED ${ASCEND_AUTOGEN_PATH}/op_proto.cc)
    target_compile_definitions(cust_opapi PRIVATE ACLNN_WITH_BINARY)
endif()
target_include_directories(cust_opapi PRIVATE
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
        ${ASCEND_CANN_PACKAGE_PATH}/include
        ${ASCEND_CANN_PACKAGE_PATH}/include/aclnn
        # ... 标准路径
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)
target_link_libraries(cust_opapi PRIVATE
        intf_pub
        ascendcl
        nnopbase
)
set_target_properties(cust_opapi PROPERTIES OUTPUT_NAME cust_opapi)

# 验证器要求此 install 规则
install(TARGETS cust_opapi
        LIBRARY DESTINATION packages/vendors/${vendor_name}/op_api/lib)
install(FILES ${aclnn_inc}
        DESTINATION packages/vendors/${vendor_name}/op_api/include)
```

**常见错误**：用 `file(GLOB ... "${CMAKE_CURRENT_SOURCE_DIR}/aclnn*.cpp")` 在源目录查找 ACLNN 文件会失败，因为 ACLNN 文件是 opbuild 生成的，位于 `${ASCEND_AUTOGEN_PATH}/` 目录。

---

## 变更记录

- 2026-04-24: 重构第0节复杂度判断标准——移除"TilingParse 注册"作为复杂度判据（简单算子也可使用 TilingParse），改为"CompileInfo 含 cube 硬件参数或平台特性探测"；新增注意说明；重构第2节决策树——明确仅适用于复杂算子，简单算子转交 conversion_rules.md
- 2026-04-23: 重构为自包含指南——从 conversion_rules.md 吸收 stub 策略表（第3节）、include 改写规则表（第5节）、模板化 Tiling Key（新增第8节）；移除 0.1 节（目标结构已移至 SKILL.md）；消除所有指向 conversion_rules.md 特定节的交叉引用
- 2026-04-23: 新增 cust_opapi 构建规则和调试条目（来自 BatchMatMulV3 迁移经验）
- 2026-04-21: 初始版本，覆盖 matmul 系列复杂算子适配规则
