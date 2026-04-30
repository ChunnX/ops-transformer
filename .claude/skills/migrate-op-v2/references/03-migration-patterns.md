# 迁移模式详解

本文覆盖五种迁移模式，每种包含：识别方法、处理步骤、代码示例（改前/改后）、检查清单。

迁移前必须**先确定模式**。处理步骤取决于模式，不分模式就动手会导致大量返工。

## 模式判断决策树

在源 `op_host/` 目录中执行以下检测，**按顺序判断，第一个匹配即为当前模式**：

```
1. 先做架构可用性检查（与模式分类独立）：
   # 源算子是否只有 arch35/ 和 *_apt.cpp？
   find op_kernel -name "*.cpp" ! -path "*/arch35/*" ! -name "*_apt.cpp"  # 找非 arch35 且非 _apt 的 kernel
   
   如果 op_kernel 中所有 kernel 实现都在 arch35/ 下，或只有 *_apt.cpp，且没有 membase 目录 → 模式 5

2. rg "TilingBaseClass|REGISTER_TILING_TEMPLATE" op_host → 有匹配？
   └── 是 → 模式 3：复杂 registry

3. rg "TeOpParas|OpRunInfo|nlohmann" op_host → 有匹配？
   └── 是 → 模式 4：旧接口

4. rg "IMPL_OP_OPTILING" op_host → 有匹配？
   └── 是 → 模式 2b：IMPL_OP_OPTILING 分文件

5. 检查文件结构：
   - op_host 中只有一个 .cpp 同时包含 TilingFunc + OpDef → 模式 1：简单 gert
   - op_host 有 *_def.cpp + *_tiling.cpp（分文件）→ 模式 2a：分文件 gert
```

**确定模式后再进入阶段 4 的执行。**

---

## 模式 1：简单 gert

### 识别特征

- Tiling 函数签名使用 `gert::TilingContext*`
- OpDef 中使用 `.SetTiling(optiling::TilingFunc)`
- 无 `IMPL_OP_OPTILING`、无 `TilingBaseClass`、无 `REGISTER_TILING_TEMPLATE`
- tiling 函数和 OpDef 在同一个 `.cpp` 文件中

### 典型源仓结构

```
source/operator_name/
├── op_host/
│   ├── operator_name.cpp      # TilingFunc + OpDef
│   └── operator_name_tiling.h # TilingData 定义
└── op_kernel/
    └── operator_name.cpp      # kernel 实现
```

### 处理步骤

**第1步：导入文件**

```bash
cp source/op_host/operator_name_tiling.h answer/0/op_host/
cp source/op_host/operator_name.cpp       answer/0/op_host/
cp source/op_kernel/*                     answer/0/op_kernel/

# 删除禁止导入的内容
rm -rf answer/0/op_host/op_api/
rm -rf answer/0/op_kernel/arch35/
rm -f  answer/0/op_kernel/*_apt.cpp
```

**第2步：清理 op_host/**

1. 修正 include 路径：
   ```cpp
   // 改前
   #include "../../common/inc/log/ops_log.h"

   // 改后
   #include "register/op_def_registry.h"
   #include "tiling/platform/platform_ascendc.h"
   // 如需日志宏，复制本地 common/log.h（从已迁移的同类算子复制）
   ```

2. 裁剪 OpDef 芯片配置（只保留目标平台）：
   ```cpp
   // 改前
   this->AICore().AddConfig("ascend910b")
                 .AddConfig("ascend910_93")
                 .AddConfig("ascend950")      // 删除
                 .AddConfig("ascend910_95")   // 删除
                 .AddConfig("kirinx90");      // 删除

   // 改后
   this->AICore().AddConfig("ascend910b");
   this->AICore().AddConfig("ascend910_93");
   ```

3. 确认 `.SetTiling(optiling::TilingFunc)` 存在且函数名正确。

4. 确认 `OP_ADD(OpName)` 参数是 PascalCase 的 `OpName`。

**第3步：清理 op_kernel/**

1. 删除非目标架构代码：`arch35/` 目录、`*_apt.cpp`（910_95 入口）、Kirin/310P/ascend950 专用代码
2. 修正 include 路径：外部 include 改为本地或 CANN 标准头
3. 确认 kernel 入口签名 `GM_ADDR` 顺序与 OpDef 一致
4. 确认 `GET_TILING_DATA` 参数与 `_tiling.h` 结构体名一致

**第4步：CMake**

模式 1 默认**不需要**自定义 CMake。msopgen 默认 `op_host/CMakeLists.txt` 已通过 `aux_source_directory` 收集当前目录 `.cpp`。

### 示例：Gelu

**迁移后**（`tasks/level2/Activation/Gelu/answer/0/`）：
```
answer/0/                    # 无 CMakeLists.txt
├── op_host/
│   ├── gelu.cpp             # TilingFunc + OpDef（芯片配置已裁剪）
│   └── gelu_tiling.h        # 不变
└── op_kernel/
    └── gelu.cpp             # kernel（非目标架构代码已删除）
```

### 模式 1 检查清单

- [ ] include 路径已修正（不依赖源仓根目录）
- [ ] OpDef 芯片配置只保留 `ascend910b` 和 `ascend910_93`
- [ ] `.SetTiling(optiling::TilingFunc)` 函数名正确
- [ ] `OP_ADD(OpName)` 是 PascalCase
- [ ] `op_api/` 未导入
- [ ] `arch35/`、`*_apt.cpp` 已删除
- [ ] kernel `GM_ADDR` 顺序与 OpDef 一致
- [ ] `GET_TILING_DATA` 参数与 `_tiling.h` 结构体一致
- [ ] 无需自定义 CMake

---

## 模式 2a：分文件 gert（`.SetTiling()`）

### 识别特征

- 文件分为 `*_def.cpp` + `*_tiling.cpp` + 可选的 `*_infershape.cpp`
- OpDef 中用 `.SetTiling()` 注册
- 无 `IMPL_OP_OPTILING`、无 `TilingBaseClass`

### 处理步骤

**第1步：导入文件**

```bash
cp source/op_host/*_tiling.h     answer/0/op_host/
cp source/op_host/*_def.cpp       answer/0/op_host/
cp source/op_host/*_tiling.cpp    answer/0/op_host/
cp source/op_host/*_infershape.cpp answer/0/op_host/ 2>/dev/null
cp source/op_kernel/*             answer/0/op_kernel/

# 删除禁止的
rm -rf answer/0/op_host/op_api/
rm -rf answer/0/op_kernel/arch35/
rm -f  answer/0/op_kernel/*_apt.cpp
```

**第2步：修正分文件间 include**

`_def.cpp`、`_tiling.cpp`、`_infershape.cpp` 之间的互引用从外部路径改为本地：

```cpp
// 改前
#include "ops/operator_name/operator_name_tiling.h"
#include "ops/operator_name/operator_name_def.h"

// 改后（同目录下）
#include "operator_name_tiling.h"
#include "operator_name_def.h"
```

**第3步：清理各文件**

- `_def.cpp`：裁剪芯片配置至 `ascend910b` + `ascend910_93`，确认 `.SetTiling()` 函数名正确
- `_tiling.cpp`：修正外部 include；确认 `TilingFunc` 使用 `gert::TilingContext*`
- `_tiling.h`：TilingData 结构不动
- `_infershape.cpp`（如存在）：保留并修正 include；如输出 shape 简单等于输入可省略
- `_def.cpp` 中 `#include "register/op_def_registry.h"`（模式 2a 用 `.SetTiling()` 注册，不需要 `register/op_impl_registry.h`）

**第4步：清理 op_kernel/**

同模式 1。

**第5步：CMake**

通常不需要自定义。分文件的 `.cpp` 都会被默认 `aux_source_directory` 收集。如果在 `op_host/` 下有子目录包含 `.cpp`，需要自定义 CMake（见 `04-cmake-guide.md`）。

### 模式 2a 检查清单

- 所有 include 已修正（分文件间和外部）
- `.SetTiling()` 指向正确的 tiling 函数
- `_def.cpp`、`_tiling.cpp` 都被 CMake 收集
- 其余同模式 1

---

## 模式 2b：IMPL_OP_OPTILING 分文件

### 识别特征

- 有 `IMPL_OP_OPTILING` 宏
- **但无** `TilingBaseClass` 和 `REGISTER_TILING_TEMPLATE`（无多策略 registry）
- 文件分为 `*_def.cpp` + `*_tiling.cpp` + 可选的 `*_infershape.cpp`

这个模式介于模式 2a 和模式 3 之间。典型例子：ScatterElementsV2（`IMPL_OP_OPTILING` 但无多策略 registry）。也可能包含 `TilingParse`——如 GeluMul、SwiGlu 等用 `IMPL_OP_OPTILING` + `TilingParse<CompileInfo>` 但无 `TilingBaseClass` 的算子。

### 关键区别

与模式 2a 相比：
- `_def.cpp` 中**没有** `.SetTiling()`
- `_tiling.cpp` 中使用 `IMPL_OP_OPTILING(OpName).Tiling(TilingFunc)` 注册

与模式 3 相比：
- 没有 `TilingBaseClass` 子类化
- 没有 `REGISTER_OPS_TILING_TEMPLATE` 多策略注册
- 可能有 `TilingParse<CompileInfo>`（如需要编译期平台信息），也可能没有

### 处理步骤

**第1步：导入文件（同模式 1）**

**第2步：清理 `_def.cpp`**

```cpp
namespace ops {
class MyOp : public OpDef {
public:
    explicit MyOp(const char* name) : OpDef(name)
    {
        this->Input("x")...;
        this->Output("y")...;

        // 注意：不写 .SetTiling() — tiling 由 _tiling.cpp 中 IMPL_OP_OPTILING 注册
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};
OP_ADD(MyOp);
}
```

**第3步：清理 `_tiling.cpp`**

简单变体（无 CompileInfo / TilingParse）：
```cpp
#include "my_op_tiling.h"
#include "register/op_impl_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    // ... tiling 实现 ...
}

IMPL_OP_OPTILING(MyOp)
    .Tiling(TilingFunc);
}
```

带 `TilingParse` 的变体（如 GeluMul、SwiGlu 等，源仓使用 CompileInfo 传递平台信息，但无多策略 registry）：

> **警告：aclnn 直调路径下 TilingParse 不会被调用！** NPUKernelBench 通过 pybind → aclnn 调用算子，走的是 aclnn 直调路径。在此路径下，`.TilingParse<CompileInfo>(TilingPrepare)` 注册了也不会被触发，`context->GetCompileInfo<T>()` 始终返回 nullptr。
>
> 迁移时必须：删除 `TilingPrepare` 函数、删除 `CompileInfo` 结构体、删除 `.TilingParse<CompileInfo>(...)` 注册，在 `TilingFunc` 中直接用 `PlatformAscendC(context->GetPlatformInfo())` 获取平台信息。详见 `references/02-tiling.md` 第 8 节。

```cpp
// 迁移后：删除 TilingParse 和 CompileInfo，直接用 PlatformAscendC
#include "my_op_tiling.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint32_t aivCoreNum = platform.GetCoreNumAiv();
    // ... 使用 ubSize 和 aivCoreNum 计算 tiling ...
}

IMPL_OP_OPTILING(MyOp)
    .Tiling(TilingFunc);   // 注意：已删除 .TilingParse<...>(...)
}
```

**第4步：CMake**

通常不需要自定义。如果 `op_host/` 下有子目录，需要自定义 CMake。

### 模式 2b 检查清单

- [ ] `_def.cpp` 中**没有** `.SetTiling()`
- [ ] `_tiling.cpp` 中 `IMPL_OP_OPTILING(OpName)` 的 OpName 与 `OP_ADD` 一致
- [ ] `TilingFunc` 签名正确（`gert::TilingContext*`，返回 `ge::graphStatus`）
- [ ] 已删除 `TilingParse` 和 `CompileInfo`，改用 `PlatformAscendC(context->GetPlatformInfo())` 直接获取平台信息
- [ ] 其余同模式 1

---

## 模式 3：复杂 registry

### 识别特征

- `TilingBaseClass`
- `REGISTER_TILING_TEMPLATE` 或 `REGISTER_OPS_TILING_TEMPLATE`
- `IMPL_OP_OPTILING` + `.TilingParse<CompileInfo>(...)
- 多个 tiling 策略类

> **警告：aclnn 直调路径下 TilingParse 不会被调用！** 迁移时必须删除 `TilingParse` 函数、`CompileInfo` 结构体，以及 `.TilingParse<CompileInfo>(...)` 注册。各个 tiling 策略类中通过 `GetCompileInfo()` 获取平台信息的代码，改为直接用 `PlatformAscendC(context->GetPlatformInfo())` 获取。详见 `references/02-tiling.md` 第 8 节。

### 处理步骤

复杂 registry 需要本地化公共 tiling 基础设施。

**第1步：导入文件**

```bash
cp -r source/op_host/*     answer/0/op_host/
cp -r source/op_kernel/*   answer/0/op_kernel/

# 删除禁止的
rm -rf answer/0/op_host/op_api/
rm -rf answer/0/op_kernel/arch35/
rm -f  answer/0/op_kernel/*_apt.cpp
```

**第2步：本地化公共依赖**

识别所需的外部 include：
```bash
rg '^#include "[.][.]/' op_host
```

常见需要本地化的：
- `tiling/tiling_base.h` → 复制到 `op_host/tiling_base/`
- `tiling/tiling_templates_registry.h` → 复制到 `op_host/tiling_base/`
- `log/ops_log.h` / `error/ops_error.h` → 复制到 `op_host/common/log.h`

**推荐的本地化方法**：从本仓已通过的同类算子复制 `tiling_base/` 和 `common/`：

```bash
# 找已迁移的模式 3 算子
rg -l "IMPL_OP_OPTILING.*TilingParse" tasks/ --include="*.cpp" | head -5
# 复制其 tiling_base 和 common 目录
```

然后修改 tiling 代码中的命名空间以匹配复制来的 base。

**第3步：修改 `_def.cpp`**

只定义 OpDef，不写 `.SetTiling()`。只保留 `ascend910b` 和 `ascend910_93`。

**第4步：修改 `_tiling.cpp`**

删除 `TilingParse` 函数和 `.TilingParse<...>(...)` 注册。各 tiling 策略类中涉及 `GetCompileInfo()` 的代码改为 `PlatformAscendC(context->GetPlatformInfo())` 直接获取。

```cpp
#include "my_op_tiling.h"
#include "strategy_a_tiling.h"
#include "strategy_b_tiling.h"
#include "common/log.h"
#include "tiling_base/tiling_templates_registry.h"

namespace optiling {
ge::graphStatus Tiling4MyOp(gert::TilingContext* context)
{
    return OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

// TilingParse 已删除 — aclnn 路径下不会被调用

REGISTER_OPS_TILING_TEMPLATE(MyOp, StrategyATiling, 50000);

IMPL_OP_OPTILING(MyOp)
    .Tiling(Tiling4MyOp);
    // 注意：已删除 .TilingParse<MyOpCompileInfo>(TilingPrepareForMyOp)
}
```

**第5步：编写自定义 op_host/CMakeLists.txt**

因为 `op_host/` 下有子目录（`common/`、`tiling_base/`），默认 CMake 不会收集它们。参见 `references/04-cmake-guide.md` 第3节获取完整自定义 CMake。

**第6步：处理 kernel 侧**

删除 `arch35/` 和 `*_apt.cpp`。确认保留的 kernel 路径的 tiling key 能被 host 设置。

### 模式 3 检查清单

- [ ] 公共依赖已本地化（`common/`、`tiling_base/`）
- [ ] `_def.cpp` 中**没有** `.SetTiling()`
- [ ] `_tiling.cpp` 中命名空间与本地化的 tiling_base 一致
- [ ] `REGISTER_OPS_TILING_TEMPLATE` 的 OpName 与 `OP_ADD` 一致
- [ ] 已删除 `TilingParse` 函数和 `.TilingParse<CompileInfo>(...)` 注册（aclnn 路径不会调用）
- [ ] 各 tiling 策略类中的 `GetCompileInfo()` 已改为 `PlatformAscendC(context->GetPlatformInfo())` 直接获取
- [ ] `answer/0/op_host/CMakeLists.txt` 正确收集子目录中的 `.cpp`
- [ ] 非目标芯片的 tiling 策略已删除或不可达
- [ ] `op_api/`、`arch35/`、`*_apt.cpp` 已删除
- [ ] 打包产物包含 `liboptiling.so` 且无未解析符号

---

## 模式 4：旧接口

### 识别特征

- `TeOpParas`、`OpRunInfo`、`nlohmann::json`
- tiling 函数返回 `bool`，参数为 `(const TeOpParas&, OpRunInfo&)`

### 处理步骤

旧接口是完全的重写。必须用 gert API 重新实现整个 tiling 函数。

**第1步：保留 TilingData 定义**

`_tiling.h` 不变。

**第2步：重写 tiling 函数**

使用 `references/02-tiling.md` 第 5 节的对照表逐行替换。完成后 tiling 函数签名必须为：
```cpp
static ge::graphStatus TilingFunc(gert::TilingContext* context)
```

**第3步：重新实现 OpDef**

旧接口算子通常没有 `_def.cpp`。需要根据 README、proto 或 op_api 推导 Input/Output/Attr。

**第4步：InferShape**

如果输出 shape 简单等于输入，可省略 InferShape。否则手写 `*_infershape.cpp`。

### 模式 4 检查清单

- [ ] 所有 `TeOpParas` / `OpRunInfo` / `nlohmann::json` 已替换
- [ ] 平台信息通过 `PlatformAscendC` 获取
- [ ] TilingData 结构不变
- [ ] 新 OpDef 完整定义了所有 Input/Output/Attr
- [ ] 返回值改为 `ge::GRAPH_SUCCESS` / `ge::GRAPH_FAILED`

---

## 模式 5：阻塞 — 只有 arch35/apt 实现

### 识别特征

- 源算子的 `op_kernel/` 中所有 kernel 实现都在 `arch35/` 目录下
- 或只有 `*_apt.cpp` 入口
- 没有 membase 目录（910B/910_93 实现）

### 处理方式

**不要强行迁移。** arch35 代码是为 regbase 架构编写的，无法在 910B/910_93 上运行。

**下一步：**
1. 在其他三个源仓中搜索同名的算子目录（可能有 910B 版本）
2. 搜索同仓中的 membase 子目录
3. 如果四个源仓都没有 910B 实现 → 报告阻塞，停止迁移
