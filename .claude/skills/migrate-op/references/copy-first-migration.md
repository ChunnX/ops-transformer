# Answer-first 迁移模式

本文用于迁移 `answer/0`。这里的“导入/本地化”是 agent 将开源仓必要的 `op_host`、`op_kernel` 和依赖文件放入 `tasks/.../answer/0`；“框架复制”专指测试框架在编译前把 `answer/0` 覆盖到 `runs/.../src`。迁移时只维护 task 目录，不手动编辑 `runs/.../src`。

目标是先形成自包含的单算子工程，再根据结构和日志做最小适配。

## 1. 定位与导入

定位源目录：

```bash
find /home/zhangqijia/repo/ops-nn /home/zhangqijia/repo/ops-cv /home/zhangqijia/repo/ops-math /home/zhangqijia/repo/ops-transformer -type d -name '<source_name>' -print
```

优先选择含 `op_host/` 和 `op_kernel/` 的目录。导入到目标时保留源内相对结构：

```text
source/op_host/*   -> answer/0/op_host/*
source/op_kernel/* -> answer/0/op_kernel/*
公共 include/.cpp  -> answer/0/op_host/common/ 或 answer/0/op_kernel/
```

源 `tests/`、`examples/`、仓库根 `cmake/` 不作为默认迁移对象。`op_api/` 可作为接口语义参考，但不导入到 `answer/0`；`aclnn_*.cpp` 和 `aclnn_*.h` 只由 `opbuild` 从 OpDef 自动生成。被 include 的公共头和 `.cpp` 按最小闭包本地化。

## 2. 先识别，不全文读

在目标 `answer/0` 运行：

```bash
find . -maxdepth 4 -type f | sort
rg -n 'gert::TilingContext|SetTiling|IMPL_OP_OPTILING|REGISTER_TILING|TilingBaseClass|TeOpParas|OpRunInfo|nlohmann|arch35|_apt|kirin|ascend950|__CCE_AICORE__|__NPU_ARCH__|extern "C"|__global__|GM_ADDR|TILING_KEY_IS|GetTilingKey'
rg -n '^#include "[./]' op_host op_kernel
```

用结果确定适配模式。只打开命中附近代码、OpDef 构造函数、TilingData 定义、kernel 入口和编译日志指向的位置。

## 3. host 侧适配

### 简单 gert

特征：`gert::TilingContext*` tiling 函数，无多策略 registry。

做法：

- 保留 tiling 函数和 TilingData 字段
- 补齐或裁剪 `OpDef`
- 用 `.SetTiling(optiling::TilingFunc)`
- 保留 `OP_ADD(OpName)`
- `AICore().AddConfig()` 保留 `ascend910b`、`ascend910_93`

### 分文件 gert

特征：已有 `*_def.cpp`、`*_tiling.cpp`、`*_infershape.cpp`。

做法：

- 保持分文件结构
- 修正 include 为本地路径或 CANN 公开头
- 只裁剪非目标芯片配置和非目标 kernel 文件
- 若多个 `.cpp` 都在 `op_host` 根目录，默认 CMake 通常够用

### 复杂 registry

特征：`TilingBaseClass`、`REGISTER_TILING_TEMPLATE`、多个 tiling 类或公共 registry。

做法：

- 本地化 registry 依赖的 `common/`、`tiling_base/` 和对应 `.cpp`
- 使用 `IMPL_OP_OPTILING` 路径时，`_def.cpp` 中不再写 `.SetTiling`
- 添加 `op_host/CMakeLists.txt`，递归收录 registry 相关 `.cpp`
- 确认注册宏、TilingData 注册名、`OP_ADD(OpName)` 使用同一个 op type

### 旧接口

特征：`TeOpParas`、`OpRunInfo`、`ge::Operator`、`nlohmann::json`。

做法：

- 改为 `gert::TilingContext*`
- 从 `context` 读取 shape、dtype、attr、workspace 和 platform
- 用 `PlatformAscendC(context->GetPlatformInfo())` 获取 UB、核数和 SoC 信息
- 用 `context->SetBlockDim()`、`SetTilingKey()`、`SaveToBuffer()`、`SetDataSize()` 写回结果

## 4. OpDef 核对

OpDef 是 `opbuild` 生成 proto 和 aclnn API 的输入。核对：

- `Input()`、`Output()` 名称和顺序与原算子语义、kernel `GM_ADDR` 顺序一致
- `ParamType(REQUIRED/OPTIONAL/DYNAMIC)` 与原定义一致
- dtype、format、unknown shape format 列表长度一致
- attr 顺序与 tiling 中 `GetAttrs()->Get...(...idx...)` 索引一致
- `OP_ADD(OpName)` 与任务目录 `OpName` 一致
- 只保留 `ascend910b` 和 `ascend910_93`

## 5. TilingData 和 ABI

TilingData 是 host 与 kernel 的 ABI。迁移时：

- 保持字段类型、顺序、数组长度和含义
- `REGISTER_TILING_DATA_CLASS(OpName, XxxTilingData)` 的 op 名与 `OP_ADD(OpName)` 一致
- kernel `GET_TILING_DATA` 后读取的字段与 host 定义一致
- kernel 使用 tiling key 时，host 设置同一 key
- 设置 raw tiling data 后写入 `SetDataSize(tiling.GetDataSize())`

## 6. kernel 侧适配

只保留 910B/910_93 可用计算路径：

- 保留 `#include "kernel_operator.h"`
- 删除 `arch35/`、`*_apt.cpp`、Kirin、310P、ascend950-only 入口
- 多架构宏中保留可用于当前目标的分支
- 被 include 的公共头本地化到当前 `op_kernel/`，并改成本地 include
- kernel 入口保持 `extern "C" __global__ __aicore__`
- `GM_ADDR` 顺序为输入、输出、workspace、tiling，和 OpDef/pybind/autogen 对齐

如果只存在非目标架构实现，停止迁移该路径并报告缺少 910B/910_93 可用源码。

## 7. CMake 添加条件

默认不添加 `answer/0/CMakeLists.txt`。

添加 `answer/0/op_host/CMakeLists.txt` 的常见条件：

- `op_host` 子目录中有必须编译的 `.cpp`
- registry/tiling_base 有独立实现文件
- 需要额外 include 目录、编译宏或链接库

添加后保持 msopgen 产物结构：仍要生成并安装 `cust_op_proto`、`cust_optiling`、`cust_opapi`，并保证 `liboptiling.so` 位于 custom OPP root 的 `op_impl/ai_core/tbe/op_tiling/` 下。`cust_opapi` 继续使用 `opbuild` 生成的 autogen `aclnn_*` 源码，不替换为本地 `op_api/` 实现。

## 8. 修改优先级

优先顺序：

1. 本地化缺失 include 和源码
2. 裁剪非目标架构
3. 修 OpDef、tiling 注册和 CMake 收录
4. 修 pybind 参数桥接
5. 根据 precision 日志修 validation 或 host/kernel 对齐问题

只在当前证据指向计算路径错误时查看 kernel 内部实现；改计算语义前先确认是否其实是 tiling、参数顺序、dtype 或布局桥接问题。
