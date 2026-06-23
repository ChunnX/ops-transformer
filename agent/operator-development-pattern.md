# 单算子开发模式静态分析

## 典型目录职责

```text
<domain>/<op_name>/
  CMakeLists.txt
  op_host/
  op_host_aclnn/
  op_host_aclnnInner/
  op_host_aclnnExc/
  op_graph/
  op_kernel/
  examples/
  tests/
```

| 子目录 | 作用 |
| --- | --- |
| `op_host/` | 算子定义、shape 推导、tiling、注册、config。 |
| `op_host_aclnn/` | 对外 aclnn API 定义和封装。 |
| `op_host_aclnnInner/` | 内部 aclnn API 或非公开入口。 |
| `op_host_aclnnExc/` | exception 或扩展 API 相关入口。 |
| `op_graph/` | graph 模式接入、GE/IR 注册和适配。 |
| `op_kernel/` | AscendC kernel 源码，是后续性能优化的主要位置。 |
| `examples/` | eager 或 graph 调用样例，可用于冒烟和 profiling。 |
| `tests/ut/` | 新 UT 框架 case。 |
| `tests/utest/` | 老 UT 框架或特殊 case。 |
| `tests/st/` | ST case、json 配置、executor。 |

并非所有算子都有所有子目录。判断优化影响面时，要先看该算子的实际目录结构。

## `op_host` 常见内容

典型文件类型：

- `*_tiling.cpp`、`*_tiling.h`：tiling 逻辑，决定 block、tile、workspace、tiling key 等。
- `*_infer_shape.cpp`：shape 推导。
- `*_op.cpp`、`*_def.cpp`：算子定义和注册。
- `config/`：按 SoC 或场景组织的算子配置。

性能相关性：

- tiling 参数会直接影响 kernel 执行路径和性能。
- shape/dtype/format 判定可能决定是否走高性能分支。
- `op_host/config` 是 SoC 支持信号之一，但不是唯一信号。

## `op_kernel` 常见内容

典型文件类型：

- `<op>.cpp`：kernel 入口或主实现。
- `<op>.h`、`*_impl.h`：模板实现、核心计算和搬运逻辑。
- `*_tiling_data.h`：host 和 kernel 共享的 tiling data 结构。
- `*_tiling_key.h`：tiling key 定义。
- `arch22/`、`arch35/` 等：面向特定 SoC 架构的实现。
- `.apt` 或相关文件：部分算子存在自动调优或特定模板输入。

性能相关性：

- 用户当前 `ascend910b` 对应 `arch22`，后续优先查看 `op_kernel/arch22` 或 `#if` 中的 `arch22` 分支。
- 复杂 attention/GMM/MC2 算子通常有多套 kernel 或模板，性能差异可能由 tiling key 选择决定。
- 只改 kernel 但不改 tiling，可能无法触发新路径；只改 tiling 也可能导致 kernel 假设不匹配。

## CMake 组织约定

常见函数和变量：

- `add_op_to_compiled_list()`：把算子加入待编译列表。
- `op_add_subdirectory()`：发现和加入算子子目录。
- `op_add_depend_directory()`：加入算子依赖目录。
- `add_modules_sources_with_soc()`：按 SoC 选择源码。
- `add_ops_compile_options()`：添加编译选项。
- `${op_name}_depends`：声明算子依赖。

后续修改时要注意：

- 新增 kernel 源码后，需要确认 CMake 是否纳入对应 source list。
- 新增 tiling key 或配置后，需要确认 op info 和 binary compile 脚本能生成对应 kernel。
- 修改公共依赖后，要确认依赖该公共模块的所有算子都被覆盖。

## 调用路径

### eager/aclnn

典型链路：

```text
example/test_aclnn_*.cpp
  -> aclnn API
  -> op_host tiling
  -> runtime launch AI Core kernel
  -> op_kernel AscendC
```

相关目录：

- `op_host_aclnn*`
- `op_host`
- `op_kernel`
- `examples/test_aclnn_*.cpp`
- `tests/ut/framework_normal/op_api`

### graph

典型链路：

```text
example/test_geir_*.cpp
  -> GE/graph
  -> op_graph/op_host
  -> runtime launch AI Core kernel
  -> op_kernel AscendC
```

相关目录：

- `op_graph`
- `op_host`
- `op_kernel`
- `examples/test_geir_*.cpp`
- `tests/ut/framework_normal/op_graph`

## 后续看单个算子的推荐顺序

1. 看 `<domain>/<op_name>/CMakeLists.txt`，确认 build 如何纳入源码。
2. 看 `op_host/config` 和 CI list，确认目标 SoC 支持线索。
3. 看 `op_host/*tiling*`，确认 shape/dtype/format 到 tiling key 的映射。
4. 看 `op_kernel/*tiling*`，确认 kernel 读取的 tiling data。
5. 看 `op_kernel/` 主实现和 `arch22` 分支。
6. 看 `examples/` 是否可生成可 profiling 的二进制。
7. 看 `tests/ut`、`tests/utest`、`tests/st`、pytest 是否已有验证资产。
