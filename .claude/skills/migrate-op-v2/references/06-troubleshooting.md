# 故障诊断方法论

当编译或精度验证失败时，不要立即搜索"与错误消息匹配的已知问题"。先按以下系统化流程缩小问题范围，确定根因所在的层次。

## 1. 最重要的规则

**永远不要通过修改 kernel 的数学计算逻辑来"修复"精度或运行时错误。** 问题根因几乎总是在 tiling（参数计算或 ABI 不匹配）、pybind 桥接（类型不匹配或参数顺序错误）、include 路径、或 CMake 选源中——而不是在 kernel 的数学计算中。修改计算代码来"绕过"问题会使数据集的正确性验证失效。如果 kernel 数学逻辑在源仓是正确的，迁移后也应该是正确的。

## 2. 调试的第一原则：分层定位

迁移问题通常分布在四个层次之一。按从高到低的顺序排查，每一层确认正确后再看下一层：

```
层次 1：环境 / 路径（CANN 未安装、路径错误、环境变量缺失）
   ↓ 确认环境正确
层次 2：工程 / 编译（CMake 选源错误、include 缺失、链接错误、autogen 失败）
   ↓ 确认编译通过
层次 3：运行时 / 加载（so 加载失败、符号冲突、tiling 未命中、同名内置算子冲突）
   ↓ 确认 kernel 能运行
层次 4：语义 / 精度（GM_ADDR 顺序、TilingData ABI 不匹配、dtype 分支、计算逻辑）
```

**核心原则：永远先在当前层次确认问题，不要跳到下一层。** 例如，编译报 "undefined reference"，不要在 kernel 代码中找问题——这是层次 2（链接错误）。

## 3. 按失败阶段切入

### 3.1 编译阶段失败

编译失败时，阅读错误输出的**最后 30 行**（不是最前面）。编译器的第一条错误可能引发级联错误，根因在底部。

**第一步：确认 CANN 环境**

```bash
echo $ASCEND_HOME_PATH
echo $ASCEND_CANN_PACKAGE_PATH
ls $ASCEND_CANN_PACKAGE_PATH/include/register/op_def_registry.h
```

如果 `ASCEND_CANN_PACKAGE_PATH` 为空或头文件不存在 → **层次 1 问题**。设置环境变量后重试。

**第二步：分类错误消息**

| 错误类型 | 关键字 | 所属层次 | 排查方向 |
|---------|--------|---------|---------|
| 头文件未找到 | `fatal error: ... No such file or directory` | 编译 | include 路径不对；本地化缺失的头文件 |
| 宏/符号未定义 | `was not declared / has not been declared` | 编译 | 缺少 include；删除或替换不可用的宏 |
| 链接错误 | `undefined reference to` | 链接 | CMake 未收录源文件；命名空间不匹配；缺少库链接 |
| autogen 失败 | `aclnn_*.h not found` | 编译 | OpDef 定义有误，`opbuild` 无法生成 API |

**第三步：逐错误修复**

- **头文件未找到**：确定该头文件的来源（CANN 标准头？开源仓公共头？其他算子？）。CANN 标准头通过 `ASCEND_CANN_PACKAGE_PATH/include` 提供；开源仓公共头需要本地化到 `answer/0/op_host/` 或 `answer/0/op_kernel/`
- **符号未定义**：确定符号的来源。如果是日志宏（`OPS_ERR_IF`、`OP_TILING_CHECK`），本地化 `common/log.h`。如果是 tiling API，检查 CANN 版本和 include 路径
- **链接错误**：确认对应的 `.cpp` 文件是否被 CMake 收集。默认 `aux_source_directory` 只收集当前目录——子目录中的 `.cpp` 需要自定义 CMake

### 3.2 编译通过但运行时失败

**第一步：确认产物完整性**

```bash
# 必需的产物
find runs -path "*<OpName>*" -name "kernel_gen_ops*.so"    # Python 扩展
find runs -path "*<OpName>*" -name "libcust_opapi.so"      # aclnn API
find runs -path "*<OpName>*" -name "liboptiling.so"         # tiling 库
```

缺任何一样 → **层次 2 问题**（编译不完整，检查 CMake 打包规则）。

**第二步：确认产物可用性**

```bash
# 检查符号
nm -D runs/.../build/kernel_gen_ops*.so | grep <kernel_name>
# 检查 tiling 库的链接依赖
ldd runs/.../liboptiling.so | grep "not found"
```

如果 `liboptiling.so` 有未解析符号 → **层次 2 问题**（链接不完整）。

如果 `kernel_gen_ops*.so` 中没有 <kernel_name> 符号 → **层次 2 问题**（pybind 编译有误，检查 pybind.cpp 和 validation/CMakeLists.txt）。

**第三步：检查运行时错误类型**

| 错误 | 可能层次 | 排查方向 |
|------|---------|---------|
| `_check_not_null` / `Check out != nullptr` | 运行时 | EXEC_NPU_CMD 参数顺序错误，输出张量必须放最后 |
| `std::bad_alloc` / `MemoryError` | 运行时 | ListFloat 属性未拆分（pybind 桥接无 `vector<float>` 转换） |
| `compileInfoPtr is nullptr` | 运行时 | 代码使用了 `GetCompileInfo<T>()` 但 aclnn 直调路径不走 `TilingParse`。改为用 `PlatformAscendC(context->GetPlatformInfo())` 直接获取平台信息 |
| `do tiling failed, ret is -1` | 运行时 | tiling 执行失败。常见原因：`GetCompileInfo()` 返回 nullptr 导致未初始化的数据被使用、`OP_LOGE_IF` 等宏被错误重定义导致真实错误被吞掉 |
| 所有输出为 0 | 语义 | TilingData ABI 不匹配；tiling key 未命中；blockDim=0 |
| 随机大误差 / NaN | 语义 | GM_ADDR 顺序错误；workspace 大小不够；dtype 分支不完整 |
| 崩溃 / SIGSEGV | 运行时 | workspace 指针未判空；TilingData 超缓冲区；UB 溢出 |

### 3.3 精度失败

当某条测试用例精度不通过时，按以下步骤缩小范围：

**第一步：失败的范围有多大？**

```bash
# 在精度输出中观察通过/失败的模式
python scripts/run_test.py -task_name <OpName> -stages precision -n_case 5 2>&1 | grep -E "PASS|FAIL|abs|rel"
```

- **全部 case 失败**：偏向 层次 3（kernel 调用正确但计算结果全错）或 层次 4（golden 语义与 kernel 不一致）
- **部分 case 失败**：检查失败 case 的共性——相同 dtype？相同 shape 模式？触发了某个 tiling key？
- **单条 case 失败**：边界条件问题

**第二步：隔离到最简单的失败 case**

只测试最简单的失败 case：

```bash
# 如果 case 0 失败，只跑 case 0
python scripts/run_test.py -task_name <OpName> -stages precision -n_sample 1 -n_case 1
```

在 `test_cases.csv` 中临时只保留失败的那个 case 和一条通过的 case 做对比。

**第三步：加入 tiling dump 对比 host/kernel 参数**

```bash
python scripts/run_test.py -task_name <OpName> -stages compile precision -tiling_dump
```

比较 tiling 日志中 host 计算的参数与 kernel 实际使用的参数是否一致。

**第四步：在 golden 中增加诊断代码**

在 `module.py` 的 `Model.forward()` 和 `ModelNew.forward()` 中临时添加：
```python
print(f"input shape: {x.shape}, dtype: {x.dtype}")
print(f"output min/max: {y.min().item():.6f} / {y.max().item():.6f}")
```

对比两个 forward 的中间值，找到分歧点。

## 4. 诊断命令工具箱

```bash
# === 环境检查 ===
echo "CANN home: $ASCEND_HOME_PATH"
echo "CANN package: $ASCEND_CANN_PACKAGE_PATH"
ls $ASCEND_CANN_PACKAGE_PATH/include/register/

# === 源文件检查 ===
# 确认某个模式的所有 include 文件都存在
rg '^#include "(\.\./|[^/])' answer/0/op_host answer/0/op_kernel

# 确认没有残留的非目标架构代码
rg -n 'arch35|_apt|kirin|310P|ascend950' answer/0/

# === 编译产物检查 ===
# 查看 autogen 是否生成
find runs -path "*<OpName>*" -name "aclnn_*.h" | head -3

# 查看编译日志的最后错误
find runs -path "*<OpName>*" -name "*.log" -exec tail -80 {} \; | grep -i error

# 确认 tiling 库的符号完整性
find runs -path "*<OpName>*" -name "liboptiling.so" -exec nm -D {} \; | grep -i "tiling\|undefined"

# === 精度检查 ===
# 手动验证 golden 正确性
python -c "
import torch
import sys; sys.path.append('tasks/level{N}/{Category}/{OpName}/validation')
from module import Model
m = Model()
x = torch.randn(2, 4)
print(m(x))
"
```

## 5. 当系统化排查仍然找不到问题

1. **寻找同 category 已迁移算子对比**：
   ```bash
   find tasks/level{N}/{Category}/ -name "answer" -path "*/0/*" | head -5
   ```
   阅读同 category 算子的 `_def.cpp`、`_tiling.cpp` 和 `op_host/CMakeLists.txt`，对比结构差异。

2. **逐步回退到最简形态**：
   - 将 `test_cases.csv` 缩减为 1 条最简单的 case
   - 在 tiling 函数中硬编码参数（跳过 `context->GetInputShape` 计算），验证 kernel 是否正确
   - 如果硬编码 tiling 参数后精度通过 → 问题在 tiling 计算逻辑
   - 如果硬编码后仍然失败 → 问题在 kernel 或 pybind 桥接

3. **隔离框架与计算**：如果以上两步仍无法定位，用最简操作（如 element-wise copy）临时替换 kernel 主体来验证编译/加载/调用链路是否正常。若替换后能运行，问题在 kernel 计算侧（极其罕见）；若仍失败，问题在 tiling/ABI/桥接层。
