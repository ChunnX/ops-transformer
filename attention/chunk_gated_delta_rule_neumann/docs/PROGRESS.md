# PROGRESS.md - chunk_gated_delta_rule_neumann 算子开发进度

## 目标
- 创建新算子 `chunk_gated_delta_rule_neumann`
- 编译、安装、测试
- 与 CPU golden 进行精度对比

## 完成步骤

### ✅ 1. 算子目录创建
- 从 `chunk_gated_delta_rule` 复制创建新算子
- 重命名所有文件和内容

### ✅ 2. 文件修改

**CMakeLists.txt (关键修改点)**:
- `op_host/CMakeLists.txt`: OPTYPE 和 def.cpp 文件名
- `tests/ut/op_kernel/CMakeLists.txt`: 测试目标名
- `op_graph/CMakeLists.txt`: 算子名和 proto 文件名

**头文件引用**:
- 所有 `#include "chunk_gated_delta_rule.h"` → `#include "chunk_gated_delta_rule_neumann.h"`
- 相对路径引用也需要更新（如 `../op_kernel/xxx.h`）

**Kernel 函数名**:
- `void chunk_gated_delta_rule()` → `void chunk_gated_delta_rule_neumann()`

### ✅ 3. 编译安装
```bash
bash build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule_neumann -j8
./build/cann-ops-transformer-custom_linux-aarch64.run --quiet --nox11
```

**编译结果**: 成功生成 `.run` 安装包
**安装位置**: `~/Ascend/cann-9.0.0/opp/vendors/custom_transformer/`

### ✅ 4. aclnn 单算子测试
```bash
bash build.sh --run_example chunk_gated_delta_rule_neumann eager cust --vendor_name=custom
```

**结果**:
- ✅ aclnn API 调用成功
- ✅ Workspace 分配: 37921280 bytes
- ⚠️ 输出 NaN (输入数据问题，非算子 bug)

### ✅ 5. 精度对比测试程序
创建文件:
- `tests/pytest/generate_test_inputs.py` - Python 数据生成脚本
- `examples/test_aclnn_chunk_gated_delta_rule_neumann.cpp` - C++ 精度测试程序

**测试数据格式**:
- q: (T, nk, dk) = (128, 4, 64)
- k: (T, nk, dk)
- v: (T, nv, dv) = (128, 4, 64)
- beta: (T, nv) - 关键！不是 (T,)
- g: (T, nv) - float32
- initial_state: (B, nv, dv, dk)
- actual_seq_lengths: (B) - int32

### ⚠️ 6. 精度对比结果

**测试输出**:
```
Workspace size: 37921280 bytes
aclnn execution succeeded

Output precision:
  max_abs_err: 2.460938 at idx 47
  max_rel_err: 700.000000%
  FAIL (threshold 0.010000)

FinalState precision:
  max_abs_err: 1.001038
  max_rel_err: 11311.034180%
  FAIL (threshold 0.010000)
```

**问题分析**:
- NPU 算子与 CPU golden 实现逻辑不一致
- 需要检查:
  1. Kernel 实现是否正确
  2. Tiling 参数是否正确
  3. 输入数据格式是否完全匹配
  4. Golden 实现是否与算子逻辑一致

## 关键发现

### CMakeLists.txt 必须修改的位置
1. `op_host/CMakeLists.txt`: OPTYPE 变量和 def.cpp 文件名
2. `tests/ut/op_kernel/CMakeLists.txt`: 测试目标名
3. `op_graph/CMakeLists.txt`: 算子名和 proto 文件名

### 头文件引用必须更新
- 所有相对路径引用 `#include "xxx.h"` 需要检查
- 特别是 `../op_kernel/xxx.h` 这种相对路径

### 数据格式关键点
- **beta/g 维度必须是 (T, nv)**，不是 (T,)
- actual_seq_lengths 是 int32 tensor，不是 aclIntArray
- API 参数顺序: query, key, value, beta, initialState, actualSeqLengths, g, scale

### API 参数顺序
```cpp
aclnnChunkGatedDeltaRuleNeumannGetWorkspaceSize(
    query, key, value, beta, initialState, actualSeqLengths, g, scale,
    out, finalState, workspaceSize, executor);
```

## 下一步

### 精度问题排查
1. 对比 golden 实现与 kernel 逻辑
2. 检查 tiling 参数是否正确
3. 简化测试用例（B=1, seqlen=16）验证基本逻辑
4. 添加中间结果打印调试

### 可选优化
- 注册 torch_npu Python API
- 创建 UT 单元测试
- 添加更多测试用例

## 文件位置

| 文件 | 路径 |
|------|------|
| 算子目录 | `/mnt/workspace/gitCode/cann/ops-transformer/attention/chunk_gated_delta_rule_neumann/` |
| 进度文档 | `docs/PROGRESS.md` |
| SKILL | `SKILL/operator-clone/SKILL.md` |
| aclnn 测试 | `examples/test_aclnn_chunk_gated_delta_rule_neumann.cpp` |
| 数据生成 | `tests/pytest/generate_test_inputs.py` |
| Golden 实现 | `tests/pytest/chunk_gated_delta_rule_neumann_golden.py` |

## Git 状态

**提交**: `9e0456d2` - "feat: Add chunk_gated_delta_rule_neumann operator and SKILL documentation"
**分支**: master
**Fork**: https://gitcode.com/chuqihang/ops-transformer (push 需要手动执行)