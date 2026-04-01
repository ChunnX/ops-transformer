## MC2算子重构：使用自动生成的头文件替代内联函数声明

### 背景说明

在编译MC2算子时，编译框架会自动生成对应的aclnn接口定义文件，存放在`build/autogen/inner/`目录下。例如：
- `build/autogen/inner/aclnnInner_all_gather_matmul.h`
- `build/autogen/inner/aclnnInner_all_gather_matmul_v2.h`

然而，在现有的MC2算子代码中（如`mc2/all_gather_matmul/op_api/aclnn_all_gather_matmul.cpp`），仍然存在内联的外部函数声明，这导致了代码冗余和维护困难。

### 改动目标

将MC2算子中内联的外部函数声明移除，改为包含编译框架自动生成的头文件，以保持代码的一致性和可维护性。

### 具体改动点（以all_gather_matmul为例）

#### 1. 添加头文件包含
```cpp
// 原代码没有包含自动生成的头文件
// 新增以下行：
#include "aclnnInner_all_gather_matmul.h"
```

#### 2. 移除内联的外部函数声明
```cpp
// 删除以下内联声明：
extern aclnnStatus aclnnInnerAllGatherMatmulGetWorkspaceSize(
    const aclTensor *x1, const aclTensor *x2, const aclTensor *bias,
    const char *group, bool transposeX1, bool transposeX2,
    int64_t gatherIndex, int64_t commTurn, int64_t rankSize,
    bool isGatherOut, const aclTensor *output, const aclTensor *gatherOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

extern aclnnStatus aclnnInnerAllGatherMatmul(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream);
```

#### 3. 适配函数调用参数类型
```cpp
// 原调用（group参数为const char*）：
aclnnStatus ret = aclnnInnerAllGatherMatmulGetWorkspaceSize(
    x1, x2, bias, group, transposeX1, transposeX2, ...);

// 新调用（适配自动生成的头文件中group为char*类型）：
aclnnStatus ret = aclnnInnerAllGatherMatmulGetWorkspaceSize(
    x1, x2, bias, const_cast<char*>(group), transposeX1, transposeX2, ...);
```

### 实施步骤

1. **检查自动生成的头文件**：确认`build/autogen/inner/`目录下存在对应的头文件
2. **添加头文件包含**：在算子源文件中添加`#include "aclnnInner_xxxx.h"`
3. **移除内联声明**：删除源文件中重复的外部函数声明
4. **适配参数类型**：检查函数签名差异，必要时使用`const_cast`进行类型转换
5. **验证编译**：确保修改后代码能正常编译通过

### 注意事项

1. **参数类型差异**：自动生成的头文件中`group`参数类型为`char *`（非const），而原始声明可能为`const char *`，需要使用`const_cast<char*>(group)`进行转换
2. **头文件命名规则**：头文件名称与算子名称对应，格式为`aclnnInner_{算子名}.h`
3. **编译依赖**：确保在编译时已经生成了对应的头文件
4. **作用范围**：此改动需要应用到`mc2/`目录下的所有算子
5. **最小改动原则**：不要试图优化代码格式，不要做格式化，代码改动保持最小

### 参考示例

已完成的改动：
- `mc2/all_gather_matmul/op_api/aclnn_all_gather_matmul.cpp`
- `mc2/all_gather_matmul_v2/op_api/aclnn_all_gather_matmul_v2.cpp`

### all_gather_matmul_v2算子改动详解

#### 特殊情况：多处函数调用适配

all_gather_matmul_v2算子在CCU模式和AIV模式两处调用`aclnnInnerAllGatherMatmulV2GetWorkspaceSize`，都需要适配参数类型。

**CCU模式调用（第462-466行）：**
```cpp
// 原调用：
aclnnStatus ret = aclnnInnerAllGatherMatmulV2GetWorkspaceSize(
    x1, transX2, bias, x1Scale, transX2Scale, quantScale, group,
    transposeX1, transposeX2, gatherIndex, commTurn,
    rankSize, blockSize, groupSize, isGatherOut, isAMaxOut,
    outDtype, commMode, output, gatherOut, amaxOut, workspaceSize,
    executor);

// 新调用（group和commMode需要const_cast）：
aclnnStatus ret = aclnnInnerAllGatherMatmulV2GetWorkspaceSize(
    x1, transX2, bias, x1Scale, transX2Scale, quantScale, const_cast<char*>(group),
    transposeX1, transposeX2, gatherIndex, commTurn,
    rankSize, blockSize, groupSize, isGatherOut, isAMaxOut,
    outDtype, const_cast<char*>(commMode), output, gatherOut, amaxOut, workspaceSize,
    executor);
```

**AIV模式调用（第499-503行）：**
```cpp
// 原调用：
aclnnStatus ret = aclnnInnerAllGatherMatmulV2GetWorkspaceSize(
    x1, x2, bias, x1Scale, x2Scale, quantScale, group,
    transposeX1, transposeX2, gatherIndex, commTurn,
    rankSize, blockSize, groupSize, isGatherOut, isAmaxOut,
    yDtype, commMode, output, gatherOut, amaxOut, workspaceSize,
    executor);

// 新调用（group和commMode需要const_cast）：
aclnnStatus ret = aclnnInnerAllGatherMatmulV2GetWorkspaceSize(
    x1, x2, bias, x1Scale, x2Scale, quantScale, const_cast<char*>(group),
    transposeX1, transposeX2, gatherIndex, commTurn,
    rankSize, blockSize, groupSize, isGatherOut, isAmaxOut,
    yDtype, const_cast<char*>(commMode), output, gatherOut, amaxOut, workspaceSize,
    executor);
```

### 所有算子改动总结

#### 已完成修改的算子列表（共22个算子，38个cpp文件）

| 算子名称 | 修改文件数 | 对应头文件 |
|---------|-----------|-----------|
| matmul_all_reduce | 8 | aclnnInner_matmul_all_reduce.h |
| matmul_all_reduce_add_rms_norm | 3 | aclnnInner_matmul_all_reduce_add_rms_norm.h |
| moe_distribute_combine | 1 | aclnnInner_moe_distribute_combine.h |
| moe_distribute_combine_v2 | 4 | aclnnInner_moe_distribute_combine_v2.h |
| moe_distribute_combine_v3 | 1 | aclnnInner_moe_distribute_combine_v3.h |
| moe_distribute_combine_add_rms_norm | 1 | aclnnInner_moe_distribute_combine_add_rms_norm.h |
| moe_distribute_dispatch | 1 | aclnnInner_moe_distribute_dispatch.h |
| moe_distribute_dispatch_v2 | 4 | aclnnInner_moe_distribute_dispatch_v2.h |
| moe_distribute_dispatch_v3 | 1 | aclnnInner_moe_distribute_dispatch_v3.h |
| moe_distribute_dispatch_setup | 1 | aclnnInner_moe_distribute_dispatch_setup.h |
| moe_distribute_dispatch_teardown | 1 | aclnnInner_moe_distribute_dispatch_teardown.h |
| moe_update_expert | 1 | aclnnInner_moe_update_expert.h |
| matmul_reduce_scatter | 1 | aclnnInner_matmul_reduce_scatter.h |
| matmul_reduce_scatter_v2 | 1 | aclnnInner_matmul_reduce_scatter_v2.h |
| distribute_barrier | 1 | aclnnInner_distribute_barrier.h |
| ffn_to_attention | 1 | aclnnInner_ffn_to_attention.h |
| attention_to_ffn | 1 | aclnnInner_attention_to_ffn.h |
| allto_all_all_gather_batch_mat_mul | 1 | aclnnInner_allto_all_all_gather_batch_mat_mul.h |
| allto_allv_grouped_mat_mul | 1 | aclnnInner_allto_allv_grouped_mat_mul.h |
| grouped_mat_mul_allto_allv | 1 | aclnnInner_grouped_mat_mul_allto_allv.h |
| batch_mat_mul_reduce_scatter_allto_all | 1 | aclnnInner_batch_mat_mul_reduce_scatter_allto_all.h |
| grouped_mat_mul_all_reduce | 1 | aclnnInner_grouped_mat_mul_all_reduce.h |

#### 修改模式

所有算子遵循相同的修改模式：

1. **添加头文件包含**：在cpp文件的include区域添加`#include "aclnnInner_{算子名}.h"`
2. **移除内联声明**：删除`extern aclnnStatus aclnnInner...`声明（可能跨越多行）
3. **适配参数类型**：将`group`和`commMode`参数从`const char*`转换为`char*`，使用`const_cast<char*>()`

#### 批量修改脚本

可以使用以下脚本批量修改算子：

```bash
#!/bin/bash
# fix_mc2_operators.sh

# 1. 编译生成自动生成的头文件
bash build.sh --pkg --soc=ascend910b

# 2. 执行批量修改脚本
bash /tmp/fix_mc2_operators_v3.sh
```

#### 验证方法

修改完成后，执行编译验证：
```bash
bash build.sh --pkg --soc=ascend910b
```

编译成功后，检查生成的包文件：
```bash
ls -la build/cann-ops-transformer-custom_linux-aarch64.run
```

### 相关信息

- 自动生成的头文件位置：`build/autogen/inner/`
- 编译命令：`bash build.sh --pkg --soc=ascend910b --ops={算子名}`
- 参考PR：https://gitcode.com/cann/ops-transformer/pull/3258