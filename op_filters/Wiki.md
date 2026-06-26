# MC2 算子测试选择工具 Wiki

## 1. 概述

本工具根据 Git PR 变更文件列表，通过多层过滤和依赖分析，自动确定需要执行的算子测试用例。

**核心目标**：避免每次 PR 都跑全量测试，智能地只测试受变更影响的算子及其依赖者。

**当前策略**：
- 路径不以 `mc2/` 开头 -> 全量测试
- `mc2/common/` 或 `mc2/3rd/` 变更 -> 全量测试（不再做特殊映射）
- 仅算子目录变更 -> 部分测试（变更算子 + 反向依赖算子）
- 白名单/忽略规则命中的变更 -> 无测试

---

## 2. 整体流程图

```mermaid
flowchart TD
    A["PR 变更文件列表输入"] --> B["Step 0: 动态生成依赖关系<br>扫描 mc2/ 解析 CMakeLists<br>生成 result.json"]
    B --> C["Step 1: 白名单路径过滤<br>whitelist.txt 路径被忽略"]
    C --> D{"过滤后无剩余文件？"}
    D -- 是 --> E1["无测试 - 输出空列表"]
    D -- 否 --> X{"路径不以 mc2/ 开头？"}
    X -- 是 --> J0["全量测试 - 流程终止"]
    X -- 否 --> F["Step 2: 忽略规则过滤<br>.md 文件 + tests/ 被忽略"]
    F --> G{"过滤后无剩余文件？"}
    G -- 是 --> E2["无测试 - 输出空列表"]
    G -- 否 --> H["Step 3: 非 mc2 目录检测"]
    H --> I{"存在 mc2 外文件？"}
    I -- 是 --> J1["全量测试 - 流程终止"]
    I -- 否 --> K["Step 4: common/3rd 检测"]
    K --> L{"common 或 3rd 有变更？"}
    L -- 是 --> J2["全量测试 - 流程终止"]
    L -- 否 --> M["Step 5: 算子名称提取<br>从路径提取算子名<br>检测 arch22/arch35 限制"]
    M --> N["Step 6: 反向依赖搜索<br>变更算子 + 依赖它的算子<br>只传播一层"]
    N --> O["Step 7: 输出与最终过滤<br>算子白名单屏蔽<br>arch 裁剪 supported_products<br>SOC 类型过滤<br>输出需要测试的算子名"]
```

---

## 3. 全分支输入文件与逐步过滤演示

以下构造一个覆盖所有分支的输入文件，逐行标注每个文件在哪一步被踢出或保留进入下一步。

### 3.1 输入文件（10 行，覆盖所有分支）

```
mc2/all_gather_matmul/tests/test_main.py
mc2/all_gather_matmul/README.md
mc2/matmul_all_reduce/tests/test_perf.py
ops-math/CMakeLists.txt
mc2/common/op_api/mc2_aclnn_util.h
mc2/3rd/quant_batch_matmul_v3/quant_batch_matmul_v3.cpp
mc2/all_gather_matmul/op_host/all_gather_matmul_def.cpp
mc2/matmul_all_reduce/op_host/op_tiling/arch22/matmul_all_reduce_tiling.cpp
mc2/matmul_reduce_scatter/op_host/op_tiling/arch35/matmul_reduce_scatter_tiling.h
mc2/common/op_kernel/unknown_file.xyz
```

### 3.2 逐步过滤过程（合并视图）

> 图例：~~删除线~~ = 被踢出，← 后标注踢出的 Step 和原因；普通文本 = 保留到最终

~~mc2/all_gather_matmul/tests/test_main.py~~ ← Step1 通配符匹配 mc2/*/tests/
~~mc2/all_gather_matmul/README.md~~ ← Step2 .md 后缀忽略
~~mc2/matmul_all_reduce/tests/test_perf.py~~ ← Step2 tests/ 目录忽略
~~ops-math/CMakeLists.txt~~ ← Step1后 路径不以 mc2/ 开头，全量测试
~~mc2/common/op_api/mc2_aclnn_util.h~~ ← Step4 common 文件，全量测试
~~mc2/3rd/quant_batch_matmul_v3/quant_batch_matmul_v3.cpp~~ ← Step4 3rd 文件，全量测试
~~mc2/common/op_kernel/unknown_file.xyz~~ ← Step4 common 文件，全量测试
mc2/all_gather_matmul/op_host/all_gather_matmul_def.cpp → Step5 提取算子: all_gather_matmul (无 arch 限制)
mc2/matmul_all_reduce/op_host/op_tiling/arch22/matmul_all_reduce_tiling.cpp → Step5 提取算子: matmul_all_reduce (arch22)
mc2/matmul_reduce_scatter/op_host/op_tiling/arch35/matmul_reduce_scatter_tiling.h → Step5 提取算子: matmul_reduce_scatter (arch35)

Step 6 反向依赖扩展后算子集合：

~~matmul_all_reduce~~ ← Step7 [ascend910b] 白名单屏蔽
~~matmul_all_reduce_add_rms_norm~~ ← Step7 [ascend910b] 白名单屏蔽
~~inplace_matmul_all_reduce_add_rms_norm~~ ← Step7 [ascend910b] 白名单屏蔽
all_gather_matmul → 保留
matmul_reduce_scatter → 保留

Step 7 arch 裁剪 + SOC 过滤后（假设 --soc ascend910b）：

all_gather_matmul → supported_products: [ascend910b]
matmul_reduce_scatter → supported_products: [ascend910b] (arch35 裁剪为 [910b,950]，与 --soc 交集)

**最终输出**：

```json
{
  "ascend910b": ["all_gather_matmul", "matmul_reduce_scatter"]
}
```

---

## 4. 关键配置与数据

### 4.1 白名单内容（whitelist.txt）

匹配到此文件中路径模式的变更文件会被忽略（不参与后续过滤）。

| 类型 | 路径 | 说明 |
|------|------|------|
| 通配符 | `tests/` | 顶层 tests 目录 |
| 通配符 | `mc2/*/tests/` | 各算子下的 tests 目录 |

### 4.2 算子依赖关系（result.json）

由 Step 0 动态扫描 `mc2/` 目录生成，共 35 个算子。以下列出有依赖关系的算子：

| 算子 | 依赖 | 支持芯片 |
|------|------|----------|
| all_gather_matmul | all_gather_matmul_v2 | ascend910b, ascend910_93, ascend950 |
| allto_all_matmul | matmul_allto_all | ascend910b, ascend910_93 |
| distribute_barrier_extend | distribute_barrier | ascend910b |
| grouped_mat_mul_allto_allv | allto_allv_grouped_mat_mul | ascend910b, ascend910_93 |
| inplace_matmul_all_reduce_add_rms_norm | matmul_all_reduce_add_rms_norm, matmul_all_reduce | ascend910b |
| matmul_all_reduce_add_rms_norm | matmul_all_reduce | ascend910b, ascend910_93 |
| matmul_reduce_scatter | matmul_reduce_scatter_v2 | ascend910b, ascend910_93 |
| moe_distribute_combine | moe_distribute_dispatch_v2, moe_distribute_combine_v2, moe_distribute_dispatch_v3, moe_distribute_combine_v3 | ascend910b |
| moe_distribute_combine_add_rms_norm | moe_distribute_dispatch_v2, moe_distribute_combine_v2, moe_distribute_dispatch_v3, moe_distribute_combine_v3 | ascend910b |
| moe_distribute_combine_teardown | moe_distribute_combine_setup | ascend910b |
| moe_distribute_combine_v2 | moe_distribute_dispatch_v2, moe_distribute_dispatch_v3 | ascend910b |
| moe_distribute_combine_v3 | moe_distribute_dispatch_v2, moe_distribute_dispatch_v3, moe_distribute_combine_v2 | ascend910b |
| moe_distribute_dispatch | moe_distribute_dispatch_v2, moe_distribute_dispatch_v3 | ascend910b |
| moe_distribute_dispatch_setup | moe_distribute_dispatch_v2, moe_distribute_dispatch_v3 | ascend910b |
| moe_distribute_dispatch_teardown | moe_distribute_dispatch_setup | ascend910b |
| moe_distribute_dispatch_v3 | moe_distribute_dispatch_v2 | ascend910b |
| quant_all_reduce | quant_reduce_scatter | ascend910b, ascend910_93 |
| quant_grouped_mat_mul_allto_allv | allto_allv_quant_grouped_mat_mul | ascend910b, ascend910_93 |

### 4.3 arch 限制与支持芯片对应关系

| arch 类型 | 支持芯片 | 说明 |
|-----------|----------|------|
| arch22 | ascend910b, ascend910_93 | 路径含 `op_host/op_tiling/arch22` |
| arch35 | ascend910b, ascend950 | 路径含 `op_host/op_tiling/arch35` |
| 无限制 | 全部芯片 | 同一算子同时有 arch22 和 arch35 路径时视为无限制 |


