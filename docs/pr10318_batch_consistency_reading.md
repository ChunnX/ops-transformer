# PR #10318 分层阅读笔记：pytest batch consistency

> PR：[#10318](https://gitcode.com/cann/ops-transformer/pull/10318)  
> 标题：`pytest: batch consistency`  
> 分支：`SH_jingsong/ops-transformer_0716:master` → `cann/ops-transformer:master`  
> 对比基线：`5a2c52a706d6394845c325cc09391d4ab18e5648`  
> 变更规模：17 个文件，约新增 1196 行、删除 262 行。

## 1. 一句话结论

本 PR 针对 PR 基线中的旧 batch consistency 测试代码进行迁移和重构，将实现放到 `attention/sparse_flash_mla/tests/pytest/batch_consistency`，使它成为 SMLA 家族测试的共享实现；同时增加格式适配、token 来源对齐、独立样本关系测试、deterministic level 3 守卫和严格 bitwise 比较，并让 MQSMLA、QSMLA 接入同一套框架。

它主要改的是测试基础设施和测试覆盖，不改 attention 算子的 kernel、host tiling 或 aclnn 接口。

## 2. 修改范围总览

| 层次 | 目录/文件 | 作用 |
|---|---|---|
| PR 基线中的旧路径 | `attention/common/tests/pytest/batch_consistency` | PR 基线里曾存在的旧测试框架路径；本 PR 将其迁移/重写到 SMLA 测试目录 |
| 共享框架 | `attention/sparse_flash_mla/tests/pytest/batch_consistency` | 新增/迁移 schema、模型、转换、运行、比较和 pytest 支持代码 |
| SMLA 测试 | `attention/sparse_flash_mla/tests/pytest/test_sparse_flash_mla_batch_consistency.py` | 改用新路径，并增加独立关系测试 |
| MQSMLA 测试 | `attention/mixed_quant_sparse_flash_mla/tests/pytest/...` | 接入新框架，增加独立关系测试，改为动态加载本算子模块 |
| QSMLA 测试 | `attention/quant_sparse_flash_mla/tests/pytest/...` | 新增 batch consistency 测试入口及配置 |

文件级变化可概括为：1 个旧文件删除；9 个框架文件从旧路径重定位并重写；1 个新框架支持文件；MQSMLA 2 个测试文件修改；QSMLA 新增测试文件并修改配置/工具；SMLA 测试入口和工具修改。

## 3. 推荐阅读顺序

建议按下面的调用链阅读：

```text
test_*_batch_consistency.py
        ↓ 收集 Excel/pytest case 与 independent relation
pytest_support.py
        ↓ 装载算子 process、构造变换 case
runner.py
        ↓ level 3、baseline/derived 执行、结果汇总
adapter.py + schema.py
        ↓ 统一 SMLA/MQSMLA/QSMLA 数据格式并校验
transform.py
        ↓ 生成 reorder/split/partition/shape/isolation 关系
compare.py
        ↓ 按 TokenOrigin 对齐 output/LSE，做精度与 bitwise 比较
```

## 4. 各文件职责

### 4.1 `adapter.py`：统一三种 case 格式

新增 `CaseAdapter`，通过输入字段自动识别：

- SMLA：tensor 位于 `input`，元数据使用 `N1/N2/D/B`；
- MQSMLA：tensor 位于 `op_input`，使用 `num_heads_q/num_heads_kv/head_dim/batch_size`；
- QSMLA：同样使用 `op_input`，但额外存在 `q_descale/ori_kv_descale/cmp_kv_descale` 等反量化 scale。

它还统一 batch size、Q/KV layout、元数据读写，并提供逻辑 KV 长度获取。对 `BSND`、`TND` 和 `PA_BBND` 分别处理；`PA_BBND` 没有可推导的逻辑长度时会显式报错，避免把物理 padding 当成有效 token。

### 4.2 `schema.py`：输入结构和不变量

负责校验三类算子的必需顶层字段、tensor 字段、metadata 字段，并计算物理/有效 Q 长度。还生成 baseline 的 `TokenOrigin`，为后续 batch 重排或拆分后的结果回溯到原始 token 提供依据。

### 4.3 `model.py`：测试对象模型

补充/扩展 `RunResult`、`ConsistencyCase`、`TokenOrigin` 等数据结构，使测试关系不仅携带输入，还携带：

- 输出 token 的来源；
- 输出坐标（适用于跨 case 的独立关系）；
- 变换描述和比较范围。

### 4.4 `transform.py`：生成 batch 关系

支持多种关系变换，包括：

1. batch reorder：改变样本顺序；
2. batch split：把一个 batch 拆成多个 batch；
3. token partition：按 token 范围切分；
4. shape change：改变物理形状但保留有效 token 语义；
5. token isolation：将 token 放入独立样本，检查结果是否受其他样本影响。

变换过程中同步 batch、长度、layout 和 SMLA 特有的 top-k 字段，并记录有效输出范围。

### 4.5 `compare.py`：origin-aware 比较

新增两级比较：

- 数值精度比较：复用各算子已有的 `result_compare_method.check_result`；
- 严格 bitwise 比较：把 tensor 拷贝到 CPU 后按字节计算 MD5，要求 shape 相同且字节完全一致。

比较前不会简单按物理 batch 顺序比较，而是根据 `TokenOrigin` 将 baseline 的 output/LSE 对齐到 derived case 的 token 顺序。这样可以正确比较 reorder、split、TND/BSND 和跨 case 映射。

### 4.6 `runner.py`：执行和状态管理中心

`ConsistencyRunner` 完成一次关系测试的主流程：

1. 校验 schema 和不变量；
2. 设置 NPU device；
3. 进入 `DeterministicLevelGuard`，设置 deterministic level 3；
4. 执行 baseline；
5. 执行每个变换后的 case；
6. 同时进行 batch consistency bitwise 比较和 baseline precision 比较；
7. 恢复进入测试前的 deterministic level。

对不支持 level 3、算子包/SoC 不匹配、case 参数不支持等情况进行边界分类，输出 `UNSUPPORTED` 或 `ENVIRONMENT_UNSUPPORTED`，避免把环境限制误报成算子精度失败。

### 4.7 `pytest_support.py`：pytest 接入层

负责从 Excel/测试配置收集普通 case 矩阵和 independent relation 矩阵，动态加载算子自己的 `result_compare_method`、参数校验和 process 模块，并提供：

- `run_pytest_case`：同一原始 case 内生成关系并运行；
- `run_independent_pytest_case`：baseline 与 derived 来自不同 case，按显式 token 映射比较；
- `load_operator_module`：避免把 MQSMLA/QSMLA 的同名模块错误地当成公共模块导入。

## 5. 三个算子的接入方式

### SMLA

原测试入口改为从自己的 `tests/pytest` 目录导入 `batch_consistency`，并增加 independent relation 测试。原先放在 common 下的框架代码随之迁移到 SMLA 目录。

### MQSMLA

不再直接导入公共 `result_compare_method` 和 process，而是使用 `load_operator_module` 从 MQSMLA 自己的测试目录动态加载；新增独立 batch consistency 测试。这样可保留算子特有的参数检查和输出比较逻辑。

### QSMLA

新增 `test_quant_sparse_flash_mla_batch_consistency.py`，实现与 SMLA/MQSMLA 相同的 executor 适配；同时新增/修改 pytest 配置和工具，使 QSMLA 的量化输入、反量化 scale、LSE 结果也能进入 level-3 一致性检查。

## 6. 这次 PR 解决的测试盲点

- 过去公共框架默认一种输入结构，难以同时覆盖 SMLA、MQSMLA、QSMLA；现在通过 `CaseAdapter` 统一差异。
- 仅按 tensor 物理位置比较会误判 reorder/split；现在通过 `TokenOrigin` 做语义对齐。
- 只跑同一个 case 的 batch 变换不足以验证样本独立性；现在加入 independent relation 和显式坐标映射。
- batch consistency 不应只看近似误差；现在同时要求数值通过和必要时 bitwise 一致。
- deterministic level 只在主算子调用期间设置可能不完整；现在由守卫覆盖 baseline 和 derived 调用，并负责恢复状态。
- 物理 padding、有效 Q/KV 长度和 TND/BSND layout 混用容易产生假阳性；现在集中校验并按逻辑长度比较。

## 7. 阅读时需要注意的边界

1. PR 页面描述区基本为空，变更意图主要需要从代码和标题还原。
2. 该 PR 的验证重点是 pytest batch consistency 框架；它没有修改算子 kernel/tiling，因此不能据此判断算子实现本身新增了 level-3 能力。
3. `deterministic level 3` 依赖安装的 `torch_npu` 和算子包支持；框架会把不支持情况分类为环境/能力边界，实际运行时仍需结合日志判断。
4. 具体测试 case 仍由各算子自己的 Excel/pytest 数据提供，框架只是统一收集、变换、执行和比较。

## 8. 快速定位命令

```bash
git diff 5a2c52a706d6394845c325cc09391d4ab18e5648..147df997 --stat
git diff 5a2c52a706d6394845c325cc09391d4ab18e5648..147df997 --name-status
rg "DeterministicLevelGuard|run_independent_pytest_case|TokenOrigin|CaseAdapter" \
  attention/sparse_flash_mla/tests/pytest attention/*sparse_flash_mla/tests/pytest
```
