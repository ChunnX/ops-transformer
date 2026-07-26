# mixed_quant_sparse_flash_mla算子测试框架
## 功能说明
基于pytest测试框架，实现mixed_quant_sparse_flash_mla算子的功能验证：
- **CPU侧**：复现算子功能用以生成golden数据
- **NPU侧**：通过torch_npu进行算子直调、图模式调用获取实际数据
- **精度对比**：进行CPU与NPU结果的精度对比验证算子功能

## 当前实现范围
### 参数限制

- **数据格式**:
    - **layout_q**：BSND、TND
    - **layout_kv**：PA_BBND

### 环境配置

#### 前置要求
1、 确认torch_npu为最新版本

#### custom包调用
支持custom包调用

## 文件结构
#### pytest文件结构说明
- test_run.sh                                               # 用例执行脚本
- mixed_quant_sparse_flash_mla_golden.py                   # 算子入参处理及cpu侧golden实现
- result_compare_method.py                                  # cpu golden与npu结果精度对比脚本
- pytest.ini                                                # 创建测试标记

单用例测试：
- test_mixed_quant_sparse_flash_mla_single.py              # pytest测试单用例运行主程序
- mixed_quant_sparse_flash_mla_paramset.py                 # 单用例入参数配置

批量测试：
- test_mixed_quant_sparse_flash_mla_batch.py               # 用例批量测试主程序，读取pt用例并获取npu输出，结果保存至excel文件
- ./batch/test_mixed_quant_sparse_flash_mla_pt_save.py     # 读取excel表格批量生成用例pt文件
- ./batch/mixed_quant_sparse_flash_mla_process.py          # 调用算子接口获取npu输出

## 使用方法
在pytest文件夹路径下执行：

### 运行测试用例
#### 单用例测试
1、手动配置mixed_quant_sparse_flash_mla_paramset.py的参数

2、执行指令：
``` bash
bash test_run.sh single
```
#### 批量测试

支持三种批量测试模式，均可通过选项指定excel文件、sheet名、pt保存目录：

| 命令 | 功能 |
|------|------|
| `batch_save` | 从excel读取用例 → golden计算 → 保存pt文件 |
| `batch_exec` | 读取pt文件 → NPU测试（需先`batch_save`） |
| `batch` | 全流程：`batch_save` + `batch_exec`，默认完成后清理pt文件 |

##### 选项参数

| 选项 | 说明 | 默认值 | 适用命令 |
|------|------|--------|---------|
| `--excel <路径>` | 指定excel文件路径 | `./excel/example.xlsx` | batch_save / batch_exec / batch |
| `--sheet <名称>` | 指定excel sheet名 | `decode` | batch_save / batch |
| `--pt-dir <目录>` | 指定pt文件保存/读取目录 | `qsmla_testcase` | batch_save / batch_exec / batch |
| `--keep-pt` | 执行完成后保留pt文件 | 默认清理 | 仅batch |

##### 使用示例
``` bash
# batch_save：只生成pt文件
bash test_run.sh batch_save
bash test_run.sh batch_save --excel my.xlsx --sheet prefill --pt-dir my_pt

# batch_exec：只执行NPU测试（需先batch_save生成pt文件）
bash test_run.sh batch_exec
bash test_run.sh batch_exec --pt-dir my_pt

# batch：全流程（默认清理pt文件）
bash test_run.sh batch

# batch：全流程，保留pt文件
bash test_run.sh batch --keep-pt

# batch：全流程，自定义所有参数
bash test_run.sh batch --excel my.xlsx --sheet prefill --pt-dir my_pt --keep-pt
```

## pytest_v2 batch 一致性扩展

本目录是 MQSMLA 自己的实现，不再从 SMLA pytest 借用主算子 executor。原输入生成、量化 CPU
golden、PT save/load、process 和 `check_result` 均保留；一致性入口在这些原流程上增加模式
变换、level 3、MD5 和 metadata 白盒分析。

| 文件 | 职责 |
| --- | --- |
| `test_mixed_quant_sparse_flash_mla_batch_consistency.py` | 模式 1-4 的 MQ PT 入口 |
| `batch/mixed_quant_sparse_flash_mla_consistency.py` | MQ 专用 executor |
| `batch/mixed_quant_sparse_flash_mla_process.py` | 原两段调用；新增可选 `metadata_sink`，旧返回值不变 |
| `batch/mixed_quant_sparse_flash_mla_consistency_prepare.py` | Excel→原量化golden→baseline/derived PT的CPU-only验收 |
| `batch/consistency/transform.py` | 双 layout/PA 变换、FP8 bit比较与实际输入语义检查 |
| `batch/consistency/metadata_coverage.py` | FA/FD展开与PR floor/需求ceil公式双轨报告 |
| `batch/consistency/reporting.py` | 原PT来源、shape/dtype/device/byte/digest/sample输入摘要 |
| `batch/consistency/offline.py` | 模式1-4 derived golden scope与PT round-trip |
| `batch/consistency/relation_manifest.py` | versioned baseline/derived TokenOrigin关系、内容指纹、预检与覆盖清单 |
| `batch/consistency/suite_config.py` | versioned per-PT suite配置、精确selector及shape-aware预检 |
| `batch/consistency/result_artifact.py` | 逐模式结果JSON、level3 PASS门禁、完整性检查与聚合 |
| `batch/consistency/runtime_capability.py` | level3可执行探针、异常恢复短压测及runtime/custom OPP内容指纹 |
| `batch/consistency/metadata_stub.py` | CPU-only单核到多核候选规划及默认关闭hook约束 |
| `excel/mqsmla_batch_consistency_cases.xlsx` | MQSMLA专属4行CPU关系矩阵；压力候选在独立sheet且不参与执行 |
| `mixed_quant_sparse_flash_mla_golden.py` | 原量化golden；ori mask 0/3/4统一计算并裁剪可见KV窗口 |
| 其余 `batch/consistency/*.py` | adapter、schema、layout、compare、flow和命名配置 |

```bash
# B=8 -> B=3 + B=5
bash test_run.sh consistency --pt-dir ./mqsmla_testcase \
  --modes reorder,split \
  --split-groups '0,1,2;3,4,5,6,7'

# 无NPU离线验收
bash test_run.sh consistency_prepare \
  --excel ./excel/mqsmla_batch_consistency_cases.xlsx \
  --sheet MQSMLA --pt-dir /tmp/mqsmla_pt --report-dir /tmp/mqsmla_report
```

当前本机 runtime 会在第一条 NPU 操作时拒绝 level 3；必须等支持 level 3 的
runtime/custom包组合后，才能验收FD与主算子一致性。临时metadata-only probe已从正式`batch/`
删除；`consistency_prepare`只验证CPU输入/golden/PT链，不会声称NPU PASS。
每个baseline和算子汇总目录还会输出relation manifest；其中PASS只代表TokenOrigin边界、scope、
语义oracle、PT内容指纹及mode/layout/mask/S2覆盖预检通过。

### 正式在线配置与结果

多个模式参数优先写入suite文件，不把一个`3+5`或`4K+12K`配置强套到所有PT：

```bash
bash test_run.sh consistency \
  --pt-dir /tmp/mqsmla_pt/baseline \
  --config-file ../../../../requs/cases/SMLA_MQSMLA_batch一致性执行配置.json \
  --report-dir ./result/mqsmla_consistency \
  --probe-attempts 1 \
  --device-id 0 \
  --custom-opp-root "$STAGING" \
  --expected-custom-opp-manifest "$MANIFEST"
```

调用自定义算子前先加载`/home/Ascend/vendors/custom_transformer/bin/set_env.bash`。`STAGING`应指向
待验证构建的`packages/vendors/custom_transformer`，`MANIFEST`是该根的期望全树SHA256；shell会
把staging和`op_api/lib`分别放到两个搜索路径首位。身份不符会在set level和NPU smoke前失败。

suite只加载一次，但每个PT分别按operator/case name/PT basename/content SHA256精确匹配，并按
当前B和有效Q长度校验。`--config-file`不能和`--modes/--profile/--seed/groups/sizes`混用；
`-M 1,2`只保留在旧CLI转换层。

每次执行创建独立run目录，`results/*.json`对应一个`(operator,PT,mode)`，汇总为
`consistency_result_summary.json`，并先写`runtime_capability.json`。探针执行
`set 3 -> readback -> 普通NPU dispatch -> restore`，不能以setter/getter成功替代可执行性；
`--probe-attempts 10`可做异常恢复短压测。探针失败时各模式仍完成纯CPU transform/semantic
oracle预检，但不导入MQ executor，再绑定同一能力证据写`UNSUPPORTED`；因此不会掩盖
`INVALID_TRANSFORM`。`pytest.log`仅用于诊断。只有探针通过、真实metadata+主算子执行报告通过且requested/observed
deterministic level均为3，逐模式状态才可写成`PASS`。聚合器还会检查预期artifact数和pytest
子进程异常，因此skip或退出码0不会被当成NPU通过。完整PASS返回0，真实FAIL返回1，
`UNSUPPORTED/PARTIAL/INCOMPLETE`等未完成状态返回2。汇总还校验唯一PT数和完整PT/mode矩阵；
损坏、重复或异算子旧artifact会写入汇总错误项，不能因文件总数刚好相等而通过。

当前模式1-4统一属于`same-source-cross-call`。独立编写PT之间的
`independent-cross-case`和同一次调用内位置互比`single-call-intra-case`是补充场景，artifact
会如实列为gap，不把模式1/3改名冒充。metadata stub planner目前只从自然metadata找候选并给出
拟拆分S2 block区间；FD契约、公式、安装包和level3关闭阻断前不会生成或执行synthetic tensor。

端到端命令、Excel列、状态/退出码和排障见
`../../../../requs/SMLA_MQSMLA_pytest_batch一致性测试使用指南.md`。
