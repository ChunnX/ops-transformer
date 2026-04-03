# 需求
/home/j60100428/pytest/pytest_qsfa是已经写好了的pytest框架，这个测试框架是适配attention/kv_quant_sparse_flash_attention算子的（简称qsfa）
现在要求把/home/j60100428/pytest/pytest_qsfa框架移到attention/kv_quant_sparse_flash_attention/test/下面（目录不存在就创建），形成如下的目录结构
attention/kv_quant_sparse_flash_attention/test/pytest/框架实现代码
说几点详细要求：
    1. 代码严格按照本仓的开源格式要求写，Copyright等格式要求可以参照attention/fused_infer_attention_score/tests/pytest
    2. pytest框架的README.md资料（pytest_qsfa/pytest/README.md）也要适配
    3. pytest_qsfa的perf_run.sh相关脚本文件可以不用适配，不用移到qsfa算子目录下
    4. 在移动过程中优化pytest框架（但要保证核心功能没有损失：single和batch），有部分函数功能舍弃：
        run_npu_only不再需要，舍弃这部分功能（包括适配的函数）
        kv_quant_sparse_flash_attention_golden.py文件中generate_and_save_testdata函数舍弃decode_hifloat8参数，舍弃hifloat8 decode的功能，放弃hifloat8解码，因为compute_golden为True时的代码也要舍弃，不会上库，compute_golden为True时直接打印：当前golden CPU测计算尚存在问题，会导致cpu_result和npu_精度对比失败，暂不支持golden CPU。你可以把这话优化一下，同时cpu_result = None。但result_compare_method.py等result对比功能要保留
    5. 优化注释，把一些参数的要求尽量放进pytest框架的README.md资料中，具体参数要求可以查看pytest框架代码和attention/kv_quant_sparse_flash_attention下的算子实现和文档
    6. 不允许动/home/j60100428/pytest/pytest_qsfa目录下的框架源码，不允许动本仓内的代码实现，你只能在attention/kv_quant_sparse_flash_attention/下区创建test/pytest/，去实现qsfa的pytest框架
# 实现说明

## 已完成内容
1. 已在 `attention/kv_quant_sparse_flash_attention/test/pytest/` 下新建 pytest 框架目录，并将原 `/home/j60100428/pytest/pytest_qsfa/pytest` 框架按本仓目录组织方式适配到算子目录下。
2. 已保留 single、batch、graph 三条主流程，其中：
   - `test_kv_quant_sparse_flash_attention_single.py` 用于单用例执行；
   - `batch/test_kv_quant_sparse_flash_attention_pt_save.py` + `test_kv_quant_sparse_flash_attention_batch.py` 用于批量生成与批量回放；
   - `test_kv_quant_sparse_flash_attention_batch_graph.py` 用于图模式回放。
3. 已删除 `run_npu_only` 相关入口与依赖，不再提供该模式。
4. 已按需求不迁入 `perf_run.sh` 相关脚本。
5. 已保留 `result_compare_method.py` 以及结果落盘逻辑，便于后续恢复精度对比链路。

## 代码侧适配说明
1. `kv_quant_sparse_flash_attention_golden.py`
   - 保留测试数据构造能力；
   - 去掉 `decode_hifloat8` 相关接口和实现；
   - 去掉 CPU golden 计算分支；
   - 当 `compute_golden=True` 时只打印“当前 CPU golden 暂不支持”的提示，并令 `cpu_output=None`。
2. `utils.py`
   - 重新整理参数组合、Excel 读取、结果写表逻辑；
   - 当 `cpu_output is None` 时，统一记为 `SkipCompare`，表示仅校验 NPU 连通性。
3. `check_valid_param.py`
   - 将当前框架实际支持的参数约束集中校验，避免无效用例直接进入执行阶段。
4. `README.md`
   - 已按新目录重写；
   - 已把主要参数限制、能力边界、运行方式和当前 CPU golden 限制写入文档。
5. `test_run.sh` / `pytest.ini`
   - 已删除 `npu_only` 相关说明与 marker；
   - 保留 `single`、`batch`、`graph` 执行入口。

## 额外说明
1. 严格未修改 `/home/j60100428/pytest/pytest_qsfa` 目录下的源码。
2. 严格未修改本仓已有算子实现代码，仅在 `attention/kv_quant_sparse_flash_attention/` 下新增 `test/pytest/` 框架。
3. 已补充 `excel/.gitkeep`，便于后续放置 batch Excel 用例。

## 本地校验
1. 已对新增 Python 文件执行 `python3 -m py_compile attention/kv_quant_sparse_flash_attention/test/pytest/*.py attention/kv_quant_sparse_flash_attention/test/pytest/batch/*.py`，语法校验通过。
2. 由于当前环境未实际执行 NPU pytest 用例，本次主要完成了框架落库、能力裁剪和文档注解。
