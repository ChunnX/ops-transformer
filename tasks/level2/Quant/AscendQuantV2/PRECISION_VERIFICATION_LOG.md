# AscendQuantV2 精度验证记录

- **日期**：2026-04-01  
- **环境**：`conda activate nkb` → `source /root/miniconda3/envs/nkb/Ascend/ascend-toolkit/set_env.sh` → `cd NPUKernelBench_workspace` → `source set_framework_env.sh`（与 `.codex/nkb-kernel-bench-validation/SKILL.md` / `env-setup.md` 一致；本机可再 `source $CONDA_PREFIX/Ascend/cann-*/set_env.sh`）  
- **命令**：`python run_multi_test.py -task_name AscendQuantV2 -stages compile precision -n_case 1 -config single_npu_config.yaml`  
- **约束**：调试 **≤3 轮**；**仅改算子目录**（`answer/0`、`validation/` 等），不改 `framework/`、`libs/common_include/` 等公共工程。

## 最终结果（单例 case_id=0）

| 项目 | 状态 |
|------|------|
| **compile_pass** | **是** |
| **precision_pass** | **是**（`n_success=1, n_failed=0`） |

## 调试轮次（共 3 轮）

### 第 1 轮（precision-debug §7 / §8）

- **现象**：`aclnnAscendQuant … not in libopapi.so`（实际应绑定 **`aclnnAscendQuantV2`** + **`libcust_opapi.so`**）。  
- **根因**：`pybind.cpp` 写死 `aclnnAscendQuant`，与 msopgen **`aclnn_ascend_quant_v2.h`** 不一致；且 autogen 含 **`axis`**、`dstType` 为 **`int64_t`**。  
- **修改**：`validation/pybind.cpp` — `EXEC_NPU_CMD(aclnnCustomOp, …)`；补 **`axis=-1`**；**`int64_t` dst**；**`char* round_mode`** 用可写 `vector<char>` 缓冲。

### 第 2 轮（precision-debug §4 / optiling 技能）

- **现象**：**EZ9999**，`AscendQuantV2 do tiling failed` / `MatchCache`。  
- **根因**：**`.Tiling`** 回调里 **`GetCompileInfo<AscendQuantV2CompileInfo>()`** 在 Bench 路径常为空。  
- **修改**：`answer/0/op_host/ascend_quant_v2_tiling.cpp` 与 **`ascend_quant_v2_regbase_tiling.cpp`** 中 **`GetCompileInfo()`** 改为 **`GetPlatformInfo()` + `PlatformAscendC`**（`GetCoreNumAiv`、`GetCoreMemSize(UB)`、`GetCurNpuArch` 判定 910B）。

### 第 3 轮（precision-debug §6）

- **现象**：`custom_check_precision` 中 **`ad`** 已 `reshape(-1)`，分母仍为多维，`/` 广播报错。  
- **修改**：`validation/prepare_inputs.py` — **`o32/n32` 均 `reshape(-1)`** 后再算 **`ad`/`rd`**。

## 后续

- 单例通过后按 SKILL 执行全量：**`-n_case 16`**（与 `test_cases.csv` 行数一致）。

## 本轮变更文件（算子内）

- `validation/pybind.cpp`  
- `validation/prepare_inputs.py`  
- `answer/0/op_host/ascend_quant_v2_tiling.cpp`  
- `answer/0/op_host/ascend_quant_v2_regbase_tiling.cpp`  
