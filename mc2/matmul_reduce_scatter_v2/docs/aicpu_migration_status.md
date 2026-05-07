# MatmulReduceScatterV2 AICPU 改造状态说明

## 改造背景

将 `MatmulReduceScatterV2` 算子在 Ascend 950 (DAV_3510 / arch35) 平台上的 HCCL 服务端从 CCU 切换为 AICPU。

参考 PR: [cann/ops-transformer#4531](https://gitcode.com/cann/ops-transformer/pull/4531)

## 已完成修改

### 修改文件清单

| 文件 | 修改内容 |
|------|---------|
| `op_api/aclnn_matmul_reduce_scatter_v2.cpp` | 封装 `SetNnopbaseHcclServerTypeByArch` 函数，在 3 处调用 |
| `op_host/matmul_reduce_scatter_v2_def.cpp` | 删除 `HcclServerType(AICPU)` 配置行 |
| `op_host/op_tiling/arch35/matmul_reduce_scatter_v2_tiling.cpp` | 使用 `mc2tiling::AICPU_ENGINE` 常量，添加日志 |
| `op_host/op_tiling/arch35/quant_bmm_reduce_scatter_tiling.cpp` | 使用 `mc2tiling::AICPU_ENGINE` 常量，添加日志 |
| `op_kernel/arch35/matmul_reduce_scatter_fp16_bf16.h` | 使用 V2 接口 `InitV2/SetCcTilingV2`，删除参数 |
| `op_kernel/arch35/matmul_a2a_vec_reduce_fp16_bf16.h` | 使用 V2 接口 `InitV2/SetCcTilingV2`，删除参数 |
| `op_kernel/arch35/quant_bmm_reduce_scatter_fp8_hif8.h` | 使用 V2 接口 `InitV2/SetCcTilingV2`，删除参数 |
| `op_kernel/arch35/quant_bmm_a2a_vec_reduce_fp8_hif8.h` | 使用 V2 接口 `InitV2/SetCcTilingV2`，删除参数 |
| `op_kernel/matmul_reduce_scatter_v2_apt.cpp` | 删除宏中的 `mc2InitTiling/mc2CcTiling` 参数 |
| `mc2/common/op_host/op_tiling/mc2_tiling_utils.h` | 添加 `AICPU_ENGINE=2` 常量 + `IsUseAicpuCommMode()` 函数 |

## 环境变量控制说明

**环境变量**：`ENV_MC2_COMM_MODE_AICPU`

**控制逻辑**：
- 有设置时（值为 `1`/`y`/`Y`）：使用 AICPU 通信模式
- 无设置时：使用 CCU 通信模式

**影响范围**：

| 层级 | 文件 | 控制方式 |
|------|------|---------|
| Tiling | `matmul_reduce_scatter_v2_tiling.cpp` | `SetCommEngine(AICPU_ENGINE/CCU_ENGINE)` |
| Tiling | `quant_bmm_reduce_scatter_tiling.cpp` | `SetCommEngine(AICPU_ENGINE/CCU_ENGINE)` |
| ACLNN | `aclnn_matmul_reduce_scatter_v2.cpp` | `SetHcclServerType(AICPU/CCU)` |
| Kernel | `*.h` headers | 模板参数固定为 AICPU（不受环境变量影响） |

**使用方法**：
```bash
# 使用 AICPU 模式
export ENV_MC2_COMM_MODE_AICPU=1
python test_script.py

# 使用 CCU 模式（默认）
unset ENV_MC2_COMM_MODE_AICPU
python test_script.py
```

**注意事项**：
- Kernel 层 HCCL 模板参数为编译时确定，固定为 `HCCL_SERVER_TYPE_AICPU`
- 环境变量只影响 Tiling 和 ACLNN 层的运行时配置
- 待通信引擎参数接口适配完成后，移除环境变量控制

## 已完成修改

## 参考

- PR 4531: https://gitcode.com/cann/ops-transformer/pull/4531
- PR 4653 (改造模式参考): https://gitcode.com/cann/ops-transformer/pull/4653