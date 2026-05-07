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

## 待解决问题

### 1. 环境变量动态切换通信引擎

**需求**：通过环境变量 `ENV_MC2_COMM_MODE_AICPU` 控制 CCU/AICPU 模式切换。

**问题**：Kernel 层 HCCL 模板参数是编译时确定的，无法通过环境变量动态切换：

```cpp
// 模板参数在编译时确定
Hccl<HcclServerType::HCCL_SERVER_TYPE_AICPU> hccl_;  // AICPU 模式
Hccl<HcclServerType::HCCL_SERVER_TYPE_CCU> hccl_;    // CCU 模式
```

**可选方案**：

| 方案 | 说明 | 优点 | 缺点 |
|------|------|------|------|
| A. 编译两套 Kernel | 在 apt.cpp 中根据 tiling key 选择不同 kernel | 真正支持动态切换 | 代码量增加，需要两套模板 |
| B. 只保持 AICPU | Kernel 保持 AICPU 模板，环境变量只影响 Tiling/ACLNN 层 | 改动最小 | 实际通信引擎由模板参数决定，环境变量不完全生效 |
| C. 等接口适配完成 | 保持当前状态，等通信引擎参数接口适配后再修改 | 最稳妥 | 无法立即测试 CCU 模式 |

**建议**：当前采用方案 B（只保持 AICPU），待通信引擎参数接口适配完成后统一修改。

**环境变量影响范围（方案 B）**：

```cpp
// Tiling 层 - 可以动态切换
uint8_t commEngine = mc2tiling::IsUseAicpuCommMode() ? mc2tiling::AICPU_ENGINE : mc2tiling::A5_CCU_ENGINE;
mc2CcTilingConfig.SetCommEngine(commEngine);

// ACLNN 层 - 可以动态切换  
if (IsUseAicpuCommMode()) {
    NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_AICPU);
} else {
    NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_CCU);
}

// Kernel 层 - 无法动态切换（模板参数）
Hccl<HcclServerType::HCCL_SERVER_TYPE_AICPU> hccl_;  // 固定 AICPU
```

## 下一步工作

1. [ ] 确认环境变量方案选择（方案 A/B/C）
2. [ ] 添加环境变量判断逻辑到 Tiling 和 ACLNN 层
3. [ ] 等通信引擎参数接口适配完成后，移除环境变量控制
4. [ ] 跑通功能验证测试

## 参考

- PR 4531: https://gitcode.com/cann/ops-transformer/pull/4531
- PR 4653 (改造模式参考): https://gitcode.com/cann/ops-transformer/pull/4653