# MatmulReduceScatterV2 AICPU 改造说明

## 背景

参考 PR [cann/ops-transformer#4653](https://gitcode.com/cann/ops-transformer/pull/4653)（在 `all_gather_matmul_v2`、`matmul_all_reduce` 等算子上展示的改造模式），将 `MatmulReduceScatterV2` 算子在 <term>Ascend 950PR/Ascend 950DT</term>（DAV_3510 / arch35）平台上的 HCCL 服务端类型从 **CCU**（Communication Co-processor Unit）切换为 **AICPU**。

## 改造范围

- **改动平台**：仅 DAV_3510（Ascend 950）的 CCU 路径切换为 AICPU
- **不动平台**：DAV_2201（A2/A3）的 MTE / aiv 路径保持原状
- **不动逻辑**：算子计算逻辑、tiling 算法、kernel 内部实现均无改动，仅切换 HCCL 服务端引擎

## 改造模式（来自 PR 4653）

每个被改造的 MC2 算子遵循 4 类统一改动：

| # | 文件类型 | 改动 |
|---|---------|------|
| 1 | `op_api/aclnn_*.cpp` | DAV_3510 分支：`NNOPBASE_HCCL_SERVER_TYPE_CCU` → `NNOPBASE_HCCL_SERVER_TYPE_AICPU` |
| 2 | `op_host/*_def.cpp` | 在 ascend950 配置的 `MC2().HcclGroup("group")` 之后新增 `this->MC2().HcclServerType(HcclServerType::AICPU);` |
| 3 | `op_host/op_tiling/arch35/*.cpp` | 在 `Mc2CcTilingConfig` 构造之后新增 `mc2CcTilingConfig.SetCommEngine(2);`（2 = `AICPU_ENGINE`） |
| 4 | `op_kernel/arch35/*.h` | `Hccl<HcclServerType::HCCL_SERVER_TYPE_CCU>` → `Hccl<HcclServerType::HCCL_SERVER_TYPE_AICPU>` |

## 文件改动清单（共 8 个文件，9 处修改）

### 1. ACLNN 入口（1 处）

**文件**：`mc2/matmul_reduce_scatter_v2/op_api/aclnn_matmul_reduce_scatter_v2.cpp`

```diff
     if (NnopbaseSetHcclServerType) {
         if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2201) {
             NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_MTE);
         } else if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
-            NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_CCU);
+            NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_AICPU);
         }
     }
```

### 2. 算子定义（1 处）

**文件**：`mc2/matmul_reduce_scatter_v2/op_host/matmul_reduce_scatter_v2_def.cpp`

仅在 `ascend950` 配置块后追加，A2/A3（`ascend910b`/`ascend910_93`）配置块**不变**：

```diff
     this->AICore().AddConfig("ascend950", aicore_config);
     this->MC2().HcclGroup("group");
+    this->MC2().HcclServerType(HcclServerType::AICPU);

     OpAICoreConfig aicore_config_910b;
```

### 3. arch35 Tiling（2 处）

**文件 A**：`mc2/matmul_reduce_scatter_v2/op_host/op_tiling/arch35/matmul_reduce_scatter_v2_tiling.cpp`

```diff
     AscendC::Mc2CcTilingConfig mc2CcTilingConfig(
         group, opType, rsConfig, reduceType, dataType, dataType
     );
+    mc2CcTilingConfig.SetCommEngine(2);

     OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(matmulReduceScatterV2TilingData_->mc2InitTiling) != 0,
```

**文件 B**：`mc2/matmul_reduce_scatter_v2/op_host/op_tiling/arch35/quant_bmm_reduce_scatter_tiling.cpp`

```diff
     AscendC::Mc2CcTilingConfig mc2CcTilingConfig(
         group, opType, rsConfig, reduceType, dataType, dataType
     );
+    mc2CcTilingConfig.SetCommEngine(2);

     OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(quantBmmMatmulReducescatterTilingData_->mc2InitTiling) != 0,
```

> 说明：`SetCommEngine(2)` 的常量 `2` 对应 `mc2/common/op_host/op_tiling/mc2_tiling_utils.h` 中定义的 `AICPU_ENGINE`（参见 PR 4653 中新增的常量定义注释）。

### 4. arch35 Kernel HCCL 模板参数（4 处）

将 4 个 arch35 kernel header 中的 `Hccl` 模板实例化从 CCU 改为 AICPU：

| 文件 | 改动行 |
|------|--------|
| `op_kernel/arch35/matmul_reduce_scatter_fp16_bf16.h` | L61 |
| `op_kernel/arch35/matmul_a2a_vec_reduce_fp16_bf16.h` | L77 |
| `op_kernel/arch35/quant_bmm_reduce_scatter_fp8_hif8.h` | L74 |
| `op_kernel/arch35/quant_bmm_a2a_vec_reduce_fp8_hif8.h` | L92 |

```diff
-    Hccl<HcclServerType::HCCL_SERVER_TYPE_CCU> hccl_;  // CCU模式
+    Hccl<HcclServerType::HCCL_SERVER_TYPE_AICPU> hccl_;  // AICPU模式
```

## 兼容性影响

| 维度 | 影响 |
|------|------|
| Atlas A2/A3（DAV_2201） | **无影响**，aiv 模式继续走 AI Core / MTE |
| Ascend 950（DAV_3510） | HCCL 通信任务从 CCU 迁移到 AICPU 执行；算子语义和接口不变 |
| ACLNN API | **无破坏性变更**，参数列表、返回值、错误码完全不变 |
| 算子精度 | 无影响（仅切换通信引擎，不改变计算路径） |
| Graph 模式 | `op_def` 中新增 `HcclServerType(AICPU)` 用于注册阶段告知框架 |

## 验证建议

1. **编译验证**

   ```bash
   bash build.sh --pkg --soc=ascend950 --ops=matmul_reduce_scatter_v2 -j16
   ```

2. **功能验证**：通过 [examples/test_aclnn_matmul_reduce_scatter_v2.cpp](../examples/test_aclnn_matmul_reduce_scatter_v2.cpp) 在 Ascend 950 多卡环境运行端到端测试。

3. **回归验证**：在 Atlas A2/A3 上重跑既有 UT，确认 aiv 路径未受影响。

## 参考

- 参考 PR：[cann/ops-transformer!4653](https://gitcode.com/cann/ops-transformer/pull/4653)
- 算子文档：[aclnnMatmulReduceScatterV2.md](./aclnnMatmulReduceScatterV2.md)
