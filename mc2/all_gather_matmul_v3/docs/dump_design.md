# 异常 Dump 方案 A：peermem 按实际数据大小 dump

## 背景

当前 `Mc2DumpUrmaContext` 固定 dump 1MB（`WIN_SIZE`），但实际 peermem 数据大小取决于 tiling 参数（M/K/rankSize/dtype）。方案 A 在 tiling 结构体中存储精确的 peermemDataSize，异常回调时从 tilingGM 读取该值，按实际大小 dump。

## peermemDataSize 计算公式

两算子统一公式（MX 量化场景）：
```
peermemDataSize = rankSize * M_per_rank * (K * typeSize + CeilDiv(K, 64) * 2)
```
- `typeSize = 0.5`（FP4）/ `1.0`（FP8）
- `CeilDiv(K, 64) * 2`：MX scale 每行字节数（对应 scale shape `[M, CeilDiv(K,64), 2]` × sizeof(fp8_e8m0)=1B）

| 算子 | rankSize | M_per_rank | K | 公式 |
|------|----------|------------|---|------|
| v2 | `worldSize_` | `m_` | `k_` | `worldSize_ * m_ * (k_ * typeSize + CeilDiv(k_,64)*2)` |
| v3 | `shapeInfo.rankSize` | `shapeInfo.mPerRank` | `shapeInfo.k` | `rankSize * m * (k * typeSize + CeilDiv(k,64)*2)` |

## dump 大小决策流

```
peermemSizeOffset != 0 && tilingGmArgIdx != 0
  → D2H 读 8B from tilingGM + offset → peermemDataSize
    → 0 < peermemDataSize <= 64MB → dumpSize = peermemDataSize（精确）
    → peermemDataSize == 0 或 > 64MB → dumpSize = WIN_SIZE（1MB fallback）
  → D2H 失败 → dumpSize = WIN_SIZE
else → dumpSize = WIN_SIZE
```

## 改动清单

### 共享基础设施

#### 文件 1: mc2/common/utils/mc2_exception_dump.h
- 新增常量 `MAX_PEERMEM_DUMP_SIZE = 64MB`
- Mc2DumpUrmaContext 新增参数 `tilingGmArgIdx` 和 `peermemSizeOffset`
- 新增逻辑：从 tilingGM D2H 读 8B peermemDataSize，决定 dumpSize
- #else no-op stub 同步更新签名

### v2 (AlltoAllMatmulV2) 增量改动

#### 文件 2: all_to_all_matmul_tiling_data.h
- allToAllMatmulTilingData 末尾新增 `uint64_t peermemDataSize{0}`

#### 文件 3: allto_all_matmul_v2_tiling_base.h — DoOpTiling()
- 计算 `td->peermemDataSize = worldSize_ * m_ * (k_ * typeSize + CeilDiv(k_,64)*2)`

#### 文件 4: allto_all_matmul_v2_tiling.cpp
- Mc2DumpUrmaContext 调用新增 tilingGmArgIdx=9, peermemSizeOffset=offsetof(peermemDataSize)

### v3 (AllGatherMatmulV3) 全新实施

#### 文件 5: all_gather_mx_matmul_urma_tiling_data.h
- 新增 include mc2_tiling_struct.h
- AllGatherMxMatmulUrmaTilingData 末尾新增 workspaceLayout + peermemDataSize

#### 文件 6: all_gather_mx_matmul_urma_impl.h
- include mc2_runtime_info.h
- 新增 runtimeInfo_ 成员 + execTurn_ + position 常量 + SetRuntimeInfo
- AllGatherProcess: WriteGmByPassDCache/IncCommitCount/IncWaitCount/WritePhaseByPassDCache
- MatmulProcess: WriteGmByPassDCache

#### 文件 7: all_gather_matmul_v3_apt.cpp
- include mc2_runtime_info.h
- 删除 (void)workspaceGM
- 初始化 Mc2RuntimeInfo + SetRuntimeInfo

#### 文件 8: all_gather_matmul_v3_tiling.cpp
- include mc2_exception_dump.h
- SetWorkSpace: workspaceSize = libApiSize + 64KB
- SetTilingData: 填充 workspaceLayout（2段）+ peermemDataSize
- 注册异常回调：Mc2DumpTilingData(tilingGmArgIdx=10, workspaceGmArgIdx=9) + Mc2DumpUrmaContext(contextArgIdx=0, tilingGmArgIdx=10, peermemSizeOffset=offsetof(peermemDataSize))

## 异常时 dump 产出

| 文件 | 内容 | 大小 | 来源 |
|------|------|------|------|
| args_info_* | kernel args 全部 GM 指针 | 256B | Mc2DumpTilingData |
| tiling_data_* | tiling 结构体（含 workspaceLayout + peermemDataSize） | 2048B | Mc2DumpTilingData |
| workspace_seg0_type0_* | LIB_API 段原始数据 | ~16MB | Mc2DumpTilingData |
| workspace_seg1_type12_* | Mc2RuntimeInfo — 每核执行进度 | 64KB | Mc2DumpTilingData |
| comm_context_* | CommContext — 通信拓扑 | ~1KB | Mc2DumpUrmaContext |
| peermem_window_* | peermem 窗口数据（实际大小，上限 64MB） | peermemDataSize | Mc2DumpUrmaContext |

## v2 vs v3 参数对照

| 参数 | v2 | v3 |
|------|----|----|
| op | "AlltoAllMatmulV2" | "AllGatherMatmulV3" |
| tilingGmArgIdx | 9 | 10 |
| workspaceGmArgIdx | 8 | 9 |
| contextArgIdx | 0 | 0 |
| tiling 结构体 | allToAllMatmulTilingData | AllGatherMxMatmulUrmaTilingData |
| usedCoreNum 字段 | tileQbmmTilingData.usedCoreNum | mmTile.usedCoreNum |
| impl 类 | AllToAllMxQuantMatmulUrmaImpl | AllGatherMxMatmulUrmaImpl |
| impl 入口 | Run() | Process() |
| AIV 通信方法 | RunAllToAll() | AllGatherProcess() |
| AIC 计算方法 | RunMatmul()+RunLocalMatmul() | MatmulProcess() |
