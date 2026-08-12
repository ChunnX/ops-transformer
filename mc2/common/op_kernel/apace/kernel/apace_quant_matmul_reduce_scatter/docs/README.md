# apace_quant_matmul_reduce_scatter_v2 算子

## 基本信息

- **算子名称**：apace_quant_matmul_reduce_scatter_v2
- **算子类别**：MC2 通算融合（Quant MatMul + AllToAll PUT 通信 + reduceSum 搬运，3 级流水线）
- **数学公式**：C_r = Σ_{i=0}^{R-1} dequant(A_i[r·M/R : (r+1)·M/R]) × dequant(B_i)，R=4（rankSize）
- **开发状态**：✅ 开发完成（精度全部通过 + 性能达标）
- **平台**：Ascend 950（`--npu-arch=dav-3510`），`KERNEL_TYPE_MIX_AIC_1_1`

## 接口

| 参数 | 方向 | shape (per rank) | dtype | 说明 |
|------|------|------------------|-------|------|
| hcommCtx | 输入 | - | CommContext | UDMA 通信上下文（host 建链） |
| A | 输入 | [M, K/rankSize] | fp8_e4m3fn / fp8_e5m2 | K-split |
| ScaleA | 输入 | [M, CeilDiv(K/rankSize,64)*2] | fp8_e8m0 | MXFP8 scale |
| B | 输入 | [K, N] | 同 A | 全量复制（K = rankSize × Ka） |
| ScaleB | 输入 | [CeilDiv(K,64)*2, N] | fp8_e8m0 | 全量 MXFP8 scale |
| C | 输出 | [M/rankSize, N] | bfloat16 | ReduceScatter 输出（M-split） |
| workspace | 输入 | PIPELINE_NUM × coreNum × baseM × baseN | bfloat16 | mm 中间输出（host `aclrtMalloc`） |
| tiling | 输入 | - | ApaceQuantMatmulReduceScatterV2TilingData | host 推导 |

**输入校验约束**：M%rankSize==0，K%rankSize==0，ka=K/rankSize 且 ka%64==0，通信 tile 数 ≤ 10（flagId 冲突规避）。

## 架构（v2 3 级流水）

```
AIV (Vector 核, 时分复用)                    AIC (Cube 核)
─────────────────────────                    ─────────────────
Phase 1: AllToAll PUT 通信                   RunMatmul():
  for tid: Commit(scale) + Commit(data)        while BlockScheduler.GetTileIdx():
    → Wait<BARRIER_DEVICE> → SyncAll              回压 WaitFlag<mode2,M>(3) [cubeNum≥4]
    → SetFlag<0x2,MTE3>(tid) ──────────→          通信等待 WaitFlag<0x2,MTE2>(tid)
Phase 2: reduceSum（复用 BlockScheduler，         per-tile gmCWorkspace Tensor
  与 AIC 同一枚举序，DESIGN.md §2.16）             mm: A[Win]×B → L0C FP32 跨 rank 累加
  while bsAiv.GetTileIdx():                        → fixpipe → workspace[BF16]
    WaitFlag<mode2,MTE2>(5) ←────── SetFlag<mode2,FIX>(5)
    DataCopyPad(workspace→UB, 2D)
    [event MTE2→MTE3 同步]
    DataCopyPad(UB→yGm[mPos*N+nPos], 2D)
    SetFlag<mode2,MTE3>(3) ──────→  [回压: slot 可复用]
```

- **PIPELINE_NUM=4** workspace 多 buffer 轮转：AIC mm 可领先 AIV reduceSum 最多 4 个 tile
- **无 AtomicAdd**：L0C 跨 rank FP32 累加后单次 fixpipe 写 workspace，reduceSum 仅 2D DataCopyPad 搬运，yGm 无需清零
- **AIV 复用 BlockScheduler**（§2.16 方案 b）：MIX_AIC_1_1 下 AIC/AIV 的 GetBlockIdx()/GetBlockNum()/GetTaskRatio() 一致，同一 problemShape/schParams 枚举同一 tile 序列，mPos/nPos 由 `GetTileCoord` 推导，`outOffset = mPos*N + nPos`
- **流水同步**：GM→UB(MTE2) 与 UB→GM(MTE3) 之间显式 event 同步（RAW: MTE2_MTE3，WAR: MTE3_MTE2）；UB 行距 64B 对齐 pitch（cpIn/cpOut stride 一致）
- **flagId 冲突规避**：通信就绪 flagId 避开 3（回压）、5（mm 完成）、[11-14]（SyncAll 硬同步）

## 目录结构

```
apace_quant_matmul_reduce_scatter_v2/
├── kernel/apace_quant_matmul_reduce_scatter_v2/   # 算子层（3 个头文件）
│   ├── apace_quant_matmul_reduce_scatter_v2_udma_impl.h    # Impl：RunAllToAll/RunReduceSum/RunMatmul
│   ├── quant_matmul_mx_kernel.h                           # Blaze mm 调度（mm→workspace + flag 编排）
│   └── apace_quant_matmul_reduce_scatter_v2_tiling_data.h # tiling 结构体 + 流水线常量
├── block/ tiling/ basic/ utils/                  # [REUSE] 共享层（禁止修改）
├── st/apace_quant_matmul_reduce_scatter_v2/      # ST 工程（main.cpp + kernel_launcher.h + scripts/）
├── docs/                                         # DESIGN.md / PLAN.md / environment.md / perf/
└── cmake/ third_party/                           # 构建辅助 + ops-tensor
```

## 构建与测试

```bash
source /home/l00947526/Ascend806/cann/set_env.sh
cd st/apace_quant_matmul_reduce_scatter_v2

# 编译
cmake -S .. -B ../build && cmake --build ../build --target apace_quant_matmul_reduce_scatter_v2 -j4

# 精度测试（4 rank，qrun 排队执行）
bash run.sh 1        # cases.csv 第 1 行（T1: M=2048 K=4096 N=2048 e4m3）
bash run.sh all      # 全部 10 行
bash run.sh --cli 4096 8192 4096 4 512 e4m3e4m3   # T4 大 shape

# 性能采集（msprof + L2 flush）
bash run.sh --perf 1
python3 scripts/parse_prof.py --all
```

## 测试结果（2026-08-12）

| 用例 | shape | dtype | 结果 |
|------|-------|-------|------|
| T1 基础精度 | 2048×4096×2048 | e4m3e4m3 | ✅ PASS |
| T2 基础精度 | 2048×4096×2048 | e5m2e5m2 | ✅ PASS |
| T3 混合 dtype | 2048×4096×2048 | e4m3e5m2 / e5m2e4m3 | ✅ PASS |
| T4 大 shape | 4096×8192×4096（tileCnt=2, mTailTile=3 尾块拆分） | e4m3e4m3 | ✅ PASS |
| T5 小 shape | 512×1024×512（baseN=48 非 64B 对齐） | e4m3e4m3 | ✅ PASS |
| 非对齐 | 1120×2048×1024（mLocal=280 尾块） | 全 4 种 dtype | ✅ PASS |
| T6 localMatmul=0 | 默认 REMOTE-only | - | ✅（全部用例均为 localMatmul=0） |

精度标准：bfloat16 rtol=atol=1e-2（小 shape 用例实测 max_err=0 bit 级一致；大 shape 用例 max_abs_err 为 bf16 ulp 量级，rel_err 远小于阈值）。

## 性能（P1: 2048×4096×2048, e4m3, 4 卡）

| 指标 | 值 |
|------|-----|
| Task Duration (v2) | **58.0 us**（msprof task-based，4 卡平均） |
| Task Duration (v1 基线) | 143 us |
| 加速比 | **≈ 2.47×** |
| 主导流水 | cube ≈ 91%（通信/reduceSum 已被流水掩盖） |

数据归档：`docs/perf/round_001/`（summary.txt + parse_output.txt + prof_raw/）

## 参考资料

- 技术设计：docs/DESIGN.md（§2.16 design_issue 解决记录：AIV 复用 BlockScheduler + totalMmTiles 废除）
- 开发计划与调试记录：docs/PLAN.md
- 环境信息：docs/environment.md
