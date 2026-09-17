# AlltoAllQuantMatmul HiF4 特性 ST 测试设计（黑盒 + 白盒）

| 项目 | 内容 |
| :--- | :--- |
| 被测特性 | 47b981b42 `alltoallmatmul新增hif4全量化特性`（Apace Hcomm 通路） |
| 测试框架 | `mc2/common/op_kernel/apace/tests/st/all_to_all_quant_matmul/`（run.sh + main_hif4.cpp 直调工程） |
| 执行入口 | `QUANT_TYPE=hif4 [TRANS_B=false] bash run.sh --csv cases_hif4.csv <N|all>` |
| 验收容差 | `verify_result.py`：HiF4 `rtol=atol=1e-1`（BF16 输出） |
| 关联文档 | HiF4_Apace_Template_Design.md、HiF4_UT_Case_Design.md |
| shape 原则 | 在覆盖前提下取最小：Ka=128（约束下限）、m/n 以 16 倍数覆盖整块/尾块、rank 2/3/4/8 |

## 1. 测试对象与执行框架

### 1.1 工程组成

| 组件 | 说明 |
| :--- | :--- |
| `src/main_hif4.cpp` | ST 直调入口：fork 多进程 → HcclCommInitRootInfoConfig 建域 → 进程内构造 tiling（SWAT + comm tiling）→ 直调 `AllToAllQuantMatmulKernelHiF4HiF4_Hcomm`（CCU，与主线 apt.cpp 同一 `AllToAllMxQuantMatmulHcommImpl<hif4_scale>`） |
| `src/apace_a2a_kernel_launcher.h` | kernel 入口封装，transB 运行时双分支模板实例化 |
| `scripts/gen_data.py` | hif4 模式：生成 S1P2 打包数据 + hif4_scale 三级元数据 + CPU golden（反量化 + AllToAll 模拟 + matmul） |
| `scripts/verify_result.py` | 逐 rank allclose（hif4 tol=1e-1） |
| `cases_hif4.csv` | 用例表：`m,k,n,rank_num,head_m_size` |

### 1.2 执行前提（重要）

1. **devkit/blaze 依赖**：`hif4_scale` 类型、`IsHif4<>()` 分发、BlockMmad HiF4 路径就位后方可编译执行（main_hif4.cpp 头部 NOTE 已声明，当前为静态交付）；
2. **ST 工程前置修复**（白盒分析发现，见 §3.7）：
   - **F1**：`main_hif4.cpp` 通信 tiling 构造使用 `mTotal`（M_total）计算 `headTileCnt/tailMSize`，与主线 `SetHcommTilingInfo`（splitAxis 覆盖 per-rank m）及 AllToAll stride 数学不一致，多卡时会越界（§3.7-R6），执行前需改为 `m / headMSize`、`m % headMSize`；
   - **F2**：`gen_data.py` 需扩展定向数据模式（§4.3）支撑数据级白盒；
3. **观测开关**：`ASCEND_GLOBAL_LOG_LEVEL=0` 可打印 SWAT tiling dump；main_hif4 固定打印 `TileCnt/HeadTileCnt/HeadMSize/TailTileCnt/TailMSize`。

## 2. 黑盒用例设计（接口规格驱动）

### 2.1 等价类与边界分析

| 维度 | 等价类/边界 | 取值 |
| :--- | :--- | :--- |
| Ka | 下限 128；baseK 分档 128/256 | 128、256、512 |
| rankDim | 下限 1；常用 2/4；非 2 幂 3；上限 8 | 1,2,3,4,8 |

> 注：R=3 仅 ST 直调框架可测——aclnn 主线通路受公共校验 `SUPPORT_RANK_SIZE={2,4,8,16}` 限制，非 2 幂卡数在 tiling 层被拒。
| transB | DN / ND | true / false |
| m | 整块（%256=0）；尾块（%256=16）；多块（>256）；最小对齐 16 | 16,144,256,272,512 |
| n | 整块；DN 尾块（%256=16）；ND 尾块（%256=80，ND baseN 对齐 64） | 16,256,272,336 |
| bias / y dtype | ST 直调固定无 bias、BF16 输出；FP16/FP32/bias 经 aclnn 主线补充 | — |
| 异常 | k%R≠0、Ka%128≠0、R>8、参数非法 | §2.3 |

### 2.2 正向用例表（cases_hif4_small.csv 建议）

| 编号 | m | k | n | R | transB | 覆盖目标 | 规模说明 |
| :--- | :-: | :-: | :-: | :-: | :--- | :--- | :--- |
| ST-BB-01 | 256 | 256 | 256 | 2 | true | 基准最小规格：Ka=128 下限，baseM/N/K=256/256/128，单块整算 | x1 32KB / y 128KB |
| ST-BB-02 | 256 | 256 | 256 | 2 | false | ND 布局基准（LayoutB/ScaleBND 路径 + golden 转置重打包） | 同上 |
| ST-BB-03 | 128 | 512 | 128 | 4 | true | 4 卡（K 4 段拼接） | x1 32KB |
| ST-BB-04 | 272 | 256 | 256 | 2 | true | M 尾块（272=256+16，mBlockCnt=2，mTail=16） | y 136KB |
| ST-BB-05 | 256 | 256 | 272 | 2 | true | N 尾块（DN，nTail=16） | y 136KB |
| ST-BB-06 | 256 | 256 | 336 | 2 | false | ND N 尾块（baseN 对齐 64 → 256，nTail=80） | y 168KB |
| ST-BB-07 | 128 | 384 | 256 | 3 | true | 3 卡（非 2 幂 rank，Ka=128） | x1 24KB |
| ST-BB-08 | 64 | 1024 | 128 | 8 | true | 8 卡上限（8 段 K 拼接，scale 通信量上限） | x1 32KB |
| ST-BB-09 | 512 | 256 | 256 | 2 | true | 多 M 块（mBlockCnt=2）+ fullMTile=false（L2 hint 不设置分支） | y 256KB |
| ST-BB-10 | 128 | 512 | 256 | 2 | true | Ka=256 → baseK=256 分档（min(Ka,256)） | x1 32KB |
| ST-BB-11 | 128 | 1024 | 256 | 2 | true | Ka=512 → baseK=256 clamp + stepK 多轮 K 深度循环 | x1 64KB |
| ST-BB-12 | 16 | 256 | 16 | 2 | true | 极小块（baseM=16 最小对齐；通信 head-only，见 WB-03） | y 512B |
| ST-BB-13 | 16 | 128 | 16 | 1 | true | 单卡退化（AllToAll R=1 自收；可选边界） | y 512B |
| ST-BB-14 | 2048 | 256 | 2048 | 2 | true | 中规模通信 head+tail 复合（与 WB-05 共用） | y 8MB |
| ST-BB-15 | 4040 | 8192 | 4096 | 4 | true | 大规格冒烟（保留原 cases_hif4.csv 代表性条目） | 可选 |

> head_m_size 列：main_hif4 当前仅打印不生效（自算 headMSize），填 512 占位；该列生效性随 F1 修复一并复核。

### 2.3 异常参数用例（参数校验层，预期报错退出）

| 编号 | 构造 | 预期 |
| :--- | :--- | :--- |
| ST-BB-E01 | m=128, k=320, R=2（k%R=160≠0） | main_hif4：`k must be divisible by rankNum`（非 0 退出） |
| ST-BB-E02 | m=128, k=384, R=2（Ka=192，%128≠0） | main_hif4：`Ka (k / rankNum) must be 128-aligned` |
| ST-BB-E03 | R=9 | main_hif4：`rankNum <= 8` |
| ST-BB-E04 | transB=yes（非法值） | main_hif4：`transB must be 'true' or 'false'` |
| ST-BB-E05 | mode=foo | main_hif4：`mode must be 'precision' or 'perf'` |
| ST-BB-E06 | m=0 / k=-1 / n=0 | main_hif4：`must be positive` |
| ST-BB-E07 | gen_data 单独调用同 E01/E02 形态 | gen_data.py：`ValueError: K must be divisible` / `Ka must be 128-aligned` |

### 2.4 aclnn 主线补充用例（ST 直调不覆盖的规格维度）

ST 直调工程固定 bias=nullptr、输出 BF16；以下维度经 aclnn 主线（960dt 环境）补测，shape 复用 ST-BB-01/02：

| 编号 | 变更点 | 覆盖 |
| :--- | :--- | :--- |
| ST-BB-A01 | bias=[256] FP32 | bias L1 预算/epilogue 加法路径（isBias_=true） |
| ST-BB-A02 | y=FLOAT16；y=FLOAT32 | 输出 dtype 三支持 |
| ST-BB-A03 | x2 传 .t() 非连续视图 + transposeX2=false | op_api 自动转连续 + transposeX2 置 true |
| ST-BB-A04 | ND 布局 aclnn 主线 | ⚠ 已知问题 R1：host tiling 会误拒，暂不执行，修复后启用 |

### 2.5 验收标准

- 全部正向用例：所有 rank `allclose(rtol=atol=1e-1)` PASS；
- 异常用例：报错信息匹配且退出码非 0；
- perf 模式（可选）：输出 per-iter 统计无异常（不设性能门槛，本特性无性能指标承诺）。

## 3. Apace Hcomm 实现详细分析（白盒用例设计基础）

以下逐层分析 `AllToAllMxQuantMatmulHcommImpl<hif4_scale>` 及其调用的 `AllToAllQbmmMxKernel` 的可执行分支。**所有结论均以代码为据**，标注源码位置。

### 3.1 通信数据通路（Init，hcomm_impl.h:170-235）

**维度约定**（推导依据：AllToAll stride 数学 + URMA 对照 + host `SetHcommTilingInfo`/`AlignCutResForKernel`（tileM ≤ mValue=m））：

```
m            = 每卡输出行数（x1 行数 M_total = m×R）
Ka           = x1 列数 = K/R（nonSplitAxisSize/R）
splitAxis    = 通信切分轴，覆盖 m（每目标卡分片），splitAxisTileSize×Cnt + TailSize×Cnt ≡ m
headTileSize = splitAxisTileSize（m 单位）
```

**workspace 布局**（从 workspaceGM 起始）：

```
[ scale 收包区 x1ScaleLen = AlignUp(M_total×Ka/64×4B, 512) | 数据收包区 commDataLen = M_total×Ka/2 B ]
（all2all_out 非空时数据区改用 all2all_out——HiF4 被 tiling 拒绝，恒走 workspace）
```

**三个 AllToAll 任务**（HCCL kernel API：block j 位于 sendBuf 偏移 j×strideCount，发往 rank j；recv 侧来自 rank i 的块落在 i×strideCount）：

| handle | 下发条件 | dataCount（块大小） | strideCount（目的块间距） | repeat |
| :--- | :--- | :--- | :--- | :--- |
| scaleHandle | 恒下发 | `m×Ka/64` 个 hif4_scale（代码传 `scaleSendCount×sizeof` 字节，**单位歧义见 R5**） | `m×Ka/64`（scale 元素） | 1 |
| dataHeadHandle | splitAxisTileCnt>0 | `tileSize×Ka/2` B（UINT8，2:1 打包） | `m×Ka/2` B | tileCnt |
| dataTailHandle | splitAxisTailCnt>0 | `tailSize×Ka/2` B | `m×Ka/2` B | tailCnt |

send 侧块 j = x1 行 `[j×m + t×tileSize, +tileSize)`（目的 j 的 m 分片按 tile 逐段发送）；recv 侧布局 `[R, m, Ka]`（源 i 的贡献在行 `i×m` 起）——与 kernel `layoutA=(R×m, Ka)`、`actualMPos=rank×m+mPos` 精确对应。

**HiF4 相对 MX 的泛化点**：

1. scale 通信类型 `UINT32`（MX: FP8E8M0）——hif4_scale 为 4 字节完整元数据；
2. `scaleKPerRank = MxScaleKLen<hif4_scale>(Ka) = Ka/64`（MX: Ka/64×2）；
3. 数据 2:1 打包：`isHiF4Scale_` 判定（不依赖 IsMxFp4 模板参数），`rankForComm = CeilDiv(Ka,2)` 字节/行。

### 3.2 等待与同步（HcommCommWaitPolicy + qbmm kernel）

```
WaitTile(tileIdx):
    tileIdx == 0          → Wait(scaleHandle)        // scale 仅首个 tile 等待一次
    tileIdx < headTileCnt → Wait(dataHeadHandle)
    else                  → Wait(dataTailHandle)
```

- 计算侧 `CalcDependTileIdx = min((mPos+blockM-1)/headTileSize, totalTiles-1)`，`while (readyTileIdx < dependTileIdx)` **去重等待**（同一 tile 只 Wait 一次）；
- `totalTiles = CeilDiv(m, headTileSize)`（oriM=m）；
- **尾核兜底**：调度块耗尽后 `while (readyTileIdx < totalTiles-1) WaitTile(++readyTileIdx)`——防止未消费的通信 tile 阻塞 `Finalize`；
- `Run()`：`localMatmul==0` → 仅 `MatmulProcess(REMOTE)` → `SyncAll` → `Finalize`。

### 3.3 计算分支（qbmm ProcessSingleBatch，REMOTE 主体）

```
for (mPos,nPos) in 调度块:                       // BlockSchedulerQuantBatchMatmulV3
    空块保护: singleShape M/N ≤ 0 → return
    去重等待 (§3.2)
    for rank in [0,R):
        actualMPos = rank×m + mPos
        rank==self: A 块与 scaleA 块切到 gmALocal/gmScaleALocal（本卡输入 GM 短路, localMatmul==2 才跳过）
        rank!=self: 切到 gmA/gmScaleA（通信收包区）
        B 块 = gmB[rank×Ka, nPos]（本卡 x2 全量 K）；scaleB 块 = gmScaleB[rank×Ka/64, nPos]
        mmadOp_(..., remoteRankCnt++)             // L0C 逐源累加，末次 fixpipe
```

- `splitKNum = R`（localMatmul=0）；
- 布局：`LayoutB = TransB ? DN : ND`；scale framing `SCALE_C0_A/B = MxScaleC0 = 1`（HiF4）；
- scaleKLen = Ka/64（HiF4 单元数）；
- `kCacheLineAlignMask = 0x7f`（hifloat4x2_t 非FP4 → 128B 判据）。

### 3.4 SWAT tiling 决策树（ST 侧进程内构造，与 host DoHcommMxQuantMMTiling 一致）

```
baseM = Align(min(m,256), 16)
baseN = transB ? Align(min(n,256),16) : Align(min(n,256),64)     // HiF4 ND 对齐 64
baseK = Align(min(Ka,256), 64)                                    // 4-bit 档 256
mBlockCnt/nBlockCnt/totalBlockCnt/tailBlockCnt →
    usedCoreNum = (totalBlockCnt>1 || tailBlockCnt==0) ? aicNum : tailBlockCnt×mTailTile×nTailTile
    （mTailTile/nTailTile 由 CalcTailBasicBlock 在 aicNum 预算内生长，上限 ⌈baseM/16⌉）
dbL0c = baseM×baseN×8 ≤ l0cSize ? 2 : 1
depthA1/B1 → stepK(≤4) → scaleFactor → scaleKL1 = min(sfA×stepKa×baseK, sfB×stepKb×baseK)
headMSize(ST) = CeilDiv(usedCoreNum, CeilDiv(n,baseN)) × baseM   // m 单位
通信 tile: headTileCnt = m/headMSize, tailMSize = m%headMSize, tailCnt = tailMSize?1:0
```

三种通信形态（**均可在小 shape 下构造**）：

| 形态 | 条件 | handle 组合 |
| :--- | :--- | :--- |
| head-only | m 整除 headMSize 且 headMSize≤m | scale + dataHead |
| head+tail | headMSize<m 且 m%headMSize≠0 | scale + dataHead + dataTail |
| tail-only | headMSize>m | scale + dataTail |

### 3.5 L2 cache 分支（SetL2Cache/SetScaleL2Cache，qbmm）

生效前提：`fullMTile = baseM ≥ m`（整 M 块）且 B 维=1：

| 分支 | 判据（128B） | HiF4 可构造性 |
| :--- | :--- | :--- |
| DN scale DISABLE | `(K/64)×4B % 128==0 且 scaleKL1 组×4B %128==0` → K%2048==0 | K=2048（WB-16） |
| DN scale NORMAL | 上述不满足 | K=256：16B 不对齐（WB-01） |
| ND scale DISABLE | `N×4B %128==0 且 baseN×4B %128==0` → N%32==0 | N=256（WB-02） |
| ND scale NORMAL | N%32≠0 | N=336（WB-17） |
| DN B DISABLE/NORMAL | K%128 | **HiF4 恒 DISABLE**（K=Ka×R=128R ⇒ K%128==0 恒真），NORMAL 分支对本特性不可达 |
| ND B DISABLE/NORMAL | N%128 且 baseN%128 | N=256 → DISABLE；N=336 → NORMAL |
| gmC DISABLE | isAtomicAdd_ | 不可达（localMatmul=0） |

### 3.6 分支清单与可达性汇总

| # | 分支/决策点 | 位置 | 主线可达性 | 覆盖用例 |
| :-: | :--- | :--- | :--- | :--- |
| 1 | GetX1HcclDataType→UINT8（hifloat4x2_t） | hcomm_impl:56 | ✓ | 全部 |
| 2 | mxScaleHcclDataType_=UINT32（IsHif4） | hcomm_impl:148 | ✓ | 全部 |
| 3 | isHiF4Scale_ 2:1 打包 rankForComm | hcomm_impl:204 | ✓ | 全部 |
| 4 | all2all_out==nullptr → workspace 数据区 | hcomm_impl:193 | ✓（非空被 tiling 拒） | 全部 |
| 5 | splitAxisTileCnt>0（dataHead 下发） | hcomm_impl:218 | ✓ | WB-03/05 |
| 6 | splitAxisTileCnt==0（不下发） | hcomm_impl:218 | ✓ | WB-04 |
| 7 | splitAxisTailCnt>0（dataTail 下发+headOffset） | hcomm_impl:225 | ✓ | WB-04/05 |
| 8 | splitAxisTailCnt==0（不下发） | hcomm_impl:225 | ✓ | WB-03 |
| 9 | WaitTile tileIdx==0 → scale wait | hcomm_impl:79 | ✓ | 全部 |
| 10 | WaitTile <headTileCnt → dataHead wait | hcomm_impl:82 | ✓ | WB-03/05 |
| 11 | WaitTile else → dataTail wait | hcomm_impl:84 | ✓ | WB-04/05 |
| 12 | 去重 while（readyTileIdx<dependTileIdx） | qbmm:490 | ✓（多 tile 时多次迭代） | WB-05 |
| 13 | 尾核兜底 while | qbmm:531 | ✓ | WB-04/05 |
| 14 | REMOTE self-rank gmALocal/gmScaleALocal 短路 | qbmm:507 | ✓ | 全部 |
| 15 | REMOTE 远端 rank 收包区读取 | qbmm:497 | ✓ | R≥2 用例 |
| 16 | 单卡退化（R=1 仅 self） | qbmm:494 | ✓ | WB-12 |
| 17 | 空块保护 singleShape≤0 → return | qbmm:414 | 防御性（正常调度不可达） | 代码检视 |
| 18 | needUpdateTail_/UpdateTailTile | qbmm:395 | ✓ | WB-06/07 |
| 19 | fullMTile=true 设置 L2 hint | qbmm:289 | ✓ | m≤256 用例 |
| 20 | fullMTile=false 不设置 | qbmm:292 | ✓ | WB-05/BB-09 |
| 21 | DN/ND scale·B L2 四分支 | qbmm:257-276 | ✓（除 DN-B-NORMAL 恒 DISABLE） | WB-01/02/16/17 |
| 22 | LayoutB DN/ND、ScaleBDN/BND、SCALE_C0=1 | qbmm:106-129 | ✓ | WB-01/02 |
| 23 | baseK 128/256 分档 + stepK 多轮 | swat:312 | ✓ | BB-01/10/11 |
| 24 | baseN DN(16)/ND(64) 对齐 | swat:309 | ✓ | BB-01/02 |
| 25 | M/N 尾块分裂 CalcTailBasicBlock/OptimizeEdge | swat:119/353 | ✓ | WB-06/07 |
| 26 | dbL0c 1/2 | swat:334 | ✓（依 l0cSize，大 baseM×baseN 触发 1） | 大块冒烟 |
| 27 | LOCAL 先行 / DEFERRED_SYNC / localMatmul==2 跳 self / isAtomicAdd / gmC DISABLE | qbmm:320/365/439/504/285 | ✗ 保留能力（主线 localMatmul=0） | 不设计（激活时补） |

### 3.7 实现缺陷与风险发现（白盒分析产出）

| 编号 | 发现 | 分析 | 处置建议 |
| :--- | :--- | :--- | :--- |
| **R5** | **scale AllToAll dataCount 单位歧义**（hcomm_impl:214-216）：HiF4 传 `scaleSendCount×sizeof(hif4_scale)` 作为 dataCount、`scaleSendCount` 作为 strideCount。若 HCCL 语义按元素计数（头文件 fp16 示例：4 rank×4 fp16 → dataCount=1），则 dataCount 4 倍超发；若按字节计数（cBuffSize 风格）则正确。MX 路径 E8M0（1 字节）两约定等价，无法在存量用例区分 | 多卡场景 scale 通路错乱 → 精度失败或越界写 | ① 与 HCCL 确认 dataCount 单位；② 以 WB-SCALE 定向用例（rank 标签 scale）在 2 卡实测；③ 若确认按元素，修正为 `scaleSendCount`（stride 单位同查） |
| **R6** | **ST main_hif4.cpp 通信 tiling 单位错误**：`headTileCnt = mTotal/headMSize`、`tailMSize = mTotal%headMSize` 使用 M_total；而主线 `SetHcommTilingInfo` 的 splitAxis 覆盖 per-rank m（`AlignCutResForKernel` 以 mValue=m 封顶；AllToAll stride=`splitAxis×Ka/2` 必须=`m×Ka/2`；URMA 实现同以 `axisM=m` 传入 LocalParams）。splitAxis=M_total 时 dataStrideCount=整缓冲区，目的块 j≥1 偏移越界，recv 侧 2 倍溢出 | 多卡 ST 执行必然失败（因 devkit 依赖当前未执行而未暴露） | 修复为 `m/headMSize`、`m%headMSize`（F1）；修复后 WB-03/04/05 的形态断言才成立 |
| R1（回顾） | host tiling `CheckMxQuantShapeInfo` ND 布局 x2Scale 校验假定 [N,K]，与 op_api/kernel 的 [K/64,N,1] 约定冲突 → aclnn ND 通路误拒 | 详见设计文档 §10 | ST 直调 ND 用例（BB-02/06、WB-02/17）不受影响；aclnn ND（ST-BB-A04）待修复后启用 |
| R2（回顾） | devkit/blaze HiF4 分发未就位，ST 为静态交付 | — | 版本对齐后先跑 WB-SCALE 与 WB-01/02 冒烟 |

## 4. 白盒用例设计

### 4.1 设计原则

- 每条用例锚定 §3.6 分支清单中的具体编号，一例多分支时逐一列出；
- shape 取触发该分支的最小值；通信形态（head/tail）依赖 `headMSize = CeilDiv(usedCoreNum,nTile)×baseM`，与 aicNum 相关——用例给出**形态断言方法**（读 INFO_LOG）与 aicNum∈[33,64] 区间内的形态鲁棒性说明；
- 全部用例复用黑盒执行框架（同一 CSV，`QUANT_TYPE=hif4`）。

### 4.2 白盒用例表

| 编号 | m | k | n | R | transB | 目标分支（§3.6#） | 形态/日志断言 |
| :--- | :-: | :-: | :-: | :-: | :--- | :--- | :--- |
| WB-01 | 256 | 256 | 256 | 2 | true | #1-4,9,14,15,19,21(DN scale NORMAL),22(DN),23(baseK=128) | 精度 PASS；tail-only（headMSize≫m） |
| WB-02 | 256 | 256 | 256 | 2 | false | #21(ND scale/B DISABLE),22(ND/C0=1),24(ND 对齐 64) | 精度 PASS |
| WB-03 | 16 | 256 | 16 | 2 | true | #5,8(head-only),10；最小块 baseM/N=16 | 日志 `HeadTileCnt≥1 且 TailTileCnt=0`（headMSize=baseM=16 时 HeadTileCnt=1；若 usedCoreNum>1 则 headMSize=usedCoreNum×16，仍 <m=16 仅当 usedCoreNum=1——**依赖核数，以日志校准**） |
| WB-04 | 256 | 256 | 256 | 2 | true | #6,7,11(tail-only),13(兜底) | 日志 `HeadTileCnt=0 且 TailTileCnt=1`（headMSize>CeilDiv(aicNum,1)×256 时成立，aicNum≥3 即满足） |
| WB-05 | 2048 | 256 | 2048 | 2 | true | #5,7,10,11,12,13(head+tail),15,20(fullMTile=false),18(多块) | 日志 `HeadTileCnt≥1 且 TailTileCnt=1`（headMSize=⌈aicNum/8⌉×256，aicNum∈[33,120] 内形态稳定） |
| WB-06 | 272 | 256 | 256 | 2 | true | #18,25(M 尾块 mTail=16) | 精度 PASS；tiling dump mTailTile≥1 |
| WB-07 | 256 | 256 | 272 | 2 | true | #18,25(N 尾块 DN nTail=16) | 精度 PASS |
| WB-08 | 128 | 1024 | 256 | 2 | true | #23(baseK=256 clamp + stepK 多轮)，Ka=512 | 精度 PASS；tiling dump stepK≥1、scaleKL1≤512 |
| WB-09 | 128 | 512 | 256 | 4 | true | #15(4 源遍历)，splitKNum=4，scale SendCount=m×Ka/64 | 精度 PASS |
| WB-10 | 64 | 1024 | 128 | 8 | true | #15(8 源)，rankDim 上限 | 精度 PASS |
| WB-11 | 128 | 384 | 256 | 3 | true | #15(非 2 幂 rank) | 精度 PASS |
| WB-12 | 16 | 128 | 16 | 1 | true | #16(单卡退化，splitKNum=1，仅 self 短路) | 精度 PASS（可选） |
| WB-16 | 256 | 2048 | 256 | 2 | true | #21(DN scale DISABLE：K/64×4=128B 对齐) | 精度 PASS；确认 scaleKL1 行对齐性（不满足则仍 NORMAL，记录实际分支） |
| WB-17 | 256 | 256 | 336 | 2 | false | #21(ND scale/B NORMAL：N×4=1344B 非对齐) | 精度 PASS |
| WB-SCALE | 256 | 256 | 256 | 2 | true | **R5 定向验证**：gen_data scale 填 rank 标签（rank r 的全部 hif4_scale 填 E6M2=r+30、E1_8/E1_16=0），golden 同步 | 若 scale 通路错乱（超发/错位）→ 精度 FAIL 且错误模式与 scale 布局强相关；PASS 则 R5 按字节语义成立 |

> 说明：#17（空块保护）与 #27（保留能力分支）无用例，见 §3.6；#26（dbL0c=1）依赖 l0cSize 与 baseM×baseN，256×256×8B=512KB，960dt l0cSize 若 ≥512KB 则 dbL0c=2 恒成立，该分支由大块冒烟（ST-BB-15）观察。

### 4.3 定向数据用例（数据级白盒，需 F2 扩展 gen_data.py）

在 gen_data.py 增加 `--pattern` 参数（默认 random，兼容既有调用）：

| pattern | 构造 | 目标 |
| :--- | :--- | :--- |
| `pure_e6m2` | E1_8=0、E1_16=0，E6M2 正常随机 | 三级缩放中两级为 0 的退化：V=E6M2×S1P2 |
| `e6m2_boundary` | E6M2 指数 ∈ {30,31,32}、尾数 ∈ {0,3}（避开 em=255 NaN） | E6M2 解码边界（2^±1 档、尾数满/零） |
| `s1p2_zeros` | S1P2 码值全 0（数据全 0） | 零数据（acc=0，验证无 NaN 污染） |
| `s1p2_neg_all` | 符号位全 1、码值遍历 8 档 | 负数域 + 打包偶低奇高逐位正确性 |
| `s1p2_maxmag` | 码值全 6 + E1_16 全 1 | 幅值上界（E6M2(32 档)×2^16×6 ≈ 3.1e5，BF16 可表示） |
| `scale_rank_tag` | rank r 的 scale 全部填 E6M2=2^(r+30)、E1 置 0；数据正常随机 | R5 通信正确性定向（WB-SCALE） |

对应用例（复用 WB-01 shape，`m=256,k=256,n=256,R=2,transB=true`）：

| 编号 | pattern | 验证点 |
| :--- | :--- | :--- |
| WB-D01 | pure_e6m2 | E1_8/E1_16 全 0 位选择路径 |
| WB-D02 | e6m2_boundary | E6M2 指数/尾数边界解码 |
| WB-D03 | s1p2_zeros | 全 0 数据（含 0×scale、0 输出行） |
| WB-D04 | s1p2_neg_all | 符号位与码值表全遍历、2:1 打包位序 |
| WB-D05 | s1p2_maxmag | 组内动态范围上界（E1_16 全 1 × 最大码值） |
| WB-SCALE | scale_rank_tag | scale 通信通路（R5） |

### 4.4 观测与断言手段

| 手段 | 用途 |
| :--- | :--- |
| `INFO_LOG`（main_hif4 固定输出） | `TileCnt/HeadTileCnt/HeadMSize/TailTileCnt/TailMSize` → 通信形态断言（WB-03/04/05） |
| `ASCEND_GLOBAL_LOG_LEVEL=0` | SWAT `PrintTilingData` dump（baseM/N/K、stepK、scaleKL1、mTailTile） |
| `verify_result.py`（tol=1e-1） | 逐 rank 精度断言；FAIL 时打印 abs/rel err 分布与角点数据，用于 R5 错误模式定位 |
| `mode=perf` | per-iter 耗时（avg/min/max），通信形态变化对耗时的影响佐证 |
| 退出码 + stderr | 异常用例（§2.3）断言 |

## 5. 用例汇总与执行顺序

| 类别 | 数量 | 编号段 |
| :--- | :-: | :--- |
| 黑盒正向（ST 直调） | 15 | ST-BB-01..15 |
| 黑盒异常 | 7 | ST-BB-E01..E07 |
| aclnn 主线补充 | 4（1 条暂缓） | ST-BB-A01..A04 |
| 白盒分支用例 | 14 | WB-01..12,16,17 |
| 白盒定向数据 | 6 | WB-D01..05 + WB-SCALE |
| **合计** | **46** | — |

**执行顺序**：

```
0. 前置：devkit 就位 → F1(ST 修复) → 编译冒烟（BB-01）
1. R5 定向：WB-SCALE（2 卡）→ 结论驱动是否修复 scale AllToAll
2. 分支主链：WB-01 → WB-02（ND）→ WB-03/04/05（通信三形态）
3. 尾块/多卡/多 K：WB-06..12 → WB-16/17（L2 分支）
4. 定向数据：WB-D01..D05
5. 黑盒全量：ST-BB-01..15 + 异常 E01..E07
6. aclnn 补充：A01..A03（A04 待 R1 修复）
7. 回归：原 cases_hif4.csv 7 条大规格（含 ST-BB-15）
```

**通过标准**：全部正向/白盒用例精度 PASS（tol=1e-1）+ 形态日志断言匹配 + 异常用例报错匹配；R5/R6 两个发现闭环（确认语义/修复验证）。
