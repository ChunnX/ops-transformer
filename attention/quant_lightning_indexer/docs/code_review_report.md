# QLI 算子代码检视报告

**检视日期**: 2026-06-12  
**检视范围**: `op_host` 和 `op_kernel` 中非 arch22 部分的所有代码  
**检视方法**: 三轮深度检视，每轮 4 个独立 Agent 并行检视，交叉验证后汇总  
**文件总数**: 17 个文件，约 6,623 行代码

---

## 检视历程

| 轮次 | 检视范围 | Agent 分工 | 结果 |
|------|---------|-----------|------|
| **第一轮** | 全量代码扫描 | op_host / common+入口 / arch35核心 / vf TopK | 发现 28 个问题（8高/10中/10低） |
| **第二轮** | 聚焦 MXFP8 完整路径 | service_cube / kernel.h / service_vector / common+tiling+vf_topk | 新发现 5 个（N1/N2/C1/A1/P1），推翻 H2、M1 |
| **第三轮** | 验证 + 边界条件 + 同步追踪 | N1验证+TopK数据流 / C1验证+buffer审计 / A1验证+Cube→Vector流 / kernel.h边界+跨核同步 | 推翻第二轮全部 5 个，新发现 3 个待确认项（D1/D2/D3） |

### 问题状态统计

| 类别 | 数量 | 说明 |
|------|:----:|------|
| ✅ 确认存在问题 | **21** | 经验证确实需要修复或关注 |
| ⚠️ 待确认硬件语义 | **3** | D1/D2/D3，需结合实际硬件行为判定 |
| ❌ 已推翻误报 | **7** | H2/M1/N1/N2/C1/A1/P1+P2 |

---

## 一、最终确认问题清单

### 🔴 高严重度（6 个确认 + 1 个待确认）

#### H1 [仅非MXFP8] kScale ping-pong 读写不匹配

- **文件**: `kernel.h:661-665` + `service_vector.h:387-394,423`
- **影响 MXFP8**: ❌ 不影响（MXFP8 路径 Vector 核完全跳过 kScale，`if constexpr (!IS_MXFP8)` 守护）
- **问题描述**: `kScaleLoop` 基于绝对 s2Idx 位置递增（`(s2LoopIdx - s2Start) % 16 == 0` 时自增），而 `ProcessVec1` 中读取 kScale 的索引基于相对 s2Start 的位置。当 SplitCore 分配给某核的 `s2Start` 不是 16 的倍数时，写入和读取使用不同的 ping-pong buffer，导致读取过期数据。
- **复现**: s2Start=5, s2Idx=5 → kScaleLoop=1 → 写入 ping-pong **1**，但 (5-5)%16=0 → 读取 ping-pong **0**
- **建议**: 将 `kScaleLoop` 改为基于 `(s2LoopIdx - s2Start) / 16` 计算，或在 ProcessVec1 中使用 `runInfo.kScaleLoop` 推导读取偏移

---

#### H3 [影响MXFP8] 16-bit TopK 缓冲区溢出 1024 字节

- **文件**: `quant_lightning_indexer_topk.h:100-108`
- **影响 MXFP8**: ✅ 影响（MXFP8 使用 `LITopk<uint16_t>`）
- **问题描述**: `GetSharedTmpBufferSize` 中 `reUseBufferSize = Align(topK, 256) * sizeof(uint32_t)` 被从总缓冲区大小中减去，意图让 `tmpIndexLocal` 与 `hisIndexLocal2` 共享内存。但 `InitBuffers`(L127-128) 中 `tmpIndexLocal` 定位在 `nkValueLocal[64]` 之后（远在 hisIndexLocal2 之后），并不与其重叠。
- **数值验证** (topK=512, trunkLen=16384):
  - 缓冲区容量：9472 × 2 = **18944 uint16_t**
  - tmpIndexLocal 结束位置：2560 + 16896 = **19456 uint16_t**
  - **溢出 512 uint16_t = 1024 字节**
- **建议**: 移除 `reUseBufferSize` 减法；或将 `tmpIndexLocal` 重新定位到 `hisIndexLocal2` 起始位置以实现真正的内存复用

---

#### H4 [影响MXFP8] layout_key 校验逻辑反转

- **文件**: `quant_lightning_indexer_tiling.cpp:204-208`
- **影响 MXFP8**: ✅ 影响（PA_BBND 布局的 MXFP8 用例会被错误拒绝）
- **问题描述**: 当前逻辑拒绝 `BNSD` 和 `PA_BBND`，接受其他所有值。但 PA_BBND 在 layoutKeyMap(L341) 中是合法映射，却被错误拒绝。同时任意非法字符串可通过校验。
- **建议**: 改为正向白名单校验：layout_key 必须是 BSND/TND/PA_BSND/PA_BBND 之一

---

#### H5 [不影响MXFP8] NPU 架构硬编码为 DAV_3510

- **文件**: `quant_lightning_indexer_tiling.cpp:105`
- **影响 MXFP8**: ❌ 不影响（MXFP8 本身就是 DAV_3510 特性，硬编码值与实际一致）
- **问题描述**: `npuArch_` 硬编码为 DAV_3510，未从 platform_ascendc 实际获取。在 910B 等非 DAV_3510 平台上走错误的校验分支。
- **建议**: 从 `platform_ascendc::PlatformAscendC` 获取实际 NPU 架构

---

#### H6 [不影响MXFP8] TilingData n2Size 字段未设置

- **文件**: `quant_lightning_indexer_tiling.cpp:950-961`
- **影响 MXFP8**: ❌ 不影响（kernel 侧未读取该字段）
- **问题描述**: QLITilingData 有 n2Size 字段，但 DoTiling 中从未 set_n2Size()，buffer 中包含未初始化值。
- **建议**: 添加 `tilingData_.set_n2Size(tilingInfo->n2Size)`

---

#### H7 [影响MXFP8] stride 属性空指针解引用

- **文件**: `quant_lightning_indexer_tiling.cpp:168-169`
- **影响 MXFP8**: ✅ 影响（若属性缺失则 tiling 阶段崩溃）
- **问题描述**: `attrs->GetAttrPointer<int64_t>()` 可能返回 nullptr，代码直接解引用 `*(...)` 未做检查。
- **建议**: 先获取指针并判空再解引用

---

#### ⚠️ D1 [影响MXFP8，待确认] CrossCoreSetFlag 首次迭代死锁风险

- **文件**: `kernel.h:642-645`
- **严重程度**: 高（潜在死锁）
- **影响 MXFP8**: ✅ CVRATIO=2 时存在风险
- **问题描述**: 所有 AIV 核（包括 AIV0+AIV1）均执行 `CrossCoreSetFlag(CROSS_VC_EVENT + 0/1)`，即只设置 event ID=0 和 1。但 AIC 核等待事件取决于 BlockId 关系，首次迭代时可能存在配对缺口。
- **前提假设**:
  - 如果 MODE4 (QLI_SYNC_MODE4) 具有"per-pair 广播"语义（AIC-AIV 配对内自动路由），则 event 命名空间是 per-pair 私有的，D1 **不存在**
  - 如果 MODE4 是全局 flat 语义（所有核共享统一事件 ID 空间），则需要按 pair 隔离事件，D1 **存在**
- **建议**: 以硬件文档或实测为准。若非 MXFP8 路径在 CVRATIO=2 下已稳定运行，大概率是 per-pair 语义

---

### 🟡 中严重度（8 个确认 + 2 个条件性）

#### M2 FloatToSortableKey NaN 检测不完整

- **文件**: `quant_lightning_indexer_vector1.h:85-88`（及 float 版本）
- **影响 MXFP8**: ⚠️ 低风险
- **问题描述**: 仅比较 canonical NaN（bf16=0x7FC0, float=0x7FC00000），不覆盖 signaling NaN 等其他位模式。
- **风险评估**: Ascend 硬件通常产出 canonical NaN，实际触发概率极低

---

#### M3 [死代码] s2BasicBlock_==256 分支缺失 MXFP8 scale 加载

- **文件**: `service_cube.h:264-351`
- **影响 MXFP8**: ❌ 当前不影响（isMXFP8 恒等于 false 在该分支，因为 arch35 MXFP8 固定 s2BasicBlock_=128）
- **建议**: 添加编译期断言 `static_assert(s2BasicBlock_ != 256 || !IS_MXFP8)`

---

#### M4 int64_t → uint32_t 静默截断

- **文件**: `quant_lightning_indexer_tiling.cpp:952,955-956`
- **影响 MXFP8**: ⚠️ 当前安全（stride=blockSize×(headDim/32)=blockSize×4，uint32_t 足够）
- **问题描述**: keyDequantScaleStride0 在 tiling.h 定义为 uint32_t，但实际 stride 可能需要更大范围。
- **建议**: 评估是否需要扩展为 uint64_t

---

#### M5 Workspace uint32_t 乘法溢出风险

- **文件**: `quant_lightning_indexer_tiling.cpp:921-923`
- **建议**: 中间计算使用 uint64_t

---

#### M6 ConstInfo 关键字段未初始化

- **文件**: `quant_lightning_indexer_common.h:141-150`
- **问题描述**: kHeadNum、headDim、keyStride0、keyDequantScaleStride0、sparseCount、outputLayout 未提供默认值

---

#### M8 [影响MXFP8性能] 遗留 DumpTensor 调试代码（Cube 核 6 处）

- **文件**: `service_cube.h:549-551, 595-597, 671`
- **影响 MXFP8**: ✅ 每次 L0 循环均执行 DumpTensor，严重影响性能
- **代码示例**: `DumpTensor(queryL1_[...], 1003, 128*128, shapeInfo1)`
- **建议**: 删除或用 `#ifdef QLI_DEBUG` 包裹

---

#### M9+M10 [仅MXFP8触发] 遗留 printf + DumpTensor（Vector 核）

- **文件**: `service_vector.h:410-413`
- **影响 MXFP8**: ✅ **仅在 IS_MXFP8=true 时触发**
- **代码**: `printf("HHHHHHH mx\n")` + `DumpTensor(qkBase, 1001, 128*128, shapeInfo1)`
- **建议**: 删除

---

#### ⚠️ D2 [条件性] MXFP8_SCALE_EXTRA 空间硬编码

- **文件**: `service_cube.h:76`
- **问题描述**: MXFP8_SCALE_EXTRA=512 bytes，恰好够 128 tokens × 4 scales/token = 512 bytes。当 gSize > 64（mBaseSize > 128）时不够用。
- **当前评估**: tiling 层保证 gSize ≤ 64，因此**当前参数下安全**
- **建议**: 改为动态计算 `CeilDiv(mBaseSize, MXFP8_BLOCK_SIZE) * FP8_TWO * sizeof(K_T)`

---

#### ⚠️ D3 [待确认] Mmad n 参数未 CeilAlign 到 BLOCK_CUBE

- **文件**: `service_cube.h:616`
- **问题描述**: m 维度已对齐到 BLOCK_CUBE=16，但 n 维度直接用 s2L0RealSize（原始 token 数）。mmad_mx 对 n 的对齐要求未知。
- **建议**: 参考 mmad_mx ISA 规格确认

---

### 🔵 低严重度（10 个）

| # | 问题 | 文件 | 影响 MXFP8 |
|---|------|------|:----------:|
| H8 | KERNEL_TASK_TYPE_DEFAULT 注释写 "1:1" 实际是 1:2 | quant_lightning_indexer.cpp:48 | ❌ |
| M7 | MXFP8 偏特化允许 W_T=float（语义不合理） | common.h:57-71 | ❌ |
| M8b | static_assert(std::is_same_v\<uint16_t, uint16_t\>) 恒真 | service_vector.h:402 | ❌ |
| L1 | `#if (true)` 使 arch22 成为死代码 | cpp:20,50 + tiling_key.h:35 | ❌ |
| L2 | V_MTE2 vs MTE2_V 事件定义不对称 | service_vector.h:64-65,387,393 | ❌ |
| L3 | HistogramFirstVFImpl vreg1U16/ThirdVFImpl vreg3U16 变量未使用 | vf_topk.h:57,170 | ❌ |
| L4 | LiTopKGatherVF outputValueLocal/inputValueLocal 参数声明但未使用 | vf_topk_16_gather.h:409-430 | ❌ |
| L5 | CeilAlign(num,rnd) 中 num+rnd-1 在极端大 num 时溢出 | common.h:175 | ❌ |
| L6 | LI_TPL_BF16 定义但未被任何 template key 使用 | tiling_key.h:24 | ❌ |
| L7 | GetAttrParaInfo 中 layOutQuery/layOutKey 重复赋值 | tiling.cpp:157-163 | ❌ |
| L8 | error log 使用 %u 打印 int64_t | tiling.cpp:236,239 | ❌ |
| L9 | QliGetHeadDim 中 TND case DIM_IDX_TWO 冗余赋初值 | tiling.cpp:520-523 | ❌ |
| L10 | 宏名拼写 WEIGTHS_INDEX → WEIGHTS_INDEX | tiling.h:49 | ❌ |

---

## 二、已推翻问题详细论证

### H2 [第二轮推翻] 16-bit TopK EQ 输出机制理解偏差

**原始声称**: GT 输出前缺少 ClearSpr，AR 被先前 Squeeze 操作污染，EQ 无法输出。

**推翻依据**:
- 16-bit 版本与 32-bit 版本的 Filter 设计根本不同
- `FindIdxGTOutputVFImpl`(L230) 内部包含 ClearSpr，AR 被正确清零
- `FindIdxEQOutputVFImpl`(L257) **故意不设 ClearSpr**，让 EQ 元素追加在 GT 之后
- 数学事实：count(x ≥ kth_value) ≥ topK，全量写出后取前 topK 即可

---

### M1 [第二轮推翻] FindKth DIST_NORM_B16 截断分析错误

**原始声称**: StoreAlign\<DIST_NORM_B16\> 只写 16-bit 会丢失高位。

**推翻依据**: kth-value 本身最大为 0xFFFF（16-bit 值域），不需要 32-bit 存储。后续 FindIdxEQ/GT 中 LoadAlign\<DIST_BRC_B16\>(..., nkValueBuf) 正确将其零扩展回 32-bit lane。

---

### N1 [第三轮推翻] FindKth 高字节移位——DIST_BRC_B8 广播语义被忽视

**原始声称**: `And(idxHigh, 0xff00)` 应为 `ShiftLefts(idxHigh, 8)`。

**推翻依据**:
1. idxHigh 来源链：Squeeze 输出 bin 索引 0-255 → 存为 uint32\_t 0x00000042 → **LoadAlign\<DIST_BRC_B8\>** 读取
2. **DIST_BRC_B8 语义**：读取最低有效字节并广播至所有 4 字节 → 0x00000042 → **0x42424242**
   - 代码证据：vf_topk.h:148 注释 "0x000000fc -> 0xfcfcfcfc"
3. And 掩码构建：Duplicate(0xff00) 得到 [0xff00,0xff00,...]，reinterpret_cast 到 uint32\_t 得 **0xff00ff00**
4. 组合过程（high=0x42, low=0x21）：

| 步骤 | 运算 | 结果 |
|------|------|------|
| idxHigh (BRC_B8) | broadcast | 0x42424242 |
| And mask | & 0xff00ff00 | 0x42004200 |
| idxLow (BRC_B16) | broadcast | 0x00210021 |
| Add | 0x42004200 + 0x00210021 | 0x42214221 |
| StoreNorm_B16 | 取低 16-bit lane | **0x4221** ✓ |

**关键洞察**: 16-bit 版本刻意利用 DIST_BRC_B8 广播 + bitwise AND 完成字节提取，相比 32-bit 版本省去多次 ShiftLefts；低 16-bit lane 最大值 0xFF00+0x00FF=0xFFFF，绝不会产生进位污染高位。

---

### N2 [第三轮推翻] EQ 输出不受控——混淆了两个版本的设计选择

**原始声称**: 16-bit 版本没有 EQ 限量机制会导致越界写入。

**推翻依据**: 同 H2。16-bit 版本采用"全量写出+取前N"策略，32-bit 采用"精确计数+逐块判断"策略，两者都是正确实现，只是工程取舍不同。

---

### C1 [第三轮推翻] outBuf_ buffer 溢出——未识别交错寻址

**原始声称**: outBuf\_(pingpong=1) = 256B offset 后仅剩 256B，第二行写入越界。

**推翻依据 — 代数证明**:
- 设 n = CeilDiv(s1BaseSize_, CVRATIO)，分配总量 = 2·n·s2BaseSize\_·sizeof(uint16\_t) = **512n** bytes
- UB_BANK_STRIDE=256, UB_BANK_DEPTH_STRIDE=512
- Pingpong 0: row i 写入 offset i×512，占用 [i×512, i×512+256)
- Pingpong 1: row j 写入 offset 256+j×512，占用 [256+j×512, 256+(j+1)×512)
- 两侧互不相交: 0 用偶 256B，1 用奇 256B
- Max access offset = 256 + (n-1)×512 + 256 = **512n** = 分配量 ✓

---

### A1 [第三轮推翻] Fixp dstNdStride 越界——Cube/Vector 寻址未对照

**原始声称**: dstNdStride=8192 导致第二个 ND 块越界或与 Vector 端 qkVLstride=8192 不一致。

**推翻依据**:
- Cube Fixp 写入: ND block 0 → floats [0, 8191]，ND block 1 → floats [8192, 16383]
- Vector ReadFixp 读取: qkBase + p×qkVLstride + batch_offset×dstStride
  - p=0 → offset 0 对应 ND block 0 ✓
  - p=1 → offset qkVLstride=8192 对应 ND block 1 ✓
- **完美匹配**：dstNdStride=8192 == qkVLstride=8192

---

### P1+P2 [第三轮推翻] SCALE_EXTRA 不足 + srcStride 错误

**P1 推翻**: QUERY_L1_PINGPONG_STRIDE = data(32768) + scale_extra(512) = **33280**。qScale 需求 = 128 tokens × 4 fp8_e8m0 / 2 (packed as bf16) × 1 byte = **256 bytes** ≪ 512。足够。

**P2 推翻**: Scale 经 Dn2Nz 后格式为 NZ(N=2, D=128)，相邻 C0 tile 间隔 BF16_TWO=2 个 bf16 元素。srcStride=2 **正好匹配 NZ 步长**。

---

## 三、MXFP8 Cube 执行路径验证结论

经过三轮交叉验证，以下模块的逻辑正确性已被**确认**：

### Host 侧（Tiling 计算）

| 项目 | 验证点 | 结论 |
|------|--------|:----:|
| OpDef 注册 | MXFP8 类型组合位于 position 4/5 | ✅ |
| ShapeInfer | 输出 shape/dtype 推导逻辑 | ✅ |
| ACLNN 接口 | stride 计算(-1 default)、参数转发顺序 | ✅ |
| CheckScaleShape | isMxFp8Scale=true 分支：scale shape=[B,S,N,D/32] | ✅ |
| GenerateInfo | keyDequantScaleStride0 = blockSize × (headDim/32) | ✅ |
| Layout 转换 | PA_BSND ↔ PA_BBND 映射 | ✅ |

### Kernel 主控制（kernel.h）

| 项目 | 验证点 | 结论 |
|------|--------|:----:|
| Init 函数 | IS_MXFP8 下 AIC 传 scaleGm，AIV 跳过 | ✅ |
| ProcessMain 三层循环 | bN2/gS1/s2 上下边界 + LoopEnd 处理 | ✅ |
| CalcRunInfo MXFP8 分支 | tensorQScaleOffset/tensorKScaleOffset 偏移算法 | ✅ |
| SplitCore 负载均衡 | totalBlockNum=0、deal1MoreBlockCoreNum、尾核处理 | ✅ |
| Causal Mask | validS2Len clip 到 [1, actS2SizeOrig] | ✅ |
| PageAttention | startBlockTableOffset=0 边界、kCacheBlockSize=0 短路 | ✅ |
| 零序列 DealActSeqLenIsZero | INVALID_IDX 填充完整性 | ✅ |
| CrossCore 事件循环内配对 | 事件ID 一致性、Ping-Pong 翻转正确 | ✅ |

### Cube 核矩阵乘（service_cube.h）

| 项目 | 验证点 | 结论 |
|------|--------|:----:|
| InitParams | s2BasicBlock\_=128 硬编码合理 | ✅ |
| InitBuffers | queryL1\_/keyL1\_/cL0\_/mm1ResUB\_ 尺寸公式 | ✅ |
| AllocEventID/FreeEventID | 9 种 flag Set/Wait 严格对称 | ✅ |
| KeyNd2Nz / QueryNd2Nz | NZ 转换参数正确 | ✅ |
| LoadKScaleToL1（BSND） | non-PageAttn 地址连续拷贝 | ✅ |
| LoadKScaleToL1（PA_BSND） | blockTable 分段拷贝、首尾 partial 处理 | ✅ |
| LoadQScaleToL1 | s1g 粒度加载到 L1 尾部额外区 | ✅ |
| LoadQueryToL0a MXFP8 | LoadData2DMx(A), ifTranspose=false, srcStride=2 正确 | ✅ |
| LoadKeyToL0b MXFP8 | LoadData2DMx(B), ifTranspose=true, srcStride=2 正确 | ✅ |
| ComputeL0c | Mmad(mx_fp8_e4m3_t) → mmad_mx 指令触发 | ✅ |
| Fixp | dualDstCtl=1, dstNdStride=8192, 双 ND 块拆分正确 | ✅ |
| ComputeMm1 调度 | L0 分块 mL0/nL0/kL0、Ping-Pong 轮换、flag 时序 | ✅ |

### Vector 核后处理（service_vector.h + vector1.h）

| 项目 | 验证点 | 结论 |
|------|--------|:----:|
| InitBuffers MXFP8 分支 | 跳过 kScaleBuf\_/qScaleBuf\_ 分配 | ✅ |
| ProcessVec1 MxFP8 | BatchMulWeightAndReduceSum_MxFP8 参数传递 | ✅ |
| BatchMulWeightAndReduceSum_MxFP8 | Float→BF16 Cast + ReLU(bf16,bf16) + BrcMul(bf16) + ReduceSum | ✅ |
| qkVLstride 访问边界 | qkBase + (mBaseSize/2−1)×128 + gSize×8192 < 32768 ✓ | ✅ |
| outBuf_ 交错布局 | 代数证明 max_offset = allocation = 512n | ✅ |
| FloatToSortableKey XOR 变换 | pos/neg/zero/-0 排序语义正确 | ✅ |
| Score DataCopy | blockCount/blockLen/strides 三维搬运参数 | ✅ |

### TopK（vf_topk.h + vf_topk_16_gather.h + topk.h）

| 项目 | 验证点 | 结论 |
|------|--------|:----:|
| 32-bit 4 级直方图 | First/Second/Third/Last AR 计数精确控制 | ✅ |
| 16-bit 2 级直方图 | High/Low 全量写出+取前 N 策略正确 | ✅ |
| DIST_BRC_B8 广播 | idxHigh 0x00000042 → 0x42424242 符合硬件语义 | ✅ |
| kth_value 组装 | 0x42004200 + 0x00210021 → 低 16-bit lane = 0x4221 ✓ | ✅ |
| LiTopKVF\<false\> 单 trunk | 完整 histogram→findkth→filter 流程 | ✅ |
| LiTopKVF\<true\> + Gather 多 trunk | double-buffer loopIdx%2 轮换 | ✅ |
| FindRealIndexVFImpl | idx < topK → gather, else → idx − Align(topK,256) + loopBasicIdx | ✅ |
| LITopk\<uint16_t\> Buffer 布局 | hisIndex[idx]/histograms[idx]/idxHigh/idxLow/nkValue/tmpIndex 排布 | ✅ |

---

## 四、MXFP8 影响问题速查表

| 优先级 | # | 问题简述 | 是否必须修复 |
|:------:|:-:|---------|:----------:|
| **P0** | **D1** | CrossCoreSetFlag 首次迭代配对 (待确认硬件) | 待确认 |
| **P0** | **H3** | 16-bit TopK tmpIndexLocal 溢出 1024B | **必须** |
| **P0** | **H4** | layout_key PA_BBND 拒绝、任意串放行 | **必须** |
| **P0** | **H7** | stride 属性 nullptr 解引用风险 | **必须** |
| **P1** | **M8/M9/M10** | 8 处 DumpTensor + 1 处 printf（MXFP8-only printf） | **强烈建议** |
| **P1** | **D2** | SCALE_EXTRA=512 边界过紧（当前参数恰好够） | 建议加固 |
| **P1** | **D3** | mmad_mx n 参数对齐要求（待确认） | 待确认 |
| **P2** | **M2** | NaN bit pattern 覆盖不全 | 视精度要求 |

---

## 五、修复建议优先级

```
═══════════════════════  P0 · 立即修复  ═══════════════════════

[D1]  CrossCoreSetFlag 首次迭代 → 查阅 MODE4 硬件语义词典，必要时改用 CROSS_CORE_SYNC_PAIR
[H3]  TopK tmpIndexLocal buffer → 修正 GetSharedTmpBufferSize 计算公式
[H4]  layout_key regex 校验 → 改为枚举白名单 {BSND,TND,PA_BSND,PA_BBND}
[H7]  stride 属性 nullptr 防御 → 加 CHECK_NULL_OR_THROW

═══════════════════════  P1 · 计划迭代修复  ═══════════════════════

[M8/M9/M10] 调试残留清理 → 用 #ifdef QLI_DEBUG_DUMP 包裹或直接删除
  - service_cube.h:   549-551, 595-597, 671
  - service_vector.h: 410-413 （注意 L410 printf 仅 MXFP8 触发）
[D2]  SCALE_EXTRA 动态化 → CeilDiv(mBaseSize, BLOCK_SIZE) × FP8_TWO
[D3]  Mmad n 对齐 → 查阅 mmad_mx ISA 规约；若需对齐，在 s2L0RealSize 计算处增加 CeilAlign(16)
[H1]  kScale ping-pong（非 MXFP8） → 用 runInfo.kScaleLoop 替代 (s2Idx-s2Start)%16 推导 kScaleBase

═══════════════════════  P2 · 可选改进  ═══════════════════════

[M2]  NaN detection 增强 → 加入指数域全1 + 尾数非0 的判断逻辑
[H5]  NPU arch hardcode → 接入 platform_ascendc 运行时探测
[M4]  tiling.h 字段 widening → keyDequantScaleStride0 改 uint64_t
[M5]  workspace size calc → 用 uint64_t 中转乘法
[M6]  ConstInfo field init → 补充默认值和 static_assert 校验

═══════════════════════  P3 · 代码卫生  ═══════════════════════

[H6/H8/M7/L1-L10] → 死代码清理、typo 修复、变量未使用警告消除
```

---

## 六、检视过程记录

### 第一轮 Agent 分工

| Agent | 职责 | 检视文件 |
|-------|------|---------|
| Agent 1 | op_host 全部 | def.cpp, infershape.cpp, tiling.h/cpp, aclnn.cpp/h, CMakeLists.txt |
| Agent 2 | common + tiling_key + 入口 | common.h, tiling_key.h, quant_lightning_indexer.cpp |
| Agent 3 | arch35 核心服务 | kernel.h, service_cube.h, service_vector.h |
| Agent 4 | VF TopK 模块 | quant_lightning_indexer_topk.h, vector1.h, vf_topk_16_gather.h, vf_topk.h |

### 第二轮 Agent 分工（聚焦 MXFP8 路径）

| Agent | 职责 | 重点验证 |
|-------|------|---------|
| Agent A | service_cube.h MXFP8 路径 | InitParams/Buffers, LoadQ/KScale, LoadQuery/Key, Mmad, Fixp |
| Agent B | kernel.h MXFP8 路径 | Init(Cube vs Vector scale 区分), CalcRunInfo offset 溢出, CrossCore 配对 |
| Agent C | service_vector.h + vector1.h + topk.h MXFP8 | InitBuffers 跳过 scale, ProcessVec1 MxFP8, MulWeightAndReduceSum_MxFP8, ProcessTopK |
| Agent D | common.h + tiling.cpp + vf_topk 系列 | MXFP8 偏特化模板匹配, isMxFp8Scale 校验, stride 计算, 16-bit TopK 全流程 |

### 第三轮 Agent 分工（定向验证第二轮发现）

| Agent | 任务 | 结果 |
|-------|------|------|
| Agent A | 验证 N1 (FindKth And vs ShiftLefts) | **推翻** — DIST_BRC_B8 广播语义使 And 操作正确 |
| Agent B | 验证 C1 (outBuf_ 溢出) + 全部 buffer 审计 | **推翻** — 代数证明交错布局严格吻合；发现 F1/F2 边界过紧但无越界 |
| Agent C | 验证 A1 (Fixp dstNdStride) + Cube→Vector 数据流追踪 | **推翻** — dstNdStride=qkVLstride=8192 完美匹配；P1/P2 一并推翻 |
| Agent D | kernel.h 边界条件遍历 + CrossCoreSetFlag 逐指令配对 | 新增 D1(首次迭代疑问)/D2(SCALE_EXTRA)/D3(mmad_mx n align) |

---

## 七、检视文件清单

| 文件 | 行数 | 第一轮 | 第二轮 | 第三轮 | 综合状态 |
|------|-----:|:------:|:------:|:------:|:--------:|
| quant_lightning_indexer_def.cpp | 145 | — | — | — | ✅ 已确认无问题 |
| quant_lightning_indexer_infershape.cpp | 95 | — | — | — | ✅ 已确认无问题 |
| aclnn_quant_lightning_indexer.cpp | 101 | — | — | — | ✅ 已确认无问题 |
| aclnn_quant_lightning_indexer.h | 62 | — | — | — | ✅ 已确认无问题 |
| CMakeLists.txt | 29 | — | — | — | ✅ 已确认无问题 |
| quant_lightning_indexer_tiling.h | 198 | L10 | — | — | 低优 typo |
| quant_lightning_indexer_tiling.cpp | 979 | H4,H5,H6,H7,M4,M5,L7-L9 | — | — | 多项中低优问题 |
| quant_lightning_indexer_template_tiling_key.h | 113 | L1,L6 | — | — | 低优先死代码 |
| quant_lightning_indexer_common.h | 210 | M6,M7,L5 | — | — | 多项中低优问题 |
| quant_lightning_indexer.cpp | 82 | H8,L1 | — | — | 低优先 |
| quant_lightning_indexer_kernel.h | 706 | H1 | B确认 | D新增D1 | H1+D1(待确认) |
| quant_lightning_indexer_service_cube.h | 712 | M3,M8 | A深入确认 | C追踪Cube流 | M3(死代码)+M8+D2+D3 |
| quant_lightning_indexer_service_vector.h | 589 | M9,M10,L2 | C深入确认 | — | M9,M10(MXFP8-only) |
| quant_lightning_indexer_topk.h | 171 | H3 | D验证TopK | — | **H3 确认** |
| quant_lightning_indexer_vector1.h | 1242 | M2 | C验证MxFP8 | — | M2 低风险 |
| vf_topk_16_gather.h | 432 | N1(误),N2(误),M1(误),L4 | A,B,D分别验证 | A彻底推翻N1 | L4 低优 |
| vf_topk.h | 678 | L3 | — | — | L3 低优先死变量 |
