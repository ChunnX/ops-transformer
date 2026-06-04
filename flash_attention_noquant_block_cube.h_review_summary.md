# 代码检视报告

## 检视概览
- 代码文件：/mnt/workspace/transformer-issue/attention/common/op_kernel/arch35/flash_attention_noquant_block_cube.h
- 代码侧别：Kernel
- 检视文档：cpp-secure.md, cpp-general.md, ascendc-api.md, ascendc-perf.md, ascendc-topk.md
- 总条例数：67
- 检视时间：2026-06-04

## 检视统计
| 状态 | 条例数 | 占比 |
|------|------|------|
| PASS | 61 | 91.0% |
| FAIL（发现问题）| 5 | 7.5% |
| FAIL（需关注）| 1 | 1.5% |

---

## 发现问题（HIGH 置信度）

### [API-1] 禁止 GlobalTensor::SetValue/GetValue
- **问题描述**：`IterateBmm1Nz` 函数中，FP8 反量化分支使用 `GlobalTensor<float>::GetValue()` 逐元素读取 GM 中的 dequantization scale 值。`GetValue()` 是逐元素 GM 访问操作，性能极差，在生产代码中禁止使用。共 3 处调用（行 1339、1340、1343）。
- **代码片段**（flash_attention_noquant_block_cube.h 行 1322-1345）：
  ```cpp
  1322:     float deScaleValueTile1 = 1.0f;
  1323:     float deScaleValueTile2 = 1.0f;
  1324:     if constexpr (isFp8) {
  1325:         int64_t deScaleQOffset = 0;
  1326:         int64_t deScaleKvOffset = 0;
  1327:         // 训练模板：128*128 推理模板：128*256
  1328:         int64_t s1BlockCnt = CeilDivision(constInfo.s1Size, 128);
  1329:         int64_t s2BlockCnt = CeilDivision(constInfo.s2Size, 256);
  1330:         /* Q的反量化scale内容在Gm中的偏移 原始shape为 [B, N2, G, Ceil(S1, 128), 1] */
  1331:         deScaleQOffset = runInfo.boIdx * constInfo.n2G * s1BlockCnt +
  1332:                                 runInfo.n2oIdx * constInfo.gSize * s1BlockCnt +
  1333:                                 runInfo.goIdx * s1BlockCnt + runInfo.s1oIdx;
  1334:         /* KV的反量化scale内容在Gm中的偏移 原始shape为 [B, N2, Ceil(S2, 256), 1] */
  1335:         runInfo.deScaleKvOffset = runInfo.boIdx * constInfo.n2Size * s2BlockCnt +
  1336:                                 runInfo.n2oIdx * s2BlockCnt +
  1337:                                 (runInfo.s2StartIdx >> 8) + (runInfo.s2LoopCount * 2);
  1338:         deScaleKvOffset = runInfo.deScaleKvOffset;
  1339:         float deScaleQValue = this->deScaleQGm.GetValue(deScaleQOffset);
  1340:         float deScaleKValue = this->deScaleKGm.GetValue(deScaleKvOffset);
  1341:         deScaleValueTile1 = constInfo.scaleValue * deScaleQValue * deScaleKValue;
  1342:         if (runInfo.s2RealSize > 256) {
  1343:             deScaleValueTile2 = constInfo.scaleValue * deScaleQValue * this->deScaleKGm.GetValue(deScaleKvOffset + 1);
  1344:         }
  1345:     }
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | 行 1339/1340/1343 直接调用 `this->deScaleQGm.GetValue(...)` 和 `this->deScaleKGm.GetValue(...)`，`deScaleQGm`/`deScaleKGm` 声明为 `GlobalTensor<float>`，完全匹配 API-1 禁止的模式 |
| 上下文防御缺失 | +30% | 当前作用域内无任何批量搬运替代方案（DataCopy/DataCopyPad），而是逐元素从 GM 读取 float 标量值 |
| 领域关联 | +10% | 代码位于 `__aicore__ inline` 函数 `IterateBmm1Nz` 内，属于 Kernel 侧生产代码 |

负向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 无 | 0% | 规则允许调试代码中使用 GetValue，但此处是生产路径中读取反量化 scale 参与计算，不适用豁免 |

自信值 = 40% + 30% + 10% = **80%** ≥ 70% → 判定违规

- **修复建议**：将 `GetValue()` 逐元素读取替换为 `DataCopy` 批量搬运到 UB 局部变量。先将所需的 1-2 个 float scale 值通过 `DataCopy` 从 GM 搬运到 UB LocalTensor，再从 LocalTensor 读取标量值。

---

### [PERF-2] 禁止写死硬件参数
- **问题描述**：`InitLocalBuffer()` 函数中，非 MLA 路径的 L0A/L0B/L0C BufferManager 初始化和子 Buffer 分配使用了硬编码的原始数字（magic numbers），而非命名常量或 API 动态获取。开发者自身在注释中也承认"当前写死"并标注"要改成通过计算获取"。同一函数的 MLA 路径（行 356-361）使用了命名常量，形成了明显的不一致。
- **代码片段**（flash_attention_noquant_block_cube.h 行 383-401）：
  ```cpp
  383:
  384:         // L0A B C 当前写死，能否通过基础api获取
  385:         l0aBufferManager.Init(tPipe, 65536); // 64 * 1024
  386:         l0bBufferManager.Init(tPipe, 65536); // 64 * 1024
  387:         l0cBufferManager.Init(tPipe, 262144); // 256 * 1024
  388:         // L0A B C当前写死，要改成通过计算获取
  389:         if (IsSameType<INPUT_T, float>::value) {
  390:             mmL0ABuffers.Init(l0aBufferManager, 64 * 1024);
  391:             mmL0BBuffers.Init(l0bBufferManager, 64 * 1024);
  392:         } else {
  393:             mmL0ABuffers.Init(l0aBufferManager, 32 * 1024);
  394:             mmL0BBuffers.Init(l0bBufferManager, 32 * 1024);
  395:         }
  396:
  397:         if constexpr (s1BaseSize * s2BaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4 && ...) {
  398:             mmL0CBuffers.Init(l0cBufferManager, (L0C_SIZE / NUM_4) * KB_TO_BYTES);
  399:         } else {
  400:             mmL0CBuffers.Init(l0cBufferManager, (L0C_SIZE / NUM_2) * KB_TO_BYTES);
  401:         }
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | 行 385-387 使用原始数字 `65536`/`262144`，行 390-394 使用 `64 * 1024`/`32 * 1024`，均为硬件缓存大小参数 |
| 上下文防御缺失 | +30% | 代码中无任何动态获取 L0 缓存大小的机制，开发者注释明确承认未实现 |
| 领域关联 | +10% | 代码位于 Kernel 侧 Buffer 初始化函数中，直接对应 PERF-2 条款描述的场景 |

负向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 无有效防御 | 0% | MLA 路径使用命名常量，但不构成对非 MLA 路径的防御 |

自信值 = 40% + 30% + 10% = **80%** ≥ 70% → 判定违规

- **修复建议**：将行 385-394 的原始数字替换为命名常量，参照 MLA 路径使用 `L0C_SIZE * KB_TO_BYTES` 等。若存在获取 L0 缓存大小的基础 API，应优先使用 API 替代编译期常量。

---

### [GEN-2.6] 禁止头文件 using namespace
- **问题描述**：头文件中使用了 4 条 `using namespace` 指令（行 27-30），这将导致所有包含此头文件的翻译单元都被强制导入这 4 个命名空间，造成命名空间污染，可能引发符号冲突和难以排查的编译错误。
- **代码片段**（flash_attention_noquant_block_cube.h 行 15-32）：
  ```cpp
  15: #ifndef FLASH_ATTENTION_NOQUANT_BLOCK_CUBE_H_
  16: #define FLASH_ATTENTION_NOQUANT_BLOCK_CUBE_H_
  17: #include "util_regbase.h"
  18: #include "../offset_calculator.h"
  19: #include "../matmul.h"
  20: #include "../FixpipeOut.h"
  21: #include "../CopyInL1.h"
  22: #include "../memcopy/copy_gm_to_l1.h"
  23:
  24: #include "infer_flash_attention_comm.h"
  25: #include "flash_attention_score_common_regbase.h"
  26: #include "kernel_operator_list_tensor_intf.h"
  27: using namespace AscendC;
  28: using namespace AscendC::Impl::Detail;
  29: using namespace regbaseutil;
  30: using namespace fa_base_matmul;
  31:
  32: namespace NoQuantCube {
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | 行 27-30 明确存在 4 条 `using namespace` 指令 |
| 上下文防御缺失 | +30% | 文件作用域的 `using namespace` 无法通过任何防御手段消除对包含者的污染 |
| 领域关联 | +10% | 文件为 `.h` 头文件，精确命中规则触发条件 |

负向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 无 | 0% | 规则无 Kernel 侧例外条款 |

自信值 = 40% + 30% + 10% = **80%** ≥ 70% → 判定违规

- **修复建议**：移除头文件中的 `using namespace` 指令，改为在 `.cpp` 实现文件中使用，或在头文件中使用完全限定名（如 `AscendC::Buffer`）。若 Ascend C Kernel 侧代码因模板宏展开机制确实依赖文件作用域的 `using namespace`，应在每个函数/类作用域内局部使用 `using` 声明。

---

### [GEN-4.2] 禁止魔鬼数字
- **问题描述**：文件中多处使用未命名的数字字面量（魔鬼数字），且与同函数内 MLA 路径使用命名常量的写法形成明显不一致。代码自身注释也承认了该问题（"当前写死"、"不要写死"）。
- **代码片段 1**（flash_attention_noquant_block_cube.h 行 384-395）：
  ```cpp
  384:         // L0A B C 当前写死，能否通过基础api获取
  385:         l0aBufferManager.Init(tPipe, 65536); // 64 * 1024
  386:         l0bBufferManager.Init(tPipe, 65536); // 64 * 1024
  387:         l0cBufferManager.Init(tPipe, 262144); // 256 * 1024
  388:         // L0A B C当前写死，要改成通过计算获取
  389:         if (IsSameType<INPUT_T, float>::value) {
  390:             mmL0ABuffers.Init(l0aBufferManager, 64 * 1024);
  391:             mmL0BBuffers.Init(l0bBufferManager, 64 * 1024);
  392:         } else {
  393:             mmL0ABuffers.Init(l0aBufferManager, 32 * 1024);
  394:             mmL0BBuffers.Init(l0bBufferManager, 32 * 1024);
  395:         }
  ```
- **代码片段 2**（flash_attention_noquant_block_cube.h 行 1306-1347）：
  ```cpp
  1306:     if (runInfo.s2RealSize > s2SplitSize) {
  1307:         CopyToL1Nd2Nz<INPUT_T>(mm1BTensor, GetKeyGm(runInfo, constInfo)[runInfo.keyOffset], 256,
  1308:                                 constInfo.dSize, constInfo.mm1Kb);
  1309:         CopyToL1Nd2Nz<INPUT_T>(mm1BTensor[L1Boffset], GetKeyGm(runInfo, constInfo)[runInfo.keyOffset + gmSplitOffset],
  1310:                                 runInfo.s2RealSize - 256, constInfo.dSize, constInfo.mm1Kb);
  1311:     } else {
  1312:         CopyToL1Nd2Nz<INPUT_T>(mm1BTensor, GetKeyGm(runInfo, constInfo)[runInfo.keyOffset], runInfo.s2RealSize,
  1313:                                 constInfo.dSize, constInfo.mm1Kb);
  1314:     }
  ...
  1342:         if (runInfo.s2RealSize > 256) {
  1343:             deScaleValueTile2 = constInfo.scaleValue * deScaleQValue * this->deScaleKGm.GetValue(deScaleKvOffset + 1);
  1344:         }
  ...
  1347:     uint32_t nLoop = (runInfo.s2RealSize + 255) >> 8;
  ```
- **代码片段 3**（flash_attention_noquant_block_cube.h 行 1988-1992）：
  ```cpp
  1988:     if constexpr (useDn && isFp8) {
  1989:         fixpipeParams.dstStride = 64;
  1990:     } else {
  1991:         fixpipeParams.dstStride = fixpipeParams.nSize / 2;
  1992:     }
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | 多处直接使用数字字面量，且同函数 MLA 路径已示范命名常量用法 |
| 上下文防御缺失 | +30% | 非 MLA 路径未定义对应命名常量；`s2SplitSize` 已定义(=256)但未被一致使用 |
| 领域关联 | +10% | 硬件 Buffer 大小是关键参数，可维护性要求高 |

负向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 注释说明 | -10% | 部分魔鬼数字有注释解释（如 `// 64 * 1024`），但注释不能替代命名常量 |

自信值 = 40% + 30% + 10% - 10% = **70%** ≥ 70% → 判定违规

- **修复建议**：
  1. 行 385-394：参照 MLA 路径，使用 `L0C_SIZE * KB_TO_BYTES` 等命名常量
  2. 行 1307/1310/1342/1347：将 `256` 替换为 `s2SplitSize`，`255` 替换为 `(s2SplitSize - 1)`，`>> 8` 替换为 `CeilDivision`
  3. 行 1989：定义命名常量说明 `64` 的含义（如 `DN_FP8_DST_STRIDE = 64`）

---

### [GEN-15.2] 入参 const T&
- **问题描述**：`ConstInfo` 在整个文件中是纯只读入参，但几乎所有函数签名都使用非 const 的 `ConstInfo<isInfer, hasRope> &constInfo`，违反规则"入参用 `const T&`"。共 38 处函数签名受影响。同文件 `GetKvByTensorList`（行 187）已正确使用 `const ConstInfo &`，证明技术上可行。
- **代码片段**（flash_attention_noquant_block_cube.h 行 172-218 声明 + 行 2196 实现）：
  ```cpp
  172:     __aicore__ inline void IterateBmm1(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &output,
  173:         RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo);
  ...
  185:     __aicore__ inline void CalcS1Coord(RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo);
  186:     __aicore__ inline void CalcS2Coord(RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo);
  187:     __aicore__ inline void GetKvByTensorList(RunInfo<isInfer> &runInfo, const ConstInfo<isInfer, hasRope> &constInfo,
  ...
  218:     __aicore__ inline bool IsGS1Merge(ConstInfo<isInfer, hasRope> &constInfo);
  ...
  2196: __aicore__ inline bool FANoQuantBlockCube<TEMPLATE_ARGS>::IsGS1Merge(ConstInfo<isInfer, hasRope> &constInfo)
  2197: {
  2198:     return (Q_FORMAT == GmFormat::BSNGD || ...) && constInfo.isPfaGS1Merge;
  2199: }
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | 规则明确要求"入参用 `const T&`"，38 处函数签名使用非 const 引用 |
| 上下文防御缺失 | +30% | 无任何注释或文档解释为何需要非 const；无技术原因阻止 const 限定 |
| 领域关联 | +10% | 同文件行 187 已正确使用 `const ConstInfo &`，证明可行 |

负向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 无 | 0% | 无有效防御可引用 |

自信值 = 40% + 30% + 10% = **80%** ≥ 70% → 判定违规

- **修复建议**：将所有函数签名中的 `ConstInfo<isInfer, hasRope> &constInfo` 改为 `const ConstInfo<isInfer, hasRope> &constInfo`。涉及声明（行 173, 177, 185, 186, 189, 190, 193, 195, 198, 200, 202, 205, 207, 209, 214, 217, 218）和对应实现及 Dummy 类。

---

## 需关注（MED 置信度）

### [GEN-10.6] 指针 const
- **问题描述**：多个函数的指针/引用形参仅被读取但未标记 `const`。涉及两类参数：
  1. `CVSharedParams<isInfer, isPa> *sharedParams` — 在 `InitCubeInput`（行 297）和 `InitGmTensor`（行 407）中仅读取成员（共 46 处读取，0 处写入），但未标记 const。
  2. `ConstInfo<isInfer, hasRope> &constInfo` — 与 GEN-15.2 重叠，在多个函数中仅读取未修改。
- **代码片段**（flash_attention_noquant_block_cube.h 行 295-325）：
  ```cpp
  295: TEMPLATES_DEF_NO_DEFAULT
  296: __aicore__ inline void FANoQuantBlockCube<TEMPLATE_ARGS>::InitCubeInput(
  297:     __gm__ uint8_t *key, __gm__ uint8_t *value, CVSharedParams<isInfer, isPa> *sharedParams,
  298:     AttenMaskInfo *attenMaskInfo, __gm__ int64_t *actualSeqQlenAddr, __gm__ int64_t *actualSeqKvlenAddr,
  299:     __gm__ uint8_t *keySharedPrefix, __gm__ uint8_t *valueSharedPrefix, __gm__ uint8_t *actualSharedPrefixLen)
  300: {
  301:     if ASCEND_IS_AIC {
  302:         if constexpr (isInfer) {
  303:             if (sharedParams->fromFused) {
  304:                 ...
  308:                 if (sharedParams->isKvContinuous == 1) {
  309:                     ...
  323:             attenMaskInfo->preTokens = sharedParams->preTokens;
  324:             attenMaskInfo->nextTokens = sharedParams->nextTokens;
  325:             attenMaskInfo->compressMode = sharedParams->compressMode;
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | sharedParams 和 constInfo 仅被读取但未标记 const |
| 上下文防御缺失 | +30% | 函数体内无任何机制需要非 const 访问 |
| 领域关联 | +10% | 规则标注 [适用: All]，Kernel 侧无豁免 |

负向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 无 | 0% | 无有效防御可引用 |

自信值 = 40% + 30% + 10% = **80%**（MED 因 Ascend C Kernel 侧 const 限定可能影响模板展开兼容性）

- **修复建议**：将 `sharedParams` 改为 `const CVSharedParams<isInfer, isPa> *sharedParams`。`constInfo` 的修复与 GEN-15.2 合并处理。

---

## 通过条例

[SEC-1.1] PASS — 静态类型安全
[SEC-1.2] PASS — 内存安全
[SEC-1.3] PASS — 禁止未定义行为
[SEC-2.1] PASS — 有符号整数不溢出
[SEC-2.2] PASS — 无符号整数不回绕
[SEC-2.3] PASS — 除零保护
[SEC-3.1] PASS — 禁止未初始化变量
[SEC-3.3] PASS — 数组索引校验
[SEC-3.4] PASS — 禁止sizeof指针
[SEC-3.5] PASS — 指针判空
[SEC-4.1] PASS — 外部输入合法性校验
[SEC-4.2] PASS — 内存操作长度校验
[SEC-9.1] PASS — 禁止逐位操作非trivially copyable
[SEC-10.3] PASS — 敏感信息清零
[SEC-10.4] PASS — 结构体字段末尾添加
[SEC-10.5] PASS — 接口变更兼容性
[TOPK-3] PASS — 局部变量指针野指针
[TOPK-6] PASS — nan/inf等特殊值处理
[TOPK-8] PASS — gm内存偏移用int64
[TOPK-9] PASS — atomic累加需清零
[TOPK-10] PASS — 可整数计算时不转浮点
[TOPK-11] PASS — 通信算子核间同步
[TOPK-12] PASS — 宏命名不冲突
[API-2] PASS — 禁止std::计算函数
[API-3] PASS — DataCopy对齐
[API-4] PASS — Compare 256字节对齐
[API-6] PASS — AllocTensor/FreeTensor配对
[API-7] PASS — 禁止动态内存分配
[API-8] PASS — repeatTimes≤255
[API-9] PASS — Cast RoundMode
[API-10] PASS — DataCopyParams单位差异
[API-12] PASS — CrossCoreSetFlag/WaitFlag对称
[PERF-1] PASS — 循环内禁止逐元素操作
[PERF-3] PASS — Double Buffer使用
[PERF-4] PASS — PipeBarrier优化
[PERF-5] PASS — 单次搬运量优化
[PERF-6] PASS — 避免GM重复读取
[PERF-7] PASS — 尾块处理正确性
[PREC-1] PASS — 流水线同步正确性
[PREC-2] PASS — 除零保护
[PREC-3] PASS — FP16中间精度保护
[TIL-1] PASS — 多核负载均衡
[TIL-2] PASS — 片上缓存容量不溢出
[GEN-1.1] PASS — 外部数据合法性检查
[GEN-1.2] PASS — 函数结果优先返回值
[GEN-1.3] PASS — 删除无效冗余代码
[GEN-2.2] PASS — 禁止循环依赖
[GEN-2.3] PASS — 禁止无用头文件
[GEN-2.4] PASS — 禁止extern声明
[GEN-2.5] PASS — 禁止extern C含头文件
[GEN-3.1] PASS — 避免滥用typedef
[GEN-4.1] PASS — 禁止宏常量
[GEN-4.3] PASS — 常量单一职责
[GEN-5.2] PASS — 避免全局变量
[GEN-5.3] PASS — 禁止自增自减再引用
[GEN-6.1] PASS — 比较左变右不变
[GEN-6.2] PASS — 括号明确优先级
[GEN-8.1] PASS — switch有default
[GEN-10.7] PASS — 数组传长度
[GEN-12.1] PASS — 断言不用于运行期
[GEN-14.4] PASS — 强类型参数
[GEN-15.1] PASS — 入参在前
[GEN-15.5] PASS — 单参数explicit
