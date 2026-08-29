# mega_moe 权重 L2 预取功能检视报告

| 项 | 内容 |
|---|---|
| 检视对象 | 双窗口 W8/W4 权重 L2 预取（commit `a328fda16`，7 文件，+372/-17） |
| 检视重点 | 功能正确性、流水完整性 |
| 检视方法 | 预检精读 + 双波假设检验（H1-H7 功能正确性 / H10-H17 流水完整性），关键结论主检视逐行亲验 |
| 检视日期 | 2026-08-31 |
| 结论 | **正确性问题 2 项（Q1 条件触发数据损坏风险 / Q2 脆弱不变量），建议项 6 组，其余通过** |

## 一、功能概述

各 block 的 AIV0 在两段"天然等待窗口"以 fire-and-forget 的 GM→UB `DataCopyPad`（16KB chunk）
把本卡专家权重 W1+W2 原始字节拉进 L2（UB 数据从不消费），使后续 GMM 权重读命中 L2：

- **窗口 0**（输入准备/SendMask 期间）：批内钩子（每批 topkIds 读完成后发 1 块）+ SendMask 返回后
  余量补发（fire-and-forget，不 Drain）。预算 = 32 chunk/AIV0 × 16KB × blockNum。
- **窗口 1**（GMM1 首专家 token-count 自旋间隙）：从窗口 0 已覆盖总量无缝接续，预算 16MB，
  `window2Used` 一次性标志（仅 e0 首等待启用），会话以 `WeightL2PrefetchDrain()`
  （`SetFlag/WaitFlag<MTE2_V>(ID5)` 紧邻配对）收尾。

涉及文件（`mega_moe/op_kernel/arch35/`）：
`common/mega_moe_constants.h`（L67-83 常量）、`common/mega_moe_utils.h`（L271-421 游标 API、
L500-519 等待变体）、`mega_moe_arch35.h`（L480-495 窗口 0 构造、L592-618 挂接、L628-636 补发）、
`mega_moe_wave_a4w4.h` / `mega_moe_wave_a8w4.h` / `mega_moe_wave_a8w8.h`（窗口 1 构造与使用）、
`stage/mega_moe_send_mask.h`（L94-98 批内钩子）。

## 二、正确性问题

### Q1【红线，CONFIRMED 条件触发】SendMask 自适应 UB 布局可与预取 scratch [192K,208K) 并发冲突【已修复：host 侧钳位】

**数据流冲突链：**

1. host 侧 [mega_moe_tiling.cpp:2354-2364](../../mega_moe/op_host/op_tiling/arch35/mega_moe_tiling.cpp#L2354-L2364)
   将平台 UB 全量（`GetCoreMemSize(UB)`）透传给 `SetAdaptiveBufferConfigs`；
2. `CalcSendMaskBufferConfig` Stage 2（[mega_moe_tiling.cpp:1059-1081](../../mega_moe/op_host/op_tiling/arch35/mega_moe_tiling.cpp#L1059-L1081)）
   固定 ring 深度后**用剩余 UB 扩张 routeItemsPerBatch**，布局高水位撑满可用 UB；
3. kernel 侧 [mega_moe_arch35.h:452-478](../../mega_moe/op_kernel/arch35/mega_moe_arch35.h#L452-L478)
   布局尾部 sendMask ring / sendGatherOut / sendCntAcc 随扩张**越过 192K，横跨预取 scratch [192K,208K)**；
4. 窗口 0 批内钩子（[mega_moe_send_mask.h:96-98](../../mega_moe/op_kernel/arch35/stage/mega_moe_send_mask.h#L96-L98)，
   MTE2 队列 fire-and-forget 写 scratch）与 SendMask 自身的 **V 队列写**（`GatherMask` 写 sendGatherOut、
   `CompareScalar` 写 ring slot，[mega_moe_send_mask.h:113-119](../../mega_moe/op_kernel/arch35/stage/mega_moe_send_mask.h#L113-L119)）
   **跨队列并发写同一 UB 区间，无任何顺序保证** → gather 序数 / mask 位被垃圾覆写 →
   对端 expert token count 与 Dispatch 序数错乱 → **最终输出错误**。

**触发阈值量化：** 布局高水位 = fixed(~67KB 最坏) + 2×route×4B + N×(route/8+32B) > 192KB，
即 route items > ~15K。换算：topK=8 时 `numMaxTokensPerRank` > ~1900（常见 prefill 形状即触发）；
topK=4 时 > ~3500。

**为何现有验证未暴露：**
- [mega_moe_tiling.h:106-114](../../mega_moe/op_kernel/arch35/mega_moe_tiling.h#L106-L114) 注释仅按基准批
  12288 item（合计 176064B ≈ 172KB）保守验证，未覆盖 Stage 2 扩张后情形；
- [mega_moe_constants.h:69-71](../../mega_moe/op_kernel/arch35/common/mega_moe_constants.h#L69-L71)
  "与输入准备布局（decode ~62K）之间有大量空洞"的论断**仅 decode 档成立**；
- decode/小 batch 测试形状布局 < 192K，功能正常——缺陷随 prefill 大 batch 显形，且表现为
  偶发数据错乱（跨队列竞态时序相关），难定位。

**修复方向（二选一，待确认后实施）：**
1. host 侧：预取开启档位（mode 判定与 kernel 侧 5 处构造门控同源）将 `availableUbBytes` 钳到
   `WEIGHT_L2_PREFETCH_UB_OFFSET`（192KB），为 scratch 预留 [192K,208K)；
2. kernel 侧：scratch 偏移不再硬编码 192K，改由 tiling 按 SendMask 布局实际高水位下发，
   恒置于布局末端之上。

### Q2【红线-脆弱不变量，CONFIRMED 机制】a8w8+共享专家分支窗口 0 在飞块无显式排空，安全性寄生于无关屏障【已修复：防御性 Drain + 守护注释】

- [mega_moe_wave_a8w8.h:962-970](../../mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h#L962-L970)：
  `sharedExpertNum_ > 0` 分支**没有** `PipeBarrier<PIPE_ALL>`（仅 else 分支 L967-969 有，为既有结构）；
  本次新增的窗口 0 补发（L957-960，fire-and-forget 不 Drain）使在飞块在此分支下持续写
  AIV0 UB [192K,208K) 流入 GMM 流水；
- 实际未出错的唯一原因：`PrepareSharedExpertInput` 尾部
  [mega_moe_shared_expert_input.h:82](../../mega_moe/op_kernel/arch35/stage/mega_moe_shared_expert_input.h#L82)
  有一行**与预取毫无注释关联的**无条件 `PipeBarrier<PIPE_ALL>()`（tokenRange 为空也执行到此处）；
- 该屏障之后 AIV0 的首个 UB 消费点——共享专家 GMM1 epilogue 的 C tile **VECIN [128K,256K)**
  （[mega_moe_gmm1_activation.h:719-728](../../mega_moe/op_kernel/arch35/stage/mega_moe_gmm1_activation.h#L719-L728)
  → `block_epilogue_activation_mx_quant.h` 的 `l0cOutUbFirst_/Second_`，256 行 tile × bf16 = 128KB×2）
  **物理覆盖 [192K,208K)**。屏障是唯一防线：若 `PrepareSharedExpertInput` 未来被重构出
  `tokenRange.count == 0` 早退（仓内 `QuantizeLocalTokens` 已有此早退模式，
  [mega_moe_token_quant.h:105-107](../../mega_moe/op_kernel/arch35/stage/mega_moe_token_quant.h#L105-L107)），
  在飞预取写与 epilogue 读即发生跨队列竞态，**静默污染 C tile**；
- 对照：基类路径（a4w4/a8w4）在 [mega_moe_arch35.h:658-660](../../mega_moe/op_kernel/arch35/mega_moe_arch35.h#L658-L660)
  为**无条件** `PipeBarrier`，窗口 0 在飞块在 SyncAll 前排空，安全。

**修复方向：** `a8w8.h` 补发循环后（或 `sharedExpertNum_ > 0` 分支内）补一次显式
`WeightL2PrefetchDrain()`；至少在 `mega_moe_shared_expert_input.h:82` 屏障处加守护注释
声明其承担的预取排空职责。

## 三、建议项（不阻塞合入）

| 编号 | 问题 | 位置 | 说明与建议 |
|---|---|---|---|
| P2 | 事件 ID 占用图注释失真 | constants.h:68 | "ID0/1/2 占用，3-5 空闲取 5"漏记 **ID4 已被** [gmm2_combine.h:215](../../mega_moe/op_kernel/arch35/stage/mega_moe_gmm2_combine.h#L215)（`SyncFuncStatic<MTE2_V, ID4>`）占用；unpermute 运行时动态使用 0..5（`MAX_UNPERMUTE_INPUT_BUFFER_COUNT=6`）。取 ID5 本身仍安全（Drain Set+Wait 紧邻配对、与 unpermute 时序隔两道全屏障），仅注释需补全 |
| P3 | 窗口门控不对称 | arch35.h:483-494 / a8w8.h:547-557 / a8w4.h:147-158 / a4w4.h:150-161 / a8w8.h:841-855 | 基类窗口 0 查 `topoType==MTE`、a8w8 窗口 0 不查、三处窗口 1 均不查 mode/topo。mode↔dtype 严格互斥（tiling.cpp:838-864 + 608-619 校验）排除 mode 交叉；但 **fp4+URMA 拓扑**时窗口 0 不构造而窗口 1 仍构造，起始偏移跳过从未预取的前缀（性能空洞 + 口径不一）。建议统一门控 |
| P4 | 注释失真 4 处 | 见下 | ① send_mask.h:95 "预算写死 4 块/核"→实际 32 块/AIV0（constants.h:83）；② arch35.h:631-633 "由 ProcessWave 尾部排空"→实际 [arch35.h:659](../../mega_moe/op_kernel/arch35/mega_moe_arch35.h#L659) 阶段 1 后即排空（更早更安全，结论不变）；③ constants.h:67 与 utils.h:271-272 标题"A8W4/A4W4"未覆盖已接入的 W8(fp8)；④ arch35.h:131-134 称窗口 0"Drain + PipeBarrier 收尾"与 fire-and-forget 无 Drain 的实现矛盾 |
| P5 | 死常量 | constants.h:80 | `WEIGHT_L2_PREFETCH_UB_BYTES` 全仓零使用（TryIssue 以 copyBytes 构造 scratch，utils.h:405）。建议删除，或加 `static_assert(copyBytes <= WEIGHT_L2_PREFETCH_UB_BYTES)` 启用为区间背书 |
| P6 | 命名混淆 | utils.h:345-346 / 294 | `WeightL2PrefetchInit` 形参 `hiddenDim` 在全部 5 处调用点实收 `commonConfig_.gmm1OutputDim`（= tiling 的 hiddenDim = GMM1 的 N = SwiGLU gate+up 拼接宽），与模型隐维 h（=tokenHiddenDim=K）极易混淆——W2 公式 `hiddenDim/ACTIVATION_N_HALF` 的正确性完全依赖该实参。建议形参改名 `gmm1OutputDim`；`window2Used` 实为"窗口 1 已用"，建议改名 `countWaitPrefetchUsed` |
| P7 | 口径漂移 | constants.h:74/81 | "窗口 0 共 16MB / 两窗口合计 32MB"仅 blockNum=32 成立（实际 = 32×16K×blockNum）；"窗口 1 lane 无缝接续"仅 regionBytes 为 16K 整除时精确（fp8 只保证 128 对齐），接缝处每 worker 最多 1 chunk 重复/漏预热，L2 去重/损失命中，属性能完备性 |

## 四、通过项（证据链）

**功能正确性（H1-H7 假设检验全通过）：**

1. **地址公式同源**：W1 = `expertIdx×gmm1OutputDim×tokenHiddenDim/epb`、
   W2 = `expertIdx×tokenHiddenDim×(gmm1OutputDim/2)/epb`，与消费侧
   [gmm1_activation.h:995-997](../../mega_moe/op_kernel/arch35/stage/mega_moe_gmm1_activation.h#L995-L997)、
   [gmm2_combine.h:900-902](../../mega_moe/op_kernel/arch35/stage/mega_moe_gmm2_combine.h#L900-L902)
   逐字节一致（含 `GetTensorAddr` 两级指针表同源）；epb 参数化正确（fp4=2 默认 / fp8=1 经
   `PackedElementTraits`）。
2. **per-expert tensor 无越界**（H2 证伪）：host 断言链（tiling.cpp:1449-1481 list 内 shape 强一致、
   1627-1638 维度关系、2257-2271 对齐、556-562 专家数匹配）保证 per-expert tensor 字节数与公式
   严格相等；若公式偏大消费侧自己就会越界——不存在"仅预取越界"的输入。
3. **覆盖完整性**（H1）：16K 整除时区域内交错认领为精确前缀划分（零 gap/overlap，数值推演验证）；
   非整除/回绕/预算耗尽产生的 gap/overlap 均属性能完备性（拷贝恒钳在 regionEnd 内，无越界）。
4. **预算与边界**（H4/H6）：双钳位防 uint64 回绕；均摊零头可忽略（blockNum=40 时全局缺口 16B）；
   窗口 1 起始落点边界正确（Init L370 `>=` 为正确选择，不漏跳不多跳）；超界（startOffset ≥ 总量）
   安全退化为纯自旋，无 nullptr 解引用。
5. **DataCopyPad 非对齐长度合法**（H3）：blockLen 为元素数（uint8 即字节数），DataCopyPad 设计
   即支持非 32B 对齐尾块；仓内既有同类用法（send_mask.h:137 pushBytes ≡4 mod 32）。
6. **构造惯例**（H5）：`LocalTensor(TPosition::VECCALC, 字节偏移, 元素数)` 与全仓一致，
   scratch [192K,208K) ⊆ 248KB UB 上限。

**流水完整性（H10-H17）：**

7. **窗口 1 同步纪律**：Drain 为 Set+Wait 紧邻配对，排空语义正确；自旋每轮**先查 flag 再发块**
   （不推迟就绪检测）；`window2Used` 一次性使 e1+ 退化为原 100-cycle 退避自旋（防 AIV0 在飞
   epilogue 读期间 UB 干净性不可证明的问题，设计自洽）；游标跨等待窗口预算结转。
8. **事件 ID5 无实际并发冲突**（H12）：MTE3_V 方向的 EVENT_ID0-5（SendMask ring）与 MTE2_V ID5
   不同 pipe-pair 独立编址；unpermute 动态 event（可取 5）与 Drain 时序隔 `PipeBarrier<PIPE_ALL>`
   + `SyncAll` 至少两道全屏障。
9. **基类路径窗口 0 排空安全**：arch35.h:658-660 无条件 PipeBarrier 在 SyncAll 前排空全部在飞块；
   a8w8 shared==0 分支同理（a8w8.h:967-969）。
10. **补发与后续操作的交叠安全**（H16）：ResetSyncStatus 走 V+MTE3（无 MTE2）且区间 <192K，
    与补发队列/空间双零重叠；PrepareSharedExpertInput 复用 quant scratch（<192K），其 MTE2 拷贝
    与预取块同队列按序。
11. **AIC/AIV1 零行为变化**：全部经 nullptr → 原路径（arch35.h:601-612 `g_coreType==AIV` 门控 +
    构造限 AIV0；a8w8.h:667-671 AIV1 提前 return）；apt.cpp:109 三 Wave 类仅在
    `TILINGKEY_TPL_MTE` 实例化，layered（URMA）路径零改动。
12. **指针表解析频度**（H7）：`GetTensorAddr` 每区域恰一次（Init/AdvanceRegion），自旋内仅
    flag 轮询 + TryIssue。
13. **unpermute 布局越 192K 但无害**（H10d）：AIV0+AIV1 都执行且布局可达 ~245K，但时序在全部
    预取会话 Drain + 排空之后。

## 五、问题分级汇总

| 分级 | 编号 | 一句话结论 |
|---|---|---|
| 红线（条件触发数据损坏） | Q1 | prefill 大 batch（route items > ~15K）下 SendMask 布局越 192K，窗口 0 预取与 gather/mask 跨队列并发写 → 需 host/kernel 协同预留 scratch |
| 红线（脆弱不变量） | Q2 | a8w8+共享专家分支窗口 0 在飞块仅靠无关屏障排空，C tile [128K,256K) 覆盖 scratch 区，需显式 Drain 或守护注释 |
| 建议 | P2-P7 | 注释失真 / 门控不对称 / 死常量 / 命名混淆 / 口径漂移，均不阻塞合入 |
| 通过 | 其余 | 地址同源、边界、同步纪律、退化路径、零影响面等 13 组证据链完整 |

## 六、修复建议清单（待确认后实施）

1. **Q1**【已实施】：host 侧 `SetAdaptiveBufferConfigs`（mega_moe_tiling.cpp:1346-1356）在
   `topoType==TOPO_TYPE_MTE` 档位将 SendMask 可用预算钳到 `WEIGHT_L2_PREFETCH_UB_OFFSET`（192KB，
   直接引用 kernel 侧常量，单一事实源）；同步在 mega_moe_tiling.h 与 mega_moe_constants.h 注释
   双向标注同步要求。钳位后基准批最坏规格合计约 171KB 仍低于 192KB，双 buffer 下限不受影响。
2. **Q2**【已实施】：`a8w8.h` Process 的 shared>0 分支在 `PrepareSharedExpertInput` 之后补防御性
   `WeightL2PrefetchDrain()`（仅 AIV0 预取档位触发，正常路径 MTE2 已空、Drain 空转 ~百 ns）；
   `shared_expert_input.h` 尾部屏障加守护注释（禁止新增早于屏障的提前返回）；补发处注释改为
   准确描述两个分支的排空点。基类路径（arch35.h:658-660 无条件 PipeBarrier）安全无需改。
3. **P2-P7**【已实施】：
   - P2：constants.h 占用图补全（ID4 占用、unpermute 动态 0..N-1 及其与 Drain 的时序隔离说明）；
   - P3：三处窗口 1 + a8w8 窗口 0 构造注释补门控依赖声明（Wave 类仅 TILINGKEY_TPL_MTE 实例化、
     mode 与类经 tiling 互斥校验一一对应，基类窗口 0 的 topoType 判定为同义冗余防御，
     复用到其他拓扑时须补门控）——采用注释声明而非补冗余判断，避免暗示存在其他路径；
   - P4：4 处注释失真全部修正（send_mask.h 32 块、arch35.h 两处排空点描述、constants.h/utils.h
     标题改为"W8/W4（A8W8/A8W4/A4W4）"）；
   - P5：删除 `WEIGHT_L2_PREFETCH_UB_BYTES`（单缓冲 scratch 容量恒等于单 chunk，常量冗余），
     相关注释统一改引 `WEIGHT_L2_PREFETCH_CHUNK_BYTES`；
   - P6：`hiddenDim` 形参全部改名 `gmm1OutputDim`（含公式注释与"勿与模型隐维混淆"说明）；
     `window2Used` 改名 `countWaitPrefetchUsed`；
   - P7：窗口 0 覆盖量口径改为"32×16KB×blockNum（32 block 时 16MB）"；"无缝接续"措辞补精确性
     条件（16KB 整除时精确，否则接缝处每 worker 最多 1 chunk 重复/漏预热，L2 去重）。
