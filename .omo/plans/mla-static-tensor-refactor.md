---
slug: mla-static-tensor-refactor
status: plan
intent: clear
review_required: false
predecessors: [.omo/drafts/mla-static-tensor-refactor.md, .omo/plans/mla-align-flash-attn.md]
approach: 以 flash_attn 的静态 tensor（InitBuffers + BUFCNT/BUF_BUF_BYTES constexpr + LocalTensor(TPosition,offset,size)）+ Mutex 同步架构为基准，把 flash_mla_with_kvcache 的 kernel/cube/vec/fd 四层从 TPipe/TQue/TBuf 动态模型改造为全静态模型；行为零变化（PRELOAD_N=1、槽数、tiling 数值、72 实例化面不变）；验收 = 编译通过 + UT 绿 + 行为零变化证据
---

# Plan: mla-static-tensor-refactor

## TL;DR (For humans)

把 `attention/flash_mla_with_kvcache` 的 buffer 管理与同步机制从 **TPipe/TQue/TBuf 动态模型** 改造为 flash_attn 的 **静态 tensor + Mutex 模型**：

- **去掉全部 TPipe 依赖**：kernel 的 `pipe_` 成员、vec 的 9 处 TQue/TBuf、FD 的 16 个 TBuf、4 处 `GetTPipePtr()->AllocEventID` 全部消除。
- **改为编译期静态布址**：各 block 的 `InitBuffers()` 用 `LocalTensor<uint8_t>(TPosition, 偏移, 大小)` + `X_BUFCNT/X_BUF_BYTES` constexpr 常量直接圈定 UB（VECIN）/ L1（A1）/ L0C（CO1 2×128K）；L0A/L0B 改用 `BufferManager::Init(64K)` 静态重载 + `BuffersPolicyDB(LOCK_UNLOCK, EXTERNAL)` 手填 ID。
- **核内同步换 Mutex**：`Mutex::Lock/Unlock<PIPE>(显式 ID)`（硬件 get_buf/rls_buf），替代 TPipe 事件与 SetFlag/WaitFlag 事件对。
- **核间同步显式化**：AIC↔AIV 用显式 `CrossCoreSetFlag/CrossCoreWaitFlag<mode=4>` 硬件 intra-block 旗标（AIV1 偏移 +16），与 flash_attn 的 `InitCrossCoreSync/UnInitCrossCoreSync` 模式同构。
- **行为零变化**：PRELOAD_N=1、缓冲槽数与字节数（含 L0C：2×128K policy → 2×128K 静态槽，槽数与总量均不变，仅机制换为 静态 + 2 个 Mutex）、tiling 数值、72 实例化面（36 组合 × 2 dtype）全部不变；不改 aclnn/def 接口；不改 `attention/flash_attn/` 与 `attention/common/` 与 `common/`（只读参考）。
- **验收硬性条件**：`bash build_mla.sh`（-j20）退出码 0 + `tests/ut` 全绿；有 NPU 跑 pytests、无则如实列未验证清单。

风险最高点：UB 预算按源事实精确核算 ≈ 245.4K（CV 160K + vec2Res 64K + stage1 8.25K + mask 8K + softmax 1.5K + LSE 1K + max/sumBrdcst 2K + tmp 0.5K），满足 ≤248K 静态断言，但余量仅 ~3K——静态布局必须与现状容量逐一核对（表 1 已按源容量修正）；FD 静态布局必须放 [0..95K] 与 bmm2 CV 区时间复用（注 4），pop-buffer 与 CV 槽瞬时别名依赖 c2v2 反向旗标门控（注 5）；跨核旗标 ID 必须在 cube/vec 两侧一致——**数据面**（删 kernel 级 manager + 签名变化）是编译原子，**同步面 ID** 是各 TU 独立编译期常量、不一致只运行时失败，故单一大提交是好习惯而非编译硬需求（git 恢复点 + wip 检查点策略见 Verification strategy）。

## Scope

**IN（要做的）**
1. K1 kernel 层：删除 `AscendC::TPipe pipe_`（flash_mla_with_kvcache_kernel_noquant_mla.h:91）、`InitMMResBuf`（:175-186）、kernel 级 `ubBufferManager_/l1BufferManager_/bmm1Buffers_/bmm2Buffers_/l1KvPBuffers_`（:61-68）；`ComputeMm1/Mm2/ComputeVec1/Vec2`（:619-638）改为 runInfo-only 直调 block 内部静态 buffer；`CrossCoreBufferInit/UnInit`（:342-369）移除并迁移为 block 级 `InitCrossCoreSync/UnInitCrossCoreSync`。
2. K2 cube block（flash_mla_with_kvcache_block_cube_noquant_mla.h）：`InitBuffers()` 静态化——UB CV 区（mm2@VECIN 0..128K、mm1@128K..160K）、L1（Q 72K + KV/P 3×144K，A1）、L0C（CO1 2×128K=256K，静态 + 2 Mutex ID，见注 3）、L0A/L0B 静态 `BufferManager::Init(BUFFER_SIZE_BYTE_64K)` + `BuffersPolicyDB<..., INNER_CORE_SYNC, LOCK_UNLOCK, IdSource::EXTERNAL>`；`IterateBmm1/Bmm2`（:287-372）中 L1/L0C 的 `WaitFlag/SetFlag(GetEventID)` 换 `Mutex::Lock/Unlock<PIPE_MTE2/MTE1/PIPE_M/PIPE_FIX>`；去 `TPipe *tPipe` 成员；`AllocEventID/FreeEventID/InitCrossCoreSync/UnInitCrossCoreSync` 对齐 FA。
3. K3 vec block（flash_mla_with_kvcache_block_vec_noquant_mla.h）：`InitBuffers()` 静态化——UB `struct UbLayout`（与 cube 的 CV 别名 + mask + vec1Res + vec2Res + softmax×3 + LSE + tmp）`static_assert ≤ 248K`；9 处 TQue/TBuf（:102-114,661-699）全静态 index 轮转；`GetTPipePtr()->AllocEventID/ReleaseEventID`（:703-718）换固定 Mutex ID；`SoftmaxInitBuffer`（:661-679）静态化；去 `tPipe` 成员；**`ClearOutput` 清零机制替换**：删除 `InitOutputSingleCore`/`InitLseOutputSingleCore`（:200-239）与 `initOutputEventId` 节拍，改用 flash_attn 式 `AttentionCommon::InitOutput<..., EVENT_ID0/1, popBufAddr, eleSize, true>` 协同清零 + `SyncAll`（参照 vec_nd.h:298-326；pop buffer 地址/EVENT_ID 见附录 A 表 1 注 5、表 2）。
4. K4 FD block（flash_mla_with_kvcache_block_vec_flashdecode_mla.h）：16 个 TBuf（:121-137）+ `InitBuffers(TPipe*)`（:163-189 含 pipe->Reset()）→ 静态绝对偏移布局（**放 VECIN 0..95K**，按 MLA FD 自身尺寸 6K×5+16K×4+1K 顺序排布；风格参照 FA 绝对偏移，但**不得镜像 FA 的 160K BASE/32K vec2Res 常量**——FA 公式在 MLA 上溢出 248K 且与 64K vec2Res 竞态，见注 4）；kernel `FlashDecode`（kernel :837 `vecFdBlock_.InitBuffers(&this->pipe_)`）改无参调用。
5. K5 同步机制：显式 `CROSSCORE_*` 旗标 ID 表 + Mutex ID 表（见附录 A），cube/vec 两侧 ID 一致；kernel `Process()`（kernel :845-879）编排对齐 flash_attn_kernel_nd.h:626-652（AIV 先 InitBuffers→InitCrossCoreSync→AllocEventID；AIC 先 InitCrossCoreSync→InitBuffers→AllocEventID；section 循环；收尾 FreeEventID→UnInitCrossCoreSync）。
6. 编译门 + 回归 + 行为零变化验证：build_mla.sh -j20 / 等价 build.sh 命令退出码 0；tests/ut 全绿；有 NPU 跑 pytests；行为门产物对比（有机会跑时）。

**OUT（Must NOT have）**
- 不深改流水：PRELOAD_N 保持 1、ring 仍 2；CV UB 仍 2 槽、L1 KV/P 仍 3 槽、Q 仍单槽、softmax 仍 2 槽；L0C 从 2×128K policy 改为 2×128K 静态 + 2 Mutex（槽数与总量不变，见附录 A 注 3）。
- 不做性能调优/精度调整；不采集性能基线；不改 tiling 数值与实例化面（72 不变）。
- 不改 `attention/flash_mla_with_kvcache/op_kernel/arch35/flash_mla_with_kvcache_template_tiling_key.h`（DECL/SEL 宏体与位宽 token 一字不动；本计划不触碰该头）。
- 不改 `attention/flash_attn/`、`common/`、`torch_extension/`、`docs/`（只读参考）。
- 不重命名算子/语义类名；不删除 examples。
- 不迁移 flash_attn 测试用例资产；不新建 utils/ 目录。
- 不引入新的外部依赖/新库。

## Verification strategy

**每 todo 的 QA 门**（除注明外）：
- 编译门：`bash build_mla.sh --incremental`（或等价 `build.sh --pkg --soc=ascend950 --ops=flash_mla_with_kvcache -j20`）退出码 0。**全局硬约束：所有编译类命令一律 `-j20`，禁止省略或改值。**
- 回归门：`tests/ut` 现有（git 跟踪的）用例全绿——**以 Todo 1 现场核实为准**（仓库当前 git 跟踪内 `tests/ut/` 仅通用框架，算子级 arch35/UT 与 `.py` pytests 未被跟踪，见 Todo 1 动作 ④）；`tests/pytests` functional_stc/functional_nc 有 NPU 才跑。无 NPU 时在交付说明如实列出未验证项，禁止伪造。
- 行为门（Todo 2/3 之后）：有机会跑时，对同一输入采集改动前后 `attention/flash_mla_with_kvcache/examples/test_aclnn_flash_mla_with_kvcache.cpp` 输出做 diff；无环境则降级为编译+UT 并如实记录。

**关键工程约束**：
- **跨核旗标 ID 一致性**：c1v1（BMM1 UB 2 槽）、c2v2（BMM2 UB 2 槽）、v1c2（L1 P 3 槽）的 `CrossCoreSetFlag/WaitFlag` ID 在 cube 与 vec 两侧必须逐字节一致（附录 A 表 2），否则死锁——为此 Todo 2 将数据面+同步面合成 **单一原子提交**，禁止拆开独立提交。
- **UB/L1 预算**：UB 静态布局 `static_assert(sizeof(UbLayout) <= 248 * 1024)`、L1 `static_assert(sizeof(L1Layout) <= 512 * 1024)`（参照 flash_attn_block_vec_nd.h:219、flash_attn_block_cube_nd.h:167）。若超限按附录 A 注 1 的 trim 优先级裁减并记录 deviation。
- **Dummy/Real 约束（核对后放宽）**：`ASCEND_IS_AIC`/`ASCEND_IS_AIV` 展开为 constexpr（sys_macros.h:68/80），`if ASCEND_IS_AIC{...}` 是 if-constexpr——未命中分支被编译器丢弃，各 TU 的*被命中*路径只调用 Real 块方法；**Dummy 块无需镜像全部接口**，只须保留被非丢弃分支使用的类型别名与静态常量（如 mBaseSize）。改动后 AIC/AIV 两 TU 编译通过即可；Todo 3 ③ 只做 sanity 核对，不为追求"接口齐平"给 Dummy 补无用空方法。
- **常量不漂移**：静态布局字节数与改动前的 `mm1ResultSize/mm2ResultSize/mm12RightSize/Q（576宽）` 数值保持一致（见附录 A 表 1）；MatmulK/N 的 L0A/L0B policy 参数保留（Matmul 接口需要 Buffer 参数），仅切换 SyncMode/IdSource。
- FD `isFd` 是 kernel 级模板参数，不挪动（见 mla-align 计划约束，与本次无关但防回归）。

## Execution strategy

- **顺序**：Todo 1（基线快照）→ Todo 2（大波：数据面+同步面，全静态化+显式旗标+Mutex，单原子提交）→ 编译门 → Todo 3（编排面：kernel Process 对齐 FA，runInfo-only 终核，Dummy 接口核对）→ 编译门 → Todo 4（行为/回归验证）→ Todo 5（全量终检）→ F1~F4。
- **只读参考**（改造动作以 flash_attn 对位文件为基准）：`attention/flash_attn/op_kernel/arch35/flash_attn_block_cube_nd.h`（InitBuffers:153-183、常量:93-121、Mutex/L0C:342-364、L1 Mutex:328-340、CrossCore:221-231,296-394）、`flash_attn_block_vec_nd.h`（InitBuffers:196-245、常量:96-133、Mutex:467-478,635-694、CrossCore:256-294）、`flash_attn_block_vec_flashdecode.h:173-232`、`flash_attn_kernel_nd.h:580-591,626-652`、`attention/common/op_kernel/buffer_manager.h:44-48`、`attention/common/op_kernel/buffers_policy.h`（BuffersPolicyDB）、`attention/common/op_kernel/buffer.h:209-227,276-322`（LOCK_UNLOCK/EXTERNAL）。
- **git 纪律**：Todo 2 大波开工前确认工作区干净，并 `git rev-parse HEAD` 把**恢复点 hash 记录进 `.omo/baselines/mla-static-tensor-baseline.md`**（Todo 1 产物）；波内允许 wip 检查点提交或 `git add -p` 分段暂存，**最终原子提交仅在验收门通过后 author**；若大波中途无法继续：已提交过 wip → `git reset --hard <恢复点 hash>`，未提交 → `git checkout -- attention/flash_mla_with_kvcache`；每个 todo 一个原子提交（Todo 2 本身单提交）。`git checkout --` 只能还原未提交改动，无法撤销已提交的 wip——恢复一律以 hash 为准。
- **实现顺序内部建议（不限死）**：先按附录 A 表 1 精确布局常量与 static_assert，再逐 block 机械替换数据面，最后统一同步面与 kernel 编排。

## 附录 A：静态布局与 ID 表（决策固化）

**表 1 静态布局**（INPUT_T=2B、T=float=4B、CV_RATIO=2、mBase=64/s2=128/d=576/dv=512；数值 = 现状容量逐一核算，改动前后一致）：

| 区 | 位置 | 大小/槽 | 依据（现状容量） |
|---|---|---|---|
| UB mm2Res（CV） | VECIN 0 | 2 × 64K = 128K | mm2ResultSize=64/2×512×4=64K；cube 写、vec 读 |
| UB mm1Res（CV） | VECIN 128K | 2 × 16K = 32K | mm1ResultSize=64/2×128×4=16K；cube 写、vec 读 |
| UB vec2Res/stage2 | VECIN 160K | 64K | stage2OutBuf=64/2×512×4=64K（单槽） |
| UB stage1/vec1Res | VECIN 224K | 8.25K（1 槽） | stage1OutQue 深度 1：33×128×2=8448B |
| UB mask | VECIN 后 | 2 × 4K = 8K | attenMaskInQue[2] 各 4096B |
| UB softmax sum/max/exp | VECIN 后 | 3 × (2 × 256B) = 1.5K | SoftmaxInitBuffer ACT_BYTES=256B、槽 [mloop%2] |
| UB LSE out | VECIN 后 | 1K（1 槽） | softmaxLseQueue=32×4×8=1024B |
| UB maxBrdcst/sumBrdcst | VECIN 后 | 2 × 1K = 2K | 现状各 1024B |
| UB tmp（commonTBuf） | VECIN 后 | 512B | 现状 commonTBuf |
| UB attenOut pop-buffer | VECIN 0（init-only 瞬时别名） | 32K | InitOutput 协同清零用，见注 5；**不占预算** |
| UB LSE pop-buffer | VECIN 32K（init-only 瞬时别名） | 32K | InitOutput LSE 用，见注 5；**不占预算** |
| L1 Q | A1 0 | 72K（1 槽） | 64×576×2=72K |
| L1 KV/P | A1 72K | 3 × 144K = 432K | 128×576×2=144K；P 写 rope 段、偏移语义不变 |
| L0C | CO1 0 | 2 × 128K = 256K | 槽数与总量均不变（注 3） |
| L0A/L0B | A2/B2 | 静态 manager 64K + DB 2×32K | LOCK_UNLOCK + EXTERNAL ID |

**UB 合计 = 128+32+64+8.25+8+1.5+1+2+0.5 ≈ 245.25K ≤ 248K 静态断言 ✓**（L1 合计 504K ≤ 512K ✓）。
注 1：UB 合计 245.25K ≤ 248K 静态断言，**默认无需 trim**。若 `static_assert`/编译实测超限（例如软对齐抬升），裁减优先级：① LSE 并入 commonTBuf/tmp 区（省 1K）；② stage1 与 mm1Res 空闲槽别名复用（DataCopy 消费后语义等价，需注释说明）；③ mask 降 1 槽（省 4K）。任何 trim 必须记录 deviation。
注 2：L1 合计 504K ≤ 512K（3×144K + 72K）；Q 与 KV/P 的 Mutex ID 独立轮转（表 2）。vec 写 P 的目标 L1 偏移必须与 cube 的 L1 静态布局（A1 72K 起、每槽 144K、P 在槽内 rope 段偏移 `s2BaseSize*dVBaseSize`）逐字节一致——cube/vec 两侧的偏移常量在各自块内定义时，注释必须互相引用同一公式，防止漂移。
注 3：L0C 从 2×128K（policy DB，内部 Set/WaitFlag 事件）改为 **2×128K 静态槽 + 2 个 Mutex ID（PIPE_M/PIPE_FIX）**。硬约束：bmm2 单次 `Mmad` 的 L0C 结果足迹 = 64×512×fp32 = **128K**（MakeMMParam(mBaseSize,dSizeV) 见 cube :357；n=512 见 matmul.h:841-853），**槽必须保持 128K、不得拆成 4×64K**（64K 槽会让一次 bmm2 跨两槽，而 mm1 的 32K 足迹在同一 Mutex 轮转下并发写相邻槽造成结果破坏）；总量 256K 与槽数均不变，仅机制替换。按 FA cube_nd.h:342-364 手写 M/FIX Mutex 对，槽轮转 `l0cBufId_ = (l0cBufId_+1) % 2`。
注 4（修订）：**FD block 的静态 UB 布局放在 VECIN 0..95K**（顺序排布 FD 自身 buffer：6K×5 + 16K×4 + 1K = 95K，参照 FA 的绝对偏移风格但用 MLA 实际尺寸），**不得镜像 FA 的 160K BASE/32K vec2Res 常量**——FA 公式在 MLA 上会溢出（OFF_BUF1B 末端 ≈272K > 248K 可用 UB）且其硬编码的 32K vec2Res 跳过量与 MLA 实际的 64K stage2OutBuf 不符（放 176K 会与 [160K..224K] 活跃 vec2Res 竞态）。**该区落在 bmm2 CV 区 [0..128K] 之上属时间复用**：FD 窗口内唯一 cross-core 活跃写是 AIC 的 mm1 Fixpipe（目标 [128K..160K]，不相交），AIC 下一 section 的 mm2(0/1) 写被 c2v2 反向旗标门控于 AIV 其后 vec2 消费（cube :350-351 / vec :424），AIV 自写区由 FD 阶段串行承担 + 前置 `SyncAll()`（kernel :838-840）——与现状 `pipe->Reset()` 的有效布局一致。FD 各 buffer 的物理字节属于 CV 区已有分配，**静态布局不再额外占用，不得将其累加进表 1 预算**。
注 5（修订）：**InitOutput 协同清零的 pop-buffer 瞬时别名**：flash_attn 式 `AttentionCommon::InitOutput<..., true>` 需要静态 UB pop-buffer（atten-out @ VECIN 0、LSE @ VECIN 32K，各 32K，沿用 FA 语义，Mutex ID 见表 2 EVENT_ID0/1）。该区落在 bmm2 CV 槽上（0..128K）。**安全论证不依赖 `SyncAll()`（核内屏障，非 cross-core 保证）**，真正的不变量是：AIC 对 [0..64K] 的首次写入是 mm2(0)，被 c2v2 反向旗标门控于 AIV 其后的 vec2(0) 消费（cube :350-351 / vec :424），而 AIV 的 vec2(0) 严格晚于 ClearOutput 完成；AIC 首个 CV 写（mm1(0)）目标 [128K..160K]，与 pop 区不相交。若未来改动破坏该门控次序（如加深 PRELOAD 或改旗标序），立即触发回退：把 pop-buffer 移到 vec 空闲区（160K 之后）并记 deviation。

**表 2 同步 ID（按 flash_attn 常量语义映射，cube/vec 两侧一致）**：

| 用途 | pipe 对 | ID（base） | 槽 | 侧 |
|---|---|---|---|---|
| CROSSCORE_BMM1_0/1（c1v1） | FIX(写)/V(读) | 0/1，AIV1 +16 | 2 | cube+vec |
| CROSSCORE_BMM2_0/1（c2v2） | FIX(写)/V(读) | 2/3，AIV1 +16 | 2 | cube+vec |
| CROSSCORE_L1P_0/1/2（v1c2） | MTE3(写)/MTE1(读) | 5/6/7，AIV1 +16 | 3 | cube+vec |
| L0C_BUFFER_ID0/1 | M / FIX | 0/1（cube 核内） | 2 | cube |
| Q_L1_BUFFER_ID0/1 | MTE2/MTE1 | 2/3（cube 核内） | 2 | cube |
| KV_L1_BUFFER_ID0..2 | MTE2/MTE1 | 4..6（cube 核内） | 3 | cube |
| L0A_BUFFER_ID0/1 | MTE1/M（policy 内） | 7/8（cube 核内） | 2 | cube |
| L0B_BUFFER_ID0/1 | MTE1/M（policy 内） | 9/10（cube 核内） | 2 | cube |
| UB_OUT_VEC2_RES_EVENT0 | V / MTE3 | 0（vec 核内） | — | vec |
| UB_OUT_VEC1_RES_EVENT0 | V / MTE3 | 2（vec 核内） | — | vec |
| UB_IN_MASK_EVENT0/1 | MTE2 / V | 6/7（vec 核内） | 2 | vec |
| EVENT_ID0（InitOutput atten-out） | **Mutex（PIPE_V/PIPE_MTE3）** | 1（vec 核内） | — | vec |
| EVENT_ID1（InitOutput LSE） | **Mutex（PIPE_V/PIPE_MTE3）** | 3（vec 核内） | — | vec |

注 6：Mutex ID 空间 0-26 按核隔离（AIC 与 AIV 各自独立），表内"cube 核内/vec 核内"表示互不冲突即可（与 flash_attn_block_cube_nd.h:77-91、flash_attn_block_vec_nd.h:84-93 同构）；核间一律走 CROSSCORE 旗标，禁止用 Mutex 做核间同步。表 2 中 L0A/L0B 的 7..10 为 MLA 自洽分配的占位值——唯一硬约束是本核内不与其他行冲突、与 flash_attn 语义等价（FA 在 AllocEventID 传 L0A 6/7、L0B 8/9），worker 可微调并统一在注释中维护映射。
注 7：宏 `CROSS_CORE_SYNC_MODE = 4`、`AIV0_AIV1_OFFSET = 16` 常量在 cube/vec 各自块内定义（参照 FA）。
注 8：**cross-core 旗标预置/drain 语义**（照 FA 模式，防首轮/末轮死锁）：AIV 侧 `InitCrossCoreSync()` 预置 c1v1+c2v2 四个旗标（CROSSCORE_BMM1_0/1、CROSSCORE_BMM2_0/1）；AIC 侧 `UnInitCrossCoreSync()` 收尾 drain 8 个（四个旗标 × {本 AIV0, +AIV0_AIV1_OFFSET}）；v1c2 的 L1P_0/1/2 三旗标随任务 ping-pong 无需预置——若实现中发现首个 BMM2 等待挂起，把 L1P 三旗标也纳入 AIV 预置并记录 deviation。flag ID 归属、计数必须与 FA 的 InitCrossCoreSync/UnInitCrossCoreSync（vec_nd.h:256-262、cube_nd.h:221-231）逐一对齐。
注 9：**EVENT_ID0/1 是 Mutex ID（PIPE_V/PIPE_MTE3），不是 SetFlag 事件 ID**——`AttentionCommon::InitOutput` 在 `ENABLE_LOCK=true` 路径把 SYNC_ID 映射为 `Mutex::Lock<PIPE_V>/<PIPE_MTE3>`（attention/common/op_kernel/init_output.h:43-59）。禁止将其计入 SetFlag 事件池；它们仅在 kernel Init 阶段执行一次（FA kernel_nd.h:137 同构），与 vec 核内其他同步用户时序不重叠。
注 10：**vec 侧事件 ID 全额枚举是 Todo 2 ③ 的前置动作**：现状 vec 的 4 个 TPipe 事件（`mte3ToVId[0/1]`、`vToMte3Id[0/1]`，:703-708）属 **MTE3_V 与 V_MTE3 两个不同 HardEvent 类**（每类独立 0..7 空间），另有 maxBrdcst/sumBrdcst/softmaxLseQueue/stage1 队列内部事件。表 2 当前 0/2/6/7 仅为示意，**非封闭清单**——执行者开工前必须按 FA vec_nd.h:84-93 的布局（0..7，含 LSE-out 4/5）逐类枚举锁定并回填本表，防止事件 ID 双重分配（静默失效，编译期不可见）。

## Todos

- [x] 1. 基线快照 + 环境确认
    - 参考：现场核实的测试资产——`find tests -iname '*flash_mla*'` 与 `git ls-files tests/ | grep -i mla`（注意：`test_flash_mla_with_kvcache_tiling.*` 与算子下 `.py` pytests 目前**未被 git 跟踪**，以 Todo 1 动作 ④ 现场核实为准）；`attention/flash_mla_with_kvcache/examples/test_aclnn_flash_mla_with_kvcache.cpp`；仓根 `build_mla.sh`（前一计划已交付）
    - 动作：① 确认工作区干净（`git status`）；② 确认 `build_mla.sh` 支持 `--incremental` 参数（`grep -n incremental build_mla.sh`；不支持则本计划内统一改用等价 `build.sh` 全量命令并记录）；③ 跑 `bash build_mla.sh`（-j20）确认改动前基线编译通过并存留产物，**同时记录构建警告计数**（供 Todo 5"无新增"比对）；④ 跑 `tests/ut` 中实际存在的套件并记录基线通过/失败清单（`find tests -iname '*flash_mla*'` 与仓库框架用例；**文件不存在时如实记录"该用例未被 git 跟踪/不存在"**，不得编造通过项）；⑤ 有机会跑则保存行为门 before 基准（详见 Todo 4 dump 路径约定）；⑥ 记录当前 pytests functional_stc/functional_nc 通过状态（无 NPU 记为"未跑"）
    - 验收：基线记录写入 `.omo/baselines/mla-static-tensor-baseline.md`（编译产物路径、构建警告数、UT 通过/失败清单、before 产物路径、pytests 状态），文件存在且字段完整
    - QA：happy=`test -f .omo/baselines/mla-static-tensor-baseline.md` 且字段非空；failure=基线编译失败为阻断项，先修复环境再继续
    - 提交：无（不提交，仅记录；`.omo/` 本就在 git 忽略或在 .omo 分支管理）
- [x] 2. 大波：数据面+同步面全静态化（单原子提交）
    - 参考：flash_attn 静态范式与附录 A 表 1/表 2；`attention/flash_attn/op_kernel/arch35/flash_attn_block_cube_nd.h:93-183,217-247,296-394`、`flash_attn_block_vec_nd.h:96-294,467-478,635-694`、`attention/common/op_kernel/buffer_manager.h:44-48`、`attention/common/op_kernel/buffers_policy.h`（BuffersPolicyDB）、`attention/common/op_kernel/buffer.h:209-227,276-322`
    - 动作（全部在本提交内完成，中间态不要求编译；开工前先 `git stash` 或确认工作区干净 + 记录恢复点）：
      ⓪ 预检：`grep -n "Matmul[KN]<" flash_mla_with_kvcache_block_cube_noquant_mla.h` 确认 L0C 现传参为 `mm*ResL0C.GetTensor<T>()`（Buffer 槽）→ 改后传静态 `LocalTensor`（C 参数）；同时确认 ascendc `MatmulK/N` 的 C 参数接受 LocalTensor（ascendc-api-best-practices 或本地 matmul.h 签名核实），若模板要求 Buffer 类型则保留 L0C 的 BuffersPolicyDB 但切换 LOCK_UNLOCK+EXTERNAL（记 deviation）；
      ① kernel（flash_mla_with_kvcache_kernel_noquant_mla.h）：删 `pipe_`(:91)、`InitMMResBuf`(:175-186)、kernel 级 `ubBufferManager_/l1BufferManager_/bmm1Buffers_/bmm2Buffers_/l1KvPBuffers_`(:61-68) 及注释掉的 gmBufferManager_/bmm2ResGmBuffers_；`Init()` 内删 `InitMMResBuf(workspace)` 调用(:150)；
      ② cube（flash_mla_with_kvcache_block_cube_noquant_mla.h）：删 `TPipe *tPipe`(:59)；`InitBuffers()`(:107-120) 重写为静态：UB CV 区两个 LocalTensor<uint8_t>(VECIN, 偏移, BUFCNT*BUF_BYTES)、L1 Q+KV/P（A1，按附录 A 表 1）、L0C（CO1, 0, 2×128K，静态 + 2 Mutex ID，见注 3）；L0A/L0B `BufferManager::Init(BUFFER_SIZE_BYTE_64K)` 静态重载 + `BuffersPolicyDB<L0A/L0B, INNER_CORE_SYNC, LOCK_UNLOCK, IdSource::EXTERNAL>` 手填表 2 ID（7..10）；`InitCubeBlock`(:99) 去 pipe/l1BuffMgr 参数；L1/L0C 的 `WaitFlag/SetFlag<MTE1_MTE2/MTE2_MTE1>(GetEventID)`(:302-307) 换 `Mutex::Lock/Unlock<PIPE_MTE2/PIPE_MTE1/PIPE_M/PIPE_FIX>(表 2 ID)`；新增 BUFCNT/BUF_BYTES constexpr 常量集 + `static_assert(L1Layout<=512K/UbLayout<=248K)`；
      ③ vec（flash_mla_with_kvcache_block_vec_noquant_mla.h）：删 `TPipe *tPipe`(:88)；**开工前按表 2 注 10 逐 HardEvent 类枚举 vec 事件 ID 并回填表 2（含 mte3ToVId[0/1]、vToMte3Id[0/1] 与各队列内部事件的落点）**；`InitBuffers()`(:681-699) 重写为静态：UB `struct UbLayout`（mm2/mm1 CV 区与 cube 同一偏移别名 + vec2Res + vec1Res + mask + softmax sum/max/exp + LSE + max/sumBrdcst + tmp，表 1）；9 处 TQue/TBuf(:102-114,661-699) 改静态 index 轮转（softmax sum/max/exp 仍 [mloop%2]、mask 仍 [loop%2]、stage1 单槽语义不变）；`GetTPipePtr()->AllocEventID/ReleaseEventID`(:703-718) 换固定 Mutex ID（表 2 vec 核内）；`InitVecBlock`(:126) 去 pipe 参数；`SoftmaxInitBuffer`(:661-679) 并入静态布局；**`ClearOutput`(:187-198) 改 FA 协同清零**：删 `InitOutputSingleCore`/`InitLseOutputSingleCore`(:200-239)、删静态 `initOutputEventId`(:67) 及其 Set/WaitFlag<MTE3_V> 节拍(:190,195,214-216,234-237)，调用 `AttentionCommon::InitOutput<OUT_T, EVENT_ID0, UB_ATTEN_POP_BUF_ADDR, UB_ATTEN_POP_BUF_ELE, true>(attentionOutGm_, totalOutputSize, vecCoreNum, 0)` + `AttentionCommon::InitOutput<float, EVENT_ID1, UB_LSE_POP_BUF_ADDR, UB_LSE_POP_BUF_ELE, true>(softmaxLseGm_, lseTotalSize, vecCoreNum, 3e+99)` + `SyncAll()`（tSize/layout 分支与 `IsInitAttentionOutGm()`/`isSoftmaxLseEnable` 门控语义保留不变；vecCoreNum=2×coreNum；精确参照 flash_attn_block_vec_nd.h:298-326 + attention/common/op_kernel/init_output.h:39-81；pop-buffer 静态地址/EVENT_ID 按表 1 注 5、表 2）；
      ④ fd（flash_mla_with_kvcache_block_vec_flashdecode_mla.h）：16 个 TBuf(:121-137) + `InitBuffers(TPipe*)`(:163-189) → 静态绝对偏移，**布局放 VECIN 0..95K**（6K×5+16K×4+1K=95K 顺序排布；不得镜像 FA 的 160K BASE/32K vec2Res 常量——见注 4），去 `pipe->Reset()`/`pipe->InitBuffer`；
      ⑤ cross-core 显式化（cube+vec 同批）：`WaitCrossCore/SetCrossCore`（cube :289,329,350-351,371；vec :158-161,409-410,424,433,457）改为显式 `CrossCoreSetFlag/CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE=4, PIPE>(表 2 CROSSCORE ID)`；kernel `CrossCoreBufferInit/UnInit`(:342-369) 拆除，`SetEventID/GetEventID`(:350-364) 换显式旗标；预置/drain 语义按表 2 注 8；
      ⑥ kernel `ComputeMm1/Mm2/ComputeVec1/Vec2`(:619-638) 改 runInfo-only 直调 `cubeBlock_.IterateBmm1(runInfo)` / `vecFaBlock_.ProcessVec1(runInfo)`（block 内部改用自身静态 buffer 与槽位成员）；Dummy 块（cube/vec）**无需镜像接口**（`if ASCEND_IS_AIC/AIV` 是 if-constexpr，未命中分支被丢弃，见 Verification strategy）——仅当编译报缺成员时补类型别名/静态常量；FD 无 Dummy 对应（kernel 模板 FD 槽恒为 Normal，无需同步）；
      ⑦ 同批锁定两侧 CROSSCORE/Mutex ID 与附录 A 表 2 完全一致。
    - 验收：`bash build_mla.sh --incremental`（或等价）退出码 0（-j20）；以下 grep 全部零命中（词边界正则，含注释/宏干扰免疫）：`rg -n '\b(TPipe|TQue|TBuf|GetTPipePtr|InitBuffer|AllocEventID|ReleaseEventID)\b' attention/flash_mla_with_kvcache/op_kernel/`（AllocEventID/ReleaseEventID 改为 block 内私有空实现的行为除外——以编译期零残留为准）；`rg -n 'initOutputEventId|InitOutputSingleCore|InitLseOutputSingleCore' attention/flash_mla_with_kvcache/op_kernel/` 零命中（ClearOutput 已换 FA 协同清零）；`static_assert` 未被触发；`tests/ut` 全量绿（对照 Todo 1 基线无新增失败）
    - QA：happy=编译门 0 + 词边界 grep 零命中 + UT 全量无新增失败；failure=编译报错/static_assert 触发 → 按附录 A 注 1 trim 后重编并记录 deviation；grep 残留 → 查漏补缺
    - 提交：`refactor(flash_mla_with_kvcache): static-tensor buffers + mutex/cross-core sync (single atomic wave)`（**仅在验收通过后** author 本提交；波内可用 wip 检查点提交，交付前 squash）
- [x] 3. 编排面：kernel Process 对齐 flash_attn（runInfo-only 终核 + Dummy 接口核对）
    - 参考：`attention/flash_attn/op_kernel/arch35/flash_attn_kernel_nd.h:580-591,626-652`（Process 内 InitBuffers/InitCrossCoreSync/AllocEventID 顺序、FlashDecode 段、FreeEventID/UnInitCrossCoreSync 收尾）；`flash_mla_with_kvcache.cpp:186-194`（`__DAV_C310_CUBE__` 选型）
    - 动作：① kernel `Process()`(:845-879) 改为 FA 同构：AIV 分支 `vecFaBlock_.InitBuffers(); vecFaBlock_.InitCrossCoreSync(); vecFaBlock_.AllocEventID();`，AIC 分支 `cubeBlock_.InitCrossCoreSync(); cubeBlock_.InitBuffers(); cubeBlock_.AllocEventID();`；section 循环调 `FlashAttention(sectionIdx)`/`FlashDecode(sectionIdx)`；收尾 `FreeEventID()`+`UnInitCrossCoreSync()`（收尾语义按表 2 注 8）；② `FlashDecode`(:831-843) 改 `vecFdBlock_.InitBuffers()` 无参（去 &pipe_），保留 `ICachePreLoad(2); SyncAll; FlashDecode; SyncAll` 序列；③ Dummy/Real sanity 核对：`grep -n "class FlashMlaWithKvcache.*Dummy" op_kernel/arch35/*.h` 枚举全部 Dummy 类，仅需确认被非丢弃分支使用的类型别名/静态常量存在（`if ASCEND_IS_AIC/AIV` 为 if-constexpr，无需接口齐平），编译通过即视为 Dummy 侧 OK；④ 复核 Compute*/IterateBmm*/ProcessVec* 签名全部 runInfo-only；
    - 验收：编译门退出码 0；**内核产物验证链（基线 §3 实测修正，build 树不驻留 .o）**：`build_out/*.run` 再生（mtime 晚于源改动）+ `sh <run> --extract` 内嵌 `FlashMlaWithKvcache_{d11cc332,e76bc2b1}*.o/.json` 2 组齐全 +（已重装时）installed custom_opp kernel 目录 .o mtime 更新；`rg -n '\bpipe_\b' attention/flash_mla_with_kvcache/op_kernel/arch35/` 零命中
    - QA：happy=编译 + 产物非空 + grep 零 `pipe_`；failure=某 TU 编译失败（多半是签名/头文件遗漏或 Macro 分支问题）→ 修复后重编
    - 提交：`refactor(flash_mla_with_kvcache): align kernel Process orchestration with flash_attn`
- [x] 4. 行为零变化 + 回归验证
    - 参考：Todo 1 基线清单（`.omo/baselines/mla-static-tensor-baseline.md`）；`tests/ut` 现有（git 跟踪）用例；算子 `tests/` 下实际存在的 pytests（以 Todo 1 现场核实为准）
    - 动作：① 有机会跑行为门：固定随机种子、单核单线程跑 `attention/flash_mla_with_kvcache/examples/test_aclnn_flash_mla_with_kvcache.cpp`（或稳定 pytests 用例），输出 dump 到 `build/baseline/attn_out_after.bin`，与 Todo 1 的 `attn_out_before.bin` 做字节级 diff（`cmp`/`diff -q`）；**显式覆盖"单任务 section"场景（仅槽 0 被使用）**——该 corner 直接检验表 2 注 8 的 drain 8 个 Wait 是否各有配对 Set（防悬死），用例加入行为门/UT 清单；② 跑 Todo 1 实际存在并记录的 UT 套件，对照基线（无新增失败）；③ 有 NPU 跑 `tests/pytests` functional_stc/functional_nc；④ 整理"未验证项清单"（无 NPU 时：行为门 diff、上板 pytests 未跑，如实列出）
    - 验收：行为门 `cmp` 全等 或 声明为"编译期+host 侧证据支持、运行时未实测"（不得默认写成已验证）；UT 无新增失败；未验证项清单完整
    - QA：happy=`cmp -s build/baseline/attn_out_before.bin build/baseline/attn_out_after.bin`（exit 0）+ UT 零新增失败；failure=diff 异常或 UT 新增失败 → 检查顺序（① 两侧 CROSSCORE/Mutex ID 是否与表 2 一致；② L0C 2 槽 128K Mutex 轮转语义是否破坏 bmm2 足迹；③ 静态布局偏移与表 1 一致；④ L1 P 写入偏移 cube/vec 一致），修复后追加提交 `fix(flash_mla_with_kvcache): ...` 并重跑本 todo
    - 提交：无（修复走追加提交）
- [x] 5. 全量终检
    - 参考：仓根 `build_mla.sh`；`op_host/CMakeLists.txt`（无改动预期）
    - 动作：① `bash build_mla.sh`（非 incremental 全量，-j20）；② 复核产物齐全（build/ 内 run 包/kernel bin）；③ 复核无构建警告新增；④ 复核 Scope OUT 无越界：`grep` 确认 `attention/flash_attn/`、`common/`、`torch_extension/` 零改动
    - 验收：退出码 0 + 产物齐全 + 越界 grep 零命中
    - QA：happy=编译 0 + grep 零命中；failure=编译失败按验证策略环境门处理（无 CANN 环境时如实标注"未真机验证"）
    - 提交：无（若 Todo 4 有修复则并入最后修复提交）

## Final verification wave

- [x] F1. Plan compliance audit
    - 动作：对照本计划逐条核验 Todos 1-5 完成且 Scope OUT 无越界（`git diff --stat -- attention/flash_attn attention/common common torch_extension docs` 为空 + `git diff -- template_tiling_key.h 相关路径` 为空 + 越界路径 grep 零命中）
    - 验收：检查清单全绿；越界 diff 全空
- [x] F2. Code quality review
    - 动作：评审全部改动：静态布局与附录 A 表 1 数值一致（逐字节核对 UB 合计 ≤248K、L1 ≤512K）、CROSSCORE/Mutex ID 表 2 在 cube/vec 两侧一致、`static_assert` 就位、Dummy/Real 接口齐平、注释规范（引用 flash_attn 行号段）、`rg -n '\b(TPipe|TQue|TBuf|GetTPipePtr)\b' op_kernel/` 零命中
    - 验收：评审意见清零或全部闭环
- [x] F3. Real manual QA：编译 + 回归必须通过
    - 动作：真实环境 `bash build_mla.sh`（非 incremental）记录退出码与产物；跑 `tests/ut` 全量对照 Todo 1 基线；有 NPU 跑 pytests 并对 Todo 1 基线无新增失败；交付说明显式列出未验证项；复核构建警告数不高于 Todo 1 基线
    - 验收：退出码 0 + 产物存在 + UT 无新增失败 + 警告数 ≤ 基线 + 未验证项清单完整（无 NPU 时逐项列出）
- [x] F4. Scope fidelity
    - 动作：确认行为零变化：① 实例化面计数一致（改动前后分别 `grep -c` template_tiling_key.h 的 DECL/SEL/INSTANTIATE 宏实例计数，相等）；② template_tiling_key.h 零 diff；③ def/aclnn 接口未变；④ 缓冲槽数/字节数与附录 A 表 1 一致（L0C 槽数与总量均不变，仅机制换为静态 + 2 Mutex）；⑤ examples 未删；⑥ `attention/flash_attn/`、`attention/common/`、`torch_extension/`、`docs/` 零改动
    - 验收：逐项（①-⑥）证据齐全

## Commit strategy

- Todo 2 为**单一大原子提交**（数据面同步面纠缠，跨核 ID 须同批锁定；git 恢复点 = 开工时 HEAD hash，Todo 1 已记入 `.omo/baselines/`）。
- 其余 todo 各一个原子提交：Todo 3 `refactor(flash_mla_with_kvcache):`、Todo 4 修复 `fix(flash_mla_with_kvcache):`、Todo 5 无提交。
- Git 写操作前执行 pre-commit 检查（ops-transformer-skill：pre-commit 检查、代码格式）。
- 不 rebase 已推送历史；全部提交留在当前分支（建议从 `mla-align-flash-attn` 完成后的当前分支继续）。
- 任何 trim/deviation（附录 A 注 1/注 3 触发）必须写进提交信息与代码注释。

## Success criteria

1. `bash build_mla.sh` 退出码 0，产物生成（编译必须通过——用户硬性要求）。
2. Todos 1-5 + F1-F4 全部完成。
3. 行为零变化：72 实例化面、tiling key 位编码、def/aclnn 接口、缓冲字节数均与改动前一致；PRELOAD_N 仍 1。
4. `attention/flash_mla_with_kvcache/op_kernel/` 零 TPipe/TQue/TBuf/GetTPipePtr。
5. 现有 tests/ut 套件（以 Todo 1 现场核实的基线为准）无新增失败；构建警告数不高于基线；pytests 有 NPU 跑、无则未验证项清单完整。
6. `attention/flash_attn/`、`attention/common/`、`common/`、`torch_extension/`、`docs/`、`template_tiling_key.h` 零改动（git diff/grep 验证）。