---
slug: mla-static-tensor-refactor
status: drafting
intent: clear
review_required: true
pending-action: 双高精度复核（momus + Oracle）→ 折叠发现 → 交付执行
---

# Draft: mla-static-tensor-refactor

> 前置：已完成的前一计划 `.omo/plans/mla-align-flash-attn.md`（todo 全部 [x]）只做"组织模式对齐"，**未包含**静态 tensor / Mutex 改造——本次请求正是其下一层。本 draft 独立记录新计划。

## Components (topology ledger)
<!-- id | outcome (one line) | status: active|deferred | evidence path -->
- K1 | kernel 层静态化：删 `AscendC::TPipe pipe_`、InitMMResBuf 改静态布局、kernel 级 BuffersPolicy（UB bmm1/bmm2、L1 l1KvP）改静态 LocalTensor + 显式 cross-core ID | active | op_kernel/arch35/flash_mla_with_kvcache_kernel_noquant_mla.h:61-91,175-186,342-369
- K2 | cube block 静态化：InitBuffers 重写、L0A/L0B 静态 manager + BuffersPolicyDB(LOCK_UNLOCK,EXTERNAL)、L0C 静态 CO1 + Mutex、L1 Q/KV/P 静态 + Mutex、IterateBmm1/Bmm2 同步改 | active | op_kernel/arch35/flash_mla_with_kvcache_block_cube_noquant_mla.h:59-120,287-372
- K3 | vec block 静态化：TQue/TBuf 全静态、UB struct 布局（与 cube 的 CV 区别名）、GetTPipePtr AllocEventID→固定 Mutex/旗标 ID、stage1/softmax/LSE 队列静态化 | active | op_kernel/arch35/flash_mla_with_kvcache_block_vec_noquant_mla.h:88-114,303-467,661-719
- K4 | FD block 静态化：TBuf(pipe->InitBuffer)→静态绝对偏移（镜像 flash_attn_block_vec_flashdecode.h:173-232 别名布局）、每 section InitBuffers | active | op_kernel/arch35/flash_mla_with_kvcache_block_vec_flashdecode_mla.h:121-189
- K5 | 同步机制对齐：显式 CROSSCORE_* 旗标 ID + InitCrossCoreSync/UnInitCrossCoreSync 模式、Mutex ID 分配表（0-13 区段）统一 | active | 四个 block 文件 + flash_attn 参照（flash_attn_block_cube_nd.h:68-91,217-231；flash_attn_block_vec_nd.h:75-93,256-274）
- B1 | 编译门 + 回归：build_mla.sh -j20（或等价 build.sh 命令）退出码 0、tests/ut 全绿、pytests functional（有 NPU 才跑、无则如实标注） | active | 仓根 build_mla.sh（前一计划已交付）+ tests/
- B2 | 行为等价验证：产物对比（有机会跑时）或编译+UT+诚实未验证清单；静态布局 static_assert 生效 | active | examples/test_aclnn_flash_mla_with_kvcache.cpp + build 产物

## Findings (cited: path:lines)
- **目标范式（flash_attn，改造成的样子）**：
  - 全 `attention/flash_attn/op_kernel/` 零 TPipe/TQue/GetTPipePtr（grep 证实）；buffer 全部 `LocalTensor<uint8_t>(TPosition, 字节偏移, 字节数)` 编译期确定。
  - 命名规范：`X_BUFCNT`=轮转槽数、`X_BUF_BYTES`=单槽字节，总分配 = BUFCNT×BUF_BYTES，取槽 `tensor[loop%BUFCNT*BUF_BYTES]`（flash_attn_block_cube_nd.h:93-121；flash_attn_block_vec_nd.h:96-133）。
  - 地址计算：struct + OFFSET_OF_MEMBER/SIZE_OF_MEMBER（cube_nd.h:153-183, vec_nd.h:196-245）或顺序 addr+=（cube_dn.h:149-173, vec_dn.h:176-218）。
  - L0A/L0B：`fa_base_matmul::BufferManager<L0A/L0B>::Init(BUFFER_SIZE_BYTE_64K)`（静态重载，buffer_manager.h:44-48）+ `BuffersPolicyDB<BufferType, INNER_CORE_SYNC, LOCK_UNLOCK, IdSource::EXTERNAL>` 手填 mutex ID（cube_nd.h:119-130,233-247）。
  - L0C：静态 `LocalTensor<uint8_t>(TPosition::CO1, 0, L0C_BUFCNT(4)×L0C_BUF_BYTES(64K))` + `Mutex::Lock/Unlock<PIPE_M>/<PIPE_FIX>(L0C_BUFFER_ID0+l0cBufId_)`（cube_nd.h:342-364）。
  - L1：静态 A1 + `Mutex::Lock/Unlock<PIPE_MTE2>/<PIPE_MTE1>(Q_L1_BUFFER_ID0/KV_L1_BUFFER_ID0+id)`（cube_nd.h:328-340）。
  - UB：静态 VECIN + `Mutex::Lock<PIPE_MTE3>/<PIPE_V>`（vec_nd.h:467-478,635-694）。
  - 核间：`CrossCoreSetFlag/CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE=4, PIPE>(id)` → 硬件 set_intra_block/wait_intra_block（dav_3510/kernel_operator_sync_impl.h:789-843）；AIV1 用 id+16（AIV0_AIV1_OFFSET，buffer.h:29-30）；AIV InitCrossCoreSync 预置 4 个 BMM 旗标（vec_nd.h:256-262）、AIC UnInitCrossCoreSync drain 8 个（cube_nd.h:221-231）；c1v1/c2v2 为 UB 2 槽、v1c2 为 L1 P 3 槽（CC_BMM1_0/1=0/1、CC_BMM2_0/1=2/3、CC_L1P_0/1/2=5/6/7）。
  - Mutex 类 = AscendC 硬件 pipe 互斥（get_buf/rls_buf，MAX_MUTEXID=27，kernel_common.h:136-189），**intra-core only**；全仓无 GM mutex、无原子量。
  - FD block 静态化：绝对偏移计算 + 与 FA block UB 区别名（flash_attn_block_vec_flashdecode.h:173-232）。
  - UB 预算断言：cube L1Layout ≤ 512KB（cube_nd.h:167）、vec UbLayout ≤ 248KB（vec_nd.h:219）。
- **MLA 现状（改造对象）**：
  - kernel_noquant_mla.h:91 `AscendC::TPipe pipe_`；:175-186 InitMMResBuf 用 `BufferManager::Init(&pipe_, 524288)`(L1) / `Init(&pipe_, mm1*2+mm2*2=160K)`(UB)；:64-68 kernel 级 BuffersPolicy：`bmm1Buffers_/bmm2Buffers_`(UB DB, CROSS_CORE_SYNC_BOTH)、`l1KvPBuffers_`(L1 3buff, CROSS_CORE_SYNC_FORWARD)、`l1BufferManager_/ubBufferManager_`(TPipe)。
  - kernel CrossCoreBufferInit/UnInit（:342-369）：AIV 侧 SetCrossCore×4；AIC 侧 SetEventID×3 + SetFlag/WaitFlag<MTE1_MTE2>（GetTPipePtr AllocEventID 分配）。
  - cube block_cube_noquant_mla.h:59 `TPipe *tPipe`；:113-119 `l0aBufferManager.Init(tPipe,65536)`、`l0cBufferManager.Init(tPipe,262144)`（注释"L0A B C 当前写死"）；:83-85 BuffersPolicyDB<L0A/L0B/L0C>（默认 SyncMode=SET_WAIT_FLAG, IdSource=INTERNAL）；:87 l1QBuffers SingleBuffer<L1>(TPipe)；:287-372 IterateBmm1/Bmm2 用 WaitFlag/SetFlag<MTE1_MTE2/MTE2_MTE1>(kvpSharedBuf.GetEventID...) + Wait/SetCrossCore。
  - vec block_vec_noquant_mla.h:88 `TPipe *tPipe`；:102-114 TQue/TBuf×9（stage1OutQue、attenMaskInQue[2]、stage2OutBuf、softmaxMax/Sum/ExpBuf[2]、maxBrdcst/sumBrdcst、softmaxLseQueue、commonTBuf）；:661-699 tPipe->InitBuffer 全部；:703-706 GetTPipePtr()->AllocEventID<MTE3_V/V_MTE3>×4、:715-718 ReleaseEventID×4；:158-161,409-410,424-457 Wait/SetCrossCore。
  - FD block_vec_flashdecode_mla.h:121-137 TBuf×13；:163-189 `InitBuffers(TPipe *pipe)` + pipe->Reset() + pipe->InitBuffer 每 section 重建（kernel :837 `vecFdBlock_.InitBuffers(&this->pipe_)`）。
  - 实例化面：Config9（mBase=64/s2=128/d=576=dv=512 之外另有 nope512+rope64 语义）→ {TND,BNSD,BSH}×{PA_BBND,PA_BNBD,PA_NZ}×mask×fd=36 组合×2 dtype=72 实例（template_tiling_key.h:21-27,203-208）。
  - buffer 数值（dtype 2B/4B）：mm1ResultSize=64/2×128×4=16K、mm2ResultSize=64/2×512×4=64K、UB CV=2×16K+2×64K=160K；mm12RightSize=128×576×2=144K×3=432K（L1 KV/P）；Q=64×576×2=72K（单槽）；L0A/L0B=64K、L0C=256K。
- **方案映射**：见 approach；逐组件任务 = K1→K5+B1+B2。
- **风险**：
  - UB 静态全额预留 ≈ 160K(CV)+vec(64K stage2+8.4K stage1+16K mask+2.25K softmax+4K LSE+0.5K tmp) ≈ 255K vs 256K UB（flash_attn 用 ≤248K 保守断言）——需精确布局 + static_assert；超限 trim 优先级（方案内写死）：stage1 复用 CV 空闲槽 → mask 降单槽 → LSE 复用 commonTBuf 区。
  - L1 512K 预算内 P/Q/KV 布局与 flash_attn 的 L1Layout 结构不同（MLA 有 576 宽 k_cache、k==v、P 与 K-rope 复用）：**不得照抄 FA 的 L1Layout 结构**，必须按 MLA 现布局（3×144K KV/P + 72K Q）静态化，L1 P 首槽偏移与 vec 写 P 的目标偏移保持一致。
  - L0C 现为 2×128K(DB policy) vs flash_attn 4×64K 静态——总量 256K 不变，但槽数/ID 分配随架构对齐变化，需在 IterateBmm1/Bmm2 内同步 l0cBufId_ 轮转逻辑。
  - 编译宏 `__DAV_C310_CUBE__`/`ASC_DEVKIT_MAJOR >= 9`（kernel_basic_intf.h）已就绪，Mutex 可用（同 flash_attn 编译面）。

## Open assumptions (announced defaults, 可在 gate veto)
- **行为零变化**（延续前一计划原则）：不改 tiling 数值、不改实例化面（36×2=72）、不改 def 输入集合、不改 aclnn 接口；只改 kernel 内部 buffer 管理与同步机制。
- **保持现有 buffer 槽数**：CV UB 2 槽、L1 KV/P 3 槽、Q 单槽、L0C 2→4 槽（总量 256K 不变，槽数对齐 FA 的 4×64K 以契合 Mutex ID 轮转，若导致行为/性能疑虑回退为 2×128K 静态）。**不加深流水**（PRELOAD_N 仍 1，ring 2）——性能调优不在本次范围。
- **语义命名保留**：FlashMlaWithKvcache* 类名不改；对齐的是机制不是命名。
- **FD block 默认纳入**（全量对齐 flash_attn 静态 FD）。
- **cross-core 同步显式化**：从 Buffer 策略层（SetCrossCore/WaitCrossCore）改为 flash_attn 式显式 CrossCoreSetFlag/WaitFlag + InitCrossCoreSync/UnInitCrossCoreSync。
- **L0A/L0B/L0C 的 policy 层保留**（MatmulBase/MatmulK/N 接口需要 Buffer 参数），仅切换 SyncMode=LOCK_UNLOCK + IdSource=EXTERNAL + 静态 manager Init。
- **验证惯例**：编译门（-j20）+ tests/ut 全绿 + pytests（有 NPU 才跑，无则如实列未验证项）+ 有机会跑行为产物对比；沿用前一计划"诚实未验证清单"约定。
- 不改 `attention/flash_attn/`、`common/`（只读参考）；不新建 utils/ 目录。

## Decisions (with rationale)
- D1 架构全部对齐 flash_attn：静态 InitBuffers + BUFCNT/BUF_BYTES + Mutex + 显式 cross-core 旗标（用户点名"静态tensor编程"与"mutex这种同步"）。
- D2 保留 BufferManager/BuffersPolicy 库层（fa_base_matmul 是共享库，flash_attn 同样在用）——只改调用参数（静态 Init、LOCK_UNLOCK、EXTERNAL ID）。
- D3 新计划 slug=mla-static-tensor-refactor（与已完成 mla-align-flash-attn 分离；后者作为结构前提已就位，不改动其成果）。

## Scope IN
- K1-K5 全部：kernel/cube/vec/fd 四层静态化 + 同步机制对齐 + Mutex ID 表 + CROSSCORE ID 表。
- B1 编译门与回归；B2 行为等价验证。
- 若 UB/L1 静态预算超限，按方案内 trim 优先级裁减（记录为 announced deviation）。

## Scope OUT (Must NOT have)
- 不深度改流水（PRELOAD_N 保持 1）；不做性能调优/精度调整；不改 tiling 数值与实例化面；不改 aclnn/def 接口。
- 不改 attention/flash_attn/、common/、torch_extension/、docs/（只读参考）。
- 不重命名算子/类为 GQA 语义；不删除 examples。
- 不迁移 flash_attn 测试用例资产。
- 不引入新的外部依赖。

## Open questions (interview)
- ~~Q1 验收口径~~ → 已答：**A. 纯架构对齐，行为冻结**（保持 PRELOAD_N=1、槽数、tiling 数值全部不变；验收 = 编译通过 + UT 绿 + 行为零变化证据；不做性能调优、不采集性能基线）
- ~~Q2 FD block 范围~~ → 已答：**A. 一并静态化**（K4 active，FD block 全量对齐 flash_attn 静态布局）
- ~~Q3（增量追加）~~ → 用户补充：**`InitOutputSingleCore()` / `InitLseOutputSingleCore()` 也替换**为 flash_attn 式协同清零（`AttentionCommon::InitOutput<..., EVENT_ID0/1, popBufAddr, eleSize, true>` + SyncAll；删 `initOutputEventId` 节拍）。已折叠进 plan：Scope K3 项、表 1（pop-buffer 两行 + 注 5）、表 2（EVENT_ID0/1 行）、Todo 2 ③ 动作与验收 grep。

## Approval gate
status: approved（用户"开始写计划"）→ plan 已写入 .omo/plans/mla-static-tensor-refactor.md
approach: 按 D1-D3 全量静态化改造（K1 kernel / K2 cube / K3 vec / K4 fd / K5 同步机制 + B1 编译回归 + B2 行为等价验证），**行为零变化**：PRELOAD_N=1、缓冲槽数（L0C 2×128K→4×64K 为唯一结构例外）、tiling 数值、72 实例化面全部不变；验收硬性条件 = 编译通过（build_mla.sh -j20 退出码 0）+ tests/ut 全绿 + 无 NPU 时如实列未验证清单；FD 一并静态化。
metis-review: 已执行（NO-GO 复核）→ 折叠修正：表 1 预算按源事实修正为 245.25K（≤248K 断言，默认无需 trim）；表 2 增补 L0A/L0B Mutex ID 行（7..10）+ 注 8 预置/drain 语义；注 4 新增 FD/vec UB 重叠别名说明；Todo 1 基线写入 .omo/baselines/、Todo 2 增 Matmul-L0C 预检与词边界 grep、Todo 3 增 Dummy 全集枚举、Todo 4 行为门改字节级 cmp + 确定性、F1/F4 可执行化、Success criteria 增 docs/。所有 BLOCKER/MAJOR/MINOR 已闭环。
high-accuracy-review（用户"复核一下"触发）: 已执行 momus + oracle 双评审并折叠（详见下方 sum 摘要）：
- momus：GO（条件）——F1 全量 MLA 文件名短名（已全局补全 flash_mla_with_kvcache_ 前缀）；F2 `common/op_kernel`→`attention/common/op_kernel`（已 replaceAll，F1/F4 pathspec 补 attention/common）；F3 测试资产未被 git 跟踪（tests/ut 仅框架、无跟踪 .py、无 test_flash_mla_with_kvcache_tiling.*）→ Todo1/4/Success5 改为"现场核实为准"；F4 examples 全路径（已替换）；F5 FD TBuf 13→16（已改）；F6 vec 4 事件变量 1:1 映射（已并入注 10 全额枚举前置）；F7 WIP 提交无法用 git checkout 撤销（已改 git reset --hard <开工时 HEAD hash>）。
- oracle：NO-GO→修订后 GO——B1 BLOCKER：L0C 4×64K 装不下 bmm2 的 128K 足迹（64×512×fp32），已把 2×128K 静态+2 Mutex 转正为主设计（表1/注3/表2/Scope/Todo2② 全部同步，剔除"唯一结构例外"字样）；B2 BLOCKER：FD 镜像 FA 160K BASE 会溢出 248K 且与 64K vec2Res 竞态 → FD 静态布局改放 VECIN 0..95K（注4/Scope4/Todo2④ 改）；#3 MAJOR vec 事件 ID 全额枚举前置（注10）；#4 MAJOR EVENT_ID0/1 是 Mutex 非事件（表2 已标 Mutex + 注9）；#5 MINOR pop-buffer 安全论证改 c2v2 反向门控（注5 重写，SyncAll 非跨核保证）；#6 MINOR 单任务 section corner（Todo4 ① 已加）；#7 MINOR Dummy 接口镜像不必要（if-constexpr 丢弃分支，Verification/Todo2⑥/Todo3③ 已放宽）；#8 MINOR 原子波理由精化（数据面编译原子/同步面运行时失败，风险段已改）。
plan-status: 双复核已完成且全部发现折叠闭环；计划可交付执行。
next-workflow-action: 交付说明 → 用户 /start-work 执行。