# Learnings — mla-static-tensor-refactor

Conventions, patterns, and successful approaches discovered during work on this plan.

_Auto-scaffolded by /start-work. Append new entries below - never overwrite._

---

## [2026-08-25 02:31 UTC] T1: 基线快照

1. **基线数字**：recovery hash `b2d3abaa903a21d74c81bdefc5a641adb8abdf59`（分支 feat/flash_mla_with_kvcache）；工作区净度仅 `?? .omo/`；全量编译（`rm -rf build build_out && bash build_mla.sh`）EXIT=0 + `[build_mla] SUCCESS`，kernel .o 2 个（d11cc332/e76bc2b1），.run 包 6547675B；**构建警告 = 10**（host 侧：-Wclass-memaccess×1、-Wunused-parameter×2、-Wundef(__CCE_AICORE__)×2、-Wshadow(bnStride/n2Stride)×4、-Wunused-macros×1）——Todo 5 比对基准。
2. **重大环境事实：当前 HEAD 上 flash_mla_with_kvcache 的 tests 资产已整体移除**——提交 `a3377fc25 "test: remove flash_mla_with_kvcache/tests (pytests/assets/ut)"`（在 HEAD^ 位置）。后果：UT 门 `build.sh -u --ophost --ops=flash_mla_with_kvcache --soc=ascend950 -j20` 实测 **0 用例**（exit 0；CMake 打印 `_OpsTestUt_UTestCaseLibraries and _OpsTestUt_UTestAclnnCaseLibraries both empty`，0 个 Built target）；pytests harness `.py` 源全部缺失（目录只剩 `__pycache__`/pyc）：`test_flash_mla_with_kvcache.py`、`test_cases/functional_stc.py`、`functional_nc.py`、`assets/impl/{golden,compare,graph,inputs}.py`、`assets/{gen_cases,spec}.py` 均不存在。**Todo 2/4 的"UT 全绿"约束自动退化为空套件**，36P/7F 与 10P/4F 是资产尚在时代的历史参照、无法复跑（不编造用例名，仅保留 KEY-1 根因：host prefill tilingKey 恒 3672322 vs CSV 1575170）。
3. **上板现状**：有 NPU（Ascend950PR 单卡空闲）；custom_opp 已装 flash_mla_with_kvcache（kernel bin 与构建产物 hash 同名），**flash_mla_with_kvcache_metadata 只有 proto 头、无 kernel bin** → examples 行为门需先补装 metadata 包；pytests 因资产缺失本轮未跑。行为门 before 产物无法产生 → Todo 4 改为"重装双算子包后同轮 before/after"策略。
4. **build_mla.sh --incremental**：grep 确认支持（:27-28 注释），但按 MUST DO ⑤ 基线 & Todo 5 终检一律全量命令，不采信增量 0 退出码（已知静默跳过 kernel 重编陷阱，trigger 后按 mla-align learnings Todo4-4 判定法核对 .o mtime > 源）。
5. **UT 资产恢复线索（如需）**：git 历史 a3377fc25 的父提交（如 40ca9dc59/a3377fc25 之前的树）可 `git show <sha>:attention/flash_mla_with_kvcache/tests/...` 出原文件；当前任务不恢复（Scope OUT），仅记录。
## [2026-08-25] T2: 大波静态化

### 注10 枚举结果（详见 decisions.md）
vec 核内 Mutex：0(vec2Res)、1(EVENT0)、2(vec1/stage1)、3(EVENT1)、4(lse)、5(sumBrdcst)、6/7(mask 预留注释)、8(maxBrdcst)。
mte3ToVId[0]/[1]、vToMte3Id[0]/[1] 旧 TPipe 事件全部收敛：
- mte3ToVId[0] + vToMte3Id[0/1] → Mutex 0（单槽 vec2Res，V/MTE3 成对）
- mte3ToVId[1] 无 mid-run 使用 → 删除
- stage1/lse/sumBrdcst/maxBrdcst TQue 内部事件 → Mutex 2/4/5/8
- mask 走 common 4 参自同步（FetchEventID + Set/Wait MTE2_V），不占显式 ID；6/7 仅注释对齐 FA 排布

### 实际改动点（5 个文件）
1. kernel 头：删 `AscendC::TPipe pipe_`、InitMMResBuf、kernel 级 5 个 buffer manager；Init 去 `InitMMResBuf(workspace)`；
   `InitVecBlock/InitCubeBlock` 去 TPipe/l1BuffMgr 参数；删 `CrossCoreBufferInit/UnInit`；Process 改 FA 编排
   （AIV: InitBuffers→InitCrossCoreSync→AllocEventID；AIC: InitCrossCoreSync→InitBuffers→AllocEventID；收尾 FreeEventID→UnInitCrossCoreSync）；
   FlashDecode 去 `&this->pipe_`；Compute* 全 runInfo-only（IterateBmm1/2、ProcessVec1/2）。
2. cube 头：静态 InitBuffers（UB mm2 2×64K@0 + mm1 2×16K@128K；L1 Q 72K@0 + KV/P 3×144K@72K，static_assert ≤512K；
   L0C 2×128K 静态 + l0cBufId_ 轮转 + M/FIX Mutex 对；L0A/B `BufferManager::Init(64K)` 静态重载 +
   `BuffersPolicyDB<..., INNER_CORE_SYNC, LOCK_UNLOCK, IdSource::EXTERNAL>` Init(2×32K, id 7/8, 9/10)（AllocEventID 内）；
   KV/P 槽 Mutex 保持从 bmm1 数据就绪到 bmm2 MatmulN 消费完释放（= 旧 MTE1_MTE2 语义）；
   Wait/Set<MTE1_MTE2/MTE2_MTE1/FIX_M/M_FIX>(GetEventID) 全部换 Mutex::Lock/Unlock<PIPE_*>；cross-core 显式化。
3. vec 头：静态 UbLayout（struct + static_assert ≤248K，字节序=表 1）；9 处 TQue/TBuf 全改静态 LocalTensor +
   Mutex；ClearOutput 换 `AttentionCommon::InitOutput<OUT_T, EVENT_ID0=1, 0, 32K/ele, true>` +
   `InitOutput<float, EVENT_ID1=3, 32K, 32K/ele, true>` + SyncAll（删 initOutputEventId/InitOutputSingleCore/InitLseOutputSingleCore）；
   去 tPipe；AllocEventID/FreeEventID 空实现（FA 同款）；ProcessVec1/2 显式 CrossCoreWait/SetFlag。
4. fd 头：16 TBuf → 13 个静态 LocalTensor（fdSinkCopyInBuf/fdSinkValueBuf/fdSinkTmpBuf 三个从未 InitBuffer/使用 → 直接删除）；
   InitBuffers() 无参静态布局 VECIN 0..95K（6K×5+16K×4+1K）；去 pipe->Reset()。SYNC_*_FLAG raw 字面量旗标保留。
5. flash_mla_with_kvcache.cpp：仅 1 行过时注释改写（"kernel 自持 TPipe" 已被本波移除 → 写明静态模型）。**DEVIATION**：
   "唯一改动面=4 文件/不改 .cpp" 与验收门 A（op_kernel/ 全目录 0 个 TPipe token，含注释）冲突——.cpp:220 注释含
   2 个 `TPipe` token，不改则门 A 恒失败；纯注释改动（无行为影响，.o 哈希 d11cc332/e76bc2b1 保持 dtype 派生未变），
   提交含第 5 个文件的注释行。

### DEVIATIONS（无其他 trigger）
- mask 6/7 未实际分配 Mutex ID（common 自同步覆盖）——表 2 语义等价，注释维护。
- Kernel 级 CrossCoreBufferInit 的 3×MTE1_MTE2 AIC 预置不再需要：静态 Mutex 初始为解锁态，且末任务 bmm2 必然 Unlock。

### 编译门证据
`rm -rf build build_out && bash build_mla.sh`（内置 -j20）：日志 `[build_mla] SUCCESS`；
构建警告 10（= Todo 1 基线 10，host 侧不变，kernel 零新警告）；
`build_out/*.run` 再生（11:17:08 > 源改动 11:13:30）；
`--extract` 内嵌 `FlashMlaWithKvcache_d11cc332fbed1b93524f5b5a4b634f08.o/.json` +
`e76bc2b10bf573b91e21de09b0aa527a.o/.json` 2 组齐全（**哈希未变**：dtype 派生，模板签名不变，与基线 §3 一致）。

### grep 结果
- A 组 `\b(TPipe|TQue|TBuf|GetTPipePtr|InitBuffer)\b` op_kernel/ → **0 行**
- B 组 `initOutputEventId|InitOutputSingleCore|InitLseOutputSingleCore` → **0 行**
- 越界：`git diff --stat -- attention/flash_attn attention/common common torch_extension docs` = 空；
  `template_tiling_key.h` diff = 空；实例化面计数不变（tiling key 零 diff + 产物仍 2 组 .o）。
### static_assert
cube `L1Layout ≤ 512K`、vec `UbLayout ≤ 248K` 编译通过未触发；L0C 2×128K 足迹由 bmm2 单次 Mmad 128K 约束（注 3）。

## [2026-08-25] T2 追加：oracle 检出缺陷修复（vec softmax 折叠丢失首 S2 片守卫）

**缺陷**：206aadb75（T2 初提交）中 `Vec1PostProcess` 的 `UpdateExpSumAndExpMax` 被无条件折叠。
基线 b2d3abaa9:413-414 有守卫 `if (likely(!runInfo.isFirstS2Loop))`。
首 S2 片走 NoUpdate VF 族（ProcessVec1Vf updateFlag=false）——只写 slice 局部 sum/max，不填共享快照
`vec1ApiTmpBuf_`；无条件折叠读脏快照，破坏单 S2 片任务的 sum/max/exp 轮转（LastDivNew/LSE/RowInvalid 出错）。

**一行修复**（按 oracle 指定，不引入 ResetSoftmaxBuffer）：
```cpp
if (likely(!runInfo.isFirstS2Loop)) {
    UpdateExpSumAndExpMax<T>(sumUb, maxUb, expUb, sumUb, maxUb, vec1ApiTmpBuf_, runInfo.actVecMSize);
}
```
`SoftmaxDataCopyOut`（内部 isLastS2Loop 守卫）保持原条件。

**复验**：`rm -rf build build_out && bash build_mla.sh` 退出 0、`[build_mla] SUCCESS`、error 0、warning 10（基线）、
`.run` mtime 11:50:36 > 修复编辑；`--extract` 内嵌 2 组 `FlashMlaWithKvcache_*.o`；A 组/B 组 grep 零命中；
越界 diff 空；tiling key 零 diff；`grep isFirstS2Loop` 现于折叠点 454 行有守卫。
pre-commit 全 Passed → **amend**（非新增 commit）：`206aadb75` → **`e89731566`**（消息不变，仍单原子在 b2d3abaa9 之上）。

## [2026-08-25] T3: 编排面核验（Process 对齐 FA + Dummy sanity）——1 处小修

### 核验结论（详见 /tmp/t3_evidence.txt，纯验证为主）
1. **Process() 已 FA 同构**（kernel_noquant_mla.h:791-830 vs flash_attn_kernel_nd.h:626-652）：AIV/AIC 分支
   InitBuffers/InitCrossCoreSync/AllocEventID 顺序、section 循环 FA/FD、收尾 FreeEventID+UnInitCrossCoreSync
   逐行比对一致；cross-core 预置/drain（注 8）：AIV 预置 c1v1+c2v2 4 旗标、AIC drain 8 旗标（×{本,+16}）与
   FA vec_nd.h:256-262 / cube_nd.h:221-231 完全对齐；CROSS_CORE_SYNC_MODE=4、BMM1_0/1=0/1、BMM2_0/1=2/3、
   AIV0_AIV1_OFFSET=16。
2. **Process 外层多包 `aicIdx<coreNum` 门**：AIC aicIdx=GetBlockIdx()∈[0,coreNum)（:167）、AIV aicIdx=aivIdx/
   SubBlockNum（:170），FA_AIV_CORE_NUM 分发下恒真 ⇒ 与 FA 语义等价、非缺口（防御性冗余，不改）。
3. **FlashDecode 缺口修复（本 todo 唯一改动）**：T2 交付的 FlashDecode 缺 FA:590 的尾 `SyncAll()`——计划契约
   action② 明确"保留 ICachePreLoad(2); SyncAll; FlashDecode; SyncAll 序列"。FreeEventID（FD 内部 8 旗标
   WaitFlag drain）不等价于全 pipe 屏障（FA:226 注释：防跨 section 流水残留/死锁）。补 1 行
   `AscendC::SyncAll();`（:786，FlashDecode 与 FreeEventID 之间）+ FA:590 参照注释。无死锁风险：SyncAll 只依赖
   pipe 完成、不依赖旗标状态。
4. **Dummy sanity**：2 个 Dummy（cube:482 / vec:866），成员足够：cube Dummy 有 mBaseSize/LAYOUT/PAGE_ATTENTION/
   Q_T/KV_T/MM_T 等；vec Dummy 有 HAS_MASK/FLASH_DECODE/OUT_T 等；FD 无专属 Dummy（AIC TU FD 槽复用 vec 哑块，
   cpp:186-194）。`if ASCEND_IS_AIC/AIV` 常量条件分支 + 弃用分支不实例化依赖调用，编译通过即 OK，无需接口齐平。
5. **签名 runInfo-only 全绿**：kernel ComputeMm1/2(:564/:569)、ComputeVec1/2(:574/:579) → cube IterateBmm1/2
   (:372/:440)、vec ProcessVec1/2(:204/:463)，全部 (RunInfo) 单参；ProcessVec1Nd/ProcessVec2OnUb 为 block 私有
   辅助（LocalTensor 参数），不涉契约。
6. **编译门**：全量重编 EXIT=0 + SUCCESS；.run 提取 2 组 .o/.json（d11cc332/e76bc2b1 hash 不变）；mtime 晚于源；
   `\bpipe_\b` grep 0 命中。警告 10 基线由 Todo 5 复核（本 log 原始 grep 71 行含构建噪声，不作数）。
7. **给 Todo 5 的提醒**：本次修复改变了 kernel 头，但 FA 模板签名不变、产物 hash 不变（dtype 派生）——实例化面
   计数仍为 2 组；tiling key 零 diff。UT 空套件（a3377fc25 已移除），如实不重跑。

## [2026-08-25] T4: 行为门实测（A档部分达成；FD 路径发现真实行为差异）

### 最终判定（诚实降级声明）
- **字节级全等：3/4 官方配置达成**（fp16-b2 / bf16-b2 / b1-single-section；attn_out+lse+metadata+输入全 cmp 相等）。
- **1/4 不等：fp16-causal-b8-multisection**（batch=8, kvLen=32768, 17 heads, 8 section，含 FlashDecode）。
  b8 差异：85.25% 元素不等、1~80 ULP、absdiff p50=1.14e-5/max=8.2e-5；golden(b4=0.00% fail / b5=6.88%)；
  b4/b5 **两侧各自确定性复现**（各连跑两次字节全等）→ 真实行为差异而非调度噪声。
- 差异定位：kvLen 扫描 probe（batch1, 2 token, 1 head）——kvLen∈{128,140,192,256,512} 全等 + golden PASS；
  kvLen=1024/4096 不等 + b5 golden FAIL(33~44%)。→ **差异集中在 FD(FlashDecode) 参与的大 KV 路径**；
  FD 区块 diff 仅 `TBuf→LocalTensor` 机械替换（同偏移/轮转式），无算术语句变化；vec 块变化=ClearOutput 换
  FA 协同清零（值与语义同基线）+ Mutex/旗标显式化 + kernel FD 段 SyncAll(T3 已加)。嫌疑面=FD 与 vec 互操作/
  协同清零时机，具体根因留给后续 fix 任务（本 todo 不得改产品代码，未动任何代码）。

### 可复用事实（下轮直接用）
1. **example 上板需要 `ASCEND_CUSTOM_OPP_PATH=<live vendor dir>`**：不设置 → AICPU metadata 报 107000
   （ACL_ERROR_RT_PARAM_INVALID，根因 `BinaryGetFunctionByName` 找不到 kernel，非 .so 陈旧——重编 .so 与
   现装 md5 相同）；设置 → metadata 立即可跑。安装 .run 时必须 unset（默认装到 $ASCEND_OPP_PATH/vendors）。
2. **example 自带的 3 个用例在本环境并非全 PASS**：bf16-b2 恒 FAIL(2.0%, atol1e-4/rtol7.8e-3 超)；
   b8-multisection v5(before) PASS、after FAIL(6.88%)。before 也非"全绿"——行为门判据 = **before/after
   字节比较**，不是 example 的 golden 通过率。
3. **b8 差异的模式**：每个 batch 的 token 0..6 各 head 均异、token 7（末 token）17 头全等——
   与 mask_mode=3 因果窗 / FA-FD 窗口边界相关，可作为 fix 任务的二分锚点。
4. **b1-single（单任务 section）角**：kvLen=4/1 token/1 head → sectionNum=1 且 metadata FA 区仅 core0 词非空、
   cores1..35 全 0 → 仅 slot 0 被使用；运行不挂、输出与基线字节全等 → **注 8 的 drain 8 旗标配对位在
   单任务边界无悬死**（直接验证）。
5. **双树对比方法**：同一 driver 二进制（rpath 指向 live op_api/lib）+ 两套内核包轮换安装，
   每次安装后核对已装 kernel .o md5 与对应 .run 提取物一致；libcust_opapi.so b4/b5 同 md5
   （host aclnn 层未变）→ 差异纯粹来自内核 .o。
6. **构建噪声**：全新 worktree 首次全量 build 会多 28 行 vendored protobuf 三方警告（protobuf_src-prefix），
   应用级警告 b4=10/b5=10 一致；比对警告一律排除 `protobuf_src` 行。
7. **b8 案例 dump 体积**：in_k.bin 302MB/case，4 案例全量约 620MB——/tmp 空间充足即可，勿重复多跑。
8. **UT/pytests**：UT 空套件 exit 0（CMake both empty），pytests 资产缺失未跑；两门"如实记录"。
## [2026-08-25] T4-b: 掩码槽竞态 RCA + 修复（行为门从"DIFFER"翻转为字节全等）

### RCA 结论（修正 T4 的 FD 假说）
- **isFd=0 实证**：probe-1024/4096/b8 的 metadata header 均为 `isFd=0`、FD 区全零 → **无 FlashDecode 参与，全部纯 FA**。原"kvLen≥1024 触发 FD"定位错误（准确说法：多 S2 片 + 掩码尾键）。
- **真因 = mask 槽跨 S2 轮次槽复用的 V→MTE2 屏障丢失**：静态化后 `ubMaskBuffers_[loop%DB]` + common 4 参自同步（内部 FetchEventID MTE2_V Set/Wait + PipeBarrier<V>）只保证单次加载的 MTE2→V 顺序；旧深度1队列的 EnQue/DeQue（V 读完成前不许覆写）语义缺失。末 S2 片掩码可能读到上一轮数据 → 被掩码键被当有效键 → sum/lse 偏大 → 整行 ULP 漂移。
- **确定性/竞态实证**：基线 5/5 样本全部确定净值；修复前 probe-512 在"基线值/脏值"间翻转（5 样本 3 脏 2 净，run2==基线字节全等）；probe-1024/4096/b8 恒脏（构建时序定向解决）。E1 实验（ClearOutput 换基线 per-stripe）证明 ClearOutput 与主 delta 无关。

### 修复（commit 0937e04ee，1 文件 16+/3-）
镜像 FA `flash_attn_block_vec_nd.h:92-93,728-746` 的 per-slot 双 pipe 生产-消费握手，落地表 2 预留的 `UB_IN_MASK_EVENT0/1=6/7`：
- 加载侧 `Mutex::Lock/Unlock<PIPE_MTE2>(6+slot)`（main+pre 双槽）包 AttenMaskCopyIn；
- 消费侧 ProcessVec1Nd 尾部 `Lock/Unlock<PIPE_V>(6+slot)`×2。
mask 仍走 common 自同步，仅在调用点补跨轮次缺口。无算术/布局/接口/tiling/实例化面变化。

### 复验矩阵（修复后 3-5 样本/配置）
4 官方配置 + 3 probe **全部确定性 + attn_out/lse/metadata 与基线(b4 dumps)字节全等**；
golden：b8 恢复 PASS（修复前 FAIL 6.88%）、bf16-b2 保持改前既有 FAIL（字节全等，未动）；
probe-512 修复前翻转→修复后 5/5 恒定且==基线。构建 exit 0、警告 10、grep/越界门全 0、UT 空套件 0P/0F。

### 沿用要点（下轮直接用）
1. **mask 槽静态轮转必须补 per-slot 双 pipe 握手**（FA 5 参版本在 common 内做，MLA common 是 4 参无 ID → 调用点补 Lock<MTE2>）。其他 vec 异步槽（vec2Res/vec1/lse/brdcst）T2 已用 Mutex 覆盖，无同类缺口。
2. **行为门判据永远先跑"确定性"**：竞态可能被构建时序伪装成确定性回归；probe-512 这类临界形状是竞态探针（修复后必 5/5 恒定）。
3. **isFd/FA 结构判据兜底**：分析 kernel 行为先解 metadata（FA/FD 区、每 section 每核区间），避免凭规模猜路径。
4. 证据：/tmp/t4b_evidence.txt、/tmp/t4b_fix{1..5}/、/tmp/t4b_fin{1..3}/、build/baseline/t4_after_fix/、/tmp/t4b_b4s{1..5}/（基线 5 样本）。
