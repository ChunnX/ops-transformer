# BSA post pooling 二次融合 — 旧 Session 交接摘要

> 来源 session：`ses_f99c9c237ffe27u7CiIMDuo51Q`（"BSA post pooling二次融合"，2026-09-03 ~ 09-08）。
> 该 session 上下文已超限无法继续，新 session 请以本文件为起点。路径未标注时均相对 `/home/sll/bsa_select_block_mask/post_pool/ops-transformer`。

## 1. 任务与代码改动状态

目标：BSASelectBlockMask 算子 post pooling 二次融合（post 折叠并入 softmax 二趟，删除独立 post 相位）。

已完成（静态自检通过）：
- `reduce.h` AR 语义确认（任意行数支持），设计敲定
- 重写 `bsa_vec_post_pool_service.h`：融合折叠（列折叠 + 行折叠 + 除 count + Cast + 写 GM）
- `bsa_vec_sm_service.h`：SoftmaxSecondPassAndCast 二趟内调用 post pool 折叠
- `bsa_vector_service.h`：接口转发
- `bsa_select_block_mask_base.h`：删除独立 post 相位 + 行分区 postXBlock 对齐
- op_host tiling：post 模式 `attnScoreSize=0` + staging 条件分配
- arch35 全套镜像改动

**未完成**：编译装包 + 上板验证（todo 唯一 pending 项）。

## 2. 性能分析定稿结论（客户 case：B1N1 S=794880, bx8 + post16×16）

实验目录 `794880_bx8_by8_post16x16`（另有 `794880_bx8_by8_singlepool` 对照）。

端到端 192.64ms 拆解：

| 环节 | 耗时 | 占比 |
|---|---|---|
| Cube mac（纯矩阵乘） | 9.6 ms | 5% |
| Cube fixpipe（写 ~39.5GB FP32 scoreFp32 scratch） | 104.3 ms | 56% |
| AIV mte2（读回 scratch） | 46.6 ms | 与 vec 部分重叠 |
| softmax 两趟 + post 折叠 | 62.6 ms | 33%（post 融合增量≈0，被流水掩盖） |
| Q/K 一次 pool | 4.8 ms | 2.5% |
| TopK | 6.9 ms | 3.6%（粗粒度 6210²=38.5M 上 radix 选择，已非瓶颈） |

核心结论：
- **~155ms（83%）是 scoreFp32 scratch 写+读桥**（fixpipe 104 + mte2 47）。
- 流水结构：k 循环内 matmul↔pass1 健康重叠（AIC 慢侧 2.78ms/chunk，AIV pass1 1.65ms 被掩盖）；**pass2 期间 AIC 因 V1C1 单缓冲 WAR 完全挂空 ~45–51ms**；AIV tile 内 mte2/vec 串行（scoreUb 单缓冲 131KB 放不下双份）。
- 双缓冲 scratch 实测收益 +9ms（消除 AIC 空转）。
- fixpipe 104.3ms 构成（仿真 sim_c2_kfull 外推 × 真机 msprof 交叉验证，mac 外推比 1.00× 完美）：
  - 已建模指令+访存（FIX_L0C_TO_DST）~57.8ms（55%）
  - 仿真未建模 GM 写协议税 ~46.5ms（45%）：write-allocate / 512B burst / 20 核写仲裁
- 真机 mte2 比仿真快 3 倍：kCmp（51MB）重扫被 L2 命中兜住，仿真 L2 模型保守。
- **优化方向排序：FP16 scratch（fixpipe 104→~52ms）是唯一真路；指令/同步优化只有几 ms 空间；双缓冲已拿到 +9ms。**

## 3. NZ 直拷实验（失败收场，教训重要）

目的：matmul 结果以 CFG_NZ 直拷跳过 ND 转换，取证 ND 转换在 fixpipe 中的开销。

结果：6 卡、0 卡两轮分 pipe 数据**全部失真**（墙钟 ~19.8ms 但 mac 90–333ms、fixpipe 23.7–127.8ms、mte2 346–640ms 全部远超墙钟，且卡间分布不一致）→ kernel 进入异常状态，非环境干扰。

根因（锁定）：**CFG_NZ 下 `dstStride` 语义完全不同**（NZ 是 32B 单位的 burst 间跳距，ND 是 GM 行距元素数）。传入 `K_TILE_COLS=512` 在 NZ 语义 = 每 16B burst 跳 16KB → NZ fractal 写散布 8 倍地址范围，越界踩其他核的 kCmp/qCmp scratch → 坏数据/NaN → 异常路径+硬件重试，pipe 计数失真，墙钟 19.8ms 是假象。

若要测干净 NZ，必须配套改三件事：
1. `dstStride` 按 NZ fractal 布局重算（128×128 块内 16 个 16×16 fractal 排列，burst=512B=16 个 32B 单位）
2. scratch 布局按 NZ 格式重排（NZ 128×128 块 ≠ ND 64KB 连续）
3. AIV 读侧改 NZ 取数索引，或先做**"纯 AIC 写 + AIV（pass1/pass2）完全跳过"版**干净测 fixpipe 纯写对比

可信对照数据：ND 版 cube(aicore)=104.5ms、fixpipe=104.3ms。

**丢失状态**：用户最后要求"换 5 卡跑一下"，压缩时未确认是否执行，5 卡结果未知。

## 4. 关键文件

- kernel 服务头：`bsa_vec_post_pool_service.h`、`bsa_vec_sm_service.h`、`bsa_vector_service.h`、`bsa_select_block_mask_base.h`、`bsa_vec_pool_service.h`、`bsa_radix_topk_service.h`、`bsa_matmul_service.h`、`bsa_select_block_mask_common.h`
- 测试：`test_aclnn_bsa_select_block_mask.cpp`
- 仿真 prof：`sim_q2k3/`、`sim_q2k33/`（OPPROF 目录）
- 实验工作目录：`794880_bx8_by8_post16x16`（融合版）、`794880_bx8_by8_singlepool`（单池化对照）

## 5. 下一步建议（优先级）

1. 编译装包 + 上板验证融合版（唯一 pending todo）：精度对齐 + 客户 case 耗时复测（对照 192.64ms 基线）
2. 主攻 FP16 scratch（halve fixpipe 写量 + AIV mte2 读量同减）
3. NZ 直拷若继续：先实现"纯 AIC 写 + AIV 跳过"版取干净数；否则按 ND 结论收口
4. 双缓冲 scratch（已验证 +9ms）可合并进融合版一起上板

## 用卡约定（2026-09-23 更新）

- **默认用 7 卡**（空闲且状态正常时）。
- 7 卡被大量占用时，再选其他空闲且 Health=OK 的卡；避开有进程驻留的卡（测前 `npu-smi info` 查 Process 列表确认空闲）。
- 教训：卡 2 曾被 43.8GB GPTQ 任务驻留导致测量漂移 +0.7ms；跨卡差异 ~0.2ms、跨日漂移 ~0.4ms，对比实验须同卡同日。

## 当前状态（2026-09-23）

- 最新提交：`27c03d2e0`（已推送），精度 PASS（122290 逐位一致）。
- 卡 7 复验：纯融合 34.92ms（vs 竞品 34ms 持平）、全流水 37.04ms。
- pass1/pass2/pool 双侧结构性地板；剩余可选项均为实测负结论留档。

## 预存问题专项调查（2026-09-23 深夜，二分定论）

**全部 6 个遗留失败指向单一提交 `9b3927cce`（score scratch NZ直拷写读+背压同步，优化链第一刀）**：

| 用例 | 43d66dc95(v2终态) | 9b3927cce(链#1) | 171d2e03e(当前) | 结论 |
|---|---|---|---|---|
| base0(非post S=256 blklen) | **EXACT_PASS** | NPU_FAIL | NPU_FAIL | 链#1 引入回归 |
| tail(S=4032/3904 非整除尾) | FAIL(错值) | NPU_FAIL | NPU_FAIL | v2 就错值，链#1 变崩溃 |
| p16var / tndtail / bfvar | FAIL(错值) | NPU_FAIL | NPU_FAIL | 同上 |
| blk8(bx=by=8 S=1024) | NPU_FAIL | **BOUNDARY_TOLERANT_PASS** | FAIL(错值) | 链#1 修了 v2 崩溃；当前又错值(w8 列参数已修，另有残留) |

- 5 个 NPU FAIL 的错误类型：507015 挂死，plog 实锤 **"VEC UB address not aligned"**（aivector，与 w8 调试同错误类）
- 下轮入手：9b3927cce 的 NZ 读/背压路径在微型 shape（xBlocks=2、kColsAlign=16、1 行 slice）与变长 batch 下的 VEC 对齐
- 测试资产：/tmp/pp_root/<case>/（含 input+golden，可直接复测）；e62900f6e 也全挂（v2 修复在 43d66dc95）
- 注意：checkout 老提交后 third_party/ops-tensor/.../CMakeLists.txt untracked 冲突需删除再 build；grep -c error 后必须核对 .run 时间戳（曾两次装旧包误测）

## 预存问题专项第二轮（2026-09-23 深夜续，本地 af2b64126）

**定位进展（5个NPU FAIL 的根因排查）**：
1. 分阶段标记法实锤：base0 小grid(2AIC+4AIV)；aiv0/2(chunk持有者)卡在 **pass1 内部的 V_MTE2 门**
   （等 poolQ 残留 V 队列排空）→ 故障 V op 在 poolQ；aiv1/3 卡 TopK（等 aiv0/2，级联）
2. 寄存器证据：cube error=0（mmad 成功）、fixp/mte error 非零 → fixpipe/MTE 侧违规
3. 模拟器跑通 base0 且 mask 正确（sim 不校验对齐）→ 数据路径对，真机特有
4. blklen 关掉则 base0 不挂（mask 也对）→ 根因在 poolQ 变长 block 路径；但单独复现
   block_len=[28,6] 不挂（数据/上下文相关），S=256 系列非 post case compare 全 FAIL
   但 mask 正确（次要字段问题，待查 compare 细节）
5. 已做防御修复：fixpipe mSize 对齐 16（实测行为不变，语义更安全）

**调试基建（复用）**：
- 挂死可 dump 的 harness：run_bsa_select_block_mask_0828_dbg（源 test_aclnn_*.cpp 已改，
  编译命令在 shell history：g++ ... -lascendcl -lopapi -lnnopbase）
- pc→符号：plog 的 pc_start=模块基址(_0_mix_aic@0)，current−start=模块偏移，
  llvm-objdump 对 opp vendors 的 .o（符号 _0/_1/_16/_17 = tilingKey = Q/KV layout 组合）
- sim：装当前构建到 0909 CANN + msprof op simulator（LD_LIBRARY_PATH 带 simulator/dav_2201/lib
  和 vendor op_api/lib）
- 标记法源码已从 base/sm/vector 服务移除（会污染 post 模式 partSum），需要时按提交历史恢复

**下轮**：poolQ 变长路径逐 op 对齐审计（Cast/ReduceMean RA {actualLen,128} 的 counter 模式）；
S=256 非post compare 的次要字段差异；blk8 错值（bx=by=8 几何）。

## 专项第三轮结论（2026-09-24 凌晨，本地 83242b859）

**base0 挂死已修复**：fixpipe mSize 对齐 16（M=1 违反 fixpipe 粒度）。关键教训：调试标记
扰动小 grid 时序曾掩盖修复效果（标记在场误判无效、移除后 6/6 全过显现）——标记法用完即撤。

**套件终态**（26 用例）：21 独立通过（14 EXACT + 6 TOL + base0 修复后通过），套件内
base0 的 NPU FAIL 是前面真挂死 case 的卡残留污染（独立复跑 6/6 过）。

**剩余 4 挂死**：tail/tndtail/bfvar/p16var（共同点：变长/非整除——但 M=32 已对齐，
另有根因，疑 kCols 奇数 31→nSize 32 的 pass1 NZ 读或变长行切分路径）+ blk8 错值。
**下轮**：对 4 个 case 做注释二分（用户建议）：先 tail（B1N4 定长非整除最简）——
pass1 的 kCols=31/kColsAlign=32 路径逐块注释定位。

## 变长挂死二分（2026-09-24 上午，本地 9ebb7de25）

**干净二分链**（每步独立构建验证）：
TopK关→过 ｜ radix跳过(留SyncAll)→过 ｜ finalize跳过→过 ｜ 组装段跳过(cutA)→过 ｜
Cast+同步留/GM写跳过(cutC)→过 ⇒ **故障=finalize 组装段每 unit 的 8B DataCopyPad GM 写**
（代码与 n1 逐字相同、值健康无 NaN——疑 chunk 重排时序下 MTE3 竞态；base0 亦呈二进制
布局敏感的偶发，同族特征）

**已修**：尾列修正 GetValue/SetValue → colScale 向量化（尾列分支才触发，tail 实锤）。
**失败尝试留档**：窗口级批量写（staging+跨步行写）——n1 反而回退，疑 srcStride 单位
是 32B 块而非字节；如重试需先验证 DataCopyPad UB 源 stride 单位。

**教训（再次）**：二分调试的 skip 改动必须当场还原再进下一刀——本轮 finalize-skip
残留污染了 cutA/cutB/尾列禁用三步的结论（全部重测才纠偏）。

**当前**：客户 PASS/37.33ms；n1/tnd/blk64/base4 EXACT；tail/tndtail/bfvar/p16var/base0(偶发)
仍挂。下轮：①查 DataCopyPad UB src stride 单位后重试批量写 ②或对 GM 写改 32B 块对齐
粒度（pooledScore 段 padding）③base0 偶发性与 MTE3 竞态统一排查。

## 变长挂死第四轮（2026-09-24 上午续）

**新关键事实（tail 深挖）**：
1. 挂死是**偶发竞态**：同构建 10/10 不挂、其它时段必挂（二进制/时序敏感，base0 同族）
2. **finalize 的 pooledScore GM 写从未落地**：正确偏移(16128, qCmp 无 B×N 因子)与全 workspace
   扫描均无特征值——不挂的运行里 TopK 读到全零（mask 全零、mismatch 32/64=尾列+随机列）
3. n1 同代码写出正常（EXACT）——tail 的差异仅 yBlocks=31/尾块 partial
4. 附带发现：qCmp/kCmp 无 per-head 偏移（多 head 共写同区，n4/n8/n40 却过——待解释，
   疑 matmul 侧也按共享读或 tolerance 掩盖）

**下轮**（按优先级）：
1. 重新引入最小标记（GeneralFinalize 入口/写出前后各一）确认 finalize 是否执行到写出
   （标记会扰动时序——只判"执行/未执行"，不判时序）
2. 若执行到：查 DataCopyPad 描述符（dst tensor 切片构造）在 tail 几何下的正确性
3. 若没执行到：查 finalize 入口前的早退条件（sliceNum/窗口循环边界）
4. qCmp/kCmp per-head 语义独立核实（可能牵出多 head 系列问题）

## 尾块挂死第五轮（2026-09-24 上午，标记法终局定位）

**FinMark 执行轨迹（tail，挂时 dump）**：
- aiv3（空闲 subvec1）：到达 finalize 包装器入口（phase6）→ sliceNum==0 早退 → TopK 等待
- **aiv0/2/4/6（chunk 持有者）：连包装器都没到——卡在 pass1 或更早**
- n1 对照：aiv0 完整 1→5（finalize 全程）✓

**统一图景（tail/tndtail/bfvar/p16var/base0 全族）**：
- 竞态在 chunk 持有者的 MTE3 通路（pass1 的 w8 折叠 M3 跨步写 + partMax + finalize 写），
  症状两态：写静默丢弃（不挂但 pooledScore 全零）或队列故障（挂死）
- 偶发（同二进制 10/10 不挂 vs 必挂时段）——单次运行的二分结论全部不可信（本轮已
  多次被坑），后续判据必须 ≥10 次重复
- 已排除：dstStride 单位歧义（224B=7×32B 两解释同值）、NaN/Inf、地址未对齐（静态全审过）

**下轮候选**：
1. pass1 的 w8 折叠 M3 写改逐行连续写（放弃跨步）看丢写是否消失——隔离「跨步 MTE3 写」
   这一嫌疑元件
2. 或 w8 折叠暂存改用 band 级（复用 w16 的 foldOut 通道——需 16KB rs 区，查挤占）
3. msprof 采集挂时 MTE3 队列状态（aic-metrics 的 mte3 指标）

## 尾块挂死第六轮（2026-09-24，标记法极限 + 陈旧数据修正）

**新发现**：
1. P1Mark 轨迹：仅 aiv4 的 pass1 入口标记落地（1/几十次写）——chunk 持有者确实进入 pass1；
   其余标记全被竞态吃掉（标记写本身走 MTE3，同样被丢弃）
2. **陈旧数据修正**：partSum 区的"4032 正值"疑为跨 run 残留（workspace 复用不清零）——
   "折叠已执行"的推断作废；scratch 环同因不可作证据（n1 的 scratch 也空）
3. 干净构建 20 连跑：0 挂但 pooledScore 仍全零（mask=确定性 top-k(全零)，32/64 错）
4. **核心症状统一**：chunk 持有者的 MTE3 写选择性丢弃（pooledScore 必丢、标记大多丢、
   partSum 偶落），偶发队列故障（挂）——时序竞态，窗口与 K=31 类形状的 AIC fixpipe
   时序相关（n1 的 K=32 不进窗口）

**方法论极限确认**：标记法已到极限——标记写本身被竞态吃掉（沉默≠未执行），且标记的
额外同步对会抑制竞态（有标记必不挂）。注释二分同受偶发性污染（单次运行结论不可信）。

**下轮（新会话，按优先级）**：
1. w8 折叠路径的事件协议审计：slice 循环内 V_MTE3/MTE3_V(eventSft) 与 band 末
   V_MTE3(eventSft)/MTE3_V(evCross) 的时序交互——疑 eventSft 复用与 slice/band
   嵌套存在窗口
2. msprof aic-metrics 多次采集抓 MTE3 队列异常态（需在挂的时段跑）
3. 结构性绕过：w8 折叠改 band 级暂存（消除 slice 循环内的同步对，复用 w16 的
   foldOut 通道——需查 rs 区 16KB 挤占）
4. workspace 清零验证陈旧数据问题（对比 run 间残留）

## 尾块挂死——根因破案（2026-09-24 上午终局，本地 f9747b3de）

**根因**：子带尾块掩码 `Duplicate(subUb[r×foldCols + kChunkSize], NEG_INF, ...)` 目标地址
= 行首 + kChunkSize×4B。**kChunkSize%8≠0 时非 32B 对齐 → VEC 队列异常死亡 → 下游所有
MTE3_V/V_MTE3 等待永挂**（plog 实锤 "VEC UB not aligned"）。

**证据链闭环**：
- kCols=31(tail)/11(p16var)/2(base0) 等 %8≠0 → 全挂；kCols=32(n1)/64(blk64) 等 %8==0 → 全过
  ——「同代码 n1 过 tail 挂」之谜的答案
- workspace 清零 + 探针定位：折叠 M3 落地（写完成）但 MTE3_V 事件永不触发（队列已死）
- 挂死两态症状（静默丢写/队列故障）= VEC 队列死亡的不同表现时机

**修复**：未对齐尾段(≤7列/行)标量 SetValue（S 管道无对齐约束）+ 对齐段 Duplicate。pass1 与
旧 pass2 两处同修。

**套件终态**（26 用例）：挂死 0；EXACT 11 + TOLERANT 9 + 值错 6（blk8/tndtail/bfvar/p16var
+ n1 偶发 ~10%（MTE3 写丢弃竞态，独立残留问题））

**残留**：① tndtail/bfvar/p16var 值错（v2 基线就错——变长 kCmp 值 bug，非本轮引入）
② blk8 值错（bx=by=8 几何）③ n1/客户偶发 MTE3 写丢弃竞态（~10%）④ 客户全流水 37.76ms
（vs 37.33 前值——路径无变化，疑卡间噪声，下轮同卡对比确认）

## blk8 链回归二分（2026-09-24 续，本地 f9747b3de 基线）

**二分链**：9b3927cce PASS → 45745f5a5 **PASS(2/1024)** → a9bc89ce1 **FAIL(544/1024)** → … → HEAD FAIL
⇒ **回归 = a9bc89ce1（softmax 单趟化）**——与 post_by≠16 硬编码同源提交；171d2e03e 泛化
修复了主族，但 blk8（kCols=128 恰好 64 倍数、无掩码路径）仍有残留。

**blk8 失败特征**：
- 确定性 544/1024（每次同值）；pooledScore 全零（前8列确定性 top-k 模式）
- plog: VEC UB not aligned (core 13)——单趟化路径内另有未对齐 VEC op
- partMax 全空 + partSum 仅个别槽有数据（渐进停摆——但 partSum 偏移未校准，不可全信）
- 已排除：pool 批量路径（禁用后仍 FAIL）
- blk8 特有几何：bx=by=8, kCols=128(foldCols 恰 64 倍数无 padding), 16 chunk-owner AIV

**下轮**：blk8 用 FinProbe 技术（用完即删！）定位单趟化路径内 blk8 特有的未对齐 VEC op
——嫌疑：kCols=128 时 foldCols=128 的 BRS repeat=cnt×2 / RA{cnt,64} in-place / finalize 窗口。

## blk8 深挖第七轮（2026-09-24 午后，SafeProbe 技术）

**定位链**（安全探针 = slot 区后空闲 UB 值写 + aivIdx37 槽 M3，foldCols≤240 守卫）：
1. pass1：16/16 chunk 持有者完整走完 9 阶段（含 partMax M3 + MTE3_V）——pass1 无辜
2. finalize：16/16 到达入口(21) → tile 循环末(22) → Cast 后(23)且探针自带 V_MTE3 对通过
   ——V 队列活着；组装段代码全部执行
3. **结论：finalize 完整执行，但 pooledScore GM 写被静默丢弃**（全 dump 扫不到特征值）
4. GM 写跳过/保留、Mul padding 64 均无变化；加更多探针(24/25)后二进制变化竞态吞掉全部
   探针写（含 21）——blk8 竞态命中率 ~100%（n1 ~10%，同族 MTE3 写丢弃）

**与 a9bc89ce1 回归的关系**：单趟化引入 finalize 后 blk8 即错值——finalize 的 MTE3 写在
blk8 形状（16 AIV×4 head 高密度交错）下 100% 命中竞态窗口。

**下轮根本性方案**（三选一）：
1. msprof aic-metrics 连续采集 blk8 的 MTE3 队列状态（找队列异常的直接证据）
2. 协议重构：finalize 的 eventSft/evCross 与 pass1 复用改为独立 event ID（消除窗口）
3. 时序解耦：finalize 的 GM 写聚合到 TopK 前一次性批量写（减少交错密度）

## blk8 第八轮（2026-09-24 午后，eventFin + 布局考古修正，本地 945251b46）

**eventFin(7) 试验**：finalize 独立事件 ID——blk8 写丢未解（非 event 复用机制）；
n1 10/10（弱阳性）；无回归。已提交保留（协议正确性改进）。

**布局考古重大修正**（blk8，workspace 28.6MB）：
- partSum 实际 @ ~26.7MB（**非**链式估算 10.6MB！）——前段 section 实际尺寸远大于
  假设（aicNum/Q_CHUNK 待查，scratch 环数据在 [16MB, 24MB)，512KB 间隔=环槽）
- **折叠写确认落地**（26.7MB 处 2058 非零 = 16 槽×128 组 + partMax）——pass1 完好
- 唯一未落地：finalize 的 pooledScore 写（65536 处全零、全 dump 无特征值）
- 之前多轮"partSum/partMax 全空"结论因偏移错误全部作废

**blk8 剩余谜题**：finalize 完整执行（探针至 Cast 后）+ partSum 读正常 +
pooledScore 写（{1,32B,0,0} 到 32B 对齐地址）不落地——同形状探针写（到 partSum
槽37）落地！写读地址一致（kernel 链式 + radix 同 tensor）。

**下轮**：①用数据签名法定位 pooledScore 写的真实落点（可能写到别的偏移——
   pooledScoreOffset 的 kernel 链 vs tiling set 值差异）②查 tiling 的
   set_pooledScoreOffset(=attnScoreFp16Size=0!) 是否被某处消费（radix? harness?）
   ——该死值与 kernel 链式(65536)不一致，若 radix 用了 tiling 值会读 offset 0！
③msprof aic-metrics MTE3 队列证据

## blk8 第九轮（2026-09-24 下午，地址标定定案，本地 945251b46 干净态）

**FinCal 地址标定**（finalize 入口写 partSum 槽37 → dump 定位）：
- **partSumOffset = 27,331,540B ≈ 26.06MB**（链式自洽——探针写落地即证）
- 折叠数据在正确偏移处**完美落地**：槽0 = 行0-7×组0-16 非零（值~14-115 exp和），
  行8+全零——签名精确匹配 → **pass1 折叠完全无辜**
- 16 AIV 中仅 aiv12 的标定探针落地（写丢弃竞态仍在，但 chain 地址正确性已证）

**blk8 最终谜题（收窄到一点）**：
- partSum ✓（26.06MB 数据正确）｜ finalize 读 ✓（探针证执行）｜ corr/acc/Mul/Cast ✓（探针证到 Cast 后）
- **pooledScore 写（{1,32B,0,0} → 65536，链式值确定无歧义）不落地**——全 dump 无特征值
- 无重叠覆写方（ring@66KB+、topkWS@~25MB、qCmp/kCmp@0-64KB 均不覆盖 [64KB,66KB)）
- 探针写（同队列同 drain，到 26MB 区）落地 vs pooledScore 写（到 64KB 区）不落地
  ——**唯一残留差异：目标地址段**（低地址 64-66KB vs 高地址 26MB+）

**下轮终极二选一**：
1. 把 pooledScore 写的目标改到高地址区（如 partSum 尾部空闲槽）验证「低地址段写
   丢弃」假说——若落地则为 workspace 低段（qCmp/kCmp 后紧邻区）的写路径异常
2. msprof aic-metrics 采集 MTE3 写事务按地址段分布

**附**：n1 的 partSum 同样需按各自链式偏移重查（历史 [16MB] 观察全是错位结论）。

## blk8 第十轮（2026-09-24 下午终局，tiling 陈旧 .so 破案 + 写不可见之谜）

**重大修复（隐性 bug）**：`build/` 目录陈旧导致 **tiling .so 未重编**——旧 tiling 的
workspaceSize(28,611,796) 比 kernel 链式需求(28,662,740) 少 51KB → **partMax 区整个越界**
（写被丢弃）。全清重编后 workspaceSize=28,662,740，partMax 落地（20 槽非零验证 ✓）。
**教训：改 tiling 源码后必须全清 build/ 重编，增量编译对 host 侧 .so 不生效！**

**blk8 残余之谜（收窄到极致）**：
- fresh tiling + FinTag 验证：16/16 AIV finalize 全程执行（入口→写出→退出全过）
- 写经 pooledScoreGmTensor（成员/局部/新构/任意 index/常数 3.25）**全部不可见**——
  连 radix 同流读都读到零（写从未到达 GM 的 64KB 段）
- 同位置写经 partSumGmTensor（27MB 段）**全部落地**
- **同一 pooledScoreGmTensor 在 w16 快路径（n1/p16）写正常落地！**
- n1(w8, pooled@16,384)落地 vs blk8(w8, pooled@65,536)不可见——疑 64KB 边界地址段
  的 MTE3 写异常（blk8 的 pooledScore 恰在 65536=64KB 整边界）

**下轮终极手段**：
1. 泄漏 PhyAddr 验证：把 pooledScoreGmTensor.GetPhyAddr() 的值按位写出到可读区——
   确认 tensor 基址是否真的是 workspace+65536
2. msprof aic-metrics MTE3 写事务按地址段分布（64KB 边界假说的直接证据）
3. 若 64KB 边界坐实：pooledScore 段加 2KB 头 padding 错开边界（tiling+kernel 两处）

**当前套件状态（fresh tiling 干净构建）**：tail PASS / base0 EXACT / n1/tnd/blk64/base4
EXACT / tndtail/bfvar/p16var FAIL（v2 基线值错）/ blk8 FAIL（写不可见之谜）/ 客户 PASS。

## blk8 第十一轮（2026-09-24 晚，边界假说证伪 + 低区写不可见矩阵，本地 945251b46 干净态）

**64KB 边界假说证伪**：pooledScore 前插 2KB padding（tiling total+2048 & kernel 链式偏移，
65,536→67,584）——blk8 仍 FAIL、新偏移处仍零。边界不是原因。

**颠覆性发现——低区写不可见矩阵**（fresh tiling、全清重编、无 memset）：
| 区域 | 写方 | dump 可见 | 内核后续读 |
|---|---|---|---|
| qCmp [0,32KB) | pool MTE3 | **零** | matmul(MTE1/L2) 读到真值（分数真实）|
| kCmp [32,64KB) | pool MTE3 | **零** | 同上 |
| pooledScore [64,66KB) | finalize MTE3 | **零** | radix(MTE2) 读到零 ✗ |
| scratch [16MB+] | AIC fixpipe | ✓ | ✓ |
| partSum [27MB+] | pass1 fold MTE3 | ✓ | finalize(MTE2) 读到真值 ✓ |
| partMax [28.6MB] | pass1 MTE3 | ✓（修完 OOB 后）| ✓ |

即：**低区（<64KB+）的 MTE3 写对 dump 与跨引擎 GM 读不可见**（但同 L2 引擎如 MTE1
能读到）；**高区（16MB+）的写全部持久**。n1 的 pooledScore@16,384 为何可见——n1 的
qCmp+kCmp=16KB，pooledScore 紧随其后…16,384 也在"低区"却可见！→ 可见性分界不在
64KB，而在更微妙处（可能与总分配布局/L2 分 bank 有关）。

**下轮（需硬件级手段）**：
1. msprof aic-metrics：MTE3 写事务的地址分布与 L2 行为——直接看低区写是否到达 L2/GM
2. 对比 n1 vs blk8 的 L2 cache 分区（n1 pooled@16K 可见、blk8@65K 不可见——差异
   可能在 L2 way/bank 映射越过了某页边界）
3. 兜底工程方案：把 pooledScore/qCmp/kCmp 全部挪到 workspace 高区（>16MB）——
   若高区写全部持久，一次布局重排绕过硬件异常

## blk8 第十二轮（2026-09-24 晚，**根因链基本锁定**：尾部重排修复写可见 + fold 输入 tile 被逐行 ×2^k 损坏）

### 已确证的修复（保留在 working tree，未提交）
**pooledScore/postStaging 尾部重排**（bsa_select_block_mask_base.h InitConstInfo 链式偏移调整：
pooledScore 移到 partMax 之后）——**修复 blk8 的"写不可见"**：重排后 finalize 的 GM 写落地、
radix 读到真实数据（mask 模式与写入值自洽）。全用例回归无害：n1/p16/p816/tail/base0/tnd/blk64
全 PASS + 客户 BOUNDARY_TOLERANT_PASS ✓。qCmp/kCmp 低区写不可见异常仍存在但已绕过。

### blk8 剩余 bug：fold 输入 tile 损坏（探针实证链）
| 检查点 | 结果 |
|---|---|
| 原始 tile（NzRead 后/Muls 前）| **正确** raw dots ~30.4-34（CPU 对照吻合，832/1024 非零）|
| post-exp tile（fold 时刻）| **损坏**：行0=正确值×16、行1-3=×2 精确、行4-7=混合、B_r=m−ln2 特征 |
| partMax（tmpMax band 末）| **精确正确**（与 CPU head3 逐行 4 位小数吻合）|
| globalMax/globalSum | sane（globalSum≈正确行总和 115.7）|
| BRS 微实验（全1/行值模式）| **BRS 本身语义正确**（×8窗口、按行分组 ✓）|
| partSum 槽（pitch-64/16有效）| = 损坏 tile 的组和（115.7/14.4 混合）✓ 因果闭合 |

**损坏窗口锁定**：Muls→RowMaxCascade→Brcb(globalMaxBroc)→BSASubRowBroadcast→Exp 链内。
逐行 ×2^k（二进制幂）特征 + 行4-7 混合 = 部分覆写/错读广播缓冲。

### 头号嫌疑（下轮首查）
1. **RowMaxCascade/RowSumCascade 硬编码 REDUCE_WIDTH=K_TILE_COLS=512**（bsa_vec_sm_service.h:290/313）
   —— stage-1 按 rowStride(=foldCols=128) 写 rows×16 组，stage-2/3 按 rows×64 读——**结构性错位实锤**
   （n1 foldCols=64 错位轻 → 恰好过；blk8 foldCols=128 错位 4×）。需读 RowMaxCascade 体并改为按
   rowStride 推导。
2. Brcb 的 brocRepeat=ceil(rows/BROC_BASE_NUM)——rows=8 时若 repeat=1 只广播 m_0，行1-7 读
   **陈旧 globalMaxBroc**（前一相位 finalize 的 corrBroc≈0.0087 残留？→ exp(x−0)=raw exp 特征）。
   查 BROC_BASE_NUM 值与 globalMaxBroc 陈旧内容。
3. 逐点 bisect 探针：post-Muls tile / tmpMax / globalMaxBroc 内容 / post-Sub tile 四点 dump 定位。

### 探针基础设施（已验证可用）
- `partSumGmTensor[39*8192+X]`（AIV39 槽，blk8 下大概率空 AIV）写 GM 标记/dump——落地可见
- 探针注意：DataCopyPad 写方向 5 参数形式、aicore 禁 float↔uint cast（int 用 ReinterpretCast<int32>）、
  LocalTensor 读值用 GetValue()、**微实验必须先 dump 后填充**（自己污染 tile 教训）
- partSumOffset(blk8,重排后)=27,329,492（探针标定）；partMax=28,640,212；槽 stride=Q_CHUNK(128)×foldG(64)×4B

### 下轮行动
1. 读 RowMaxCascade 体（line 281-306）确认 512-hardcode；改 REDUCE_WIDTH=rowStride 修复级联
2. 修 Brcb repeat 或 globalMaxBroc 每次全量重建
3. 修复后 blk8 应 PASS → 连同尾部重排一起提交（fix: w8路径折叠级联布局泛化）
4. 套件全量回归 + 客户精度&性能终验

## 新会话快速上手（2026-09-24 终稿补充）

### 当前状态一句话
HEAD=945251b46（7 本地提交待推送）+ working tree 含**尾部重排修复**（base.h，pooledScore 移
partMax 后——已验证全用例无害，待与 fold 级联修复一起提交）。客户 PASS、套件 ~20/26、
blk8 根因已锁定待修、tndtail/bfvar/p16var 未查。

### 命令序列（复制即用）
```bash
# 构建（改 tiling 后必须 rm -rf build 全清！增量不重编 tiling .so）
cd /home/sll/bsa_select_block_mask/post_pool/ops-transformer
source /home/sll/cann_9-2-0_master_0828/cann-9.2.0/set_env.sh
rm -rf build && bash build.sh --pkg --soc=ascend910b --ops=bsa_select_block_mask 2>&1 | grep -cE "error:"   # 必须=0
ls -la build_out/*.run   # 必须是刚才的时间戳！编译失败会装旧包（已踩 3 次）

# 安装
rm -rf /home/sll/cann_9-2-0_master_0828/cann-9.2.0/opp/vendors/custom_transformer
export ASCEND_OPP_PATH=/home/sll/cann_9-2-0_master_0828/cann-9.2.0/opp
export TMPDIR=/tmp/opencode
./build_out/cann-ops-transformer-custom_linux-aarch64.run 2>&1 | grep SUCCESS

# 运行 + 比对（卡：先 npu-smi info 选空闲 OK 卡；7 卡被 VLLM 占）
cd /home/sll/bsa_select_block_mask/self_test_final
source /home/sll/cann_9-2-0_master_0828/cann-9.2.0/set_env.sh
export ASCEND_OPP_PATH=/home/sll/cann_9-2-0_master_0828/cann-9.2.0/opp
./run_bsa_select_block_mask_0828 /tmp/pp_root/<case>/input /tmp/pp_root/<case>/npu <card>
python3 compare.py --input_dir /tmp/pp_root/<case>/input --golden_dir /tmp/pp_root/<case>/golden --npu_dir /tmp/pp_root/<case>/npu | grep RESULT

# 客户场景
cd /home/sll/bsa_select_block_mask/post_pool/794880_bx8_by8_post16x16
/home/sll/bsa_select_block_mask/self_test_final/run_bsa_select_block_mask_0828 input npu <card>
python3 /home/sll/bsa_select_block_mask/self_test_final/compare.py --input_dir input --golden_dir golden --npu_dir npu | grep RESULT

# 套件全量（26 用例，卡号按需改）
bash /home/sll/bsa_select_block_mask/self_test_final/run_pp_cpu_golden_0828.sh
```

### wsdump 约定（关键！）
dbg 版 harness（run_bsa_select_block_mask_0828_dbg）**仅当 input_params.txt 尾部有
`pooledScoreOffset=0` + `pooledScoreSize=<workspaceSize>` 两行才 dump 工作区**到
`<npu_dir>/../npu_wsdump.bin`。blk8/n1/tail 已带；新用例需先跑一次普通版拿 workspaceSize 再追加。
blk8(重排后)：workspaceSize=28,662,740；partSum@27,329,492；partMax@28,640,212；
pooledScore@28,660,692（fp16, 4头×16×16）；AIV39 槽@partSum+39×8192×4B 可作探针区。

### CPU 对照脚本思路（探针 dump 验证用）
gen_golden_cpu.py 语义：q_pooled=pool_blocks(Q,8token均值) → score=q_pooled@k_pooled.T×scale
→ att=softmax(score,dim=-1) → pooled=avg_pool2d(att,(post_bx,post_by))。
inline 对照：np 读 input_query/key.bin(fp16,reshape [1,N,1024,128]) → 上述链 → 与 dump 比值。

### 下轮任务（按序）
1. 读 bsa_vec_sm_service.h:281-306 RowMaxCascade 体 → REDUCE_WIDTH=K_TILE_COLS(512) 硬编码
   改为 rowStride 推导（RowSumCascade 同病，:307-326）→ 全清重编 → blk8 复测
2. 若仍错：四点 bisect 探针（post-Muls tile/tmpMax/globalMaxBroc/post-Sub tile）
3. blk8 PASS → 连同尾部重排一起提交 → 套件全量 + 客户精度&性能终验（用户铁律）
4. tndtail/bfvar/p16var（v2 遗留，可用 pooledScore dump 分 head/batch 对比 golden 定位）

## 09-08~09-23 性能优化链存档摘要（本 session 摘要抢救，防丢失）

### 已 push 提交链（6 个，结果性记录）
e3b83facc pass1 chunk上下文 → 170680301 pass1 slice合并32行链 → 82886d1dd L2跨tile预读+receipt前移
→ 82886d1dd/376398166 TopK合并 → b07ce86a0 pass2 finalize批处理BATCH 4→8 → 27c03d2e0 poolQ/K批量化池化
性能：全流水 45.9→37.3ms（-19%），纯融合 39.23→34.92ms（vs 竞品 34ms 持平）
负结论留档：M3 写出拆半 -0.65ms 回退（VEC 多气泡>藏掉的 M3 传输）不做

### 关键教训（此前未入档）
1. **invCols 偏移**：pass2 BATCH=8 布局下 invCols 在 rs0[1536]（旧 1280 会错位）——BATCH 布局改动必查 rs 尾区偏移
2. **if(false&&) 短路陷阱**：`if(false && expr)` 中 expr 的编译器语义——含 VEC 副作用的代码不能靠 false&& 关掉，必须整块注释/编译开关
3. **标记扰动时序**：调试标记（额外 VEC/MTE3 写）会扰动流水时序、**掩盖竞态**——加标记后"复现消失"≠修复，结论需去标记复验；单次运行不可下结论（竞态类≥10 次）
4. **事件协议全景**：eventSft(3) 被 pass1 五个方向共享（MTE3_MTE2/V_MTE2/MTE2_V/V_MTE3/MTE3_V）；eventFin(7) finalize 专用；evCross(6) 跨核；evSlot(4/5) 子带乒乓；pool(0/1) 池化；eventPost(2)
5. **用户验证矩阵要求**：poolQ、matmul/pass1/pass2、TopK 的任何改动需泛化精度验证——post pool 关闭用例 + postXBlock/postYBlock ∈ {8,16,32,64,128} + q/k 尾块变长

### 补遗：工具链细节（2026-09-24 终稿第二补）

**dbg harness 重编**（源码 self_test_final/test_aclnn_bsa_select_block_mask.cpp；已编好的
run_bsa_select_block_mask_0828_dbg 一般无需重编）：
```bash
g++ test_aclnn_bsa_select_block_mask.cpp -o run_bsa_select_block_mask_0828_dbg \
  -I<op_api 头文件路径> -L$ASCEND_OPP_PATH/../lib64 -lascendcl -lopapi -lnnopbase
```

**/tmp/pp_root 用例重建**：无需手工——`run_pp_cpu_golden_0828.sh` 自带 gen_data.py 固定 seed
生成 input+CPU golden（数据可复现），跑一遍套件即重建全部用例。单补某用例：
`bash run_pp_cpu_golden_0828.sh <用例名>`。

**套件 29 用例清单**：n1 n4 n8 n40 p16 p816 p168 bf16 tnd sp03 sp098 b2 tail blk64 blk8
blklen tndvar tndtail bnsdvar bfvar p16var bottomk base0 base4 basetnd basesp bigsq sq256k sq256t
（bigsq/sq256k/sq256t 为大 shape 用例，跑前看时长；当前 FAIL×5：blk8 + tndtail/bfvar/p16var）

**性能测量口径**（用户铁律配套）：客户场景性能 = msprof 采集 kernel duration，
重复≥3 次取中位数（客户目录下 prof_bare_*/prof_cv_* 为历史采样样例，纯融合 vs 全流水两口径）。
对比实验必须同卡同日；单次运行不下结论。另 self_test_final/sem_perf_run.sh 为
单/双 pooling 语义对照专用（用 0730 CANN 环境，勿与精度流程混用）。

**git 状态约定（本文档提交时点）**：7 个代码提交已 push；working tree 留
base.h 尾部重排（未提交，故意——待 blk8 fold 级联修复验证 PASS 后合并提交
`fix: w8路径折叠级联布局泛化+pooledScore尾部重排`）。
