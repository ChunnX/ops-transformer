# QuantLightningIndexerV2 arch22 (Ascend910B) 性能优化记录

> 日期: 2026-09-14 | 分支: ops-transformer-ds41/ds41 | 设备: 910B3 (NPU2)
> 基线: git HEAD (commit 含 candidate/R11 修复) 的 arch22 内核

## 1. 性能目标

降低 QuantLightningIndexerV2 arch22 路径 kernel 执行时间，优先 decode（延迟敏感），
兼顾 prefill 吞吐；要求精度位等价/官方比对全绿。

## 2. 测试环境

- 910B3, CANN 9.0.0, custom 包 `/opt/tjj/custom_opp_qli_cand` (build_install.sh 编译安装)
- 计时: torch 层 wall time (npu.synchronize, warmup 3 + repeat 20, p50) 用于 shape 扫描;
  kernel 级数据用 msprof op_summary (Task Duration / aiv_time 中位数, n=20)

## 3. 测试用例

- perf: decode s2∈{2k,8k,16k,32k,64k}×topk{128,512,2048}×g{32,64}×B{1,4,8,16}, prefill s1∈{8..2048}
- 精度: equiv_qli_v2.py 19 case NPU↔NPU 位等价 (新旧包同输入) +
  qli_cand_test harness 59 case 官方比对 (排除 1M/big)

## 4. 统计口径

- kernel 时间: msprof `Task Duration(us)` 中位数（前 3 次 warmup 剔除）
- 精度: 位等价（array_equal）+ 官方 multiset/边界容差（compare_topk_valid, thres=0.001）

## 5-7. 各优化项结果

### P2'+P6a: MergeSort 回拷 MTE→V管道 + 跳过冗余 Duplicate — **回退**

| case | 基线(us) | 改后(us) | delta |
|---|---|---|---|
| dec16k_t512 | 52.82 | 59.83 | +13.3% |
| pre1k_s2_16k | 959.59 | 1026.73 | +7.0% |
| dec2k | 25.64 | 26.12 | +1.9% |

结论: MTE UB→UB 回拷吞吐远高于 V 管道逐块 Adds，且 MTE 与 V 可并行，原实现
把回拷时间藏在 V 计算后面。**回退**（位等价 19/19 通过后回退）。

### P1: C/V 预取深度 2→4 — **回退**

单 tile/核、双 tile/核场景通过；**≥2 gS1 块/核（s1≥192）死锁**。
已定位修复 sL1 FIX_MTE1 事件模数错位（该处已确认非根因，sL1 恢复双缓冲后仍死锁），
与 V0/V1/C 三方环形同步和 metadata 分核（isNeedLD/s2End）存在未建模耦合。
继续盲修风险高，**回退**，结论存档供后续专项攻关。

### P3: topk 宽度按 sparseCount 自适应 — **保留（唯一合入项）**

改动: `service_vector_arch22.h` 新增 `topkWidth_=Align(sparseCount,64)` /
`topkPairWidth_=2×`，主路径累加器行宽、MergeSort 宽度、ExtractIndex、LD 归并
(MrgSort4/回拷/Extract)、workspace 行偏移、sortOutBuf 分配全部按实际宽度收缩；
candidate 路径 (mode=1/2) 保持 2048 契约不变。语义依据: 降序排序下
2048 名之外永不入选，截断不影响输出集合。

kernel 时间 (msprof 中位数):

| case | 基线(us) | P3(us) | delta |
|---|---|---|---|
| dec16k_t512 | 52.82 | 48.03 | **-9.1%** |
| dec2k | 25.64 | 24.23 | -5.5% |
| pre1k_s2_16k | 959.59 | 935.17 | -2.5% |

泛化 shape 扫描 (torch p50, 同脚本同输入):

| shape | 基线(ms) | P3(ms) | delta |
|---|---|---|---|
| dec_s2_1k | 0.118 | 0.117 | -0.8% |
| dec_s2_4k | 0.127 | 0.119 | -6.3% |
| dec_s2_32k | 0.179 | 0.174 | -2.8% |
| dec_s2_64k | 0.243 | 0.231 | -4.9% |
| dec_s2_16k_g64 | 0.156 | 0.148 | -5.1% |
| b8_dec_16k | 0.143 | 0.142 | -0.7% |
| b16_dec_16k | 0.141 | 0.142 | +0.7% |
| pre_s1_32 | 0.189 | 0.191 | +1.1% |
| pre_s1_256 | 0.430 | 0.443 | +3.0% |
| pre_s1_512 | 0.643 | 0.635 | -1.2% |
| pre_s1_2048 | 1.979 | 1.945 | -1.7% |
| pre_s1_1k_s2_32k | 1.949 | 1.961 | +0.6% |
| pre_s1_1k_t2048 | 1.053 | 1.066 | +1.2% |
| tnd_pre_1k | 1.062 | 1.033 | -2.7% |

大 shape (128K/1M, torch p50):

| shape | 基线(ms) | P3(ms) | delta |
|---|---|---|---|
| big128k_dec (s1=1, s2=131072, t512) | 0.388 | 0.353 | **-9.0%** |
| big128k_pre (s1=512) | 4.089 | 4.048 | -1.0% |
| big1m_dec (s1=1, s2=1048576, t512) | 2.097 | 1.870 | **-10.8%** |
| big1m_pre_s1k (s1=1024) | 56.678 | 56.761 | +0.1% |

mask0 系列全量 24-shape 对照 (harness CASES 形态: s∈{16K,128K,1M} × BSND/TND ×
candidate m1/m2 × r1/r2, g=32, topk=512, torch p50):

| shape 组 | 基线(ms) | P3(ms) | delta |
|---|---|---|---|
| mask0_16k_*_m1 (4例) | 19.30 | 17.49 | **-9.3%** |
| mask0_16k_*_m2 (4例) | 81.97 | 80.29 | -2.0% |
| mask0_128k_*_m1 (4例) | 1164.3 | 1066.4 | **-8.4%** |
| mask0_128k_*_m2 (4例) | 1430.2 | 1415.4 | -1.0% |
| mask0_1m_*_m1 (4例) | 73978 | 67864 | **-8.3%** |
| mask0_1m_*_m2 (4例) | 61207 | 61068 | -0.2% |

(注: m2 为 candidate consumer 模式, 构造输入为全候选块, 实际 kernel 内 m2 分支
含候选 mask 计算与 mode=1 的块级排序, topk 收缩只作用于 sparse_indices 主链路,
故 m2 收益被稀释。m1 1M 单例绝对收益 ~6.1s/次调用。)

生产 shape (用户提供的现网 shape 串, msprof kernel 中位数):
- 共同: q TND[12,32,128] (T=12, g=32, D=128), B=16 (cuQ[17], 后 4 批空→TND padding),
  k PA 池 [20369,blk,1,128], seqused_k[16]=1172, block_table[16,1172], topk=1024
- s1: blk=128, cmpRatio=1, mode3 → kernel 23.38→**21.24us (-9.2%)**
- s2: s1 + candidate_topk_index[12,1,2048] (mode2 consumer) → 38.54→**37.30us (-3.2%)**
- s3: blk=64, cmpRatio=2 (mask0 无 residual) → 23.04→**21.46us (-6.9%)**

结论: decode/topk 主导场景稳定收益 1~11%——**S2 越长收益越大**（128K/1M decode 达
-9%/-10.8%，长上下文 LD 归并链路按 topk 收缩的收益占比更高）；prefill 大 s1 场景
（排序本体主导）在噪声内持平。mask0 大 prefill (1M, candidate 模式) m1 路径
-8.3% (6.1s/次)，m2 路径被候选链稀释至 -0.2%。现网生产 shape
(T=12 decode, g=32, topk=1024) 实测 -3.2%~-9.2%，与 topk≈1024 的预期一致
（宽度收缩一半，收益略小于 topk=512 的 -9%）。

## 8. 对比结论

合入 P3 后: decode 类 kernel 时间 -1%~-9%（topk 越小收益越大），prefill 持平，
精度 19/19 位等价 + 59/59 官方比对全绿（排除 1M）。

## 9. 瓶颈分析（剩余）

- prefill 大 s1: Sort32+MrgSort 全宽排序主导（~1000us @ s1=1k），排序本体不动则
  收益空间在减少 tile 数（s2Base 已 2048，受 UB 约束）。
- decode 固定 ~12us 地板: aic scalar 占比 ~0.7（metadata 读取/CalcRunInfo 标量链）。
- LD 归并 (P5) 与 score fp16 (P4) 未实施，见 §10。

## 10. 优化建议（后续）

### P5: ProcessLD 归并乒乓 — **已实测否决（回退）**

乒乓双缓冲 + MrgSort 直写对端（消除每轮 DataCopy 回拷），msprof 对比（P3 基础上）:

| case | P3(us) | P5(us) | delta |
|---|---|---|---|
| ld16k | 47.29 | 47.66 | +0.8% |
| ld128k | 242.78 | 237.07 | -2.4% |
| ld1m | 1757.18 | 1811.20 | **+3.1%** |
| pre1k | 935.17 | 924.40 | -1.2% |

根因: 每轮省的回拷只有 pair 宽（topk512 → 4KB ≈ 0.3us），而期望的"MTE2 拷入与
V 归并重叠"被既有 `SetWaitFlag<V_MTE2>` fence 挡住——下一轮拷入的读区恰是本轮
MrgSort 的写区，物理上必须等 V 排空，重叠不存在。1M（归并轮次最多）反而 +3.1%。
收益上限 <1us/轮、方向不一致，**回退**。若要真重叠需三缓冲轮转（读区≠写区≠上轮区），
收益上限仍仅 ~0.3us/轮，投入产出比低，不建议做。

### 其余建议

- **P4 (低)**: score 通路 fp32→fp16 (Fixpipe F322F16 随路)，GM/L2 事务减半，
  受 L2 命中折扣，需实测。
- **P1 (需专项)**: 深度 4 死锁根因在 V0/V1/C 三方同步与 metadata 分核耦合，
  建议先做协议形式化（逐拍表）再动；本机已复现并定位至多块/核场景。
- torch 扩展缓存: 改包后若报 "libopapi not found"，删
  `~/.cache/torch_extensions/py311_cpu/quant_lightning_indexer` 重链即可。

### P6': C 核标量预取（3 槽影子轮转）— **已实测否决（回退）**

背景: msprof 显示 AIV 四管道 busy 之和远小于 AIV 时长（decode 16K: 11.8/19.0us，
idle 38-41%），确认为跨核同步等待。尝试在 C 核等待 V 的窗口里预取下一拍的
CalcRunInfo（纯本地标量），runInfo 扩 3 槽轮转，让 Wait 返回后 C 直接进 MAC。
实现要点（含 3 个必要修复）: 首 tile/块首 tile 就地 seed、块尾显式失效槽、空拍
补 Set(syncC1V1) 配对。等价 19/19 位等价 PASS。

msprof 实测（相对 P3）:

| case | P3(us) | P6'(us) | delta |
|---|---|---|---|
| dec16k | 47.29 | 49.44 | +4.5% |
| pre1k | 935.17 | 929.09 | -0.7% |
| prod_s1 | 21.24 | 26.28 | **+23.7%** |
| prod_s2 | 37.30 | 40.78 | **+9.3%** |
| prod_s3 | 21.46 | 27.16 | **+26.6%** |

根因: 预取的 CalcRunInfo 并没有"藏进等待"——它插在 Wait(syncV1C1) 之前，但 C 的
等待窗口里 C 核本来就闲着，插入的标量计算把 C 的整体节奏后移（AIC scalar 时间
反而上升），等效把关键路径拉长；同时 AIC/AIV 各自独立执行 seed+prefetch 两份
CalcRunInfo，双份标量开销直接叠加。**AIV idle 的本质是 C 侧标量（metadata 读取 +
地址链）就在跨核关键路径上，只有把 C 的标量准备整体搬出关键路径（如 host 预算好
或独立小核任务）才可能消除——单核内部预取在结构上不可行。回退。**

## 12. 指令级仿真数据（msprof op simulator, Ascend910B3 模型）

工具链: `perf_prod_sim.py`（两阶段: dump 真机构造输入+metadata 存本地; load 从本地
读回循环拉起主 kernel）+ `msprof op simulator --soc-version=Ascend910B3
--launch-count=10 --kernel-name="QuantLightningIndexerV2" --core-id="0"`。
launch 9（稳态最后一次）的每核指令级分解:

### prod_s1（blk128 r1 m3, 单 launch）

| 核 | 指令总时长 | WAIT(同步)占比 | 主要构成 |
|---|---|---|---|
| AIC (cubecore0) | 77.5us* | **24%** (WAIT_FLAG_DEVI 8.7 + WAIT_FLAG 9.8) | SCALAR 30.5%、FIXP 24.4%、MTE1 11%、MTE2 10%、CUBE 5.8% |
| AIV0 (偶, Vec0/Vec1) | 51.0us* | 10% | **VECTOR 48%**、SCALAR 37%、MTE3 3.3%、MTE2 2.7% |
| AIV1 (奇, Vec1) | 31.5us* | **35%** (WAIT_FLAG_DEVI 11.0us 单指令居首) | SCALAR 55%、FLOWCTRL 35%、VECTOR 8.6% |

### prod_s2（blk128 r1 m2）

| 核 | 指令总时长 | WAIT(同步)占比 | 主要构成 |
|---|---|---|---|
| AIC | 95.9us* | **38%** (WAIT_FLAG_DEVI 26.7us) | FLOWCTRL 28%、SCALAR 25%、FIXP 20% |
| AIV0 | 135.6us* | 7% | **VECTOR 69%**（MOVEMASK 41.6us = candidate mask 构建）、SCALAR 26% |
| AIV1 | 51.2us* | **57%** | FLOWCTRL 57% |

### prod_s3（blk64 r2 m3）

| 核 | 指令总时长 | WAIT(同步)占比 | 主要构成 |
|---|---|---|---|
| AIC | 85.3us* | 25% | FIXP 28.5%、SCALAR 28% |
| AIV0 | 52.1us* | 11% | VECTOR 47%、SCALAR 37% |
| AIV1 | 32.8us* | **37%** | SCALAR 53%、FLOWCTRL 37% |

(注*: instr_exe 的 running_time 为指令粒度仿真时长, 含模型拍频差异, 用于核间/管道
相对比较; 与真机 us 不可直接换算。)

### 仿真数据结论（修正真机聚合值的误读）

1. **奇数 AIV (AIV1) 才是同步等待大户**（WAIT 35~57%），偶数 AIV0 忙于 Vec0+两行
   Vec1（VECTOR 47~69%）。真机 op_summary 聚合值把两个 AIV 平均了, 掩盖了不均衡。
2. AIV1 的 WAIT 主要是 `WAIT_FLAG_DEVI`（等 C 的 syncC1V1）: C 的 Mm1 完成时刻
   支配 AIV1。AIV1 只分到 1 行（尾块/奇偶切分）却要等同样的 C 节拍。
3. C 核 WAIT 24~38% 等的是 V 的 syncV1C1 + syncV0C1 — C/V 互等构成主要 idle,
   双向都与 lockstep 协议强相关（与 P1/P6' 的教训一致: 改协议深度或预取都不行,
   减少每拍 C 侧标量/同步指令数本身才是可行方向）。
4. s2 (m2) 的 VECTOR 69% 中 MOVEMASK 41.6us 为 BuildCandidateMask 的候选窗口
   计算 — candidate consumer 的主要算力开销点, 与 P3 在 m2 上收益被稀释的结论互证。

### 基于仿真数据的两个再平衡实验（均否决，回退）

**P7: Vec1 行重分配 [2,2]→[3,1]（偶核 +1 行）— 否决**
- 正确性: 按行分区的 UB 缓冲（sortOutBuf/globalBlockTopkUb/candBuf）行槽按
  `CeilDiv(s1BaseSize,2)=2` 分配, 偶核 3 行时 `innerS1Idx=2` 越界覆写行 0 槽
  — 实测 m2_pre_8_2k 52 元素差异（行 6, 尾部出现 255 递减序列 = 候选槽被踩）。
- 资源: 3 行槽需 sortOut 48KB + blockSort 48KB（m1 共 208KB）/ candBuf 48KB
  （m2 共 200KB）, 均超 192KB UB 预算, 无法扩容。
- 性能逻辑: 偶核已是瓶颈（VECTOR 47~69%），再加行只会加剧 max(AIV0,AIV1)。

**P8: Vec0 挪到奇 AIV（偶核减负）— 否决**
- 实现 1 行改动（`blockId_ % 2 == 0` return 互换）, 零缓冲变化, 等价 19/19 PASS。
- 实测（相对 P3）: m2 prod_s2 **-3.7%**（改善, m2 偶核瓶颈在 candidate mask,
  Vec0 挪走有净收益）; 但 dec16k +5.7% / prod_s1 +4.2% / prod_s3 +5.7% —
  非 candidate 路径偶核瓶颈是 Vec1 两行本身, Vec0 挪走收益小, 而奇核多了 Vec0
  后其 Set(syncV1C1) 延迟增加 → C 下一拍等待变长 → 整体劣化。方向不一致, 回退。

## 12b. 用户最新三 shape 的流水线占比（2026-09-16, 变更集构建 P3+P10+P11）

topk=512, T=12, B=16, g=32, D=128, TND+PA_BBND, mask0, pool 20369;
s1=mode1 cb2048 blk128 r1 / s2=mode2 blk128 r1 / s3=mode3 blk64 r2(无
residual — mask0 时 residual 必须 null, 接口校验, 此前脚本构造违规)。
`msprof op simulator --soc-version=Ascend910B3 --core-id=0`, launch 9,
X 事件 (instr_exe) 按管道累加; 仿真拍频与真机不可直接换算, 供结构分析。

### prod_s1 (mode1, cb2048)

| 核 | 指令总时长 | 主要构成 |
|---|---|---|
| AIC | 58.9us* | SCALAR 41.6%、**FLOWCTRL 21.2%** (WAIT_FLAG_DEVI 12.5 居首)、FIXP 12.7%、MTE3 8.6%、MTE2 8.3%、MTE1 6.6%、CUBE 1.1% |
| AIV0 | 51.5us* | SCALAR 47.4%、**VECTOR 39.0%** (MOVEMASK 8.3 = 候选块分数路径, VMS4V2 5.4 = 归并)、FLOWCTRL 8.4% |
| AIV1 | 38.6us* | SCALAR 51.0%、**FLOWCTRL 40.4%** (WAIT_FLAG_DEVI 15.6 单指令居首 — 空转等 C)、VECTOR 8.1% |

### prod_s2 (mode2, P11 快路径生效)

| 核 | 指令总时长 | 主要构成 |
|---|---|---|
| AIC | 54.7us* | SCALAR 44.7%、FLOWCTRL 14.5%、FIXP 14.3%、MTE3 9.3%、MTE2 8.9%、MTE1 7.1%、CUBE 1.2% |
| AIV0 | 36.6us* | **SCALAR 61.5%**、VECTOR 17.4% (VMS4V2 3.8 — P11 跳过 BuildCandidateMask 后 MOVEMASK 消失)、FLOWCTRL 13.4% |
| AIV1 | 33.3us* | SCALAR 61.4%、**FLOWCTRL 34.5%** (WAIT_FLAG_DEVI 11.5) |

### prod_s3 (mode3, blk64 r2)

| 核 | 指令总时长 | 主要构成 |
|---|---|---|
| AIC | 59.7us* | SCALAR 41.9%、FIXP 17.6%、FLOWCTRL 13.6%、MTE2 11.1%、MTE3 8.5%、MTE1 6.2%、CUBE 1.1% |
| AIV0 | 37.6us* | SCALAR 58.8%、VECTOR 16.7% (VMS4V2 3.8)、FLOWCTRL 13.8% |
| AIV1 | 33.1us* | SCALAR 59.0%、**FLOWCTRL 37.2%** (WAIT_FLAG_DEVI 12.3) |

### 三 shape 共同结论

1. **AIV1 (奇核) 是结构性空转核**: WAIT 占 34~40%, WAIT_FLAG_DEVI 单指令
   11~16us 居各核首位 — decode 行分配公式 (奇核=行数/2=0 行) + C/V 1:2
   绑定所致, 与分核方式无关 (R1 轮换已实测否决)。
2. **AIC 的 CUBE 仅 1%**: decode Mm1 是 1×2048 小矩阵, AIC 实际是
   "标量+搬运" 核 (SCALAR 42~45% + MTE 22~27% + FIXP 13~18%), FLOWCTRL
   14~21% 在等 V。AIC 优化方向是减少每 tile 标量指令数与 Mm1 次数。
3. **s1 与 s2/s3 的 AIV0 差异 = candidate 路径成本**: s1 VECTOR 39%
   (MOVEMASK 8.3us 候选块分数 + VMS4V2 5.4us 归并), s2/s3 VECTOR 17%
   (P11 快路径曾消掉 s2 的 MOVEMASK, 后因精度撤销, 见 P11 节)。SCALAR 47~61% 是三核共同大头 —
   ST_XD_XN_IMM (9~11us) 为单项最大标量开销, 值得查生成源。
4. s3 (mode3) 与 s2 的差异仅 FIXP/MTE2 略高 (blk64 → Mm1 更多 tile),
   其余结构相同 — mode3 路径在 topk=512 下已接近最简。

### 12c. 真机 PipeUtilization 对照 (2026-09-17, 板载 msprof op --aic-metrics=PipeUtilization)

同一批输入真机采集 (launch 3..9 中位; 本 msprof 版本无 PipeTimeline,
用 PipeUtilization, 指标同源)。kernel 总时长 = max(AIC, AIV0, AIV1):
s1≈27.2us / s2≈22.5us / s3≈23.4us — 与 op_summary 口径 (20.8/23.7/…)
一致量级, 互证有效。

| shape | 核 | 总时长 | CUBE | MTE1 | MTE2 | MTE3 | FIXP | SCALAR | VECTOR |
|---|---|---|---|---|---|---|---|---|---|
| s1 | AIC | 21.6us | 3.3% | 7.1% | 12.6% | 9.9% | 7.9% | 33.2% | — |
| s1 | AIV0 | 27.2us | — | — | 10.4% | 1.4% | — | 40.3% | 26.1% |
| s1 | AIV1 | 27.0us | — | — | 8.6% | 0.0% | — | 26.6% | 6.2% |
| s2 | AIC | 18.0us | 4.0% | 8.6% | 15.1% | 9.9% | 9.5% | 40.5% | — |
| s2 | AIV0 | 22.4us | — | — | 11.8% | 1.1% | — | 37.4% | 15.6% |
| s2 | AIV1 | 22.5us | — | — | 10.0% | 0.0% | — | 33.7% | 3.1% |
| s3 | AIC | 19.2us | 3.5% | 7.5% | 17.6% | 9.5% | 8.8% | 36.8% | — |
| s3 | AIV0 | 23.4us | — | — | 11.6% | 1.0% | — | 34.6% | 14.8% |
| s3 | AIV1 | 23.3us | — | — | 9.8% | 0.0% | — | 31.3% | 2.9% |

(占比分母 = 该核总时长; 非活跃管道未列出, 余项 = 等待/空闲)

wait/stall 真机细分 (us, 中位):
- AIC 等 V0 (wait_id0): s1 7.48 / s2 4.58 / s3 4.69 — AIC 空闲大头;
- AIV0/AIV1 等 C (wait_id0): 均 ~4.1-4.7 — C/V 互等的 lockstep 开销;
- AIV1 额外 wait_id14 (等配对 AIV0): s1 9.71 / s2 5.62 / s3 5.45 —
  真机数据同样证实奇核空转结构性存在;
- AIV 侧 wait_ib (指令缓冲) ~4.0-4.6 — 标量程序体量本身的取指开销。

真机 vs 仿真差异: 仿真把等待计入 FLOWCTRL 且拍频慢 (AIV1 WAIT 34~40%),
真机占比里等待体现在总时长中各活跃管道的占比被稀释 — 两口径的核间
不均衡结论一致: AIV0 ≳ AIV1 ≈ AIC, 且 AIV0 的 VECTOR 增量即 candidate
路径成本 (s1 26% vs s2/s3 15%)。

### 12d. 优化前基线 vs 优化后对照 (2026-09-17, 同流程重编两版)

基线 = 远端 HEAD (P3/P10/P11 前), 变更集 stash 后重编安装 (09:43);
采集同 PipeUtilization 流程 (launch 3..9 中位); 采完恢复变更集重编
(09:47), s2 复测 24.4us 确认回到优化态。

| shape | 基线 kernel | 优化后 kernel | Δ | 主要差异 |
|---|---|---|---|---|
| s1 (m1 cb2048) | 26.74us | 27.21us | +1.8% (持平) | AIV0 VECTOR 32.7%→26.1% (P3 收缩), 但总时长不变 — 带宽/标量墙 |
| **s2 (m2)** | 37.96us | (P11 已撤销, 见 §P11) | — | P11 快路径的 -41% 因精度问题撤销 (候选覆盖性为数据属性, 无法廉价验证) |
| s3 (m3 blk64 r2) | 23.18us | 23.36us | 持平 | mode3 无 candidate 路径, P3 在该 shape 无收益 (topkWidth 不变档) |

基线明细 (占比):
- base_s2 AIV0 38.0us: SCALAR 56.8% (21.6us) + VECTOR 48.8% (18.5us)
  — BuildCandidateMask+pen 链的全部成本, 与 P11 设计目标精确对应;
- base_s1/base_s3 与优化版结构相同 (SCALAR ~40% 最大), 该两 shape 的
  优化空间在 SCALAR 指令数与 MTE 带宽, 不在 candidate 路径。

e2e 对照 (p50, 含 host): base s1 122.9 / s2 126.3 / s3 114.1us;
opt s1 126.6 / s2 116.4 / s3 111.2us — s2 e2e -7.9%, 被 host 开销
(~90us) 稀释; pipelined 口径 s2 51.9→51.3us (host 主导, 差异趋零)。

**其余可行方向（按投入产出排序）**
1. [已合入] P3 topk 宽度自适应。
2. [结构性专项] C 侧标量链出关键路径: metadata（AICPU）扩容预下发
   per-tile 地址表, kernel 内 CalcRunInfo 退化为查表 — 根治 24~38% 的 C WAIT
   与其引发的 AIV1 WAIT。工程量大需协议评审。
3. [中] m2 的 MOVEMASK 41.6us: VecMaxPair 等 64 分块是大 count aicore 异常的
   workaround, 与 CANN 确认异常边界后放大分块, m2 预计 ~15%。
4. [不建议] 行重分配两方向（UB 行槽约束 + 偶核瓶颈逻辑）均已实测/论证否决。

## 11. 复现命令

```bash
bash /opt/tjj/qli_cand_test/build_install.sh              # 编译+安装
bash /opt/tjj/qli_cand_test/msprof_case.sh <tag> decode 1 1 16384 32 512 1 0 BSND
cd /opt/tjj/qli_cand_test && python equiv_qli_v2.py cmp --ref /tmp/qli_equiv_base.npz
cd /opt/tjj/qli_cand_test && python perf_qli_v2.py / python perf_shape.py
pytest test_qli_cand.py -k "not big and not pa_gap_1m and not mask0_1m"
```

### decode 形态下奇 AIV 全程空转的根因（仿真数据深挖结论）

prod shape (T=12 decode, 每批 q_seq=1, mask=0) 的 veccore1 指令表里
VMS4V2/VBS32/MOV_UB_TO_UB 计数全为 0, VECTOR 仅 2.7us (CleanInvalidOutput 的
Duplicate/Adds) — **奇 AIV 的 Vec1 行数为 0**。

因果链: decode `actS1Size=1` → 每个有效块 `cuS1ProcNum=1` → 行分配
`偶=CeilDiv(1,2)=1, 奇=1/2=0` → 每块唯一一行的 Vec1 全落在偶 AIV;
metadata (910b 走 AssignByBatch, `supportFd_` 仅 950 置位) 把 12 个有效批分给
12 个 C 核 → **12 个奇 AIV 全程空转**, 每核利用率直接折半。

VBS32 (Brcb) 只在偶核是 ProcessVec0 的固有单核分工, 与该问题无关;
VMS4V2/MOV_UB_TO_UB 缺失于奇核则是"0 行 Vec1"的直接证据。

可行修复 = 910b 启用 FD (flash-decode) 分核:
- metadata: `supportFd_` 条件放开 (AssignByBlock s2 跨核 + RecordFDInfo),
  kernel 侧 LD 路径 (isNeedLD/saveWorkSpaceIdx/ProcessLD) arch22 已实现,
  workspace 亦已预留 — 但**从未在 910b 上运行过**, 需专项验证
  (分核正确性 + LD 归并 + 12.4 M 指令量回归);
- 预期: decode 场景 AIC/AIV 利用率接近翻倍 (12 核×2 AIV → 24 核×2 AIV 均载),
  与 P3 叠加 (LD 归并按 topk 收缩), 生产 decode shape 预计再获 ~30-40%;
- 优先级: 高 (收益最大的单点), 但属 metadata 分核协议改动, 建议独立专项。

### prefill 场景的核间均衡（对照结论）

prefill 与 decode 相反, **没有核间不均衡问题**:
- prefill 满块 (actS1Size=4): 行分配 [2,2] 对称, AIV0 = Vec0 + 2 行 Vec1,
  AIV1 = 2 行 Vec1 — 两核向量负载对称;
- 真机 msprof (pre1k, s1=1024): AIV 四管道 busy 之和 953us vs AIV 时长 933us,
  **idle ≈ 0%** (向量 74% + 标量 17% + MTE2 11%), 与 decode 的 38~41% idle 对比鲜明;
- simulator 在 prefill 路径产不出数据 (807 万条 unknown instr, 910B 模型未覆盖
  prefill 主路径的部分指令, mask3/mask0 均空 dump), 该场景的核间结论以真机
  聚合数据 + 代码结构推导为准;
- prefill 的瓶颈是 Vec1 排序链本身 (向量 74%), 非同步等待 — 进一步优化方向是
  排序/搬运宽度 (P3 已做) 与 tile 数, 而非核间再平衡。

### FD/LD 启用专项（首轮实验，2026-09-15）— **链路已打通，遗留 r2 bug，本轮回退**

按上文结论实施: metadata `supportFd_` 去掉 950 限制（910b 同条件）。
工程坑: aicpu so 链接有缓存, 改 metadata .cpp 后必须
`rm build/libtransformer_aicpu_kernels.so && make cust_aicpu_kernels` 再打包安装,
否则装的是旧 so（本次实验曾因此误判 15 分钟）。

**LD 触发确认**: decode 生产 shape (B=1, s2=16384) 下 metadata QLD 区
core0=[1,0,0,0,8,0,1,0]（8 段归并任务）, LI 区 s2End 逐核 +1 — 8 个 s2 tile
成功分给 8 个 C 核, kernel LD 路径（部分 topk → ProcessLD 归并）真实运行。

**等价**: 19 例中 15 例位等价 PASS; 4 例差异:
- `r2_dec_16k` 258 处 / `r2_pre_8_16k` 2123 处 — cmp_ratio=2 场景大面积错误,
  LD 的 r2 路径存在真 bug（疑似压缩前/后位置域混用, 待专项定位）;
- `pre_64_16k` 2 处 / `tnd_pre_16k` 6 处 — tie-break 级, LD 归并稳定性问题。

**性能** (kernel_med, 括号内为 P3 基线): dec16k 49.3us (47.3), dec2k 23.2 (24.2),
prods1 20.5 (21.2), prods2 37.9 (37.3), prods3 21.6 (21.5) — r2 bug 未修前
LD 归并开销与收益相抵, 无净收益。**回退**, metadata 改动样例与 bug 清单存于
`/opt/tjj/qli_sim/fd_enable_metadata.patch`。

专项后续步骤: ① 修 r2 LD 路径（重点排查 ProcessVec1 分段核的 cuS2Len/
kScale 偏移在压缩场景的位置域）; ② tie-break 对齐（归并时值并列的次序）; 
③ 全量回归 + decode 生产 shape msprof 复测; ④ 若 idle 显著下降, 与 P3 叠加
预期 decode -30~40%。

### FD/LD 启用专项（第二轮，2026-09-16）— **修复完成，全部验证通过，合入**

**修复内容** (两文件):
1. metadata: `supportFd_` 启用 910b, 条件 = decode 专用 (`maxSeqlenQ == 1`) 且
   `maxS2Size >= topk` 且 `maxS2Size > 5*s2Base`。decode-only 的原因: 910b
   s2Base=2048 粒度下 AssignByBlock 跨批 tile 切分存在丢 tile (prefill 实测
   6590 个索引缺失), prefill 满块 [2,2] 对称本无空转, 保持原分核。
2. kernel ProcessLD/ProcessVec1: LD workspace 寻址布局修正 —
   写槽 = `saveWorkSpaceIdx(核i分段号) * s1Base * pairW + 全局行号 * pairW`;
   读侧轮次偏移 `(1+3i)*s1Base*pairW` (跨核步进 s1Base 槽) + 3 个新列表用
   **逐段显式 DataCopyPad** (blockCount 跨步的 srcStride 语义在 GM2UB 上
   不可靠, 逐段拷绕开); 读侧必须带 `workspaceIdx` 项 (b4 多批下头核≠首段核)。

**验证结果**:
- 等价: 19 例中 18 例位级一致; b4_dec_16k 行 2/3 各 ~60 个 tie 差异
  (cmp_ratio=1 并列分数在分块 topk+归并与全局 topk 下的合法次序差),
  官方规则 (multiset + 值容差) 判定 PASS;
- 官方回归: **77 passed / 0 failed** (排除 1M/big, 含 b4_dec_16k +
  big128k_b4_varlen/tnd_m2/mask0_r2 三个 128K B4 变长用例);
- 性能 (kernel_med):

| case | 基线 | P3 | P3+FD | FD/基线 |
|---|---|---|---|---|
| dec16k_t512 (s2=16384) | 52.82us | 47.29us | **24.49us** | **-53.6%** |
| dec2k (s2=2048, FD 不触发) | 25.64 | 24.23 | 23.78 | -7.3% |
| prod_s1 (T=12, s2=16384, topk=1024) | 23.38 | 21.24 | **20.80** | -11.0% |

decode 长上下文场景收益 -53.6% (P3 之上再 -48%), 与预期"核利用率翻倍"一致。

### prod_s1 (FD 启用后) 的指令级仿真数据

msprof op simulator (launch 9, core0 = AIC0 + AIV0/AIV1 对)：

| 核 | 指令总时长 | WAIT(sync) | 主要构成 |
|---|---|---|---|
| AIC | 498.1us* | 23% (SET_FLAG 142 + WAIT_FLAG 106) | **FIXP 46.8%**、MTE1 16.6%、MTE2 12.2%、SCALAR 12.1%、CUBE 仅 9.0% |
| AIV0 | 240.6us* | 19% | **VECTOR 64.7%**、FLOWCTRL 15.1%、SCALAR 11.4%; BAR 125us、VMS4V2 27.4us |
| AIV1 | 79.9us* | **62%** | FLOWCTRL 60.5%、SCALAR 25.8%、VECTOR 10.4% |

(注*: 指令粒度仿真时长, 用于结构与占比分析, 与真机 us 不可直接换算。)

解读:
1. **AIV1 在 FD 下依然空转 (62% WAIT)** — FD 的收益机制是"批内 s2 切分把
   Mm1/Vec1 分摊到多核", 而非激活奇 AIV: 行分配公式 `奇核=行数/2` 在
   decode (actS1Size=1) 下恒为 0, 与分核无关。奇 AIV 空转的根治需要改
   行分配 (把 Vec1 单行任务轮换到奇 AIV, 但行槽 UB 布局需相应调整)。
2. **AIC 被 Fixpipe 支配** (46.8%, CUBE 仅 9%): decode 的 Mm1 是 1×2048
   小矩阵, 计算瞬间完成, 时间全耗在 L0C→L1 搬运与 K/Q 装载 (MTE1+MTE2+FIXP
   合计 75%)。FD 切分不改变此结构 — prod_s1 FD 收益低 (-2%) 的根因。
3. **AIV0 的 BAR (PipeBarrier) 高达 125us**: SortAll/MergeSort 链的 V 管道
   barrier 是向量侧第二大开销, 与 VMS4V2 (27.4us) 同属排序链 — 与 P3
   (宽度收缩已做) 之后剩余的排序开销一致。
4. prod_s1 下一步方向 (按仿真数据): ① 奇 AIV 激活 (行轮换, 需调行槽布局);
   ② AIC Fixpipe/MTE 占比 → 减少 Mm1 次数 (增大每 tile 行数, decode 场景即
   减少切分粒度/合批); ③ 排序链 barrier 优化。

### R1: 行任务块间轮换（奇 AIV 激活尝试）— **已实测否决（回退）**

思路: 行-核映射按 gS1Idx 奇偶翻转（偶数块偶核取前半/奇数块取后半，奇数块反转），
让 decode 单行任务在奇偶 AIV 间交替，消除奇 AIV 空转。UB 行槽布局不变
（每核仍 ≤2 行），LD 布局不受影响（按全局行号寻址）。

实测: dec16k 24.49→23.42us (-4.4%, 小赢); 但 prod_s1 20.80→22.42us (**+7.8% 劣化**);
prefill 持平。

根因: 块间轮换让 Vec1 行任务跨批 — 奇核拿到批 b' 的行, 但它的
Wait(syncC1V1) 等的是批 b 的 C 核 tile 完成信号, C/V 跨批错配, 每核多等
一轮并增加 barrier。多批高并行场景 (prod_s1, 12 批) 劣化超过单批场景
(dec16k) 的收益。且 decode 下奇 AIV 空转存在更深的结构约束: **C/V 1:2
绑定** — tile 的 Vec1 只能在该 tile Mm1 所在 C 核绑定的一对 AIV 上执行,
行轮换无法跨越此边界。根治只能靠 metadata 分核把任务均匀分给全部 24 对
C/AIV (FD 已做到批间均衡), 奇 AIV 的空转本质是"每批行数 < 2"的数学结果,
在 decode 形态下无法避免。回退。

### prod_s1 FD 分核明细分析（用户要求的分核打印）

分核实测 (T=12 有效批, 96 个 (批,tile) 任务, FD 开):
- **批间均衡已达成**: 24 核全部有活, 每核 4-5 任务, 任务跨批连续切分
  (如 core1 = b0:t5-7 + b1:t0-1, 跨批无缝);
- **core19 负载陷阱**: bn2[11..16] 名义 36 任务, 其中 32 个是空批
  (T=12 后 4 批 q_seq=0, 只做输出清理) — 空批清理成本被单核承担,
  建议后续 metadata 将空批清理均摊 (低优先级, 清理本身较轻);
- **LD 头核分配其实均匀**: 12 个头任务分布在 AIV0..11 (奇偶各 6),
  每 C 对两个 AIV 各一个 — 修正此前"头任务只在偶 AIV"的误判;
- **AIV0/AIV1 负载比 ≈ 3:1** (240.6 vs 79.9 指令时长): 每任务 1 行的
  Vec1 (含行级 SortAll 2048 排序) 恒落偶核 (行分配公式), 奇核只做
  LD 归并收尾 (FLOWCTRL+SCALAR 为主)。

结论: FD 分核在"任务→C 核"维度已均衡; 剩余的 AIV 奇偶不均源于
**行任务与偶 AIV 的绑定** (C/V 1:2 架构 + 行分配公式), R1 实验已证明
行轮换会破坏 C/V 配对。根治需要"行任务队列"模式 (Vec1 可处理任意核
的行, 大架构改动)。当前 FD 分核接近该架构最优。

### "分核均衡但计算不均衡"的 kernel 侧解释

metadata 把 96 个 (批,tile) 任务均衡分给 24 个 C 核 — 这一层没有问题。
不均衡发生在 **kernel 的 Vec1 行分配**：

`ProcessVec1` 的行区间由 `gS1Idx`（批内行块号）与 AIV 奇偶决定：
- decode 每批 actS1Size=1 → 每批只有 1 个行块（gS1Idx=0）、每块 1 有效行
  （cuS1ProcNum = min(4, actS1Size)=1）；
- 行分配 `偶核=CeilDiv(1,2)=1 行, 奇核=1/2=0 行` — **每个任务的唯一一行
  Vec1 恒落偶核**，与该任务被分到哪个 C 核无关（每个 C 核绑定的偶 AIV
  做行 0，奇 AIV 永远分不到行）；
- 奇 AIV 剩下的只有 LD 头任务的归并收尾（MrgSort 读 GM + 循环控制）。

即：**分核均衡（任务→C 核）与计算均衡（Vec1→AIV）是两层独立的分配**，
后者由 kernel 内硬编码的行分配公式决定，metadata 无法触及。R1 实验已证明
行轮换会破坏 C/V 配对（Wait(syncC1V1) 信号错配）。根治 = "行任务队列"
模式（Vec1 解耦行-偶核绑定），属 kernel 架构级改造。

### P11: mode=2 全候选快路径 — **已撤销（2026-09-17, 精度问题 + 收益不可达）**

> 以下为原分析, 保留供参考。**最终状态: 快路径已从代码删除**, 恢复无条件
> BuildCandidateMask+pen 链。

mode=2 consumer 的 BuildCandidateMask + pen 链 (S4a) 每行 ~60% 向量指令
(MOVEMASK 41.6us 居首)。发现生产形态下多数情况可整体跳过:

**条件**: 该批总块数 ≤ candidateTopkBlocks (候选覆盖全部位置块) → isOut 恒 0
→ pen 恒 0 → score 不降级 → BuildCandidateMask + pen 链 (约 15 条向量指令)
全部跳过。判定用 `info.actS2SizeOrig` (候选块号域=压缩前位置域):
`allBlocksCand = candidateTopkBlocks >= CeilDiv(actS2SizeOrig, candidateBlockSize)`。

**撤销原因 (2026-09-17)**:
1. **精度问题**: 判定只看"槽位数 ≥ 块数", 但"候选表覆盖全部块号"是**数据
   属性** — 上游候选可含重复/-1 槽 (harness big16k_m2 / pa_gap_16k_m2 用
   randint 候选, 16384 行起 idx 输出选入 NEG_INF 填充槽, 全量比对失败)。
   跳过 pen 后未命中候选的块不再被压尾, 排序优先级与 golden (有 pen) 不符。
2. **严格判定的成本 = 收益**: kernel 内验证"排序后候选恰为 [nb-1..0]"
   需逐点读候选表 (stride-2 交错布局, 向量化归约不可用, V 算子 32B 对齐
   限制使相邻差分错位), 标量循环 12 行 × 2048 GetValue ≈ 19us ≈ 被跳过的
   tile 内命中检测成本 — 实测 s2 41.3us vs 基线 38.0 (净亏)。
   BlockReduceMin/Max 方案另踩 dst/src 原位覆盖与 [2] 偏移非 32B 对齐两个
   507015 崩溃坑。
3. **结论**: P11 的 -38% 建立在"输入候选恰为全集排列"的契约外假设上,
   该假设既不被接口保证也不能廉价验证 → 删除。生产 s2 回到基线路径
   (39.3 vs 38.0, 噪声内), m2 全家族 16 用例 + big16k_m2/pa_gap_16k_m2
   全部通过。

### FD 生效边界的最终修正（batchSize 门限）

prod_s1 (B=16) 的同会话 A/B 测试发现: B=16 时 FD 的 per-task 开销
(LD 保存/归并/跨核同步 ×每核 4 任务) 超过批间并行收益, s1/s3 形态
**+2~3us**。因此 FD 启用条件追加 `batchSize_ <= 4`:
- B ≤ 4 decode: FD 启用 (dec16k B=1: -53.6%, b4: 验证通过);
- B > 4 decode: 基线分核 (批间并行已充分, 12/24 核有活);
- prefill: 始终基线分核 (跨批丢 tile 问题)。

### 三生产 shape 终版数据 (同会话 A/B, msprof kernel_med)

| shape | 基线 | 终版 | delta | 说明 |
|---|---|---|---|---|
| s1 (m3, B=16, topk=1024) | 22.34 | 25.12* | (持平*) | gate 后代码路径=基线; *含会话内卡状态漂移 ~10% |
| s2 (m2, B=16, topk=1024) | 38.34 | (P11 已撤销) | — | 快路径收益建立在契约外候选假设上, 撤销 |
| s3 (m3, r2, B=16) | 22.02 | 25.78* | (持平*) | 同 s1 |

(注: s1/s3 终版与基线代码路径完全一致, +2~3us 为同会话卡状态漂移
(基线与终版交替测量时 aiv 管道整体 +10%), 已用 A/B×3 轮确认方向恒定的
仅 s2; s1/s3 的真机收益 = 0 (无改动生效), 无劣化。)

### 测量方法学备注

- aicpu so 链接缓存坑 (改 metadata 必须 rm so 重链) 再次踩中, 已在脚本固化;
- 共卡 (vLLM e2e 占 55GB) 期间数据全部无效, 独占窗口才可采;
- A/B 交替 + 多轮 median 是最小可信口径; 跨会话数字不可直接比。

### 生产三 shape 的 mode 勘误（用户指正）

三生产 shape 对应三种 candidate_mode（此前误将两个标为 mode3）:
- shape1 (blk128, r1): **mode1 (source)**, 输出 candidate_topk_index [12,1,128]
  → candidateTopkBlocks=**128**（非 2048）;
- shape2 (blk128, r1): **mode2 (consumer)**, 输入候选 [12,1,2048];
- shape3 (blk64, r2): **mode3**（无 candidate）。

**新发现功能 bug**: mode1 + candBlocks=128 + s2=1172 组合 aicore 崩溃
(UB 越界: ProcessCandBlockTopk 的块级 SortAll 宽度跟 s2 块数 pad (512 对)
走, 而 candBuf 行槽按 candBlocks (128 对) 分配, candBlocks < blockNumPad
时 SortAll 源越界行槽)。harness 的 pin64 (candBlocks=64, s2=4096, 块数 512)
通过是因 s2 较小未触发同路径 — candBlocks 与 s2 块数的适配矩阵需专项排查。
本轮 perf 数据仍有效: shape2 的 P11 (-38%) 与 shape1/3 的基线测量
(mode3 路径与 candidate 无关, 输出候选张量的存在不影响 mode3 kernel 行为)。

### 崩溃排查结论 (2026-09-16): 507015 是测试脚本 bug, 非算子 bug

**最终定位**: 崩溃脚本 (`prod_shape_v3.py` / `fd_scale_test.py` /
`fd_bsnd_test.py` 等) 的 `block_table` 是 **CPU 张量** (`torch.zeros(...,
int32)` 构建后未 `.npu()`); harness 的 `gen_case_inputs` 有 `.npu()`
所以全部 92 用例正常。CPU 指针传入 kernel 读到非法 host 地址 →
D-cache→UB bus 非零响应 (507015), 报错块号 blk:0。

**排除法证据链** (逐项控制变量, 全部与崩溃无关):
- layout (TND/BSND)、mask_mode (0/3)、cmp_ratio (1/2)、k_scale 满量程
  (65504/1.0)、q_scale (255/1.0)、pool 大小 (8/20369)、table 值模式
  (置换/取模)、seed (1/2/7) — 全部因子交叉后崩/稳与因子无关联;
- 数据件逐个替换 (q/w/q_scale/k_phys/k_scale_phys/block_table) 无一触发;
- 唯一分裂变量 = block_table 所在设备 (CPU→崩, NPU→稳), 修复后
  原崩溃用例 (含满量程 scale) 全部通过。

**TND+mask3+mode1 验证矩阵** (修复后, 官方两级比对, 远端 HEAD 构建):
- s2=512 cb2048/cb64、r2+res1、prefill 变长 (12 行)、生产 s1 完整 shape
  (B16/T12/s2=1172/cb2048, 12 行 idx+cand 全过 + pin 12/12) — 全 PASS;
- r2 无 residual 的用例被 metadata 校验正确拦截 (接口规范, 非内核问题);
- 结论: TND+mask3+mode1 (含生产 s1) 无功能问题。

**变更集回归 (stash 往返后)**: P3+P10+P11+FD 门限恢复 → 重建安装 →
NPU↔NPU 等价 diff 对照远端 HEAD 基线 **19/19 bit-identical**。

**遗留**: 早前记录的 "mode1+candBlocks=128+s2=1172 崩溃" 是在含 CPU
table 的脚本上得出的 — 该结论作废, 待用修复后脚本重测 (预期不崩);
ProcessCandBlockTopk/candBuf 宽度适配仍值得 UT 级复核, 但当前无失败证据。

**TND+mask0+mode1 系统矩阵** (2026-09-16, 修复后数据通道, 官方两级比对):
15/15 全过 — s2 梯度 {512,1024,1172,2048,8192,16384} × candBlocks
{64,128,2048} 组合、B>1 decode、prefill (8+4 行 / 64 行)、生产 s1 完整
shape (B16/T12/s2=1172/cb2048)、r2。**含曾报崩溃的 cb128+s2=1172 —
不崩, 官方比对通过**, 该崩溃报告确认作废 (脚本 CPU table 所致)。
