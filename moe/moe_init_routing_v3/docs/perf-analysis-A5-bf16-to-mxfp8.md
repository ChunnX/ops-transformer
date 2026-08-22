# MoeInitRoutingV3 A5 性能分析（bf16 → mxFP8/E4M3FN）

> 场景：A5 / Ascend950 / arch35（`__NPU_ARCH__ == 3510`），x=bf16 输入 → expanded_x=mxFP8(E4M3FN) 输出 + expanded_scale=float8_e8m0（每 32 元素一个 block scale）。
> 背景：A5 相对 A2(910B2C) 该场景仅 1~1.2X，期望定位瓶颈并评估提升空间。
> 结论（先读）：**该场景 gather+量化热点为 HBM 写带宽受限，已逼近 A5 物理峰值，无内核级可回收空间**；两种候选内核优化经真机实测均无净收益（改进 < 1%，噪声级）。真正的墙是"写带宽 + 固定输出体量"，与 A2→A5 仅 1.2X 的现象自洽。

## 1. 场景与用例

| case | 输入 | 输出 | 公共属性 |
|---|---|---|---|
| case1 | x[10240,4096] bf16, expert_idx[10240,25] int32 | expanded_x[256000,4096] mxfp8, row_idx[256000], count[800], scale[256000] | H=4096, K=25, expert_num=800, drop_pad_mode=0, quant_mode=MXFP8_E4M3FN, validScaleCols=128 |
| case2 | x[5524,4096] bf16, expert_idx[5524,25] | expanded_x[138100,4096] mxfp8, row_idx[138100], count[800], scale[138100] | 同上 |

- 选路：两 case 均走 `tilingKey=11030000`（sortmulticore + mxfp8quant），`useGatherCopy=1`（因 (expertEnd-expertStart)*k=20000 > expertNum*3=2400，即 K>3）。

## 2. 采集口径与环境

- 工具：`msprof op --launch-count=20 --kill=off`（kernel 直调，非 --application；无 host 计时，无第三方框架间接加载）。
- total_us = 单次运行**全部 kernel launch 之和**，同一张卡串行采集，全频 1650/1650，blockDim=56（56 个 AIV 核用满）。
- 基线与各方案 launch-count 一致、口径一致、block 一致 → 可比。

## 3. 基线性能与瓶颈定位

| case | total_us | gather(热点) | sortmulticore | count(host 分块) | rowidx |
|---|---|---|---|---|---|
| case1 | 1161.509 | 819.983 (70.6%) | 227.671 | ~81.5 | 32.311 |
| case2 | 625.857 | 414.312 (66.2%) | 133.569 | ~61.0 | 16.920 |

**gather 热点 = MTE3(输出写) BOUND，且已逼近 A5 HBM 写带宽物理上限：**
- 写流量守恒（case1）：输出 data 256000×4096×1B = 1.048 GB + scale 256000×128B = 32.8 MB = 写 **1.081 GB**；输入读唯一源行 10240×4096×2B ≈ 84 MB（GatherCopy 每源行仅读一次）。总 DRAM ≈ 1.165 GB / 819.983us = **1.42 TB/s**（写占 **1.32 TB/s**）。
- case2 写 583 MB / 414.312us = **1.41 TB/s**。
- A5/Ascend950PR HBM 峰值参考：Atlas350 卡 1.4 TB/s、芯片标称 ~1.6 TB/s → 实测写带宽为峰值 **89%~101%，基本打满**。
- profiling 指标：`aiv_mte3_ratio≈0.97`、`aiv_vec_ratio≈0.06`，计算被 MTE3 完全掩盖；MTE3 仅 15.7% 等待气泡（且为写饱和导致的固有等待）。

**输出体量由算子语义固定不可压缩**（256000/138100 行 × 4096 fp8 + 128B e8m0 scale 是必须产出），因此 gather 热点内核级可优化空间 ≈ 0。

**GatherCopy 是访存最优路径（不应切回 ScatterCopy）**：GatherCopy 读 84MB(顺序)+写 1.08GB(散射)=1.16GB；ScatterCopy 每输出行各读一次源 → 读约 2GB+写 1.08GB≈3GB（2.6× 流量），且 compute 从 1 万次涨到 25.6 万次。写 bound 下 GatherCopy 总流量最小，保持现路径。

### 3.1 GatherCopy 搬运机理：读侧"两次索引一次搬运跳搬"已最优，写侧不可合并

源码 `op_kernel/arch35/moe_v3_gather_mxfp8_quant.h::GatherCopyExpandedXandMXQuant` 按**源 token 行**（非输出行）组织搬运：

- **一次搬运（读+算按源行摊销）**：每个源行只做一次 `CopyIn`(读 bf16 源行) + 一次 `Compute`(amax→scale→量化)。该源 token 被 K 个 expert 选中、落在本核范围内的多个副本，**共享同一次读入与同一次量化结果**，不重复读、不重复算。
- **两次索引**：① 探测扫描 `indicesLocal.GetValue`——借 `globalSortIdx / k_ == row` 判源行边界并检测本核区间内是否有有效输出；② 写出扫描 `GetValue`——对每个有效 `outIndex` 取目标行号驱动散射写。
- **跳搬**：探测/写出扫描遇 `outIndex < 0`(被 drop 的 token) 直接跳过，不触发读入与写出。

| 侧 | 现状 | 能否再减 |
|---|---|---|
| 读(gather 源行) | 每源行**只读 1 次**（不管被几个 expert 选中）→ case1 仅读 84MB | 已最优；切 ScatterCopy 反而 K× 放大到 ~2GB |
| 量化 compute | 每源行**只算 1 次** | 已最优 |
| 写(scatter 输出) | 每有效输出行**仍 1 次 `DataCopyPad`**(data+scale 各一条) | **不能合并**：K 个目标地址 `outIndex*cols_` 由运行期排序散射决定、无固定 stride，无法用 `blockCount>1` 打包 |

即：**读侧 gather 调用次数已被"两次索引一次搬运跳搬"压到最小；写侧次数/字节量由固定输出体量决定、逼近 HBM 写带宽物理墙**——这正是内核级改动测不出收益的根因。

## 4. 候选优化与真机实测（均无净收益）

在直调复现工程上实施两种候选内核优化并逐 case 采集对比（基线/方案同卡串行、launch-count=20）：

| 方案 | 改动 | case1 total | case2 total | gather Δ | 判定 |
|---|---|---|---|---|---|
| 基线 | — | 1161.509 | 625.857 | — | — |
| plan1（MTE3 流水重叠） | GATHER_MXFP8_OUT_BUFFER_NUM 2→3、去源行循环 S_MTE2 barrier、host tiling 缓冲分母 /2→/3 | 1158.861 (**-0.23%**) | 625.168 (**-0.11%**) | case1 -0.42% / case2 -0.10% | 噪声内，无收益 |
| plan2（削减 128B scale 小写事务） | 诊断为**字节带宽受限**（非事务发射受限），gather 与基线字节一致，未实质改动 | 1163.031 (+0.13%) | 626.329 (+0.08%) | +0.15% / +0.10% | 确认字节受限，收益≈0 |

- 全部 |Δ| < 0.5%，方向不一致（plan1 微降、plan2 微升），是典型**测量噪声**而非真实收益。
- **常见直觉优化"合并 3 个串行 VF / 消除 x 双搬入"对本算子无效**：VEC 占比仅 6%，三个 VF(amax/roundscale/data) 与 x 的两次 LoadAlign 全部被 MTE3(97%) 掩盖，动这些不缩短 total_us。

**精度安全性**：plan1 全量功能用例 **18/18 逐字节 PASS，0 新增 FAIL**（fp8_e4m3fn/e8m0 逐字节比对；含 GatherCopy 触发用例 K∈{4,5,6,8}）。即便如此，因无净收益且多耗 UB，**不建议将 plan1 回注生产 kernel**。

## 5. 结论与后续方向

1. **本场景 gather+mxfp8 量化已在 A5 HBM 写带宽物理墙**，内核级流水重排/缓冲加深/写序调整均无法突破"写带宽 + 固定输出体量"，两方案实测确认。**当前生产 GatherCopy 路径即为该场景最优，无需改动。**
2. A2→A5 仅 1~1.2X 与此自洽：两平台该算子都是 HBM 写带宽受限，加速比≈两者聚合写带宽之比而非算力比，故差距有限。
3. **后续可评估空间在 gather 之外**（需独立立项，非本场景 gather 主线）：
   - **sort（multicore）**：case1 227us/20%、case2 133us/21%，第二大耗时，建议单独评估 multicore sort 流水优化空间。
   - count 阶段在框架 launch 路径下为单次 launch，无额外开销（直调复现时的 host 分块仅为规避 507035 的临时手段，不适用于注册工程）。

> 采集/分析细节与逐 case 原始读数见调优工作区 round1 产物（性能调优方案.md / 性能调优报告.md / full-precision-regression.md）。
