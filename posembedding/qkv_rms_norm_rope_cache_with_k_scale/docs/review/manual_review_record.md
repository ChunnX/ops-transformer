# QKV M-RoPE MX 人工 Review 记录

本文件是本轮人工 Review 问题、处理方案和最终状态的唯一记录。代码、测试结果和性能原始数据仍是各自的事实源；本文件只记录评审判断并引用它们，避免复制出第二套实现或测试事实。

状态取值：

- `OPEN`：问题已提出，尚未形成一致结论。
- `AGREED`：处理方案已达成一致，尚未修改代码。
- `FIXED`：代码或文档已经修改，尚未完成验证。
- `VERIFIED`：修改及约定的验证均已完成。
- `DEFERRED`：有明确原因暂缓处理。
- `REJECTED`：评审后决定不修改，并记录理由。

## REV-001 Q 执行模式命名及 tail 路由

- 提出日期：2026-08-24
- 提出问题：`QKV_K_SCALE_EXEC_MODE_VECTOR` 表示什么？为什么引入 `QKV_K_SCALE_EXEC_MODE_Q_TAIL_PAIR` 和 `QKV_K_SCALE_EXEC_MODE_Q_TAIL_GLOBAL`？能否消除？
- 当前状态：`FIXED`

### 修改前的代码事实

1. `EXEC_MODE` 是 tiling key 的最后一个编译期模板维度。Kernel 用 `if constexpr` 选择 VF，运行时不在热循环中判断该值。
2. 三个当前可选值都运行在 AIV SIMD Vector 上：
   - `VECTOR`：处理没有 8-head 尾部的基础 RowBatch16 路径，覆盖 `Nq=16/32/48/64`。
   - `Q_TAIL_PAIR`：先处理完整的 16-head prefix，再把相邻 token 的 8-head tail 合成 RowBatch16，覆盖 `Nq=8/40`。
   - `Q_TAIL_GLOBAL`：把 tile 内所有完整 prefix 和跨 token tail wave 放进同一次 `Stage -> Scale -> Quant` 流水，覆盖 `Nq=24/56`。
3. `Q_TAIL_PAIR` 与 `Q_TAIL_GLOBAL` 不只选择不同函数，还选择不同的 UB 临时区生命周期：
   - pair 路由在 prefix VF 结束后复用临时区，保留较大的 token tile；
   - global 路由为 tile 内所有 prefix 和 tail wave 分配独立临时槽，用更多 UB 换取更大的流水窗口。
4. `SIMT` 和 `SIMT_VK` 虽仍出现在 Host 枚举和 Kernel 宏中，但 tiling selector、controller 和当前 Host 选路都不接收它们。`mropeMxExecMode_` 只被初始化为 `VECTOR`，当前代码没有把它改成其他值。

### 性能事实

跨 token tail8 的历史搜索已经比较过以下结构：

- v5：统一 tail-tile 流水；`Nq=8/24/40/56` 分别为 `12.105/17.813/22.581/27.388 us`，已淘汰。
- v7：把两种 tail 调度合入一个 tiling key 和 kernel 实例，在 AIV controller 中运行时选择两个独立 `VF_CALL`；small-tail 因合并后的 kernel 代码布局退化，已淘汰。
- v8：用两个 tiling key 分别编译 v4 small-tail 与 v6 global-tile；四个 Nq 相对各自最强对照的差值都不超过 `0.235 us`，并且有正有负。它保留了两条已知最强路线，但没有带来新的稳定大幅收益。

原始性能事实源：`/tmp/qkv_mrope_qtail8_cross_token_rewrite_v8_20260821.json`。该文件属于实验身份，不随仓库交付；合入前若需要以这些数据作为保留双路由的正式依据，应把匹配 A/B 的结论整理进设计文档并绑定 package/result SHA256。

### 追问：为什么 `Nq=24/56` 不使用 `Q_TAIL_PAIR`

`Nq=24/56` 可以使用 `Q_TAIL_PAIR`，两条路由的数值语义相同。Host 选择 `Q_TAIL_GLOBAL` 是性能决策，不是正确性或 shape 限制。

`Q_TAIL_PAIR` 把完整 prefix 和 8-head tail 分开处理：prefix VF 对每个 token 完成一次 `Stage -> Scale -> Quant`，prefix 结束后，tail VF 才复用同一临时区处理跨 token wave。`Q_TAIL_GLOBAL` 则为当前 tile 的每个 prefix 批和 tail wave 分配独立临时槽，先完成全部 Stage，再完成全部 Scale 和 Quant。因此它把 prefix 与 tail 的同步窗口一起合并了。

以下 barrier 数是根据 VF 源码和对应 `tokenTile` 推导的完整 tile 静态计数，不是 profiler 动态计数：

| `Nq` | 完整 prefix 批/token | pair：`tokenTile / scratch / barrier` | global：`tokenTile / scratch / barrier` | pair v4 | global v6 | 当前选择 |
|---:|---:|---:|---:|---:|---:|---|
| 8 | 0 | `24 / 17,664B / 18` | `17 / 79,488B / 3` | `11.418 us` | `12.186 us` | pair |
| 24 | 1 | `10 / 17,664B / 39` | `6 / 79,488B / 3` | `17.694 us` | `17.401 us` | global |
| 40 | 2 | `6 / 26,496B / 21` | `4 / 88,320B / 3` | `21.760 us` | `22.355 us` | pair |
| 56 | 3 | `4 / 35,328B / 15` | `3 / 97,152B / 3` | `27.211 us` | `26.130 us` | global |

测试条件为 `T=768`、seed 42、TTK ACLNN task-prof、内部 warmup、run 50、性能测试关闭 Golden。v4 与 v6 来自不同 package 身份且没有交错执行，因此这些数据用于解释历史选路，不应表述为当前 package 的严格单变量 A/B。后续 v8 分 key 实现相对各自 fresh 最强对照的差值不超过 `0.235 us`，说明最终分 key 没有破坏各自路线。

对 `Nq=24`，global 虽把 `tokenTile` 从 10 降到 6，却把 prefix 每 token 的三段同步和 tail 分组同步合并为每 tile 三次；实测快 `0.293 us`。对 `Nq=56`，global 把 `tokenTile` 从 4 降到 3，但能让三个 prefix 批和 tail wave 共享一个更大的依赖窗口；实测快 `1.081 us`。

barrier 少并不必然更快。`Nq=8/40` 的 global 路由也只有三个静态 barrier，但更大的临时区、更小的 `tokenTile`、更长的值生命周期和不同的编译布局抵消了收益，实测反而更慢。因此 `fullBatchCount == 1 || fullBatchCount == 3` 是当前硬件和实现上的实测路由表，不能解释成通用算法规律。

评审处理要求：设计文档应明确写成“pair 语义可用，global 为实测性能选择”，并保存可复核的 package、输入、结果 SHA256。Host 中的 `1/3` 条件应通过有名称的 Q schedule 选择函数表达，避免留下没有出处的 magic number。

### 设想：pure Vector 只保留一个 tiling key

该方案在结构上可行。一个 tiling key 只决定生成一个 kernel 实例，不要求所有 Q 算法写进同一个 `__simd_vf__` 函数。Kernel 可以根据运行期常量 `qHeadNum` 在 AIV controller 中选择以下三个独立 VF：

- `Nq%16==0`：普通 RowBatch16；
- `Nq%16==8` 且完整 prefix 批为 0 或 2：tail tile wave；
- `Nq%16==8` 且完整 prefix 批为 1 或 3：all-Q tile wave。

Host 仍须按同一规则搜索 `tokenTile` 和计算 UB 布局。不能为了单 key 给所有 shape 分配 global 路由的最坏 UB，否则会无谓降低 `Nq=8/40` 的 `tokenTile`。当前 shape 已足以推导 schedule，不应新增 tiling data 字段。

v7 是该设想的部分验证：它仍保留无 tail 的 `VECTOR` key，但把 pair/global 两个 tail key 合成一个 key。运行时判断位于 controller，两个 `__simd_vf__` 实现仍然分开。v7 的 `Nq=8/24/40/56` 和 `T=1/2/3` 精度用例全部通过，说明运行时选路本身可正确实现。

历史性能如下；“最强分 key”使用 v4 pair 或 v6 global，各结果来自同配置但不同 package、非交错执行：

| `Nq` | 最强分 key | v7 单 tail key | 差值 |
|---:|---:|---:|---:|
| 8 | `11.418 us` | `12.060 us` | `+0.642 us / +5.623%` |
| 24 | `17.401 us` | `17.243 us` | `-0.158 us / -0.908%` |
| 40 | `21.760 us` | `21.767 us` | `+0.007 us / +0.032%` |
| 56 | `26.130 us` | `26.103 us` | `-0.027 us / -0.103%` |

4a3 对象中，v7 合并后的 tail kernel symbol 为 `9,440B`；v8 拆分后的 pair/global 分别为 `8,940B/8,572B`，普通 RowBatch16 为 `8,408B`。因此直接成本不只是一次标量判断，还包括更大的 kernel 代码、不同的指令地址和潜在 I-cache/编译调度变化。v7 在 `Nq=8` 的 `0.642 us` 退化证明这种影响可以超过分支本身。

完整“只留一个 Vector key”还会把普通 RowBatch16 也放入同一 kernel；`Nq=16/32/48/64` 的性能和最终 symbol 大小没有测试，状态为 `NOT_EVALUATED`。不能用 v7 的 tail 数据推断这些 shape 无损。

如果继续验证，候选必须满足：

1. 运行时判断放在 AIV controller，不能放进 token/head 的 VF 内层循环；三个 VF 保持独立。
2. Host 和 device 共用一个由 `qHeadNum` 推导 schedule/UB 的 helper；不新增 schedule 字段，不按最坏情况统一分配 UB。
3. 对当前 package 做 `Nq=8,16,...,64` 的固定 seed TTK ACLNN 匹配 A/B，同时记录 kernel symbol 大小、最终 `tokenTile` 和五输出精度。
4. 若以性能不回退为目标，现有证据不支持直接采用单 key；至少要先消除 `Nq=8` 已观察到的约 `5.6%` 退化，并证明无 tail 的四个 shape 不退化。

以上是测试前的阶段判断。随后已按上述四项门禁实现并测试真正的“三合一”候选，结论见下一节；该候选覆盖普通 RowBatch16、tail tile wave 和 all-Q tile wave，不再沿用 v7 只合并两条 tail 路由的局限。

### 真正“三合一”实测

实验候选只生成公开 key `302059776`。Kernel 在 `ProcessQRowBatch16` 入口根据 `qHeadNum` 选择三条独立 VF helper：

- 无 8-head tail：`RowBatch16`；
- 有 tail 且完整 16-head batch 数为 0 或 2：`TailTileWave`；
- 有 tail 且完整 16-head batch 数为 1 或 3：`AllQTileWave`。

运行时判断只执行一次，不进入 token/head 的 VF 内层循环。Host 仍根据 `qHeadNum` 分别计算 `tokenTile` 和 UB layout，没有新增 tiling data，也没有让所有 shape 使用最坏 UB 布局。

测试使用 Ascend950PR_9589、CANN 9.1.0、TTK ACLNN task-prof、seed 42、内部 warmup、run 50、单进程且不复用进程/HBM。性能阶段在精度门通过后关闭 Golden；每个 case 的 `QkvRmsNormRopeCacheWithKScale_AiCore` 都有 49 次调用。

| `Nq/Nk/Nv` | baseline key | 三合一 key | baseline 性能 | 三合一性能 | 差值 |
|---:|---:|---:|---:|---:|---:|
| `8/1/1` | `4597027072` | `302059776` | `11.859 us` | `11.413 us` | `-0.446 us / -3.761%` |
| `16/2/2` | `302059776` | `302059776` | `13.800 us` | `13.874 us` | `+0.074 us / +0.536%` |
| `24/3/3` | `8891994368` | `302059776` | `17.259 us` | `17.421 us` | `+0.162 us / +0.939%` |
| `32/4/4` | `302059776` | `302059776` | `18.712 us` | `19.066 us` | `+0.354 us / +1.892%` |
| `40/5/5` | `4597027072` | `302059776` | `21.680 us` | `21.921 us` | `+0.241 us / +1.112%` |
| `48/6/6` | `302059776` | `302059776` | `23.807 us` | `24.283 us` | `+0.476 us / +1.999%` |
| `56/7/7` | `8891994368` | `302059776` | `26.210 us` | `26.038 us` | `-0.172 us / -0.656%` |
| `64/8/8` | `302059776` | `302059776` | `26.393 us` | `26.573 us` | `+0.180 us / +0.682%` |

八个 shape 的非加权平均差值为 `+0.109 us / +0.343%`；最大回退是 `Nq=48` 的 `+0.476 us / +1.999%`。评审确认该性能幅度可以接受。

两版 opkernel 和 package 构建都通过。精度阶段两版均为 `11/11 PASS`，每条用例五个输出均为 `100%`；覆盖上述八个 `T=768` shape，以及 `T=1` smoke、`T=2` 四个 D32 scale 顺序和 `T=3` position `1023/1024/1025`。baseline 与 candidate 的精度结果 CSV 字节相同。

baseline 的 4a3 对象包含三个 M-RoPE MX symbol，大小分别为 `8,408B/8,940B/8,572B`；三合一对象只含一个 `9,832B` symbol。对象总大小从 `112,912B` 降为 `86,488B`，编译日志没有 spill 诊断。

实验中还暴露了三项测试资产问题：Torch 的 `float8_e8m0fnu` 名称没有转换为 ACL 的 `float8_e8m0`；旧性能 CSV 仍使用 `[2,2]` scale shape；旧 Golden 仍按五维 K scale cache 写入。实验只在隔离 runner/assets 中修正这些问题，使其符合当前公开的 flat-four scale 合同；这些修正不是算子数值模拟，也没有写入本次待评审实现。

完整实验事实：`/tmp/qkv_mrope_vector_singlekey3_20260824_001/single_vector_key3_result.json`，SHA256 `092d8e0227232b528a2a6b34f515d0a1b996cbc8d84b357fc7d943caf7c7bede`。原始结果：baseline performance SHA256 `15a7a55e5a00eb9afea98bba911dc820997a4fdd3e06f627a3c4a0e56df009df`，candidate performance SHA256 `d309b4c4ff74bb3736cfa61138552d9cace199b95ab6994c3c02bd9248bef803`。

### 评审判断

`VECTOR` 这个名称不准确。它容易让人理解为“Vector 与其他执行引擎二选一”，但三个当前路由都是 SIMD Vector。这里真正变化的是 Q 的组批范围和临时区生命周期，不是执行引擎。

三条路由在数学语义上可以共用一个公开 tiling key，但不能把三种实现强行改成一个 VF 数据流。真正三合一实验说明：保留三个独立 VF helper、只在 Q 入口做一次运行时选择，并继续按 shape 分配 UB，可以消除两个 tail tiling key；Nq sweep 的最大回退为 `1.999%`，在本次评审可接受范围内。历史 v7 的 `Nq=8` 回退不是运行时分支的必然成本，而是当时不完整合并后的代码布局结果。

### 建议处理方案

1. 公开 tiling key 只保留 `QKV_K_SCALE_EXEC_MODE_VECTOR` 的数值 `302059776`；删除 selector/controller 对 `Q_TAIL_PAIR` 和 `Q_TAIL_GLOBAL` 的选择。
2. `ROW_BATCH16`、`TAIL_TILE_WAVE`、`ALL_Q_TILE_WAVE` 只作为 Kernel 内部 Q schedule/helper 名称，不再成为 tiling key 维度。
3. 在 Q 入口根据 `qHeadNum` 一次性选择 helper，不在 token/head 的 VF 内层循环判断。
4. Host 继续按相同 shape 规则计算三种 schedule 所需的 `tokenTile` 和 UB layout；不新增 schedule tiling 字段，不按最坏布局统一分配 UB。
5. 删除始终为 `VECTOR` 的 Host 成员 `mropeMxExecMode_`，并清理不再可选的 `SIMT`、`SIMT_VK`、两个 tail key 名称。
6. 把隔离候选移植到当前待评审目录后，先核对 M-RoPE MX 生成物只含 key `302059776`，再跑完整回归；完整回归通过后将本问题从 `FIXED` 更新为 `VERIFIED`。

### 完成条件

- 三合一实现合入当前待评审目录，生成物只保留公开 key `302059776`。
- Host、controller、Kernel、SRS/STC 和相关 UT 对“一个公开 key、三个内部 Q schedule”的表述一致。
- 完整回归通过；若合入后的 package identity 与本实验不同，重新执行上述固定 seed Nq sweep，最大回退不超过本次已接受的 `1.999%`。

### 最终处理和状态

- 最终处理：评审接受“一个公开 M-RoPE MX tiling key + Kernel 内三个独立 Q schedule helper”的方案。三合一源码已合入当前待评审目录：Host 用七个有选择意义的字段生成 key `302059776`，tiling selector 只生成这一条 M-RoPE MX 记录，Q 入口根据 `qHeadNum` 选择三个独立 helper；相关 Host UT 和 SRS/STC 已同步。原先恒为 `VECTOR(0)` 的 `EXEC_MODE` 已按 `REV-002` 删除，不再作为 key 维度。
- 构建证据：最终非增量 `opkernel` PASS，日志 `/tmp/ops_build_3p5j4yqu.log`；最终非增量 package PASS，日志 `/tmp/ops_build_irx2c43y.log`；Host UT `134/134 PASS`，日志 `/tmp/ops_build_5eluvsji.log`。八个 `Nq=8..64` Host case 全部断言 key `302059776`。当前 4a3 对象 SHA256 为 `2fdb56b64d7c01134504bb85dbfa32a1f3884342bf4568e32311c3b0fee18a80`，大小 `86,488B`，与已完成 11 条精度和 8 个 Nq sweep 的隔离候选对象逐字节相同；对象中 M-RoPE MX 仅有 `302059776` 一个 symbol，大小 `9,832B`；三个 dtype 模板 JSON 都只引用该 M-RoPE MX key。当前 package SHA256 为 `f455177cbd545e591c7909f5ef8eb1838b019fda37d69e8b4f3cac4af5697cd8`。一次增量 `opkernel` 重试复用了 Host-UT 配置的 build 目录，在编译前因缺少 `ops_transformer_kernel` target 退出，日志 `/tmp/ops_build_ifcz1xce.log`；随后重新配置并执行非增量构建通过，因此该次失败不属于源码编译失败，也不计入验收结果。
- 最终状态：`FIXED`。源码、测试期望和文档已修改，且构建与 kernel 对象身份已闭合；当前待评审 package 的完整分层回归尚未结束，因此不能标记为 `VERIFIED`。

## REV-002 删除单值 `EXEC_MODE` 模板维度

- 提出日期：2026-08-24
- 提出问题：`qkv_rms_norm_rope_cache_with_k_scale_tiling_key.h` 中 `EXEC_MODE` 只有 `VECTOR` 一个选项，是否应删除？
- 当前状态：`VERIFIED`

### 代码事实和评审判断

`EXEC_MODE` 已经失去选择能力。删除两个 tail key 后，Host 始终传 `VECTOR(0)`；tiling selector
只声明这一个值；kernel 入口、总 Controller、M-RoPE MX Controller 和 Vector 类只透传该模板参数，
各层唯一用途是 `static_assert(EXEC_MODE == VECTOR)`。Q 的三种组批方式由 Kernel 在入口根据
`qHeadNum` 选择，与该模板参数无关。

评审接受删除。保留单值模板维度会让读者误以为仍存在多个编译期执行后端，也要求 Host、selector
和 Kernel 维护无效参数。删除它不会合并三个 VF，也不会把 schedule 判断移入热循环。

### 处理方案

1. 从 tiling 参数声明和五条 selector 记录中删除 `EXEC_MODE`，同时删除 `QKV_K_SCALE_EXEC_MODE_VECTOR`。
2. Host 计算 key 时只编码七个有选择意义的字段。
3. kernel 入口、Controller 和 M-RoPE MX Vector 类删除该模板参数；M-RoPE MX 改为七参数显式特化。
4. SRS 的可读 key 改为 `(D128,TND,TND,MROPE,E4M3FN,MX,MX)`；早期多执行模式设计文档明确标为历史快照。
5. 通过 Host UT 和最终 package metadata 验证整数 key 仍为 `302059776`；通过对象 SHA 和 symbol
   对比判断是否需要重新执行精度与性能测试。

### 最终处理和状态

- 最终处理：上述源码和当前 SRS 已修改。Host、tiling selector、kernel 入口、Controller 和 Vector 类均不再声明或透传 `EXEC_MODE`；Q 内部仍保留 `ROW_BATCH16`、`TAIL_TILE_WAVE`、`ALL_Q_TILE_WAVE` 三个独立 schedule。
- 静态验证：目标算子的 Host、Kernel、UT 和当前 SRS 中已无代码级 `EXEC_MODE`/`QKV_K_SCALE_EXEC_MODE_VECTOR` 引用；SRS 只保留一处说明旧字段已经删除。`git diff --check -- posembedding/qkv_rms_norm_rope_cache_with_k_scale` 通过。
- Host UT：`134/134 PASS`，日志 `/tmp/ops_build_pkhg9ixn.log`。八个 `Nq=8..64` 路由用例仍断言 key `302059776`。
- 构建验证：Ascend950 `opkernel` PASS，日志 `/tmp/ops_build_48q00gzl.log`；package PASS，日志 `/tmp/ops_build_j74c4m_4.log`。当前 package 文件 SHA256 为 `6d276f02f4ace0b9438c54f587bd361e4a0432442d91783eb6c3665d020fdeac`。
- 产物身份：三个 dtype 模板 JSON 的 key 集合保持不变，M-RoPE MX 仍只有 `302059776`。4a3 设备对象大小为 `86,488B`，SHA256 为 `2fdb56b64d7c01134504bb85dbfa32a1f3884342bf4568e32311c3b0fee18a80`，与删除该字段前、且已经完成 11 条精度门禁和八个 Nq 性能 sweep 的三合一对象逐字节相同。因此本次不重复执行 NPU 精度和性能测试；依据是设备对象字节身份相同，而不是仅凭源码推断。
- 最终状态：`VERIFIED`。单值模板维度已端到端删除，整数 key、kernel symbol 和设备代码均未改变。

## REV-003 按数据类型隔离 tiling key

- 提出日期：2026-08-24
- 提出问题：`qkv_rms_norm_rope_cache_with_k_scale_tiling_key.h` 没有用数据类型宏隔离 selector，导致每个静态 dtype 对象都编译无效的 dtype 与 tiling key 组合。
- 当前状态：`VERIFIED`

### 修改前的代码事实

OpDef 只声明三组相关的数据类型契约，但修改前每个设备对象都生成相同的五个 tiling key：

| 静态 dtype 对象 | 有效场景 | 修改前实际 key | 应保留 key |
|---|---|---|---|
| K cache E4M3、K scale FP32、Q out E4M3、Q scale FP32 | RoPE | `0, 256, 4352, 17895680, 302059776` | `0, 256, 4352` |
| K cache INT8、K scale FP32、Q out BF16、Q scale FP32 | M-RoPE | `0, 256, 4352, 17895680, 302059776` | `17895680` |
| K cache E4M3、K scale E8M0、Q out E4M3、Q scale E8M0 | M-RoPE MX | `0, 256, 4352, 17895680, 302059776` | `302059776` |

修改前生成物已冻结到 `/tmp/qkv_mrope_dtype_guard_review_20260824_before/`。三个 JSON 的 key 集合均由文件内容直接确认，不是按源码推断。

### 评审判断和处理方案

评审意见成立。tiling key 中的 `K_CACHE_DTYPE` 只参与整数 key 编码，不能替代静态输入/输出 dtype 对象之间的编译隔离。继续让所有 dtype 对象编译五个 key 会生成 Host 永远不会合法选择的 kernel，并扩大对象和 package。

处理采用编译器已经提供的 `ORIG_DTYPE_*` 宏，不新增 tiling key 数据类型字段：

1. Host 侧没有定义 `__CCE_AICORE__`，继续向 `GET_TPL_TILING_KEY` 暴露全部五个合法 selector，保持 Host 的 key 合法性校验完整。
2. Device 编译侧同时检查 `ORIG_DTYPE_K_CACHE`、`ORIG_DTYPE_K_SCALE_CACHE`、`ORIG_DTYPE_Q_OUT` 和 `ORIG_DTYPE_Q_SCALE`，只展开与当前静态 dtype 对象完全匹配的 selector。
3. 三组场景继续复用已有七字段 key 编码；不增加 dtype 模板维度，不改变 `0/256/4352/17895680/302059776` 的数值。
4. 如果静态 dtype 元组不属于 OpDef 声明的三组契约，Device selector 不生成 kernel；不把未知元组错误地归入某个默认场景。

### 完成条件

- Host UT 通过，证明 Host 仍能生成并校验五个现有整数 key。
- Ascend950 `opkernel` 和 package 构建通过。
- 三个生成 JSON 的 key 集合分别严格等于 `{0,256,4352}`、`{17895680}`、`{302059776}`。
- 五个保留 kernel symbol 与修改前相同 key 的 symbol 逐字节一致；如果设备代码发生变化，再决定是否重跑相应 NPU 精度与性能测试。

### 最终处理和状态

- 最终处理：已在 tiling key selector 外增加 `ORIG_DTYPE_*` 场景门禁；Host 保留全部 selector，Device 按四个相关输入/输出 dtype 选择一个场景组。
- Host 与构建验证：Host UT `134/134 PASS`，日志 `/tmp/ops_build_8j7lgdgd.log`；Ascend950 `opkernel` PASS，日志 `/tmp/ops_build_jri0otyd.log`；package PASS，日志 `/tmp/ops_build_4u3wpyms.log`。最终 package SHA256 为 `58258ced70dd5e504aff7267c9a1e4e5951432bbd9902c9429fa7583d69b7614`。
- 生成物验证：RoPE、M-RoPE、M-RoPE MX 三个静态 dtype 对象的 key 集合分别为 `{0,256,4352}`、`{17895680}`、`{302059776}`。三个设备对象大小分别从 `86,488B` 降至 `49,952B`、`19,248B`、`27,568B`，总大小从 `259,464B` 降至 `96,768B`，减少 `162,696B / 62.705%`。
- 设备代码身份：修改前后同 key 的九个 AIC/AIV kernel symbol 均逐字节一致，包括 key `0/256/4352/17895680/302059776` 的全部保留 symbol。因此本项不重复运行 NPU 精度和性能测试；依据是保留设备代码字节身份相同，而不是只凭源码推断。
- 最终状态：`VERIFIED`。无效 dtype 与 tiling key 组合已从 Device 生成物中删除，Host 合法 key 集合、整数 key 和所有保留 kernel 的设备代码均未改变。

## REV-004 删除未接入的 SIMT 实验实现和当前死代码

- 提出日期：2026-08-24
- 提出问题：删除不必要的 `qkv_rms_norm_rope_cache_with_k_scale_mrope_mx_simt.h` 和
  `qkv_rms_norm_rope_cache_with_k_scale_mrope_mx_simt_vk.h`；同时删除无用字段、函数、类型和头文件。
- 当前状态：`VERIFIED`

### 代码事实和评审判断

两个 SIMT 文件属于未接入生产通路的实验代码，并且原本都是待上库的未跟踪文件：

| 文件 | 行数 | SHA256 | 生产引用 |
|---|---:|---|---:|
| `qkv_rms_norm_rope_cache_with_k_scale_mrope_mx_simt.h` | 399 | `f218e69c215180308a84b9cf7039bb79353dc0123a64422bfad1971cc29c071f` | 0 |
| `qkv_rms_norm_rope_cache_with_k_scale_mrope_mx_simt_vk.h` | 199 | `e8bb4223c107fe42839334373aa79c1234f28b51cc4323cefd725b0fe6bbd8e8` | 0 |

生产 selector、Host、controller 和 kernel 入口均未包含或调用这两个文件。SIMT 文件自己的
`simt_api` include、结构体、函数和 `V_S` 同步只形成一个内部依赖闭环；删除整个闭环不会改变
当前 SIMD 的 `MTE2_V`、`V_MTE3`、`MTE3_S` READY/FREE 协议。因此评审接受直接删除两个文件，
不保留空壳、兼容别名或失效 selector。

沿当前 SIMD 调用图继续检查后，发现 `mrope_mx_vf.h` 还保留了一组已经没有调用方的旧 helper，
以及一个只实例化 `<false>`、从未实例化 `<true>` 的模板分支。这些代码不属于当前三条 Q schedule，
也不参与 K batching：

- 删除 `MxQuantCublasScaleRowBatch16`、`MxQuantCublasScaleRowBatch16ToPadded`、
  `FinishMropeMxRowBatch16` 和 `MxQuantCublasDataRowBatch8`；
- 删除 `QRmsNormMropeMxD128TailPair16VfImpl` 和 `QRmsNormMropeMxD128TailSingle8VfImpl`；
- 将只使用 `<false>` 的 `QRmsNormMropeMxD128RowBatch16PrefixVfImpl` 改为普通函数，删除整段
  `HAS_TAIL=true` 分支和该分支独占的局部变量。

当前 VF 文件由 1,833 行降到 1,450 行，共删除 383 行；加上两个实验文件的 598 行，本次从待上库
代码中共清理 981 行。清理后逐函数引用检查未发现定义后无调用的 helper。`MropeMxTileDesc`、当前
layout 字段、常量和 include 均有生产调用方或明确的地址/同步职责，因此没有为了扩大清理范围继续删除。
当前 `ROW_BATCH16`、`TAIL_TILE_WAVE`、`ALL_Q_TILE_WAVE` 三条 Q schedule 和 K 的 batch 路由均保留。

历史设计文档继续保存已经分析过的 SIMT/SIMT-VK 方案，避免丢失实验结论；文档顶部已明确标注它们
是历史方案、源码和路由已经删除，不能作为当前实现事实源。当前事实以 SRS 和生产源码为准。

### 处理和验证

1. 删除两个未跟踪 SIMT 实验头文件及其完整内部依赖闭环。
2. 按实际调用图删除上述死 helper 和未实例化模板分支；不修改当前 SIMD 数据流和同步协议。
3. 更新 `tests/st/generate_st_cases.py` 的源码门禁，使其检查当前三个 Q schedule、K batch 和
   deferred-RMS helper；CSV 中 `Q_TAIL_PAIR/Q_TAIL_GLOBAL` 仍作为覆盖分类名称保留，不再当作代码符号。
4. 生产源码和测试中已无 `mrope_mx_simt`、`SIMT_VK`、`simt_api`、`__simt_`、上述死 helper 或
   `HAS_TAIL` 引用；package staging 中也没有名称含 `simt` 的文件。
5. ST 资产检查通过：`generate_st_cases.py check` 确认 1,200 行，分层计数保持 Kernel 250、
   ACLNN L0 200、L1 650、L2 50、Torch 50；CSV 内容哈希未被本次源码门禁修正改变。
6. Host UT 为 `134/134 PASS`，日志 `/tmp/ops_build_v0392u78.log`。后续清理只触及 device helper、
   历史文档状态和 ST 源码门禁，没有修改 Host 生产代码。
7. 最终 Ascend950 `opkernel` PASS，日志 `/tmp/ops_build_4dtsa7nu.log`；最终 package PASS，日志
   `/tmp/ops_build_wggd7sx8.log`。实际 `build_out` package SHA256 为
   `0cc83b8e5423d1da742d5fa6709da7de514d36fde385e382f48334dbf192c075`。
8. 最终三个 dtype 对象的 key 集合仍分别为 `{302059776}`、`{0,256,4352}`、`{17895680}`。
   对象大小/SHA256 分别为：M-RoPE MX `27,568B / e55f200d7b0bb0051738f065b4baf2da8d538740ea3c3d0ff6deb0de66152197`，
   RoPE `49,952B / a4c6c13c6487361b9699d044f794819f801faa2ede79634ed871dd24be4bcb64`，
   M-RoPE `19,248B / d4b0285166eea7355ce5947f2acfc220419104b5467cbf1ecb6fa3814db606b4`。
   三个对象与清理前逐字节相同，证明删除内容没有进入设备代码，因此本项不重复运行 NPU 精度和性能测试。
9. `git diff --check -- posembedding/qkv_rms_norm_rope_cache_with_k_scale` 通过。

### 最终处理和状态

- 最终处理：两个未接入的 SIMT 实验文件已从待上库文件集合中删除；当前 SIMD 调用图中的死 helper
  和未实例化模板分支也已删除。继续使用的类型、字段、include、Q schedule、K batching 和同步协议均保留。
- 最终状态：`VERIFIED`。源码引用、ST 资产、Host UT、Ascend950 构建、package 内容和设备对象字节身份均已闭合。

## REV-005 将 cache scatter stride 校验覆盖到 RoPE 和旧 M-RoPE

- 提出日期：2026-08-24
- 原问题：`CheckStrides` 不能只处理 M-RoPE MX；RoPE、M-RoPE 非 MX 场景也存在同样的 cache scatter。
- 当前状态：`VERIFIED`

### 修改前代码事实

1. ACLNN 入口通过 `CreateCacheViews` 保留 cache 的逻辑 shape、storage shape、storage offset 和 stride，
   并未在进入算子前把非连续 cache 稠密化。
2. legacy device 实现和 M-RoPE MX 实现都从 `slotMapping` 计算 block/token，再使用 tiling data 中的
   `kvCacheStrideBlock/Head/Token` 与 `kScaleCacheStrideBlock/Head/Token` 形成 GM scatter 地址。因此
   RoPE、旧 M-RoPE、M-RoPE MX 共享同一种 cache 地址契约；差别只是旧场景每 token 的 K scale
   尾维为 1 个 FP32，MX 为 4 个 E8M0。
3. 算子 README 和 ACLNN 文档已经把 K/V/KScale 的正 stride、K/V 前三维 stride 相同以及尾维连续
   约束写成三个场景的共同约束。
4. 修改前 Op API `CheckStrides` 对非 MX 场景直接返回；graph tiling 虽对三个场景检查了基础 stride，
   但 K/V/KScale 的 non-overlap 与 checked-span overflow 检查只在 MX 分支执行。这与设备寻址和公开
   文档不一致。

### 评审判断与解决方案

评审意见成立。不能把 `CheckStrides` 解释为 MX 专属保护。处理采用已有
`QkvRmsNormRopeCacheWithKScaleStrideUtils::IsNonOverlappingPositiveStrides` 作为唯一公共判定，
不新增第二套 stride 规则：

1. 三个场景统一要求 K/V 的 Block、Head、Token stride 为相同正值，Head/Token stride 至少覆盖
   一个完整 D 行，D 维 stride 为 1，并满足按 Block/Head/Token/D 语义顺序的 non-overlap 与
   checked-span 无溢出。
2. 三个场景统一要求 KScale 四维 stride 为正、尾维 stride 为 1、token stride 覆盖完整 scale 尾维，
   并满足按 Block/Head/Token/Scale 语义顺序的 non-overlap 与 checked-span 无溢出。尾维元素数直接
   取逻辑 shape，因此自然覆盖旧场景的 1 和 MX 的 4，不派生新场景常量。
3. M-RoPE MX 的 `qOut/qScale` 连续性仍保留为 MX 专属约束，不错误扩展到 RoPE 或旧 M-RoPE。
4. graph tiling 使用同一个 stride helper，将原先 MX 分支内的 K/V/KScale non-overlap 检查提升为
   三个场景共同检查，保持 ACLNN 与 graph 两条入口一致。

### 实际修改

- `op_host/op_api/qkv_rms_norm_rope_cache_with_k_scale_checker.h`：删除非 MX 提前返回，将 cache stride
  和 non-overlap 校验推广到三个场景；保留 MX 输出连续性检查的场景门禁。
- `op_host/op_tiling/arch35/qkv_rms_norm_rope_cache_with_k_scale_base_tiling.cpp`：将 K/V/KScale 的
  non-overlap 与 span-overflow 检查从 MX 分支提升为共同检查。
- `tests/ut/op_api/test_aclnn_qkv_rms_norm_rope_cache_with_k_scale.cpp`：新增 RoPE K/V block overlap
  拒绝、旧 M-RoPE KScale block overlap 拒绝，以及 RoPE padded non-overlapping view 接受用例。

### 验证证据

1. `git diff --check -- posembedding/qkv_rms_norm_rope_cache_with_k_scale` 通过。
2. Op API UT `140/140 PASS`；新增三个非 MX stride 用例全部通过。
3. Host UT `134/134 PASS`，覆盖现有 RoPE、旧 M-RoPE、M-RoPE MX tiling 与 mixed-stride 正向资产。
4. 最终 Ascend950 `opkernel` PASS，日志 `/tmp/ops_build_vlebhmgm.log`。
5. package 首次在 sandbox 内失败，首个硬错误是 ES wheel 构建依赖无法从受限网络取得
   `setuptools>=40.8.0`，退出码 2，日志 `/tmp/ops_build_g782stth.log`；这是构建环境网络门禁，
   不是算子编译错误。随后在 device-visible/非受限路径以同一精确命令重跑，package PASS，退出码 0，
   日志 `/tmp/ops_build_5us7dnp5.log`。实际 `build_out` package SHA256 为
   `f20209688262c2fca62a9557ec8df4e38f97de9ac11d97324e5fce3f61eff118`。
6. 最终三个静态 dtype 对象的 key 集合仍分别为 M-RoPE MX `{302059776}`、RoPE
   `{0,256,4352}`、旧 M-RoPE `{17895680}`，相关 kernel symbol 均存在。
7. 三个设备对象大小/SHA256 分别为 M-RoPE MX
   `27,568B / e55f200d7b0bb0051738f065b4baf2da8d538740ea3c3d0ff6deb0de66152197`、RoPE
   `49,952B / a4c6c13c6487361b9699d044f794819f801faa2ede79634ed871dd24be4bcb64`、旧 M-RoPE
   `19,248B / d4b0285166eea7355ce5947f2acfc220419104b5467cbf1ecb6fa3814db606b4`；与 REV-004
   基线逐字节相同。整包 hash 因 Host checker/tiling 库变化而变化，设备对象未变化。

### 最终处理和状态

- 最终处理：cache scatter stride 契约已统一覆盖 RoPE、旧 M-RoPE 和 M-RoPE MX；MX 输出连续性仍按
  场景隔离。
- 未执行项：未重复运行 NPU 精度和 TTK 性能测试。依据是三个设备对象与 REV-004 基线逐字节相同，
  本项只收紧 Host 对非法重叠 view 的拒绝范围。
- 最终状态：`VERIFIED`。两条 Host 入口、单测、最终 Ascend950 构建、package metadata、key、symbol
  和设备对象字节身份均已闭合。

## REV-006 调整 L0 接口中 Q/K 量化模式的参数顺序

- 提出日期：2026-08-24
- 原问题：`QkvRmsNormRopeCacheWithKScale` 的 Op API L0/L2 入参中，`kQuantMode` 应直接跟在
  `qQuantMode` 后，而不是放在 `qOutDtype` 后。
- 当前状态：`VERIFIED`

### 当前代码事实

1. 公开 ACLNN L2 头文件与实现原本已经是 `qQuantMode, kQuantMode` 相邻，L2 无需调整。
2. L0 头文件、实现和 L2 到 L0 的直接调用原本是 `qQuantMode, qOutDtype, kQuantMode`，与 Q/K 成对参数
   的公开接口顺序不一致。
3. graph OpDef 的属性 ABI 仍是 `q_quant_mode, q_out_dtype, k_quant_mode`，infer dtype 依赖既有属性索引；
   该 schema 不属于本条 L0/L2 C++ 参数顺序评审，不能随意重排。
4. 示例源码和 ACLNN 文档中的调用示例还漏传了 `kQuantMode`，会使示例与当前 L2 原型不一致。

### 评审判断与解决方案

评审意见成立，但只需要修正不一致的 L0 C++ 接口及调用闭包。L2 已符合意见；graph OpDef 属性顺序作为
既有 ABI 保持不变。同步补齐示例中的 K 量化模式，避免文档形成第二套错误接口事实。

### 实际修改

- `op_host/op_api/qkv_rms_norm_rope_cache_with_k_scale.h/.cpp`：L0 参数和 DFX 顺序改为
  `qQuantMode, kQuantMode, qOutDtype`；内部 `OP_ATTR` 仍按 graph schema 顺序传递。
- `op_host/op_api/aclnn_qkv_rms_norm_rope_cache_with_k_scale.cpp`：L2 到 L0 调用按新顺序传参。
- `examples/test_aclnn_qkv_rms_norm_rope_cache_with_k_scale.cpp`、
  `docs/aclnnQkvRmsNormRopeCacheWithKScale.md`：补齐并相邻传入 Q/K 量化模式。

### 验证证据

1. 静态检查确认 L0 声明、定义、直接调用和 DFX 顺序一致；L2 声明/定义仍保持 Q/K 相邻；graph
   `OP_ATTR` 顺序未改变。
2. Op API UT `143/143 PASS`；Host UT `136/136 PASS`。
3. Ascend950 `opkernel` PASS，日志 `/tmp/ops_build_4h91fcks.log`；最终 package PASS，日志
   `/tmp/ops_build_8y2192i0.log`。
4. `git diff --check -- posembedding/qkv_rms_norm_rope_cache_with_k_scale` 通过。

### 最终处理和状态

- 最终处理：L0/L2 对外可见参数均为 `qQuantMode, kQuantMode` 相邻；graph 属性 ABI 未被误改。
- 最终状态：`VERIFIED`。

## REV-007 按 Q/K 语义区分 MX 量化模式常量

- 提出日期：2026-08-24
- 原问题：`op_host/op_api/aclnn_qkv_rms_norm_rope_cache_with_k_scale.cpp` 中的
  `QUANT_MODE_MX` 没有按已有风格区分 Q/K。
- 当前状态：`VERIFIED`

### 当前代码事实

同一文件的 PerTokenPerHead/NoQuant 常量和解析函数已经按 `Q_`、`K_` 语义命名，只有字符串值同为
`"MxQuant"` 的 MX 常量被 Q/K 共用。它不会造成运行时数值错误，但破坏了本文件的语义归属和维护风格。

### 评审判断与解决方案

评审意见成立。复用同一公开字符串值，但分别定义 `Q_QUANT_MODE_MX`、`K_QUANT_MODE_MX`，让 Q/K
解析与反向字符串转换各自只引用对应常量；不新增额外解析路径或改变公开字符串。

### 实际修改

- `op_host/op_api/aclnn_qkv_rms_norm_rope_cache_with_k_scale.cpp`：拆分 MX 常量并更新
  `ParseQQuantMode`、`ParseKQuantMode`、`QQuantModeToString`、`KQuantModeToString`。

### 验证证据

1. 目标文件已无通用 `QUANT_MODE_MX` 引用，Q/K 解析和 ToString 各自闭合。
2. Op API UT `143/143 PASS`；Host UT `136/136 PASS`。
3. Ascend950 opkernel/package 均 PASS，日志分别为 `/tmp/ops_build_4h91fcks.log`、
   `/tmp/ops_build_8y2192i0.log`；`git diff --check` 通过。

### 最终处理和状态

- 最终处理：MX 字符串值不变，Q/K 常量归属已按既有风格分离。
- 最终状态：`VERIFIED`。

## REV-008 去除 checker 中不准确的 MROPE_LEGACY 命名

- 提出日期：2026-08-24
- 原问题：`qkv_rms_norm_rope_cache_with_k_scale_checker.h` 使用 `MROPE_LEGACY` 不合适，
  该场景不是 legacy。
- 当前状态：`VERIFIED`

### 当前代码事实

checker 的 `SceneType::MROPE_LEGACY` 实际表示公开支持的非 MX M-RoPE 场景，并非兼容性遗留分支；
同一枚举还与 `ROPE`、`MROPE_MX` 并列。名称会错误暗示该场景已废弃。

### 评审判断与解决方案

评审意见成立。将 checker 内场景枚举直接命名为 `MROPE`，并同步 checker 诊断文本及对应 Op API
测试名。没有借此扩大到生成资产中作为历史分类标签的其他 `legacy` 文本，避免越界重命名。

### 实际修改

- `op_host/op_api/qkv_rms_norm_rope_cache_with_k_scale_checker.h`：`MROPE_LEGACY` 改为 `MROPE`，
  相关判断和错误信息同步改为 M-RoPE/non-MX 表述。
- `tests/ut/op_api/test_aclnn_qkv_rms_norm_rope_cache_with_k_scale.cpp`：对应测试名机械同步，语义不变。

### 验证证据

1. checker 和对应测试中已无 `MROPE_LEGACY` 引用。
2. Op API UT `143/143 PASS`；Host UT `136/136 PASS`。
3. Ascend950 opkernel/package 均 PASS；`git diff --check` 通过。

### 最终处理和状态

- 最终处理：checker 的当前公开非 MX M-RoPE 场景不再被标为 legacy。
- 最终状态：`VERIFIED`。

## REV-009 将 262144 token 上限统一为所有场景的 T 约束

- 提出日期：2026-08-24
- 原问题：`MAX_MROPE_MX_TOKEN_NUM` 不是 M-RoPE MX 专属限制，而是所有场景的 T 限制。
- 当前状态：`VERIFIED`

### 当前代码事实

1. 修改前 ACLNN checker 只在 M-RoPE MX 场景以 `MAX_MROPE_MX_TOKEN_NUM` 拒绝 `T>262144`；graph
   tiling 也只在 MX 分支使用 `MAX_MROPE_MX_TOKENS`。
2. README 和 ACLNN 文档把该上限写在 MX 场景附近，未明确为 RoPE、M-RoPE 和 M-RoPE MX 的共同约束。
3. 该限制属于输入 token 数的公共接口契约，不依赖 MX scale 数或 M-RoPE position 的特殊形态。

### 评审判断与解决方案

评审意见成立。现有常量足以表达公共事实，只需去掉错误场景限定并提升校验位置，不新增派生常量：
ACLNN 使用 `MAX_TOKEN_NUM`，graph tiling 使用 `MAX_TOKENS`，三个场景统一在进入具体分支前校验。

### 实际修改

- `op_host/op_api/qkv_rms_norm_rope_cache_with_k_scale_checker.h`：常量改为 `MAX_TOKEN_NUM`，
  对所有场景统一拒绝 `T>262144`。
- `op_host/op_tiling/arch35/qkv_rms_norm_rope_cache_with_k_scale_base_tiling.h/.cpp`：常量改为
  `MAX_TOKENS`，graph tiling 公共标量校验覆盖所有场景。
- `README.md`、`docs/aclnnQkvRmsNormRopeCacheWithKScale.md`：把 `1<=T<=262144` 写为三场景共同约束。
- Op API UT 新增 RoPE、M-RoPE、M-RoPE MX 三个超限拒绝用例；Host CSV 新增 RoPE、M-RoPE
  两个超限拒绝用例，补足此前 MX 分支已有覆盖。

### 验证证据

1. 三个新增 Op API 超限用例通过；Op API UT 总计 `143/143 PASS`。
2. Host CSV 中 `rope_token_count_above_limit`、`mrope_token_count_above_limit` 均通过；Host UT 总计
   `136/136 PASS`。
3. Ascend950 `opkernel` PASS，日志 `/tmp/ops_build_4h91fcks.log`；package 的沙箱首次运行因无法下载
   `setuptools>=40.8.0` 退出 2，日志 `/tmp/ops_build_plyodwg6.log`，属于网络门禁。以相同命令在允许
   依赖访问的环境重跑后 PASS，日志 `/tmp/ops_build_8y2192i0.log`。
4. 最终 `build_out` package SHA256 为
   `7cecc8dfddbd1eceb9d0591cf457689823d2499cadf7923d6d55a66296dc9bd4`。

### 最终处理和状态

- 最终处理：262144 已成为 ACLNN 与 graph tiling 两条入口、三个场景一致的公共 T 上限。
- 最终状态：`VERIFIED`。

## REV-010 精简 KScale cache shape 校验中的重复分支

- 提出日期：2026-08-24
- 原问题：checker 中 `kScaleShapeValid` 的场景三元表达式过长，rank/head 条件重复；只需把不同的
  head dim 放入 `scaleCount` 即可精简。
- 当前状态：`VERIFIED`

### 当前代码事实

原实现的两个分支都检查 KScale rank 为 4、head 数为 `nk`，唯一差异是尾维：M-RoPE MX 使用
`GetMxScaleCount(d)`，非 MX 使用 `K_SCALE_CACHE_SCALE_DIM`。重复表达式没有独立语义。

### 评审判断与解决方案

评审意见成立。先按场景计算唯一差异 `scaleCount`，再用一条公共布尔表达式检查 rank/head/head-dim；
不新增 helper 或中间场景实体。

### 实际修改

- `op_host/op_api/qkv_rms_norm_rope_cache_with_k_scale_checker.h`：`scaleCount` 改为场景选择值，
  `kScaleShapeValid` 合并为一条无重复的 rank/head/head-dim 校验。

### 验证证据

1. Op API UT `143/143 PASS`；Host UT `136/136 PASS`。
2. Ascend950 opkernel/package 均 PASS；`git diff --check` 通过。
3. 最终三个 dtype 对象 key 集分别为 M-RoPE MX `{302059776}`、RoPE `{0,256,4352}`、M-RoPE
   `{17895680}`，相关 kernel symbol 均存在。
4. 对象大小/SHA256 分别为 M-RoPE MX
   `27,568B / e55f200d7b0bb0051738f065b4baf2da8d538740ea3c3d0ff6deb0de66152197`、RoPE
   `49,952B / a4c6c13c6487361b9699d044f794819f801faa2ede79634ed871dd24be4bcb64`、M-RoPE
   `19,248B / d4b0285166eea7355ce5947f2acfc220419104b5467cbf1ecb6fa3814db606b4`；与 REV-005
   基线逐字节相同。

### 最终处理和状态

- 最终处理：差异只保留在 `scaleCount`，公共 shape 条件只表达一次。
- 未执行项：未重复运行 NPU 精度和 TTK 性能测试；三个 device 对象与 REV-005 逐字节相同，
  本批没有改变设备代码。
- 最终状态：`VERIFIED`。

## REV-011 将原始模式组合归一化为唯一 Scene 事实源

- 提出日期：2026-08-24
- 原问题：场景由 `ropeMode_ + qQuantMode_ + kQuantMode_` 在多个函数中临时拼装，模式组合、dtype、
  rank/shape、cache shape、stride、UB 资源、tiling、blockDim 和 compileInfo 等场景知识分散；新增场景时
  容易漏改。建议建立“原始属性 -> 已验证 Scene -> SceneTraits”的内部唯一事实源，同时保留
  `ComputeTilingData()` 的策略分派和 `GetTilingKey()` 的既有 ABI 编码。
- 当前状态：`FIXED`

### 当前代码事实

1. 修改前 `ropeMode_`、`qQuantMode_`、`kQuantMode_` 的组合合法性由独立判断完成，后续校验和 tiling
   又分别用原始字段或 `IsMropeMx()` 推导场景差异。
2. 三个公开场景实际具有一组稳定的声明式差异：Q 输出、K cache、KScale cache dtype，VScale rank，
   rotation/M-RoPE position 是否存在，MX scale layout、AIV-only kernel 以及 UB 输入行容量。
3. `GetTilingKey()` 的字段和值属于既有 ABI；内部归一化不能改变它。M-RoPE MX 与其余场景的
   `ComputeTilingData()` 算法确实不同，顶层策略分派应保留。

### 评审判断与解决方案

评审意见成立。新增一个内部 `Scene` 和一张只读 `SceneTraits` 表；所有原始属性和可选输入解析完成后，
由唯一的 `ResolveScene()` 校验组合、layout 和 rotation presence，并写入 `scene_`。后续字段规则只读取
traits，不能再拼装三个原始字段。额外纳入 `qkvInputRowsPerAiv`，是因为 UB 资源差异也属于本意见明确
指出的分散场景知识，不另建第二份映射。

### 实际修改

- `op_host/op_tiling/arch35/qkv_rms_norm_rope_cache_with_k_scale_base_tiling.h`：新增 `Scene`、
  `SceneTraits`、`scene_`、`ResolveScene()` 和 `GetSceneTraits()` 声明。
- `op_host/op_tiling/arch35/qkv_rms_norm_rope_cache_with_k_scale_base_tiling.cpp`：建立 ROPE、MROPE、
  MROPE_MX 三项 traits；将组合、layout、rotation presence 的合法性集中到 `ResolveScene()`；dtype、
  shape/rank、cache scale layout、stride、UB token tile、M-RoPE tiling 字段、compileInfo 和 blockDim
  规则统一读取 traits。
- `ComputeTilingData()` 仍按 `scene_` 做 M-RoPE MX 的顶层算法分派；`GetTilingKey()` 仍直接按
  `ropeMode_`、`qQuantMode_`、`kQuantMode_` 及既有 cache dtype 字段编码，没有修改 key ABI。
- 删除旧的 `ResolveRopeMode()`、`ValidateModeCombination()` 和 `IsMropeMx()` 场景派生入口。

### 验证证据

1. 静态引用检查确认三个原始模式字段只剩属性解析、`Reset()`、`ResolveScene()`、诊断输出和
   `GetTilingKey()` ABI 编码用途；其余业务规则不再重新推导场景。
2. `git diff --check -- posembedding/qkv_rms_norm_rope_cache_with_k_scale` 通过。
3. Host UT 命令
   `bash build.sh --ophost_test --ops=qkv_rms_norm_rope_cache_with_k_scale -j1` 返回 0，
   `136/136 PASS`；已有用例覆盖三场景的合法/非法组合、layout、rotation、dtype、rank/shape、cache、
   stride、UB、compileInfo、tiling key、blockDim 和并发调用。
4. 本条曾启动 Ascend950 opkernel 构建；收到“所有检视意见完成前不跑 opkernel/package”的要求后立即
   主动中止，命令退出码 130，日志 `/tmp/ops_build_62m15raa.log`。这是按用户要求取消，不是编译失败；
   package 未启动。

### 最终处理和状态

- 最终处理：三个原始属性只在 `ResolveScene()` 中归一化一次，后续由唯一 traits 表声明场景差异；
  策略分派和 tiling key ABI 保持原有职责。
- 未执行项：Ascend950 opkernel/package 及最终对象 key、大小、SHA、symbol 核对，按用户要求推迟到
  全部检视意见完成后统一执行。
- 最终状态：`FIXED`；待最终批次设备构建和产物审计通过后再更新为 `VERIFIED`。

## REV-012 补充 GetSceneTraits 入参语义说明

- 提出日期：2026-08-24
- 原问题：`GetSceneTraits` 的入参没有注释，不容易判断传入的 `Scene` 是否已完成合法性校验。
- 当前状态：`FIXED`

### 当前代码事实

`GetSceneTraits(Scene scene)` 只应接收 `ResolveScene()` 生成的已验证场景，但声明处未说明此前置条件。

### 评审判断与解决方案

评审意见成立。该信息属于函数契约，直接在声明处添加 `@param` 注释即可；不新增 wrapper、校验分支
或第二套 scene 类型。

### 实际修改

- `op_host/op_tiling/arch35/qkv_rms_norm_rope_cache_with_k_scale_base_tiling.h`：说明 `scene` 是由
  `ResolveScene()` 产出的已验证 operator scene。

### 验证证据

1. `git diff --check` 通过。
2. Host UT `136/136 PASS`。

### 最终处理和状态

- 最终处理：入参来源和合法性前置条件已在函数声明处明确。
- 未执行项：按用户要求，本批不运行 opkernel/package。
- 最终状态：`FIXED`；待最终批次构建后更新为 `VERIFIED`。

## REV-013 在 graph tiling 中按 Q/K 区分量化模式常量

- 提出日期：2026-08-24
- 原问题：`base_tiling.cpp` 的通用 `QUANT_MODE_MX` 没有按已有代码风格区分 Q/K；如有更清晰且不过度
  抽象的方法更好。
- 当前状态：`FIXED`

### 当前代码事实

1. Q/K 的公开字符串值虽然同为 `"MxQuant"`，但分别属于 `q_quant_mode` 和 `k_quant_mode` 属性。
2. 修改前 K 的 PerTokenPerHead 解析也复用了 `Q_QUANT_MODE_PER_TOKEN_PER_HEAD`，使字段归属不清晰。
3. ACLNN 和 InferShape 当前均使用独立的 Q/K 常量名；建立通用转换框架会增加不必要实体。

### 评审判断与解决方案

评审意见成立。沿用现有最直接的字段命名：分别定义 `Q_QUANT_MODE_MX`、`K_QUANT_MODE_MX` 和
`K_QUANT_MODE_PER_TOKEN_PER_HEAD`；各 parser/ToString 只引用本字段常量。字符串值仍复用同一公开 ABI，
不引入额外映射层。

### 实际修改

- `op_host/op_tiling/arch35/qkv_rms_norm_rope_cache_with_k_scale_base_tiling.cpp`：删除通用
  `QUANT_MODE_MX`；Q/K 的解析和诊断字符串各自闭合。

### 验证证据

1. 静态搜索确认目标文件不存在通用 `QUANT_MODE_MX`，只剩 `Q_QUANT_MODE_MX` 和
   `K_QUANT_MODE_MX`。
2. Host UT `136/136 PASS`；`git diff --check` 通过。

### 最终处理和状态

- 最终处理：Q/K 常量命名和使用已与 ACLNN、InferShape 的既有风格一致，属性值与 tiling key ABI
  均未改变。
- 未执行项：按用户要求，本批不运行 opkernel/package。
- 最终状态：`FIXED`；待最终批次构建后更新为 `VERIFIED`。

## REV-014 收缩 ACLNN 参数检查并由 graph tiling 统一维护语义规则

- 提出日期：2026-08-24
- 原问题：ACLNN 接口不应维护过于复杂的判断逻辑；ACLNN 后续仍会进入 InferShape/optiling，在两边
  重复校验场景、dtype、shape、layout、stride 等规则会造成漂移。
- 当前状态：`FIXED`

### 当前代码事实

1. 修改前 ACLNN checker 为 1034 行，独立实现 scene 分类、可选参数组合、empty、dtype、shape、layout、
   cache stride 和 token 上限；其中规则在 InferShape/graph tiling 中已有对应校验。
2. ACLNN 构造 task 前必须自行保证的边界只有：会被直接解引用或用于创建 executor 的必选指针，以及
   public `qScale` 是否按 Q 量化模式提供。后者无法完全下沉，因为 L0 会为 null optional output 创建
   内部占位 tensor，下游看不到调用者原始的 null 状态。
3. cache scatter 的 stride 合法性属于全部场景的 graph/kernel 契约，graph tiling 已统一校验 K/V/KScale
   stride、非重叠和 span；不需要 ACLNN 再保留第二份 `CheckStrides()`。

### 评审判断与解决方案

评审意见成立，但不能把所有检查都删除。采用明确分层：ACLNN 只检查任务构造安全所需的必选指针和
public `qScale` presence；场景组合、可选输入组合、empty、dtype、shape、layout、token 上限和 stride
统一交给 InferShape/graph tiling。未知量化字符串也转交 graph 校验，ACLNN 的输出后置检查只对已识别的
Q 量化模式要求 qScale。

### 实际修改

- `op_host/op_api/qkv_rms_norm_rope_cache_with_k_scale_checker.h`：删除 `SceneType`、
  `DetermineScene()`、`CheckSceneParams()`、`CheckEmptyTensor()`、`CheckDtypeValid()`、`CheckShape()`、
  `CheckFormat()`、`CheckStrides()` 及其 helper；checker 从 1034 行收缩为 122 行。
- `op_host/op_api/aclnn_qkv_rms_norm_rope_cache_with_k_scale.cpp`：新增最小的
  `IsQScaleRequired()`，避免未知 qQuantMode 被 ACLNN 输出后置检查提前截获。
- `tests/ut/op_api/test_aclnn_qkv_rms_norm_rope_cache_with_k_scale.cpp/.csv`：保留原有 128 行输入矩阵，
  将 CSV 末列明确命名为 operator-level 预期，并由测试代码独立推导 ACLNN boundary 预期；语义错误
  用例改为验证 ACLNN 成功转交，下游拒绝仍由 Host UT 覆盖。相关 direct test 从 `Rejects...` 改为
  `Forwards...ToGraphValidation`。
- `tests/st/generate_st_cases.py`：仅机械同步 checker source gate token；1200 用例分类和内容未改变。

### 验证证据

1. Op API UT `143/143 PASS`：必选指针和 qScale presence 仍在 ACLNN 层拒绝；其余语义错误矩阵确认
   不再被 ACLNN 重复判断。
2. Host UT `136/136 PASS`，其中 graph tiling 的 115 个 CSV 用例继续覆盖场景组合、dtype、shape、layout、
   stride、token 上限及资源规则，未削弱最终算子校验。
3. `python3 posembedding/qkv_rms_norm_rope_cache_with_k_scale/tests/st/generate_st_cases.py check`
   返回 0；1200 用例资产检查通过。
4. 静态搜索确认 ACLNN checker 已无 `DetermineScene`、`CheckSceneParams`、`CheckDtypeValid`、`CheckShape`、
   `CheckFormat` 和 `CheckStrides`；cache stride helper 只由 graph tiling 使用。`git diff --check` 通过。

### 最终处理和状态

- 最终处理：ACLNN 不再是第二份 operator 语义事实源；API 边界检查和 graph 语义检查职责已分开。
- 未执行项：按用户要求，本批不运行 opkernel/package；最终设备对象、key、SHA 和 symbol 审计留到全部
  review 完成后统一执行。
- 最终状态：`FIXED`；待最终批次设备构建和产物审计通过后更新为 `VERIFIED`。

## REV-015 清理当前实现和测试资产中不准确的 legacy 用词

- 提出日期：2026-08-24
- 原问题：算子目录下仍存在大量 `legacy` 用词；RoPE 和非 MX M-RoPE 都是当前公开支持场景，不能用
  `legacy` 暗示其已废弃或只为兼容保留。
- 当前状态：`FIXED`

### 当前代码事实

1. 本条处理前，当前 InferShape 变量和诊断、tiling-key 编译门、通用 device controller 断言、算子定义
   注释仍使用 `legacy`；这些实体均参与当前生产实现，不是历史代码。
2. Golden、ST 生成器、测试设计和 1200 行生成资产又使用 `mrope_legacy`、`LEGACY` route、
   `_legacy_dynamic` 等分类，导致正常 M-RoPE 场景在测试事实源中继续被标成遗留路径。
3. `manual_review_record.md` 中 REV-005/REV-008 的 `legacy` 和 `MROPE_LEGACY` 是已发生评审的原问题、
   修改前事实与验证结论。修改这些原文会篡改历史证据，因此不属于当前命名实体。

### 评审判断与解决方案

评审意见成立。当前代码和测试事实源统一使用 `ROPE`、`MROPE`、`MROPE_MX`；仅用于区分两种实现族的
route 标签改为 `NON_MX`。Golden helper 按真实算法改名为 `per_token_per_head_dynamic`，不再以时间先后
命名。历史 manual review 原文保持不变，并在本条明确其只描述修改前事实。

### 实际修改

- `op_host/qkv_rms_norm_rope_cache_with_k_scale_infershape.cpp`：`isLegacyMrope*` 改为 `isMrope*`，
  诊断改为 RoPE/M-RoPE/M-RoPE MX 当前场景名称。
- `op_kernel/arch35/qkv_rms_norm_rope_cache_with_k_scale_tiling_key.h`：编译门改为
  `QKV_K_SCALE_COMPILE_ROPE`、`QKV_K_SCALE_COMPILE_MROPE`；模板值和 key ABI 未变。
- `op_kernel/arch35/qkv_rms_norm_rope_cache_with_k_scale_controller.h`、算子定义注释和 tiling 日志：
  分别改为 RoPE/M-RoPE、`unusedFields` 等真实语义。
- Golden、ST 生成器、测试设计、相关设计文档和 Host UT 测试名同步改为 M-RoPE/non-MX 表述；由生成器
  重新生成 ACLNN L0/L1/L2、Kernel、Torch E2E 和因子表，不手改派生 CSV。

### 验证证据

1. 排除只保存历史原问题的 `manual_review_record.md` 后，大小写不敏感搜索确认算子目录中已无
   `legacy`、`mrope_legacy`、`MROPE_LEGACY` 或 `QKV_K_SCALE_COMPILE_LEGACY*`。
2. `generate_st_cases.py generate` 返回 0，重新生成并自检 1200 行：ACLNN L0 200、L1 650、L2 50、
   Kernel 250、Torch 50；分类数量和用例合同不变。
3. Host UT 返回 0，`136/136 PASS`；Golden 和生成器 Python 源码 `compile()` 检查通过。
4. `git diff --check -- posembedding/qkv_rms_norm_rope_cache_with_k_scale` 通过。

### 最终处理和状态

- 最终处理：当前实现、当前设计和当前测试资产不再把 RoPE/非 MX M-RoPE 称为 legacy；manual review
  中保留的命中只属于不可改写的历史原文。
- 未执行项：按用户要求，本批不运行 opkernel/package；宏改名后的 device 构建留到全部 review 完成后。
- 最终状态：`FIXED`；待最终批次 device 构建通过后更新为 `VERIFIED`。

## REV-016 保留 ACLNN 运行所需的 kernel-config 兼容别名安装

- 提出日期：2026-08-24
- 原问题：`op_host/CMakeLists.txt` 的修改是否必要；为什么其他算子不需要相同处理。
- 当前状态：`FIXED`

### 当前代码事实

1. 该规则把生成的 `qkv_rms_norm_rope_cache_with_k_scale_apt.json` 在同一 package 中以逻辑 opType 名
   `qkv_rms_norm_rope_cache_with_k_scale.json` 再安装一次；内容仍只有 `_apt.json` 一份生成事实源。
2. 删除规则后构建、安装的 Ascend950 custom package 只包含 `_apt.json`。真实 ACLNN 调用在
   `AclOpKernelInit` 阶段失败，opType 为 `QkvRmsNormRopeCacheWithKScale`，尚未进入 kernel。
3. 在同一已安装 OPP 中仅补入逐字节相同的逻辑名 JSON 后，同一个 T=768/Nq=8 case 立即从
   `AclOpKernelInit failed` 变为五个输出均 `100%`、`PRECISION_STATUS: PASS`。因此当前 CANN 9.1
   composite ACLNN loader 的确先按逻辑 opType 查找 config，`opFile.value` 不能替代这个首查找入口。

### 评审判断与解决方案

原“该 CMake 修改不必要”的判断被设备反事实推翻。虽然其他 `_apt` 算子没有同类规则，但这只能说明仓库
惯例，不能覆盖本算子当前 composite ACLNN 注册链的真实失败。兼容别名不是第二份人工 metadata：它由
同一生成文件在 package 阶段派生，内容一致且不会独立漂移。最小修复是恢复原安装块，不改 opFile、kernel
目录或公共打包框架。

### 实际修改

- `op_host/CMakeLists.txt`：恢复仅在 custom Ascend950 package 中从 `_apt.json` 安装逻辑名兼容别名的规则。

### 验证证据

1. 删除别名的 package：opkernel/package 构建成功，但 ACLNN T=768/Nq=8 在 `AclOpKernelInit` 失败。
2. 仅补入与 `_apt.json` SHA256 同为
   `16b46e33d7660311f5295bbb0d7113386923335537d5f2f0273089fb73391aa1` 的逻辑名文件后，同 case
   五输出 `100%`、PASS；随后基线 T=1/2/3 为 3/3 PASS，证明差异点就是 config 首查找名称。
3. 恢复后的 `op_host/CMakeLists.txt` 通过 `git diff --check`。

### 最终处理和状态

- 最终处理：恢复已由设备反事实证明必要的 Ascend950 custom-package 兼容别名；`_apt.json` 仍是唯一生成源。
- 未执行项：恢复 CMake 后尚未重新生成最终 package；本次 A/B 的两侧 OPP 使用与该规则等价的逐字节复制
  继续验证。最终 package metadata 审计留到全部 review 完成后统一执行。
- 最终状态：`FIXED`；运行时必要性已证明，待恢复规则后的最终 package 构建通过再更新为 `VERIFIED`。

## REV-017 评估 M-RoPE MX tile 预取是否值得保留

- 提出日期：2026-08-24
- 原问题：`qkv_rms_norm_rope_cache_with_k_scale_mrope_mx_controller.h` 实现了预取；算子印象中是
  Vector-bound，需确认预取是否影响性能；若没有影响，希望删除以简化代码。
- 当前状态：`VERIFIED`

### 当前代码事实

1. controller 并非在当前 Vector 计算完成后才同步加载下一 tile：它先等待 tile i 的 input ready，随后
   向另一 input slot 发射 tile i+1 的 slotMapping、QKV 和 raw cos/sin 搬运，再执行 tile i 的
   `V -> K -> Q`。MTE2 与 Vector 因此存在真实的跨 tile 重叠窗口。
2. `MTE2_V INPUT_READY` 是数据就绪依赖，`V_MTE2 INPUT_FREE` 是 Q 最后读取后的 buffer 复用依赖；
   两套 input UB slot 物理区间分离。它们不是仅因源码相邻即可删除的冗余 fence。
3. 单 tile 时不会发射 next-tile lookahead；多 tile 时若改成单 slot 串行加载，则每个后续 tile 的 QKV、
   slotMapping、cos/sin GM->UB 及 Scalar 地址准备都会位于该 tile Vector 计算之前。
4. 本条性能实验前的正式数据只证明含预取的完整实现，并没有同源码、同对象合同、同输入、只关闭 lookahead 的
   matched A/B。`Vector-bound` 描述当前关键流水占比，不能反向证明预取没有价值；它也可能正是 MTE2
   延迟已被隐藏后的结果。

### 评审判断与解决方案

“应验证复杂度是否有收益”的意见成立，但“既然 Vector-bound 就可直接删除预取”的推论证据不足。
当前静态证据确认它建立了合法的生产者/消费者重叠，未发现死预取、错误同步或过宽等待，因此本批不删除。
用户进一步确认必须以实际性能测试结果决策，不能仅凭静态判断决定保留或删除。
已按用户要求提前制作仅关闭 next-tile lookahead 的单变量候选：只把 `PreloadTile(next)` 从当前 tile 的
`ProcessV/K/Q` 之前移动到 `ReleaseInput(current)` 之后，其他 schedule、数值、双 slot 和事件均不变。
实际结果未显示可重复、可泛化的预取收益，因此接受删除显式 next-tile lookahead。用户进一步明确双 input
slot 与逐槽 READY/FREE 协议必须保留；它们负责 buffer 所有权和流水依赖，不属于本次删除范围。

### 实际修改

- controller 删除 `current/next/hasNext` lookahead 状态机，改为单个 tile 描述符按
  `Load -> READY -> V/K/Q -> FREE` 顺序循环；下一个 tile 不再于当前 tile 的 `V/K/Q` 之前显式发射。
- `PreloadTile()` 更名为 `LoadTile()`，使函数名与当前调度语义一致；函数内部的数据搬运和 READY 事件不变。
- 保留两套 input UB、`tileOrdinal & 1` 槽位轮转、两套 `INPUT_READY`/`INPUT_FREE`/`SLOT_READY`
  事件及其 wait/set；未修改 UB layout、tiling 数据或 kernel key。
- 性能实验候选仍保留在 `/tmp/qkv_mrope_prefetch_ab_20260824_E2a9Jl/candidate`，未写回仓库。

### 验证证据

1. 修改前静态调用闭包确认唯一 lookahead 点位于 `WaitTileReady(current)` 之后、`ProcessV/K/Q(current)` 之前；
   `ReleaseInput(current)` 只在 Q 最后消费之后发出。
2. 同步审计确认 input slot 的 READY/FREE 分别使用 `MTE2_V`/`V_MTE2`，两者语义未混用；position window
   另由 `MTE2_S`/`S_MTE2` 管理，不把未来 window 混入当前 Vector ready event。
3. 当前仓库性能记录 `14.474/20.901/21.517/32.669 us` 只作为现实现状，不作为预取因果证据。
4. A/B 身份：基线 controller SHA256 为
   `85972bc0fef53fe09fa14f5bd63b37e9e338f812afb6b24b98994ae8d44f5adc`，候选为
   `c0a56f24cf3f95b8c19776f0176ef6c2502d2a3d707af6070a0bf3d7cc92ddd5`；只有 key `302059776`
   所在对象不同（均 27568 B），另两个对象逐字节相同。
5. 设备为 `Ascend950PR_9589`。基线 T=1/2/3 为 3/3 PASS；候选 T=768/Nq=8 与 T=1/2/3 为
   4/4 PASS；每个性能 case 均为 49 次 AiCore task、8/8 PASS。
6. 全量反序 B2→A2 的 49 次中位数中，无重叠候选在 Nq=8/16/24/32/40/48/64 更快，在 Nq=56
   基本相同；但该轮仍受执行顺序影响。随后紧邻交叉复测：Nq=8 当前预取/候选为
   `13.584/11.728 us`，Nq=64 为 `26.664/26.878 us`。预取只在 Nq=64 快 `0.214 us（0.8%）`，
   Nq=8 反而慢 `1.856 us`；方向不一致，且多轮出现毫秒级系统长尾，不能声称存在稳定收益。
7. 完整命令、对象哈希和原始 task_time 归档在
   `/tmp/qkv_mrope_prefetch_ab_20260824_E2a9Jl`。
8. 修改后静态闭包确认 controller 中已无 `next`、`hasNext`、`PreloadTile`；双槽 layout、`inputSlot` 轮转和
   两套 READY/FREE 事件仍有完整定义与调用。最终 controller SHA256 为
   `1027305ea686fb50323c6064b8eb977c9d2bf63e11e79937623ca18b3e229ec9`。
9. 用户要求重新测试后，以最终源码重新构建 Ascend950 package；命令到达 `ops_transformer_kernel` 并生成
   package，rc=0。package 文件 SHA256 为
   `20208ce420c23c5db2f5af2b9600c7b7da7eae3ad1cd8d570d73812c48f749a1`。
10. logical/`_apt` config 逐字节相同；M-RoPE MX key `{302059776}` 对象为 25,008 B，SHA256 为
    `5b7665a338a8cfa6090325146c4167a2baf50eb2fee56c640b7eb99b066248cf`；另外两个对象的 key 集仍为
    `{0,256,4352}` 和 `{17895680}`。
11. 最终 package 的 T=768、Nq=8/64 精度端点均为五输出 `100%`、PASS。
12. 最终 ACLNN task-prof 在 Nq=8/16/24/32/40/48/56/64 的 49 次 AiCore 中位数分别为
    `12.056/14.249/15.990/17.606/21.373/23.595/24.910/26.600 us`；8/8 TTK PASS，统一命中
    key `302059776`、blockDim 64、MixBlockDim 0，未发现 AIC_ERROR、缺任务或 kernel 长尾。
13. 完整身份、命令、原始 task CSV 和报告位于
    `/tmp/qkv_mrope_no_lookahead_nq_sweep_20260824_LqnDCL`。

### 最终处理和状态

- 最终处理：已删除显式 next-tile lookahead 状态机，同时按用户要求保留双缓冲及逐槽 READY/FREE 协议。
- 未执行项：未单独再跑一次 `build.sh --opkernel`；本次成功的 package 构建已实际到达
  `ops_transformer_kernel` 并生成、安装和上板执行目标对象。
- 最终状态：`VERIFIED`；最终源码、package、精度端点和全 Nq 性能曲线均已闭环。

## REV-018 为 M-RoPE MX Vector 控制代码补充必要注释

- 提出日期：2026-08-24
- 原问题：`qkv_rms_norm_rope_cache_with_k_scale_mrope_mx_vec.h` 需要增加合适的注释。
- 当前状态：`VERIFIED`

### 当前代码事实

1. 该文件同时承载双 input slot、position window、V/K/Q staging 的硬件 pipe 同步协议；其中 READY 表示
   当前内容可被下游读取，FREE 表示物理 UB 可由上游覆盖，不能按普通 CPU 线程同步理解。
2. K 路径存在 RowBatch16Even 与通用尾块两种 schedule；Q 路径按 Nq 一次性选择 RowBatch16、
   GlobalTileWave 或 RowBatch16 + TailTileWave，选择原因仅从函数名和条件表达式不易直接读出。
3. M-RoPE gather index、position window 的 refill/release，以及 E8M0 copy 参数的字节单位均属于容易误改的
   隐含契约；相反，普通赋值、tensor 绑定和显然的 API 调用不需要逐行复述。

### 评审判断与解决方案

意见成立。采用最小注释补强，不新增 helper、状态或派生事实源，也不把设计文档整段复制进代码。注释只解释
同步所有权、buffer 生命周期、schedule 分派、M-RoPE gather/window 语义和 copy 单位；保留现有实现与调用顺序。

### 实际修改

- `op_kernel/arch35/qkv_rms_norm_rope_cache_with_k_scale_mrope_mx_vec.h`：
  - 补充类级流水与双槽 READY/FREE 语义；
  - 解释初始化/析构 event token、tile 的 Scalar/Vector 双就绪域和 Q 后释放 input slot 的原因；
  - 解释 K RowBatch16Even 的适用条件，以及 Q 三种 schedule 对应的 Nq 分类；
  - 解释 M-RoPE gather index、position window 生命周期、E8M0 DataCopy 字节单位及 UB buffer 归属。

### 验证证据

1. 将修改前后文件删除独立 `//` 注释行后执行逐行 diff，结果为空，证明 device 代码 token、常量、分支和
   调用均未变化。
2. controller 调用闭包仍为 `LoadTile -> WaitTileReady -> ProcessV -> ProcessK -> ProcessQ -> ReleaseInput`，
   READY/FREE 最后消费者关系未改变。
3. 新增注释均不超过 120 列；目标文件与记录文件通过 whitespace 检查。
4. 未执行 Host UT、opkernel/package 或设备测试：本条只有注释变化，无编译产物或运行时行为变化；并遵循
   用户要求，在全部检视意见完成前不重复进行 opkernel/package 编译。

### 最终处理和状态

- 最终处理：已在复杂且容易误改的控制点补充原因型注释，未给显然代码增加复述性注释。
- 未执行项：无本条必要验证未执行；最终全量构建仍按整体 review 收口计划统一进行。
- 最终状态：`VERIFIED`；静态证据确认只有注释变化。

## REV-019 删除 Kernel 内不可达的 K head 零值防御

- 提出日期：2026-08-24
- 原问题：`const uint16_t kHeadCount = static_cast<uint16_t>(kvHeadNum_);` 对应的判断为什么允许
  `kHeadCount == 0`；是否存在 `kvHeadNum_ == 0` 的场景，按合同应至少为 1。
- 当前状态：`FIXED`

### 当前代码事实

1. `head_nums` 的公开合同要求三个元素均为正数；InferShape 明确校验
   `0 < Nq <= 64`、`Nq = 8 * Nk`、`Nk = Nv`，因此合法范围为 `Nq=8..64`、`Nk=Nv=1..8`。
2. tiling 的 `ParseHeadNumsAttr()` 在写入内部字段前拒绝任一 `head_nums[i] <= 0`，随后
   `ValidateHeadNums()` 再次拒绝 `numKHeads == 0` 并校验上述比例关系。
3. `GetShapeAttrsInfo()` 必须先完成 `BuildContractInput()` 和 `ValidateParsedInput()`；成功后
   `FillMropeMxTilingData()` 才把 `input_.numKHeads` 写入 `tilingData_.kvHeadNum`。Kernel 读取到的合法
   `kvHeadNum_` 因而只能为 1 至 8，转为 `uint16_t` 也不会截断。
4. 现有负向资产已包含 `headNums.Nk=0`，期望在参数/Host 校验阶段失败且不进入 tiling 或 Kernel。

### 评审判断与解决方案

意见成立。当前支持场景中没有 `kvHeadNum_ == 0`；Kernel 本身信任已经验证并序列化的 tiling data。
`kHeadCount != 0U` 只是在消费端重复 Host 合同，还会让读者误以为存在零 K head 场景。最小修复是删除该条件，
保留偶数 head 和整除 16-row batch 的真实 schedule 条件。若绕开 Host 人工构造非法 tiling data，不属于公共
算子合同，也不应在这个热路径分支中单独兜底。

### 实际修改

- `op_kernel/arch35/qkv_rms_norm_rope_cache_with_k_scale_mrope_mx_vec.h`：从 K schedule 条件中删除
  `kHeadCount != 0U`，保留 `(kHeadCount & 1U) == 0U` 与 `16 % kHeadCount == 0U`。

### 验证证据

1. 静态约束闭包确认 Parse、InferShape 和 tiling Validate 三处均禁止 `Nk=0`；tiling data 仅在验证成功后生成。
2. 静态引用检查确认目标算子中 `kHeadCount != 0U` 已无残留；本次源码 diff 只有该条件删除及自然换行合并。
3. 现有负向设计资产 `C-HEAD-001` 和 ACLNN L2 manifest 均覆盖 `headNums.Nk=0`，并声明不得启动 Kernel。
4. 目标改动通过 whitespace 检查。
5. 未执行 Host UT：Host 合同和测试资产未修改；未执行 opkernel/package 和设备测试，遵循用户要求在全部
   检视意见完成前不重复编译，最终构建验证统一收口。

### 最终处理和状态

- 最终处理：删除不可达的 K head 零值防御，不改变合法输入上的 schedule 选择。
- 未执行项：最终 opkernel/package 编译及设备回归留到全部 review 完成后统一执行。
- 最终状态：`FIXED`；源码与约束闭包已完成，待最终 device 构建后更新为 `VERIFIED`。

## REV-020 评估 ProcessK 只保留通用路径的性能代价

- 提出日期：2026-08-24
- 原问题：如果 `ProcessK` 删除 RowBatch16Even 分支、只保留通用处理，性能差异有多大。
- 当前状态：`VERIFIED`

### 当前代码事实

1. `Nk=Nq/8`，合法范围为 1 至 8。RowBatch16Even 仅覆盖 `Nk=2/4/8`，即 `Nq=16/32/64`；
   `Nk=1/3/5/6/7`（`Nq=8/24/40/48/56`）当前已经全程走通用 VF。
2. RowBatch16Even 分别把 8、4、2 个 token 的 K head 合成 16 行，将 MX scale 阶段从单 head 的
   `4/64` lane 使用率提升为 16 行的满载 `64/64`，并摊薄 scale 构造和流水 barrier 固定开销。
   通用 VF 则逐 token、逐 head 完成 RMSNorm、M-RoPE 和 MX 量化。
3. 最终源码的 ACLNN task-prof 总耗时在 Nq=16/32/64 分别为 `14.249/17.606/26.600 us`；这些只是
   当前含 RowBatch16Even 的整算子基线，不是该分支的独立耗时或收益。
4. 原 Nq sweep 不能作为分支 A/B：不同 Nq 同时改变 Q schedule、Q/K/V 行数、UB layout 和搬运量。
   因此本项另建隔离 baseline/candidate，只改变 `ProcessK` 分派并对相同 Nq 做 matched pair。

### 评审判断与解决方案

“只保留通用路径能否用少量性能换取明显简化”是有效评审问题。现已完成只改变 `ProcessK` 分派的真实设备
matched A/B：全通用候选在 Nq=16/32/64 上分别造成 `3.56%/5.73%/6.61%` 的合并中位数回退，
而无行为变化的 Nq=48 控制点差异仅 `-0.09%`。三个受影响 shape 的两轮结果方向一致，且回退随 Nk
增加而扩大；因此这不是采样漂移，RowBatch16Even 对整算子有可重复的实际收益。

解决方案是保留当前 `Nk={2,4,8}` 的 RowBatch16Even 专用路径以及其他 Nk/尾块的通用路径。该分支只存在
于 Q 入口外的 K 调度层，不改变双缓冲、READY/FREE 协议、tiling 或公开 key；相对于约 3% 至 7% 的稳定
收益，当前局部复杂度是必要的。

### 实际修改

- 仓库算子源码未修改，保留当前 RowBatch16Even 实现。
- 在 `/tmp/qkv_mrope_k_generic_ab_20260824_wyFeTU` 创建隔离 baseline/candidate；candidate 仅把
  `ProcessK` 改为无条件调用 `KRmsNormMropeMxD128VfImpl`，未修改 Q、VF 实现、双缓冲、READY/FREE、
  tiling、key 或 UB layout。
- 将完整构建日志、package/对象身份、精度结果和两轮性能原始数据保存在上述 `/tmp` 根目录；仓库中只更新
  本评审记录。

### 验证证据

1. 静态调用闭包确认 RowBatch16Even 的可达集合为 `Nk={2,4,8}`，通用集合为
   `Nk={1,3,5,6,7}` 以及批处理余数。
2. VF 对比确认通用路径逐 head 调用 `MxQuantCublasD128`，RowBatch16Even 将 16 行汇总后调用
   `MxQuantCublasScaleRowBatch16DeferredRms*` 和 `MxQuantCublasDataRowBatch16`。
3. 用户明确授权本项性能对比后，baseline 与 candidate 的 Ascend950 package 均构建 PASS。package SHA256
   分别为 `4272cc4f39d07719a1537a2909f0c5f0b86caa0fe221864a5d8cbb224b07cd83` 和
   `1e7c488725522b5621ad4ac0c70217583c74c4a0b00327bfa47ad51c4de0304c`。
4. 两侧 dtype 对象 key 集均保持 `{302059776}`、`{0,256,4352}`、`{17895680}`。只有 M-RoPE MX
   对象变化：baseline 为 25008 bytes、SHA256 `d85acfb0cef539b92ea1f26c43bb1b568aa0f4f31c1355141b28b8f65f4cb8ff`；
   candidate 为 22240 bytes、SHA256 `e915ac1d98a4e0e648c7b776da375fd03b9d5ece1553d0151986b45e05e6a5c1`。
   另外两个对象逐字节相同；MX JSON 除对象 SHA 外相同，key 仍为 `302059776`。
5. 设备可见环境为 Ascend950PR_9589、device 0；`/dev/davinci0`、`/dev/davinci_manager`、
   `/dev/hisi_hdc` 可见。candidate 对 Nq=16/32/64 完成精度 smoke，五个输出均为 100%，三例均
   `PRECISION_STATUS: PASS`。
6. 两轮采用反向交错顺序；每个版本、每个 shape、每轮各 49 个 `AI_VECTOR_CORE` task。逐轮中位数差异为：
   - Nq=16：`+3.28%`、`+3.00%`；合并每侧 98 个 task 后为 `14.030 -> 14.529 us`，`+3.56%`。
   - Nq=32：`+4.69%`、`+6.57%`；合并后为 `17.370 -> 18.364 us`，`+5.73%`。
   - Nq=48 控制点：`+0.61%`、`-1.49%`；合并后为 `23.375 -> 23.354 us`，`-0.09%`。
   - Nq=64：`+6.38%`、`+6.85%`；合并后为 `26.346 -> 28.087 us`，`+6.61%`。
7. 16 次性能运行全部 TTK PASS，原始 task CSV 每次均有 49 条目标 key `302059776` 的
   `AI_VECTOR_CORE` 记录，未发现 AIC_ERROR。完整证据见
   `/tmp/qkv_mrope_k_generic_ab_20260824_wyFeTU/RESULT.md`。
8. 本次仅针对 M-RoPE MX 当前公开 key 的 K schedule 做决策；未扩大到其他场景，也未修改或覆盖既有测试结果。

### 最终处理和状态

- 用户最终决定：保留 K 的 RowBatch16Even 专用分支。
- 最终处理：保留当前 RowBatch16Even 专用分支；全通用候选不进入仓库。
- 未执行项：未运行全量 ST；候选已由三个受影响 shape 的设备 Golden smoke 和两轮 matched 性能覆盖，
  最终仓库源码未发生 Kernel 修改，因此本项不要求把实验候选纳入全量回归。
- 最终状态：`VERIFIED`；专用路径对 Nq=16/32/64 的整算子收益已由反向交错 matched A/B 验证。

## REV-021 Golden 是语义实现还是仿照 NPU 实现

- 提出日期：2026-08-24
- 原问题：检查 `tests/assets/qkv_rms_norm_rope_cache_with_k_scale_golden.py` 是独立算子语义实现，
  还是仿照 NPU/Kernel 实现。
- 当前状态：`OPEN`

### 当前代码事实

1. 四类 Kernel/ACLNN/Torch frontend adapter 最终都调用唯一的 `_golden_core`；Golden 不调用被测 ACLNN、
   NPU `DynamicMxQuant`、AscendC helper 或 Kernel 对象作为 oracle。
2. `_golden_core` 用 NumPy 按逻辑张量表达以下数据流：layout 归一化、RMSNorm、RoPE/M-RoPE、rotation、
   Q/K 量化、V 静态量化、`slotMapping` cache scatter 和未命中 cache 继承。它没有 tiling key、blockDim、
   RowBatch16、Q/K schedule、READY/FREE 或 VF 分支；`_CHUNK_T=256` 只是 CPU 内存分块，不参与结果定义。
3. `_rms_rope` 直接写数学公式；`_golden_core` 直接按 `slot/blockSize` 计算 cache 位置。正常有限值主路径因此是
   独立的语义实现，而不是按设备指令或当前 schedule 逐步复刻。
4. `_mx_quant_cublas` 包含 FP32 bit 拆解、E8M0 raw byte、非有限 D32 的 `0xFF` scale，以及
   DAV_3510 `SatMode::SAT + CTRL[50]=0` 下 NaN 转 E4M3FN raw `0x00` 的显式模拟。这一段虽然接近设备
   位算法，但 M-RoPE MX 合同已明确锁定 `DynamicMxQuantV3(scaleAlg=1)`，包括 zero、NaN、Inf、上下溢、
   rint 和饱和转换，因此它属于“硬件派生但已进入合同的语义”，不是拿当前 Q/K VF schedule 当 oracle。
5. `_per_token_per_head_dynamic` 也显式写了“Match DynamicQuantD128 exactly”，并把 `amax==0` 的 `0/0`
   在 DAV_3510 上转换为 FP8 raw `0x00`。这一 helper 同时服务普通 RoPE FP8 和旧 M-RoPE INT8；其中 FP8
   零行分支是直接按当前 Kernel cast 行为修正 `ml_dtypes`，不是由公开公式独立推出。
6. 当前公开文档只定义 `scale=amax/M` 和 `cast(x/scale)`，没有定义 `amax=0`、NaN、Inf 时的具体字节。
   `implementation_deviations.md` 的 DEV-013 又明确要求普通 RoPE 特殊值先定合同，不能依据 DAV3510 dump
   修改 Golden。该要求与 `_per_token_per_head_dynamic` 当前的 FP8 零行强制 raw `0x00` 存在事实冲突。
7. compare 层对 E8M0 使用 raw-byte exact，对普通 E4M3FN 允许相邻一个编码差异；这进一步说明 Golden
   正常路径不是试图逐指令复制当前 Kernel 的所有舍入顺序。

### 评审判断与解决方案

结论不是二选一，而是“语义主干 + 合同锁定的硬件数值细节 + 一处尚未合同化的 NPU 行为模拟”：

- 整体架构和正常有限值计算是独立语义 Golden，设计方向正确；不能称为仿照 NPU 实现。
- M-RoPE MX 的 E8M0/FP8 位级边界来自锁定的 `DynamicMxQuantV3` 合同，保留位级实现是必要的；否则无法
  验证 E8M0 raw byte、指数边界和非有限值合同。
- 普通 RoPE `PerTokenPerHead` 的 FP8 零行 raw `0x00` 尚未被公开合同定义，却在 Golden 中按 DAV_3510
  行为硬编码。这一小段目前确实是“跟 NPU”，并会把普通 RoPE 的实现偏差掩盖成 PASS，属于真实的独立性问题。

建议保持 `_golden_core` 的语义结构不变，并按场景拆清数值依据：M-RoPE MX helper 继续严格实现锁定合同；
普通 RoPE/旧 M-RoPE 的特殊值先完成 DEV-013 合同决策，再决定 Golden 输出。合同未确定前，不应继续用
DAV_3510 raw `0x00` 作为普通 RoPE 的真值，也不应仅为让当前 Kernel PASS 而修改 Golden。

### 实际修改

- 本条未修改 Golden、Kernel、测试 CSV 或设计合同；仅新增本评审记录。

### 验证证据

1. 完整阅读 620 行 Golden，并追踪四类 TestSpec adapter 到 `_golden_core` 的调用闭包。
2. 静态检索确认 Golden 中没有 AscendC、VF、tiling、blockDim、RowBatch16、READY/FREE 或被测 NPU 算子调用。
3. 将 `_rms_rope`、`_mx_quant_cublas`、`_per_token_per_head_dynamic`、cache scatter 与当前 Kernel VF、
   M-RoPE MX contract、ACLNN 公开公式逐项对照。
4. M-RoPE MX contract 明确要求 `DynamicMxQuantV3(scaleAlg=1)` 的 zero/NaN/Inf/上下溢/rint/饱和行为
   逐项一致；普通 PerTokenPerHead 文档没有对应特殊值定义。
5. `implementation_deviations.md:27-37` 明确把普通 RoPE 特殊值合同列为 DEV-013 未决项，并禁止用设备 dump
   反向修正 Golden；与当前 `_per_token_per_head_dynamic` 的 FP8 零行分支形成可定位冲突。
6. 未执行编译或设备测试：本条是源码与合同的静态独立性审计，未修改任何运行代码。

### 最终处理和状态

- 最终处理：确认 Golden 主体是独立语义实现；M-RoPE MX 位级细节有合同依据；普通 RoPE FP8 零行仍含
  未合同化的 DAV_3510 行为模拟。
- 未执行项：尚未决定普通 RoPE 对 zero、NaN、Inf 的公开支持域和精确输出合同，也未据此修订 Golden/用例。
- 最终状态：`OPEN`；等待 DEV-013 合同决策后再消除普通 RoPE Golden 与独立性要求之间的冲突。

## REV-022 reciprocal scratch 使用 BF16 还是 FP32 的性能对比

- 提出日期：2026-08-24
- 原问题：当前 RowBatch reciprocal scratch 使用 128B BF16，后续读取后再做 BF16→FP32；若改为
  256B FP32，可以删除 BF16 打包和恢复转换并保留 FP32 数值。两者不能只按搬运字节数判断，需要做
  单变量实机 A/B，并覆盖多个 Nq。
- 当前状态：`VERIFIED`

### 当前代码事实

1. 修改前每个 RowBatch16 scratch record 包含 8192B FP32 data、256B UINT32 max、128B BF16 reciprocal
   和 256B FP32 rms-squared；每个 scratch record 总计 8832B。BF16 路径在 scale 阶段将 64 个 FP32
   reciprocal 截掉低 16 位并 Pack/Store，data 阶段再 `DIST_E2B_B16` load 和 Cast 回 FP32。
2. FP32 候选把 reciprocal 区域改为 256B，record 总 scratch 增加 128B 至 8960B；scale 阶段直接
   `StoreAlign<float>`，data 阶段直接 `DIST_E2B_B32` load。候选不改变三个 Q schedule、K 专用分支、
   双缓冲、READY/FREE、tiling key 或公开接口。
3. FP32 候选除了性能机制外也消除了 reciprocal 的 BF16 截断，因此不是逐字节数值等价候选；必须先过
   Golden 精度门槛。最终候选在 Nq=8/24/64 的五个输出均为 100%，三例全部 PASS。
4. 候选第一次精度 smoke 在 Nq=64 暴露 Q 输出约 50.07% 匹配。根因不是 FP32 数值，而是多 batch
   `reciprocalBatchStride` 仍以 `sizeof(uint16_t)` 换算；修正为 `sizeof(float)` 后 Nq=64 恢复五输出
   100%。最终 A/B 只使用修正后的 package。
5. scratch 大小会参与 tiling，但不需要在 `op_tiling` 中维护第二份数值：Host tiling 直接包含共享的
   `mrope_mx_layout.h`，用 `TryMakeMropeMxUbLayout()` / `TryMakeMropeMxQTailWaveUbLayout()` 计算
   `layout.totalBytes` 并按 `compileInfo_->ubSize` 搜索 `tokenTile`；Device 又用同一常量族重新物化 layout，
   以 `layout.mxScratchOffsetBytes` / `layout.mxScratchBytes` 绑定 UB。因此 8832B→8960B 已自动传播到
   Host 资源判定和 Device 地址布局，边界 shape 如需缩小 `tokenTile` 也会由现有搜索自动完成。

### 评审判断与解决方案

这是有效的精度与性能权衡问题。理论上 FP32 多搬 128B，但删除了 Shift/Pack、BF16 store/load 和
BF16→FP32 Cast；两项代价无法仅由字节数推出整算子结果。用户进一步明确，本项目标不是要求 FP32 更快，
而是在性能基本相当时优先采用保留 FP32 reciprocal 的更高精度路径。

按该准则，实机结果满足采用条件：Nq=8 基本持平，Nq=16/24/32/40/56/64 合并中位数提升
0.81%～2.15%，最差的 Nq=48 回退 1.16%。这属于小幅波动而非显著性能退化；同时 FP32 路径消除了
reciprocal 的 BF16 截断以及后续 BF16→FP32 恢复转换。因此采用 FP32 reciprocal scratch，但不把它
表述为普遍性能优化。

### 实际修改

- 仓库将 `qkv_rms_norm_rope_cache_with_k_scale_mrope_mx_layout.h` 的 reciprocal scratch 从 128B BF16
  改为 256B FP32，并在 `qkv_rms_norm_rope_cache_with_k_scale_mrope_mx_vf.h` 中改为直接 FP32
  store/load，同时把所有 reciprocal scratch 指针和 batch stride 改为 FP32 单位。
- 未改三个 Q schedule、K 专用分支、双缓冲、READY/FREE、tiling key、公开接口、测试 CSV 或 Golden。
- 未单独修改 `op_tiling`：scratch 大小由 Host/Device 共用的 layout 事实源计算，新增独立 tiling 常量反而会
  形成易漂移的第二事实源。
- 在 `/tmp/qkv_mrope_recip_fp32_ab_20260824_qGG5zl` 创建隔离 baseline/candidate。candidate 仅修改
  `qkv_rms_norm_rope_cache_with_k_scale_mrope_mx_layout.h` 和
  `qkv_rms_norm_rope_cache_with_k_scale_mrope_mx_vf.h`，实现 256B FP32 reciprocal scratch、直接 FP32
  store/load 以及对应 FP32 batch stride；仓库最终可执行代码与该已构建、已实测候选一致。
- 构建日志、package/对象身份、精度结果、38 次性能运行及原始 task CSV 均保存在上述 `/tmp` 根目录；
  仓库未写入构建或性能测试产物。

### 验证证据

1. baseline 与最终 candidate 的 Ascend950 package 均构建 PASS。package SHA256 分别为
   `8eb39f1bcf8d8e8df85cf13b0049befa470fcc6230068573ae2eef03c1c716e2` 和
   `f97dc9187b43c7dea3f0dea4a4f19768c12024418bfeca41bdc1ce2a37da1253`。
2. 两侧 dtype 对象 key 集保持 `{302059776}`、`{0,256,4352}`、`{17895680}`。仅 M-RoPE MX 对象变化：
   baseline 25008 bytes、SHA256 `d85acfb0cef539b92ea1f26c43bb1b568aa0f4f31c1355141b28b8f65f4cb8ff`；
   candidate 24976 bytes、SHA256 `6c79e72444f8ae03b75362beac40abd93dc1d171016bcb06fa4e31c779bad321`。
   另外两个对象逐字节相同。两侧 MX 对象均导出 key `302059776` 的同名 Kernel symbol，MX JSON 除对象
   SHA256 外相同。
3. 最终 candidate 在 Ascend950PR_9589 device 0 上对 Nq=8/24/64 完成精度 smoke，五个输出均为
   100%，三例 `PRECISION_STATUS: PASS`。
4. Nq=8～64 两轮采用反向顺序 matched A/B；每侧每 shape 每轮 49 个 key `302059776` 的
   `AI_VECTOR_CORE` task。对两轮方向不一致的 Nq=8/40/48 增加第 3 轮。逐轮与合并中位数如下：

   | Nq | FP32 逐轮差异 | BF16 合并中位数/us | FP32 合并中位数/us | FP32 合并差异 |
   |---:|:---|---:|---:|---:|
   | 8  | `+2.88% / -1.36% / +0.00%` | 12.131 | 12.132 | `+0.01%` |
   | 16 | `-0.48% / -2.48%` | 14.247 | 14.041 | `-1.44%` |
   | 24 | `-0.94% / -2.18%` | 16.431 | 16.160 | `-1.65%` |
   | 32 | `-2.32% / -1.13%` | 17.608 | 17.230 | `-2.15%` |
   | 40 | `-1.93% / +0.43% / -0.80%` | 21.146 | 20.975 | `-0.81%` |
   | 48 | `-0.50% / +0.54% / +2.13%` | 22.999 | 23.266 | `+1.16%` |
   | 56 | `-0.57% / -1.00%` | 25.087 | 24.833 | `-1.01%` |
   | 64 | `-1.97% / -0.27%` | 26.297 | 26.024 | `-1.04%` |

5. Nq=8/40/48 合并每侧 147 个 task，其余 shape 合并每侧 98 个 task。38 次性能运行全部 TTK PASS，
   未发现 AIC_ERROR。完整命令、身份和原始数据索引见
   `/tmp/qkv_mrope_recip_fp32_ab_20260824_qGG5zl/RESULT.md`。
6. 仓库 layout 文件 SHA256 为
   `4d6b5a008c4b4bf3d9b90c3370d2945346f0434ee5c71a3df7ffdc00cb859db9`，与已构建候选逐字节一致。
   VF 与已构建候选的可执行代码一致，文本差异仅为候选测试完成后新增的两处解释性注释；静态检索确认已无
   `uint16_t *reciprocalScratch`、`reciprocalB16` 或 `reciprocalHighBits`，batch stride 使用
   `sizeof(float)`。因此不重复构建 package，既有 candidate package、精度与性能结果直接覆盖最终实现。
7. 理论指令节省的精确 cycle 上界和剩余可节省量未单独用流水工具量化，状态为 `NOT_EVALUATED`；本项决策
   依据整算子 matched task time，不把理论指令数当作实测收益。

### 最终处理和状态

- 最终处理：采用 256B FP32 reciprocal scratch。它不是统一更快的性能优化，但在 Nq=8～64 实测中
  性能基本相当，并消除了 BF16 reciprocal 截断，符合“性能相当时优先更高精度”的决策准则。
- 未执行项：未跑全量 ST，也未重新构建与已测候选功能代码一致的 package；已有 candidate package 构建、
  三类 Q schedule 代表 shape 的设备 Golden smoke 和 Nq=8～64 matched 性能覆盖。
- 最终状态：`VERIFIED`；FP32 实现、精度门槛、性能等价性和最终源码到已测候选的身份关系均已闭环。
