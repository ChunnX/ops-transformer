# FlashAttnGrad Tiling 侧架构设计

> 目的：把 `attention/flash_attention_score_grad`（下称 **FAG**）arch35 的能力迁移到
> `attention/flash_attn_grad`（下称 **FAGrad**）时，tiling 侧应该长成什么样。
>
> 适用范围：`ascend950` / `NpuArch::DAV_3510`，kernel 为 pypto（`op_kernel/flash_attn_grad.py`）。
>
> 基线分支：`fag_pto_tiling_reconstruct` @ `3cd842ea3`。决策状态见 §9。

---

## 1. 结论摘要

1. **不要整体照搬 FAG arch35 的 tiling 类结构。** arch35 host tiling 约 6900 行，其体量主要来自两块，而这两块在 FAGrad 里都不成立：
  - 一是**已被接口缩减砍掉的特性**（pse / dropout / prefix / rope / 量化 / fp32 / 8 种 sparse_mode…），它们撑起了 `FuzzyBaseInfoParamsRegbase` 的上百个字段和 1443 行 tilingkey 白名单；
  - 二是**分核枚举**（`blockStarts/Ends[36]`、TND 前缀和、确定性调度），这部分在 FAGrad 已经下移到 AICPU metadata。
2. **应该采用"薄入口 + 只读 POD + 单一职责策略文件 + 按 layout 分模板"的分层**。这套分层已经存在于 `feat/pr9617_followup` 分支的提交 `aa1be54a5`（标题「【FAG】【PTO】tiling reconstruct」），其父提交正是当前分支 HEAD `3cd842ea3`；当前分支上的 433 行单函数与它**逻辑逐字相同**。因此落地方式是 cherry-pick 那个提交而非重写，属**零行为变化的纯重构**，应当作为后续迁移的地基（见 §3.2、§7-P0）。
3. **三条必须写进架构、并用 UT 钉住的不变量**（详见 §6）：
  - **A. host tiling 不依赖输入张量的数值**（不声明 `TilingInputsDataDependency`）；
  - **B. host 的分核公式与 AICPU metadata 同源**，且 host 只允许"多发不少发"；
  - **C. host 与 pypto kernel 的三份契约（TilingData 字段序 / TilingKey 位域 / Workspace 分区）单一真源**。
4. 迁移按 layout 与特性**分阶段**推进（§7），每阶段以"新增一个模板类 + 一组 UT"收口，而不是一次性把 arch35 搬平。
5. **路由是最长期的维护成本，要单独设计**（§5.13）。tiling 的分支主要来自路由，而路由规则会随上板实测反复变动。分层只能让改动局部化，防错要靠**能力/偏好分离 + 决策可追溯 + 路由黄金基线**三件事；其中基线（§6-D）性价比最高，是唯一能让"一条规则改动影响了哪些 shape"变得可见的机制。

---



## 2. 两侧现状



### 2.1 规模对比


| 侧                | 文件                                                                               | 行数   |
| ---------------- | -------------------------------------------------------------------------------- | ---- |
| FAG arch35 host  | `op_host/arch35/*_tiling_common_regbase.cpp`                                     | 1901 |
|                  | `op_host/arch35/*_tiling_normal_regbase.cpp`                                     | 2365 |
|                  | `op_host/arch35/*_tiling_varlen_regbase.cpp`                                     | 1873 |
|                  | `op_host/arch35/*_tiling_common_regbase.h`                                       | 636  |
| FAG arch35 契约    | `op_kernel/arch35/*_tiling_data_regbase.h`                                       | 1504 |
|                  | `op_kernel/arch35/*_template_tiling_key.h`                                       | 1443 |
| FAGrad host（当前）  | `op_host/flash_attn_grad_tiling.cpp`                                             | 433  |
|                  | `op_host/flash_attn_grad_tiling_check.cpp`                                       | 438  |
|                  | `op_host/flash_attn_grad_block_outer.h`                                          | 72   |
| FAGrad 分核（AICPU） | `attention/flash_attn_metadata/op_kernel_aicpu/flash_attn_grad_metadata_split.h` | 1142 |
| FAGrad kernel    | `op_kernel/flash_attn_grad.py`                                                   | 2812 |




### 2.2 FAG arch35 的 tiling 组织方式

```
TilingFlashAttentionGradScore(context)            // flash_attention_score_grad_tiling.cpp
  ├─ CheckParams / LogConstraintWarnings          // 面向 22 输入 / 7 输出 / 14 属性
  ├─ IsEmptyOutput → RunEmptyTilingRegbase        // 空 tensor 独立 tilingkey
  └─ TilingRegistryArch::DoTilingImpl(context)    // 按 NpuArch 查表，按优先级试
       ├─ FlashAttentionScoreGradTilingVarlenRegbase   (DAV_3510, priority 900)  // TND
       └─ FlashAttentionScoreGradTilingNormalRegbase   (DAV_3510, priority 950)  // 泛化
```

- 两个模板类都继承 `TilingBaseClass`（`common/include/op_host/tiling_base.h`），走固定 8 步：
`GetShapeAttrsInfo → GetPlatformInfo → IsCapable → DoOpTiling → DoLibApiTiling → GetWorkspaceSize → PostTiling → GetTilingKey`。
- Varlen 继承 Normal，只重写 TND 相关虚函数（`GetSparseUnpadBlockInfo` / `CalcleTNDDeterParam` / `CalcTNDSwizzleParam` …）。
- 跨类共享状态放在 `FuzzyBaseInfoParamsRegbase`（轴信息 + 特性开关 + 分核结果 + 确定性调度表），另有 `TndBaseInfo`、`DeterPrefixData`。
- 跨类共享逻辑以自由函数放在 `*_tiling_common_regbase.{h,cpp}`：参数处理（`ProcessPseInfo` / `ProcessDropoutInfo` / `ProcessQuantInfo` / `ProcessSinkInfo` / `ProcessSparseModeInfo` / `ProcessTokensInfo`）、shape/dtype 校验、模板路由（`SetSplitAxis`）、基本块选择（`GetS1S2TemplateType` / `GetDTemplateType`）、sparse 枚举与确定性调度。

**值得继承的三点**：注册表 + 优先级 + `IsCapable` 的路由方式；"一个 POD 承载跨阶段状态"的思路；normal / varlen 按 layout 分家。
**不值得继承的三点**：把所有特性字段堆进一个结构体；host 侧 sparse 逐块枚举；用 1443 行 `ASCENDC_TPL_SEL` 白名单剪组合。

### 2.3 FAGrad 接口缩减后，arch35 里可以**直接不迁**的逻辑

对比 `flash_attention_score_grad_def.cpp`（22 输入 / 7 输出 / 14 属性）与 `flash_attn_grad_def.cpp`（6 必选 + 7 可选输入 / 3 输出 / 9 属性）：


| 被砍掉的接口                                            | 连带可删除的 arch35 tiling 逻辑                                                                                                                                          |
| ------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `pse_shift`                                       | `ProcessPseInfo` / `ProcessPseNormal` / `ProcessInnerPseInfo` / `ProcessPseSparseMode8` / `SetPseLayout`，`PseShapeType`(6) × `PseType`(4) 组合，tilingkey `IsPse` 位 |
| `drop_mask` + `keep_prob`/`seed`/`offset`         | `ProcessDropoutInfo` / `ProcessDropoutIsDivisibleBy8`、`dropMaskOuter` 对模板路由的干扰、tilingkey `IsDrop` 位                                                              |
| `padding_mask`                                    | 对应校验分支                                                                                                                                                           |
| `prefix` + `sparse_mode` 5/6                      | `SetPrefixSparseParams` / `CheckPrefixNExist` / `GetSparsePrefixBlockInfo` / `prefixN[2048]` / `PREFIX_COMPRESS_S1_SIZE`                                         |
| `sparse_mode` 1/2/7/8                             | `AttenMaskCompressMode`(7) / `SparseMode`(9) 的多数分支；FAGrad 只有 `mask_mode ∈ {0,3,4}`                                                                               |
| `q_start_idx` / `kv_start_idx`                    | `SetQKVStartIdx` 及其对 token 修正的影响                                                                                                                                 |
| `d_scale_*` / `ds_scale` / `p_scale` + FP8/HIFP8  | `ProcessQuantInfo`、`DtypeEnum` 的 4/5/6 档、tilingkey `InputDType` 的 4 个取值                                                                                          |
| `query_rope` / `key_rope` / `dq_rope` / `dk_rope` | `hasRope` 全链路、`ROPE_D_192`/`ROPE_D_64`/`ROPE_POST_BASE`、tilingkey `IsRope` 位                                                                                     |
| `dpse` 输出                                         | pre/post 的 dpse 清零与分核                                                                                                                                            |
| `softmax_max`+`softmax_sum` → `softmax_lse`       | `CheckSoftmaxMaxShape` / `CheckSoftmaxSumShape`、`tndMaxSumLayout`、`softmax_in_layout` 属性分支                                                                       |
| fp32 输入 / `out_dtype`                             | `DtypeEnum::FLOAT32` 全部分支、tilingkey `OutDType` 位                                                                                                                 |
| `head_num` / `inner_precise` 属性                   | 属性校验；FAGrad 的 N1/N2/G 全部由 shape 推导                                                                                                                               |


**FAGrad 新增/保留需要 tiling 处理的**：`sinks`、`seqused_q/kv`（等长 padding 下的变长）、`metadata`（AICPU 分核结果）、`mask_mode` + `win_left/win_right`、`max_seqlen_q/kv`、`layout_q/kv/out` 三个 layout 属性、以及 `D ≠ Dv`。

> 结论：FAGrad 的 tiling 只需覆盖 arch35 的**一个正交子集**：轴推导 + 布局视图 + 模板路由 + D 分档 + swizzle + workspace + tilingkey。规模量级应在 **1000 行以内**，而不是 6900 行。

---



## 3. 最大的架构差异：分核已经下移到 AICPU

这是 FAGrad 与 FAG arch35 在 tiling 架构上**唯一的本质区别**，也是所有分层决策的出发点。

### 3.1 差异本身


|                                     | FAG arch35                                                                                                 | FAGrad                                                                 |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| 每核 `[start, end)` 区间                | host `DoSparse` / `GetSparseBlockInfo` / `GetSparseUnpadBlockInfo` 算出 `blockStarts/Ends[36]`，写进 TilingData | AICPU `flash_attn_grad_metadata_split.h` 算出，写进 metadata 张量，kernel 运行期读 |
| `s1Outer` / `s2Outer` / `splitAxis` | host 算                                                                                                     | AICPU 算（`InitFagBaseDims` / `SetFagSplitAxis`）                         |
| 变长实际序列长度                            | host 需要值 → `.TilingInputsDataDependency({12,13,14,15,16})`                                                 | host 只看 shape；实际长度由 AICPU 读                                            |
| pre/post 行区间                        | host `DoPreTiling` / `DoPostTiling`                                                                        | AICPU `FagSplitRows`（`FAG_DQ_ROW_STARTS_OFFSET` …）                     |




### 3.2 由此得到的目标形态

host tiling 退化为**纯 shape/attr 的函数**，只负责 5 件事：

1. 从 context 解析轴（B/S1/S2/N1/N2/G/D/Dv）与属性；
2. 把 layout 差异折叠成 4 维视图参数（`viewD0Q/viewD2Q/coefB0Q/...`），使 kernel 只需一份 `pl.load`；
3. 决定编译期分档 → tilingkey（template / layout / has_atten_mask / swizzle / d_align / dv_align / is_bn2_multiblk）；
4. 决定启动核数 `SetBlockDim` 与 `SetScheduleMode`；
5. 计算 workspace 大小。

`feat/pr9617_followup` @ `aa1be54a5` 里就是这个形态，共 6 个文件、约 630 行（`build/binary/ascend950/src/flash_attn_grad/op_host/` 下的 9/5 快照是构建该分支的产物）：

```
flash_attn_grad_tiling.cpp          51   薄入口：ParsePlatform → CheckParams → ParseInfo → DoOpTiling
flash_attn_grad_tiling_info.h      104   索引/属性常量 + FagTilingInfo（解析结果 + host 决策）
flash_attn_grad_tiling_info_parser 218   唯一读 context 取轴/属性的地方
flash_attn_grad_tiling_split       109   SelectSplitAxis（BN2/BN2GS1S2 路由）+ ApplyLaunch（核数/调度模式）
flash_attn_grad_tiling_swizzle     146   DecideSwizzle（L2 工作集判据）
flash_attn_grad_tiling_padded      172   FillTilingData + GenTilingKey + SetWorkspace 编排
```

当前分支的 `flash_attn_grad_tiling.cpp`（433 行）是上述 5 个文件的合并版，注释与判据逐字相同 —— **改成分层不改变任何行为**。两分支在 `attention/flash_attn_metadata/` 上零差异，即这次重构完全没碰 AICPU 分核那一侧，符合 §6-B 的分工不变量。

---



## 4. 建议的目标分层



### 4.1 依赖方向

```
                 ┌───────────────────────────────────────────────┐
   L0  入口      │ flash_attn_grad_tiling.cpp                    │  IMPL_OP_OPTILING
                 └───────┬───────────────────────────────┬───────┘
                         │                               │
   L1  契约      ┌───────▼──────────┐          ┌─────────▼─────────────────┐
                 │ tiling_info.h    │          │ 生成契约（唯一真源在 py） │
                 │  索引/属性常量   │          │  TilingData / TilingKey   │
                 │  FagTilingInfo   │          └───────────────────────────┘
                 └───────▲──────────┘
                         │  只读写 POD，不碰 context
   L2  解析      ┌───────┴───────────┐
                 │ tiling_info_parser│  ← 唯一 GetInputShape/GetAttrs 的地方
                 └───────────────────┘
   L3  校验      ┌───────────────────┐
                 │ checkers/*        │  ← 唯一 OP_LOGE 报参数错的地方
                 └───────────────────┘
   L4  策略      ┌──────────┬──────────┬──────────┬──────────┐
                 │ split    │ swizzle  │ dbucket  │ workspace│  纯函数(FagTilingInfo&)
                 └──────────┴──────────┴──────────┴──────────┘
   L5  模板      ┌───────────────────┬───────────────────┐
                 │ tiling_padded     │ tiling_varlen     │  按 layout 分家，注册进 registry
                 │ (BSND / BNSD)     │ (TND / seqused)   │
                 └───────────────────┴───────────────────┘
```

**硬性规则**：L4/L5 只读写 `FagTilingInfo`，除 `SetBlockDim` / `SetTilingKey` / `SetScheduleMode` / `GetWorkspaceSizes` / `GetTilingData` 这几个**写回**接口外，不得触碰 `gert::TilingContext`。轴与属性一律从 POD 拿。这条规则是"策略可单测"的前提。

### 4.2 目标文件布局

```
attention/flash_attn_grad/
├── op_host/
│   ├── flash_attn_grad_tiling.cpp            # L0 薄入口 + IMPL_OP_OPTILING + TilingParse
│   ├── flash_attn_grad_tiling.h              # CompileInfo；TilingData 契约转发（见 §5.7）
│   ├── flash_attn_grad_tiling_info.h         # L1 索引常量 + FagTilingInfo
│   ├── flash_attn_grad_tiling_info_parser.*  # L2
│   ├── flash_attn_grad_tiling_base.h         # L5 自持的 FagTilingBase（复制 FiaTilingBase 形态，见 D1）
│   ├── checkers/                             # L3（形态参照正向，代码自持，见 D3）
│   │   ├── fag_checker.*                     #    组合器
│   │   ├── layout_checker.*                  #    layout_q/kv/out 一致性、维数
│   │   ├── shape_checker.*                   #    B/S/N/D 一致性、D ≤ 192、Dv ≤ D
│   │   ├── mask_checker.*                    #    mask_mode × attn_mask × win_left/right
│   │   ├── seq_len_checker.*                 #    cu_seqlens / seqused / max_seqlen
│   │   └── dtype_checker.*                   #    q/k/v/dout/attn_out/dq/dk/dv 同 dtype、lse fp32
│   ├── flash_attn_grad_split_policy.h        # L4 与 AICPU 共享的分核公式（现 block_outer.h）
│   ├── flash_attn_grad_tiling_split.*        # L4 SelectSplitAxis + ApplyLaunch
│   ├── flash_attn_grad_tiling_swizzle.*      # L4 DecideSwizzle
│   ├── flash_attn_grad_tiling_dbucket.*      # L4 D/Dv 分档（从 GenTilingKey 里抽出来）
│   ├── flash_attn_grad_tiling_workspace.*    # L4 workspace 分区（与 kernel make_ptr 对齐）
│   ├── flash_attn_grad_tiling_padded.*       # L5 非 TND 模板
│   └── flash_attn_grad_tiling_varlen.*       # L5 TND / seqused 模板（待新增）
├── op_kernel/flash_attn_grad.py              # TilingData / TilingKey 的唯一真源
└── tests/ut/op_host/                         # 待新增，见 §6
```

`op_host/CMakeLists.txt` 现在通过 `add_tiling_modules()` 收集，新增 `*_tiling*.cpp` 无需改构建脚本；`checkers/` 子目录需要确认是否被 glob 覆盖，否则显式列出（正向 `flash_attn/op_host/CMakeLists.txt` 是可参考的先例）。

---



## 5. 逐项设计决策



### 5.1 用扁平 POD 取代 `FuzzyBaseInfoParamsRegbase`

`FagTilingInfo`（现 104 行头文件里的结构体）已经是正确形态：约 30 个标量字段，无数组、无 `std::vector`、无 36 核调度表。为避免它退化成第二个 `FuzzyBaseInfoParams`，定三条**字段准入规则**：

1. **只放跨阶段传递的量**。单个策略内部的中间量（如 `liveHeads`、`perHeadBytes`）留在函数里。
2. **不放 kernel 内部常量**。`CUBE_BASE_M/N`、UB 分区偏移属于 kernel，host 只在需要同判据时按 §6-C 共享。
3. **不放 AICPU 已经负责的量**。`blockStarts/Ends`、per-batch seqlen、pre/post 行区间一律不进 host POD；host 侧的 `s1OuterMeta`/`s2OuterMeta` 仅用于**推算核数**，命名上要保留 `Meta` 后缀以示"这是 AICPU 口径的镜像值"。

字段建议按"解析结果"与"host 决策"两段分区并加注释分隔 —— 当前结构体已经这么排（`b/s1/s2/...` 与 `tmpl/swizzle/dAlign/tilingKey/...`），保持即可。

### 5.2 索引与属性常量单点化

当前 `flash_attn_grad_tiling.cpp` 与 `flash_attn_grad_tiling_check.cpp` **各自**用魔数读属性：

```cpp
// tiling.cpp
auto scalePtr = attrs->GetAttrPointer<float>(0);
auto maskModePtr = attrs->GetAttrPointer<int64_t>(1);
auto maxSeqlenQPtr = attrs->GetAttrPointer<int64_t>(4);
// check.cpp 又把 1/2/3/4/5/6/7/8 重新读了一遍
```

问题有两个：魔数分散，以及**同一属性被解析两次**（校验解析一遍、tiling 再解析一遍，两处默认值必须手工保持一致）。

建议：属性索引常量与输入索引常量统一放 `flash_attn_grad_tiling_info.h`（`ATTR_SOFTMAX_SCALE_INDEX` … `ATTR_LAYOUT_OUT_INDEX`，9/5 版本已经这么做），**属性只在 parser 解析一次**，checker 从 `FagTilingInfo` 读值校验。这同时把执行顺序改成 `ParsePlatform → ParseInfo → Check → DoOpTiling`（当前是 `Check → ParseInfo`），需要注意 checker 里"依赖原始 context 才能查的东西"（如可选输入是否存在、维数）仍走 context，而"值的合法性"走 POD。

对齐参考：正向 `flash_attn/op_host/fa_tiling_info.h` 就是这个形态（名字常量 + 索引常量 + `FaTilingInfo`），FAGrad 与它同属一个特性族，术语应当一致。

### 5.3 校验层：拆成组合式 checker

> **已落地。** 原先 `FlashAttnGradCheck` 是一个 456 行、6 个静态方法的类，`CheckShape` 单函数约 200 行、覆盖 B/S/N/D × 8 个张量，随着 TND/sparse/sinks 接入会线性膨胀。现已按下述关注点拆到 `op_host/checkers/`，实际落法见本节末的「落地记录」。

建议按**关注点**拆到 `checkers/`，**形态**参照正向 `flash_attn/op_host/checkers/fa_checker.h` 的组合模式：一个 `FagChecker` 持有 `std::vector<std::unique_ptr<FagBaseChecker>>`，分 `CheckParaExistence` / `CheckSinglePara` / `CheckMultiPara` / `CheckFeature` 四轮。收益不只是行数，更重要的是错报可控：错误信息集中，便于统一成 `OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON` 这类带原因的格式（arch35 用的就是这套）。

**决策（D3）：checker 代码一律本算子自持，不与正向共用、不抽到 `attention/common/op_host/`。**
`layout` / `mask_mode` / `cu_seqlens` / `sinks` / `metadata` 这几组参数虽然与正向 `FlashAttn` 语义相同，但仍在 `flash_attn_grad/op_host/checkers/` 下**复制一份**实现。理由是解耦：正向改校验口径不应牵动反向，反之亦然；代价是要自己防口径漂移。

漂移由两个非代码手段兜住，不靠共享实现：

1. **文件头对照注释**：每个 checker 头部写明「形态参照 `flash_attn/op_host/checkers/<x>.h`；语义对应 arch35 `<函数名>`」，让 review 时能三方对读；
2. **正反向口径对照表**：把易漂移的具体口径列成表，随校验改动一起更新 —— 例如 `attn_mask` 是否必须 `[2048,2048]`、TND 下 `cu_seqlens` 是否必传、`layout_q/kv/out` 是否必须三者相同、`max_seqlen_*` 的默认值语义。表放在本文档或 checker 目录的 README，作为正反向对比的唯一落点。

拆分不必一次做完：先把 `CheckShape` 按 B/S/N/D 四个维度拆成四个私有函数，再按上表落到文件，是可增量的。

#### 落地记录

`op_host/checkers/` 下的文件与 checker：

| 文件 | checker | 覆盖 |
| --- | --- | --- |
| `fag_base_checker.h` | `FagBaseChecker` / `FagCheckCtx` | 基类与共享上下文 |
| `fag_feature_checker.*` | `FagDeterministicChecker` | D2 确定性拒绝 |
| `fag_input_checker.*` | `FagInputExistenceChecker`、`FagDtypeChecker` | 必选输入齐全、8 个张量 dtype 一致 |
| `fag_attr_checker.*` | `FagAttrRangeChecker`、`FagMaskChecker`、`FagLayoutChecker` | 属性取值范围、mask_mode↔attn_mask↔window、layout 枚举与三者相同 |
| `fag_shape_checker.*` | `FagShapeChecker` | 维度数、B/S/N/D 一致、softmax_lse、cu_seqlens |
| `fag_checker.*` | `FagChecker` | 组合器，注册顺序即校验顺序 |

`flash_attn_grad_tiling_check.{h,cpp}` 收成门面（22 行），只把 `CheckParams` 转发给 `FagChecker::Process`，所以 `flash_attn_grad_tiling.cpp` 不受影响。子目录靠 `op_host/CMakeLists.txt` 里 `file(GLOB_RECURSE ... checkers/*checker*.cpp)` + `target_sources` 收进编译 —— 父级的 tiling glob 只认 `*_tiling*.cpp`，不显式收会静默漏编。写法与正向 `flash_attn/op_host/CMakeLists.txt` 一致。

**两处对设计的偏离，都是有意的：**

1. **没有分四轮。** 正向分四轮是因为它有八个 checker、多数要参与多轮；FAGrad 只有六个 checker，每个都只落在一轮里。做成四轮会让每个 checker 实现一个方法、另外三个返回 `SUCCESS`，正是 §8.2 第一条列为反面写法的空虚函数。所以退成「单个 `Check` + 有序列表」，顺序由 `FagChecker` 构造函数显式表达。等 checker 数量涨上来再分轮。
2. **checker 吃 `gert::TilingContext` 而不是已解析的 POD。** 正向的 checker 吃 `FaTilingInfo`，但 FAGrad 的校验发生在 parser 之前（入口顺序 `CheckParams` → `ParseInfo`），所以只能直接读 context。属性集中在 `FagAttrRangeChecker` 里读一次、存进 `FagCheckCtx` 供后续复用，不让每个 checker 各读一遍。若将来要改成吃 POD，得先调整入口顺序，那是行为相关改动，不在本次范围内。

`CheckShape` 的收敛靠三个表驱动辅助函数（`CheckDimNum` / `CheckDimEqual` / `GetAxisIndex`）替掉逐张量重复的 `OP_CHECK_IF`：原先加一个张量要改八处，现在只改一处张量表。顺带修掉两个小问题：`qN % kN` 缺 `kN == 0` 守卫（空张量可达，原为除零 UB），以及 `GetDimNum()` 返回 `size_t` 却用 `%ld` 打印。

### 5.4 模板层与注册表

建议 FAGrad 也走注册表，而不是在入口函数里 `if (isTnd) ... else ...`：

```cpp
REGISTER_TILING_TEMPLATE_WITH_ARCH(FlashAttnGrad, FlashAttnGradTilingVarlen,
                                   static_cast<int32_t>(NpuArch::DAV_3510), 900);
REGISTER_TILING_TEMPLATE_WITH_ARCH(FlashAttnGrad, FlashAttnGradTilingPadded,
                                   static_cast<int32_t>(NpuArch::DAV_3510), 950);
```

理由：

- **路由条件下沉到** `IsCapable()`，与 arch35 习惯一致（varlen 900 优先于 normal 950），新增模板不需要改入口；
- 未来接 BN2S2（tilingkey 已预留 `template=3`）或确定性模板时，是加文件而不是加分支；
- 注册表天然支持"本类不支持就往下试"的 `GRAPH_PARAM_INVALID` 语义，比手写 if-else 更容易表达 fallback（AICPU 侧 `DoFagSparse` 就是三层 fallback，host 侧路由要能跟得上）。

**基类二选一，需要一次定死**：


|     | `TilingBaseClass`（`common/include/op_host/tiling_base.h`）  | `FiaTilingBase`（`attention/common/op_host/fia_tiling_base.h`）          |
| --- | ---------------------------------------------------------- | ---------------------------------------------------------------------- |
| 流程  | 8 步固定骨架，含 `GetWorkspaceSize`/`PostTiling`/`DumpTilingInfo` | 3 步（`InitTilingInfo`/`IsCapable`/`DoOpTiling`），写回封装成 `SetXxx`          |
| 注册  | `TilingRegistryArch`（按 arch）                               | `FiaTilingRegistry`（按 arch，`DoTilingImpl(context, tilingInfo)` 可带 POD） |
| 使用者 | FAG arch35                                                 | 正向 `FlashAttn`                                                         |


**决策（D1）：采用 `FiaTilingBase` 的形态，但在本算子目录下自持一份，不 include 正向那份。**

具体落法：新增 `flash_attn_grad/op_host/flash_attn_grad_tiling_base.h`，把 `FiaTilingBase` 的三步骨架（`InitTilingInfo` / `IsCapable` / `DoOpTiling`）与 `SetNumBlocks`/`SetTilingKey`/`SetWorkspaceSize`/`SetScheduleMode`/`SetTilingData` 这几个写回封装**复制**过来，改名为 `FagTilingBase`；注册表同理自持一份 `FagTilingRegistry`（或直接沿用 `REGISTER_TILING_TEMPLATE_WITH_ARCH` + `TilingRegistryArch`，因为它在 `common/include/` 下属于全仓公共设施、并非正向私有）。

选它的理由：`DoTiling(TilingInfo*)` 直接把外部解析好的 POD 传进模板类，正好匹配 §4.1 的"parser 独占 context"分层。自持一份的理由与 D3 相同 —— 与正向解耦。代价是 workspace/blockDim 的调用点要自己编排（成本很小，几个 `Set*` 转发）。

**同时必须保持与 arch35 的对照关系**（这是 D1 的附加约束）：不能因为骨架从 8 步变成 3 步，就让读者无法把 FAGrad 的实现与 arch35 的对应段落对上。维护办法见 §5.12。

无论选哪个基类，**不要混用**两套骨架。

**注意注册表只覆盖 layout 这一根轴。** 这里的 `IsCapable` 路由决定"哪个 C++ 模板类接管"（padded / varlen），粒度是 layout。而性能相关的路由——BN2 / BN2 MultiBlk / GS1S2、swizzle 开关、D 分档——最终落在 **tilingkey 取值**上，由同一个 pypto 入口内部分派，不对应 C++ 类。这部分才是会随实测反复变动的地方，其组织方式见 §5.13，不要试图用注册优先级去表达。

### 5.5 TilingKey 单点生成

FAGrad 的 tilingkey 只有 7 个字段 / 9 bit，远小于 arch35 的 19 字段 / 55 bit：


| bit | 字段                | 取值                                   | 备注                      |
| --- | ----------------- | ------------------------------------ | ----------------------- |
| 1:0 | `template`        | 0=BN2GS1S2, 1=BN2, 2=未用, 3=BN2S2(**保留位，勿复用**) | 见 D5 |
| 2   | `layout`          | 0=非TND, 1=TND                        | BNSD **不占位**，靠视图参数运行期区分 |
| 3   | `has_atten_mask`  | 0/1                                  |                         |
| 4   | `swizzle`         | 0/1                                  | 仅 `template=0`          |
| 6:5 | `d_align`         | 64/128/192                           | 决定片上 tile 宽度，必须编译期      |
| 7   | `dv_align`        | 128/192                              | `d_align=64` 时强制 128    |
| 8   | `is_bn2_multiblk` | 0/1                                  | 仅 `template=1`          |


两个要点：

1. **位域拼装只允许出现在一个函数里**（建议 `flash_attn_grad_tiling_key.h/cpp`，从现在的 `GenTilingKey` 里抽出）。当前它和 D 分档逻辑、报错逻辑混在同一个函数里，改 D 档次会误伤 key 拼装。
2. **组合剪枝的真源在 pypto 侧** `FlashAttnGradTilingKey.is_valid()`。host 必须只发 `is_valid` 为真的组合，否则运行期找不到二进制。这条不能靠人眼保证 —— 见 §6-C 的 UT 建议。**不要**把 arch35 那套 1443 行 `ASCENDC_TPL_SEL` 白名单迁过来：pypto 的 `is_valid` 已经承担了同样职责，两处白名单只会互相漂移。
3. **决策（D5）：`template=3`（BN2S2）保留不动。** 本轮不实现、不编译、host 永不产生该取值（`SelectSplitAxis` 只输出 0/1，建议加一条 `OP_CHECK_IF(tmpl == 2 || tmpl == 3, ...)` 之类的兜底断言）。`template=2` 同样标记为未用。这两个取值都**不得挪作他用** —— 将来接 BN2S2 时要能直接复用 pypto 侧已有的位定义与 `is_valid` 剪枝规则，改语义会让两侧同时失配。



### 5.6 Workspace 契约

当前 workspace = `libapi + (dq + dk + dv) * fp32`，且注释明确"必须与 kernel 的 `make_ptr` 分区完全一致"。这是典型的**跨语言隐式契约**，建议显式化：

```cpp
// flash_attn_grad_tiling_workspace.h
struct FagWorkspaceLayout {
    size_t libapiBytes;           // [0]
    size_t dqOffset,  dqBytes;    // B*S1*N1*D  fp32   (N1 = N2*G)
    size_t dkOffset,  dkBytes;    // B*S2*N2*D  fp32
    size_t dvOffset,  dvBytes;    // B*S2*N2*Dv fp32
    size_t total;
};
FagWorkspaceLayout CalcFagWorkspace(const FagTilingInfo &info, size_t libapiBytes);
```

好处：offset 而不只是 total 被写出来后，UT 可以直接对照 kernel 里 `make_ptr` 的偏移，而不是只校验一个总和；未来加 deter 用的额外缓冲时，分区表是唯一要改的地方。

### 5.7 TilingData 双声明问题（当前的真实隐患）

`flash_attn_grad_tiling.h` 里注册的是 `FlashAttnGradTilingDataTmp`：

```cpp
BEGIN_TILING_DATA_DEF(FlashAttnGradTilingDataTmp)
TILING_DATA_FIELD_DEF(int64_t, b)
...
REGISTER_TILING_DATA_CLASS(FlashAttnGrad, FlashAttnGradTilingDataTmp)
```

而 tiling 实际写的是 `context->GetTilingData<FlashAttnGradTilingData>()` —— 后者由 pypto 从 `flash_attn_grad.py` 的 `@dataclass FlashAttnGradTilingData` 生成（`build/.../FlashAttnGradTilingData_tiling.h`）。于是同一份内存布局存在**两份声明**：一份给 GE 注册用、一份给 host 写值用，靠"字段名与顺序逐一对应"的人工约定同步。py 侧注释已经点明这个约束：

> 字段名与顺序必须与 host 侧逐一对应 —— codegen 出的 `FlashAttnGradTilingData_tiling.h` 是按这里的声明顺序排布内存的。

建议（按优先级）：

1. 在 host 侧对生成头做 `static_assert(sizeof(FlashAttnGradTilingData) == ...)` 与逐字段 `offsetof` 断言，把"顺序一致"变成编译期错误；
2. 或者由 codegen 同时产出 `BEGIN_TILING_DATA_DEF` 版本，删掉手写的 `_Tmp`；
3. 至少把 `_Tmp` 改名为 `FlashAttnGradTilingDataRegistryOnly` 并在注释里指明真源，避免后人误以为它是被写入的那份。



### 5.8 平台信息应走 CompileInfo，L2 不应硬编码

两处问题：

- `TilingParseForFlashAttnGrad` 是空实现，`FlashAttnGradCompileInfo::ParamCheck` 直接 `return GRAPH_SUCCESS`。每次 tiling 都重新 `GetPlatformInfo()` 取核数。arch35 的 `TilingPrepareForFlashAttentionScoreGrad` 会把 `aivNum/aicNum/ubSize/l1Size/l0aSize/l0bSize/l0cSize/l2CacheSize/npuArch` 全部缓存进 CompileInfo。
- `DecideSwizzle` 里 `constexpr int64_t L2_CACHE_BYTES = 128L * 1024 * 1024;` **硬编码**了 L2 容量，而 arch35 用的是 `fBaseParams.l2CacheSize`（来自平台）。一旦跑在 L2 容量不同的型号上，swizzle 判据会整体失准。

建议：填齐 `FlashAttnGradCompileInfo`（复用 `CompileInfoCommon` 的字段集），`FagTilingInfo` 增加 `l2CacheSize`（以及后续可能用到的 `ubSize/l0cSize`），`DecideSwizzle` 改用该值。注意 `TilingParse` 在图模式下可能拿不到 platform（arch35 对 `platformInfoPtr == nullptr` 有回退到 CompileInfo 的分支），迁移时保留同样的双路。

### 5.9 空 tensor / 零长序列

arch35 有独立通道：`IsEmptyOutput(context) || IsTndAllSeqLenZero(context)` → `RunEmptyTilingRegbase` → 专用 tilingkey（`IsEmptyTensor` 位）+ 按 aivNum 均分清零。

FAGrad 目前没有这条路径。B/S=0 时现有代码不会崩（`FagS1S2Outer` 对 `s==0` 返回 outer=0，`FagUsedCubeCores` 回退到 1，workspace 只剩 libapi），但语义是"发一个核什么也不做"，且 TND 全零 seqlen 依赖 AICPU 的 `fagIsSeqExistZero_` 判断。建议显式化：

- 在 parser 之后加一次 `IsEmptyShape(info)` 判定，命中则走独立分支（`SetBlockDim(1)` + 不进策略层 + workspace 只留 libapi），并在日志里明确标注；
- 若将来 dq/dk/dv 需要"输出非空但计算为空"的清零（例如 TND 某个 batch 长度为 0），再考虑像 arch35 一样占一个 tilingkey 位。当前 pypto 侧没有 empty 模板，不要提前占位。



### 5.10 确定性计算：当前明确拒绝

arch35 有 4 种确定性模式（`DeterSparseType` = `DETER_OLD` / `DETER_DENSE` / `DETER_CAUSAL` / `DETER_BAND`）、`deterPrefix[132]` 系列前缀表、`SelectBlockSchedule` 调度选择，以及 `context_->GetDeterministic()` 的读取。

FAGrad **完全没有读** `GetDeterministic()`，而 dq/dk/dv 走的是 fp32 `atomicAdd` 累加 —— 也就是说开启确定性开关时结果依然不确定。这不是缩减接口能覆盖掉的问题（确定性是框架开关而非算子参数），必须显式表态：

**决策（D2）：当前拒绝，不静默放行。**

落法：在 checker 里读 `context->GetDeterministic()`，为 1 时 `OP_LOGE` 并返回 `ge::GRAPH_PARAM_INVALID`（注意是**拒绝**而非 `GRAPH_FAILED` 之外的静默路径，也不要只打 `OP_LOGW`）。报错文案要说清"不支持"而不是"参数非法"，便于用户定位，例如：

```
FlashAttnGrad does not support deterministic computation yet
(dq/dk/dv are accumulated with fp32 atomicAdd). Please disable the
deterministic switch, or use FlashAttentionScoreGrad instead.
```

这条要配一个 UT：`deterministic=1` 时 tiling 必须失败，防止后续重构不小心把这个分支丢掉、退回静默非确定。

**长期放开路径**：`template` 字段仍有 `2` 未用、tilingkey 也还有高位可扩，deter 模板作为独立 L5 模板类接入，前缀表放 AICPU（与分核同侧），不要回流 host。放开时把上述 UT 从"必须失败"改成正例即可。



### 5.11 sinks / seqused 的归属

`sinks` 与 `seqused_q/kv` 是 FAGrad 相对 arch35 的**新增**参数（arch35 有 `sink` 但无 `seqused`）：

- `hasSeqused` 目前只用于关闭 `useActualCores`（保守全片启动）。真正的变长分核在 AICPU（`InitFagLayoutAndSparse` 里 `hasSequsedQ || hasSequsedK → fagUseTndSplit_ = true`）。这个分工是对的，host 侧不要试图算 seqused 的分核。
- `sinks` 目前 host 未消费。接入时它影响的是 kernel 的 softmax 分支，属于 tilingkey 位候选（arch35 有 `sinkOptional`/`s1SinkOuter`/`s2SinkOuter`）。**决策（D4）：是否占位暂定**，留到 pypto 侧 sinks 实现方式确定后再定。判据已经明确，与 §5.5 的 D 分档一致：**改变片上 tile 形状的才占位，只改索引/数值的不占位**。在定下来之前，host 侧不要预留位、也不要预先解析 `sinks`。

---

### 5.12 与 arch35 的对照关系如何维护

D1 选了三步骨架、D3 让 checker 自持，两者都会让 FAGrad 的代码结构离 arch35 更远。而 arch35 仍是**行为基准**（精度对齐、判据来源、后续特性的参照实现），所以对照关系必须显式维护，不能靠"读者自己去找"。三条要求：

1. **每个文件头写明 arch35 出处。** 格式统一为一行注释，指到类/函数级：

   ```cpp
   // 对应 arch35: FlashAttentionScoreGradTilingNormalRegbase::DoOpTiling / SetSplitAxis
   //   (attention/flash_attention_score_grad/op_host/arch35/
   //    flash_attention_score_grad_tiling_normal_regbase.cpp)
   // 差异: 无 pse/dropout/rope/quant；分核枚举下移 AICPU（见 §3）
   ```

   "差异"一行是关键 —— 它让 review 时能立刻判断"这里少写的东西是有意裁剪，还是漏了"。

2. **骨架映射表**放在 `flash_attn_grad_tiling_base.h` 头部，把三步与 arch35 八步的关系一次讲清：

   | arch35（`TilingBaseClass` 8 步） | FAGrad 落点 |
   | --- | --- |
   | `GetShapeAttrsInfo` | L2 `ParseFlashAttnGradInfo`（入口调用，不在模板类里） |
   | `GetPlatformInfo` | L2 `ParsePlatform` |
   | `IsCapable` | L5 模板类 `IsCapable`（同名同义） |
   | `DoOpTiling` | L5 模板类 `DoOpTiling`（编排 L4 策略） |
   | `DoLibApiTiling` | 无（FAGrad 不用高阶 API tiling） |
   | `GetWorkspaceSize` | L4 `CalcFagWorkspace` + `SetWorkspaceSize` |
   | `PostTiling` | L4 `ApplyLaunch`（`SetBlockDim`/`SetScheduleMode`）+ `FillTilingData` |
   | `GetTilingKey` | L4 `GenTilingKey` |

3. **判据类代码保留 arch35 溯源注释。** `DecideSwizzle` / `SelectSplitAxis` / D 分档这三处的阈值都源自 arch35（`CheckExceedL2Cache` / `SetSplitAxis` / `GetDTemplateType`）或实测标定，现有注释已经写了实测数据，迁移时**不要精简掉**——它们是这些魔数唯一的解释来源。

---

### 5.13 路由逻辑的可演进设计

这一节针对 FAG 这类算子最长期的维护成本：**tiling 是分支最多的地方，而其中的路由规则会随上板实测反复变动。** 今天某个 shape 走 A 模板，明天实测发现 B 更好，改一条门限就可能悄悄改变一片 shape 的路由。§4 的分层只解决了"改动局部化"，不解决"改错了没人发现"——那需要本节的三个补强。

前提是一个有利条件：**kernel 侧模板是固定的几套**（流水与 buffer 分配基本不变），所以路由的目标空间是可枚举的小集合，而且这个集合已经由 pypto 的 `FlashAttnGradTilingKey.is_valid()` 声明。因此这里不需要规则引擎，一张表加一个循环就够。

#### 5.13.1 根因：三类性质不同的判断被塞进同一个控制流

if-else 会无限堆积，根因不是"分支多"，而是下面三类判断混在一起：

| 类别 | 例子 | 变化频率 | 判断错的后果 |
| --- | --- | --- | --- |
| **能力**（能不能跑对） | `d <= 128`、`n1 == n2`、`is_valid()` 是否编了该组合 | 低，跟随 kernel | 功能 bug / 运行期找不到二进制 |
| **偏好**（哪个更快） | `bn >= 256`、swizzle 的 1.25×/2×L2 阈值 | **高，每次上板都可能改** | 性能回退 |
| **副作用**（顺手改别的） | arch35 `SetSplitAxis` 里改 `isDeterministic`、`layoutType` | —— | 最难定位 |

现状实证：FAGrad `SelectSplitAxis` 里 `info.d <= BN2_MAX_D`（能力）与 `bn >= BN2_MULTIBLK_BN_256`（纯实测偏好）在同一个嵌套 if 中；arch35 更严重，`SetSplitAxis` 在选模板的同时把 `fBaseParams.isDeterministic` 置 false、把 `layoutType` 改成 `INPUT_FORMAT_TND`。后果就是：改一条偏好门限必须穿过一堆能力约束，容易误伤功能，且影响面不可见。

#### 5.13.2 补强一：路由改成「有序候选 + 能力过滤 + 偏好门」

把嵌套 if 摊平成一张表，每个候选自带**分开的**两个谓词：

```cpp
// 候选按性能优先级排列，越靠前越希望命中。逐行试：能力不满足跳过，
// 能力满足但偏好门不通过也跳过，第一个双双通过者胜出。
// 明天实测说 B 更好 => 只动 kRoutes 顺序或只改 Prefer_*，永不碰 Feasible_*。
struct FagRoute {
    const char *name;
    bool (*feasible)(const FagTilingInfo &);   // 能力：kernel 实现边界 + is_valid
    bool (*prefer)(const FagTilingInfo &);     // 偏好：上板标定门限
    const char *calibration;                   // 标定出处：每个魔数都要能追溯
};

constexpr FagRoute kRoutes[] = {
    {"BN2_single",    Feasible_Bn2Single, Prefer_Always,      "S<=128 无多核累加，省 pre/post"},
    {"BN2_multiblk",  Feasible_Bn2Multi,  Prefer_Bn2BnGate,   "BN>=256 或 BN>=128 且 S 对齐 128"},
    {"GS1S2_swizzle", Feasible_Gs1s2,     Prefer_SwizzleGate, "29 点标定，见 tiling_swizzle.cpp"},
    {"GS1S2_linear",  Feasible_Gs1s2,     Prefer_Always,      "兜底，恒可行"},
};
```

三点收益：

- **优先级顺序变成一处可见的列表**，而现在它隐含在 if 的嵌套深度里；
- **末行"兜底恒可行"从结构上保证一定选得出来**，不会出现漏判落空；
- **能力与偏好在类型上分开**，改性能标定物理上碰不到功能约束。

一条要求：**能力的真源不在 host 手写第二份。** pypto 的 `is_valid()` 已经是一张声明式能力表（哪些组合编了二进制），`Feasible_*` 应当向它对齐、并由 §6-C 的 UT 校验，而不是各写一遍再靠人眼同步。

#### 5.13.3 补强二：决策可追溯

把 §5.6 里 `SwizzleReject` 的思路推广到整个路由：

```cpp
struct FagRouteTrace {
    const char *chosen;                  // 命中哪个候选
    const char *rejected[kRouteNum];     // 每个被淘汰的候选
    const char *reason[kRouteNum];       // "capability: d>128" / "preference: bn=96 < 128"
};
```

落地是 `OP_LOGI` 一行加 UT 断言。价值出现在那个具体时刻：上板发现 B 更好，第一件事是搞清**今天为什么选了 A**——有 trace 就是读一行日志，没有就得逆向 if 树。UT 里断言"因为哪条不满足"而不是只断言 `tmpl == 0`，规则改动时的失败信息会直接指出是能力变了还是偏好变了。

#### 5.13.4 补强三：路由黄金基线（最关键）

前两个让改动**便宜**，只有这个能**防止出错**。机制与格式见 §6-D。

这里说清它解决的痛点：arch35 的 tiling UT 是 25 个以上手写 `TEST_F`、每个 60~105 行手搭 shape 与属性。改一条门限时只能看到"我手挑的 case 里哪个崩了"，看不到"哪些 shape 的路由悄悄变了"。pypto 注释里那句「GQA 下曾因此有 19 个 shape 静默不一致」正是这个失效模式的实证。

#### 5.13.5 改一次规则的操作流程

以「`(B=2, N=32, S=8192, D=128)` 今天走 `GS1S2_swizzle`，实测发现某子区间 linear 更快」为例：

1. **动一行**：改 `Prefer_SwizzleGate` 的阈值，或调 `kRoutes` 顺序。`Feasible_*`、tilingkey 拼装、workspace、parser、checker 一个都不碰。
2. **更新标定出处**：`calibration` 字段与注释里补上新实测点（保持 swizzle 现有那 29 点的记法）。
3. **跑 sweep**，`--update-baseline` 更新基线表。
4. **review 看两个 diff**：规则那一行，加基线表的差异行。基线 diff 的行数就是"这次影响了多少 shape"。
5. **若基线 diff 出现预期外的 shape**（只想动 N=32 却发现 N=8 也变了），当场发现。

第 5 步是现在完全缺失的能力。真正省力的不是第 1 步（本来就改一行），而是第 4、5 步把影响面从不可见变成可 review 的 diff。

#### 5.13.6 什么时候不要做

当前偏好规则只有两条（`bnGate`、swizzle 三档阈值），**不要造规则引擎**。上面那张表配一个 for 循环就够，几十行；提前上框架只会让 79 行的逻辑变成 300 行框架加 79 行逻辑。

升级触发条件：偏好规则超过 **5 条**，或开始出现"规则之间互相覆盖、需要排优先级"时，再把 `prefer` 一列换成数据驱动的门限表（条件 + 结论 + 标定出处 + 生效日期）。在那之前保持代码形态。

#### 5.13.7 一条纪律：路由函数无副作用

路由只读 `FagTilingInfo` 的解析字段、只写决策字段，**绝不修改解析字段**。arch35 的 `SetSplitAxis` 违反了这条（改 `layoutType` / `isDeterministic`），那是最难查的一类问题。建议用类型强制而非注释约束：

```cpp
FagRouteResult DecideRoute(const FagParsedInfo &parsed, FagRouteTrace &trace);
```

让解析结果在路由阶段是 `const`，签名本身就挡住了副作用。

#### 5.13.8 怎么知道架构没被腐蚀

这类设计的风险是纪律松掉后慢慢退回原样。四个可观测信号：

- **能力谓词的改动频率**。一次纯性能标定若需要改 `Feasible_*`，说明有能力约束被写进了偏好里（或反之），当场拆开。这是最灵敏的指标。
- **基线 diff 与预期的偏差次数**。经常"改一条规则却动了一片无关 shape"，说明候选之间条件重叠，应重新划分候选而不是继续加门限。
- **偏好规则条数**。触及 5 条即按 §5.13.6 升级。
- **`kRoutes` 行数与 `is_valid()` 可行组合数的对应关系**。失配即为漏路由（kernel 有模板但 host 永不选）或死候选（host 有候选但没编二进制）。

---



## 6. 不变量、回归基线与对应的 UT

FAGrad 的正确性依赖 host / AICPU / pypto **三方对同一批公式的一致理解**，这是它与 arch35（host 单方说了算）最危险的差别。建议把不变量写进代码注释**并**用 UT 钉住 —— 当前 `attention/flash_attn_grad/` 下**没有 tests 目录**，这是最该补的缺口。

A/B/C 三条是**不变量**（违反即为 bug）；D 是**回归基线**（不判对错，只让改动的影响面可见）。

### A. host tiling 不依赖输入数值

`IMPL_OP_OPTILING(FlashAttnGrad)` 不声明 `TilingInputsDataDependency`，而 arch35 声明了 `{12,13,14,15,16}`。好处很实在：host 不需要同步等待设备数据、图模式下不会因值变化而重新 tiling。

**保持方法**：任何需要 per-batch 实际序列长度的策略，一律放 AICPU。若某天确实需要在 host 拿值，必须是一次显式的架构决策（要同时评估 data dependency 带来的编译/下沉代价），不能顺手加。

**UT**：断言注册项没有 data dependency（或至少在 review checklist 里固化这一条）。

### B. host 的核数与 AICPU 的 `blockOuter` 同源

host `flash_attn_grad_block_outer.h` 里的 `FagS1S2Outer` / `FagDenseBlockOuter` / `FagUsedCubeCores` 是 AICPU `InitFagBaseDims` / `DoFagDenseSplit` / `DoFagBn2DenseSplit` 的**手工镜像**，注释里已经写明"must match"。同时 host 采取了保守策略：

```cpp
useActualCores = (layout != 1) && (maskMode == 0) && !hasSeqused;   // 仅 dense 非 TND 才精确算
usedCube = useActualCores ? FagUsedCubeCores(...) : aicNum;          // 其余全片启动
```

即**允许多发、禁止少发**（多发的核 `start==end` 自然 no-op）。这条"单向安全"要在注释里写成不变量，因为它决定了将来加 sparse 路径时的默认行为应是"退回全片"而不是"猜一个更小的值"。

**建议**：把镜像公式收敛到**一个头文件**，host 与 AICPU 同时 include，从"两份代码手工对齐"变成"一份代码两处包含"。AICPU 侧目前还有 `SetFagSplitAxis` 与 host `SelectSplitAxis` 的重复（BN2 门限 128/640/BN 256/128 三处各写一遍），同样应当收敛。

注意这与 D3「不与正向共代码」不冲突：这里共享的两方是 **FAGrad host 与 FAG metadata AICPU**，都属反向链路，且分核公式本身要求它们逐位一致（不一致就是 bug），与正向 `FlashAttn` 无关。放置位置建议 `attention/flash_attn_grad/op_host/flash_attn_grad_split_policy.h`（归属清晰，由 AICPU 反向 include），而不是 `attention/common/`——后者容易被误当成公共设施而被正向引用。

**UT**：对 (B, N1, N2, S1, S2, D, Dv, layout, mask_mode) 做一遍笛卡尔扫描，同时调用 host 与 AICPU 两个实现，断言
`hostUsedCube == aicpuBlockOuter`（dense 路径）或 `hostUsedCube >= aicpuBlockOuter`（其余路径）。

### C. host 与 pypto 的三份契约一致


| 契约                  | host 侧                 | pypto 侧                                 | 现状                                    |
| ------------------- | ---------------------- | --------------------------------------- | ------------------------------------- |
| TilingData 字段序      | `FillTilingData` 逐字段赋值 | `@dataclass FlashAttnGradTilingData`    | 人工对齐，见 §5.7                           |
| TilingKey 位域 + 合法组合 | `GenTilingKey` 位拼装     | `FlashAttnGradTilingKey` + `is_valid()` | 人工对齐                                  |
| Workspace 分区        | `SetWorkspace` 算总和     | `make_ptr` 分区                           | 人工对齐，见 §5.6                           |
| swizzle 判据          | `DecideSwizzle`        | `decide_swizzle()`                      | **同一判据两份实现**，注释写"Must stay identical" |


`decide_swizzle` 尤其危险：它是一段带 29 组实测标定的分段阈值（`s1Heavy ? 1.25×L2 : 2×L2`，外加方向判据与列数下限），host 一份 C++、kernel 一份 Python。任何一侧改动都会静默产生"host 发 swizzle=1 但 kernel 按 linear 算区间"这类错分核。

**UT**：

1. 遍历 tilingkey 的全部 2^9 组合，断言 host 能产生的组合 ⊆ pypto `is_valid` 为真的组合；
2. 对同一批 shape 分别跑 host `DecideSwizzle` 与 pypto `decide_swizzle`，断言逐点相等（可用 pytest 直接 import py 侧函数，host 侧经 UT 桩暴露）；
3. workspace：断言 host 各分区 offset 与 kernel `make_ptr` 偏移相等。

### D. 路由黄金基线

不是不变量，而是**让路由改动的影响面变成可 review 的 diff**。设计动机见 §5.13.4。

**形态**：一张随代码入库的 shape sweep 表，覆盖 (B, N1, N2, S1, S2, D, Dv, layout, mask_mode) 的笛卡尔子集，每行记录 host 的完整路由输出：

```
b,n1,n2,s1,s2,d,dv,layout,mask, tmpl,multiblk,swizzle,dAlign,dvAlign,tilingKey,blockDim,route
2,32,32,8192,8192,128,128,BSND,0,  0,0,1,128,128,0x...,36,GS1S2_swizzle
2,8,8,1024,8192,128,128,BSND,0,    0,0,0,128,128,0x...,36,GS1S2_linear
4,16,16,128,128,64,64,BSND,0,      1,0,0,64,128,0x...,...,BN2_single
```

`route` 列记的是 §5.13.2 里命中的候选名，比裸 tilingkey 更可读；两者都留，便于对照。

**用法**：改任何路由规则后重跑 sweep，`git diff` 直接给出哪些 shape 的路由变了。配一个 `--update-baseline` 开关更新表，**规则改动与基线更新必须在同一个 PR**，让 reviewer 同时看到"改了什么规则"和"影响了哪些 shape"。

**与 arch35 测试策略的区别**：arch35 是 25 个以上手写 `TEST_F` 的单点断言，覆盖范围取决于"写测试的人想到了哪些 shape"，失败信息是"某个 case 崩了"。基线是全扫描 + 差异集合，覆盖不依赖人的想象力，失败信息是"这些 shape 的路由变了"。两者不互斥：单点 `TEST_F` 继续用于校验拦截与边界报错，基线专门守路由。

**选点建议**：S 取 128/512/1024/2048/8192（覆盖 BN2 单块、MultiBlk 640 门限、swizzle 拐点）；N1/N2 组合覆盖 MHA 与 GQA（G=1/2/8）；D/Dv 取 64/128/192 与 D≠Dv；B 取 1/2/4 触发 BN 门限两侧。规模控制在千行量级，跑一次是秒级，可进 CI。

---



## 7. 迁移路线

每个阶段都以"可编译 + 可测"收口，且**不跨阶段混改**。


| 阶段              | 内容                                                                                                                        | 完成判据                                                    |
| --------------- | ------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| **P0** 落地分层     | cherry-pick `feat/pr9617_followup` 的 `aa1be54a5`（父提交即当前 HEAD，可干净落地）；cherry-pick 前先 `git fetch` 确认远端没有更新版本（该分支标着 `ahead 1, behind 2`） | 现有 case 的 tilingkey / blockDim / workspace 与 pick 前**逐位相同** |
| **P0.5** 决策落码   | D1 自持 `FagTilingBase`；D2 确定性拒绝 + UT；D5 `template=2/3` 兜底断言；§5.12 的对照注释与骨架映射表；§5.13 的候选表与 trace（能力/偏好分离、路由函数改为无副作用签名） | `deterministic=1` 用例必失败；每个新文件头都有 arch35 出处注释；日志能打出命中哪个候选、其余为何被淘汰 |
| **P1** 契约固化     | 建 `tests/ut/op_host/`；落地 §6 的 A/B/C 三组 UT **与 D 路由基线**；抽 `flash_attn_grad_split_policy.h` 供 AICPU include；补 CompileInfo 与 L2 取值（**py 侧 `L2_CACHE_BYTES` 同步改为入参**）；处理 §5.7 的 TilingData 双声明 | 三组 UT 全绿；基线表入库且 CI 可跑；`L2_CACHE_BYTES` 硬编码消除          |
| **P2** 校验层重整    | `CheckShape` 拆分 → `checkers/`（代码自持，见 D3）；补正反向口径对照表                                                                     | 校验逻辑无重复解析；错误信息带 reason；对照表随改动更新                        |
| **P3** 模板层与 TND | 引入注册表 + `IsCapable` 路由；新增 `FlashAttnGradTilingVarlen`（TND / seqused）；空 tensor 分支显式化                                       | TND case 走 varlen 模板；BSND/BNSD 行为不变                     |
| **P4** sparse   | `mask_mode=3/4` 接入（host 只做 tilingkey 与全片启动；枚举在 AICPU）；`sinks` 接入                                                          | causal / window case 精度通过；host 侧无 sparse 枚举代码           |
| **P5** 扩展       | 确定性模板（§5.10）、BN2S2（`template=3`）、更多 D 档                                                                                   | 每项独立模板类 + 独立 UT                                         |


---



## 8. 明确"不要迁"的清单

分两类：8.1 是**用不上的功能资产**（接口已缩减或职责已下移）；8.2 是**不该带过来的代码写法**（arch35 现存的实现问题，逐条经过代码核查）。

### 8.1 不迁的功能资产


| arch35 资产                                                                  | 不迁的理由                                    |
| -------------------------------------------------------------------------- | ---------------------------------------- |
| `FuzzyBaseInfoParamsRegbase`（上百字段）                                         | 其中约 2/3 字段对应已删除的接口；FAGrad 用 30 字段 POD 足够 |
| `blockStarts/Ends[36]`、`GetSparseBlockInfo`、`GetSparseUnpadBlockInfo`      | 已在 AICPU 实现，host 重做会立刻违反不变量 B            |
| `deterPrefix[132]` × 5、`TndBaseInfo` 的 129 项前缀和                            | 同上，属于分核数据，归 AICPU                        |
| 1443 行 `ASCENDC_TPL_ARGS_DECL` / `ASCENDC_TPL_SEL`                         | pypto `is_valid()` 已承担剪枝；两份白名单必然漂移       |
| `PseShapeType` / `PseType` / `AttenMaskCompressMode` / `SparseMode`(9) 全枚举 | FAGrad 只有 `mask_mode ∈ {0,3,4}`          |
| `DtypeEnum` 的 FP32 / FP8 / HIFP8 档                                         | 接口只有 FP16 / BF16                         |
| `ProcessQuantInfo` / rope / dpse / `softmax_max`+`sum` 相关                  | 接口已无对应参数                                 |
| `RunEmptyTiling`（arch22 版）                                                 | 只需 regbase 一份，且 FAGrad 建议先用轻量分支（§5.9）    |


### 8.2 不迁的代码写法（arch35 现存问题）

迁移时最容易发生的事，是把参考实现的**写法**一起复制过来——因为"原来就是这么写的"是最省事的理由。下表逐条给出 arch35 里不该带过来的写法，均已核对到具体位置。表中 `normal` = `op_host/arch35/flash_attention_score_grad_tiling_normal_regbase.{h,cpp}`，`common` = 同目录 `..._common_regbase.{h,cpp}`。

| # | 反模式 | arch35 实证位置 | PTO 侧要求 |
| --- | --- | --- | --- |
| 1 | **非 void 虚函数给空实现**，无 return 语句 | `normal.h:95,96,104,106`：`virtual ge::graphStatus GetSparseUnpadBlockInfo() {};`、`virtual bool IsValidUnpad(int64_t) {};` 等 | 基类不该有"假默认实现"。要么 `= 0` 纯虚，要么显式 `return ge::GRAPH_FAILED;`。见下方重点 1 |
| 2 | **零值边界未守卫**：`s2Outer==0` / `fusedOuter==0` / `blockOuter==0` 均可导致越界或除零 | `normal.cpp:971` 读 `parseInfo[s2Outer-1]`；`975→976` 用 `blockFactor` 作除数；`1002` 写 `blockEnds[blockOuter-1]` | 每个分核入口先判零再算。见下方重点 2 |
| 3 | **裸 `new[]` / `delete[]` 管理二维数组** | `normal.cpp:969、1988` new，`1035、2052` delete | 用 `std::vector` 或固定大小 `std::array`；PTO 侧本来就不该在 host 做这类枚举（不变量 B） |
| 4 | **host tiling 里跑重量级模拟** | `normal.cpp:1376、1419` 的 `while (calcNum < maxValidBBLen)` × `blockOuter` 双层循环 + `std::set` 插入 | 判据必须是闭式公式。见下方重点 3 |
| 5 | **固定容量数组绑死硬件核数**，且写入无边界检查 | `common.h:357-361、437-447、476-478` 共 11 个 `[CORE_LIST_NUM]`（=36）数组；`normal.cpp:999-1030` 局部同款 | PTO 侧 host 不存分核表（不变量 B）。若确需定长数组，容量必须来自平台查询而非手写常量，并断言上界 |
| 6 | **路由函数带副作用**，修改与路由无关的字段 | `common.cpp:1682-1712` `SetSplitAxis` 内改 `isDeterministic`、`layoutType` | 路由用 `const FagParsedInfo &` 入参 + 返回决策 struct，签名上禁止副作用（§5.13.7） |
| 7 | **用 `IsCapable()` 做特性拦截**而非路由 | `normal.cpp:800-806`：`strcmp(tndSoftmaxIn, "") != 0` 即 `return false`；且 `GetAttrPointer` 返回值未判空即传入 `strcmp` | 不支持的参数应在 checker 里报明确错误。让它走到"no valid template is found"会把参数问题伪装成内部错误 |
| 8 | **宏做成员赋值** | `normal.h:28-44` 的 `TND_TILING_DATA_COMMON_ASSIGN` / `BASE_TILING_DATA_COMMON_ASSIGN` do-while 宏 | 用函数或构造函数；宏不可调试、不参与类型检查 |
| 9 | **手写 get_/set_ 样板** | `op_kernel/arch35/*_tiling_data_regbase.h` 共 1504 行，绝大部分是每字段一对访问器 | PTO 的 TilingData 由 pypto codegen（§5.7），不要手写第二份访问器 |
| 10 | **复制粘贴 + 后缀区分**代替参数化 | varlen 1873 行大量镜像 normal：`GetSparseUnpadBlockInfo` / `CheckUnpadSparseLeftAndRight` / `IsValidUnpad` 对应无 `Unpad` 版 | TND 与非 TND 的差异应作为参数（实际序列长度来源）而非复制一套实现；PTO 侧此类差异已在 AICPU 内收敛 |
| 11 | **平台常量硬编码回退** | `flash_attention_score_grad_tiling.cpp:81-82` `aicNum = 40; // B3 default`、`151-152` `aicNum = 32; aivNum = 64; // A5 default` | 平台信息一律走 CompileInfo（§5.8）；拿不到就报错，不要用默认值继续算出一个错的 blockDim |

#### 重点 1：非 void 虚函数空实现是未定义行为

`virtual bool IsValidUnpad(int64_t blockIdx) {};` 声明返回 `bool` 却没有 return 语句。只要基类版本被调用一次就是 UB，实际拿到什么值取决于编译器。这类"占位默认实现"在 arch35 靠"派生类一定会重写"来保证安全，是一个没有编译期保障的约定。

PTO 侧如果继续用继承（D1 的 `FagTilingBase`），基类的抽象方法必须 `= 0`；如果某个钩子确实允许不实现，返回类型就应该表达"未实现"这个状态（显式返回失败码），而不是留空。

#### 重点 2：零值边界，AICPU 移植版已经补过守卫

这组问题有个很有说服力的旁证：AICPU 的 `flash_attn_grad_metadata_split.h` 在移植同一批逻辑时**主动补上了守卫**——`DoFagSparseBlockInfo` 开头 `if (fagS2Outer_ == 0) { fagBlockOuter_ = 0; ... return; }`，`DoFagDenseSplit` 里 `if (fusedOuter == 0) { ... return; }`。也就是说移植者当时就发现原版这里缺判断。

arch35 的三处具体风险：`s2Outer == 0` 时 `parseInfo[s2Outer - 1]` 是越界读；`fusedOuter == 0` 时 `blockFactor` 为 0，下一行拿它做除数；`blockOuter == 0` 时 `blockEnds[blockOuter - 1]` 是对下标 -1 写入。三者都发生在"全部块被 mask 掉"或空 shape 这类边界输入上，正好是 §5.9 要显式处理的场景。

PTO 侧要求：分核相关入口一律先判零返回，且这条要有对应的空/零长用例（§5.9）。

#### 重点 3：判据不能是模拟

`CheckExceedL2Cache()` 用"逐块走位 + `std::set` 去重累加字节"来估算 L2 工作集，循环规模是 `maxValidBBLen × blockOuter`，而 `maxValidBBLen = blockFactor = ceil(fusedOuter / aicNum)`。以 `B=2, N=32, S1=S2=8192, D=128` 估算：`s1Outer = s2Outer = 64`，`fusedOuter = 2×32×64×64 ≈ 262144`，36 核下 `blockFactor ≈ 7282`，于是内外层合计约 26 万次迭代，每次还带 `std::set` 插入——这是 host tiling 每次调用都要付的成本。

除了慢，它还有个结构性问题：它必须读 `blockStarts/Ends` 与 `IsValid()`，而这些在 PTO 侧已经归 AICPU（不变量 B）。所以这条不是"可以优化"，而是**架构上不可迁**——这也是 §1 结论里"判据保留 FAGrad 闭式版本"的由来。

> 通用原则：迁移时对每一段参考实现问一句"这个写法是**设计**还是**历史**"。上面 11 条都属于后者，复制过来只会把技术债一起继承。


---



## 9. 决策记录

2026-09-08 定稿 4 项，1 项暂定。

| # | 议题 | 决策 | 落点 |
| --- | --- | --- | --- |
| **D1** | 基类选型 | 采用 `FiaTilingBase` **形态**（三步 + POD 传入 + `Set*` 写回），但在本算子目录自持一份 `FagTilingBase`，**不 include 正向那份**；同时须保持与 arch35 的可对照性 | §5.4、§5.12 |
| **D2** | 确定性计算 | **当前拒绝**：checker 读 `GetDeterministic()`，为 1 时 `OP_LOGE` + `GRAPH_PARAM_INVALID`，并配"必须失败"的 UT | §5.10 |
| **D3** | checker 与正向共用 | **一定不共代码**：不抽 `attention/common/`、不 include 正向，本算子复制一份；靠对照注释 + 口径对照表防漂移 | §5.3 |
| **D4** | `sinks` 是否占 tilingkey 位 | **暂定**，待 pypto 侧 sinks 实现方式确定后再定；在此之前 host 不预留位、不预解析 | §5.11 |
| **D5** | BN2S2（`template=3`） | **保留位，勿复用**；`template=2` 同为未用。本轮不实现不编译，host 加兜底断言确保永不产生 | §5.5 |

D1 与 D3 合起来是一条统一的边界：**与正向 `FlashAttn` 只共享"形态"，不共享"代码"。** 参照其文件组织、类拆分、命名习惯，但实现落在本算子目录内。这样做的代价是重复实现带来的漂移风险，因此 §5.3 的对照表和 §5.12 的溯源注释不是可选项 —— 它们是这条边界的配套成本，必须一起落地。

D4 是唯一还开着的问题，判据已经明确（改变片上 tile 形状的才占 tilingkey 位），只差 kernel 侧的输入。

