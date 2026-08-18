# BSA MXFP8 全量化设计说明（kvs128 修订版）

日期：2026-08-17；板上修订：2026-08-18  
基线：`mxfp8_0814` 上已有 MXFP8 脚手架；本文件**取代** `2026-08-14-bsa-mxfp8-design.md` 的几何与内存决策。  
状态：**官方 qs128 套件精度已过线**（分支 `mxfp8_0817`；几何仍为本修订；板上修正见 §15）  
画布：Cursor `BSA-MXFP8-design-report.canvas.tsx`；仓库副本 `docs/superpowers/canvases/BSA-mxfp8-design-report.canvas.tsx`

## 1. 目标

在 BSA MXFP4 全量化路径旁共存 MXFP8 OCP 全量化，由 tiling key 选择。

相对 0814 稿的核心修订：

| | 0814 稿 | **本修订** |
| --- | --- | --- |
| 单次 QK → L0C | 128×128 | **64(kvs)×128(qs)** |
| Softmax 基块 | 256×128（两轮拼） | **128×128**（两轮 64 拼） |
| 组 G | 4（1024×128） | **8**（仍覆盖 **1024×128**） |
| `kvBaseTile` | 256 | **128** |
| Dpv / P 环 | 8 | **12** |
| UB P | 半基块双缓冲 | **整基块双缓冲** |
| L0 切 k | QK 切 D + PV 切 S2 | **默认不切**（见 §5） |

不变：

- `qs` 基块 128，D = 128  
- Q/K/V/P = `e4m3fn`，scale = `e8m0`，MX 组 32，**仅 OCP**  
- 操作数角色 **K→L0A、Q→L0B**（不可对调）  
- Softmax：**低精**（S = fp16）；rescale / oTmp：**fp32**

## 2. 术语

| 术语 | 含义 |
| --- | --- |
| QK 半面 | 一次 Cube mmad + fixpipe：`64×128` fp32→fp16 |
| 基块（base tile） | 一次 softmax / 一个 L1 P 槽单位：`128(kvs)×128(qs)`，由两轮 QK 半面拼成 |
| 组（tile group） | L0C 累加窗口 + group-max / pScale 作用域；**G = 8** 个基块，覆盖 S2 合计 1024（与 0814 的 256×4 同量级） |
| loop | 主循环全局基块序号 |
| Dpv | `PV_RE_DELAY_N`，PV 落后 softmax 的基块数 |
| G | `TILE_GROUP_N` |

数据流：

1. Cube 两次 QK（各 64 列 kvs）→ 同一 AIV 的 UB S（128×128 fp16）  
2. 该 AIV 整块 softmax → 整块 P（16KB）→ L1 槽 `loop % 12`  
3. 基块按 `loop % 2` 交替 AIV0/AIV1；一组 8 块中每 AIV 各 4 块；组尾经 L1 交换 localGlobalMax  

## 3. 流水与延迟

### 3.1 下界

同拍顺序仍为 **PV → pScale → 写 P**。组末块在 `gG+G-1` 出 S，再 `DELAY_PSCALE_N=3` 拍后 pScale 就绪；PV 消费组首需满足：

```
Dpv >= G + 3
```

G = 8 → 下界 **11**。取 **Dpv = 12**（+1 余量且为偶数，两 AIV 各 6 个 P 槽）。

环深 = Dpv：槽 `loop % 12` 在 `loop+12` 被 PV 读走；同拍先 PV 后写，**不需要 13 槽**。

### 3.2 常量

| 常量 | MXFP4 | 0814 MXFP8 | **本修订** |
| --- | --- | --- | --- |
| `TILE_GROUP_N` | 16 | 4 | **8** |
| `PV_RE_DELAY_N` | 20 | 8 | **12** |
| `DELAY_PSCALE_N` | 3 | 3 | 3 |
| `PRELOAD_TASK_CACHE_SIZE` | 21 | 9 | **13** |
| `L1_P_BUF_CNT` | 20 | 8 | **12** |
| `KVS_BASE_SIZE` | 256 | 256 | **128** |
| `QS_BASE_SIZE` | 128 | 128 | 128 |
| `UB_LOCAL_GROUP_MAX_CNT`（1K） | 20 | 8 | **12** |

组首/组尾频率：每 8 个基块一次（介于 MXFP4 的 16 与 0814 的 4 之间）。

### 3.3 尾部

`while` 条件中 delay 游标改为 delay12 / delay3；kv 基块数 &lt; 12 的短序列由同一 drain 逻辑覆盖。

## 4. L1 分配（512 KB）

规则：数据槽按 fp8 字节计；scale 分形与位宽无关（`yStep` 公式同 0814）。  
**KV 单槽仍取 32 KB**（Vᵀ+rowsum seed 镜像 18 KB &gt; 16 KB，不能把 KV 缩到 16 KB）。

| 区 | 偏移 | 单槽 | 槽数 | 合计 | 说明 |
| --- | --- | --- | --- | --- | --- |
| P 数据 | 0 | **16 KB** = 128×128×1B | **12** | **192 KB** | `loop % 12` |
| P-scale | 192 KB | **768 B** = 128×2×(128/64+1) | 12 | ≈9 KB | 随 kvs 变小 |
| Q 数据 | ≈201 KB | 16 KB | 2 | 32 KB | 128×128×1B |
| Q-scale | | 512 B | 2 | 1 KB | |
| KV 共享池 | | **32 KB** | 4 | 128 KB | 数据仅需 16KB；余量给 seed |
| KV-scale | | 2 KB | 4 | 8 KB | |
| localGlobalMax | | 512 B | 4 | 2 KB | |
| 空闲 | | | | **≈140 KB** | 多于 0814 |

## 5. L0 分配（各 64 / 256 KB）

基块 S2 = 128 后，**默认不再强制双切 k**：

| 操作数 | 几何 | 单槽 | 策略 |
| --- | --- | --- | --- |
| K | 64×128×1B | 8 KB | 双缓冲 |
| Q | 128×128×1B | 16 KB | **单缓冲**（避免 L0B 顶满） |
| Vᵀ | 144×128×1B | 18 KB | 双缓冲，**不切 S2** |
| Pᵀ | 128×128×1B | 16 KB | 双缓冲 |
| L0C QK | 64×128×4B | **32 KB** | 双缓冲 |
| L0C PV | 144×128×4B | 72 KB | 单缓冲 |

推荐布局：

| | 偏移 | 合计 |
| --- | --- | --- |
| L0A：K 2×8 + Vᵀ 2×18 | 0 / 16 KB | **52 KB**（余 12） |
| L0B：Q 1×16 + Pᵀ 2×16 | 0 / 16 KB | **48 KB**（余 16） |
| L0C：QK 2×32 + PV 72 | 0 / 64 KB | **136 KB**（余 120） |

每基块 mmad：**2 QK + 1 PV = 3 条**（回到 MXFP4 节奏；0814 为 6 条）。

可选退路（L0B 双缓 Q 时）：Q 沿 D 切 2×64 → Q 槽 8KB×2，L0B 仍 48KB，QK 变 4 条 mmad。

角色约束不变：不能把 Q 放 L0A。

## 6. UB 分配（256 KB）

S 变为 128×128 fp16 → 单槽 32KB；相对 0814 的 128KB S **省 64KB**，可恢复整块 P。

| 区 | 大小 | 说明 |
| --- | --- | --- |
| S | **64 KB**（2×32） | fp16，128×128，256B 交错 |
| oTmp | 32 KB | fp32 |
| rowSum | 512 B | |
| P | **32 KB**（2×16） | **整基块**双缓冲 |
| oTrans | 18 KB | 复用 P 区 |
| pScale | 12×768 B ≈ 9 KB | 复用 P 区尾 |
| attentionOut | 32 KB | |
| peer / softmaxMax | ≈1.25 KB | |
| localGroupMax | **12 KB** | = Dpv（1K 计） |
| localGlobalMax / dm / index | ≈2.25 KB | |
| **合计** | **≈185 KB** | 余 ≈70 KB |

## 7. 数据类型与量化

与 0814 相同：GM e4m3/e8m0；S fp16；P e4m3；P-scale e8m0；L0C fp32；Fixpipe **NO_QUANT**；seed `0x38`×`0x7F`。  
C0：`FP8_C0_ELEMS = 32`。  
组覆盖仍为 1024 S2，精度窗口与 0814 同量级（优于 MXFP4 的 4096）。

## 8. 主机侧

- `QuantMode::MXFP8_OCP_QUANT`、dtype、tiling key、分发：同 0814 方向。  
- **`CalcBaseTileTilingParams950`：MXFP8 强制 `kvBaseTile_ = 128`**（不再 256）。  
- sparse `blockShapeY` 对齐改为 **128**（或 64 的偶数倍）；workspace/idx 按新基块重算。  
- 验收尾块：kvs = 64 / 128 / 非对齐。

## 9. 文件清单

在 0814 已列 mxfp8 文件上**改几何与循环**，而非再开一套命名：

- 常量表：`KVS_BASE=128`，P 环 12，L0C QK 32KB，去掉默认 `QK_K_SPLIT`/`PV_K_SPLIT`  
- QK：`n` 方向按 64 出两轮，拼进 128 行 S  
- PV：单次 k=128；P 槽 `% 12`  
- Softmax VF：整块 128×128；删除半块 P 双槽语义  
- Kernel：`PV_RE_DELAY_N=12`，`TILE_GROUP_N=8`  
- Host：`kvBaseTile=128` + 对齐校验  

## 10. 风险

1. **Vᵀ dstStride 奇**（rowsum pad → 9）：仍在；兜底 pad 到 160，L0A 56KB / L0C 208KB 仍可。  
2. **Dpv=12**：PV 更滞后；同覆盖 1024 S2 时 **loop 次数约为 0814 的 2 倍**（基块更碎），Cube/同步开销需实测。  
3. **Q L0B 单缓冲**：Q 与 mmad 重叠变弱；若成为瓶颈再启用 Q 切 D 双缓冲。  
4. **与已实现的 256/G=4 脚手架不兼容**：需改常量与主循环，不能只热切换。  
5. P-scale **768 B** 分形需板上确认。

## 11. 验收

- L1/L0/UB `static_assert` 通过。  
- Golden：与 MXFP4 同口径（`rtol=1/128`、`atol=1e-4`、满足率 ≥99.5%、`maxRE<10`）。  
- **已过线（2026-08-18，dump-off，`cann_bsa_0817`）**：qs128 的 kvs64 / 128 / 160 / 192 GQA / 256 dense / 256 sparse；**qs64 / kvs128**。  
- **套件外未达标**：单 tile `s2=32`、`s2=96`（上次约 98.91% / 99.15%）。  
- 性能：对比 MXFP4 仍未测；Q 切 D 未启用。

## 12. 实施分期

| 期 | 内容 | 状态 |
| --- | --- | --- |
| 1 | 常量 + tiling：`kvBaseTile=128`、G=8、Dpv=12、三张内存表 | **已完成** |
| 2 | Cube：QK 64×128×2、PV 单次 k=128、L0 推荐布局 | **已完成** |
| 3 | Vec + 流水：整块 softmax/P、group-max、delay 12/3 | **已完成** |
| 4 | VF e4m3 打包 + pScale 环/分形与 kvs128 对齐 | **已完成（板上已确认）** |
| 5 | 调用链：aclnn/Torch `quantMode=4` | **已完成** |
| 6 | 板上：qs128 精度 / 尾块 / sparse | **已完成** |
| 7 | qs64 tiling + QK pad-to-128 + qs128 VF | **已完成** |
| 8 | 套件外 s2=32/96 + 性能 / 可选 Q 切 D | **待做** |

## 13. 与 0814 / MXFP4 对照摘要

| | MXFP4 | 0814 MXFP8 | **本修订** |
| --- | --- | --- | --- |
| 基块 | 256×128 | 256×128 | **128×128** |
| 组 | ×16 | ×4 | **×8** |
| Dpv | 20 | 8 | **12** |
| 基块 mmad | 3 | 6 | **3** |
| UB P | 整块 | 半块 | **整块** |
| L1 P 字节 | 320KB | 256KB | **192KB** |

## 14. 修改点说明（0817 落地）

相对 `mxfp8_0814` 脚手架（256 / G=4 / Dpv=8 / 双切 k / 半块 P），本轮在 `mxfp8_0817` 上的代码改动如下。

### 14.1 常量与内存表

| 文件 | 改动 |
| --- | --- |
| `op_kernel/arch35/kernel_utils.hpp` | `MXFP8Kernel::TILE_GROUP_N`：4 → **8** |
| `op_kernel/.../gemm/block/block_mmad_arch35_utils.hpp` | `KVS`/`S2`=128；`QK_HALF_KVS`=64；去掉 `QK_K_SPLIT`/`PV_K_SPLIT`；L0B_Q 单缓 16KB；L0C QK 32KB×2，PV 偏移 **64KB**；L1 P 环 **12×16KB**，P-scale **12×768B** |
| `op_kernel/.../epilogue/block/block_epilogue_arch35_utils.hpp` | `KVS_BASE=128`；S **2×32KB**；oTmp@64KB；P 整基块 2×16KB；pScale/localGroupMax **12**；UB 偏移按新表重排 + `static_assert` |

### 14.2 Cube / Kernel / 接口

| 文件 | 改动 |
| --- | --- |
| `.../gemm/block/block_mmad_qk_arch35_mxfp8.hpp` | n 向按 `L0_TILE_N(=64)` 两轮拼基块；D 一次打满；Q→L0B 槽 0 |
| `.../gemm/block/block_mmad_pv_arch35_mxfp8.hpp` | 去掉 S2 k-split 循环；单次 `LoadP/V` + `MatmulPV`；P 槽 `% 12` |
| `.../arch35/block_sparse_attention_kernel_arch35_mxfp8_full_quant.h` | `PV_RE_DELAY_N`：8 → **12**（`PRELOAD=13`） |
| `.../block_sparse_attention_kernel_interface.cpp` | `L0TileShapeQK`：`<128,128,128>` → **`<128,64,128>`** |

### 14.3 Softmax / Host

| 文件 | 改动 |
| --- | --- |
| `.../block_epilogue_online_softmax_arch35_reg_low_prec_fp16_mxfp8.hpp` | 注释改为 kvs=128；UB→L1 NZ 布局按 **整基块 16KB**（`UB_P_NZ_M=64`，不再用 `KVS/4`） |
| `op_host/block_sparse_attention_tiling.cpp` | `CalcBaseTileTilingParams950`：MXFP8 强制 `kvBaseTile_=128`；`CheckBlockShapeQuantConstraint`：MXFP8 的 `blockShapeY` 改为 **128 倍数** |

### 14.4 期4 代码收尾（相对 14.1–14.3）

| 文件 | 改动 |
| --- | --- |
| `block_sparse_attention_kernel_arch35_mxfp8_full_quant.h` | `delay8*` → **`delay12*`**；`pscaleNum_` 环 **`% 12 / 6`**，初值 11 |
| `block_mmad_pv_arch35_mxfp8.hpp` | 形参 `delay8*` → `delay12*` |
| `block_epilogue_compute_pscale_arch35_mxfp8.hpp` | `GROUP_MAX_SPACE_LEN=12`，`WHOLE_PROCESS_LOOP=8`；每基块 1 槽（不再 ×2） |
| `block_epilogue_online_softmax_..._mxfp8.hpp` | `GetLocalGroupMaxBufIdx`：`loop/2 % 12`（1KB/tile） |
| `vf_computeScale_dn_mxfp8.h` / `_qs64` | P-scale **yStep=3**、stride **768**、每 tile **4** 个 group-max |
| `vf_nd2nz_indexes_dn_mxfp8.h` | gather 索引步长 **×2**（两路 e4m3） |
| `mxfp8_vf/vf_softmax_*` | 四路 fp4 Or → **两路 ZERO/ONE**；qs128 写间距 4096/512 |
| `copy_l1_to_l0_mx_fp8_a5.hpp` | V srcStride 满段判断 **128**（不再 256） |

### 14.5 待板上确认（2026-08-18 已关闭大部分）

原清单已由 §15 实测关闭：kvs 64/128/160/192/256、sparse、**qs64/kvs128**、e4m3 NZ、P-scale 768B 分形均已过线。仍开放：套件外短 kvs、性能 vs MXFP4、Q 切 D。

### 14.6 调用链（quantMode=4）

| 文件 | 改动 |
| --- | --- |
| `op_host/op_api/aclnn_block_sparse_attention_v3.cpp` | `MXFP8_OCP_QUANT=4`：QKV=`e4m3fn`、scale=`e8m0`；scale 维同 mxfp4；innerPrecise=4 / 无 LSE |
| `torch_extension/.../ops/block_sparse_attention.py` | `_QUANT_MODE` 含 4；e4m3 输入 + e8m0 descale；out ∈ {fp16, bf16} |
| `torch_extension/.../ops/csrc/block_sparse_attention.cpp` | QKV 原样传递；descale wrap `ACL_FLOAT8_E8M0`；attentionOut 形状不翻倍 |
| `torch_extension/.../ops/__init__.py` | 导出 `block_sparse_attention_v3` |

## 15. 板上反向分析（2026-08-18）

对照 dump 中间量、FIA/mxfp4 参考和 dump-off 全量 pytest，几何表（§1–§6）仍然成立；原稿未写死的实现选择与若干错误路径，以下为 **as-built 约束**，后续改动不得回退。

### 15.1 精度结果（dump-off，`cann_bsa_0817`，NPU 1）

判据：`rtol=1/128`，`atol=1e-4`，满足率 ≥99.5%，`maxRE<10`。`_E8M0_QK_FLIP_PAIR=False`。`BSA_MXFP8_DUMP=0`。

| case | fulfill | 结果 |
| --- | --- | --- |
| qs128 / kvs128 dense | 99.74% | 过 |
| qs128 / kvs64 dense | 99.79% | 过 |
| qs128 / kvs160 tail | 99.95% | 过 |
| qs128 / kvs256 dense | 99.99% | 过 |
| GQA qs128 / kvs192 | 99.86% | 过 |
| sparse qs128 / kvs256 | 99.64% | 过 |
| qs64 / kvs128 dense | 100.00% | 过 |
| 套件外 s2=32 | ≈98.91% | 未纳入验收 |
| 套件外 s2=96 | ≈99.15% | 未纳入验收 |

安装注意：自定义 opp 装到 `cann_bsa_0817` 后，pytest 必须用 **与 `libcust_opapi.so` 相同的文件所有者** 跑（root 装则 root 跑，再 `chown 1007:1013`）。root 读普通用户文件会被 ACL 跳过该 so。

### 15.2 原稿 vs 落地

| 项 | 0817 原稿 | 板上落地 |
| --- | --- | --- |
| QK 半面 / 基块 / G / Dpv | 64×128 / 128×128 / 8 / 12 | **一致** |
| kvBaseTile | 128 | **一致** |
| 多 tile softmax | 未写死 | **与 mxfp4 同构**：`TILE_GROUP_N=8` + last-tile Vec2；**不要**改成 per-tile online softmax |
| P snap | 未写 −2 vs −8 | `NEG_TWO_VALE`（−2），同 mxfp4；**不要**用 −8 |
| Cube dtype | 未写 LoadData/Mmad 一致性 | LoadData dest **与** Mmad 必须是 `mx_fp8_e4m3_t`。Mmad 仍用 `fp8_e4m3fn_t` 会使 S 溢出 inf → 全 NaN |
| LoadData stride | 未写 | mx 路径 `srcStride = yStep` |
| P NZ 打包 | 「两路 ZERO/ONE」 | **mxfp4 even/odd + VL128**，再按 FP8 C0=32 **K-outer** 平面。4 行连续 load + 4-way Gather；even/odd `j` 偏移见 VF |
| CopyP / LoadP | 未写 NZ 行数 | 64 行 TLA `{M,8,8,0}` 填进 128 行 NZ L1；LoadP `srcStride = s2Base/16 = 8` |
| peer gmax | 「组尾经 L1 交换」 | AIC 上 `GetSubBlockIdx()` **恒 0**，普通 DataCopy 写不到两个 AIV UB。必须 `asc_copy_l12ub_sync`：L1 row0→AIV0、row1→AIV1。UB dest 偏移 **160KB+512**（不是 mxfp4/QFA 的 224KB+512） |
| pScale 槽 | 每基块 1 槽 | kvs128 = **1 slot/tile**。不可套 mxfp4 的 `*2+1` 再 `/2`（那是 2 slot / kvs256） |
| 尾块 pad | 未写 | kvs32 VF 必须把 groupmax pad 满 **4 组**（`computePscale` 固定 `ulmaxLoopRow=m*4`）。`PadPS2NotAlign64` 按 **128 行 × 4 个 FP8 C0 平面** 填 `[align32, align64)` |
| qs64 | 验收覆盖 | host 允许 `blockShapeX=64/128`。QK 将 Q(M) **pad 到 128**（L1 Q/Q-scale `InitConstValue` 清零），Fixpipe N=128；softmax/pscale 走 **qs128 VF**（even/odd + `SPLIT_CONSEC64`）。`qsActBaseTileAlign64` 仍为 64，供 CopyP / PV / rescale。不要用 qs64 VF 的 VL64 直接打 even/odd 128-half（约 3.125%） |
| `processUpdate` 循环长 | 未写 | 仍按 `KvsBase=256` 量级覆盖 mxfp8 32KB O，不要改成 128 |

### 15.3 板上修过的根因（保留）

1. **QK/PV Mmad 类型错误** → S inf。FIA 路径：LoadData mx 且 Mmad `mx_fp8_e4m3_t`。  
2. **P store 密度/平面顺序** → 99.74% 路径：even/odd VL128 + K-outer C0。  
3. **两 tile 独立 softmax**（kvs256 ~1.4%）→ cube `asc_copy_l12ub_sync` 交换 peer gmax 后 99.99%。AIV 侧 DataCopy L1→UB 无效；cube `GetValue` dump 曾导致 507015。  
4. **kvs160 行 NaN**（99.17%）→ kvs32 只 pad 1 个 groupmax，残留 Inf 变成 e8m0=255；PadPS2 曾按 64 行 2 平面写。  
5. **qs64 约 3.125%** → Fixpipe N=64 时 VF 按 QsBase=128 读 even/odd 128-half，VL64 不是连续 qs[0:64)。QK pad N=128 + qs128 VF 后 100%。  

### 15.4 仍开放

- 套件外 `s2=32` / `s2=96`。  
- 性能 vs MXFP4；Q L0B 单缓冲是否瓶颈。

