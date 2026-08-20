# BSA MXFP8 全量化设计说明（kvs128 修订版）

日期：2026-08-17；板上修订：2026-08-18、**2026-08-19（跨组 CopyPScale / layout+fp16out / D=64 / 对标 MXFP4 pytest 1–3）**、**2026-08-20（性能基线；P0 Q 双槽试过 −2% 后回退；S=118800 keep1 vs FIA/MXFP4）**  
基线：`mxfp8_0814` 上已有 MXFP8 脚手架；本文件**取代** `2026-08-14-bsa-mxfp8-design.md` 的几何与内存决策。  
状态：**官方 qs128 套件 + qs64/kvs128 + 跨 TILE_GROUP（s2>1024）+ BSND/TND 冒烟 + fp16out + D=64（已拷中短例）已过线**；**对标 MXFP4 的 OCP pytest 1–3（TND varlen / act_s pad / BSND·TND×D=64·sparse·非对齐 / gap_sparse / scale-stride-trap）已接线**（分支 `mxfp8_0817`；几何仍为本修订；板上修正见 §15 / §16）  
画布：Cursor [`S118800-pipe-compare.canvas.tsx`](/home/npu_user5/.cursor/projects/home-npu-user5/canvases/S118800-pipe-compare.canvas.tsx)、`BSA-MXFP8-vs-FP16-analysis.canvas.tsx`、`BSA-MXFP8-design-report.canvas.tsx`；仓库副本 `docs/superpowers/canvases/`

## 1. 目标

在 BSA MXFP4 全量化路径旁共存 MXFP8 OCP 全量化，由 tiling key 选择。

相对 0814 稿的核心修订：


|              | 0814 稿           | **本修订**                 |
| ------------ | ---------------- | ----------------------- |
| 单次 QK → L0C  | 128×128          | **64(kvs)×128(qs)**     |
| Softmax 基块   | 256×128（两轮拼）     | **128×128**（两轮 64 拼）    |
| 组 G          | 4（1024×128）      | **8**（仍覆盖 **1024×128**） |
| `kvBaseTile` | 256              | **128**                 |
| Dpv / P 环    | 8                | **12**                  |
| UB P         | 半基块双缓冲           | **整基块双缓冲**              |
| L0 切 k       | QK 切 D + PV 切 S2 | **默认不切**（见 §5）          |


不变：

- `qs` 基块 128；**D ∈ {64, 128}**（Cube / rescale / `processOut` 仍按 D=128 槽；D=64 只在 Vᵀ L0 `dstStride` 上 pad 到 144 行，同 MXFP4）  
- Q/K/V/P = `e4m3fn`，scale = `e8m0`，MX 组 32，**仅 OCP**（不做 CX）  
- 操作数角色 **K→L0A、Q→L0B**（不可对调）  
- Softmax：**低精**（S = fp16）；rescale / oTmp：**fp32**



## 2. 术语


| 术语            | 含义                                                                                |
| ------------- | --------------------------------------------------------------------------------- |
| QK 半面         | 一次 Cube mmad + fixpipe：`64×128` fp32→fp16                                         |
| 基块（base tile） | 一次 softmax / 一个 L1 P 槽单位：`128(kvs)×128(qs)`，由两轮 QK 半面拼成                           |
| 组（tile group） | L0C 累加窗口 + group-max / pScale 作用域；**G = 8** 个基块，覆盖 S2 合计 1024（与 0814 的 256×4 同量级） |
| loop          | 主循环全局基块序号                                                                         |
| Dpv           | `PV_RE_DELAY_N`，PV 落后 softmax 的基块数                                                |
| G             | `TILE_GROUP_N`                                                                    |


数据流：

1. Cube 两次 QK（各 64 列 kvs）→ 同一 AIV 的 UB S（128×128 fp16）
2. 该 AIV 整块 softmax → 整块 P（16KB）→ L1 **P 与 P-scale** 槽均为 `loop % 12`
3. 基块按 `loop % 2` 交替 AIV0/AIV1；一组 8 块中每 AIV 各 4 块；组尾经 L1 交换 localGlobalMax  
   UB `localGroupMax` 是每 AIV 自己的环：`loop/2 % 12`。CopyPScale 写 L1 时必须把该下标映射回 `loop % 12`（见 §15.2），不能直接当 L1 行号。



## 3. 流水与延迟



### 3.1 下界

同拍顺序仍为 **PV → pScale → 写 P**。组末块在 `gG+G-1` 出 S，再 `DELAY_PSCALE_N=3` 拍后 pScale 就绪；PV 消费组首需满足：

```
Dpv >= G + 3
```

G = 8 → 下界 **11**。取 **Dpv = 12**（+1 余量且为偶数，两 AIV 各 6 个 P 槽）。

环深 = Dpv：槽 `loop % 12` 在 `loop+12` 被 PV 读走；同拍先 PV 后写，**不需要 13 槽**。

### 3.2 常量


| 常量                           | MXFP4 | 0814 MXFP8 | **本修订** |
| ---------------------------- | ----- | ---------- | ------- |
| `TILE_GROUP_N`               | 16    | 4          | **8**   |
| `PV_RE_DELAY_N`              | 20    | 8          | **12**  |
| `DELAY_PSCALE_N`             | 3     | 3          | 3       |
| `PRELOAD_TASK_CACHE_SIZE`    | 21    | 9          | **13**  |
| `L1_P_BUF_CNT`               | 20    | 8          | **12**  |
| `KVS_BASE_SIZE`              | 256   | 256        | **128** |
| `QS_BASE_SIZE`               | 128   | 128        | 128     |
| `UB_LOCAL_GROUP_MAX_CNT`（1K） | 20    | 8          | **12**  |


组首/组尾频率：每 8 个基块一次（介于 MXFP4 的 16 与 0814 的 4 之间）。

### 3.3 尾部

`while` 条件中 delay 游标改为 delay12 / delay3；kv 基块数 < 12 的短序列由同一 drain 逻辑覆盖。

## 4. L1 分配（512 KB）

规则：数据槽按 fp8 字节计；scale 分形与位宽无关（`yStep` 公式同 0814）。  
**KV 单槽仍取 32 KB**（Vᵀ+rowsum seed 镜像 18 KB > 16 KB，不能把 KV 缩到 16 KB）。


| 区              | 偏移      | 单槽                           | 槽数     | 合计          | 说明                 |
| -------------- | ------- | ---------------------------- | ------ | ----------- | ------------------ |
| P 数据           | 0       | **16 KB** = 128×128×1B       | **12** | **192 KB**  | `loop % 12`        |
| P-scale        | 192 KB  | **768 B** = 128×2×(128/64+1) | 12     | ≈9 KB       | 随 kvs 变小           |
| Q 数据           | ≈201 KB | 16 KB                        | 2      | 32 KB       | 128×128×1B         |
| Q-scale        |         | 512 B                        | 2      | 1 KB        |                    |
| KV 共享池         |         | **32 KB**                    | 4      | 128 KB      | 数据仅需 16KB；余量给 seed |
| KV-scale       |         | 2 KB                         | 4      | 8 KB        |                    |
| localGlobalMax |         | 512 B                        | 4      | 2 KB        |                    |
| 空闲             |         |                              |        | **≈140 KB** | 多于 0814            |




## 5. L0 分配（各 64 / 256 KB）

基块 S2 = 128 后，**默认不再强制双切 k**：


| 操作数    | 几何         | 单槽        | 策略                 |
| ------ | ---------- | --------- | ------------------ |
| K      | 64×128×1B  | 8 KB      | 双缓冲                         |
| Q      | 128×128×1B | 16 KB     | **单缓冲**（P0 双槽已回退） |
| Vᵀ     | 144×128×1B | 18 KB     | 双缓冲，**不切 S2**            |
| Pᵀ     | 128×128×1B | 16 KB     | 双缓冲                         |
| L0C QK | 64×128×4B  | **32 KB** | 双缓冲                         |
| L0C PV | 144×128×4B | 72 KB     | 单缓冲                         |


推荐布局：


|                      | 偏移        | 合计                |
| -------------------- | --------- | ----------------- |
| L0A：K 2×8 + Vᵀ 2×18  | 0 / 16 KB | **52 KB**（余 12）   |
| L0B：Q 16 + Pᵀ 2×16     | 0 / 16 KB | **48 KB**（余 16） |
| L0C：QK 2×32 + PV 72  | 0 / 64 KB | **136 KB**（余 120） |


每基块 mmad：**2 QK + 1 PV = 3 条**。默认不切 D：Q 单缓 16KB，P 从 16KB 起。P0 整段 Q 两槽常驻（L0B 打满 64KB）已于 2026-08-20 回退。沿 D 切 2×64 的退路跨组/GQA 掉点，**未合入**。

角色约束不变：不能把 Q 放 L0A。

## 6. UB 分配（256 KB）

S 变为 128×128 fp16 → 单槽 32KB；相对 0814 的 128KB S **省 64KB**，可恢复整块 P。


| 区                           | 大小              | 说明                   |
| --------------------------- | --------------- | -------------------- |
| S                           | **64 KB**（2×32） | fp16，128×128，256B 交错 |
| oTmp                        | 32 KB           | fp32                 |
| rowSum                      | 512 B           |                      |
| P                           | **32 KB**（2×16） | **整基块**双缓冲           |
| oTrans                      | 18 KB           | 复用 P 区               |
| pScale                      | 12×768 B ≈ 9 KB | 复用 P 区尾              |
| attentionOut                | 32 KB           |                      |
| peer / softmaxMax           | ≈1.25 KB        |                      |
| localGroupMax               | **12 KB**       | = Dpv（1K 计）          |
| localGlobalMax / dm / index | ≈2.25 KB        |                      |
| **合计**                      | **≈185 KB**     | 余 ≈70 KB             |




## 7. 数据类型与量化

与 0814 相同：GM e4m3/e8m0；S fp16；P e4m3；P-scale e8m0；L0C fp32；Fixpipe **NO_QUANT**；seed `0x38`×`0x7F`。  
C0：`FP8_C0_ELEMS = 32`。  
组覆盖仍为 1024 S2，精度窗口与 0814 同量级（优于 MXFP4 的 4096）。

## 8. 主机侧

- `QuantMode::MXFP8_OCP_QUANT`、dtype、tiling key、分发：同 0814 方向。  
- `CalcBaseTileTilingParams950`**：MXFP8 强制** `kvBaseTile_ = 128`（不再 256）。  
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
3. **Q L0B 双缓冲**：P0 整段 Q 16KB×2 常驻已启用后实测仅 −2%，**2026-08-20 已回退**为单缓 16KB。沿 D 切 2×64 退路跨组/GQA 掉点，**未合入**。
4. **与已实现的 256/G=4 脚手架不兼容**：需改常量与主循环，不能只热切换。
5. P-scale **768 B** 分形需板上确认。



## 11. 验收

- L1/L0/UB `static_assert` 通过。  
- Golden：与 MXFP4 同口径（`rtol=1/128`、`atol=1e-4`、满足率 ≥99.5%、`maxRE<10`）。  
- **已过线（2026-08-18 / 19，dump-off，**`cann_bsa_0817`**）**：官方 qs128 套件；**qs64 / kvs128**；**跨组** s2=1056 / 1152 / 1280 / 2048 / 3072 / 16384（含 qs64、GQA、sparse、b=2）；套件外 **s2=32/48/96**。  
- 性能：对比 MXFP4 / FP16 已测（§17）；P0 Q 双槽 −2% **已回退**（§17.4）；S=118800 keep=1.0 vs FIA dense / BSA MXFP4 见 §17.5。



## 12. 实施分期


| 期   | 内容                                            | 状态             |
| --- | --------------------------------------------- | -------------- |
| 1   | 常量 + tiling：`kvBaseTile=128`、G=8、Dpv=12、三张内存表 | **已完成**        |
| 2   | Cube：QK 64×128×2、PV 单次 k=128、L0 推荐布局          | **已完成**        |
| 3   | Vec + 流水：整块 softmax/P、group-max、delay 12/3    | **已完成**        |
| 4   | VF e4m3 打包 + pScale 环/分形与 kvs128 对齐           | **已完成（板上已确认）** |
| 5   | 调用链：aclnn/Torch `quantMode=4`                 | **已完成**        |
| 6   | 板上：qs128 精度 / 尾块 / sparse                     | **已完成**        |
| 7   | qs64 tiling + QK pad-to-128 + qs128 VF        | **已完成**        |
| 8   | 跨 TILE_GROUP：CopyPScale L1 按下标 `loop%12` 映射 | **已完成**        |
| 9   | 套件外 s2=32/48/96 精度；性能 / Q 双槽试探              | 精度 **已完成**；P0 Q 双槽 **已回退（2026-08-20）** |
| 10  | pytest layout BSND/TND + fp16out；D=64 Vᵀ L0 pad（同 MXFP4） | layout/fp16out **已过线**；D=64 **已过线**（已拷中短例 5/5） |
| 11  | 对标 MXFP4 OCP pytest 1–3：TND varlen / `act_s` pad / BSND·TND×D=64·sparse·非对齐 / gap_sparse / scale-stride-trap | **已过线**（代表性子集 33/33 PASS，见 §16.5） |




## 13. 与 0814 / MXFP4 对照摘要


|         | MXFP4   | 0814 MXFP8 | **本修订**     |
| ------- | ------- | ---------- | ----------- |
| 基块      | 256×128 | 256×128    | **128×128** |
| 组       | ×16     | ×4         | **×8**      |
| Dpv     | 20      | 8          | **12**      |
| 基块 mmad | 3       | 6          | **3**       |
| UB P    | 整块      | 半块         | **整块**      |
| L1 P 字节 | 320KB   | 256KB      | **192KB**   |




## 14. 修改点说明（0817 落地）

相对 `mxfp8_0814` 脚手架（256 / G=4 / Dpv=8 / 双切 k / 半块 P），本轮在 `mxfp8_0817` 上的代码改动如下。

### 14.1 常量与内存表


| 文件                                                             | 改动                                                                                                                                             |
| -------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `op_kernel/arch35/kernel_utils.hpp`                            | `MXFP8Kernel::TILE_GROUP_N`：4 → **8**                                                                                                          |
| `op_kernel/.../gemm/block/block_mmad_arch35_utils.hpp`         | `KVS`/`S2`=128；`QK_HALF_KVS`=64；去掉 `QK_K_SPLIT`/`PV_K_SPLIT`；L0B_Q 单缓 16KB；L0C QK 32KB×2，PV 偏移 **64KB**；L1 P 环 **12×16KB**，P-scale **12×768B** |
| `op_kernel/.../epilogue/block/block_epilogue_arch35_utils.hpp` | `KVS_BASE=128`；S **2×32KB**；oTmp@64KB；P 整基块 2×16KB；pScale/localGroupMax **12**；UB 偏移按新表重排 + `static_assert`                                    |




### 14.2 Cube / Kernel / 接口


| 文件                                                                   | 改动                                                    |
| -------------------------------------------------------------------- | ----------------------------------------------------- |
| `.../gemm/block/block_mmad_qk_arch35_mxfp8.hpp`                      | n 向按 `L0_TILE_N(=64)` 两轮拼基块；D 一次打满；Q→L0B 槽 0          |
| `.../gemm/block/block_mmad_pv_arch35_mxfp8.hpp`                      | 去掉 S2 k-split 循环；单次 `LoadP/V` + `MatmulPV`；P 槽 `% 12` |
| `.../arch35/block_sparse_attention_kernel_arch35_mxfp8_full_quant.h` | `PV_RE_DELAY_N`：8 → **12**（`PRELOAD=13`）              |
| `.../block_sparse_attention_kernel_interface.cpp`                    | `L0TileShapeQK`：`<128,128,128>` → `<128,64,128>`      |




### 14.3 Softmax / Host


| 文件                                                                     | 改动                                                                                                                            |
| ---------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `.../block_epilogue_online_softmax_arch35_reg_low_prec_fp16_mxfp8.hpp` | 注释改为 kvs=128；UB→L1 NZ 布局按 **整基块 16KB**（`UB_P_NZ_M=64`，不再用 `KVS/4`）                                                            |
| `op_host/block_sparse_attention_tiling.cpp`                            | `CalcBaseTileTilingParams950`：MXFP8 强制 `kvBaseTile_=128`；`CheckBlockShapeQuantConstraint`：MXFP8 的 `blockShapeY` 改为 **128 倍数** |




### 14.4 期4 代码收尾（相对 14.1–14.3）


| 文件                                                        | 改动                                                             |
| --------------------------------------------------------- | -------------------------------------------------------------- |
| `block_sparse_attention_kernel_arch35_mxfp8_full_quant.h` | `delay8*` → `delay12*`；`pscaleNum_` 环 `% 12 / 6`，初值 11         |
| `block_mmad_pv_arch35_mxfp8.hpp`                          | 形参 `delay8*` → `delay12*`                                      |
| `block_epilogue_compute_pscale_arch35_mxfp8.hpp`          | `GROUP_MAX_SPACE_LEN=12`，`WHOLE_PROCESS_LOOP=8`；UB 每基块 1 槽（不再 ×2）；**CopyPScale L1 起点 `(2*groupMaxStart+sub)%12`** |
| `block_epilogue_online_softmax_..._mxfp8.hpp`             | `GetLocalGroupMaxBufIdx`：`loop/2 % 12`（1KB/tile）               |
| `vf_computeScale_dn_mxfp8.h` / `_qs64`                    | P-scale **yStep=3**、stride **768**、每 tile **4** 个 group-max    |
| `vf_nd2nz_indexes_dn_mxfp8.h`                             | gather 索引步长 **×2**（两路 e4m3）                                    |
| `mxfp8_vf/vf_softmax_*`                                   | 四路 fp4 Or → **两路 ZERO/ONE**；qs128 写间距 4096/512                 |
| `copy_l1_to_l0_mx_fp8_a5.hpp`                             | V srcStride 满段判断 **128**（不再 256）                               |




### 14.5 待板上确认（2026-08-18 已关闭大部分）

原清单已由 §15 实测关闭：kvs 64/128/160/192/256、sparse、**qs64/kvs128**、e4m3 NZ、P-scale 768B 分形、**跨组 s2>1024** 均已过线。仍开放：套件外短 kvs、性能 vs MXFP4（见 §17）。

### 14.6 调用链（quantMode=4）


| 文件                                                        | 改动                                                                                  |
| --------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `op_host/op_api/aclnn_block_sparse_attention_v3.cpp`      | `MXFP8_OCP_QUANT=4`：QKV=`e4m3fn`、scale=`e8m0`；scale 维同 mxfp4；innerPrecise=4 / 无 LSE |
| `torch_extension/.../ops/block_sparse_attention.py`       | `_QUANT_MODE` 含 4；e4m3 输入 + e8m0 descale；out ∈ {fp16, bf16}                         |
| `torch_extension/.../ops/csrc/block_sparse_attention.cpp` | QKV 原样传递；descale wrap `ACL_FLOAT8_E8M0`；attentionOut 形状不翻倍                          |
| `torch_extension/.../ops/__init__.py`                     | 导出 `block_sparse_attention_v3`                                                      |




## 15. 板上反向分析（2026-08-18）

对照 dump 中间量、FIA/mxfp4 参考和 dump-off 全量 pytest，几何表（§1–§6）仍然成立；原稿未写死的实现选择与若干错误路径，以下为 **as-built 约束**，后续改动不得回退。

### 15.1 精度结果（dump-off，`cann_bsa_0817`，NPU 1）

判据：`rtol=1/128`，`atol=1e-4`，满足率 ≥99.5%，`maxRE<10`。`_E8M0_QK_FLIP_PAIR=False`。`BSA_MXFP8_DUMP=0`。


| case                  | fulfill | 结果    |
| --------------------- | ------- | ----- |
| qs128 / kvs128 dense  | 99.74%  | 过     |
| qs128 / kvs64 dense   | 99.79%  | 过     |
| qs128 / kvs160 tail   | 99.95%  | 过     |
| qs128 / kvs256 dense  | 99.99%  | 过     |
| GQA qs128 / kvs192    | 99.86%  | 过     |
| sparse qs128 / kvs256 | 99.64%  | 过     |
| qs64 / kvs128 dense   | 100.00% | 过     |
| qs128 / kvs1152（跨组 1 tile） | 100.00% | 过（08-19） |
| qs128 / kvs2048（两整组） | 99.93%  | 过（08-19） |
| qs128 / kvs3072（三组/环回） | 99.93%  | 过（08-19） |
| qs256 / kvs16384      | 99.71%  | 过（08-19） |
| qs64 / kvs1152、2048  | 100.00% | 过（08-19） |
| GQA / sparse / b=2 × kvs1152 | ≥99.79% | 过（08-19） |
| 套件外 s2=96             | 100.00% | 过（08-19 dump-off + bf16 RT） |
| 套件外 s2=32             | 100.00% | 过（08-19 Exp 后 fp16→bf16 RT） |
| 套件外 s2=48             | 100.00% | 过（08-19 Exp 后 fp16→bf16 RT） |


安装注意：自定义 opp 装到 `cann_bsa_0817` 后，pytest 必须用 **与** `libcust_opapi.so` **相同的文件所有者** 跑（root 装则 root 跑，再 `chown 1007:1013`）。root 读普通用户文件会被 ACL 跳过该 so。

### 15.2 原稿 vs 落地


| 项                    | 0817 原稿                   | 板上落地                                                                                                                                                                                                                                                                   |
| -------------------- | ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| QK 半面 / 基块 / G / Dpv | 64×128 / 128×128 / 8 / 12 | **一致**                                                                                                                                                                                                                                                                 |
| kvBaseTile           | 128                       | **一致**                                                                                                                                                                                                                                                                 |
| 多 tile softmax       | 未写死                       | **与 mxfp4 同构**：`TILE_GROUP_N=8` + last-tile Vec2；**不要**改成 per-tile online softmax                                                                                                                                                                                      |
| P snap               | 未写 −2 vs −8               | `NEG_TWO_VALE`（−2），同 mxfp4；**不要**用 −8                                                                                                                                                                                                                                  |
| Cube dtype           | 未写 LoadData/Mmad 一致性      | LoadData dest **与** Mmad 必须是 `mx_fp8_e4m3_t`。Mmad 仍用 `fp8_e4m3fn_t` 会使 S 溢出 inf → 全 NaN                                                                                                                                                                                |
| LoadData stride      | 未写                        | mx 路径 `srcStride = yStep`                                                                                                                                                                                                                                              |
| P NZ 打包              | 「两路 ZERO/ONE」             | **mxfp4 even/odd + VL128**，再按 FP8 C0=32 **K-outer** 平面。4 行连续 load + 4-way Gather；even/odd `j` 偏移见 VF                                                                                                                                                                   |
| CopyP / LoadP        | 未写 NZ 行数                  | 64 行 TLA `{M,8,8,0}` 填进 128 行 NZ L1；LoadP `srcStride = s2Base/16 = 8`                                                                                                                                                                                                  |
| peer gmax            | 「组尾经 L1 交换」               | AIC 上 `GetSubBlockIdx()` **恒 0**，普通 DataCopy 写不到两个 AIV UB。必须 `asc_copy_l12ub_sync`：L1 row0→AIV0、row1→AIV1。UB dest 偏移 **160KB+512**（不是 mxfp4/QFA 的 224KB+512）                                                                                                           |
| pScale 槽             | 每基块 1 槽                   | **两套下标**：UB `localGroupMax` = `loop/2%12`（GetPScaleParams 读这个）；L1 P-scale = `loop%12`（与 P 同槽，PV 按此读）。CopyPScale dest = `(2*groupMaxStart+subBlockIdx)%12`，`dstStride` 隔行写该 AIV 的 tile。组 0 的 0,2,4,6 碰巧等于 groupMax 环，组 1 起必须 `*2`。不可把 UB 环下标直接当 L1 行号；也不可把 mxfp4 的 `*2+1` 套回 GetPScaleParams |
| 尾块 pad               | 未写                        | kvs32 VF 必须把 groupmax pad 满 **4 组**。`PadPS2`：P L1 固定 128 行 × 4 个 FP8 C0。`PadVS2`：按 mxfp4 推广为 **4 平面 C0=32、平面高=`s2Align64`**（V 的 L1 是 `nZ(D, align64)`，s2=32 时是 64 行不是 128）。不可照抄 mxfp4 的 `halfEmbed=64` 两段 Fill。PV 必须 **先 Wait MTE1 再 Pad**（同 mxfp4） |
| qs64                 | 验收覆盖                      | host 允许 `blockShapeX=64/128`。QK 将 Q(M) **pad 到 128**（L1 Q/Q-scale `InitConstValue` 清零），Fixpipe N=128；softmax/pscale 走 **qs128 VF**（even/odd + `SPLIT_CONSEC64`）。`qsActBaseTileAlign64` 仍为 64，供 CopyP / PV / rescale。不要用 qs64 VF 的 VL64 直接打 even/odd 128-half（约 3.125%） |
| `processUpdate` 循环长  | 未写                        | 仍按 `KvsBase=256` 量级覆盖 mxfp8 32KB O，不要改成 128                                                                                                                                                                                                                            |




### 15.3 板上修过的根因（保留）

1. **QK/PV Mmad 类型错误** → S inf。FIA 路径：LoadData mx 且 Mmad `mx_fp8_e4m3_t`。
2. **P store 密度/平面顺序** → 99.74% 路径：even/odd VL128 + K-outer C0。
3. **两 tile 独立 softmax**（kvs256 ~1.4%）→ cube `asc_copy_l12ub_sync` 交换 peer gmax 后 99.99%。AIV 侧 DataCopy L1→UB 无效；cube `GetValue` dump 曾导致 507015。
4. **kvs160 行 NaN**（99.17%）→ kvs32 只 pad 1 个 groupmax，残留 Inf 变成 e8m0=255；PadPS2 曾按 64 行 2 平面写。
5. **qs64 约 3.125%** → Fixpipe N=64 时 VF 按 QsBase=128 读 even/odd 128-half，VL64 不是连续 qs[0:64)。QK pad N=128 + qs128 VF 后 100%。
6. **s2>1024 跨组 fulfill 崩**（1152 曾 62.5%，16384 曾 5.15%）→ CopyPScale 用 groupMax 环当下标写 L1，PV 读 `loop%12`。组 1 tile 8 写到槽 4、读到槽 8（组 0 残留 scale）。L1 映射改 `*2` 后 1152=100%、16384=99.71%。
7. **单 tile s2 不齐 64 出 NaN**（s2=32/48/96 曾 93%～96%，NaN 落在 `D[32:64)`）→ PadVS2 照抄 mxfp4 两段 `halfEmbed=64`，fp8 每面只清 32B/行。改为 4 平面 + Wait-then-Pad 后 s2=96=99.77% 过；s2=32/48 NaN 消失但仍 99.31%/99.39%。
8. **s2=32/48 qs57 差 0.19pt** → 2-spike softmax 的 k=12 上 fp16 `exp=1.064453`。FIA 路径 `fp16→float→e4m3 RINT` 得到 1.125；golden / mxfp4 是 `fp16→bf16→e4m3` 得到 1.0。`1.125×0.25` 使 rowsum 1.53125 vs 1.500，该行 105/128 D 刚过 rtol。所有 softmax VF 在 Exp 后加 `BSA_MXFP8_FP16_BF16_RT4` 后 s2=32/48/64/128 = 100%。



### 15.4 仍开放 / 已关闭（2026-08-19 续）

- 套件外短 kvs 精度已关闭（s2=32/48/96 均过）。  
- 跨组精度已关闭（见 §15.1 / §15.3.6）。  
- **BSND / uniform TND** pytest runner 已放开（golden 仍 BNSD，NPU I/O permute/flatten）；qs128 冒烟 **100%**。  
- **TND varlen / `act_s` pad / gap_sparse / scale-stride-trap / BSND·TND×D=64·sparse·非对齐**：本期已把 mxfp4 driver 路径接到 `test_bsa_mxfp8_golden.py`（代表性子集，非整份 dump）。板上抽跑见 §16.5。  
- **fp16out**（D=128，已拷 mxfp4 例，跳过 `qs25600`）**7/7 PASS 100%**。  
- **D=64**：host 一直放行，但 Vᵀ `LoadData.dstStride` 阈值误用 `FP8_C0=32`，D=64 得不到与 MXFP4 相同的 pad-to-144。改 `dAct <= CONST_64` 后已拷中短例 **5/5 过线**（含 qs128/kvs128=100%、sparse qs1/kvs31=100%、GQA qs256/kvs512=99.98%）。`qs256/kvs4097_d64` 已 100%；两条 `kvs4096_d64` 长 golden 未重跑。  
- 性能 vs MXFP4 / FP16：见 §17（P0 前基线已采；P0 试探 42.4→41.5 ms 后 **已回退**；S=118800 keep1 vs FIA/MXFP4 见 §17.5）。  
- Q L0B 双缓冲：P0 已回退为单缓 16KB。  

## 16. 相对 MXFP4 未实现 / 未对齐特性（2026-08-19）

OCP 主路径（BNSD、D=64/128、fp16out、layout 冒烟）已经能对上 MXFP4。还缺的主要是 **pytest 覆盖** 和 **一条设计上不做的量化模式**，不是再开一套 Cube 几何。

对照源：host tiling、aclnn、kernel tiling key、`test_bsa_mxfp8_golden.py`、MXFP4 `copy_l1_to_l0_mx_a5.hpp`。双方 **mx 全量化共同不做**：attenMask、paged cache、LSE。

### 16.1 已经对齐（不必再补 kernel）

- OCP `quantMode=4`：qs64/128、跨组、尾块、GQA、sparse  
- D=64（Vᵀ `dstStride` pad 到 144，见 §16.2）  
- fp16 / bf16 out  
- BSND / uniform TND（qs128 冒烟 100%）  
- `blockShapeX=64`（Q pad 到 128 VF）

### 16.1b 产品 / 设计范围（不做或不测）


| 特性 | MXFP4 | MXFP8 | 说明 |
| --- | --- | --- | --- |
| CX（`quantMode=3` + `dstTypeMax`） | 已验收（约 70+ 例） | **不做** | 设计仅 OCP=`4`。VF 里有 CX 模板，无 tiling key / API |
| `blockShapeY=256` | 基块 256，Y 可 256 | **不做** | MXFP8 `kvBaseTile=128`，Y 必须是 128 倍数 |
| Q L0B 双缓冲 | 几何不同 | **已回退** | 整段 Q 16KB×2 试过 −2%；D 切退路未合入；见 §5 / §17 |
| 对 MXFP4 / FP16 的性能对比 | 基线 | **已测** | 见 §17 / §17.5。基块 128 vs 256；S=118800 keep1 对标 FIA dense |
| S = bf16 / fp32 | pytest 可改 `s_dtype` | kernel `innerPrecise=4` 固定 fp16 S | 两边 softmax 都是低精 fp16 |


### 16.2 D=64 Cube（MXFP8 独缺 → 按 MXFP4 补）

MXFP4 **没有**单独的 D=64 tiling key。做法是 **PV 永远按 D=128 槽跑**：

- `ROW_SUM_NUM=128`，`PV_MMAD_M_DIM=144`，`processOut` / rescale `EMB_ALIGN128=128`  
- Vᵀ L1→L0：`dstStride = (dAct+15)/16 + 1`，**`if (dAct <= FP4_C0)` 且 `FP4_C0=64`** → D=64 时 dstStride **5→9**（与 D=128 同构）  
- `kStep = dAct / FP4_C0` 仍按实际 D（D=64 只搬 1 个 fp4 C0）  
- PadVS2 用 `halfEmbed = ROW_SUM_NUM/2`，始终清 **D=128 的两面**（D=64 的后一面写在 KV 槽余量）  
- QK 不 pad D：`MatmulQK(..., embed)`，k=实际 D  

MXFP8 0817 原稿把同一判断写成 `if (dAct <= FP8_C0)`（**32**），D=64 不进分支，dstStride 停在 5；PV 仍 mmad M=144 → L0A 读垃圾，fulfill ~0%。改阈值后板上 5/5 过线。

**代码对齐（2026-08-19）**：

| 文件 | 改动 |
| --- | --- |
| `.../tile_common/copy_l1_to_l0_mx_fp8_a5.hpp` | 两处 V 转置 LoadData：`dAct <= CONST_64` 时 `dstStride += 64/16`（5→9） |
| `.../gemm/block/block_mmad_pv_arch35_mxfp8.hpp` | PadVS2 注释对齐 mxfp4：4 个 FP8 C0 面覆盖 D=128 槽 |

QK / softmax / rescale **不用**为 D=64 另开几何；GM 只 copy 实际 64 列（`actualOriShape.m()`）。

### 16.3 还要对齐：Kernel 多半已有，pytest 是最大缺口（本期补 1–3）

这些是 MXFP4 OCP 真正多出来的产品面。CX 除外，mxfp4 大约就是 pytest 里那 ~80+ varlen / ~20 pad / ~20 trap。MXFP8 **不整份拷贝**，只接代表性子集。


| 优先级 | 缺口 | MXFP4 | MXFP8 接线前 | 本期做法 |
| --- | --- | --- | --- | --- |
| **1** | **TND varlen**（`s1_list` / `s2_list`） | 大量 dense/sparse/GQA | runner 无 concat golden | per-batch 造数/golden/`mxfp8_quantize_native` + concat T；`actualSeq=act_q/act_kv` |
| **1** | **`act_s` pad**（实际长度 < pad） | BNSD/BSND 比有效区 | 未接 | golden 走 unpadded；NPU 0-pad 到 `s1`/`s2`；只比 `[0:act_q)` |
| **2** | **BSND/TND × D=64 / sparse / 非对齐** | 一整组 | 只有 BNSD D=64 + layout qs128 冒烟 | 复用 `_to_npu_layout`，扩代表例 |
| **3** | **gap_sparse** | 每 q-block 仅 attend kv-block 0 + 末块 | mask 只有 dense/sparse 随机 | `_build_block_sparse_mask` 加 `gap_sparse` |
| **3** | **scale-stride-trap** | BSND/TND 跨头 e8m0 不等 | 只测了 contiguous permute | `data_kind='ocp_scale_stride_trap'`。MXFP8 用相邻 binade `|x|∈(2.0,3.4)/(4.0,6.8)` → e8m0 **120/121**（不可照搬 mxfp4 的 0.60/1.20，emax=8 下 O≈0 会误挂） |

`quant_mode` 仍只允许 **4（OCP）**。`block_shape[1]` 仍须 **128 倍数**（不移植 mxfp4 `blk128x256`）。

接线位置：`attention/block_sparse_attention/tests/mxfp8_pytest/test_bsa_mxfp8_golden.py` 的 `run_bsa_mxfp8_eager`。golden 仍是 `cpu_block_sparse_attention_bnsd_mxfp8`；打包走 e4m3 native（last dim 不翻倍），不是 mxfp4 nibble pack。

### 16.4 例已拷、还没跑完（覆盖率，不是功能洞）

- 从 mxfp4 抄来的 BNSD OCP 大 shape（`qs25600`、`b56`、很多 GQA）多数没单跑  
- D=64：`qs256/kvs4096` sparse 未跑；`qs256/kvs4097` 已 100%  
- fp16out monster `qs25600` 未跑  

大 shape 抽跑即可，不必全套 `qs25600`。

### 16.5 建议补齐顺序与板上抽跑

1. **TND varlen + `act_s` pad** — 和 MXFP4 产品面差最大，kernel 已按实际长度累加  
2. **BSND/TND 扩到 D=64、sparse、非对齐** — runner 已放开，缺的是例  
3. **gap_sparse / scale-stride-trap** — 防回归  
4. 大 shape 抽跑（不必全套 `qs25600`）  
5. CX / Y=256：产品点名再做。性能迭代见 §17。  

若「对标」只指 **OCP 能跑的 layout×shape 矩阵**，补 1–3 就齐了。若还要 **CX 和 blockY=256**，那是新设计，不是现成缺口。

**合入前回归（2026-08-20）**：OCP 1–3 固定一条 `-k`，约 27 例（varlen + act_s + trap + gap + BSND/TND D=64），**不含** `kvs4096/4097`。表达式与 `MXFP8_OCP_REG_K` / `tests/mxfp8_pytest/run.sh --reg` 一致，须整段一个参数、不要换行：

```bash
python -m pytest test_bsa_mxfp8_golden.py -v -s --tb=short -k "tnd_varlen or actq or scale_stride_trap or gap_sparse or (d64 and (bsnd_mha or tnd_mha or tnd_gqa) and not kvs4096 and not kvs4097)"
```

或在 `tests/mxfp8_pytest` 下 `./run.sh --reg`。新增回归例请带 `tnd_varlen` / `actq` / `scale_stride_trap` / `gap_sparse`，或 `(bsnd_mha|tnd_mha|tnd_gqa)+d64`。

**板上抽跑（2026-08-19，`cann_bsa_0817`，代表性子集 33/33 PASS）**：


| 类 | 代表例 | 结果 |
| --- | --- | --- |
| 1 TND varlen | dense/sparse/GQA/非对齐/D=64 | **PASS**（100% / 99.94% / 99.97% / 99.93% / 99.94%） |
| 1 act_s pad | BNSD/BSND `act=200 pad=256`、decode `act_s1=1`、per-batch、D=64 | **PASS** |
| 2 layout×D=64 | BSND/TND qs128 d64、sparse、非对齐、`block=[64,128]` | **PASS** |
| 3 gap_sparse | BNSD/BSND/TND qs128/kvs512（含 D=64） | **PASS** |
| 3 trap | BNSD 对照 + BSND/TND qs128 + 非对齐 + D=64 + varlen | **PASS**（e8m0 120/121；初版照搬 mxfp4 0.60/1.20 在 BNSD 也 87.53%，已改 binade） |

定位：`BSA_MXFP8_TRAP_DIAG=1` 打印 fail 坐标、`fail_by_N`/`fail_by_D`、`|npu/gold|` 是否聚在 2/0.5。初版 trap 最差点是 `gold=0 npu~3e-4`（atol），不是 2× scale。

## 17. 性能：MXFP8 vs FP16 / MXFP4（2026-08-20）

平台：Ascend950PR PCIE，28 Cube @ 1650 MHz。峰值：FP16 Cube **378 TFLOPS**，MXFP8 **757 TFLOPS（2.00×）**，MXFP4 **1514 TFLOPS**。HBM ~1.4 TB/s，L2 112 MB。

矩阵：B=1 N=1 D=128 BNSD `block=(128,128)` `innerPrecise=4`，keep 50%/20%。msprof `--kernel-name=BlockSparseAttention --warm-up=5 --launch-count=10`，过滤 Op Name。自定义 opp：`cann_bsa_0817`。画布：`docs/superpowers/canvases/BSA-MXFP8-vs-FP16-analysis.canvas.tsx`；S=118800 keep1 对照：`docs/superpowers/canvases/S118800-pipe-compare.canvas.tsx`。

### 17.1 理论 Roofline

FLOP = `4 · S · S_kv · D`，`S_kv = keep · S`。KV 重读字节 ≈ `2 · keep · S² · D · bpe / 128`。

| 工况 | FLOP | FP16 Cube 下限 | MXFP8 Cube 下限 | FP16 HBM 下限 | MXFP8 HBM 下限 | FP16 实测（P0 前） | MXFP8 实测（P0 前） |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S=243k keep50 | 15.1 TFLOP | 40.0 ms | 20.0 ms | 84 ms（L2 掩盖） | 42 ms | 46.2 ms | **42.4 ms** |
| S=1M keep50 | 281 TFLOP | 745 ms | 372 ms | 1570 ms | 785 ms | 1447 ms | **904 ms** |

结论：理论 Cube 加速 **2.00×**。S≤243k 实测只有 **1.03–1.09×**（MXFP8 cube 占用 0.50，AIC scalar 0.90，在等 softmax / P 量化）。S=1M 拉到 **1.60×**，因为 FP16 撞上 KV 重读的 HBM 墙；MXFP8 904 ms 相对 HBM 下限 785 ms 只高 15%。keep20/keep50 FP16 全程 ~0.40。

P0 前流水（S=243k keep50，PipeUtilization 核均）：

| 路径 | Duration | cube | scalar | mte2 | fixpipe | AIV vec | AIV scalar |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FP16 | 46.2 ms | 0.93 | 0.62 | 0.87 | 0.87 | 0.88 | 0.13 |
| MXFP8 | 42.4 ms | 0.50 | 0.90 | 0.75 | 0.42 | 0.84 | 0.38 |
| MXFP4 | 22.6 ms | 0.48 | 0.98 | 0.75 | 0.60 | 0.63 | 0.32 |

### 17.2 优化方案（按收益）

主战场是 **S=243k**：Cube 还有约 2× 头寸。S=1M 流水优化只剩约 13%（HBM 地板 ~785 ms）。

| 轮次 | 内容 | 主看指标 | S=243k keep50 预期 |
| --- | --- | --- | --- |
| **P0（已回退）** | ① 整段 Q 复制进 L0B 两槽常驻（不切 D）；② QK fixpipe 延后一拍；③ `V1_C1` wait 挪到 K gather 后。D 切 2×64 退路跨组/GQA 掉点未合入。**2026-08-20 源码已回退到 Q 单缓 16KB** | cube ≥0.75 | 预期 28–32 ms；**实测 41.5 ms 后回退** |
| **P1** | QK 一次 128×128（L0C 200KB&lt;256KB）去掉 n 半面；e8m0/pScale 合并搬运；压 AIV 组分支 | fixpipe ↑ | 接近 vs FP16 的 2× |
| **P2** | 仅 S≳1M：Q tile 按 L2 112MB 分块复用 KV | 1M Duration | 243k 几乎不变；1M 挑战 ~800 ms |

验证：每轮只改一类气泡。精度 qs128 + `--reg`。性能 S∈{8422, 243000}×keep{0.5,0.2}（P0 不跑 1M）。

P0 **未达** cube≥0.75 / 28–32 ms（见 §17.4），**源码已回退**。主瓶颈仍是 AIV softmax / AIC scalar。下一轮走 **P1**。

### 17.3 P0 代码落点

| 文件 | 改动 |
| --- | --- |
| `block_mmad_arch35_utils.hpp` | `L0B_QK_BUF_CNT=2`，`SIZE=16KB`；P 从 **32KB** 起（L0B 打满 64KB） |
| `block_sparse_attention_kernel_interface.cpp` | `L0TileShapeQK` 仍 `<128,64,128>`（不切 D） |
| `block_mmad_qk_arch35_mxfp8.hpp` | 首 tile 把整段 Q 复制进两槽常驻；nSub 只装 K；`EmitFixpipeMm1` 延后一拍；`CrossCoreWaitFlag(sReadyFlag)` 在 K gather 之后、mmad 之前 |
| `block_sparse_attention_kernel_arch35_mxfp8_full_quant.h` | AIC 不再在 QK 前 wait V1_C1，flag 传入 QK |

### 17.4 P0 板上结果（2026-08-20，NPU 5，`cann_bsa_0817`）

精度（dump-off）：冒烟 7/7 PASS（含 dense qs128/kvs128、qs64、kvs256、**sparse kvs256**、kvs1056_tail、kvs1280、GQA kvs192）；`./run.sh --reg` **27/27 PASS**。

msprof：`--kernel-name=BlockSparseAttention --warm-up=5 --launch-count=10`，目录 `tests/mxfp8_pytest/bsa_msprof_perf_mxfp8_p0/`（基线 `bsa_msprof_perf_mxfp8_longseq/`）。PipeUtilization 核均（cube 行 / vector 行分别平均）。

| 工况 | P0 前 Duration | P0 后 | Δ | P0 前 cube / scalar | P0 后 cube / scalar |
| --- | --- | --- | --- | --- | --- |
| S=8422 keep50 | 92.9 µs | 91.8 µs | −1.2% | （短序列头开销大） | 0.38 / 0.78 |
| S=8422 keep20 | 53.9 µs | 52.0 µs | −3.4% | 0.28 / 0.70 | 0.29 / 0.72 |
| **S=243k keep50** | **42.4 ms** | **41.5 ms** | **−2.1%** | **0.50 / 0.90** | **0.52 / 0.89** |
| S=243k keep20 | 17.2 ms | 16.9 ms | −2.0% | 0.50 / 0.89 | 0.51 / 0.88 |

S=243k keep50 流水（P0 后 vs 前）：mte2 0.76（0.75）、fixpipe 0.43（0.42）、AIV vec 0.85（0.84）、AIV scalar 0.39（0.38）。主导仍是 **AIC scalar 空转 + AIV softmax**，不是 Q 装载。

判定：相对 P0 目标（cube≥0.75，Duration 28–32 ms）**未达标**。整段 Q 双槽常驻省掉 nSub 的 Q MTE1；wait 后移只重叠稀疏 K gather，相对 softmax 可忽略。D 切 2×64 退路精度失败。**上述 P0 改动已于 2026-08-20 回退**（Q 单缓 16KB，wait 仍在 QK 入口）。下一步 **P1**：QK 一次 128×128、scale 合并搬运、压 AIV 组分支。

### 17.5 S=118800 keep=1.0 vs FIA dense / BSA MXFP4（2026-08-20）

对齐 FIA dense（`sparse_mode=0`）的 FLOP：BSA 用 keep=1.0，S 上限 118800。BSA NPU 5，FIA NPU 1。msprof 口径同 §17 头。画布：`docs/superpowers/canvases/S118800-pipe-compare.canvas.tsx`。

| 路径 | Duration | cube | AIC scalar | MTE2 | fixpipe | AIV vec |
| --- | --- | --- | --- | --- | --- | --- |
| BSA MXFP8 keep1.0 | **20.58 ms** | 0.506 | 0.899 | 0.766 | **0.426** | 0.839 |
| FIA MXFP8 dense | **14.60 ms** | 0.691 | 0.619 | 0.769 | 0.783 | 0.880 |
| BSA MXFP4 keep1.0 | **9.78 ms** | 0.532 | 0.988 | 0.830 | **0.704** | 0.701 |

绝对时间（PipeUtilization 核均）：

| | aic_time | cube_time | fixpipe_time | aiv_vec_time |
| --- | --- | --- | --- | --- |
| BSA MXFP8 | 20.05 ms | 10.15 ms | **8.54 ms** | 16.82 ms |
| FIA MXFP8 | 13.82 ms | 9.55 ms | 10.82 ms | 12.16 ms |
| BSA MXFP4 | 9.54 ms | 5.07 ms | **6.71 ms** | 6.69 ms |

结论：

1. **同 dense FLOP**，FIA 比 BSA MXFP8 快约 **29%**（14.60 vs 20.58 ms）。FIA cube 0.69，BSA MXFP8 0.51，AIC scalar 0.90 — Cube 在等 softmax / 核间同步。两边 MTE2 ratio 都约 0.77；FIA MTE2 活跃带宽 91 GB/s，BSA MXFP8/MXFP4 约 63 GB/s。
2. **BSA MXFP4 vs MXFP8** 墙钟约 **2.10×**（9.78 vs 20.58 ms）。Cube 有效时间 5074 vs 10152 µs = **正好 2.00×**，对上峰值 1514/757 TFLOPS。keep100/keep50 Duration 约 1.98×，与 2× FLOP 一致。
3. **MXFP8 fixpipe 占比低于 MXFP4，不是 fixpipe 做得少。** ratio = `fixpipe_time / aic_time`。MXFP8 绝对 fixpipe 时间 **8.54 ms > MXFP4 6.71 ms**；占比 0.43 vs 0.70 是因为 `aic_time` 被拉到 20.05 ms（MXFP4 9.54 ms）。两边 `aic_time ≈ aiv_time`。AIV vec：MXFP8 **16.8 ms** vs MXFP4 **6.7 ms**（基块 128 vs 256，同样 S2 的 softmax 次数约 2×，且 MXFP8 还要 e4m3 打包）。MXFP8 每基块两次 64×128 **即时** fixpipe（P0 延后一拍已回退）；MXFP4 走 `EmitFixpipeMm1` 延后一拍，短时间轴上重叠更好。看占比前应先看 `aic_fixpipe_time`。

数据目录：`tests/mxfp8_pytest/bsa_msprof_perf_mxfp8_keep1/`、`tests/mxfp4_pytest/bsa_msprof_perf_mxfp4_keep1/`、`attention/fused_infer_attention_score/tests/pytest/fia_fullquant_mxfp8_test/fia_msprof_perf_mxfp8_sameseq/`。解析：`docs/superpowers/parse_pipe.py`。

