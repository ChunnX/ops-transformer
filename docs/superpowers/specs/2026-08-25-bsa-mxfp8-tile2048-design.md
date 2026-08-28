# BSA MXFP8 组窗口 1024→2048（方案 A）

日期：2026-08-25  
分支：`mxfp8_0825`（从 `mxfp8_0817` / P1 切出）  
基线：`docs/superpowers/specs/2026-08-17-bsa-mxfp8-design.md`（kvs128、G=8、覆盖 **1024** S2、Dpv=12）

状态：**设计 + 实现 + 精度冒烟 + keep1 性能已上板**。

## 1. 1024 是什么

当前 0817/P1 几何：

| 量 | 值 | 含义 |
| --- | --- | --- |
| 基块 | `KVS_BASE=128` × `QS=128` | 一次 QK + 一次 softmax + 一个 L1 P 槽（16KB e4m3） |
| 组 G | `TILE_GROUP_N=8` | L0C 累加 / group-max / pScale 作用域 |
| **组窗口** | **8 × 128 = 1024** | 跨组才换 global max；`s2>1024` 才走第二组 |
| Dpv | 12 | `Dpv ≥ G + DELAY_PSCALE_N` → `≥ 11`，取偶数 12 |
| P 环 | 12 | `= Dpv`，`loop % 12` |

softmax VF 几何、QK 128×128、e4m3 NZ、OCP `FLOOR+NEG_TWO` **不动**。P 量化 2026-08-26 起去掉 RT4，走 FIA `fp16 → float → e4m3 RINT`（见 §13）。

## 2. 三种把窗口拉到 2048 的办法

| | 做法 | 窗口 | 本修订 |
| --- | --- | --- | --- |
| **A** | G 8→16，kvs 仍 128 | 16×128=**2048** | **采用** |
| B | kvs 128→256，G 仍 8 | 8×256=2048 | 不做。L1 P 槽 32KB、L0A/L0C 超预算，等于重做 P1 |
| C | G=16 且 kvs=256 | 4096（对齐 MXFP4） | 不做 |

方案 A 只加长组、不改基块。组尾 `asc_copy_l12ub_sync` 频率减半（每 16 基块一次）。softmax 仍是 128×128，**aiv_vec 预期几乎不变**。

约束不变：

```
Dpv >= G + DELAY_PSCALE_N
G = 16 → Dpv >= 19 → 取 20（偶数，两 AIV 对齐；与 MXFP4 的 Dpv=20 同构）
```

环深 = Dpv = **20**。槽 `loop % 20` 在 `loop+20` 被 PV 读走；同拍先 PV 后写 P，不需要 21 槽。

## 3. 常量对照


| 常量 | MXFP4 | 0817 MXFP8（1024） | **本修订（2048）** |
| --- | --- | --- | --- |
| `TILE_GROUP_N` | 16 | 8 | **16** |
| 组窗口 S2 | 4096 | 1024 | **2048** |
| `KVS_BASE_SIZE` | 256 | 128 | **128（不变）** |
| `PV_RE_DELAY_N` | 20 | 12 | **20** |
| `DELAY_PSCALE_N` | 3 | 3 | 3 |
| `PRELOAD_TASK_CACHE_SIZE` | 21 | 13 | **21** |
| `L1_P_BUF_CNT` | 20 | 12 | **20** |
| `L1_P_BUF_SIZE` | 16KB（e2m1） | 16KB（e4m3 128×128） | **16KB（不变）** |
| `L1_P_SCALE_BUF_CNT` | 20 | 12 | **20** |
| `UB_LOCAL_GROUP_MAX_CNT` | 20 | 12 | **20** |
| `UB_P_SCALE_CNT` | 8 | 12 | **20** |
| `WHOLE_PROCESS_LOOP` | 16 | 8 | **16** |
| `GROUP_MAX_SPACE_LEN` | — | 12 | **20** |
| `pscaleNum_` | `% 20` /10，初值 19 | `% 12` /6，初值 11 | **`% 20` /10，初值 19** |


主循环里 `delay12*` 变量名沿用（索引仍是 `PV_RE_DELAY_N`），实际延迟是 20。不要把 delay3 段挪到 QK 之后（2026-08-25 keep1 试过 Duration +7%，已回退）。

## 4. L1（512 KB）

KV 单槽仍 32KB（Vᵀ+seed 18KB）。P 单槽仍 16KB。空闲从 ~140KB 收到 ~6KB。


| 区 | 偏移 | 单槽 | 槽数 | 合计 |
| --- | --- | --- | --- | --- |
| P 数据 | 0 | 16 KB | **20** | **320 KB**（`loop % 20`） |
| P-scale | **320 KB** | 768 B | **20** | 15 KB |
| Q 数据 | **335 KB** | 16 KB | 2 | 32 KB |
| Q-scale | **367 KB** | 512 B | 2 | 1 KB |
| KV | **368 KB** | 32 KB | 4 | 128 KB |
| KV-scale | **496 KB** | 2 KB | 4 | 8 KB |
| localGlobalMax | **504 KB** | 512 B | 4 | 2 KB |
| 空闲 | 506 KB 起 |  |  | **~6 KB** |


L0 与 P1 相同（QK 128×128，不改）。

## 5. UB（256 KB）

S / P / O / oTmp 布局与 0817 相同（P 物理窗口仍在 96.5–128.5 KB）。只加长 group-max / pScale 环，**pScale 必须跟在 index 后面**，不能停在 176 KB（会和加长后的 localGroupMax 重叠）。


| 区 | 0817 | 本修订 |
| --- | --- | --- |
| localGroupMax | 12×1 KB @ 161.75 KB | **20×1 KB**，末尾 181.75 KB |
| localGlobalMax / dm / index | 紧随 12 槽 | 紧随 **20** 槽 |
| pScale | 12×768 @ 176 KB | **20×768 @ 184 KB**（index 末尾） |


合计仍远低于 256 KB。

## 6. CopyPScale / 跨组

0817 §15.3.6：AIV 本地 `groupMax = loop/2 % 环长`，L1 必须是 `loop % 环长`。映射：

```
l1Start = (2 * groupMaxStart + subBlockIdx) % L1_P_SCALE_BUF_CNT
```

环长 12→**20** 后公式不变。组 1 的 AIV0 从 groupMax 8 起 → L1 16，skip-2 得到 16,18 再回绕 0,2,…，与 `loop%20` 的 16,18,0,2,… 对齐。

禁止再套 MXFP4 的 `*2+1`（那是 2 槽/基块）。`GetPScaleParams` 仍按一槽一 softmax。

`s2=2048`：0817 是 **两组 1024**；本修订是 **一组 2048**。online softmax / pScale 作用域变了，数字会动，`--reg` 必须重过。OCP 窗口 2048 仍小于 MXFP4 的 4096。

跨组验收从「s2>1024」改成「**s2>2048**」（例如 2176 / 2304 / 3072 / 16384）。原 1056/1152/1280/2048 在本修订里是组内或单组，不再覆盖「第二组 CopyPScale」。

## 7. 文件清单

| 文件 | 改动 |
| --- | --- |
| `op_kernel/arch35/kernel_utils.hpp` | `TILE_GROUP_N` 8→16 |
| `block_sparse_attention_kernel_arch35_mxfp8_full_quant.h` | `PV_RE_DELAY_N=20`；`pscaleNum_ % 20 / 10`，初值 19 |
| `block_mmad_arch35_utils.hpp` | L1 P/P-scale 20 槽及后续偏移 |
| `block_epilogue_arch35_utils.hpp` | `UB_LOCAL_GROUP_MAX_CNT=20`；`UB_P_SCALE_CNT=20` @ 184 KB |
| `block_epilogue_compute_pscale_arch35_mxfp8.hpp` | `WHOLE_PROCESS_LOOP=16`，`GROUP_MAX_SPACE_LEN=20` |
| 本文件 | 方案 A 唯一几何真值 |
| pytest 跨组注释 | s2>1024 → s2>2048；可补 s2=2176/2304 |

**不改**：softmax VF、QK 128×128、kvs256、Q 双槽、gmax `_sync`、主循环里 delay3 与 QK 的相对顺序。

## 8. 风险

1. L1 只剩 ~6 KB，偏移算错会和 Q/KV 撞。用 `static_assert` 卡 512 KB 和 P-scale 与 Q 相接。
2. 短 S2（基块数 &lt; 20）Dpv 填泵更空；功能仍走同一 drain，Duration 可能略差。
3. CopyPScale 回绕：组 1 写 L1 16,18 再 0…；PV `loop%20` 对不上就会重现 1152 级掉点。
4. 精度窗口加倍；sparse qs128 kvs256 的 99.707% 是 P1 已知点，不要当成新回归。
   QK scale 走 CopyL0CToUB `QF322F16_PRE`（与 MXFP4 同构），设计口径 ≥99.5%，不再要求 100%。

## 9. 性能预期（keep1 S=118800）

P1：Duration 19.86 ms，aiv_vec 16.86 ms，cube 10.15 ms。  
方案 A **砍不了 softmax**。组尾 sync 116→58 次，墙上乐观 0.3–1.0 ms。成功：Duration 降且 aiv_vec 几乎不变。vec 涨或 50% 级 fulfill → 回退。

不要把 gmax 挪到 QK 之后当本方案的一部分。

## 10. 验收

- L1/L0/UB `static_assert` 通过。
- 冒烟：dense qs128 kvs128/256/1280；**kvs2048 现为一组**；**kvs2176/2304/3072 为跨组**；GQA；qs64；sparse kvs256（≥99.5%）。
- `./run.sh --reg` ≥99.5%。
- 性能：S=118800 keep=1.0 vs P1 19.86 ms；目录 `tests/mxfp8_pytest/bsa_msprof_perf_mxfp8_g16/`。

## 11. 板上结果（2026-08-25，NPU 1，`cann_bsa_0703`）

编包安装已完成。精度：

| 用例 | fulfill |
| --- | --- |
| dense qs128 kvs128 / GQA kvs192 / kvs1280 | **100%** |
| qs64 kvs2048（一整组）/ qs64 kvs2176（跨组） | **100%** |
| qs128 kvs2176 / 2304 / 3072（跨组 + 环回绕） | **100%** |
| sparse qs128 kvs256 | **99.707%**（与 P1 相同，≥99.5%） |
| `./run.sh --reg` | 11 pass / 16 fail，失败例全部 **99.80–99.98%**（与 P1 同口径 ulp） |

精度门禁见上表（PRE 前、VF dScale 路径）。2026-08-26 起 QK scale 改 CopyL0CToUB PRE，与 MXFP4 同构，门禁改为 ≥99.5%。

## 12. 性能板上（2026-08-25）

`msprof op --kernel-name=BlockSparseAttention --warm-up=5 --launch-count=10`，Current Freq = Rated 1650。目录 `tests/mxfp8_pytest/bsa_msprof_perf_mxfp8_g16/`。P1 对照：NPU 1；本轮：NPU 0。

| | P1 G=8 keep1 | **G=16 keep1** | P1 keep0.5 | **G=16 keep0.5** |
| --- | --- | --- | --- | --- |
| Duration | 19.86 ms | **18.58 ms（−6.5%）** | 10.00 ms | **9.40 ms（−6.1%）** |
| cube_time | 10.15 ms | **10.15 ms** | 5.08 ms | 5.08 ms |
| aiv_vec | 16.86 ms | **16.42 ms** | 8.44 ms | 8.23 ms |
| bubble | 3.01 ms | 2.16 ms | 1.56 ms | 1.16 ms |
| cube occupancy | 0.511 | 0.547 | 0.508 | 0.541 |

判定：**达标**。Duration 降、cube 持平、vec 不升（组尾 sync 减半吃掉 ~0.85 ms 气泡）。softmax 仍是墙；不要据此再加大 G。

keep=50% 长序列（同矩阵，未含 S=1M）。S≥26352 相对 P1 均值 **−5.9%**。S=8422（90 µs）噪声内略慢。

| S | FP16 | P1 1024 | **G=16** | MXFP4 | vs P1 |
| --- | --- | --- | --- | --- | --- |
| 26352 | 0.58 ms | 0.55 ms | **0.52 ms** | 0.34 ms | −4.9% |
| 118800 | 10.67 ms | 10.00 ms | **9.40 ms** | 5.51 ms | −6.1% |
| 243000 | 46.25 ms | 40.65 ms | **38.16 ms** | 22.56 ms | −6.1% |

S=243k：FP16/G16=1.21×，G=16/MXFP4=1.69×（P1/MXFP4 曾是 1.80×）。cube_time 21.21 ms 与 P1 持平。

## 13. P 量化 ISA 约束（DAV 3510 / arch35）

**结论：硬件没有 `bf16 → e4m3` 单指令。** 不是 MicroAPI 漏封装，也不是 BSA 没写 `Cast<fp8_e4m3fn_t, bfloat16_t>`。写了也编不过 / 不会变成一条 vconv。

### 13.1 3510 向量 `vconv` 实际有的单指令

| 单指令 | 对应 MicroAPI | 实现头 |
| --- | --- | --- |
| **float ↔ e4m3** | `Cast<fp8_e4m3fn_t, float, CAST_RINT>` | `asc_float2e4m3_impl.h` / `asc_e4m32float_impl.h` |
| **bf16 ↔ e2m1x2** | `Cast<float4_e2m1x2_t, bfloat16_t>` | `asc_bfloat162e2m1x2_*` / `asc_e2m1x22bfloat16_impl.h` |

目录里 **没有** `bfloat162e4m3`、`half2e4m3`。e4m3 的源类型只接到 **fp32**。

官方 Cast 算子同一约束：`bf16→e4m3`、`fp16→e4m3` 都标 `CAST_TEMPLATE_TWO_CAST`，中间类型 **FLOAT**（`cast_struct.h`）。`CastTwo` = ST→mid→DT 两次转换。

### 13.2 因此三条 P 路径不能同构

| | P 量化 | 能否一条 Cast |
| --- | --- | --- |
| **FIA MXFP8** | `fp32 Exp → e4m3 RINT` | 能（直接命中 `float2e4m3`） |
| **BSA MXFP4** | `fp16 Exp → bf16 → e2m1` | 能（命中 `bfloat162e2m1x2`） |
| **BSA MXFP8** | 不能 `bf16 → e4m3` | 必须 `… → float → e4m3 RINT` |

BSA MXFP8 当前 VF（2026-08-26 起，对齐 FIA TWO_CAST，去掉 RT4）：

```
fp16 Exp → Cast<float>（SPLIT / qs64 直转）→ e4m3 RINT
```

- **去不掉的是 float 这一跳**（ISA）。没有 `Cast<e4m3, bf16>`。
- **RT4 已删除**：原先用 `fp16→bf16→fp16` 把 `1.064453` 拉到 e4m3 1.0 以对齐 MXFP4 网格；现按 FIA `fp16→float RINT`（该点会落到 e4m3 1.125）。golden 同步为 `p_local.float().to(e4m3fn)`。短 kvs 精度根因见 `2026-08-17-bsa-mxfp8-design.md` §15.3.8；门限仍 ≥99.5%。

pscale（e8m0）不受这条约束：BSA 仍是 `K_b − K_g + 125` 的 uint8；FIA 的 e8m0 才走 `float → bf16 → e8m0`。
