# FIA Antiquant 非连续 Case — Tiling Key / Shape / Stride 记录

> 维护说明：每跑通/排查一个 case，请补全对应行的 **tiling key**、**K shape**、**K stride(nc/contig)**、**结果**。
> tiling key 从 plog 的 `GenTilingKey: The tilingkey is XXX` 抓取；stride 从 `aclnn_debug: K shape=[..] stride=[..]` 或 `stride_path:` 行抓取。
> 数据来源：`cases.json`（参数） + plog（tiling key/stride，运行时实测） + `dequant_checker.cpp`（scale shape 校验）。

## 0. Antiquant Mode 与 Scale 排布（全面调查，2026-08-22）

**非连续支持判定**：只有 scale 的 **0轴=Bn(totalBlockNum)** 且 **1轴=N2(n_kv head)** 时才支持非连续传入。其余 mode 的 scale 不支持非连续（checker 应拦截）。

| mode | 名称 | scale shape（checker 强制） | axis0 | axis1 | 支持非连续 | 源码位置 |
|------|------|---------------------------|-------|-------|-----------|---------|
| **0** | per_channel / per_tensor | 1D `[H]`/`[1]`；2D `[1,H]`/`[N,D]`；3D `[1,N,D]`/`[N,1,D]`；4D `[1,1,N,D]` | 1或N | — | ❌ 无 Bn 轴 | `CheckKScaleShapeForPerChannelPerTensorMode` :2794 |
| **1** | per_token | 2D `[B,≥S]`；3D `[1,B,≥S]`（PA 下末维≥maxBlockNumPerBatch×bs） | B或1 | — | ❌ 0轴是B非Bn，blockNum折叠进末维 | `CheckKScaleShapeForPerTokenMode` :2876 + PA :2381 |
| **2** | per_tensor_head | 1D `[N]` | N | — | ❌ 无 Bn 轴 | `CheckKScaleShapeForPerTensorHeadMode` :2930 |
| **3** | per_token_head | 3D `[B,N,≥S]`（PA 下末维≥maxBlockNumPerBatch×bs） | B | N | ❌ 0轴是B非Pn，blockNum折叠进末维 | `CheckKScaleShapeForPerTokenHeadMode` :2955 + PA :2405 |
| **4** | per_token_pa | 2D `[blockNum, blockSize]`（**必须 PA**） | **Bn✅** | blockSize | ✅ **仅0轴**（1轴是block内行，block内必须连续） | `CheckKScaleShapeForPerTokenPAMode` :2987 |
| **5** | per_token_head_pa | 3D `[blockNum, N, blockSize]`（**必须 PA**） | **Bn✅** | **N2✅** | ✅ **0+1轴** | `CheckKScaleShapeForPerTokenHeadPAMode` :3014 |
| **6** | per_token_group | 5D `[1, B, N, ≥S, D/32]`（PA 下倒数第二维≥maxBlockNumPerBatch×bs） | 1(固定) | B | ❌ 0轴固定为1，无 Bn 轴 | `CheckKScaleShapeForPerTokenGroupMode` :3042 + PA :2428 |

### 结论
- **支持 scale 非连续的只有 mode 4 和 mode 5**（PA 原生模式，scale 以 blockNum 为字面轴）：
  - **mode 5** `[blockNum, N, blockSize]`：0轴(Bn) + 1轴(N2) 都可非连续
  - **mode 4** `[blockNum, blockSize]`：仅 0轴(Bn)，无 N2 轴（1轴是 blockSize，block 内必须连续）
- **mode 0/1/2/3/6 的 scale 无字面 Bn 轴** → 不支持非连续传入（KV 仍可非连续，但 scale 必须连续）
- **KV（k/v）非连续**：所有 PA mode（0-6）都支持，与 scale 的 mode 无关
- kernel 侧佐证：`CopyAntiquantParamsPageAttention`（processor.h）的 offset = `idInBlockTable×blockStride + useN2Idx×headStride + remainElmenCnt`，其中 blockStride←scaleBnStride、headStride←scaleN2Stride——只有 mode4/5 有 useN2Idx 项（per-head），mode4 无 N2 项

### 完整 checker 拦截规格（`CheckStrideForAntiquant`，2026-08-22 实现）

**KV layout 约束**：
- 非 PA → kv 非连续 → 直接拦截
- PA + kv 非 PA_BNBD/PA_BBH/PA_NZ → 拦截（paged_attention_checker 布局检查兜底）
- PA + PA_BNBD/PA_NZ → 支持 0+1 轴非连续
- PA + PA_BBH → 仅 0 轴非连续

**Scale 约束（两道门）**：
- 非 PA → scale 非连续 → 直接拦截
- PA + scale 非连续：
  - **第一道门（mode）**：mode ∉ {4,5} → 拦截（scale 无 Bn 轴）
  - **第二道门（mode×kvLayout）**：
    - mode4 `[blockNum,blockSize]`（无 N2 轴）→ 任意 kvLayout 仅允许 axis0(Bn)
    - mode5 `[blockNum,N,blockSize]`（有 N2 轴）→ BBH 仅 axis0；BNBD/NZ 允许 axis0+1

**对标 PR 6430（fullquant）**：PR 6430 在 `CheckDequantScaleBnNBsDShapeMXFP8`（:911）用 `CheckTensorContiguous + dimIndex!=0&&dimIndex!=1` 拦截 scale 0/1 轴以外非连续（MXFP8 scale `[Bn,N2,...]` 0/1 轴即 Bn/N2）。antiquant 显式按 mode 区分，更严格（mode0/1/2/3/6 scale 无 Bn 轴，即使 dim0/1 非连续也拦截）。

## 1. stride 构造规则（`_make_nc`，测试侧固定用法，禁止修改）

`_make_nc(tensor, nc)` 通过 pad+slice 构造非连续 view（**不改数据，只改 stride**）：
- `dim0` → 在轴1 pad 一行再 slice 掉 → **轴0 stride 变大**（Bn 轴非连续）
- `dim1` → 在轴2 pad 一行再 slice 掉 → **轴1 stride 变大**（N2 轴非连续）
- `dim01` = 先 dim0 再 dim1 → 0+1 轴都非连续

连续值记为 `contig`，非连续值记为 `nc`。`_apply_nc` 把逻辑轴列表 `[0,1]` 翻译成 `dim01`、`[0]`→`dim0`、`[]`→不改。

## 2. 已验证 Case（CANN 9.2.0，2026-08-21）

### BNBD 布局 `[Bn, N2, Bs, D]` (fmt=BnNBsD, paKvShapeType=1)

| Case | tiling key | K shape | K stride (nc) | K stride (contig) | kScale stride (nc) | 结果 |
|------|-----------|---------|---------------|-------------------|--------------------|------|
| INT8-M5-BNBD-DIM01-FP16 | **564399360** | [1155840] | (33024,16512,128,1) | (32768,16384,128,1) | (258,129,1) | ✅ PASS max_err=0 |
| INT8-M5-BNBD-DIM01-BF16 | **564399360** | [1155840] | (33024,16512,128,1) | (32768,16384,128,1) | (258,129,1) | ✅ PASS max_err=0 |
| INT8-M4-BNBD-DIM01-FP16 | **564399360** | [1155840] | (33024,16512,128,1) | (32768,16384,128,1) | (129,1) | ✅ PASS max_err=0 |
| INT8-M5-BNBD-CONTIG-FP16 | 564399360 | — | — (连续基线) | — | — | ✅ PASS (baseline) |

### NZ 布局 `[Bn, N2, D1, Bs, D0]` (fmt=NZ, paKvShapeType=2)

| Case | tiling key | K shape | K stride (nc) | K stride (contig) | kScale stride (nc) | 结果 |
|------|-----------|---------|---------------|-------------------|--------------------|------|
| INT8-M5-NZ-DIM01-FP16 | **2711883009** | [1216512] | (36864,18432,2048,16,1) | (32768,16384,2048,16,1) | (258,129,1) | ✅ PASS max_err=0 |
| INT8-M5-NZ-DIM01-BF16 | **564399360** | [442368] | (73728,36864,4096,16,1) | (65536,32768,4096,16,1) | (514,257,1) | ✅ PASS max_err=0 |

> ⚠️ NZ-DIM01-BF16 走的是 564399360（不是 2711883009）——B=2/bs=256 参数组合落到 564399360。tiling key 由算子 GenTilingKey 根据参数算出，非人为指定。

### BBH 布局 `[Bn, Bs, N2, D]` (fmt=BnBsH, paKvShapeType=0，仅 0 轴非连续)

| Case | tiling key | K shape | K stride (nc) | kScale stride (nc) | 结果 |
|------|-----------|---------|---------------|--------------------|------|
| INT8-M5-BBH-DIM0-FP16 | 待测 | — | (仅0轴, dim0) | — | 待验证 |
| INT8-M4-BBH-DIM0-BF16 | **564399360** | [165120] | (33024,256,1) | (129,1) | ✅ PASS max_err=0 (checker bug 修复后) |

> **BBH checker bug（08-21 发现并修复）**：`CheckStrideForAntiquant`（dequant_checker.cpp:2238）BBH 分支原代码 `if (isBbhLayout) { 无条件报错 }`，导致 BBH dim0 非连续（合法）也被拦截。已改为 `if (isBbhLayout && dimIndex != 0) { 报错 }`——dim0 放行，仅 dim≠0 报错。这就是 fork 报告里"BBH 1轴拦截不生效"的真相（实际是 BBH 0轴也被误拦）。

## 3. 待验证 Case（参数已知，tiling key/stride 待跑后补全）

### INT8

| Case | kv | mode | fmt | out | B | kv_seq | bs | q_seq | nc轴(K/V/ks/vs) | tiling key | 结果 |
|------|----|------|-----|-----|---|--------|-----|-------|------------------|-----------|------|
| INT8-M4-BNBD-DIM01-FP16 | int8 | 4 | BnNBsD | fp16 | 2 | 1024 | 128 | 1 | [0,1]×4 | ✅PASS | 08-24验证 |
| INT8-M4-BBH-DIM0-BF16 | int8 | 4 | BnBsH | bf16 | 1 | 512 | 128 | 1 | [0]×4 | ✅PASS | 08-24验证 |
| INT8-M0-BNBD-DIM01-BF16 | int8 | 0 | BnNBsD | bf16 | 1 | 1024 | 128 | 1 | [0,1]×4 | ✅PASS | 08-24验证 |
| INT8-M1-BNBD-DIM01-BF16 | int8 | 1 | BnNBsD | bf16 | 1 | 1024 | 128 | 1 | [0,1]×4 | ✅PASS | 08-24验证 |
| INT8-M0-NONPA-CONTIG-BF16 | int8 | 0 | none | bf16 | 2 | 512 | 0 | 8 | [] (非PA连续) | ✅PASS | 08-24验证 |
| INT8-M1-NONPA-CONTIG-BF16 | int8 | 1 | none | bf16 | 1 | 512 | 0 | 16 | [] (非PA连续) | ✅PASS | 08-24验证 |
| INT8-M2-BNBD-DIM01-BF16 | int8 | 2 | BnNBsD | bf16 | 1 | 1024 | 128 | 1 | [0,1](KV) [](scale) | ✅PASS max_err=0 | 08-24验证 |
| INT8-M4-NZ-DIM01-BF16 | int8 | 4 | NZ | bf16 | 2 | 512 | 256 | 1 | [0,1](KV) [0](scale) | ✅PASS max_err=0 | 08-24验证 |

### FP8E4M3

| Case | kv | mode | fmt | out | B | kv_seq | bs | q_seq | nc轴 | tiling key | 结果 |
|------|----|------|-----|-----|---|--------|-----|-------|------|-----------|------|
| FP8E4M3-M4-BNBD-DIM01-FP16 | float8_e4m3fn | 4 | BnNBsD | fp16 | 1 | 1024 | 128 | 2 | [0,1]×4 | ✅PASS | 08-24验证 |
| FP8E4M3-M4-BNBD-DIM01-BF16 | float8_e4m3fn | 4 | BnNBsD | bf16 | 1 | 1024 | 128 | 2 | [0,1]×4 | ✅PASS | 08-24验证 |
| FP8E4M3-M0-BNBD-DIM01-FP16 | float8_e4m3fn | 0 | BnNBsD | fp16 | 1 | 1024 | 128 | 4 | [0,1]×4 | ✅PASS | 08-24验证 |
| FP8E4M3-M4-BBH-DIM0-FP16 | float8_e4m3fn | 4 | BnBsH | fp16 | 1 | 512 | 128 | 1 | [0]×4 | ✅PASS | 08-24验证 |
| FP8E4M3-M4-NZ-DIM01-BF16 | float8_e4m3fn | 4 | NZ | bf16 | 2 | 512 | 256 | 8 | [0,1]×4 | ✅PASS | 08-24验证 |
| FP8E4M3-M1-BNBD-DIM01-FP16 | float8_e4m3fn | 1 | BnNBsD | fp16 | 1 | 1024 | 128 | 1 | [0,1](KV) [](scale) | ✅PASS max_err=0 | 08-24验证 |

### HIFLOAT8 / INT4 / FP4

| Case | kv | mode | fmt | out | B | kv_seq | bs | q_seq | d | nc轴 | tiling key | 结果 |
|------|----|------|-----|-----|---|--------|-----|-------|----|------|-----------|------|
| HIF8-M0-BNBD-DIM01-FP16 | hifloat8 | 0 | BnNBsD | fp16 | 1 | 1024 | 128 | 4 | 128 | [0,1]×4 | ✅PASS | 08-24验证 |
| HIF8-M0-BNBD-DIM01-BF16 | hifloat8 | 0 | BnNBsD | bf16 | 1 | 1024 | 128 | 4 | 128 | [0,1]×4 | ✅PASS | 08-24验证 |
| HIF8-M0-BBH-DIM0-FP16 | hifloat8 | 0 | BnBsH | fp16 | 1 | 512 | 128 | 2 | 128 | [0]×4 | ✅PASS | 08-24验证 |
| INT4-M0-BNBD-DIM01-FP16 | int4 | 0 | BnNBsD | fp16 | 1 | 1024 | 128 | 1 | 16 | [0,1]×4 | ✅PASS | 08-24验证 |
| INT4-M0-BNBD-DIM01-BF16 | int4 | 0 | BnNBsD | bf16 | 1 | 1024 | 128 | 1 | 16 | [0,1]×4 | ✅PASS | 08-24验证 |
| INT4-M3-BNBD-DIM01-FP16 | int4 | 3 | BnNBsD | fp16 | 1 | 1024 | 128 | 1 | 16 | [0,1]×4 | ✅PASS | 08-24验证 |
| INT4-M0-BBH-DIM0-FP16 | int4 | 0 | BnBsH | fp16 | 1 | 512 | 128 | 1 | 16 | [0]×4 | ✅PASS | 08-24验证 |
| INT4-M1-BNBD-DIM01-FP16 | int4 | 1 | BnNBsD | fp16 | 1 | 1024 | 128 | 1 | 16 | [0,1](KV) [](scale) | ✅PASS max_err=0.000415 | 08-24验证 |
| FP4-M6-BNBD-DIM01-FP16 | fp4_e2m1 | 6 | BnNBsD | fp16 | 1 | 1024 | 128 | 4 | 128 | [0,1](KV) [](scale) | ✅PASS | 08-24验证 |
| FP4-M6-BNBD-DIM01-BF16 | fp4_e2m1 | 6 | BnNBsD | bf16 | 1 | 1024 | 128 | 4 | 128 | [0,1](KV) [](scale) | ✅PASS | 08-24验证 |

> FP4-M6 的 scale 不做非连续（mode6 scale 恒 `[1,B,N,S,D/32]` BNSD 格式无 PA 轴，见 [[fia-design-impl-divergence]]）。

## 4. 异常 Case（期望 checker 拦截报错）

| Case | 场景 | 期望 | tiling key | 结果 |
|------|------|------|-----------|------|
| BBH-DIM1-ERR | BBH 1轴非连续（应拦截） | 报错 | 564399360 | ✅ PASS(expected error) 拦截生效 |
| INT8-M5-QS16-ERR | mode5 q_seq=16（mode5 INT8 限 q_seq=1） | 报错 | ✅PASS | 08-24验证 |
| INT8-M0-PA-QS8-ERR | mode0 PA q_seq=8（mode0/1 INT8+PA 限 q_seq=1） | 报错 | ✅PASS | 08-24验证 |

## 5. Tiling Key 与参数映射（实测归纳）

tiling key 由算子 `GenTilingKey`（`fused_infer_attention_score_tiling_impl.cpp:1474`）根据参数组合算出。已实测：
- **564399360**: BNBD 布局 + mode5/4/0/1 + INT8/FP8/HIF8/INT4（多 mode 共用，q_seq=1）+ NZ-BF16(B=2,bs=256)
- **2711883009**: NZ 布局 + mode5 + B=1,bs=128,kv_seq=4096

> tiling key param 字段（plog `GenTilingKey` 行）：`inOutLayoutType, config, pseMode, quantMode, hasAttenMask, hasRope, KvLayoutType, isFd, emptyTensor, enableKvPrefix, enableS1OutSplit, isReconstructTemp`。
> 例：564399360 = `inOutLayoutType:0, config:13, pseMode:9, quantMode:6, hasAttenMask:0, hasRope:0, KvLayoutType:1, isFd:0, ...`

### 关键认知：mode4/5 不靠 tiling key 区分

host `keyAntiquantMode`(0-6) 与 tiling key 的 `quantMode` 字段是**两套编码**，映射在 `UpdateTilingKeyQuantMode`（`fused_infer_attention_score_tiling_impl.cpp:1376-1418`）：
- mode 0 → quantMode `PER_CHANNEL`(0) 或 `K_PER_CHANNEL_V_PER_TOKEN`(2)
- mode 2 → quantMode `PER_CHANNEL`(0)（perHeadFlag 区分）
- **mode 1/3/4/5 → 全部 quantMode `K_PER_TOKEN`(6)**（靠运行时 flag 分流，不靠 tiling key）

因此：
- **mode4/5 与 mode1/3 共用同一 tiling key**（quantMode=6），不新增 tiling key
- kernel 侧 mode4/5 分流靠 `antiquantPageAttentionFlag`(mode4/5=true) + `antiquantPerHeadFlag`(mode2/3/5=true)，与 KV dtype 无关
- 注意：tiling key 层面共用 ≠ dtype 层面放开。checker 的 mode4/5 dtype 校验（`CheckInputKVTypeForAntiquantPerTokenPA`/`PerTokenHeadPA`）仍按 dtype 拦截——**mode4 原始仅 INT8+FP8，mode5 原始仅 INT8**，这是算子公开支持矩阵，非连续功能开发不应改动

## 6. 编译用 tiling key

- **单 key**: `bash build.sh ... --tiling_key=564399360`
- **多 key（分号 cmake list）**: `bash build.sh ... --tiling_key=564399360\;2711883009`（shell 里分号转义；build.sh 的多个 `--tiling_key=` 会后者覆盖前者，**必须用分号**）
- **全量**: 不传 `--tiling_key`（编所有 key，约 2 小时，慎用）

当前已编入包的 key：564399360 + 2711883009（覆盖四个首 case）。

**非 PA tiling key（08-24 补编）**：非 PA 连续 case 也需要 tiling key（走非 PA 路径），否则 launch 失败污染后续 case：
- `2364416`：INT8-M0-NONPA-CONTIG-BF16（mode0 非 PA 连续）
- `27530240`：INT8-M1-NONPA-CONTIG-BF16（mode1 非 PA 连续）

当前包编入 8 key：`564399360;2711883009;564399361;539233536;539233280;564399104;2364416;27530240`
