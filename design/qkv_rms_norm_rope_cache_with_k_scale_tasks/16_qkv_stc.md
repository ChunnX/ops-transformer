# QkvRmsNormRopeCacheWithKScale STC

## 1. 文档目的

本文给出 `QkvRmsNormRopeCacheWithKScale` 算子的 Software Test Cases（STC），用于回答以下问题：

1. 算子一共有多少种可能的场景与布局组合。
2. 哪些组合是接口支持的合法模式，哪些组合必须拒绝。
3. 对每种合法模式分别测试了哪些 shape、边界、Cache、位置编码和精度场景。
4. 每类测试的预期结果和验收门限是什么。

本文的有效数值用例对应已经完成的 800 例 CPU Golden/NPU 全量对拍。接口负向用例与 Host 推导用例单独列出，不计入这 800 例。

## 2. 模式是如何确定的

接口没有独立的 `scene` 属性，而是根据位置输入的 presence 判断场景。

### 2.1 RoPE 场景选择条件

```text
queryStartLocOptional: 必须传入
seqLensOptional      : 必须传入
mropePositionOptional: 不传
mropeSectionOptional : 不传或传空数组
```

同时要求：

```text
qQuantMode   = "PerTokenPerHead"
qOut dtype   = FP8 E4M3FN
qScale       = FP32 Tensor
kCache dtype = FP8 E4M3FN
```

### 2.2 M-RoPE 场景选择条件

```text
queryStartLocOptional: 不传
seqLensOptional      : 不传
mropePositionOptional: 传入 [T,3]，三列依次为 T/H/W
mropeSectionOptional : 传入非空 [t,h,w]
```

同时要求：

```text
qQuantMode   = "NoQuant"
qOut dtype   = BF16
qScale       = nullptr/None
kCache dtype = INT8
```

两组位置输入不能同时存在，也不能同时缺席；每组内部的输入必须成对存在。

## 3. 所有可能的模式

场景有 `RoPE` 和 `M-RoPE` 两种；输入和输出布局各有 `TND`、`NTD` 两种。笛卡尔积共有 8 种组合，其中只有 4 种合法。

| 模式ID | 场景 | `layoutQkv` | `layoutQOut` | 支持状态 | 800例中的数量 |
|:---:|:---:|:---:|:---:|:---:|---:|
| M01 | RoPE | NTD | NTD | 支持 | 200 |
| M02 | RoPE | TND | TND | 支持 | 200 |
| M03 | RoPE | TND | NTD | 支持 | 200 |
| U01 | RoPE | NTD | TND | 不支持 | 0 |
| M04 | M-RoPE | TND | TND | 支持 | 200 |
| U02 | M-RoPE | TND | NTD | 不支持 | 0 |
| U03 | M-RoPE | NTD | NTD | 不支持 | 0 |
| U04 | M-RoPE | NTD | TND | 不支持 | 0 |

合法模式总数：

```text
4 modes * 8 head combinations * 25 T values = 800 cases
```

默认属性组合为：

```text
layoutQkv = "TND"
layoutQOut = "NTD"
```

因此在 RoPE 参数组合下，默认布局对应 M03。

## 4. 四种合法模式的类型差异

| 项目 | M01 RoPE NTD→NTD | M02 RoPE TND→TND | M03 RoPE TND→NTD | M04 M-RoPE TND→TND |
|:---|:---|:---|:---|:---|
| `qkv` shape | `[Nq+Nk+Nv,T,128]` | `[T,Nq+Nk+Nv,128]` | `[T,Nq+Nk+Nv,128]` | `[T,Nq+Nk+Nv,128]` |
| `qOut` shape | `[Nq,T,128]` | `[T,Nq,128]` | `[Nq,T,128]` | `[T,Nq,128]` |
| `qOut` dtype | FP8 E4M3FN | FP8 E4M3FN | FP8 E4M3FN | BF16 |
| `qScale` | `[Nq,T]` FP32 | `[T,Nq]` FP32 | `[Nq,T]` FP32 | `None` |
| 位置输入 | `queryStartLoc + seqLens` | `queryStartLoc + seqLens` | `queryStartLoc + seqLens` | `mropePosition + mropeSection` |
| Q量化 | PerTokenPerHead | PerTokenPerHead | PerTokenPerHead | NoQuant |
| K Cache | FP8 E4M3FN | FP8 E4M3FN | FP8 E4M3FN | INT8 |
| K Scale Cache | FP32 | FP32 | FP32 | FP32 |
| V Cache | FP8 E4M3FN | FP8 E4M3FN | FP8 E4M3FN | FP8 E4M3FN |
| `vScale` shape | `[Nv]` | `[Nv]` | `[Nv]` | `[Nv,128]` |

所有模式均执行：

```text
Q: RMSNorm -> RoPE/M-RoPE -> rotation -> 量化或BF16输出
K: RMSNorm -> RoPE/M-RoPE -> rotation -> 动态量化 -> Cache
V: vScale缩放 -> FP8转换 -> Cache
```

## 5. 所有合法模式的公共测试维度

### 5.1 Head 组合

每个模式均覆盖以下 8 组 head，每组在该模式中执行 25 例：

```text
(Nq,Nk,Nv) =
(8,1,1), (16,2,2), (24,3,3), (32,4,4),
(40,5,5), (48,6,6), (56,7,7), (64,8,8)
```

每例均满足：

```text
0 < Nq <= 64
Nq = 8 * Nk
Nk = Nv
D = 128
```

### 5.2 Token 数

每个模式、每个 head 都完整覆盖以下 25 个 T：

```text
1, 2,
15, 16, 17,
127, 128, 129,
255, 256, 257,
1023, 1024, 1025,
4095, 4096, 4097,
16383, 16384, 16385,
65535, 131071, 196607,
262142, 262143
```

覆盖目标包括：

- 极小 T。
- 常用对齐粒度前后一个 token 的 tail。
- 64K、128K、192K 大规格。
- 严格约束 `0 < T < 256K` 的最后两个合法值。

### 5.3 Cache 内存形式

每个模式覆盖：

| Cache形式 | 每模式数量 | 重点 |
|:---|---:|:---|
| contiguous | 100 | 标准连续 Cache 写回 |
| d_pad | 100 | D 维带 padding 的正 stride 非连续 Cache |

三类 Cache shape 为：

```text
kCache      = [BlockNum,Nk,BlockSize,128]
vCache      = [BlockNum,Nv,BlockSize,128]
kScaleCache = [BlockNum,Nk,BlockSize,1]
BlockSize   = 16
```

每例额外保留至少 32 个未写 slot，并检查这些位置的 K、V 和 K Scale 保持为零。

### 5.4 slotMapping

每个模式覆盖以下四种唯一 slot 映射，每种 50 例：

| 类型 | 测试重点 |
|:---|:---|
| linear | 顺序写入 Cache |
| reverse | 逆序写入 Cache |
| random | 随机离散写入 Cache |
| affine | 仿射置换，覆盖跨 block 和非局部写入 |

单次调用内不生成重复 slot，因为接口对多个 token 写同一 slot 的最终顺序不作定义。

### 5.5 通用输出检查

每例均检查：

1. `qOut`、`qScale` 和三类 Cache 的 shape。
2. 各输出 dtype。
3. `slotMapping` 指定位置的数值。
4. 未指定 Cache slot 保持原值。
5. CPU Golden 与 NPU 输出的逐元素精度。
6. 执行过程中无 OOM。

## 6. M01：RoPE NTD→NTD 测试场景

### 6.1 模式合同

```text
qkv      = [Nq+Nk+Nv,T,128], BF16
qOut     = [Nq,T,128], FP8 E4M3FN
qScale   = [Nq,T], FP32
kCache   = FP8 E4M3FN
vCache   = FP8 E4M3FN
vScale   = [Nv], FP32
```

### 6.2 本类型内测试内容

- NTD 输入按 head-major 方式切分 Q/K/V 是否正确。
- NTD Q 输出和 NTD `qScale` 的 head-major 地址计算是否正确。
- T 为 1、对齐边界和接近 256K 时，NTD N/T 两轴是否被错误交换。
- 8 组 head 下输出 shape 是否均为 `[Nq,T,128]`。
- RoPE batch 数轮换覆盖 `1/2/4/8/16`；当 `T < batch` 时将 batch 收缩到不超过 T。
- 无历史位置与带历史位置偏移各 100 例。
- 连续与 D-padding Cache 各 100 例。
- 四种 slot 映射各 50 例。
- Cache 写入位置和 32 个未写 slot。

### 6.3 精度验收与结果

| 指标 | 门限 | 本模式最低值 | 结果 |
|:---|---:|---:|:---:|
| Q FP8 exact | ≥99% | 99.99348958% | PASS |
| Q反量化 close | ≥99% | 100% | PASS |
| Q scale close | ≥99% | 100% | PASS |
| K FP8 exact | ≥99% | 99.96093750% | PASS |
| K反量化 close | ≥99% | 100% | PASS |
| K scale close | ≥99% | 100% | PASS |
| V FP8 exact | 100% | 100% | PASS |

模式结果：

```text
M01: success=200, failed=0
```

## 7. M02：RoPE TND→TND 测试场景

### 7.1 模式合同

```text
qkv      = [T,Nq+Nk+Nv,128], BF16
qOut     = [T,Nq,128], FP8 E4M3FN
qScale   = [T,Nq], FP32
kCache   = FP8 E4M3FN
vCache   = FP8 E4M3FN
vScale   = [Nv], FP32
```

### 7.2 本类型内测试内容

- TND 输入按 token-major 方式读取和切分 Q/K/V 是否正确。
- TND Q 输出是否按 `[token,head,D]` 连续写出。
- `qScale[token,head]` 是否与对应 Q FP8 行一一匹配。
- T 跨 tile、跨 Cache block 和存在尾 tile 时，token 顺序是否保持。
- RoPE 多 batch 的 batch 边界是否影响 TND token 顺序。
- 无历史位置与带历史位置偏移各 100 例。
- 连续与 D-padding Cache 各 100 例。
- 四种 slot 映射各 50 例。
- 最大 `Nq=64` 与最大 `T=262143` 组合。

### 7.3 精度验收与结果

| 指标 | 门限 | 本模式最低值 | 结果 |
|:---|---:|---:|:---:|
| Q FP8 exact | ≥99% | 99.90234375% | PASS |
| Q反量化 close | ≥99% | 100% | PASS |
| Q scale close | ≥99% | 100% | PASS |
| K FP8 exact | ≥99% | 99.98586886% | PASS |
| K反量化 close | ≥99% | 100% | PASS |
| K scale close | ≥99% | 100% | PASS |
| V FP8 exact | 100% | 100% | PASS |

模式结果：

```text
M02: success=200, failed=0
```

## 8. M03：RoPE TND→NTD 测试场景

### 8.1 模式合同

```text
qkv      = [T,Nq+Nk+Nv,128], BF16
qOut     = [Nq,T,128], FP8 E4M3FN
qScale   = [Nq,T], FP32
kCache   = FP8 E4M3FN
vCache   = FP8 E4M3FN
vScale   = [Nv], FP32
```

### 8.2 本类型内测试内容

- 输入为 token-major、Q 输出为 head-major时的布局转换。
- `qOut[n,t,:]` 是否与 TND 语义下的同一个 token/head 对应。
- `qScale[n,t]` 与转换后的 Q FP8 行是否匹配。
- NTD 输出紧凑写回时，不同 head、token 之间是否发生覆盖或空洞。
- T=1 和各种 tail 下转置地址计算是否越界。
- 最大 Nq、最大 T 和 D-padding Cache 组合。
- 无历史位置与带历史位置偏移各 100 例。
- 连续与 D-padding Cache 各 100 例。
- 四种 slot 映射各 50 例。

Golden 比较时，物理 NTD 输出整体搬到 CPU，再按 token 分块转为逻辑 TND 视图进行逐元素比较；该处理仅改变比较方式，不减少比较元素。

### 8.3 精度验收与结果

| 指标 | 门限 | 本模式最低值 | 结果 |
|:---|---:|---:|:---:|
| Q FP8 exact | ≥99% | 99.99008179% | PASS |
| Q反量化 close | ≥99% | 100% | PASS |
| Q scale close | ≥99% | 100% | PASS |
| K FP8 exact | ≥99% | 99.98626709% | PASS |
| K反量化 close | ≥99% | 100% | PASS |
| K scale close | ≥99% | 100% | PASS |
| V FP8 exact | 100% | 100% | PASS |

模式结果：

```text
M03: success=200, failed=0
```

## 9. M04：M-RoPE TND→TND 测试场景

### 9.1 模式合同

```text
qkv           = [T,Nq+Nk+Nv,128], BF16
mropePosition = [T,3], INT32，列顺序为T/H/W
mropeSection  = [t,h,w]
qQuantMode    = "NoQuant"
qOut          = [T,Nq,128], BF16
qScale        = None
kCache        = INT8
kScaleCache   = FP32
vCache        = FP8 E4M3FN
vScale        = [Nv,128], FP32
```

### 9.2 本类型内测试内容

- T/H/W 三路 position 是否按 `[T,3]` 的 token-major 形式读取。
- `mropeSection` 对每个旋转维度选择 T/H/W cos/sin 行是否正确。
- Q 在 rotation 后直接写 BF16，不生成 `qScale`。
- K 在 rotation 后动态量化为 INT8，并生成 FP32 K scale。
- V 使用 `[Nv,128]` 的逐 head、逐通道 scale 后写 FP8。
- TND 输入与 TND Q 输出的 token/head 顺序。
- 连续与 D-padding Cache 各 100 例。
- 四种 slot 映射各 50 例。
- 最大 Nq、最大 T、INT8 K Cache 和大规格 position window 组合。

### 9.3 M-RoPE section 场景

M04 的 200 例均匀覆盖 5 组 section，每组 40 例：

| `mropeSection` | 测试场景 |
|:---:|:---|
| `[11,20,20]` | 普通三轴分配 |
| `[0,21,0]` | H lane 上界，T/W 为0 |
| `[0,0,21]` | W lane 上界，T/H 为0 |
| `[0,21,21]` | H/W 同时到达上界 |
| `[22,21,21]` | `t+h+w=64` 总容量上界 |

约束为：

```text
t >= 0
0 <= h <= 21
0 <= w <= 21
t + h + w <= 64
```

### 9.4 精度验收与结果

| 指标 | 门限 | 本模式最低值 | 结果 |
|:---|---:|---:|:---:|
| Q BF16 close | ≥99% | 100% | PASS |
| K INT8 `abs(error)<=1` | 100% | 100% | PASS |
| K scale close | ≥99% | 100% | PASS |
| V FP8 exact | 100% | 100% | PASS |
| `qScale is None` | 必须成立 | 200/200成立 | PASS |

模式结果：

```text
M04: success=200, failed=0
```

## 10. 不支持模式与负向场景

不支持模式不进入 NPU 数值矩阵，应在接口或 Host 校验阶段返回失败。

### 10.1 布局负向场景

| ID | 场景 | 输入→输出 | 预期 |
|:---:|:---:|:---:|:---|
| U01 | RoPE | NTD→TND | 参数非法/Host校验失败 |
| U02 | M-RoPE | TND→NTD | 参数非法 |
| U03 | M-RoPE | NTD→NTD | 参数非法 |
| U04 | M-RoPE | NTD→TND | 参数非法 |

同时检查布局字符串大小写敏感，`"ntd"` 等非标准字符串必须失败；空指针或空字符串则按接口默认布局处理。

### 10.2 场景选择负向场景

- `queryStartLoc` 和 `seqLens` 只传一个。
- `mropePosition` 和非空 `mropeSection` 只传一个。
- RoPE 和 M-RoPE 两组位置输入同时存在。
- 两组位置输入同时缺席。
- M-RoPE 中错误传入 RoPE batch 位置输入。

### 10.3 量化和 dtype 负向场景

- RoPE 使用 `NoQuant`。
- RoPE 的 `qOut` 使用 BF16。
- RoPE 的 K Cache 使用 INT8。
- M-RoPE 使用 `PerTokenPerHead`。
- M-RoPE 的 `qOut` 使用 FP8。
- M-RoPE 传入非空 `qScale`。
- M-RoPE 的 K Cache 使用 FP8。
- 未知 `qQuantMode`。

### 10.4 Head、D 和 shape 负向场景

- `Nq=0`、负数或 `Nq>64`。
- `Nq != 8*Nk`。
- `Nk != Nv`。
- `D != 128`。
- QKV head 维不等于 `Nq+Nk+Nv`。
- `mropePosition` 不是 `[T,3]` 或 dtype 不是 INT32。
- RoPE `vScale` 不是 `[Nv]`。
- M-RoPE `vScale` 不是 `[Nv,128]`。

### 10.5 M-RoPE section 负向场景

- section 长度不是3。
- 任意值为负数。
- `h>21` 或 `w>21`。
- `t+h+w>64`。
- position 与 section presence 不匹配。

仓内 ACLNN API CSV 共定义 128 个接口用例：

```text
SUCCESS           : 18
ERR_PARAM_NULLPTR : 16
ERR_PARAM_INVALID : 94
```

这些接口用例用于校验默认值、空指针、dtype、shape、布局和场景组合；它们与 800 个有效 NPU/Golden 精度用例属于不同测试层级。

## 11. STC 验收标准

### 11.1 所有合法模式

- 输出 shape 与模式合同完全一致。
- 输出 dtype 与模式合同完全一致。
- 所有有效 slot 写入正确。
- 所有未写 Cache slot 保持原值。
- 无 OOM。
- 单例失败列表为空。

### 11.2 RoPE

```text
Q FP8 exact         >= 99%
Q dequant close     >= 99%
Q scale close       >= 99%
K FP8 exact         >= 99%
K dequant close     >= 99%
K scale close       >= 99%
V FP8 exact         = 100%
```

### 11.3 M-RoPE

```text
Q BF16 close        >= 99%
K INT8 abs error<=1 = 100%
K scale close       >= 99%
V FP8 exact         = 100%
qScale              = None
```

### 11.4 不支持模式

接口返回参数非法或 Host 校验失败，不得进入 kernel 执行。

## 12. 执行结果汇总

| 模式 | 用例数 | PASS | FAIL |
|:---|---:|---:|---:|
| M01 RoPE NTD→NTD | 200 | 200 | 0 |
| M02 RoPE TND→TND | 200 | 200 | 0 |
| M03 RoPE TND→NTD | 200 | 200 | 0 |
| M04 M-RoPE TND→TND | 200 | 200 | 0 |
| **合计** | **800** | **800** | **0** |

执行过程中曾有一个长寿命 NPU 上下文中的通用 `ZerosLike` vector timeout；相同用例在新进程重跑后通过。最终去重报告中 800 个唯一用例全部为 PASS。

## 13. 测试资产与追溯关系

| 资产 | 路径 | 用途 |
|:---|:---|:---|
| 算子接口文档 | `posembedding/qkv_rms_norm_rope_cache_with_k_scale/docs/aclnnQkvRmsNormRopeCacheWithKScale.md` | 模式与接口合同 |
| Host shape/dtype测试 | `posembedding/qkv_rms_norm_rope_cache_with_k_scale/tests/ut/op_host/test_qkv_rms_norm_rope_cache_with_k_scale_infershape.cpp` | shape、dtype、布局推导 |
| Host tiling测试 | `posembedding/qkv_rms_norm_rope_cache_with_k_scale/tests/ut/op_host/test_qkv_rms_norm_rope_cache_with_k_scale_tiling.csv` | tiling key、tile与stride |
| ACLNN接口测试 | `posembedding/qkv_rms_norm_rope_cache_with_k_scale/tests/ut/op_api/test_aclnn_qkv_rms_norm_rope_cache_with_k_scale.csv` | 参数与负向校验 |
| CPU Golden | `/home/r00957730/qkv_test/qkv_rms_norm_rope_cache_with_k_scale_golden.py` | 数值参考实现 |
| 800例执行器 | `/home/r00957730/qkv_test/run_qkv_large_cases.py` | 用例生成、NPU执行、比较与报告 |
| 最终结果 | `/home/r00957730/qkv_test/qkv_full_800_final_report.jsonl` | 800条唯一PASS记录 |
| 泛化报告 | `/home/r00957730/qkv_test/QKV_FULL_800_GENERALIZATION_REPORT.md` | 覆盖和精度汇总 |

## 14. 最终结论

算子当前支持的所有 4 种合法模式均完成各自类型内的全量测试：

```text
RoPE NTD -> NTD : 200/200 PASS
RoPE TND -> TND : 200/200 PASS
RoPE TND -> NTD : 200/200 PASS
M-RoPE TND -> TND: 200/200 PASS
```

总结果：

```text
success=800
failed=0
```
