# M-RoPE MX 场景合同

> 状态：公开合同与历史内部路径快照。SIMT 与 `simt_vk` 相关章节只记录已经结束的内部实验，
> 其实现文件和私有路由已从当前交付源码删除；当前实现与验收事实以 `docs/srs_stc/mrope_mx_srs.md`
> 和源码为准。本文不是最终全矩阵验收报告。
> 目标源码基线：`ops-transformer@4efa682af807043e5639ae969b8d0a7fe5799e9f`。
> MX 语义参考：`ops-nn@b29b90227b8ff1a1dea97c3319cd167f9a5c8982` 中 `DynamicMxQuantV3`，固定 `scaleAlg=1`。
> 测试工具基线：`ops-test-kit@c1bac9c6298a649ec18cc7e02951a6fd234765ee`。

## 0. 与原设计冲突或特别重要、仍待确认的项

以下项目优先于正文中的一般说明。标为“待确认”的实现偏离，不能被单点精度或性能结果自动视为已接受。

| ID | 当前源码事实 | 状态 |
|---|---|---|
| 公开场景边界 | 公开 scene 仍只有本文第 2 节三种组合。SIMT 和 `simt_vk` 都不是新增 scene，不增加公开属性、输入或输出；三者只是同一 M-RoPE MX 合同的内部实现。 | 已实现，需保持此边界。 |
| A/B 私有开关 | Host 使用进程环境变量 `QKV_MROPE_MX_IMPL`：未设置或 `vector` 选择 Vector，`simt` 选择原 SIMT，`simt_vk` 选择隔离 VK 调度；非法值只让新 M-RoPE MX scene 失败，旧 scene 忽略。三路必须在独立进程中运行，避免 framework tiling/kernel cache 复用旧选择。 | 不是公开合同字段，不得写入 ACLNN/Torch schema。 |
| 公共 ABI | 为区分 K 的 `PerTokenPerHead` 与新增 `MxQuant` 算法，新增 ACLNN V2；V1 函数原型保持不变并固定补 `kQuantMode=PerTokenPerHead`。OpDef 保留既有 `q_out_dtype` 索引 6，只把 `k_quant_mode` 追加为索引 7。 | 已实现；V1 二进制/源码 ABI 与旧调用方兼容，新场景和 Torch wrapper 调 V2。 |
| D11 UB layout | 原设计要求 Host/Device 使用单一 helper。实际 Host 使用带 checked arithmetic 的 `TryMakeMropeMxUbLayout`，device 使用 uint32 算术的 `MakeMropeMxUbLayoutDevice`，以避开 device compiler-rt 宽乘依赖；两者是同一公式的两次物化。当前尚无逐字段 exact-equality UT。 | 是否接受该偏离并补 exact-equality UT，仍待用户确认。 |
| D25 M-RoPE lane 规则 | 第 6.1 节公式是规范源，但当前 Kernel Vector gather-index、SIMT `MropeMxSimtAxis` 与 Golden 各自实现相同 lane 规则，没有跨 Kernel/Golden 的单一 helper。 | 已实现但存在漂移风险；是否抽取共享事实源仍待用户确认。 |
| D26 地址窄化 | Host 采用拒绝式 shape/stride/tile/窄字段验证；device 只消费已验证 tiling，公共 copy helper 中仍有直接 `static_cast<uint16_t/uint32_t>`，没有逐处重复 checked cast。 | 是否接受 Host 作为安全边界，仍待用户确认。 |
| 当前 Vector RowBatch16 | 所有合法 `Nq` 都调用同一个 Q VF。Q 把每个 token 拆成若干16-row完整批次和可选8-row尾批；K在自己的融合VF中逐行处理。Host和device按 `max(1,floor(Nq/16))*8,576B` 分配 Q scratch。 | 不按具体head组合选择函数；不改变公开 scene、数值或 GM 写集合。 |
| 当前 SIMT 调度 | `512 threads=16 warps`；每个 warp 分成两个 16-lane subgroup，一次处理两个 head。V 以 head-pair 为行，K/Q 在 16-head group 内按 head-pair 遍历。 | 已取代旧的一 warp/一 head 实现；公开合同不变。 |
| 隔离 `simt_vk` 调度 | `512 threads=16 warps`；一个 warp 拥有一个 token，按 K/V head-pair 执行 `V->K`，全部 VK 完成后复用原 SIMT Q。 | 不保持“所有 V 后才开始任一 K”的全局阶段序；V/K 输出不重叠，公开数值合同不变。 |
| 验证边界 | 当前 RowBatch16 Vector 源码已完成干净构建；`T=768,Nq=16/24/32/64` 的 ACLNN 五输出均为 `100%`。固定seed 42、`reuse_hbm=false`、TTK内部预热和49次计时的结果分别为 `14.474/20.901/21.517/32.669 us`。 | 这四组是当前源码身份的代表性证据，不等于八组head和1200条全矩阵验收。详见 `08_validation_evidence.md`。 |

## 1. 目标与边界

现有 `QkvRmsNormRopeCacheWithKScale` 已增加一个仅支持 `TND -> TND` 的 M-RoPE MX scene：

- Q：`RMSNorm -> M-RoPE -> Dynamic MX Quant`；
- K：`RMSNorm -> M-RoPE -> Dynamic MX Quant -> cache scatter`；
- V：`per-head-per-channel static quant -> cache scatter`；
- Q/K 不再做 `rotationOptional` 矩阵乘；
- 默认 Vector、内部 SIMT 与隔离 `simt_vk` 都为 AIV-only，不使用 Cube/AIC、MMAD、Fixpipe、L0A/L0C、L1，也不使用 AIV/AIC CrossCore 同步；
- 既有 RoPE 和既有 M-RoPE 的数值、布局及输出语义保持不变。

范围保持为：

- 不新增算子名；为保持 V1 ABI 并承载 `kQuantMode`，新增 ACLNN V2；
- 不新增公开 scale 输出，复用 `qScaleOptional` 与 `kScaleCacheRef`；
- 不实现 TorchAir converter；当前只保证 Torch eager 通路，ACLGraph 若验证也只能单独记为 ACLGraph 结果；
- 本文不把某一内部实现的性能写入公开算子合同。Vector/SIMT/SIMT-VK 测量只作为当前实现证据。

## 2. 场景判定矩阵

位置输入先区分 RoPE/M-RoPE，Q/K mode 再区分旧 M-RoPE 与新 M-RoPE MX。只接受下表三种组合；其他交叉组合返回参数错误。

| 场景 | 位置输入 | `qQuantMode` | `kQuantMode` | rotation | Q/K 量化 |
|---|---|---|---|---|---|
| 既有 RoPE | `queryStartLocOptional`、`seqLensOptional` 非空；M-RoPE 位置输入缺席 | `PerTokenPerHead` | `PerTokenPerHead` | 必须非空 | 每 token、每 head，既有算法 |
| 既有 M-RoPE | `mropePositionOptional[T,3]` 与 `mropeSectionOptional=[t,h,w]` 同时存在；RoPE 位置输入缺席 | `NoQuant` | `PerTokenPerHead` | 必须非空 | Q 不量化；K 为既有 INT8 动态量化 |
| 新 M-RoPE MX | 同上 | `MxQuant` | `MxQuant` | 必须为空 | Q/K 为 MX FP8 动态量化 |

`qQuantMode`、`kQuantMode` 的空串或空指针都解析为默认值 `PerTokenPerHead`。K 的量化算法由 `kQuantMode` 指定，目标数据类型仍由实际 `kCacheRef` dtype 指定；算法与 dtype 是两个独立维度，tiling key 也分别表达它们。

Vector/SIMT/SIMT-VK 选择不参与本表 scene 判定。三者使用相同输入、输出、数值和 cache 写集合。

## 3. 公共接口变化

新增 `kQuantMode` 的直接原因是 K quant 新增了 `MxQuant` 算法，需要与既有 `PerTokenPerHead` 明确区分。已有 `qQuantMode` 不能独立表达 K 算法；仅依赖 `kCacheRef` dtype 又会把“算法”与“输出存储类型”错误耦合。

`kQuantMode` 只区分 K quant 算法：

- `PerTokenPerHead`：选择既有 K 量化路径；
- `MxQuant`：选择 Dynamic MX Quant 路径。

它不是 dtype 选择器。空串或空指针解析为 `PerTokenPerHead`，以保持既有调用的 scene 语义。

旧 ACLNN V1 原型保持不变，并将 K mode 固定为 `PerTokenPerHead`。新增 V2 在 `qQuantMode` 后携带 `const char *kQuantMode`：

```cpp
..., const aclIntArray *mropeSectionOptional,
const char *qQuantMode,
const char *kQuantMode,
aclTensor *qOut,
aclTensor *qScaleOptional,
uint64_t *workspaceSize,
aclOpExecutor **executor
```

L0/OpDef 属性顺序为：

```text
0 head_nums
1 layout_qkv
2 layout_q_out
3 epsilon
4 mrope_section
5 q_quant_mode
6 q_out_dtype
7 k_quant_mode
```

`q_out_dtype` 始终保留在索引 6；`k_quant_mode` 只能追加为索引 7。这样旧的七属性序列仍按原义解析，缺失的 K mode 默认 `PerTokenPerHead`。Torch wrapper 需要表达新场景，因此调用 V2。

Torch 两个 eager API同步增加 keyword-only 参数：

```python
k_quant_mode: str = "PerTokenPerHead"
```

同时 schema 与真实行为一致：

- `query_start_loc`、`seq_lens` 为 `Tensor?` / `Optional[Tensor]`；
- `q_scale` 返回为 `Tensor?` / `Optional[Tensor]`，因为既有 M-RoPE 公开返回 `None`。

`QKV_MROPE_MX_IMPL` 不属于上述接口，也不进入 OpDef 属性或 Torch schema。

## 4. 新场景输入合同

当前实现约束：

```text
D = 128
0 < Nq <= 64
Nq = 8 * Nk
Nk = Nv
1 <= T <= 262144
layoutQkv = TND
layoutQOut = TND
```

| 输入 | shape | dtype | presence/语义 |
|---|---|---|---|
| `qkv` | `[T,Nq+Nk+Nv,D]` | BF16 | Q/K/V 融合输入 |
| `qGamma` | `[D]` | FP32 | Q RMSNorm gamma |
| `kGamma` | `[D]` | FP32 | K RMSNorm gamma |
| `cosSin` | `[MaxSeqLen,D]` | FP32 | 每行前 `D/2` 为 cos、后 `D/2` 为 sin |
| `slotMapping` | `[T]` | INT32 | `0<=slot<BlockNum*BlockSize` 且所有 slot 唯一 |
| `kCacheRef` | `[BlockNum,Nk,BlockSize,D]` | FP8 E4M3FN | 输入输出别名 |
| `vCacheRef` | `[BlockNum,Nv,BlockSize,D]` | FP8 E4M3FN | 输入输出别名 |
| `kScaleCacheRef` | `[BlockNum,Nk,BlockSize,ceil(D/32)]` | FP8 E8M0 | 输入输出别名 |
| `queryStartLocOptional` | - | - | 必须为 `nullptr` / `None` |
| `seqLensOptional` | - | - | 必须为 `nullptr` / `None` |
| `rotationOptional` | - | - | 必须为 `nullptr` / `None` |
| `vScaleOptional` | `[Nv,D]` | FP32 | 每 head、每 channel 静态量化系数 |
| `mropePositionOptional` | `[T,3]` | INT32 | 每 token 的 T/H/W 三路 `cosSin` 行号 |
| `headNums` | `[3]` | INT64 array | `[Nq,Nk,Nv]` |
| `mropeSectionOptional` | `[3]` | INT64 array | `[t,h,w]`，保留既有 M-RoPE lane 选择语义 |

除三个 cache ref 外，新 scene 的输入输出按表中逻辑 shape 使用 dense contiguous 存储。ACLNN L2 对输入执行 contiguous，L0/Infer 分配连续 `qOut/qScaleOptional`。

三个 cache ref 允许 view：

- `BlockNum`、`BlockSize` 必须一致；
- K/V Cache 前三维 element stride 必须相同，最后一维连续；
- `kScaleCacheRef` 必须 rank 4，尾部 scale 轴长度为 `ceil(D/32)`，最后一维 stride 为 1，前三维 stride 为正。

除上述尾维规则外，每个 cache view 都必须按语义维序非重叠。令 `span=1`，从最内轴向最外轴遍历：当该轴 `dim>1` 时必须满足 `stride>=span`，随后以 checked arithmetic 更新 `span += (dim-1)*stride`；乘加溢出也拒绝。该规则允许 padding，但拒绝 token/head/block 任一层的地址覆盖。

每个 token/head 的 K scale GM 区域包含 `ceil(D/32)` 个连续 E8M0 字节；当前 `D=128` 时为 `[4]`，即 4 字节。

`mropePositionOptional` 的设备值由调用方保证 `0 <= position < MaxSeqLen`。Host tiling 不遍历 device tensor，device 也不做逐元素截断或取模；违反该前置条件的行为未定义。

## 5. 新场景输出合同

公开/DFX 输出顺序为：

```text
[qOut, qScaleOptional, kCacheRef, vCacheRef, kScaleCacheRef]
```

| 输出 | shape | dtype | 说明 |
|---|---|---|---|
| `qOut` | `[T,Nq,D]` | FP8 E4M3FN | Q MX 量化值 |
| `qScaleOptional` | `[T,Nq,ceil(D/32)]` | FP8 E8M0 | Q 每 D32 block 的 scale |
| `kCacheRef` | `[BlockNum,Nk,BlockSize,D]` | FP8 E4M3FN | 只更新 `slotMapping` 命中位置 |
| `vCacheRef` | `[BlockNum,Nv,BlockSize,D]` | FP8 E4M3FN | 只更新命中位置 |
| `kScaleCacheRef` | `[BlockNum,Nk,BlockSize,ceil(D/32)]` | FP8 E8M0 | K 每 D32 block 的 scale |

令：

```text
B = ceil(D/32)       # 有效 scale 数，也是物理存储数
```

有效 scale 线性顺序固定为 `block0 ... block(B-1)`。scale 不转置、不交织，也不按照 T/H/W lane 重排。当前 D=128 时 `B=4`。

未命中的 K/V/KScale cache 区域必须逐 bit 保留原值。

三个 scene 的 qScale presence 规则为：

- 既有 RoPE：非空、FP32，shape 随 Q 输出布局为 `[T,Nq]` 或 `[Nq,T]`；
- 既有 M-RoPE：公开输出为 `None` / `nullptr`，L0 可创建仅内部可见的 placeholder；
- 新 M-RoPE MX：非空、E8M0 `[T,Nq,ceil(D/32)]`，不得使用 placeholder。

## 6. 计算语义

### 6.1 RMSNorm 与 M-RoPE

Q/K 对每个 token/head 的 D 维向量计算；当前合法实现固定 D=128：

```text
rms = sqrt(sum(x[d]^2) / D + epsilon)
y[d] = fp32(x[d]) / rms * gamma[d]
```

随后复用既有 M-RoPE 的 T/H/W lane 选源和 half-split 旋转：

```text
low'  = low * cos - high * sin
high' = high * cos + low  * sin
```

BF16 输入在 FP32 计算域连续完成 RMSNorm 和 M-RoPE，之后直接进入 MX 量化；两者之间没有 BF16 rounding boundary，也没有 rotation 矩阵乘。第 0 节 D25 说明的是“实现实体未统一”，不是公式语义允许变化。

### 6.2 Q/K Dynamic MX Quant

每个 D32 block 独立执行 `DynamicMxQuantV3` 的语义：

```text
axis = -1
blocksize = 32
scaleAlg = 1
roundMode = rint
dstType = FLOAT8_E4M3FN
scaleType = FLOAT8_E8M0
dstTypeMax = 0.0
maxLowBound = 0.0
```

对一组 `x[0:32]` 求 `amax=max(abs(x))`，依据 E4M3FN 上限生成 FP32 scale，将其向上规整为 E8M0 可表示的 2 的幂，再用倒数缩放并转换为 E4M3FN。zero、NaN、Inf、指数上溢/下溢、`rint` 和饱和转换必须与锁定的 ops-nn 实现逐项一致。

### 6.3 V 静态量化

V 不做 RMSNorm、M-RoPE 或 MX：

```text
vQuant[u,n,d] = cast_fp8_e4m3fn(fp32(v[u,n,d]) * vScale[n,d])
```

### 6.4 Cache scatter

对 `slot=slotMapping[u]`：

```text
block  = slot / BlockSize
offset = slot % BlockSize
```

K、KScale、V 写入 `[block,head,offset,...]`；Q 按 token-major 连续写 `qOut/qScale`。Vector 与 SIMT 的 GM 写集合相同。

## 7. Workspace、确定性与内部实现

- 保留现有 workspace ABI；新 scene 的三种 AIV-only 实现都不增加任务自有 GM workspace，也不使用 workspace 承载 Q/K 中间值；
- 不新增用于 Q/K 中间结果的 GM buffer；
- 算子保持确定性，前提是 `slotMapping` 值域合法且全局唯一；
- Vector key 为 `302059776`，使用 UB tile、双输入槽、position 单窗口和 V/K/Q staging；Q 对所有合法 head 配置使用运行时 RowBatch16 VF，K使用运行时逐行融合VF；Q `mxScratch` 为 `max(1,floor(Nq/16))*8,576B`；
- SIMT key 的设计值为 `4597027072`，固定 `512 threads/16 warps`；每个 warp 分成两个 16-lane subgroup，每个 subgroup 负责一个 head。V 任务是 `(token,V-head-pair)`；Q/K 任务是至多 16 head 的 `(token,head-group)`，组内每次并行处理两个 head；每个 sublane 处理每个 D32 block 的相邻两个元素并直接读写 GM；
- `simt_vk` key 为 `8891994368`；一个 warp 先固定 token，再按 head-pair 执行 V static quant 与 K RMSNorm/M-RoPE/MX，随后复用现有 Q VF；不新增 UB、workspace 或数值公式；
- V/K/Q data 对齐时用 2-byte FP8 pair store，K/Q 四个 E8M0 scale 对齐时用 4-byte packed store；允许 padded cache stride，因此 V/K data 与 K scale 保留非对齐逐 byte fallback；
- 三个 key 都是同一公开 scene 的内部实现，不改变本合同的 shape、dtype、数值或 alias 语义。

## 8. 测试合同与当前证据

测试资产位于算子目录内，ops-test-kit 负责发现和执行，不承载本算子公式：

```text
单一语义 Golden core
  -> Kernel TestSpec 适配
  -> ACLNN TestSpec 适配
  -> Torch E2E TestSpec 适配
CSV 只描述 API 路由、shape/dtype/属性与 case
```

验收矩阵必须覆盖：

- smoke、token tail、三 tile 以上的 buffer 复用；
- position 1023/1024/1025 边界；
- slot 跨 cache block、乱序且互不重复；
- `Nq/Nk/Nv=8/1/1` 与 `64/8/8`；
- 各 D32 block 不同数量级和 `[ceil(D/32)]` scale 顺序；
- FP8 命中位置、E8M0 raw byte、未命中 cache bit-equal；
- Torch inplace 与 functional；functional 还需证明输入 cache 整块不变；
- 既有 RoPE 与既有 M-RoPE 回归；
- `T=262144` 独立 stress，不能因资源跳过后仍报告总体 PASS。

截至当前 RowBatch16 Vector 源码身份，有效证据分层如下；旧实现证据只作历史对照：

| 类别 | 结果 |
|---|---|
| Host/op-api UT | Host UT 已在新增路由后重跑 `155/155 PASS`；op-api UT `137/137 PASS` 属于接口未变化的前一轮证据 |
| 当前 BUILD/身份 | VF SHA256 `d43e6bba...7323`；layout SHA256 `89d11e1d...754b`；4a3 object SHA256 `c419b316...9d2b`；干净构建通过 |
| 当前代表shape精度 | `Nq=16/24/32/64`的ACLNN用例均五输出`100%` |
| 当前 PERF | TTK ACLNN，固定seed 42、内部warmup、`--run=50`、`reuse_hbm=false`：`14.474/20.901/21.517/32.669 us` |
| 性能分解 | 同一token的完整Q批次共享三个屏障；相对上一版MX在四组shape上降低`3.66%/1.40%/6.89%/14.51%` |
| 历史实现 | 旧性能和全矩阵结果不替代当前RowBatch16 Vector验证 |
| 当前未重跑 | `Nq=8/40/48/56`、priority-0全矩阵、最大T、Torch E2E、详细msopprof、wheel和D11 layout exact-equality UT |

当前 RowBatch16 Vector 已完成四组代表shape的干净构建、ACLNN精度和TTK ACLNN性能闭环，但没有完成当前源码身份的全矩阵、最大T、Torch E2E、详细msopprof和wheel。

## 9. 当前对齐门

实现已经进入代码和部分验证阶段，旧的“评审通过前不进入实现”门不再适用。当前只剩以下需要用户显式决定或继续验证的门：

1. D11：是否接受 Host/Device 双物化 UB layout，并增加逐字段 exact-equality UT 防漂移；
2. D25：是否抽取跨 Vector/SIMT/Golden 的 M-RoPE lane 共享事实源；
3. D26：是否接受 Host 拒绝式验证作为 device 直接 cast 的安全边界，或增加 device 防御；
4. 为当前 RowBatch16 Vector 源码身份补跑其余四组head、priority-0全矩阵、最大T、Torch E2E、详细msopprof和wheel；当前四组ACLNN精度与TTK ACLNN性能已通过，证据索引见`08_validation_evidence.md`；
5. 任何合同、Golden、tiling key 或实现选择变更都会使现有构建、精度和性能证据失效，必须重新干净构建并匹配重跑。
