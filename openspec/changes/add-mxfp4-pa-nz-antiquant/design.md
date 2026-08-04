## Context

本变更是 reference-port：它不改变 MXFP4 mode 6 的量化数学定义，而是将已有非 PA NZ 路径扩展为可按 `blockTable` 消费 PA cache。已有 PA KV loader 已区分 NZ、BNBD 和 BBH；已有非 PA mode 6 NZ scale loader 已按 group-plane 访问；已有伪量化 FP8 PA 路径已展示 page segment 的 `blockTable` 寻址模式。

本 change 仅冻结并实现 FIA 的 PA_NZ。当前树中的 FIA 直接实例化自己的 Arch35 antiquant kernel；standalone IFA 不是该调用链的一部分，且没有 `FLOAT4_E2M1` / mode 6 kernel 模板。IFA 的等价支持必须作为独立 change 设计、实现和验证，不能由 FIA 改动隐式宣称完成。

符号：

```text
B  = logical batch size
N  = KV head count
S  = logical KV sequence length
P  = total physical page count
Bs = page block size
D  = logical head dimension
G  = D / 32
```

## Goals / Non-Goals

**Goals:**

- 在 FIA 中支持 `FLOAT4_E2M1`、mode 6、`kvLayout=NZ`、`kvStorageMode=PAGE_ATTENTION`。
- 支持 offset 为空的对称反量化和 offset 存在的非对称反量化。
- 固定 scale/offset 的三阶段 producer contract，并让 kernel 只消费最终的 PA_NZ storage。
- 使用一套 page-segment 迭代抽象，避免未来 BNBD/BBH 支持复制 page-table 处理、跨页切分和 scale/offset 分支。
- 保持现有非 PA mode 6 的接口、地址计算和数值路径不变。

**Non-Goals:**

- 本 change 不支持 BNBD、BBH 或其他 PA KV layout。
- 本 change 不新增可配置 groupSize；只支持 `groupSize=32`。
- 本 change 不改变非 PA mode 6，也不为它新增 offset。
- 本 change 不做性能调优、吞吐基准或其他 FP4/FP8/INT8 格式改造。

## Decisions

### 1. PA_NZ 参数是三阶段 producer contract，而非 kernel 内转换

量化/PA cache producer SHALL 依次形成：

```text
logical parameter: [B,N,S,G]
page parameter:    [P,N,Bs,G]
PA_NZ storage:     [P,N,G,Bs]
```

Page 化使用：

```text
p = blockTable[b, s / Bs]
u = s % Bs
page[p,n,u,g] = logical[b,n,s,g]
nz[p,n,g,u] = page[p,n,u,g]
```

FIA 不在运行时执行上述 transpose；它验证并消费最后一项 `[P,N,G,Bs]`。这一约定采用 PA cache 在 producer 侧完成 storage formatting 的模型，避免每次 attention 执行都在 UB 内做 token/group 转置。

`scale` 和非空 `offset` SHALL 使用相同的 index transformation；二者只在元素 dtype 和元素字节数上不同。

### 2. 按 page segment 搬入 NZ group-plane 参数

一次 attention tile 的逻辑 token 区间可能跨多个物理 page。共享的 page-segment 迭代器 SHALL 产生：

```text
(physicalPage, tokenInPage, segmentLength, destinationTokenOffset)
```

对每一个 page segment，NZ 参数源地址的逻辑坐标为 `[physicalPage, kvHead, group, tokenInPage]`。实现 SHALL 对每个 `group` 发起一次 `blockCount=1` 的连续 DMA，并直接写入 UB 的 `[G, alignedTileS]` 对应行。跨 page 时 SHALL 循环多个 segment，不能假设跨页参数连续。

不能将这一操作编码为一次 `blockCount=G` 的目的端 strided DMA：首/尾 page segment 的 `len` 可以不是 32B 对齐，而当前 `DataCopyPad` 目的 stride 的 API 编码不适合表达任意字节 gap。每 group 的连续搬运避免该未对齐 gap，并保持相同的逻辑数据流。

该迭代器和跨页控制流 SHALL 作为可复用的内部抽象；未来 BNBD/BBH 可复用它，仅替换最内层的 TOKEN_MAJOR 搬运策略。当前 change 只实例化 GROUP_MAJOR/NZ 策略。

### 3. scale 与 offset 使用不同数据通道，并在 UB 中分别展开

`scale` 的 GM 元素是一个 E8M0 原始字节，必须以 `uint8_t` 地址单位搬运和解码。`offset` 的 GM 元素是 `Q_T`（FP16 或 BF16），必须以 `Q_T` 地址单位搬运。二者 SHALL 使用不同的 GM 类型、copy length/stride 单位及 UB buffer 视图，不能通过 `/2` 或类型重解释共享地址计算。

scale SHALL 由现有/等价的 VF helper 解码并展开为 NZ group-plane 的 BF16 计算参数。非空 offset SHALL 从 `Q_T` 转换并展开为同一 group-plane 视图。UB 的 tile stride、copy-loop slice 和 tail mask 必须对 scale/offset 分别计算并与 VF 消费布局一致。

### 4. VF 数学在 BF16 中执行，且 symmetric path 不变

对每个逻辑 FP4 元素 `x[d]`，令 `g=floor(d/32)`：

```text
symmetric:  y[d] = decode_fp4(x[d]) * scale[g]
asymmetric: y[d] = (decode_fp4(x[d]) + offset[g]) * scale[g]
```

VF SHALL 先将 FP4 decode 结果、E8M0 scale 和非空 Q_T offset 转为 BF16，再执行 Add（仅非对称）与 Mul，最后按既有舍入规则写入 Q_T。对称路径 SHALL 不读取 offset、不增加 Add，也不改变既有 mode 6 计算路径。

### 5. FIA 完成完整路由；IFA 不在本 change 内

FIA SHALL 在 checker 接受目标组合、在 tiling 中把 mode 6 选为 per-token-group 并启用 PA antiquant 路由、在 Arch35 processor/VF 中选择 PA_NZ parameter loader。

独立 IFA 不在 FIA 的运行时调用链内；当前树也未提供可承载 FP4 mode 6 的 IFA kernel 模板。因此本 change 不修改 IFA，且任何交付说明不得将 FIA 的证据表述为 IFA 证据。

### 6. 非目标组合保持隔离

对 mode 6 + PageAttention，BNBD 与 BBH SHALL 被显式判定为本 change 不支持的组合，不能落入 PA_NZ loader。非 PA mode 6 SHALL 保持现有对称参数 contract 和原有地址计算。其他量化 mode、groupSize 非 32、非 `FLOAT4_E2M1` 输入 SHALL 不因本 change 改变行为。

## Risks / Trade-offs

- **PA 参数 producer 尚未按最终 NZ storage 生成数据** → 在实施前补充 producer/test preprocessing；不接受 kernel 静默按逻辑 `[B,N,S,G]` 读取作为替代。
- **跨 page 的 group-plane DMA、tail 或 offset 地址单位错误** → 以 non-identity `blockTable`、跨页 tile、不同 group/head 值的数值用例验证；目标 CANN 环境必须编译并运行。
- **scale/offset UB 展开布局不一致** → scale 与 offset 分离 buffer，逐 copy-loop 验证，并做 BF16/FP16 asymmetric 用例。
- **standalone IFA 当前不具备 mode 6 路由** → 明确不在本 change 内；后续 IFA 支持须单独评审其 host、tiling 和 kernel 设计。
- **仅做 NZ 导致调用方误以为 BBH/BNBD 已支持** → host 层显式拒绝目标 mode 下的非 NZ PA layout，并在接口文档和测试中覆盖。

## Migration Plan

1. 合入并验证 FIA PA_NZ 的 contract、路由、搬运和 VF。
2. 部署前由 PA cache/量化参数 producer 输出最终 `[P,N,G,Bs]` scale 及可选 offset；调用方不能继续传逻辑 `[B,N,S,G]` 作为 PA_NZ 参数。
3. 若发现问题，回滚本 change 的 PA_NZ route 即可；既有非 PA mode 6、BNBD、BBH 不受修改。

## Open Questions

- 本仓库当前环境缺少目标 CANN 编译/执行环境。最终 `DataCopyPad` 对部分 segment 的支持、UB 容量、事件同步和 BF16 rounding 必须由目标环境的编译与数值测试确认；这不是可由静态审查替代的结论。

## Implementation Blueprint

本节是实施约束，不是“可选优化建议”。实现者必须按本节的数据流、地址单位和生命周期实现；若目标 CANN 版本的 API 单位与本节的语义公式不同，必须保留语义、调整 API 编码，并以编译和数值用例证明等价。

### 1. Source reference map and exact change surface

| Layer | Existing reference | Required change for this change |
|---|---|---|
| FIA mode routing | `attention/fused_infer_attention_score/op_host/arch35/fused_infer_attention_score_tiling_impl.cpp`, `UpdateTilingKeyQuantMode` | retain mode 6 → `AntiquantMode_K_PER_TOKEN`; make PA-NZ target eligible and set the PA antiquant flag only for the target tuple |
| FIA template selection | `attention/fused_infer_attention_score/op_host/fused_infer_attention_score_tiling.cpp`, `attention/fused_infer_attention_score/op_kernel/fused_infer_attention_score_template_tiling_key.h` | allow the target `K_PER_TOKEN + PA_NZ` key and reject unsupported PA layouts before a key is emitted |
| FIA host contract | `attention/fused_infer_attention_score/op_host/checkers/dequant_checker.{h,cpp}` | add a dedicated mode-6 PA-NZ checker; do not reuse a flattened BNBD checker or a logical non-PA checker |
| FIA PA page traversal | `attention/fused_infer_attention_score/op_kernel/arch35/flash_attention_score_antiquant_processor.h`, `CopyKVPageAttention` | reuse its logical-range → physical-page segmentation pattern, not its KV byte offset formula |
| FIA existing NZ group input | same processor, `CopyAntiqScaleE8M0Nz`, `CopyAntiqOffsetNz` | preserve its group-plane UB/VF contract, but add a new PA-NZ source-addressing helper; the old helper remains non-PA only |
| FIA buffer setup and task flags | `attention/fused_infer_attention_score/op_kernel/arch35/flash_attention_score_antiquant_block_vec.h`, `InitAntiquantBuffer`, `SetAntiquantTaskParam` | derive buffers from the required tile capacity; pass existing page/NZ flags to the new branch |
| FP4 vector arithmetic | `attention/common/op_kernel/arch35/vf/vf_antiquant_w4.h`, `AntiquantVFImplW4PerTokenGroupNz`, `AntiqScalePerTokenGroupByVF`, `AntiqOffsetByVFNz` | use the NZ scale/offset preparation helpers and dispatch the existing/new `hasOffset` VF specialization with BF16 parameter views |

The current MR's `LoadAntiquantParamsPerTokenGroup` checks `isKvCacheNz` before checking its PA flag. That branch computes a logical `bIdx/n2Idx/seqSize` parameter offset and calls `CopyAntiqScaleE8M0Nz`; it is not a PA-NZ implementation. The required branch order is:

```text
if (isPageAttentionAntiquant && isKvCacheNz) {
    load final PA_NZ parameter storage through blockTable;
} else if (isKvCacheNz) {
    retain the existing non-PA NZ loader unchanged;
} else {
    retain the existing non-NZ loader unchanged;
}
```

### 2. Data model, shapes, contiguity and ownership

The following are different objects. They must never be treated as alternative views of the same GM allocation during attention execution.

| Object | Owner / time | Logical or physical shape | Element type | Contiguity required by this change |
|---|---|---|---|---|
| logical scale | quantization producer, before PA cache construction | `[B,N,S,G]` | E8M0 | producer-local |
| logical offset | quantization producer, before PA cache construction, optional | `[B,N,S,G]` | Q_T | producer-local |
| page scale intermediate | PA cache producer only | `[P,N,Bs,G]` | E8M0 | producer-local |
| page offset intermediate | PA cache producer only, optional | `[P,N,Bs,G]` | Q_T | producer-local |
| final PA_NZ scale input | FIA GM input | `[P,N,G,Bs]` | E8M0 / `uint8_t` | dense C-order; `Bs` is contiguous and `G` rows have stride `Bs` |
| final PA_NZ offset input | FIA GM input, optional | `[P,N,G,Bs]` | Q_T | same dense coordinate order and strides as scale |
| FP4 PA_NZ KV input | FIA GM input | `[P,N,D/16,Bs,16]` for FP4 NZ | packed FP4 | existing KV-NZ contract; not changed by this change |
| scale raw tile | VECIN queue UB | `[G,SAlign]` | raw `uint8_t` held in queue storage | token range contiguous within every group row |
| offset raw tile | VECIN queue UB, optional | `[G,SAlign]` | Q_T | token range contiguous within every group row |
| expanded scale tile | TBuf UB | `[G,SAlign]` | BF16 bit representation held in `ANTIQUANT_PARAMS_T` storage | consumed group-row by group-row by NZ VF |
| expanded offset tile | TBuf UB, optional | `[G,SAlign]` | BF16 bit representation held in `ANTIQUANT_PARAMS_T` storage | exactly matches expanded scale coordinates |
| FP4 decoded output | VECOUT UB then L1 | existing NZ output tile | Q_T | existing NZ KV result contract |

`G` is not an FP4 packing unit. It is a quantization semantic unit: `G=D/32`. A value at D coordinate `d` uses group `floor(d/32)`. FP4 stores two logical values per byte, while the PA_NZ KV D0 storage block is 16; neither fact changes the group size of 32 logical D values.

The final parameter inputs SHALL be dense in the last two axes. The first implementation SHALL require a contiguous `[P,N,G,Bs]` storage tensor rather than accepting arbitrary user strides, because the DMA source offset is calculated from the physical coordinate formula below. Supporting non-contiguous parameter tensors is explicitly out of scope for this change.

### 3. Producer-side transformation: the only permitted three-shape conversion

The producer, not the attention kernel, creates final PA_NZ storage. For each parameter kind independently (K scale, V scale, optional K offset, optional V offset):

```text
Input logical coordinate: logical[b,n,s,g]

p = blockTable[b, floor(s / Bs)]
u = s % Bs

page[p,n,u,g] = logical[b,n,s,g]       // [P,N,Bs,G]
final[p,n,g,u] = page[p,n,u,g]         // [P,N,G,Bs]
```

Equivalent reference pseudocode:

```python
# params is scale or a non-null offset, shape [B, N, S, G]
page = zeros([P, N, Bs, G], dtype=params.dtype)
for b in range(B):
    for s in range(actual_seq_len[b]):
        p = block_table[b, s // Bs]
        u = s % Bs
        page[p, :, u, :] = params[b, :, s, :]
final = page.permute(0, 1, 3, 2).contiguous()  # [P,N,G,Bs]
```

The MXFP8 PA-NZ preprocessing reference uses the same two concepts—first scatter by `blockTable`, then NZ-specific storage reordering—although its FP8 scale has additional pair packing. It is a reference for page ownership and pre-formatting, not a license to copy its `16,2` dimensions into MXFP4.

There is no fourth input shape accepted by the operators. In particular, a logical `[B,N,S,G]`, a page-token-major `[P,N,Bs,G]`, or a flattened buffer SHALL fail the PA-NZ mode-6 shape validation in this change.

### 4. Page segmentation and exact GM address contract

For a single antiquant task, let:

```text
S0          = taskParam.s2BatchOffset       // first logical KV token of the task
T           = taskParam.copyTotalS          // number of logical tokens whose parameters are loaded
M           = taskParam.maxBlockNumPerSeq   // blockTable second dimension
n           = taskParam.n2Idx               // current KV head
Bs          = taskParam.kvCacheBlockSize
SAlign      = AlignUp(T, 32)
```

The loader SHALL iterate page segments, not individual tokens and not a single flat source range:

```text
copied = 0
curS = S0
while copied < T:
    logicalBlock = curS / Bs
    u            = curS % Bs
    p            = blockTable[bIdx * M + logicalBlock]
    len          = min(Bs - u, T - copied)
    copy one [G,len] group-plane rectangle from page p
    copied += len
    curS   += len
```

For the final dense `[P,N,G,Bs]` parameter tensor, the source coordinate and row start are:

```text
element(p,n,g,u) = (((p * N + n) * G + g) * Bs + u)
rowStart(p,n,u)  = ((p * N + n) * G) * Bs + u
```

The destination coordinate for that segment is:

```text
dst(g, copied + t) = g * SAlign + copied + t, 0 <= t < len
```

Thus one segment consists of `G` independent contiguous transfers:

```text
for g in [0, G):
  source = element(p,n,g,u)
  destination = g * SAlign + copied
  DataCopyPad(blockCount=1,
              blockLen=len * sizeof(ParamT),
              srcStride=0,
              dstStride=0)
```

For scale, `ParamT=uint8_t`; for offset, `ParamT=Q_T`. `DataCopyExtParams.blockLen` is therefore `len` raw E8M0 bytes for scale and `len * sizeof(Q_T)` bytes for offset. The source and destination offsets are type-correct element indices; no scale/offset address is reused across types. A compile test plus a `G>1`, `len<Bs` numerical case is mandatory for this point.

The page-segment loop is shared infrastructure. The future BNBD/BBH implementation will reuse its `p/u/len/copied` calculation, but will have a different, token-major inner DMA policy. This change contains only the GROUP_MAJOR/NZ policy.

### 5. GM → UB → Reg → UB → L1 dataflow

The required FIA dataflow is:

```text
KV GM [P,N,D/16,Bs,16]
  -- existing CopyKVPageAttention / CopyKVPaNz --> KV VECIN UB

scale GM [P,N,G,Bs], uint8
  -- new page-segment GROUP_MAJOR DataCopyPad --> scale raw VECIN UB [G,SAlign]
  -- EnQue / DeQue synchronization -->
  -- AntiqScalePerTokenGroupByVF (E8M0 -> BF16) --> scale TBuf [G,SAlign]

offset GM [P,N,G,Bs], Q_T, optional
  -- new page-segment GROUP_MAJOR DataCopyPad --> offset raw VECIN UB [G,SAlign]
  -- EnQue / DeQue synchronization -->
  -- AntiqOffsetByVFNz (Q_T -> BF16) --> offset TBuf [G,SAlign]

KV VECIN UB + scale/offset TBuf
  -- AntiquantVFImplW4PerTokenGroupNz --> decoded Q_T VECOUT UB
  -- existing CopyAntiquantResToL1BaseAPI --> antiquant result L1
  -- existing cube/attention flow --> output GM
```

The existing TQue lifecycle is mandatory for each raw parameter input:

```text
AllocTensor -> DataCopyPad -> EnQue -> DeQue -> VF preparation -> FreeTensor
```

The expanded scale/offset reside in distinct `TBuf` allocations because they survive beyond freeing the raw queue tensors and are sliced during the later `copyLoop` iterations. A raw scale tensor, raw offset tensor, expanded scale tensor and expanded offset tensor SHALL never alias.

The existing KV GM→UB and result UB→L1 code is not rewritten. The only changed GM dataflow is scale/offset. Existing `CopyKVPageAttention` is the reference for page crossing; it already loops while a tile spans pages and calls `CopyKVPaNz` for NZ KV.

### 6. UB sizing, padding and `copyLoop` slicing

The current source hard-codes 3072-byte scale queues, 6144-byte offset queues and 5 KiB result TBufs. Those constants SHALL NOT be retained merely because they compile. Capacity must be derived from the actual maximal tiling tuple accepted by the target route.

For one loaded parameter tile:

```text
SAlign = AlignUp(copyTotalS, 32)
E      = AlignUp(G * SAlign, 128)  // VF load/store granularity

raw scale queue bytes   >= E
raw offset queue bytes  >= E * sizeof(Q_T), when offset exists
expanded scale TBuf     >= E * sizeof(bfloat16_t)
expanded offset TBuf    >= E * sizeof(bfloat16_t), when offset exists
```

`AntiqScalePerTokenGroupByVF` and `AntiqOffsetByVFNz` currently load/store whole 128-lane register blocks. Their public code computes a tail count but their low-level implementation still issues 128-lane operations. Therefore the producer/loader must make every rounded tail readable and every expanded tail writable.

For the padding tokens `t in [T,SAlign)`, scale raw bytes SHALL be initialized to E8M0 `0x00`, which the existing helper maps to BF16 `1.0`; offset raw elements SHALL be initialized to zero. The implementation may use `DataCopyPad` padding fields or an explicit `Duplicate`/fill plus the necessary event synchronization, but it must not leave rounded tail lanes uninitialized.

Parameters are loaded once for `copyTotalS` when `taskParam.isLoadAntiqParam` is true, while the KV is processed in:

```text
loopCnt = ceil(copyTotalS / copySplitS)
current token window i = [i*copySplitS, i*copySplitS + dealRowCount)
```

For PA_NZ expanded `[G,SAlign]` tensors, the `i`th VF invocation receives the **same token offset in every group row**:

```text
paramBaseForLoop = i * copySplitS
scaleSlice  = expandedScale[paramBaseForLoop]
offsetSlice = expandedOffset[paramBaseForLoop]
```

It SHALL NOT multiply this per-loop base by `G`, by two, or by `sizeof(Q_T)`. The NZ VF adds its group-row stride `SAlign` internally. This is why the scale and offset slices have the same scalar base even though their GM raw types differ.

### 7. VF implementation contract

`AntiquantVFImplW4PerTokenGroupNz` is the required arithmetic route for `FLOAT4_E2M1` PA_NZ. Its existing organization processes NZ tiles in 8-row/16-column register blocks, uses an aligned group-row parameter stride, and invokes a `hasOffset` specialization.

For each register block the intended MicroAPI sequence is:

```text
LoadAlign<uint8_t, DIST_UNPACK4_B8>(FP4 bytes)  // unpack packed FP4
Cast<bfloat16_t>(decoded FP4)
if constexpr (hasOffset):
    LoadAlign<DIST_E2B_B16>(offset values for this group/token lanes)
    Add<bfloat16_t>(decoded, offset)
LoadAlign<DIST_E2B_B16>(scale values for this group/token lanes)
Mul<bfloat16_t>(value, scale)
StoreAlign Q_T (BF16 direct; FP16 through the existing BF16->FP16 cast trait)
```

The offset pointer supplied to this VF is an expanded BF16 view created by `AntiqOffsetByVFNz`; it is not a raw FP16 buffer reinterpreted as BF16. For `Q_T=FP16`, the conversion to BF16 must occur before the Add. For `Q_T=BF16`, the conversion helper may preserve the representation. The scale pointer is likewise an E8M0-decoded BF16 view.

The symmetric route SHALL instantiate/call `hasOffset=false`; it must not allocate, load or add an offset. The asymmetric route SHALL instantiate/call `hasOffset=true` and use the corresponding scale and offset slice. Both must use the same FP4 decode, output cast and NZ result-store path.

### 8. Host, tiling and template responsibilities

#### FIA

1. `dequant_checker.cpp` SHALL check all of the following before tiling:
   - K and V dtype are `FLOAT4_E2M1`;
   - K/V antiquant mode are both 6;
   - PageAttention is enabled and KV layout is NZ;
   - `D % 32 == 0`, `P=totalBlockNum`, `N=n2Size`, `G=D/32`, `Bs=blockSize`;
   - key/value scale shape exactly `[P,N,G,Bs]` and dtype `FLOAT8_E8M0`;
   - absent offset is legal; present key/value offset shape exactly matches its scale and dtype exactly equals Q's FP16/BF16 dtype;
   - scale/offset required storage axes are contiguous;
   - BNBD/BBH mode-6 PA is rejected here, before a kernel key is emitted.
2. `fused_infer_attention_score_tiling_impl.cpp` SHALL map antiquant mode 6 to the existing K-per-token quant category.
3. `fused_infer_attention_score_tiling.cpp` SHALL set `antiquantPageAttentionFlag` for the exact mode-6 PA-NZ tuple. It must not set the flag for non-PA mode 6 or unsupported PA layouts.
4. `fused_infer_attention_score_template_tiling_key.h` SHALL expose only keys whose template has the PA/NZ KV path and mode-6 FP4 processor compiled in. The emitted key must carry the existing PA/NZ layout selection used by `taskParam.paKvShapeType`.
5. `flash_attention_score_antiquant_block_vec.h` SHALL transfer the resulting flags to `taskParam.isPageAttentionAntiquant` and `taskParam.isKvCacheNz`, allocate derived parameter buffers, and pass both `blockTableGm` and the two parameter GM tensors to the processor.

### 9. Required implementation order

1. Implement and unit-test the producer/reference transformation `[B,N,S,G] -> [P,N,Bs,G] -> [P,N,G,Bs]` before editing kernel logic.
2. Add FIA checker and tiling route, with a negative test proving BNBD/BBH cannot reach the route.
3. Add the FIA PA-NZ page-segment loader for scale only; use unique raw E8M0 values to prove the GM→UB coordinate map.
4. Add the matching offset loader and raw/expanded buffer separation; prove offset coordinates with unique values.
5. Integrate both helpers into the PA-NZ branch before the old non-PA NZ branch.
6. Enable `hasOffset` VF dispatch and run symmetric/asymmetric numerical tests.
7. Compile and run the FIA acceptance matrix. No stage may be described as complete based only on source review.

### 10. Required reference test data and oracle checks

The reference test generator SHALL use a case that makes every addressing mistake observable:

```text
D=64 (G=2) or larger; N>=2; B>=2; Bs chosen so a tile crosses a page;
blockTable non-identity and non-monotonic across batches;
K and V scale values unique by (p,n,g,u);
K and V offsets unique by (p,n,g,u), including negative values;
FP4 values that exercise both groups and both packed nibbles.
```

The oracle SHALL first reconstruct logical scale/offset from final PA-NZ `[P,N,G,Bs]` through `blockTable`, then apply group-wise FP4 dequantization. It shall not reuse kernel address formulas as its only source of truth. FIA must be tested for:

- symmetric FP16 Q;
- asymmetric FP16 Q;
- asymmetric BF16 Q;
- non-identity page mapping;
- a parameter tile crossing at least one page boundary;
- `copyTotalS` not divisible by 32 and a final `copySplitS` tail;
- K and V with different parameter values;
- non-PA mode-6 regression;
- expected rejection of BNBD, BBH, invalid `[P,N,G,Bs]`, invalid offset dtype and `D % 32 != 0`.

The target-environment numerical gate is the direct ACLNN example
`attention/fused_infer_attention_score/examples/arch35/test_aclnn_fused_infer_attention_score_v5_antiquant_mxfp4_pa_nz.cpp`.
It uses `B=2`, `N=2`, `D=128`, `G=4`, `Bs=128`, `P=4`, actual KV lengths 130/129, and
`blockTable=[[2,0],[3,1]]`. It executes FP16 symmetric, FP16 asymmetric and BF16 asymmetric calls. The KV cache
is physical NZ `[P,N,D/16,Bs,16]`; K/V scales and optional offsets are independently filled in final
`[P,N,G,Bs]` order with values unique by `(p,n,g,u)`, including negative offsets. Its CPU golden follows logical
tokens through `blockTable`, exercises both packed FP4 nibbles, applies the BF16 antiquant arithmetic, and compares
the returned output. The wrapper `scripts/ci/run_fia_mxfp4_pa_nz_validation.sh` invokes this example through
`build.sh --run_example ... cust` and requires the caller to point `FIA_CUSTOM_OPP_PATH` at the custom package
built from this exact checkout; it must never fall back to an installed stale `opapi_transformer` library.
