# M-RoPE Kernel 分阶段实现与 UB 优化设计

## 1. 文档范围

本文只描述 `MROPE_BF16_Q_INT8_K` 场景的 Kernel 实现顺序、UB 资源和同步方案。接口、场景识别、
数学语义及 Host 校验合同仍以
[`11_mrope_bf16_int8_extension_design.md`](./11_mrope_bf16_int8_extension_design.md) 为准。

本设计分为两个阶段：

| 阶段 | 目标 | UB 与同步原则 | 当前优先级 |
|------|------|----------------|------------|
| 初版功能实现 | 先完成 M-RoPE 端到端功能与数值验证 | 继承 RoPE 顶层资源的历史基线 | 历史设计快照 |
| 后续流水优化 | 缩短 K rotation buffer 生命周期，增强 FIX/VEC/MTE3 并行 | M-RoPE 使用独立 output preprocess、K rotation 和 K quant buffer，Q 由 Fixpipe 直接写 GM | compact 与 `[Nv,D]` V scale 已落地 |

两个阶段都只作用于编译期组合
`(ROPE_MODE=MROPE, K_QUANT_MODE=INT8, Q_QUANT_MODE=NO_QUANT)`。RoPE 的三个 layout key、UB 分配、
Q/K FP8 后处理和同步协议均保持不变。

### 1.1 当前实现状态

当前 kernel 的 M-RoPE 只落地 compact direct-Q 流水；没有可由 tiling key 选择的第二条路径。
当前布局为 80 KiB input pool、80 KiB `outputDbPoolUb_`、14 KiB K rotation DB、40 KiB compact reserve，
加上 12 KiB 的 1024-token position window，总量 226 KiB。该路径
复用 flag 1 的 `FIX_OUTPUT_READY` 发布 K rotation，使用 flag 4 的 `MROPE_K_CONSUMED` 管理反向复用，
并提供两份 K rotation consumed credit。AIV 对上一拍 `lastTile` 执行 `StageMropeK` 后先返还 rotation
credit，再在同一控制迭代立即执行 `ScatterMropeK`；K quant 双槽通过逐 slot `MTE3 -> V` event 保护复用。
Q 由 Fixpipe 完成 F32->BF16 并直写 GM，不经过 Vec QCast。

当前实现把 M-RoPE `v_scale` 从历史 `[Nv]` 改为 `[Nv,D]`。在 `Nv<=8,D=128` 下常驻 scale 最大为 4 KiB，
因此 compact reserve 为 40 KiB，position window 仍为 12 KiB，总量为
`80+80+14+40+12=226 KiB`。RoPE 的 `[Nv]` scale 和 248 KiB 普通布局不变。

当前命名和 slot 归属以物理资源为准：input 使用中性 `inputPoolUb_`/`INPUT_POOL`，M-RoPE input 与 raw
cos/sin 使用独立 `inputBufferUseId_/metadataBufferUseId_`，owner 根据各自 pool/slot 容量局部推导为 4
槽；output 前处理区域统一命名为 `outputDbPoolUb_`。common 不为每个资源定义全局 `BUFFER_NUM`，所有
ring id 仍使用本资源独立的 `useId % BUFFER_NUM`，不假设 buffer 数是 2 的幂。

第 3 节和第 5 节保留分阶段设计时的参数、时序和取舍，便于追溯；其中 `198 KiB`、`24 KiB other fixed`
及 `[Nv]` V scale 等内容是历史草案，不代表当前布局。当前合同以 `01/03/04/05/11` 为准，
`ops-transformer 6ca8b3400` 已完成代码同步和验证。`16 MiB`
workspace 属于框架强制预留，不纳入本算子的 UB 优化口径。

本文不讨论图模式，也不把性能目标作为初版功能验收门禁。

## 2. 两阶段共用约束

M-RoPE 当前固定满足：

1. $D=128$。
2. `TND->TND`。
3. $0<N_q\le64$、$N_q=8N_k$、$N_k=N_v$；这与 RoPE 场景相同，是功能和 tiling 的硬约束。
4. 每个 AIC 对应两个 AIV，tile 的 Cube token 数向 2 对齐。

令实际 tile token 数为 $T_{\mathrm{valid}}$，则

$$
T_{\mathrm{cube}}=\operatorname{AlignUp}(T_{\mathrm{valid}},2),
\qquad
C=\frac{T_{\mathrm{cube}}}{2}.
$$

$C$ 是每个 AIV 的容量，当前 host 继续保留 RoPE output pool 对 Q/K 行数的资源门禁：

$$
C(N_q+N_k)\le 128.
$$

其中 $R_k=C N_k$ 是单个 AIV slot 的 K/V 最大行数。host 必须先执行 $0<N_q\le64$、$N_q=8N_k$、
$N_k=N_v$ 的功能门禁，再根据 Q/K/V 行数、UB 容量和 tile 资源公式选择 tile。资源公式可以缩小
`tokenTile`，但不能扩大合法 head 域；完整合法 tuple 为 `(8,1,1)` 到 `(64,8,8)` 的八组整数倍组合。

尾块的 buffer 容量按 $C$ 计算，数值计算和 GM 写回只处理 `aivTokenSize` 个有效 token。例如
$T_{\mathrm{valid}}=3$ 时，$T_{\mathrm{cube}}=4$，两个 AIV 的容量均为 2，实际有效 token 数分别为 2 和
1；第二个 AIV 不得计算或写回 padding row，但不能跳过必要的同步握手。

## 3. 初版功能实现（历史设计快照）

本节记录最初的功能优先方案。它描述的 shared output DB、Q staging 和旧 output ready/consumed 方案是
演进基线；当前 compact 实现已经改为 direct-Q、统一 `FIX_OUTPUT_READY` 与独立
`MROPE_K_CONSUMED`，请以第 1.1 节和
`05_kernel_design.md` 为准。

### 3.1 实现原则

初版只增加 M-RoPE 的 T/H/W 数据准备、Gather、BF16 Q 和 INT8 K 数值分支，不重构现有 buffer
ownership 和跨核流水。具体原则如下：

1. 顶层 UB 仍包含 input DB、共享 output DB 和其他固定 buffers。
2. Q/K preprocess、Cube Fixpipe 输出以及 Q/K 后处理继续按现有顺序复用共享 output DB。
3. Q 的 BF16 转换与写回仍经过 AIV；初版不使用 Fixpipe 直接写 `q_out` GM。
4. K 的 INT8 结果继续在共享 output DB 内完成原地 staging；初版不增加独立 K rotation 或 K quant DB。
5. 历史 output-consumed edge 在 Q/K 的 MTE3 写回完成后发送，AIC 不提前覆盖 output slot。
6. M-RoPE 与 RoPE 通过 `ROPE_MODE` 编译期分流，Q/K 后处理分别通过 `Q_QUANT_MODE/K_QUANT_MODE`
   编译期分流；不得通过 runtime dtype、nullable 地址或 `mrope_section` 数值再次判定分支。

这一路径的目标是最大限度复用已验证的 RoPE buffer 生命周期，先排除新算法、接口和数据搬运中的功能
问题。FIX 等待 MTE3 导致的并行度损失是初版已知代价，不在功能阶段同时处理。

### 3.2 顶层 UB 分配

初版沿用当前 248 KiB 静态分配：

| 区域 | 相对 Offset | 分配大小 | M-RoPE 初版用途 |
|------|------------:|---------:|------------------|
| input DB pool | `0 KiB` | `2 * 40 KiB = 80 KiB` | `qkv` tile staging，ping/pong |
| shared output DB pool | `80 KiB` | `2 * 64 KiB = 128 KiB` | Q/K BF16 NZ preprocess、Q/K Cube FP32 输出、BF16 Q 与 INT8 K staging |
| other fixed buffers | `208 KiB` | `40 KiB` | gamma、raw T/H/W、位置元数据、V output 及索引 |
| 合计 | - | `248 KiB` | 与 RoPE 当前顶层分配相同 |

这里的“继承”指各类 buffer 的大小、offset、共享 output DB 的复用关系和 owner 计数保持不变。M-RoPE
编译期分支绑定场景专用的 raw T/H/W、`mrope_position` 和 Gather index；其他固定 buffers 的静态布局上限为 40 KiB。

### 3.3 初版其他固定 buffer 分配

初版优先保留现有子区容量，避免功能实现同时收缩 V、`slot_mapping` 和 `v_scale` buffer。M-RoPE
其他固定 buffer 分配如下：

| 子区 | 分配大小 | 单 slot 有效数据上界 | 生命周期 |
|------|---------:|---------------------:|----------|
| Q/K gamma | `1 KiB` | `2 * 128 * sizeof(float) = 1 KiB` | 循环前加载，常驻 |
| raw T/H/W cos/sin DB | `12 KiB` | 每 slot `3 * 4 * 128 * sizeof(float) = 6 KiB` | MTE2 写、VEC 读，ping/pong |
| `slot_mapping` DB | `1 KiB` | 每 slot 最多 `4 * sizeof(int32_t) = 16 B` | MTE2 写、Scalar 读，沿用每 slot 512 B 容量 |
| `v_scale` | `512 B` | 历史 `[Nv]` 合同，最多 `80 * sizeof(float) = 320 B` | 循环前加载，常驻；已被目标 `[Nv,D]` 合同取代 |
| Q/K NZ scatter index | `512 B` | 两张 `128 * sizeof(uint16_t)` 表 | 循环前构建，常驻 |
| V output / K scale staging DB | `20 KiB` | 通用资源门禁 $C N_k\le80$，V 每 slot 最多 `80 * 128 B = 10 KiB` | VEC 写、MTE3 读；K scale 按 float view 复用，每 slot 10 KiB |
| `mrope_position` DB | `256 B` | 当前 cap=4 时每 slot最多 `4 * 3 * sizeof(int32_t) = 48 B`；按 kernel 通用上限分配 128 B/slot | MTE2 写、Scalar/Vector 读 |
| M-RoPE Gather index | `256 B` | `64 * sizeof(uint32_t) = 256 B` | 循环前构建，常驻 |
| 已分配合计 | `35.5 KiB` | - | 静态布局上限为 40 KiB |
| 未使用空间 | `4.5 KiB` | - | 初版不分配新用途 |

`mrope_position` 的 128 B 是单 slot 的 UB 容量，不是 GM 搬运长度。当前 host cap=4 时，每个 AIV 必须只从 GM 读取

$$
3\times \mathrm{aivTokenSize}\times \operatorname{sizeof}(\mathrm{int32\_t})
$$

字节，并使用等价的尾块安全路径在 UB 侧补齐；禁止按 128 B 容量越界读取 GM。

初版其他固定 buffers 的静态布局上限为 40 KiB，其中 35.5 KiB 已分配、4.5 KiB 未使用。相对 256 KiB
物理 UB，现有 248 KiB 顶层静态布局之外另有 8 KiB 未使用；这两部分未使用空间不连续，也不能相加后
描述成一个可直接分配的连续 buffer。

### 3.4 初版数据流

单 tile 的逻辑数据流为：

```text
MTE2: qkv + slot_mapping + mrope_position + raw T/H/W -> UB
VEC:  V quant/cache staging + M-RoPE Q/K preprocess
MTE3: Q/K BF16 NZ preprocess -> L1
AIC:  Q/K MMAD + Fixpipe FP32 -> shared output DB
VEC:  Q FP32 -> BF16; K FP32 -> INT8 + FP32 scale
MTE3: q_out + k_cache + k_scale_cache -> GM
```

Q 的 `q_quant_mode=NoQuant`，不执行 max/reduce/div，不分配或写出 `q_scale`。K 仍按逐 token、逐 head
计算 FP32 scale；INT8 K 原地复用 shared output slot，K scale staging 按 float view 复用当前 V output
slot，随后分别写入 `k_cache` 与 `k_scale_cache`。K cache 与 scale cache 的 GM 地址必须使用各自实际
stride，不能从逻辑 shape 重新推导连续 stride。只有 K scale 的 MTE3 写回也完成后，V output slot 才可
归还 reusable event。

### 3.5 初版同步

初版复用 RoPE 当前的四个 CrossCore 语义和 flag id：

| 语义 | Producer -> Consumer | 初版 M-RoPE 含义 |
|------|----------------------|------------------|
| `A_READY` | AIV `PIPE_MTE3` -> AIC `PIPE_MTE1` | 两个 AIV 的 Q/K preprocess 已写入 L1 |
| `A_CONSUMED` | AIC `PIPE_MTE1` -> AIV `PIPE_MTE3` | 对应 L1 logical A buffer 可复用 |
| output ready | AIC `PIPE_FIX` -> AIV `PIPE_V` | 共享 output slot 中的 Q/K FP32 可读 |
| output consumed | AIV `PIPE_MTE3` -> AIC `PIPE_FIX` | Q/K 对应 GM 写回已完成，共享 output slot 可覆盖 |

初版不新增 M-RoPE 专用 ready/consumed flag，也不改变 output-consumed 的 producer pipe。历史的一份
output credit、
`outputBufferUseId_` 顺序和 Init/End drain 逻辑继续生效。即使某个 AIV 在尾块的 `aivTokenSize=0`，
BasicBlock 仍必须参与与另一个 AIV 配平的 CrossCore Set/Wait；空数值分支不能从 controller 或
BasicBlock 提前返回。

核内仍沿用现有 input、V output 和 shared Q/K output 的 reusable event。M-RoPE 新增的 raw T/H/W 与
`mrope_position` ready/consumed 关系应优先复用同一 input slot 的生命周期；如果实现必须拆分 ready
事件，只允许增加该场景的核内事件，不得改变 RoPE event id、pipe pair 或 Set/Wait 次数。

Q/K NZ scatter index 与 M-RoPE T/H/W gather index 在 `PrepareBeforeLoop` 中由标量流水一次性构造。
所有 `SetValue` 完成后通过同一条 `S -> PIPE_V` event 发布，VF 才能读取；索引在 tile loop 中只读，
因此不增加反向 reusable event，也不改变 Init/End credit。

### 3.6 初版明确不做的优化

初版不包含以下改造：

1. Q L0C 经 Fixpipe 直接写 `q_out` GM。
2. 独立的 K rotation FP32 双缓冲。
3. 独立的 K INT8 与 K scale staging 双缓冲。
4. 将 K rotation consumed 从 MTE3 完成提前到 VEC 完成。
5. 将 128 KiB shared output DB 拆成更小的场景专用区域。
6. 为推测的 bank conflict 预先硬编码 padding。

## 4. 初版验收门禁

初版完成的判定以功能正确为主：

1. `mrope_position` 使用 `[T,3]` token-major 布局；Golden 中所有 M-RoPE case 默认按 case seed
   在合法范围内独立均匀随机生成 T/H/W 三列位置。
2. `mrope_section=[t,h,w]` 的 lane 选择正确，三项之和满足 $t+h+w\le D/2$。
3. `q_out` 为 BF16，不计算或写出 `q_scale`。
4. `k_cache` 为 INT8，逐元素与 golden 的有符号整数差值绝对值不超过 1，且所有元素均满足该标准。
5. `k_scale_cache` 为 FP32，并按实际 cache stride 写回。
6. 覆盖 odd tail、跨 tile、ping/pong wrap 和 drain；至少包含 $T\in\{1,3,4,5,17\}$。
7. 覆盖从 `(8,1,1)` 到 `(64,8,8)` 的八组合法 head tuple，包含 `(24,3,3)` 等非 2 的幂场景；
   $N_q>64$、$N_q\ne8N_k$ 或 $N_k\ne N_v$ 均作为 host 拒绝路径。
8. M-RoPE 端到端无卡死；空 AIV slice 的同步计数保持配平。
9. RoPE 的三个 layout、FP8 Q/K 数值和原有同步回归保持通过。

初版不以 FIX/VEC/MTE3 overlap 或性能提升作为通过条件。功能验证稳定后，先冻结一组相同 shape、相同
输入和相同 cache stride 的基线结果，再进入后续优化。

## 5. 后续流水优化（核心思路已落地；本节数字保留为历史草案）

### 5.1 优化目标

后续优化只改变 M-RoPE 的资源归属和流水，不改变第 2 节的输入合同与第 4 节的数值标准。当前代码已
落地其中的 direct-Q、K rotation/K quant 解耦、统一 `FIX_OUTPUT_READY`、独立
`MROPE_K_CONSUMED`；本节的旧资源表仅用于记录演进过程：

1. Q 不再返回 AIV UB，由 Fixpipe 完成 FP32 到 BF16 转换并直接写 `q_out` GM。
2. K 的 FP32 rotation 结果写入专用双缓冲。
3. VEC 将 K rotation 量化到独立的 INT8 K 与 FP32 scale staging。
4. VEC 完整读完 K rotation 并写完独立 staging 后立即返还 K rotation credit，不等待 MTE3。
5. BasicBlock 在返还 credit 后的同一控制迭代立即发起 K cache/scale scatter；K quant slot 的核内 event
   允许 MTE3 写回与后续 AIC/AIV 工作重叠。

当前稳态控制顺序为：

```text
current tile i: preprocess -> A_READY(i)
lastTile i - 1: wait FIX_OUTPUT_READY
                -> StageMropeK(i - 1)
                -> Set MROPE_K_CONSUMED
                -> ScatterMropeK(i - 1)
```

这是一拍 current/last AIC/AIV rolling，不再额外跨控制迭代保存 K scatter 状态。Q direct-GM 与 K
buffer credit 相互独立。K rotation slot 的生命周期在 VEC 量化完成时结束；K quant slot 的生命周期在 K cache 与 K
scale 两路 MTE3 均完成后结束。两个 slot id 来自各自资源 counter，不能从 tile i 推导，也不能让 K quant
借用 K rotation 的 id。

### 5.2 优化后的顶层 UB 草案（历史参数）

历史优化方案曾针对 $R_k=C N_k\le14$ 的常见 shape 选择 M-RoPE 专属静态 layout。以下段落保留
当时的资源推导，不是当前 host 的第二条路径合同；当前实现会降低 token tile，最小 tile 仍不可行则拒绝：

| 区域 | 大小 | 说明 |
|------|-----:|------|
| input DB | `2 * 40 KiB = 80 KiB` | 保持双缓冲 |
| output DB | `2 * 40 KiB = 80 KiB` | `outputDbPoolUb_` 的两个 Q/K 前处理 slot；一块 40 KiB scratch 不能承载跨 tile 流水 |
| K rotation FP32 DB | `2 * 7 KiB = 14 KiB` | 每 slot `14 * 128 * sizeof(float) = 7168 B` |
| other fixed buffers | `24 KiB` | persistent、metadata/raw、V output、K quant/scale |
| 合计 | `198 KiB` | M-RoPE 专属，不替换 RoPE 的 248 KiB layout |

顶层 buffer 静态分配 198 KiB；在 248 KiB 设计预算内尚有 50 KiB 未分配。256 KiB 物理 UB 另有 8 KiB
不纳入该设计预算，因此物理 UB 合计未分配 58 KiB。

### 5.3 优化后的其他固定 buffer 布局（历史参数）

建议按 256 B 生命周期组组织这些辅助 buffer：

| 相对 Offset | 区域 | 大小 | 有效内容 |
|-------------:|------|-----:|----------|
| `0x0000` | Q/K gamma | `0x0400` | 1024 B |
| `0x0400` | `v_scale` | `0x0040` | 历史 `[Nv]` scale，最多 56 B，补齐到 64 B；目标布局不再使用 |
| `0x0440` | Q/K NZ index | `0x0200` | 512 B |
| `0x0640` | M-RoPE Gather index | `0x0100` | 256 B |
| `0x0740` | persistent padding | `0x00C0` | 生命周期组补齐 |
| `0x0800` | `mrope_position` DB | `0x0100` | `2 * 128 B`，当前 cap=4 时每 slot 有效最多 48 B |
| `0x0900` | `slot_mapping` DB | `0x0040` | `2 * 32 B`，每 slot 有效最多 16 B |
| `0x0940` | metadata padding | `0x00C0` | 生命周期组补齐 |
| `0x0A00` | raw T/H/W DB | `0x3000` | `2 * 6144 B` |
| `0x3A00` | V output DB | `0x0E00` | `2 * 1792 B` |
| `0x4800` | K quant DB | `0x1200` | `2 * 2304 B` |
| `0x5A00` | unused | `0x0600` | 1536 B，暂不分配 |
| `0x6000` | fixed buffer layout end | - | 24 KiB |

K quant 每个 2304 B slot 包含：

$$
\underbrace{14\times128\times1}_{\text{INT8 K}=1792\ \mathrm B}
+
\underbrace{14\times32}_{\text{K scale MTE3 staging}=448\ \mathrm B}
+
\underbrace{64}_{\text{padding}}
=2304\ \mathrm B.
$$

K scale 的有效数据仍是每行一个 FP32，32 B 是 MTE3 staging row 的物理跨度。Q 在该场景没有 q_scale，
该布局不为 q_scale 分配空间。raw T/H/W 由 VF 直接 Gather 到寄存器，不增加 fused/effective cos-sin UB。
`cacheBaseOffset` 和 `scaleCacheBaseOffset` 保持为 `TileParam` 中的标量数组，也不计入 LocalTensor buffer 布局。

各逻辑子区按自身最小对齐后的容量合计为 22656 B；按上表的生命周期组和 padding 后占用 23040 B，
即 22.5 KiB，24 KiB 静态分配中有 1536 B 未使用。实现需增加等价的编译期边界检查。该 1536 B
未使用空间仅供 profiling 确认后增加 bank-phase padding，功能实现阶段不分配新用途。

### 5.4 优化后的 buffer 生命周期（当前语义的历史描述）

| Buffer | Producer -> Consumer | 可复用条件 |
|--------|----------------------|------------|
| input/raw/metadata slot | MTE2 -> VEC/Scalar | VEC 已读完 Q/K/V、raw T/H/W；metadata 的 Scalar 读取已完成 |
| Q/K preprocess slot | VEC -> MTE3 | MTE3 已将 Q/K BF16 NZ 全部写入对应 L1 logical A buffer |
| V output slot | VEC -> MTE3 | V cache 写回完成 |
| K rotation slot | AIC Fixpipe -> AIV VEC | VEC 已完整读取 FP32 K，并写完独立 K quant/scale slot |
| K quant slot | VEC -> MTE3 | K cache 和 K scale cache 两路 GM 写回均完成 |

禁止以下 alias：

1. K rotation 与 K quant 不能 alias，否则无法在 VEC 完成时提前返还 rotation credit。
2. V output 与 K quant 不能 alias，相邻 tile 可能同时占用二者。
3. raw/position 与 K quant 不能 alias，跨 tile 时 MTE2 和 MTE3 生命周期可能重叠。
4. Gather index 与 Q/K NZ index 不能 alias，同一个 VF 阶段需要同时读取两类索引。
5. input 与 Q/K preprocess 不能 alias，VF 会同时读取 input 并写 preprocess。

### 5.5 优化后的同步（当前实现）

RoPE 与 M-RoPE 复用同一个固定语义 `FIX_OUTPUT_READY`；M-RoPE 不再分配独立 ready flag：

| M-RoPE 语义 | Producer -> Consumer | 含义 |
|--------------|----------------------|------|
| `FIX_OUTPUT_READY`，flag 1 | AIC `PIPE_FIX` -> AIV `PIPE_V` | 对应 K rotation FP32 slot 可读 |
| `MROPE_K_CONSUMED`，flag 4 | AIV `PIPE_V` -> AIC `PIPE_FIX` | VEC 已读完 K rotation 并写完独立 K quant/scale slot |

K rotation 为双缓冲，因此 M-RoPE 初始化 2 个 `MROPE_K_CONSUMED` credit。每个 tile 中 AIC 在覆盖 slot
前消费一个 credit，AIV 在完成量化 staging 后归还一个 credit；End 再 drain 2 个 credit。若共有 $N$
个 tile，则 `FIX_OUTPUT_READY` 的 Set/Wait 均为 $N$ 次，`MROPE_K_CONSUMED` 的 Set/Wait 均为 $N+2$
次。`A_READY` 和 `A_CONSUMED` 继续沿用现有语义与 A_rot 物理槽计数。

对 non-empty slice，完整顺序是
`Wait FIX_OUTPUT_READY -> StageMropeK(lastTile) -> Set MROPE_K_CONSUMED -> ScatterMropeK(lastTile)`；
Stage 和 Scatter 位于同一控制迭代。`MROPE_K_CONSUMED` 由 `PIPE_V` 发送，只返还 rotation credit，不得
等待 K cache 或 K scale 的 MTE3。K quant 双槽在 Stage 前分别 wait `MTE3 -> V` reusable event，Scatter
完成两路 MTE3 后再 set 对应 event。

`aivTokenSize=0` 只跳过数值工作，仍需 `Wait FIX_OUTPUT_READY -> Set MROPE_K_CONSUMED`，以保证两个 AIV
的 CrossCore 计数一致；空 slice 不占 K quant slot，不推进 `kQuantBufferUseId_`，也不发起数值 scatter。

核内双缓冲 event 建议按物理 slot 分配：

| Pipe pair | Buffer group | event id |
|-----------|--------------|----------|
| MTE2 -> V | QKV input ready | `0/1` |
| MTE2 -> V | raw T/H/W ready | `2/3` |
| V -> MTE2 | input/raw/metadata reusable | `0/1` |
| MTE2 -> S | metadata ready | `0/1` |
| V -> MTE3，MTE3 -> V | V output DB | `0/1` |
| V -> MTE3，MTE3 -> V | output DB | `2/3` |
| V -> MTE3，MTE3 -> V | K quant DB | `4/5` |

同一数值 id 可在不同 pipe pair 复用；同一 pipe pair 内，每个具体 ping/pong slot 必须有独立 event。
K quant reusable 只能在 K cache 和 K scale 两路 MTE3 都完成后返回。所有 event 必须在 Init 建立初始
credit，并在 End 对称 drain；tail 和空 AIV 路径不得漏掉 Set/Wait。

## 6. 分阶段落地顺序（历史记录）

1. 先按第 3 节完成通用 M-RoPE 实现，建立功能基线。
2. 通过第 4 节全部功能门禁，并记录多 tile 下 FIX、VEC、MTE3 的基线 timeline。
3. 单独提交 $R_k\le14$ 的 M-RoPE UB 性能特化，引入独立 Q/K preprocess、K rotation 和 K quant
   buffers；该提交不得改 RoPE 常量、通用 M-RoPE fallback 或 host 功能合同。
4. 单独提交 Q direct-GM 与 K rotation/quant 解耦同步，静态审计 CrossCore 和核内 event 的 Init、循环、
   tail、空 AIV、End 计数。
5. 使用与初版完全相同的功能用例复验，再对比 pipeline overlap 和性能；若优化失败，可回退 M-RoPE
   专用优化实例而不影响 RoPE 或初版功能合同。

上述顺序是历史实施记录。当前代码、静态资源断言、数值路径和同步实现已经落地；后续只需维护
单一 compact 模板，并以常规及大输出 case 覆盖，不应再按“第 5 节尚未实现”或“第二条 M-RoPE
路径仍是可选项”理解当前 kernel。16 MiB workspace 属于框架强制预留。
