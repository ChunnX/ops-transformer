# megaMoe swigluQuant port4 适配报告

## 结论先说

当前分支是 `port4-swigluquant-upstream-master`。

本次最终代码只修改了一个文件：

- `mc2/mega_moe/op_kernel/arch35/block_epilogue_swiglu_mx_quant.h`

我重新审查后的结论是：

- 当前 upstream/master 默认路径，也就是 `IsInterleaved_ = false` 的非 interleaved 路径，代码结构是自洽的。
- 非 interleaved 路径现在按 256x256 的最大片内输出规模申请中间 buffer，不再依赖 128x256 假设。
- interleaved 路径已经在 swigluQuant 文件内按模板预留，核心 VF 计算复用同一套实现；但 upstream/master 目前没有端到端启用 interleaved 调用点，所以它属于“代码已准备，接入后仍需编译、精度、性能验证”的状态。
- scale copyout 现在不再做 UB 内 32B 行 staging，直接从 compact scale buffer 拷到 GM；这符合当前 scale 生成方式，也减少了一段 vector 搬运。
- `git diff --check upstream/master...HEAD` 已通过；旧的 `TransMxScaleLayout`、`quantScaleBlockOutput_`、旧 `CopyScaleFromUb2Gm` 已无残留引用。
- 代码注释已重新整理：保留/恢复仍适用的 upstream/master 原注释，并在新增 interleaved、compact scale copy、VF 模板入口处补了说明。

需要诚实说明：本轮没有完成完整编译、精度和性能复跑。之前构建卡在外部依赖拉取失败，不是代码编译错误；但“最终可合入”仍建议补一次 CI/目标 shape 实测。

## 为什么只改这个文件

你的要求是“只改我们的 swigluQuant 文件，interleaved 和非 interleaved 都适配，当前默认 interleaved=false”。所以这版没有再改 GMM 调度、workspace、flag reset/read/write 等其它模块。

这意味着当前 master 的主流程保持不变：

1. GMM1 负责把两半输出写到 UB。
2. swigluQuant epilogue 从 UB 读这两半数据。
3. vector 做 SwiGLU + MX scale + fp8/fp4 quant。
4. y 和 scale 拷回 GM。
5. atomicAdd 通知下游。

## 入口和接口变化

类模板多了一个编译期参数：

```cpp
template <..., bool IsTensorList_, bool IsInterleaved_ = false>
class BlockEpilogueSwigluMxQuant;
```

这个参数默认是 `false`，所以当前 master 里已有的 5 参数实例化不需要改调用点，默认仍走非 interleaved。

更具体地说，当前 upstream/master 里类似下面的实例化仍然只传 5 个模板参数：

```cpp
BlockEpilogueSwigluMxQuant<SwigluQuantOutType, bfloat16_t,
    QuantScaleOutType, QuantScaleOutType, true>
```

C++ 会自动使用 `IsInterleaved_ = false`。因此现在没有 interleaved 调用点时，编译出来的默认代码仍是非 interleaved 路径；`if constexpr (IsInterleaved_)` 下的 interleaved 分支不会成为运行时分支。

`operator()` 多了一个可选参数：

```cpp
operator()(blockShape, blockCoord, uint16_t pingpongIdx = 0)
```

当前 master 仍用两个参数调用，因此 `pingpongIdx` 默认是 0，不改变现有行为。未来 interleaved pingpong 接入时，可以传 0/1 选择当前 half UB。

这个设计的目的不是现在打开 interleaved，而是在 swigluQuant 文件里先把 ABI 和内部地址选择准备好：

- 当前：`swigluQuantOp(epilogueShape, epilogueOffset)`，默认 `pingpongIdx=0`。
- 未来：`swigluQuantOp(epilogueShape, epilogueOffset, pingpongIdx)`，只需要调用方传入当前 pingpong half。
- VF 主计算不需要再复制一份 interleaved 专用大函数，因为源 stride 由模板参数决定。

## UB buffer 怎么理解

这段代码涉及三类本地 UB 区域：

- `VECIN`：GMM1 写进来的输入，也就是 SwiGLU 的两半输入。
- `VECCALC`：vector 计算中间结果，比如 bf16 的 SwiGLU 结果、max exponent、halfScale。
- `VECOUT`：准备 copyout 到 GM 的量化结果 y 和 scale。

当前实现里，输入仍保留两份：

- `l0cOutUbFirst_`：第一半输入。
- `l0cOutUbSecond_`：第二半输入。

非 interleaved 下，两半输入分别来自两个 UB 区域。interleaved 下，两半输入在同一个 UB 行里相邻排布，第二半地址用 `first + singleN_` 得到。

这也是“当前还没有 interleaved，但 swigluQuant 先适配好”的核心：

- 非 interleaved：GMM1 分两块 UB 写 `x1` 和 `x2`，swigluQuant 取 `l0cOutUbFirst_` / `l0cOutUbSecond_`。
- interleaved：GMM1 后续会按 `[x1 row][x2 row]` 的一行连续布局写 UB，swigluQuant 只需要把第二半基址改成 `base + singleN_`。
- 对 VF 计算来说，差异本质只是“下一行的 stride 不一样”。所以用模板参数 `IsInterleavedSrc` 处理，不需要两套大段重复 VF 代码。

## 这次补的 buffer 尺寸修正

我在最后一轮审查时发现一个边界：当前 master 默认非 interleaved，但 GMM1 scheduler 可能按 256x256 tile 调度；如果 swigluQuant 中间 buffer 固定按 `MAX_SINGLE_MN / 2`，非 interleaved 默认路径对 256x256 tile 不够稳。

因此最新提交把 buffer 元素数改成：

```cpp
constexpr uint32_t MAX_SINGLE_MN_ALIAS =
    IsInterleaved_ ? MAX_SINGLE_MN / SWIGLU_N_HALF : MAX_SINGLE_MN;
```

含义是：

- 非 interleaved：按 `MAX_SINGLE_MN` 申请，覆盖当前 master 的 256x256 默认路径。
- interleaved：按 `MAX_SINGLE_MN / 2` 申请，保留 port4 当前 128x256 pingpong 布局。

这个改动只影响 buffer 大小，不改变计算公式和输出布局。

## 核心 VF 函数合并

之前 interleaved 和非 interleaved 有两大段相似 VF 代码。现在统一成一个模板函数：

```cpp
template <SwigluQuantMsg::QuantMode quantMode, bool IsInterleavedSrc>
VFDoSwigluAndQuantForMX(...)
```

两个路径只在源 UB 行距上不同：

- 非 interleaved：`nSrcUbAligned = ceilAlign(nSize, 32 / sizeof(DataTypeIn))`
- interleaved：`nSrcUbAligned = nSize * 2`

因为 `IsInterleavedSrc` 是编译期常量，编译器会在实例化时直接删掉不用的分支，不会在运行时多一次 if 判断。

这次合并前的问题是：interleaved 和非 interleaved 的大 VF 函数绝大部分都在做相同事情，包括读取 x1/x2、cast fp32、算 swish、乘 gate、写 bf16 中间结果、再做 MX quant。真正不同的只有源数据布局。把它们合并成模板后，代码维护风险更低：

- 后续修正 SwiGLU 或 quant 细节，只改一处。
- 非 interleaved 当前主路径不会因为 interleaved 预留而多运行时判断。
- interleaved 接入时不会复制一份可能漂移的计算逻辑。

## VF 主计算做了什么

对不熟 VF 的同学，可以把 `VFDoSwigluAndQuantForMX` 理解成一条流水线：

1. 一次从 UB 取一段 `x1` 和 `x2`。
2. 把 bf16/half 转成 fp32，保证 sigmoid 计算精度。
3. 算 `sigmoid(x1) = x1 / (1 + exp(-x1))`。
4. 算 `out = sigmoid(x1) * x2`，这就是 SwiGLU。
5. 把 fp32 结果 cast 回 bf16，写到 `gluResAddr`。
6. 对尾部不足一个 vector 的列，用 mask 只计算有效元素，并把 padding 置 0。
7. 对 `gluResAddr` 计算每 32 个元素一组的最大 exponent。
8. 根据最大 exponent 生成 MX scale 和反向用的 halfScale。
9. 用 halfScale 把 bf16 结果缩放后 cast 成 fp8/fp4，并 pack 到 `quantOutput_`。

关键点是中间数据都在 UB 内流转，没有重新从 GM CopyIn。megaMoe 的输入来自 GMM1 写好的 UB，这和独立 ops-nn 算子不同。

## ComputeMaxExp 的作用

`ComputeMaxExp` 做的是 MX quant 的第一步：每 32 个元素找一个共享 exponent。

它不是找普通浮点最大值，而是把 bf16 的 exponent bits 取出来，再做 reduce max。这样可以快速得到这一组数据需要的共享缩放尺度。

对性能有帮助的地方是：

- 直接在寄存器里用 bit mask 取 exponent。
- 用 `ReduceMaxWithDataBlock` 按 32 元素组规约。
- 输出的是紧凑 scale 中间结果，不需要额外矩阵重排。

## ComputeScale 的作用

`ComputeScale` 根据每组 max exponent 生成两份东西：

- `mxScaleLocalAddr`：最终要写到 GM 的 MX scale。
- `halfScaleLocalAddr`：后续量化 y 时用的反向 scale。

这一步处理了 0、NaN/Inf、特殊 exponent 等边界，结构和 ops-nn 的 last-axis MX scale 逻辑保持一致。

## fp8 输出怎么 pack

`ComputeDataForQuantTargetFp8` 做的是：

1. 从 `gluResAddr` 读取 bf16 SwiGLU 结果。
2. 从 `halfScaleLocalAddr` 读取每组 scale。
3. 先乘 halfScale，把值缩放到 fp8 可表示范围。
4. 分别 cast 成 fp8 的不同 lane。
5. 用 uint8 add 把四路 lane 合并。
6. 用 `DIST_NORM_B8` 写到 `quantOutput_`。

这里的核心收益是：量化和 pack 在寄存器中完成，减少多余中间 layout 操作。

## scale copyout 为什么可以 compact

`ComputeScale` 产出的 scale 在 UB 中本来就是 compact 排列：

```text
row0: blockScaleN bytes
row1: blockScaleN bytes
...
```

其中 `blockScaleN = ceil(singleN / 32) * MXFP_MULTI_BASE_SIZE`。目标常见 `singleN=256` 时，`blockScaleN=8`。

旧路径先把每行 8B scale 搬到一个 32B 行对齐 staging buffer：

```text
(m, 8B) -> (m, 32B)
```

然后再从 staging 拷 GM。

现在改成：

```cpp
DataCopyPad<int8_t, PaddingMode::Compact>
```

直接从 compact 源拷贝，仍然用：

- `blockLen = blockScaleN`
- `dstStride = scaleN_ - blockScaleN`

所以 GM 里的行距和下游期望布局没有变，只是省掉了 UB 内的 32B staging 和那段 vector copy。

这里省掉的不是 GM 输出格式，而是 UB 内临时格式：

旧路径：

1. `ComputeScale` 生成 compact scale：每行 `blockScaleN` 字节。
2. `TransMxScaleLayout` 用 vector 把每行 scale 搬到 32B 对齐 staging。
3. `CopyScaleFromUb2Gm` 从 staging 拷到 GM。

新路径：

1. `ComputeScale` 仍生成 compact scale。
2. `CopyScaleFromUb2GmCompact` 直接从 compact scale 拷到 GM。
3. 通过 `dstStride = scaleN_ - blockScaleN` 保持 GM 行距。

因此下游看到的 `yScale` 还是原来的 compact GM 布局，省掉的是中间第 2 步，也就是 UB 内 `(m, 8B) -> (m, 32B)` 的 vector 操作和对应 staging buffer。

这也是为什么代码里删除了：

- `TransMxScaleLayout`
- `quantScaleBlockOutput_`
- 旧 `CopyScaleFromUb2Gm`
- `scaleBlockN_` 成员

## y 输出布局有没有变

y 输出仍由 `CopyOutputFromUb2Gm` 控制：

- fp8：每行拷 `singleN_` 个 int8。
- fp4：每字节两个元素，所以 `blockLen` 和 `dstStride` 都右移一位。
- 非整 32B 行时，源侧用 `srcStride` 跳过 UB 内 padding。

因此 y 的 GM 布局仍是 megaMoe 下游原本看到的布局。

## interleaved 和非 interleaved 的差异

现在两条路径共享同一个 VF 计算模板，差异只保留在必要处：

- 输入地址：
  - 非 interleaved：`firstSrc = l0cOutUbFirst_`，`secondSrc = l0cOutUbSecond_`。
  - interleaved：`firstSrc = l0cOutUbBase`，`secondSrc = l0cOutUbBase + singleN_`。
- 输入行距：
  - 非 interleaved：一行就是 `singleN_`，按 UB 对齐取整。
  - interleaved：一行包含 x1+x2，所以 stride 是 `singleN_ * 2`。
- pingpong：
  - 当前 master 默认不触发。
  - interleaved 未来接入时，通过 `pingpongIdx` 选择当前 half UB base。

其余 SwiGLU、max-exp、scale、fp8/fp4 quant 流程是同一套代码。

需要注意：当前 upstream/master 没有 interleaved 端到端路径，所以这次代码里的 interleaved 是“编译期可选能力”，不是当前默认行为。当前默认行为仍是：

```text
IsInterleaved_ = false
pingpongIdx = 0
firstSrc  = l0cOutUbFirst_
secondSrc = l0cOutUbSecond_
```

未来 interleaved 接入后，预期行为才是：

```text
IsInterleaved_ = true
pingpongIdx = 0 or 1
firstSrc  = current half base
secondSrc = current half base + singleN_
row stride = 2 * singleN_
```

这就是为什么现在可以先把模板和地址选择放在 swigluQuant 内，但不需要改 GMM1 调度。

## 代码注释处理

你指出原始注释被删是对的。这里整改遵循两条原则：

1. upstream/master 里仍然适用于新代码的注释，恢复到对应位置。
2. 已经删除的旧逻辑，比如 `TransMxScaleLayout` 内部逐行 staging 注释，不机械保留在不存在的函数里，而是在新的 compact copy 位置说明“原先做什么、现在为什么不做”。

这次在代码里补充/恢复的重点注释包括：

- `// 重构UB内存`：保留原始 UB 重构说明，并补充非 interleaved / interleaved buffer 规模差异。
- `CopyOutputFromUb2Gm`：恢复 blockCount / blockLen 的规模说明。
- `CopyScaleFromUb2GmCompact`：说明 scale 源布局已经是 compact，为什么可以省掉 `(mSize*8)->(mSize,32)`。
- `ComputeMaxExp` / `ComputeScale`：恢复 max exponent、INF/0/special mask、scale/halfScale 生成过程注释。
- `ComputeDataForQuantTargetFp8`：补充 DINTLV 读取、halfScale 广播、fp8 lane cast 和连续存储说明。
- `VFDoSwigluAndQuantForMX`：补充 interleaved/非 interleaved stride 差异、SwiGLU 主计算公式、tail padding 说明。
- `VFDoSwigluForMX`：补充当前默认不传 `pingpongIdx`，未来 interleaved 只在入口切 half UB。
- `operator()` copyout：说明 scale 统一 compact copy，省掉原先重排 scale。

## 和 ops-nn 的关系

这版不是把 ops-nn 文件逐字复制进 megaMoe，也不应该逐字复制。

保持一致的部分：

- SwiGLU 公式和 VF 寄存器计算结构。
- tail mask 和 padding 处理方式。
- MX last-axis scale 的 max-exp/scale/halfScale 思路。
- fp8/fp4 量化 pack 的寄存器路径。

必须适配 megaMoe 的部分：

- 输入不是从 GM copy-in，而是来自 GMM1 已写入 UB 的两半输出。
- y 和 yScale 要写到 megaMoe 下游要求的 GM offset。
- 当前 master 没有 interleaved 调用点，所以默认保持 false。
- atomicAdd 通知沿用 megaMoe 原有 group flag 流程。

## 为什么认为当前改造可靠

1. 修改面收敛：最终相对 master 只改了 swigluQuant 一个头文件。
2. 当前默认路径保持兼容：`IsInterleaved_` 默认 false，已有调用点不用改。
3. 非 interleaved buffer 已按 256x256 恢复容量，覆盖当前 master scheduler 的默认上界。
4. 核心数学流程来自 ops-nn/port4 已验证结构：SwiGLU -> max-exp -> scale -> quant。
5. scale compact copy 不改变 GM 目标布局，只省掉 UB 内 staging。
6. 旧 staging 相关符号已经清理干净，没有悬空引用。
7. `git diff --check` 通过，说明没有基本格式/空白错误。

## 为什么可能带来性能收益

主要收益来自少做事，而不是改变数学结果：

- 不从 GM 重新读输入，直接消费 GMM1 写好的 UB。
- SwiGLU 和 quant 都在 vector 寄存器/UB 内完成。
- fp8 pack 在寄存器中合并，减少中间写回。
- scale copyout 直接 compact copy，删除 `(m, 8B) -> (m, 32B)` 的 staging vector 操作。
- interleaved/非 interleaved 合成模板后，编译期选择路径，不引入运行时分支成本。

## 仍建议补的验证

代码审查不能替代最终硬件验证。合入前建议补：

- 编译 `mega_moe`。
- 当前默认非 interleaved 目标 shape 精度。
- 当前默认非 interleaved 目标 shape 性能。
- 未来 interleaved 调用点接入后，再单独验证 pingpongIdx=0/1、y/scale copyout 和 cube/vector 并发。

当前我能明确确认的是：代码结构、默认路径兼容性、scale compact copy 逻辑、静态 diff 检查都没有发现问题。

## Review用：逐项解释 +317 / -275

当前最新 diff 统计是：

```text
1 file changed, 317 insertions(+), 275 deletions(-)
```

这里的 `+317/-275` 不是“净新增 317 行业务逻辑、删除 275 行能力”。它主要由三类变化组成：

1. 删除 upstream/master 旧 VF 写法、旧 fp8 pack 写法、旧 scale staging 写法。
2. 增加 port4/ops-nn 风格的 VF 写法、模板化 interleaved 入口、compact scale copyout。
3. 增加/恢复用于 review 的注释，所以最后统计比之前看到的 `+312/-270` 多了几行注释。

下面按 diff hunk 顺序说明每一组 `-` 和 `+` 的含义。

### 先给不熟 VF 的 reviewer 一个读法

这段代码最容易看晕，是因为它不是普通 C++ 的逐元素循环，而是在写昇腾 vector micro API。可以先把它理解成三层存储和一条流水线。

三层存储：

```text
GM      : Global Memory，全局内存，最终 y / scale 输出在这里。
UB      : Unified Buffer，片上临时内存，GMM1 把 swigluQuant 的输入先写到这里。
Reg     : Vector Register，VF 指令真正计算的地方，一次处理一串元素。
```

当前 swigluQuant 的输入不是从 GM 读进来的。GMM1 已经把两半输入写进 UB，epilogue 直接消费 UB：

```text
GMM1 output in UB
    |
    v
SwiGLU vector compute
    |
    v
bf16 intermediate in UB: gluRes_
    |
    v
MX scale compute: maxExp_ / halfScale_ / quantScaleOutput_
    |
    v
fp8 quant output in UB: quantOutput_
    |
    v
copyout to GM: y + yScale
```

几个术语可以这样理解：

- `DataCopy`：搬数据。可能是 GM 到 UB，也可能是 UB 到寄存器，或者寄存器到 UB。
- `DataCopyPad`：按二维行拷贝，支持每行长度和行间 stride。不够对齐时可以跳过 padding。
- `MaskReg`：告诉 vector 指令“这次只处理哪些 lane”。尾部不满一个 vector 时，mask 防止多算无效元素。
- `Interleave`：把两个寄存器的数据交错重排。功能正确，但会消耗额外 vector 指令。
- `CastTrait::RegLayout`：cast 时把结果放到目标寄存器的哪个 lane/byte 位置。新 fp8 路径用它少做显式重排。
- `DIST_DINTLV_B16`：一次读两路 bf16，偶数/奇数位置拆到两个寄存器。
- `DIST_UNPACK_B16`：读 bf16 时展开到适合后续 fp32 计算的寄存器布局。
- `DIST_PACK_B16` / `DIST_PACK4_B32` / `DIST_NORM_B8`：寄存器写回 UB 时的不同打包格式。

SwiGLU 本身只是一条数学公式：

```text
out = swish(x1) * x2
swish(x1) = x1 / (1 + exp(-x1))
```

MX quant 则是把 `out` 量化成 fp8，同时每 32 个 y 元素生成一组 scale。常见 `singleN_=256` 时：

```text
一行 y 有 256 个 fp8
每 32 个 y 共用 1 个 scale
所以一行 scale 有 256 / 32 = 8 个 byte
blockScaleN = 8
```

这也是 scale compact copy 的核心背景。scale 在 UB 里天然是：

```text
row0: 8B scale
row1: 8B scale
row2: 8B scale
...
```

旧代码为了 copyout，先把每行 8B 搬到 32B staging：

```text
compact UB:
row0: [8B]
row1: [8B]

staging UB:
row0: [8B + 24B unused]
row1: [8B + 24B unused]
```

新代码直接从 compact UB copy 到 GM：

```text
compact UB row: [8B]
        |
        v
GM row with dstStride: [8B] ... next row
```

所以删 `TransMxScaleLayout` 不是删 scale，而是删掉中间那次“8B 扩成 32B”的 UB 内搬运。

interleaved / 非 interleaved 可以这样看：

```text
非 interleaved 当前路径:
firstSrc  UB: row0 x1, row1 x1, ...
secondSrc UB: row0 x2, row1 x2, ...

interleaved 未来路径:
one UB row: [x1 row][x2 row]
firstSrc  = base
secondSrc = base + singleN_
next row stride = 2 * singleN_
```

因此 interleaved 和非 interleaved 不应该复制两套大 VF 函数。数学计算完全一样，只是“下一行数据隔多远”不同。现在用模板参数 `IsInterleavedSrc` 在编译期决定 stride，就是为了把差异收敛到地址计算上。

读这次 diff 时可以按执行顺序看，而不是按代码行数看：

1. 初始化 UB buffer。
2. 选择当前输入布局和 ping/pong half。
3. 从 UB 读 x1/x2 到寄存器。
4. 在寄存器里算 SwiGLU，结果写回 UB。
5. 从 SwiGLU 结果里按 32 元素一组算 max exponent。
6. 根据 max exponent 生成 scale 和 halfScale。
7. 用 halfScale 把 y 量化成 fp8。
8. y 和 scale 从 UB copy 到 GM。
9. `AtomicAdd` 通知下游。

这次 `+/-` 基本都围绕上面这 9 步展开。

### 1. 顶部常量：新增的是 vector loop 固定宽度

新增：

```cpp
QUANT_ONCE_NUM = 256
QUANT_ONCE_NUM_FP4 = 128
SCALE_ONCE_NUM = 8
CONST_64 = 64
VF_LEN_FP32 = VECTOR_REG_WIDTH / sizeof(float)
```

为什么加：

- 旧代码大量使用 `vlForHalfNumber_`、`elementAfterReduce_` 和字面值组合，读代码时很难直接知道“这一轮到底处理多少元素”。
- port4/ops-nn 的 VF 实现把关键粒度固定表达出来：一次量化处理 256 个 bf16 元素，一组 MX scale 是 8 个 scale，SwiGLU fp32 VF 主循环每次处理 `VF_LEN_FP32` 个元素。
- 这些都是编译期常量，不增加运行时成本。

这部分没有改变输出布局，只是让后面 VF/quant 函数按 port4 的粒度写得更直接。

### 2. CastTrait：删除旧的两个通用 cast，增加 lane-aware cast

删除：

```cpp
CAST_BF16_FP16_TO_FP32
CAST_FP32_TO_BF16
```

新增：

```cpp
CAST_ZERO
CAST_ONE
CAST_FP32_TO_FP16_BF16
CAST_32_TO_80
CAST_32_TO_81
CAST_32_TO_82
CAST_32_TO_83
```

为什么删旧的：

- 旧 `CAST_BF16_FP16_TO_FP32` 只表达“从 bf16/half 到 fp32”，不区分源寄存器中的 even/odd lane。
- 旧 fp8 pack 需要先 `Interleave`，再多次 `DIST_PACK4_B32` store，cast trait 本身没有表达最终 fp8 byte lane 的位置。

为什么加新的：

- `CAST_ZERO` / `CAST_ONE`：把 bf16 寄存器里的不同 lane cast 成 fp32，配合 `DIST_DINTLV_B16` 和 `DIST_UNPACK_B16`，少做显式 interleave。
- `CAST_FP32_TO_FP16_BF16`：SwiGLU 结果从 fp32 回写 bf16 中间 buffer。
- `CAST_32_TO_80/81/82/83`：把四路 fp32 结果 cast 到 fp8 的 4 个 byte lane，再通过 uint8 add 合成连续 fp8 输出。

这组新增是 fp8 port4 路径的核心前置条件。没有这些 lane-aware cast，后面的寄存器 pack 就只能回到旧的多次 interleave/store 写法。

### 3. 类模板：新增 `IsInterleaved_`，默认 false

删除的形式：

```cpp
template <..., bool IsTensorList_>
class BlockEpilogueSwigluMxQuant
```

新增的形式：

```cpp
template <..., bool IsTensorList_, bool IsInterleaved_ = false>
class BlockEpilogueSwigluMxQuant
```

为什么这么改：

- 你要求当前 upstream/master 还没有 interleaved 调用点，默认不打开 interleaved。
- `IsInterleaved_ = false` 保证现有 5 个模板参数的调用点不用改，仍然编译为非 interleaved。
- 未来 interleaved 上线时，调用点只需要显式传 `true`，swigluQuant 内部已有布局适配。

这里的关键是 `IsInterleaved_` 是编译期常量。`if constexpr (IsInterleaved_)` 不会变成运行时 if，编译器会直接删掉未选路径。

### 4. `operator()`：新增可选 `pingpongIdx`

删除：

```cpp
operator()(blockShape, blockCoord)
```

新增：

```cpp
operator()(blockShape, blockCoord, uint16_t pingpongIdx = 0)
```

为什么加：

- interleaved + GMM1 pingpong 以后，需要告诉 epilogue 当前消费的是 ping 还是 pong half。
- 当前 master 不传第三个参数，默认就是 0，所以行为不变。
- 这符合“只在 epilogue 开头处理 pingpongIdx/current half UB base”的要求，VF 中间计算不关心调度状态。

为什么不是现在改 GMM1：

- 当前 master 还没有 interleaved 主流程。
- 这次只改 swigluQuant 文件，先把入口能力准备好，不扩散到 scheduler。

### 5. 函数声明：删除 scale staging 入口，新增 compact copy 和模板 VF

删除：

```cpp
VFDoSwigluForMX(uint16_t mSize)
TransMxScaleLayout(uint16_t mSize, uint16_t scaleBlockN)
VFDoSwigluAndQuantForMX<quantMode>(...)
CopyScaleFromUb2Gm(...)
AddFlagFromUb2Gm(...)
```

新增：

```cpp
VFDoSwigluForMX(uint16_t mSize, uint16_t pingpongIdx = 0)
VFDoSwigluAndQuantForMX<quantMode, bool IsInterleavedSrc = false>(...)
CopyScaleFromUb2GmCompact(...)
```

为什么删 `TransMxScaleLayout`：

- 旧路径先把 scale 从 compact `(m, blockScaleN)` 搬到 UB staging `(m, 32B)`，再 copyout。
- 新路径确认 scale 在 UB 中已经是 compact，直接 `PaddingMode::Compact` copy 到 GM。
- 这个 staging 只服务旧 copyout 形式，删除它不会改变 GM 目标布局。

为什么删旧 `CopyScaleFromUb2Gm`：

- 旧函数默认源 UB 行已经按 32B staging 排好。
- 现在源 UB 是 compact，所以需要新的 `CopyScaleFromUb2GmCompact` 表达源布局。

为什么删 `AddFlagFromUb2Gm`：

- 它只是一个未使用 wrapper。
- 实际通知仍然在 `operator()` 尾部直接 `AtomicAdd(groupFlagListGmAddr_, 1)`，group flag 行为没有被删。

为什么 VF 函数多传 `gluResAddr/maxExpAddr/halfScaleLocalAddr`：

- 旧函数内部固定从成员 tensor base 取地址。
- interleaved pingpong 以后，入口需要先根据 `pingpongIdx` 选择当前 half 的 base，再把地址传进公共 VF。
- 这样中间 VF 逻辑不需要知道 pingpong，只处理传入的当前 buffer。

### 6. 成员变量：删除旧 staging 成员，新增安全的第二输入 offset

删除：

```cpp
quantScaleBlockOutput_
scaleBlockN_
```

为什么删：

- `quantScaleBlockOutput_` 只用于旧 `TransMxScaleLayout` staging。
- `scaleBlockN_` 只在 `operator()` 中算一次再传给 staging。
- compact copyout 后，`blockScaleN` 可以在 copy 函数里按 `singleN_` 直接计算，不需要保存成员状态。

新增：

```cpp
static constexpr uint32_t kUbSecondOffset = ...
l0cOutUbSecond_{..., kUbSecondOffset, MAX_SINGLE_MN}
```

为什么加：

- 旧代码固定把第二块 VECIN 放在 `MAX_SINGLE_MN * sizeof(DataTypeIn)` 偏移。
- 对 bf16 来说，两块 `256*256` 输入正好是 256KB；如果某些类型或布局让两块描述超出 VECIN 可描述空间，固定偏移会有风险。
- 新代码用编译期判断：能放下就按原 offset 放，放不下就让描述 alias 到 0。

需要注意：

- 当前默认非 interleaved、bf16 输入下，两块输入仍能按原布局使用。
- interleaved 未来走的是同一块 UB 行内 `[x1, x2]`，第二半通过 `base + singleN_` 得到，不依赖 `l0cOutUbSecond_`。

### 7. `Init` UB 布局：删掉 scale staging buffer，按布局决定中间 buffer 容量

删除的旧逻辑：

```cpp
gluRes_ / quantOutput_ / quantScaleOutput_ / maxExp_ / halfScale_ 都按 MAX_SINGLE_MN
额外申请 quantScaleBlockOutput_ = params_->baseM * 32B
```

新增的逻辑：

```cpp
MAX_SINGLE_MN_ALIAS = IsInterleaved_ ? MAX_SINGLE_MN / SWIGLU_N_HALF : MAX_SINGLE_MN
gluRes_ / quantOutput_ / quantScaleOutput_ / maxExp_ / halfScale_ 都按 MAX_SINGLE_MN_ALIAS
不再申请 quantScaleBlockOutput_
```

为什么这样做：

- 非 interleaved 是当前 master 默认路径，需要覆盖完整 256x256 tile，所以用 `MAX_SINGLE_MN`。
- interleaved 未来按 half tile + pingpong 使用，所以预留 `MAX_SINGLE_MN / 2`。
- scale staging buffer 已删除，少占一段 UB。

这段的 reviewer 重点：

- 当前默认路径没有被缩小 buffer。
- interleaved 分支现在是模板能力，真正启用后还要结合 GMM1 的 half tile 调度一起验证。

### 8. `CopyOutputFromUb2Gm`：新增源侧 padding stride

删除：

```cpp
blockLen = singleN_ * sizeof(int8_t)
dstStride = (n_ - singleN_) * sizeof(int8_t)
fp4 时再右移
```

新增：

```cpp
fp4: blockLen = singleN_ >> 1, dstStride = (n_ - singleN_) >> 1
fp8: blockLen = singleN_, srcStride = (nDstUbAligned - singleN_) / 32, dstStride = n_ - singleN_
```

为什么改：

- 新 VF 写 `quantOutput_` 时按 `nDstUbAligned = ceilAlign(singleN_, 32)` 在 UB 中逐行存。
- 当 `singleN_` 不是 32B 对齐时，UB 每行末尾会有 padding。
- copyout 到 GM 时，GM 下游仍只需要真实 `singleN_`，所以源侧必须用 `srcStride` 跳过 UB padding。

对常见 `singleN_=256`：

- `nDstUbAligned == singleN_`，`srcStride = 0`。
- 行为和旧路径一致。

这段 `+` 是为了补齐非 32B 对齐场景，不改变常见 256 列场景的 GM 布局。

### 9. scale copyout：删除 `CopyScaleFromUb2Gm`，新增 compact copy

删除旧逻辑：

```cpp
CopyScaleFromUb2Gm(src = quantScaleBlockOutput_)
blockLen = blockScaleN * sizeof(int8_t)
dstStride = (scaleN_ - blockScaleN) * sizeof(int8_t)
DataCopyPad(...)
```

新增逻辑：

```cpp
CopyScaleFromUb2GmCompact(src = quantScaleOutput_)
blockLen = blockScaleN
srcStride = 0
dstStride = scaleN_ - blockScaleN
DataCopyPad<int8_t, PaddingMode::Compact>(...)
```

为什么加 compact：

- `ComputeScale` 产物就是 compact：每行只有 `blockScaleN` 个 scale byte。
- 下游 GM 期望的也是按 `scaleN_` 行距排列的 compact scale。
- `PaddingMode::Compact` 正好表达“源是 compact，目标按 dstStride 展开”。

为什么这里不需要 UB 内 32B 行对齐 staging：

- 旧 staging 的目的不是改变数学结果，而是让旧 `DataCopyPad` 从 32B 行源读。
- compact copy 支持紧凑源，不需要先把每行 8B scale 搬到 32B staging。
- 删除这段 vector 搬运能省 UB 写读和一段循环。

所以这里的 `-` 是删除中间格式，`+` 是直接写最终 GM 格式。

### 10. `ComputeMaxExp`：删除无用临时寄存器，改成 port4 常量粒度

删除：

```cpp
vdExp0BF16 / vdExp1BF16
castTraitHalf2Bf16
vlForHalfNumber_ * 2
elementAfterReduce_
```

新增：

```cpp
onceNum = QUANT_ONCE_NUM
scaleNum = SCALE_ONCE_NUM
```

为什么删：

- `vdExp0BF16`、`vdExp1BF16`、`castTraitHalf2Bf16` 在旧实现里没有实际参与输出。
- `vlForHalfNumber_ * 2` 和 `elementAfterReduce_` 表达的是固定 VF 粒度，但放在成员变量里降低可读性。

为什么加：

- `onceNum=256`：每轮从 bf16 SwiGLU 结果里取 256 个元素。
- `scaleNum=8`：每 32 个元素规约出一个 exponent，一轮 256 个元素对应 8 个 scale。

算法本身没有变：

```text
读 bf16 结果 -> 按 bf16 exponent mask 取指数位 -> 两路 max -> 每 32 个元素 reduce max -> 写 maxExp
```

这部分 `+/-` 主要是整理和对齐 port4 写法，不是改变 scale 含义。

### 11. `ComputeScale`：删除无用寄存器，显式表达 MX scale 写宽

删除：

```cpp
vdExp0 / vdExp1 等未使用寄存器
vlForHalfNumber_
vlForHalfNumber_ >> 1
```

新增：

```cpp
onceNum = QUANT_ONCE_NUM_FP4
onceNumMxScale = CONST_64
```

为什么改：

- 一轮读 128 个 `maxExp`。
- 写 scale 时用 `DIST_PACK_B16`，uint16 到 int8 compact 写，实际写宽是 64。
- 旧代码用 `vlForHalfNumber_ >> 1` 表示 64，新代码直接命名成 `onceNumMxScale`。

保留的语义：

- INF/NaN mask。
- zero mask。
- invalid exponent 兜底到 `fpEmax_`。
- `scaleValue` 写到 yScale。
- `halfScale` 写到中间 buffer 给后续 y 量化使用。

所以这里的 `-` 是去掉不参与计算的临时变量和成员粒度表达，`+` 是让同一算法更贴近 port4 的寄存器实现。

### 12. fp8 quant：这是主要性能相关改造

旧 fp8 路径大致做法：

```text
读 bf16 两路
读 halfScale 并广播
乘 halfScale
Interleave bf16
cast vdExp0 到 fp32 zero/one
Interleave fp32
cast 到 fp8
对 vdExp1 重复一遍
四次 DIST_PACK4_B32 写 outLocalAddr
```

删除的重点：

- `totalCountInUB2` 和 `dataMask1/2/3/4`：旧 pack 分多阶段维护不同 mask。
- `floatScaleForMul`、`vdExp0Convert`、`vdExp1Convert`、`vdBf16Exp0FP4` 等未真正用于当前 fp8 输出的寄存器。
- 两次显式 `Interleave`。
- 四次 `DIST_PACK4_B32` store。
- 函数内局部 `castTraitZero/One/32to8`，因为现在挪到文件顶部统一复用。

新增 fp8 路径：

```text
DIST_DINTLV_B16 读取 bf16 两路
DIST_E2B_B16 广播 8 个 halfScale
两路 bf16 乘 halfScale
CAST_ZERO / CAST_ONE 拆成四路 fp32
CAST_32_TO_80/82/81/83 分别 cast 到 fp8 byte lane
三次 uint8 Add 合并四路 lane
一次 DIST_NORM_B8 连续写出 fp8
```

为什么能优化：

- 旧路径靠 interleave + 多次 pack store 把字节顺序凑出来。
- 新路径在 cast 阶段就把结果放到目标 byte lane，再用 uint8 add 合成。
- store 从四次变成一次，寄存器重排也更少。

review 时可以把这段理解成：

- `-` 的是旧的“先重排，再分四次写”。
- `+` 的是 port4 的“cast 时定位 lane，寄存器内合并，一次连续写”。

这也是当前 bf16 输入、fp8 输出、MX scale quant 目标路径最关键的性能收益来源。

### 13. fp4 quant：不是当前目标主路径，但同步清理

删除：

```cpp
dataMask2
vdExp0Convert / vdExp1Convert
vdExp0BF16 / vdExp1BF16
vdBf16Exp0FP4 / vdBf16Exp1FP4
vlForHalfNumber_ * 2
OUT_ELE_NUM_ONE_BLK 直接散落使用
```

新增：

```cpp
onceXNum = QUANT_ONCE_NUM
onceYNum = OUT_ELE_NUM_ONE_BLK
```

为什么改：

- fp4 不是当前实际目标，但和 fp8 共用 MX scale 中间结果。
- 删除未使用临时寄存器，避免后续维护时误以为这些寄存器有语义。
- 保持和 port4 风格一致：读 256 bf16、广播 halfScale、乘 scale、interleave、cast fp4、两次 pack 写。

这段不是主要性能收益点，更多是保持同一文件内 quant 代码风格一致。

### 14. VF 主函数：旧大循环被替换成一个支持两种源布局的模板

旧函数：

```cpp
template <quantMode>
VFDoSwigluAndQuantForMX(outputDst, scaleDst, firstSrc, secondSrc, mSize, nSize)
```

新函数：

```cpp
template <quantMode, bool IsInterleavedSrc>
VFDoSwigluAndQuantForMX(outputDst, scaleDst, firstSrc, secondSrc,
                        gluResAddr, maxExpAddr, halfScaleLocalAddr,
                        mSize, nSize)
```

旧主循环删除了什么：

- `sizePerRepeat` / `OneRowRepeatTimes` 这套旧循环结构。
- 每个 VF block 先 `DataCopy` 再 `Interleave` 输入的写法。
- `calBlockNum = sizeof(float) / sizeof(DataTypeIn)` 的二层拆分。
- 旧函数内部固定取 `gluRes_`、`maxExp_`、`halfScale_` base 地址。
- 旧 SwiGLU 内的 clamp：
  - `Mins(swishData, clampLimit_)`
  - `Mins(gateData, clampLimit_)`
  - `Maxs(gateData, -clampLimit_)`

新主循环新增了什么：

- `IsInterleavedSrc` 编译期决定源 stride：
  - 非 interleaved：`ceilAlign(nSize, 32 / sizeof(DataTypeIn))`
  - interleaved：`2 * nSize`
- 用 `VF_LEN_FP32` 做列方向循环，一次按 fp32 vector 宽度处理。
- 用 `DIST_UNPACK_B16` 直接读 bf16 到适合 fp32 cast 的寄存器布局。
- 显式 tail 处理：
  - `mask1` 控制有效尾部元素。
  - `mask2/mask3` 控制写 bf16 中间结果和 padding 置 0。
- SwiGLU 计算改成 port4 结构：

```text
x1 -> fp32
x2 -> fp32
neg = -x1
exp = exp(neg)
den = exp + 1
swish = x1 / den
out = swish * x2
out -> bf16
```

为什么可靠：

- 非 interleaved 和 interleaved 真正不同的是源地址行距，不是数学公式。
- 用模板参数把行距差异放在编译期，避免两份大 VF 函数漂移。
- tail padding 在 SwiGLU 阶段就处理，后续 max-exp/scale/quant 看到的是对齐后的中间 buffer。

需要 reviewer 特别关注：

- 旧 master 的 VF 里有 clamp，当前 port4/ops-nn 风格路径不再做这组 clamp。
- 这不是 accidental delete，而是随 port4 VF 主流程一起替换掉的行为差异。
- 如果某些非目标场景依赖 `clampLimit_` 的数值行为，需要额外精度用例确认；当前目标路径的移植意图是对齐已经验证收益的 port4 实现。

### 15. quant 链地址：从函数内部固定 base，变成入口传入当前 base

删除：

```cpp
__ubuf__ bfloat16_t *gluResAddr = gluRes_.GetPhyAddr()
__ubuf__ uint16_t *maxExpAddr = maxExp_.GetPhyAddr()
__ubuf__ uint16_t *halfScaleLocalAddr = halfScale_.GetPhyAddr()
```

新增：

```cpp
gluResAddr / maxExpAddr / halfScaleLocalAddr 由 VFDoSwigluForMX 传入
```

为什么改：

- 非 interleaved 当前仍传 base 地址，行为等价。
- interleaved pingpong 时，入口会先根据 `pingpongIdx` 偏移到当前 half，再传给公共 VF。
- 这满足“只在 epilogue 开头处理 pingpongIdx/current half UB base”的要求。

### 16. `VFDoSwigluForMX`：从简单 wrapper 变成布局选择入口

旧逻辑：

```cpp
quantOutput = quantOutput_.GetPhyAddr()
quantScale = quantScaleOutput_.GetPhyAddr()
first = l0cOutUbFirst_.GetPhyAddr()
second = l0cOutUbSecond_.GetPhyAddr()
VFDoSwigluAndQuantForMX(...)
```

新逻辑：

```cpp
pongMul = IsInterleaved_ && pingpongIdx == 1 ? 1 : 0
按 pongMul 偏移 input / gluRes / quantOutput / quantScale / maxExp / halfScale

if constexpr (IsInterleaved_) {
    second = base + singleN_
    VFDoSwigluAndQuantForMX<..., true>(...)
} else {
    second = l0cOutUbSecond_.GetPhyAddr()
    VFDoSwigluAndQuantForMX<..., false>(...)
}
```

为什么加：

- 这是 interleaved 适配真正集中的地方。
- 当前 master 不传 `pingpongIdx`，且 `IsInterleaved_=false`，所以实际仍走旧的两块 UB 输入。
- 未来 interleaved 接入后，第二半输入来自同一行 `base + singleN_`，不需要复制 VF 主循环。

为什么不会破坏 GMM1 pingpong：

- 这里没有关闭 pingpong，也没有改 cube/vector 并发。
- 它只是让 swigluQuant 在被调用时根据 `pingpongIdx` 选择当前 half。
- 调度仍由 GMM1/上层控制。

### 17. `TransMxScaleLayout`：整段删除是因为 compact copy 替代了它

旧函数做的事：

```text
for each row:
    从 quantScaleOutput_ 的 compact row 读 blockScaleN 字节
    写到 quantScaleBlockOutput_ 的 32B row
```

为什么删：

- 它只是在 UB 内把 `(m, 8B)` 变成 `(m, 32B)`。
- 下游 GM 不要求先在 UB 变成 32B 行。
- 新 `CopyScaleFromUb2GmCompact` 能直接从 `(m, 8B)` 源布局 copy 到 GM，并用 `dstStride` 保持 GM 行距。

所以这一大段 `-` 不是删掉 scale 输出，而是删掉 scale 输出前的中间重排。

### 18. `operator()`：删除旧 scale staging 调用，新增 pingpong copyout 选择

删除：

```cpp
scaleBlockN_ = ceilDiv(singleN_, 32) * ...
VFDoSwigluForMX(singleM_)
TransMxScaleLayout(singleM_, scaleBlockN_)
CopyScaleFromUb2Gm(..., quantScaleBlockOutput_)
```

新增：

```cpp
VFDoSwigluForMX(singleM_, pingpongIdx)
CopyScaleFromUb2GmCompact(..., quantScaleOutput_)

if constexpr (IsInterleaved_) {
    pingpongIdx == 1 时 copy quantOutput_/quantScaleOutput_ 的 pong half
}
```

为什么这么改：

- `scaleBlockN_` 不再需要作为成员保存。
- scale staging 不再需要。
- interleaved pingpong 时，输出和 scale 也要从当前 half 的 output buffer copyout。
- 非 interleaved 当前默认路径仍然直接 copy `quantOutput_` 和 `quantScaleOutput_`。

保留不变的内容：

- `singleM_ / singleN_` 初始化。
- `yOffset / yScaleOffset` 计算。
- `V_MTE3`、`MTE3_V`、`MTE3_S` flag 顺序。
- 最后的 `AtomicAdd(groupFlagListGmAddr_, 1)`。

### 19. 文件末尾 newline

diff 里还有一处：

```diff
-#endif
\ No newline at end of file
+#endif
```

这只是补文件末尾换行，符合代码规范，没有行为变化。

### 20. 哪些 `+` 只是注释，不是逻辑

最后一次整改里加了不少注释，所以 `+317` 里有一部分只是解释，不是新增执行路径。主要包括：

- 非 interleaved / interleaved buffer 规模说明。
- compact scale copy 为什么可以省掉 staging。
- fp8 pack 中 `CAST_32_TO_80/82/81/83` 的 lane 作用。
- VF 主循环里 interleaved 和非 interleaved stride 差异。
- `pingpongIdx` 当前默认不传、未来 interleaved 接入后如何使用。

这些注释的目的就是降低 review 成本。它们不改变编译出的执行逻辑，除了影响源码行数统计。

## 不懂VF版：按执行链路解释每块 `-` 和 `+`

这一节不按函数名讲，而是按数据怎么走讲。review 时可以把它当成“我看到一堆 `-` 和 `+`，它们在流水线里分别属于哪一步”。

### A. 输入阶段：为什么要加 interleaved 模板，但当前默认 false

旧代码假设输入永远是两块 UB：

```text
l0cOutUbFirst_  -> x1
l0cOutUbSecond_ -> x2
```

所以旧代码的 `-` 是：

```cpp
VFDoSwigluForMX(singleM_)
first = l0cOutUbFirst_.GetPhyAddr()
second = l0cOutUbSecond_.GetPhyAddr()
```

这对当前非 interleaved 是对的，但未来 interleaved 会变成：

```text
同一行 UB:
[x1 0..N-1][x2 0..N-1]
```

如果不提前抽象，后面接 interleaved 时容易复制一整份 VF 主函数。那样 review 和维护会很痛，因为两份代码都在做同一个 SwiGLU + quant，只是输入 stride 不同。

新代码的 `+` 是：

```cpp
if constexpr (IsInterleaved_) {
    second = base + singleN_;
    VFDoSwigluAndQuantForMX<..., true>(...);
} else {
    second = l0cOutUbSecond_.GetPhyAddr();
    VFDoSwigluAndQuantForMX<..., false>(...);
}
```

这段对不懂 VF 的人可以理解为：

```text
非 interleaved:
    去 A 仓库拿 x1，去 B 仓库拿 x2

interleaved:
    去同一个仓库拿 x1，往后偏移 N 个元素拿 x2
```

数学计算没有变，变的是“第二半输入地址怎么算”。当前 `IsInterleaved_` 默认是 false，所以 master 现在仍然走 A/B 两块 UB。

### B. SwiGLU 计算阶段：旧代码为什么删了那么大一段

旧 VF 主循环看起来很长，是因为它把“读 bf16、重排、cast、算公式、再写回”拆得比较碎。

旧路径大致是：

```text
1. 从 firstSrc 读一段 x1 bf16
2. 从 secondSrc 读一段 x2 bf16
3. Interleave x1，为 cast 到 fp32 做准备
4. Interleave x2，为 cast 到 fp32 做准备
5. x1 拆成两路 fp32
6. x2 拆成两路 fp32
7. 对两路分别做 clamp
8. 两路分别算 swish(x1) * x2
9. 两路分别 cast 回 bf16
10. 写回 gluRes_
```

对应 diff 里的 `-` 包括：

```cpp
DataCopy(swishInput0, firstSrc + l0cOutOffset)
DataCopy(gateInput0, secondSrc + l0cOutOffset)
Interleave(...)
Cast<float, DataTypeIn, CAST_BF16_FP16_TO_FP32>(...)
Mins(...)
Maxs(...)
Div<float, &DIV_MODE>(...)
Cast<bfloat16_t, float, CAST_FP32_TO_BF16>(...)
DataCopy<DIST_PACK_B32>(gluResAddr + dstUbOffset, ...)
```

新路径把同一件事写成 port4/ops-nn 结构：

```text
1. 直接用 DIST_UNPACK_B16 读 x1，寄存器布局已经适合 cast
2. 直接用 DIST_UNPACK_B16 读 x2
3. x1 cast fp32
4. x2 cast fp32
5. 算 -x1
6. 算 exp(-x1)
7. 算 exp(-x1)+1
8. 算 x1/(exp(-x1)+1)
9. 乘 x2
10. cast 回 bf16
11. 按对齐后的行宽写回 gluRes_
```

对应 diff 里的 `+` 包括：

```cpp
DataCopy<DataTypeIn, LoadDist::DIST_UNPACK_B16>(vregX1, ...)
DataCopy<DataTypeIn, LoadDist::DIST_UNPACK_B16>(vregX2, ...)
Cast<float, DataTypeIn, CAST_ZERO>(...)
Muls(negReg, vregX1F, -1)
Exp(expReg, negReg)
Adds(addsReg, expReg, 1)
Div(sigmoidReg, vregX1F, addsReg)
Mul(outFReg, sigmoidReg, vregX2F)
Cast<bfloat16_t, float, CAST_FP32_TO_FP16_BF16>(...)
DataCopy<StoreDist::DIST_PACK_B32>(gluResAddr, ...)
```

为什么这不是“重写了一个看不懂的新算法”：

- 公式仍然是 `x1 / (1 + exp(-x1)) * x2`。
- 输出中间结果仍然是 bf16 的 `gluRes_`。
- 后面的 maxExp、scale、quant 仍然消费 `gluRes_`。
- 新代码只是用更适合寄存器布局的读法，少做旧代码里的显式 `Interleave`。

这里确实有一个行为差异需要 reviewer 看见：旧代码有 clamp，新 port4 路径没有 clamp。这一点报告前面已经列为重点检查项。它不是隐藏变更，应该用目标精度验证闭环。

### C. tail 处理：为什么新增了很多 mask 变量

旧代码默认按 `OneRowRepeatTimes` 处理，尾部主要靠 `UpdateMask<DataTypeIn>(elementNum)` 兜住。新代码更显式地区分：

```text
完整 vector 段：每次处理 VF_LEN_FP32 个元素
尾部有效段：mask1 只算真实元素
尾部 padding：mask2/mask3 把对齐补出来的部分写 0
```

新增的这些变量：

```cpp
dim1VfTimes
dim1Tail
dim1TailTimes
dim1Tail2
mask1Num
mask2Num
mask3Num
firstTailAddr
secondTailAddr
swigluTailAddr1
swigluTailAddr2
```

对不熟 VF 的人来说，它们只是为了处理这种情况：

```text
真实 N 不是 vector 宽度整数倍:
[有效元素.........][尾部有效][padding]
```

为什么要把 padding 写 0：

- 后面 `ComputeMaxExp` 是按对齐后的 `nDstUbAligned` 扫 `gluRes_`。
- 如果 padding 不清 0，后面可能把无效数据也纳入 max exponent。
- 清 0 后，padding 不会影响 scale 和 quant。

常见目标 `singleN_=256` 时通常没有复杂尾部，但这段保证非整齐 N 也安全。

### D. max exponent 阶段：删除的不是功能，是未使用临时变量

MX scale 需要每 32 个 y 找一个共享 exponent。可以把 `ComputeMaxExp` 想成：

```text
输入：bf16 SwiGLU 结果
操作：每个元素取 exponent bits
规约：每 32 个元素取最大 exponent
输出：maxExp_
```

旧代码里有一些寄存器声明看起来参与了 bf16 转换：

```cpp
vdExp0BF16
vdExp1BF16
castTraitHalf2Bf16
```

但实际主流程只需要：

```text
bf16 bits -> and exponent mask -> max -> reduce max
```

所以这些 `-` 是删掉没有实际语义的临时变量，不是删掉 max exponent 计算。

新增：

```cpp
onceNum = 256
scaleNum = 8
```

是为了让 reviewer 直接看出：

```text
每轮读 256 个 bf16
每 32 个元素产生 1 个 scale
所以每轮产生 8 个 maxExp
```

### E. scale 阶段：为什么 yScale 仍然是下游期望的布局

`ComputeScale` 做两件事：

```text
maxExp_ -> quantScaleOutput_  : 写给下游的 yScale
maxExp_ -> halfScale_         : 给本 epilogue 内部量化 y 用
```

旧代码和新代码都保留这些边界处理：

```text
0       -> scale 保持 0
Inf/NaN -> scale 使用特殊值
太小/太大的 exponent -> 按 fp8/fp4 emax 处理
```

diff 里的 `-` 主要是：

```cpp
未使用寄存器
vlForHalfNumber_
vlForHalfNumber_ >> 1
```

diff 里的 `+` 主要是：

```cpp
onceNum = 128
onceNumMxScale = 64
```

这里容易误解的是 `onceNumMxScale = 64`。它不是说一行 scale 变成 64 个。它表示这一轮 vector store 的 pack 写宽。真正每行有多少 scale 仍由：

```cpp
blockScaleN = ceil(singleN_ / 32) * MXFP_MULTI_BASE_SIZE
```

决定。`singleN_=256` 时就是 8。

### F. fp8 quant 阶段：为什么旧代码四次写，新代码一次写

这是最需要给不懂 VF 的 reviewer 讲清楚的地方。

旧代码把 256 个 bf16 量化成 fp8，大致要把数据拆成四份再写：

```text
bf16 data
  |
  v
Interleave
  |
  v
fp32 lane 0
fp32 lane 1
fp32 lane 2
fp32 lane 3
  |
  v
四次 DIST_PACK4_B32 写到 outLocalAddr
```

可以理解为：旧代码先把数据摆成合适形状，然后分 4 次把 4 组 byte 写出去。

新代码改成：

```text
bf16 data
  |
  v
CAST_32_TO_80 -> 直接放到 byte lane 0
CAST_32_TO_82 -> 直接放到 byte lane 2
CAST_32_TO_81 -> 直接放到 byte lane 1
CAST_32_TO_83 -> 直接放到 byte lane 3
  |
  v
uint8 Add 合成一个连续 byte 寄存器
  |
  v
一次 DIST_NORM_B8 写到 outLocalAddr
```

为什么可以用 `Add` 合并：

- 每个 cast 写的是不同 byte lane。
- 没被写的 lane 是 0。
- 几个寄存器按 uint8 加起来，相当于把不同 lane 拼到同一个输出寄存器里。

这不是数值加法语义上的“把 fp8 值相加”，而是利用不同 byte lane 互不重叠，把四份结果合到一个连续 byte 输出。

所以 fp8 这里最重要的 `-` 是：

```text
删除显式 Interleave
删除四次 pack store
删除多套 mask 和未使用中间寄存器
```

最重要的 `+` 是：

```text
cast 时直接放到目标 byte lane
寄存器内合并
一次连续 store
```

这就是性能收益的主要来源之一：少做重排，少写 UB。

### G. scale copyout 阶段：为什么删 staging 是安全的

旧代码的 scale copyout 是三步：

```text
1. ComputeScale 生成 compact scale:
   row0: 8B
   row1: 8B

2. TransMxScaleLayout 搬到 32B staging:
   row0: 8B + 24B unused
   row1: 8B + 24B unused

3. CopyScaleFromUb2Gm 从 staging 写 GM
```

新代码是两步：

```text
1. ComputeScale 生成 compact scale:
   row0: 8B
   row1: 8B

2. CopyScaleFromUb2GmCompact 直接写 GM:
   blockLen = 8
   dstStride = scaleN_ - 8
```

注意这里“直接写 GM”不是说 GM 连续堆起来不分行。`dstStride` 仍然控制 GM 每行之间的间隔：

```text
GM yScale:
row0: [8B scale][stride 到下一行]
row1: [8B scale][stride 到下一行]
```

所以 GM 布局没有因为删除 staging 而变。变的是 UB 内不再临时扩 32B。

这也是你前面指出的“不需要先做 UB 内 32B 行对齐 staging，可以省 vector 操作”。当前代码就是按这个方向改的。

### H. copyout y 阶段：为什么要加 `srcStride`

新 VF 生成 y 时，UB 里的每行按 32B 对齐：

```text
UB row:
[真实 singleN_ bytes][padding to 32B]
```

GM 下游只要真实 y：

```text
GM row:
[真实 singleN_ bytes]
```

所以 fp8 copyout 新增：

```cpp
srcStride = (nDstUbAligned - singleN_) / 32
dstStride = n_ - singleN_
```

含义是：

```text
从 UB 拷完一行真实数据后，跳过 UB padding
写到 GM 后，跳过 GM 中当前 block 以外的列
```

常见 `singleN_=256` 时 padding 是 0，所以这段不影响目标 shape；它是为了让非 32B 对齐的 N 也不把 padding 拷出去。

### I. group flag 阶段：这次没有改通知语义

diff 里删了 `AddFlagFromUb2Gm` wrapper，但没有删真正的通知：

```cpp
AscendC::AtomicAdd(groupFlagListGmAddr_, 1);
```

因此这次 swigluQuant 文件内的 group flag 行为保持为：

```text
y copyout
scale copyout
MTE3/V/S 同步
AtomicAdd group flag
```

workspace/reset/read/write 的 cache line offset 已经不在本轮修改范围内，因为你明确要求只改 swigluQuant 文件。

## 删除/新增对照表

| diff 里的变化 | 旧代码在做什么 | 新代码在做什么 | 为什么这样改 |
|---|---|---|---|
| `- CAST_BF16_FP16_TO_FP32` / `- CAST_FP32_TO_BF16` | 通用 cast，不表达 fp8 byte lane | `+ CAST_ZERO/ONE` 和 `+ CAST_32_TO_80/81/82/83` | fp8 pack 需要 lane-aware cast，减少 interleave 和多次 store |
| `+ IsInterleaved_ = false` | 只能非 interleaved | 编译期预留 interleaved，默认 false | 当前 master 不打开 interleaved，未来接入不复制大 VF |
| `+ pingpongIdx = 0` | epilogue 不知道当前 half | 未来可选择 ping/pong half | 当前默认 0，不改变调用点 |
| `- TransMxScaleLayout` | UB 内 `(m,8B)` 转 `(m,32B)` | 直接 compact copy 到 GM | 省掉一段 vector 搬运，GM scale 布局不变 |
| `- quantScaleBlockOutput_` | scale staging buffer | 不再需要 staging | compact copy 使用 `quantScaleOutput_` |
| `- scaleBlockN_` | 只给 staging 使用 | copy 时局部计算 blockScaleN | 成员状态减少，逻辑更局部 |
| `+ MAX_SINGLE_MN_ALIAS` | 所有路径同一 buffer 上限 | 非 interleaved 用 full，interleaved 用 half | 当前 master 默认路径不缩 buffer，interleaved 预留 half tile |
| `+ srcStride` in y copyout | 没显式跳 UB padding | copyout 时跳过 UB 对齐 padding | 非 32B 对齐 N 更安全 |
| `- old fp8 four stores` | 四次 pack store 输出 fp8 | lane cast + uint8 合并 + 一次 store | 主性能优化点之一 |
| `- old VF loop` | 手动 interleave、二层拆分、固定 base | port4 模板 VF、显式 stride/tail/base | 支持 interleaved/非 interleaved 共用实现 |
| `- clamp` | 旧 master 对 swish/gate 做 clamp | port4 路径不做 clamp | 行为差异，需目标精度验证确认 |
| `- AddFlagFromUb2Gm` | 未使用 wrapper | 直接保留 `AtomicAdd` | 删除死代码，不改通知 |

## reviewer 最该逐行看的 5 个点

1. `IsInterleaved_` 默认值是不是 false。

这决定当前 upstream/master 是否无调用点改动即可保持非 interleaved。

2. `VFDoSwigluForMX` 里非 interleaved 分支是否仍使用两块 UB。

应该看到：

```cpp
l0cOutUbBase = l0cOutUbFirst_.GetPhyAddr()
l0cOutUbSecondAddr = l0cOutUbSecond_.GetPhyAddr()
VFDoSwigluAndQuantForMX<..., false>(...)
```

3. `VFDoSwigluAndQuantForMX` 里非 interleaved 和 interleaved 是否只差 stride。

应该看到：

```cpp
if constexpr (IsInterleavedSrc) {
    nSrcUbAligned = nSize * 2;
} else {
    nSrcUbAligned = ceilAlign(nSize, 32 / sizeof(DataTypeIn));
}
```

这证明没有复制两套数学逻辑。

4. scale 是否真的 compact copy。

应该看到：

```cpp
blockLen = blockScaleN;
srcStride = 0;
dstStride = scaleN_ - blockScaleN;
DataCopyPad<int8_t, PaddingMode::Compact>(...)
```

这证明 GM 行距仍保留，下游 yScale 布局没有改成 UB staging 布局。

5. fp8 是否从四次 store 改成一次 store。

旧代码有四次 `DIST_PACK4_B32`。新代码应看到：

```cpp
CAST_32_TO_80/82/81/83
Add as uint8
DataCopy<StoreDist::DIST_NORM_B8>
```

这证明 port4 fp8 pack 路径被移植进来了。

## Review结论和检查点

这次 `-275` 主要删除的是：

- 旧 VF 主循环。
- 旧 fp8 多次 interleave + 多次 pack store。
- 旧 scale 32B staging。
- staging 相关成员和函数。
- 未使用 wrapper / 未使用临时寄存器。

这次 `+317` 主要增加的是：

- port4 风格 VF 主循环。
- lane-aware fp8 pack。
- compact scale copyout。
- interleaved 编译期模板参数和 pingpong 地址选择。
- 非 interleaved 默认兼容所需的 buffer 容量处理。
- 面向 review 的解释性注释。

review 时建议重点看四个地方：

1. 当前默认实例化是否仍是 `IsInterleaved_ = false`。
2. `CopyScaleFromUb2GmCompact` 的 GM 行距是否仍满足下游 yScale 布局。
3. `VFDoSwigluAndQuantForMX` 的非 interleaved stride 是否覆盖当前 master 两块 UB 输入。
4. 旧 clamp 被 port4 VF 路径替换后，目标精度用例是否覆盖这个行为差异。

我的判断是：当前 diff 的主线是清楚的，`-` 主要是在移除旧的低效中间重排和重复 VF 写法，`+` 主要是在引入已验证收益的 port4 寄存器路径，并把 interleaved 能力作为编译期模板预埋进去。当前仍建议在合入前补目标硬件上的编译、精度和性能结果，把第 4 点也用实测闭环。
