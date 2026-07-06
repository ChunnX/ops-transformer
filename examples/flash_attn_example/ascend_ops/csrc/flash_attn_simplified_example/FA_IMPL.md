# flash_attn_simplified_example 算子详解文档

## 一、算子设计

### 1.1 功能与架构

本算子基于华为昇腾 NPU 的 AscendC 框架，采用 **Host（CPU）拉起 + Device（NPU）跑 Kernel** 的架构，实现标准自注意力计算 $Attention(Q,K,V)=Softmax(\frac{QK^T}{\sqrt{d}})V$。接口参数详见 [README.md](./README.md)。

算子约束：Q/K/V 数据类型为 `bfloat16`，形状固定为 —— Q `[1, 8192, 32, 128]`、K/V `[1, 8192, 1, 128]`（`BSND` 布局），即 `B=1, N1=32, N2=1, S1=8192, S2=8192, D=128`，因此 `G = N1/N2 = 32`（GQA，32 个 query head 共享 1 个 KV head），计算稠密（全量）注意力。

调用链路：**Python** `torch.ops.ascend_ops.flash_attn_simplified_example(q,k,v,scale)` → **C++ 入口** `FlashAttnNpu()`（`torch_interface.cpp`）→ **Kernel 调度** `FaKernel`（`fa_kernel_interface.h`）→ 按 core 类型分发 `Cube`（`fa_block_cube.h`）/ `Vector`（`fa_block_vec.h`）。

入口函数校验固定形状后，直接把 GM 指针、`blockDim`、`softmaxScale` 传给 Kernel，Kernel 内部用编译期常量描述维度，无独立的 Host Tiling 阶段。

### 1.2 基本块与 GS1 合轴

Kernel 以 **M=128、S2(N)=128、D=128** 为基本 tile（模板参数 `M_BASE / N_BASE / D_SIZE`，`fa_kernel_interface.h`）。每个 AIC Cube core 配 2 个 AIV Vector core（`KERNEL_TYPE_MIX_AIC_1_2`），每个 AIV 处理半个 M（`M_BASE/2 = 64` 行），因此 BMM 结果 UB 中以 `64×128` 为半块单位。

**GS1 合轴**：GQA 下 `G` 个 query head 共享同一 KV head，因此把 Q 的 `(G, S1)` 两维合并为一条 **GS1 = S1 × G** 轴一起切 M-block。合轴索引 `gS1Idx` 在搬运层（`memory_copy.h` 的 `CopyQueryGmToL1` / `CopyAttnOutUbToGm`）拆回真实坐标：`s1 = gS1Idx / G`、`g = gS1Idx % G`，跨 G/S1 边界读写。由此一个 128 行的 M-tile 覆盖 `128 / G = 4` 个连续 S1 行 × 全部 32 个 G，且这 128 行都 attend 同一个 KV head。

固定形状派生出的循环上界（`fa_block_cube.h` / `fa_block_vec.h`）：

| 常量 | 值 | 含义 |
|------|-----|------|
| `NUM_BN2` | `B×N2 = 1` | (batch, KV head) 组数 |
| `NUM_GS1_BLOCKS` | `S1×G / M_BASE = 262144/128 = 2048` | GS1 合轴上的 M 块数 |
| `NUM_S2_BLOCKS` | `S2/N_BASE = 64` | 每行的 KV(flash) 块数 |

### 1.3 多核切分策略

由于 `B×N2 = 1`（只有一个 KV head 组），无法再按 head 分核，改为**按 GS1 block 均分**：

- 每个 AIC core 处理 `gs1PerCore = NUM_GS1_BLOCKS / blockDim` 个 GS1 块（`Cube` / `Vector`）；起止区间 `[aicIdx × gs1PerCore, +gs1PerCore)`。
- 入口固定 `fixedAicNum = 32`、`fixedAivNum = 64`，`blockDim = CalcTschBlockDim(64, ...)`（AIC 视角为 32）。故 `gs1PerCore = 2048/32 = 64`，**每个 AIC core 处理 64 个 GS1 块**。
- AIV 侧 `aicIdx = aivIdx / 2`，`subBlockIdx = GetSubBlockIdx()`（0/1），两个 AIV 分别负责该 tile 上半 / 下半 64 行。

合轴由搬运层原生处理，Kernel 的 BMM/Softmax/Rescale 只按抽象 128×… 块计算、不感知 G。每行都 attend 全部 S2，且一行不跨 core，各 core 产出即为最终结果，无需跨核归约。

### 1.4 内存划分

NPU 上存储层次从大到小为 GM → L1 → L0A/L0B/L0C → UB。所有 buffer 均以**显式线性地址**分配（`LocalTensor<T>(TPosition, addr, elems)`），并用 `get_buf/rls_buf` 事件原语手工做多缓冲同步；Cube 与 Vector 对**共享区（UB 的 BMM 结果、L1 的 P 矩阵）使用相同地址**以完成跨核数据交换。

#### GM（Global Memory）

Q 以 `BSNGD`（`FaGmTensor(B=1, N2=1, G=32, S1=8192, D=128)`）、K/V 以 `BSND`（`N2=1`）解读（`offset_calculator.h`）。输出与 Q 同形状 `[1,8192,32,128]`。workspace 为空 tensor（`at::empty({0})`），Kernel 不使用。

#### L1（AIC 侧暂存，`fa_block_cube.h`）

| Buffer | 大小 | 缓冲策略 |
|--------|------|---------|
| `l1P`（P matrix，`M×N` bf16） | 32KB × 3 = 96KB | triple-buffer（AIV 写、AIC 读） |
| `l1Q`（Q tile，`M×D` bf16） | 32KB × 2 = 64KB | double-buffer（同一 GS1 行内 Q 复用） |
| `l1K`（K tile，`N×D` bf16） | 32KB × 2 = 64KB | double-buffer（每个 S2 块重载） |
| `l1V`（V tile，`N×D` bf16） | 32KB × 2 = 64KB | double-buffer |
| **L1 总计** | **288KB** | 从 0 地址线性排布 |

#### L0（AIC Cube 专用，`fa_block_cube.h`）

| 空间 | 本算子分配 | 缓冲策略 |
|------|----------|---------|
| **L0A** | `128×128` bf16 × 2 | double-buffer，存 Q（BMM1）或 P（BMM2） |
| **L0B** | `128×128` bf16 × 2 | double-buffer，存 K（BMM1）或 V（BMM2） |
| **L0C** | `128×128` fp32 × 4 | **quad-buffer**，隐藏 Fixpipe 延迟 |

#### UB（AIV Vector 专用 + AIC Fixpipe 写入）

AIC 通过 Fixpipe（L0C→UB）把矩阵乘结果写入 UB 供 AIV 消费。`ubBmm1/ubBmm2` 在 Cube 和 Vector 中地址一致（跨核共享），其余为 AIV 私有。为配合 PRELOAD_N=2 的流水深度，`ubBmm1`、`stage1Buf`、`softmax*` 均取 `PRELOAD_N+1=3` 份。

| Buffer | 大小 | 块数 | 说明 |
|--------|------|:---:|------|
| `ubBmm1` | 32KB（64×128 fp32） | 3（`PRELOAD_N+1`） | BMM1 结果，Cube→UB，Vec 消费 |
| `ubBmm2` | 32KB（64×128 fp32） | 2（ping/pong） | BMM2 结果，Cube→UB，Vec 消费 |
| `softmaxSum` | 256B（64 fp32） | 3（`PRELOAD_N+1`） | Softmax 分母 sum（按行索引） |
| `softmaxMax` | 256B（64 fp32） | 3 | Softmax 分子 max（按行索引） |
| `softmaxExp` | 256B（64 fp32） | 3 | Flash Update 指数比例因子（按 tile 索引） |
| `commonBuf` | 512B | 1 | 通用临时空间（sum/max 更新用） |
| `stage1Buf` | ~16.5KB（(64+2)×128 bf16） | 3（`PRELOAD_N+1`） | Softmax 结果 P（cast bf16），搬往 L1 |
| `stage2Buf` | 32KB（64×128 fp32） | 1 | Flash Update 累加器 |
| `castBuf` | 与 `stage2Buf` **同址复用**（64×128 bf16） | 1 | 输出 fp32→bf16 就地 downcast |
| **总计** | | | **~244KB（< 248KB UB 容量）** |

> `stage2Buf` 与 `castBuf` 共享同一 UB 地址：`Cast` 读 fp32（4B/elem）写 bf16（2B/elem），写指针慢于读指针，可安全就地转换以节省 UB。

AIV 侧本地 `get_buf/rls_buf` 事件号：`stage1Buf` 三槽占 `{0,1,2}`，`E_VEC2_ID`（stage2/输出）取 `PRELOAD_N+1 = 3`，两者不冲突。

### 1.5 PRELOAD 流水设计

AIC 与 AIV 是不同物理 core，跑同一份二进制但走不同分支（`#ifdef __DAV_C310_CUBE__`）。一个 tile 的生命周期为 **C1（BMM1）→ V1（Softmax）→ C2（BMM2）→ V2（FlashUpdate/输出）**，跨核用 `CrossCoreSetFlag`/`CrossCoreWaitFlag` 做生产者-消费者同步。

算子采用 **PRELOAD_N=2** 的软件流水：每个 core 在同一轮迭代中同时推进「当前 tile 的 Phase1」与「2 轮前 tile 的 Phase2」，从而用更深的流水掩盖 V1（Softmax + P 写 L1）较长的延迟。

```
        AIC Core                        AIV Core
        ────────                        ────────
loop 0: C1(T0)                          V1(T0)
loop 1: C1(T1)                          V1(T1)
loop 2: C1(T2) + C2(T0)                 V1(T2) + V2(T0)
loop 3: C1(T3) + C2(T1)                 V1(T3) + V2(T1)
...
loop k: C1(Tk) + C2(T(k-2))             V1(Tk) + V2(T(k-2))
```

从 loop=2 进入稳态：Cube 每轮发 `ComputeMm1(i)`，`i >= PRELOAD_N` 时再发 `ComputeMm2(i-2)`；Vector 每轮发 `ComputeSoftmax(i)` + `CopySoftmaxToL1`，`i >= PRELOAD_N` 时再发 `ComputeRescale(i-2)`。主循环结束后各 core 依次 flush 最后 **2** 个 tile 的 Phase2。

**Buffer 轮换索引**：`ubBmm1`/`stage1Buf`/`l1P` 用 `i % 3`；`ubBmm2`/`l1K`/`l1V` 用 `&1`（DB，Phase2 用 tile 号 `j=i-2` 的低位）；`l0C` 用 `% 4`（QB）；Q 用 `gs1Block & 1`。

**跨核 flag ID 映射**（每个 base ID 另有 `+16` 镜像对应第 2 个 AIV 子核）：

| 资源 | flag ID | 索引 |
|------|---------|------|
| bmm1（free/ready） | `{0,1,2}` | `i % 3` |
| bmm2（free/ready） | `{3,4}` | `3 + (j & 1)` |
| l1P（ready） | `{5,6,7}` | `5 + (j % 3)` |

Vector 初始预置 `{0,1,2,3,4}`（3 个 bmm1-free + 2 个 bmm2-free）；Cube 主循环结束后 barrier 等待 `{0,1,2,3,4}` 及其 `+16` 镜像，确保退出前 AIV 已消费完共享 buffer。

> **l1P 三缓冲安全性**：覆写 `l1P[j%3]` 的是 V1(j+3)，它需等 Cube 产出 bmm1(j+3)（`C1(j+3)`），而在 Cube 同核程序序中 `C2(j)`（发生在第 `j+PRELOAD_N` 轮）先于 `C1(j+3)`（第 `j+3` 轮）。只要 `PRELOAD_N < 3`，`C2(j)` 一定先消费完 `l1P[j%3]`，故 3 份 l1P 对 PRELOAD_N=2 恰好安全。

---

## 二、代码结构说明

### 2.1 整体架构概览

```
torch_interface.cpp              ← PyTorch 注册 + 入口（固定 shape 校验，直接拉起 Kernel）
  └─ op_kernel/fa_kernel_interface.h   ← Device Kernel 入口（按 core 类型 #ifdef 分发 Cube/Vector）
       ├─ kernel/fa_block_cube.h       ← AIC Cube：Q/K/V 搬运 + BMM1/BMM2 + Fixpipe
       └─ kernel/fa_block_vec.h        ← AIV Vector：Softmax + Flash Update + 输出
```

附属功能模块：

| 目录/文件 | 功能 |
|------|------|
| `op_kernel/fa_kernel_public.h` | 公共头/命名空间引入 |
| `op_kernel/memcopy/offset_calculator.h` | GM 多维索引（`FaGmTensor`，本例用到 `BSNGD`/`BSND`）+ 坐标结构体 |
| `op_kernel/memcopy/memory_copy.h` | `CopyQueryGmToL1`（ND→NZ，含 G 合轴拆分）、`CopyKvGmToL1`、`CopyAttnOutUbToGm` |
| `op_kernel/matmul/matmul.h` | `MatmulFull`（L1→L0A/L0B `LoadData` + `Mmad`）、`MMParam` |
| `op_kernel/vector/vf/vf_softmax.h` | VF 指令加速的 Flash Softmax：`ProcessVec1Vf`（update / no_update，`EQ_128` 路径）+ `UpdateExpSumAndExpMax` |
| `op_kernel/vector/vf/vf_softmax_impl/` | `vf_softmax_aligned128_{no_update,update}.h` 具体实现 + `vf_softmax_const.h` 常量 |
| `op_kernel/vector/vf/vf_rescale.h` | `FlashUpdateNew` / `FlashUpdateLastNew` / `LastDivNew`（Flash Update Rescale） |

### 2.2 Torch 接口：`torch_interface.cpp`

唯一入口 `FlashAttnNpu()`：
1. `CheckInput` 校验 bf16、4 维、`D=128`，且强约束 Q 为 `[1,8192,32,128]`、K/V 为 `[1,8192,1,128]`（否则报错返回）。
2. 取平台信息与当前 NPU stream，分配同形状输出与空 workspace。
3. 固定 `fixedAicNum=32 / fixedAivNum=64`，`blockDim = CalcTschBlockDim(64, GetCoreNumAic(), GetCoreNumAiv())`。
4. 以 `FaKernel<<<blockDim, nullptr, aclstream>>>(gq, gk, gv, go, gws, blockDim, scale)` 拉起 Kernel，随后 `aclrtSynchronizeStream` 同步。

算子通过 `TORCH_LIBRARY(ascend_ops, ...)` 注册 `flash_attn_simplified_example(Tensor q, Tensor k, Tensor v, float softmaxScale=0) -> Tensor`。

### 2.3 Device Kernel：`op_kernel/`

#### 2.3.1 入口与分发（`fa_kernel_interface.h`）

`FaKernel` 设置 `KERNEL_TYPE_MIX_AIC_1_2` 与 `InitSocState()` 后，按编译宏分发：`__DAV_C310_CUBE__` → `Cube<M,N,D>`，否则 → `Vector<M,N,D>`。模板默认 `M=N=D=128`。

#### 2.3.2 Cube 侧（`fa_block_cube.h`）

`Cube()` 建立 GM 张量、分配 L1/L0/UB，然后按 `gs1Block → s2Block` 两层循环推进流水（PRELOAD_N=2，Phase2 处理 tile `j=i-2`，主循环后 flush 最后 2 个 tile；`prevS2` 由增量计数器 `p2S2` 给出）：

- **`ComputeMm1`（BMM1：Q@Kᵀ）**：
  1. 等待 `ubBmm1` 空闲（`CrossCoreWaitFlag`）。
  2. `isFirstS2Loop` 时 `CopyQueryGmToL1`（Q GM→L1，ND→NZ，含 G 合轴拆分，整行复用）。
  3. 每块 `CopyKvGmToL1`（K GM→L1）。
  4. `MatmulFull`（L1→L0A/L0B → `Mmad` → L0C），`isRightTranspose=true` 实现 Kᵀ。
  5. `FixpipeMm1`（L0C→UB，`dualDstCtl=1` 拆成两份 `64×128` 给 2 个 AIV）→ `CrossCoreSetFlag` 通知 AIV。
- **`ComputeMm2`（BMM2：P@V）**：
  1. 等待 BMM2 UB 空闲 + 等待 AIV 把 P 写入 L1（`CrossCoreWaitFlag`）。
  2. `CopyKvGmToL1`（V GM→L1）。
  3. `MatmulFull`（P@V，`realM=M_BASE`）→ `FixpipeMm2`（L0C→UB dual-dest）→ `CrossCoreSetFlag`。

#### 2.3.3 Vector 侧（`fa_block_vec.h`）

`Vector()` 分配共享 UB（`ubBmm*`、`l1P`）与私有 UB，初始化 5 个跨核 flag，然后按 `gs1Block → s2Block` 两层循环。Phase1 用行计数 `mloop` 索引 softmax 统计量；Phase2（处理 tile `j=i-2`）用一对**增量计数器 `p2S2`/`p2Row`** 直接得到该 tile 的「行内首块 / 行末块 / 行号」，避免滞后反推：

- **`ComputeSoftmax`（V1）**：等待 BMM1 结果 → 首块走 `ProcessVec1Vf<...false>`（no_update），非首块走 `<...true>`（update）：逐行 `QKᵀ×scale → row_max → exp(x-max) → sum → cast bf16`；非首块再 `UpdateExpSumAndExpMax` 合并历史 max/sum。
- **`CopySoftmaxToL1`**：把 P（bf16）`DataCopy` 到 `l1P[i%3]`（按 `subBlockIdx` 定位上/下半），`CrossCoreSetFlag` 通知 Cube 做 BMM2、并释放 BMM1 UB。
- **`ComputeRescale`（V2）**：
  - `jIsFirst`（`p2S2==0`）→ `DataCopy` 初始化累加器 `stage2Buf`；
  - 中间块 → `FlashUpdateNew`（`stage2 = stage2·expMax + cur`）；
  - `jIsLast`（`p2S2==NUM_S2_BLOCKS-1`）→ `FlashUpdateLastNew`（累加后除以 sum）。
- **`CopyAttnOutToGm`**：仅当 tile 为某行最后一块（`jIsLast`）时触发，`Cast` fp32→bf16（就地）后 `CopyAttnOutUbToGm` 写回（输出坐标 `gS1Idx = (gs1Start + p2Row) × M_BASE`）。

每处理完一个 Phase2 tile 后推进计数器：`if (++p2S2 == NUM_S2_BLOCKS) { p2S2 = 0; p2Row++; }`。

---

## 三、典型 Case 代入走读

固定 case：**B=1, N1=32, N2=1, S1=8192, S2=8192, D=128**（`G=32`，GQA，稠密全量注意力），运行于 32 AIC + 64 AIV。

### 3.1 多核分配

- `NUM_BN2 = 1×1 = 1`（单个 KV head 组），无法按 head 分核。
- GS1 合轴长度 = `S1×G = 8192×32 = 262144`，`NUM_GS1_BLOCKS = 262144/128 = 2048`。
- `blockDim`（AIC 视角）= 32 → `gs1PerCore = 2048/32 = 64`。**每个 AIC core 处理 64 个 GS1 块**，对应 2 个 AIV core（处理每块的上/下半 64 行）。
- 每个 core 内：`64` 个 GS1 块 × `NUM_S2_BLOCKS=64` 个 KV 块 = 4096 个 tile，按 flat 序 `i` 走流水。

### 3.2 单 tile 计算流程（core 处理 GS1 块 gs1Block、KV 块 s2Block）

一个 128 行的 M-tile 对应 GS1 合轴上 `[gs1Block×128, +128)`：因 `128 = 4×G`，即 **4 个连续 S1 行 × 全部 32 个 G**，这 128 个 query 行都 attend 同一个 KV head（`n2=0`）。

**Step A — BMM1（AIC Cube）**
1. **Copy Q**（仅 `s2Block==0`）：`CopyQueryGmToL1` 按 `s1=gS1Idx/32`、`g=gS1Idx%32` 跨 G 读取 128 行 → L1（NZ，128×128 bf16=32KB），整行复用。
2. **Copy K**：`K[0, 0, s2Block×128 : +128, :]` GM→L1（NZ，32KB）。
3. **Matmul**：`Q(128×128) × Kᵀ(128×128)` → L0C。
4. **Fixpipe** L0C→UB：dual-dest 拆 2×`64×128` fp32 → 通知两个 AIV。

**Step B — Vec1 / Softmax（AIV Vector，每个子核 64 行）**
1. 等待 BMM1 → `ProcessVec1Vf`（首块 no_update / 非首块 update，`EQ_128` aligned128）：`QKᵀ×scale → max → exp → sum → cast bf16`。
2. 非首块 `UpdateExpSumAndExpMax` 合并统计量。
3. `DataCopy` P(bf16) → `l1P`，`CrossCoreSetFlag` 通知 Cube。

**Step C — BMM2（AIC Cube，滞后 2 拍）**
1. 等待 L1 中的 P → **Copy V** GM→L1（NZ，32KB）。
2. `MatmulFull` `P(128×128) × V(128×128)`（`realM=128`）→ Fixpipe L0C→UB dual-dest → 通知 AIV。

**Step D — Vec2 / Flash Update（AIV Vector，滞后 2 拍）**
- 行首块：`DataCopy` 初始化累加器；中间块：`FlashUpdateNew` 累加；行末块：`FlashUpdateLastNew`（累加 + 除以 sum）→ `Cast` bf16 → `CopyAttnOutUbToGm` 写回（`CopyAttnOutUbToGm` 同样按 G 合轴拆分写回）。

如此，每行 64 个 S2 块通过 flash online-softmax 累加为一个 `128×128` 输出块；2048 个 GS1 块（分布在 32 个 core）覆盖全部输出。

---

## 四、性能测试

以下场景在 950DT 平台上完成测试。

| B | Q_N | KV_N | Q_S | KV_S | D | Duration | MFU |
|---|-----|------|-----|------|---|----------|-----|
| 1 | 32 | 1 | 8192 | 8192 | 128 | 2542.00 | 87.17% |
