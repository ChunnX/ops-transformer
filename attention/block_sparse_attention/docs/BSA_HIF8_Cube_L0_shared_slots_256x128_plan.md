# BSA HIF8 Cube L0 共享槽位与 256×128 改造方案

## 0. 文档状态：两个候选方案并列保留

本文当前记录两个**并列待选**的实现方案，尚未确定最终采用哪一个：

- **候选方案 A（第 1～7 节）**：MM1 整体采用一个 `256×128` L0 tile；每个逻辑阶段使用
  32 KiB 的 L0A/L0B 外层槽和 128 KiB 的 L0C 外层槽。
- **候选方案 B（第 8 节）**：总体空间仍为相同的外层双槽，但每个外层槽内部再拆成两个
  `128×128` 微槽，用两次 `128×128` 完成 MM1 的 `256×128`，尝试形成 Cube/FixPipe 细粒度穿插。

两个方案的内容都必须保留，并分别实现、编译、验证和采集性能数据。最终选择以 pytest、
simulator 时间线和 msprof 总耗时为依据，当前不预设方案 A 或方案 B 胜出。

## 1. 目标

仅调整 BSA HIF8 路径，使 Cube 基础块和 L0 资源组织与 QFA HIF8 的全载路径一致：

- MM1：`K[256,128] × Q^T[128,128] -> S^T[256,128]`
- MM2：`P[128,256] × V[256,128] -> OTmp[128,128]`
- L0A：64 KiB 共享池，两个 32 KiB 槽位
- L0B：64 KiB 共享池，两个 32 KiB 槽位
- L0C：256 KiB 共享池，两个 128 KiB 槽位
- MM1 和 MM2 按真实发射顺序循环复用同一组物理槽位
- 使用绝对 `stage` 计数，进入 MM1/MM2 后通过 `stage % 2` 选择槽位

非 HIF8 的 FP16/BF16/FP8/MXFP4 路径保持不变。

## 2. 已确认的前提

Host tiling 中 HIF8 已固定：

```cpp
qBaseTile_ = 128;
kvBaseTile_ = 256;
```

HIF8 支持的 `D` 为 64 或 128，按 128 对齐后仍只需一个 MM1 K stage。由此每个 KV 基块满足：

| 阶段 | M | N | K | L0A | L0B | L0C FP32 |
|---|---:|---:|---:|---:|---:|---:|
| MM1 | 256 | 128 | 128 | 32 KiB | 16 KiB | 128 KiB |
| MM2 | 128 | 128 | 256 | 32 KiB | 32 KiB | 64 KiB |

MM2 的“256×128”指 V 的 `K×N`，MM2 的 L0C 输出仍是 `128×128`。为了与 MM1 共用 L0C，MM2 也以 128 KiB 为一个槽位的所有权单位，剩余空间不另作分配。

## 3. 绝对 stage 和真实发射顺序

在每个 Q 基块进入 KV 流水前初始化：

```cpp
uint64_t l0Stage = 0;
```

MM1/MM2 共用这个变量，并以引用传递：

```cpp
blockMmadQK(..., l0Stage, ...);
blockMmadPV(..., l0Stage, ...);
```

每次 MMAD 调用内部先取得本次 stage：

```cpp
const uint64_t currentStage = l0Stage++;
const uint32_t bufId = currentStage % L0_STAGES; // L0_STAGES == 2
```

`PRE_LAUNCH=2` 时，槽位自然跟随真实控制流：

| 发射顺序 | 绝对 stage | bufId |
|---|---:|---:|
| MM1(0) | 0 | 0 |
| MM1(1) | 1 | 1 |
| MM1(2) | 2 | 0 |
| MM2(0) | 3 | 1 |
| MM1(3) | 4 | 0 |
| MM2(1) | 5 | 1 |

该方式直接记录真实发射顺序，不再根据 KV tile 下标和 `PRE_LAUNCH` 反推前缀。

## 4. L0 地址和事件规划

### 4.1 地址

```text
L0A slot0: offset 0
L0A slot1: offset 32 KiB

L0B slot0: offset 0
L0B slot1: offset 32 KiB

L0C slot0: offset 0
L0C slot1: offset 128 KiB
```

MM1 与 MM2 都从上述地址取槽位，不再将 L0C 静态分为 MM1/MM2 两个半区。

### 4.2 Event ID

不同硬件事件类型拥有独立的同步通道，保留以下映射：

```text
L0A MTE1 <-> M : event bufId       // 0, 1
L0B MTE1 <-> M : event bufId + 2   // 2, 3
L0C M <-> FIX   : event bufId       // 0, 1
```

MM2 当前使用的 L0C `eventId = bufId + 2` 要改为 `eventId = bufId`。初始化阶段只需要为共享 L0C 初始化 event 0、1。

事件负责保证：即使 AIC 已经发射下一次 MM1/MM2，旧槽位在 MTE1、M 或 FixPipe 完成前也不会被覆盖。

## 5. 候选方案 A 的文件级修改

### 5.1 HIF8 专用 dispatch policy

文件：

```text
attention/block_sparse_attention/op_kernel/attn_infra/gemm/bsa_gemm_dispatch_policy.hpp
```

增加两个仅供 HIF8 使用的策略类型：

```cpp
template <bool transposedMm1_ = true>
struct MmadAtlasA5BsaQKHif8 : public MmadAtlasA5 {
    static constexpr uint32_t L0_STAGES = 2;
    static constexpr bool transposedMm1 = transposedMm1_;
};

template <bool transposedMm1_ = true>
struct MmadAtlasA5BsaPVHif8 : public MmadAtlasA5 {
    static constexpr uint32_t L0_STAGES = 2;
    static constexpr bool transposedMm1 = transposedMm1_;
};
```

这样 `_fp8.hpp` 与原 MMAD 模板特化隔离，不影响非 HIF8 路径。

### 5.2 引入 HIF8 MMAD 文件

文件：

```text
attention/block_sparse_attention/op_kernel/attn_infra/gemm/block/bsa_block_mmad.hpp
```

在 310 分支加入：

```cpp
#include "block_mmad_k_qt_st_arch35_fp8.hpp"
#include "block_mmad_pv_arch35_ABf16_C_to_UB_fp8.hpp"
```

### 5.3 MM1 HIF8

文件：

```text
attention/block_sparse_attention/op_kernel/attn_infra/gemm/block/block_mmad_k_qt_st_arch35_fp8.hpp
```

修改内容：

1. 改为独立 include guard。
2. 模板特化切换到 `MmadAtlasA5BsaQKHif8<true>`。
3. L0C 槽位大小改为整个 L0C 的一半：

   ```cpp
   L0C_PINGPONG_BUF_SIZE = ArchTag::L0C_SIZE / L0_STAGES;
   ```

4. `operator()` 的两个 prefix 参数改为一个 `uint64_t &l0Stage`。
5. 入口处执行：

   ```cpp
   const uint32_t bufId = (l0Stage++) % L0_STAGES;
   ```

6. 本次 MM1 的 L0A、L0B、L0C 都使用该 `bufId`。
7. 保留当前两路 scaled FixPipe，两次 FixPipe 完成后才释放 L0C slot。

MM1 L0 tile 改为 `Shape<256,128,128>` 后，完整块和尾块的 M/N/K 循环次数均不超过 1。

### 5.4 MM2 HIF8

文件：

```text
attention/block_sparse_attention/op_kernel/attn_infra/gemm/block/block_mmad_pv_arch35_ABf16_C_to_UB_fp8.hpp
```

修改内容：

1. 改为独立 include guard。
2. 模板特化切换到 `MmadAtlasA5BsaPVHif8<true>`。
3. L0C 槽位大小改为 `ArchTag::L0C_SIZE / L0_STAGES`。
4. 删除原来的 `L0C_HALF_BUF_SIZE` 起始偏移。
5. 两个 prefix 参数改为同一个 `uint64_t &l0Stage`。
6. 入口处以绝对 stage 计算统一 `bufId`。
7. L0A、L0B、L0C 使用相同的 `bufId`。
8. L0C event 从 `bufId + 2` 改为 `bufId`。

MM2 L0 tile 改为 `Shape<128,128,256>`，使 P 和 V 的 KV=256 在一次 MMAD reduction tile 中完成。

### 5.5 HIF8 kernel 调度

文件：

```text
attention/block_sparse_attention/op_kernel/arch35/block_sparse_attention_kernel_arch35_hif8_full_quant.h
```

修改内容：

1. 在 KV 流水循环前声明 HIF8 Cube 绝对 stage。
2. MM1/MM2 调用都传入该变量的引用。
3. 删除 `CalcCrossMm1Mm2PrefixSumL0ABStages()`。
4. 删除四个 MM1/MM2 L0 stage 统计成员及其计算。
5. L0C 初始化只保留共享 slot 对应的 event 0、1。

stage 只在 MM1 和 MM2 所在的 `__DAV_CUBE__` 控制流中维护；Vector 侧不读取或修改。

### 5.6 HIF8 interface

文件：

```text
attention/block_sparse_attention/op_kernel/block_sparse_attention_kernel_interface.cpp
```

只修改 `BsaInferInterfaceHIF8FullQuant`：

```cpp
using L0TileShapeQK = Shape<Int<256>, Int<128>, Int<128>>;
using DispatchPolicyQK = Gemm::MmadAtlasA5BsaQKHif8<true>;

using L0TileShapePV = Shape<Int<128>, Int<128>, Int<256>>;
using DispatchPolicyPV = Gemm::MmadAtlasA5BsaPVHif8<true>;
```

原 FP8/非量化接口继续使用原 dispatch policy 和原 MMAD 文件。

## 6. 候选方案 A 的实施与验证顺序

1. 增加 HIF8 dispatch、独立 include guard 和 include 接线。
2. 修改 MM1/MM2 tile、共享 L0 地址和 event。
3. 引入绝对 `l0Stage` 并删除前缀反推函数。
4. 编译：

   ```bash
   bash build.sh --pkg --ops="block_sparse_attention" --soc=ascend950 -j32
   ```

5. pytest 先验证：

   - `qS=128, kvS=256, D=128, dense`
   - `qS=128, kvS=256, D=64, dense`
   - Q/KV 尾块
   - `kvS>=1024`，覆盖 `MM1, MM1, MM1, MM2` 流水

6. 小用例通过后运行 BNSD/BSND/TND、padding、稀疏场景和 generated 600 cases。

## 7. 候选方案 A 的验收标准

- HIF8 interface 命中新 MMAD 文件；
- 非 HIF8 路径没有行为变化；
- 编译和安装成功；
- 完整块、尾块和多 KV tile 均无死锁或 event 错误；
- pytest 精度通过；
- 后续 msprof 能观察到 MM1 `256×128` 和 MM2 `K=256`。

## 8. 候选方案 B：外层共享双槽，槽内再切两个 128×128 微槽

### 8.1 设计动机

方案 A 把一个完整 MM1 `256×128` 结果作为一个 L0C 槽位的所有权单位：Cube 完成
`256×128` 后，再由 FixPipe 搬出整个结果。备选方案 B 保持总 L0 容量和外层共享关系不变，
但把每个外层槽再切成两个微槽，将 MM1 的 `256×128` 沿 M 方向拆成两次相互独立的
`128×128` 计算：

```text
K[  0:128, 0:128] × Q^T[0:128, 0:128] -> S^T[  0:128, 0:128]
K[128:256, 0:128] × Q^T[0:128, 0:128] -> S^T[128:256, 0:128]
```

两个输出不存在 reduction 依赖，因此可以尝试形成以下穿插：

```text
Cube(micro 0) -> FixPipe(micro 0)
                    ||
                 Cube(micro 1) -> FixPipe(micro 1)
```

这里的性能收益是候选假设，不作为硬件既定事实。是否真正形成 Cube/FixPipe DB，必须看
simulator 或 msprof 时间线；如果生成的事件等待仍将两条流水串行化，则拆分只会增加指令开销。

### 8.2 两级空间划分

总体资源保持：

| 资源 | 总容量 | 外层 | 每个外层槽的内层划分 |
|---|---:|---:|---:|
| L0A | 64 KiB | `2 × 32 KiB` | `2 × 16 KiB` |
| L0B | 64 KiB | `2 × 32 KiB` | `2 × 16 KiB` |
| L0C | 256 KiB | `2 × 128 KiB` | `2 × 64 KiB` |

容量计算：

```text
128 × 128 × sizeof(HIF8)  = 16 KiB
128 × 128 × sizeof(float) = 64 KiB
```

物理地址规划：

```text
L0A/L0B outer 0, inner 0: offset   0 KiB
L0A/L0B outer 0, inner 1: offset  16 KiB
L0A/L0B outer 1, inner 0: offset  32 KiB
L0A/L0B outer 1, inner 1: offset  48 KiB

L0C outer 0, inner 0: offset   0 KiB
L0C outer 0, inner 1: offset  64 KiB
L0C outer 1, inner 0: offset 128 KiB
L0C outer 1, inner 1: offset 192 KiB
```

对应地址公式：

```cpp
uint32_t outerId = logicalStage % 2;
uint32_t innerId = microIdx % 2;

uint32_t l0ABOffset = outerId * 32_KiB + innerId * 16_KiB;
uint32_t l0COffset  = outerId * 128_KiB + innerId * 64_KiB;
```

`logicalStage` 仍由 MM1/MM2 共享，并按照真实的逻辑阶段发射顺序递增。`microIdx` 只表示
当前逻辑阶段内部的第几个 `128×128` 微计算，不能代替跨 MM1/MM2 的绝对阶段。

### 8.3 MM1 的计算和流水

MM1 对外仍处理一个 `256×128` KV 基块，但 L0 tile 保持为：

```cpp
Shape<128, 128, 128>
```

每个逻辑 MM1 执行两个 M 微块：

| microIdx | L0A | L0B | L0C | 输出位置 |
|---:|---:|---:|---:|---|
| 0 | 16 KiB | 16 KiB | 64 KiB | `S^T[0:128, :]` |
| 1 | 16 KiB | 16 KiB | 64 KiB | `S^T[128:256, :]` |

候选发射顺序：

```text
1. MTE1 搬入 K0/Q -> A[outer,0]、B[outer,0]
2. Cube 计算 C[outer,0]
3. 向 FixPipe 发布 C[outer,0]
4. MTE1/Cube 使用 inner 1 计算 C[outer,1]
5. FixPipe 搬出 C[outer,0]，并与步骤 4 尝试并行
6. 向 FixPipe 发布并搬出 C[outer,1]
```

Q 的 `128×128` 数据在两个 M 微块中相同。第一版可以将 Q 分别搬入两个 L0B 微槽，优先保证
事件和所有权关系清晰；正确性稳定后，再评估固定一个 Q 微槽并跨两个 M 微块复用，以减少一次
L1->L0B 搬运。复用优化不能破坏下一逻辑阶段对该物理槽的所有权。

当前 HIF8 MM1 的 scaled FixPipe 还会沿 N 方向分别向两个 AIV 输出。槽位只有在该
`128×128` 微块对应的两路 FixPipe 都结束后才能释放，不能在第一路搬出完成后提前复用。

### 8.4 MM2 的处理边界

MM2 为：

```text
P[128,256] × V[256,128] -> OTmp[128,128]
```

沿 K=256 拆成两个 K=128 时，两次计算是同一个 `128×128` 输出的部分和：

```text
OTmp = P[:, 0:128]   × V[0:128, :]
     + P[:, 128:256] × V[128:256, :]
```

因此 MM2 可以用两个 16 KiB L0A/L0B 微槽对 K 分块做搬入和 Cube DB，但两次 MMAD 必须
累加到同一个 64 KiB L0C 微槽。第一个 K 微块结束后不能启动最终 FixPipe，必须等第二个 K
微块完成。方案 B 的主要 Cube/FixPipe 重叠收益来自 MM1，而不是单个 MM2 的 K 分块。

初版建议 MM2 固定使用当前 `outerId` 下的 `innerId=0` 作为累加输出；另一个 64 KiB L0C
微槽暂不利用。只有在确认存在两个相互独立的 MM2 输出并能安全并发后，才考虑使用第二个微槽。

### 8.5 Event 规划

L0C 有四个可独立复用的 64 KiB 微槽，候选映射为：

```cpp
uint32_t l0CEventId = outerId * 2 + innerId; // 0, 1, 2, 3
```

每个 L0C 微槽必须完成闭环：

```text
Wait FIX_M -> Cube -> Set/Wait M_FIX -> FixPipe -> Set FIX_M
```

只有使用不同 `l0CEventId`，FixPipe(micro 0) 才不会阻止 Cube(micro 1) 获取自己的 L0C 微槽。

L0A/L0B 的事件预算需要单独验证。四个 A 微槽和四个 B 微槽如果全部独立管理，会需要比当前
方案更多的 MTE1<->M event。第一版可将同一个 `(outerId, innerId)` 下的 A/B 视为一个生命周期
配对：两者均完成搬入后统一通知 Cube，Cube 完成后统一释放。若现有事件原语不能安全表达这种
配对，则需要减少同时在途的 A/B 微槽数量，不能仅按地址容量推断事件一定够用。

### 8.6 与方案 A 的对比

| 项目 | 方案 A：整块 256×128 | 方案 B：两个 128×128 |
|---|---|---|
| MM1 Cube 次数 | 1 个大 M tile | 2 个 M 微块 |
| 单个 MM1 L0C 所有权 | 128 KiB | `2 × 64 KiB` |
| Cube/FixPipe 粒度 | 256×128 | 128×128 |
| MM1 Cube/FixPipe 重叠机会 | 较粗 | 更细，可尝试微块间 DB |
| 指令/event 开销 | 较少 | 较多 |
| L0B Q 复用 | 一次 | 初版可能重复搬入一次 |
| 尾块处理 | 单路径 | 需要分别处理 micro 0/1 有效行数 |
| 实施风险 | 地址和事件较简单 | 事件预算、跨微块所有权更复杂 |

### 8.7 验证方法和选择标准

方案 A 与方案 B 必须使用同一份代码基线、同一组 case 分别编译测试：

1. pytest 验证完整块、`D=64/128`、KV 尾块以及多 KV tile 精度。
2. simulator 查看 M、FIX、MTE1 时间线，确认是否出现：

   ```text
   FixPipe(micro 0) 与 Cube(micro 1) 时间区间重叠
   ```

3. 统计新增的 MMAD、MTE1、event wait/set 指令开销。
4. msprof 比较 `Cube Utilization`、FixPipe 等待和总耗时。
5. 重点覆盖 `kvS>=1024`，确认共享槽在 `MM1, MM1, MM1, MM2` 发射序列下无覆盖和死锁。

选择标准不预设结论：方案 B 只有在时间线确认形成稳定的 Cube/FixPipe 穿插，并且总耗时优于
方案 A 时才胜出；如果方案 A 更快或方案 B 的额外指令/event 开销抵消了穿插收益，则选择方案 A。
在两组结果完成前，两个方案都保持待选状态，不删除任何一个方案。
