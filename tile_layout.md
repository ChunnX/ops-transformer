# PTO Tile 系统：存储格式、内存大小与数据搬运指令

## 一、内存单元（Tile）的存储格式与大小

### 1.1 七种 Location（物理位置类型）

| Location | 含义 | 是否装箱 | 内层分形尺寸 | 装箱大小 | 用途 |
|----------|------|---------|------------|---------|------|
| `Vec` | 通用向量/寄存器 tile | `NoneBox` (无) | `1×1` | 无 | 逐元素运算（EXP/ADD/MUL...） |
| `Left` (L0A) | 矩阵乘左操作数 | `SLayout::RowMajor` | `16 × (256/bits)` | **512B** | Q×K^T 的 Q 侧、P×V 的 P 侧 |
| `Right` (L0B) | 矩阵乘右操作数 | `SLayout::ColMajor` | `(256/bits) × 16` | **512B** | Q×K^T 的 K 侧、P×V 的 V 侧 |
| `Acc` | 矩阵乘累加器 | `SLayout::RowMajor` | `16×16` (仅 FP32) | **1024B** | QK^T 和 P×V 的输出 |
| `Scaling` | 缩放因子 tile | `NoneBox` | — | 无 | HiF4 MX 量化的 per-tile scale |
| `Bias` | 偏置 tile | (未使用) | — | — | 保留 |
| `Mat` | 矩阵 tile | (未使用) | — | — | 保留 |

以 FP16 为例，`Left`(Nz 格式) 的物理结构：

```
外层 BLayout::ColMajor          ← box 之间按列主序遍历
  内层 SLayout::RowMajor        ← 每个 512B box 内部按行主序切 16×16 子块
    物理最小单元: 32B lane      ← 每个 lane 宽度 256 bit
```

### 1.2 大小约束

| 约束 | 数值 |
|------|------|
| 最小 tile 大小 | 512B |
| 最大 tile 大小 | 64KB (65536B) |
| 大小步进 | 512B × 2^n（512, 1024, 2048, ..., 65536） |
| 32B 对齐条件 (NoneBox/RowMajor) | Cols 必须是 32 字节的整数倍 |
| 32B 对齐条件 (NoneBox/ColMajor) | Rows 必须是 32 字节的整数倍 |
| 分形约束 (Left/Right) | Rows 必须整除 16；Cols 必须整除 `256/bits` |
| 分形约束 (Acc) | Rows、Cols 必须整除 16 |

### 1.3 数据通路分类

- **CUBE**: 矩阵乘/累加引擎 — `TMATMUL`、`TMATMUL_ACC`、`ACCCVT`
- **TMA**: 张量内存访问 — `TLOAD`、`TSTORE`、布局转换
- **TEPL**: 逐元素处理通道 — 算术、数学函数、规约、广播、类型转换
- **MTE**: 多线程引擎 — 软件模拟/向量 kernel

---

## 二、数据移动/copy 指令（按源→目标格式）

### 2.1 Global ↔ Tile（TMA 通路）

| 指令 | 从 | 到 | 支持的布局转换 |
|------|----|----|--------------|
| `TLOAD` | Global (RowMajor/ColMajor) | Tile (Left/Right/Vec) | NORM、ND→NZ、DN→ZN、ND→ZN |
| `TSTORE` | Tile (Left/Right/Vec) | Global (RowMajor/ColMajor) | NZ→ND、ZN→DN、NORM |
| `TLOAD2_*` | Global | 2 个 Tile | ND→NZ、ND→ZN、DN→ZN |
| `TLOAD4_*` | Global | 4 个 Tile | ND→NZ、ND→ZN、DN→ZN |
| `TSTORE2_DN2DN` | 2 个 Tile (DN) | Global | NORM |

### 2.2 Acc → Vec（CUBE 通路）

| 指令 | 从 | 到 | 布局转换 |
|------|----|----|---------|
| **`ACCCVT`** | `Tile<Acc>` (Nz, 1024B) | `Tile<Vec>` | NZ→ND、NZ→DN、NZ→ZN、NORM |
| `ACCCVT_RMAX_SCALE` | Acc (Nz) | Vec + row_max + scale | NZ→DN（附带行最大提取和缩放） |
| `ACCSCALE` | Acc (Nz) | Vec (Nz) + scale | NZ→NZ（带标量缩放） |
| `ACCSCALE_NZ2DN` | Acc (Nz) | Vec (ColMajor) + scale | NZ→DN（带标量缩放） |

### 2.3 Tile ↔ Tile 布局转换

| 指令 | 从 | 到 | 硬件通路 |
|------|----|----|---------|
| **`TCVT`** | 任意 Tile (含 Acc) | 非 Acc Tile | Acc→走 TEPL `ACCCVT`；装箱→走 TMA `TMOV_*`；非装箱→走 `TCVT_T` (TEPL) |
| `TCOPY` | 同类型 Tile | 同类型 Tile | MTE（按位精确拷贝） |
| `TRESHAPE` | 同 Numel Tile | 任意形状 Tile | 直接 data 赋值（reinterpret） |
| `TTRANS` | `Tile[Rows×Cols]` | `Tile[Cols×Rows]` | MTE（转置） |
| `TEXTRACT` | 大 Tile | 子区域 Tile | MTE |
| `TFILLPAD` | 有效区 Tile | 补齐后 Tile | MTE |

### 2.4 Vec ↔ Left 装箱/拆箱

| 指令 | 效果 | 布局转换 |
|------|------|---------|
| `TCVT(Left, Vec)` | Vec→Left 装箱 | NORM → NZ (512B box) |
| `TCVT(Vec, Left)` | Left→Vec 拆箱 | NZ → NORM |

---

## 三、ACCCVT 与 TCVT 的区别

`ACCCVT` 和 `TCVT` 是两个完全不同的硬件路径，不是同一指令的变体：

| | ACCCVT | TCVT |
|---|---|---|
| **输入格式** | `TileAcc`（矩阵乘累加器格式） | `Tile<Vec>` 或 `TileLeft` |
| **输出格式** | `Tile<Vec>`（必须是 Vec） | 任意 tile 类型 |
| **硬件路径** | CUBE 引擎（`BSTART.CUBE ACCCVT`） | TEPL 引擎 |
| **类型转换** | 可选（Acc 内部是 float，输出可转 dtype） | 必含（src 和 dst 的 DType 可以不同） |
| **布局转换** | Acc/NZ → Vec ColMajor | Vec ↔ Left（任意组合） |
| **典型用途** | `TMATMUL` 后把结果从 Acc 取出来 | float→dtype 转换、Vec→Left 装箱 |

### 为什么是两个不同指令？

Acc tile 的物理布局和 Vec tile 完全不同：Acc 是矩阵乘引擎内部的专有格式（类似 NZ 分形），常规的逐元素运算（EXP、ADD、MUL）不认识 Acc 格式。所以矩阵乘的输出**必须**先经过 `ACCCVT` 转换成 Vec 后才能做后续处理。

而 `TCVT` 走的是 TEPL（细粒度逐元素处理单元），只操作 Vec 或 Left 等常规 tile，负责类型转换和装箱/拆箱。两者硬件数据通路不同，没法合并。

---

## 四、FA 算子中的典型数据流

```
Global Q (RowMajor, [kTm,qD])
  ↓ TLOAD (TMOV_ND2NZ)
TileLeft Q (Nz, 512B box)                    ← L0A

Global K (RowMajor, [kTk,qD])
  ↓ TLOAD (TMOV_ND2ZN)
TileRight K (Zn, 512B box)                   ← L0B

  ↓ TMATMUL (CUBE)
TileAcc W_out (Nz, 1024B box, float)         ← ACC

  ↓ ACCCVT (CUBE)
Tile<Vec> W (ColMajor, float, unboxed)       ← Vec (可逐元素操作)

  [online softmax: EXP/ADD/MAX... (TEPL)]

  ↓ TCVT (TMA: Vec→Left 装箱)
TileLeft W_left (Nz, 512B box)               ← L0A

Global V (RowMajor, [kTk,vD])
  ↓ TLOAD
TileRight V (Zn, 512B box)                   ← L0B

  ↓ TMATMUL (CUBE)
TileAcc PV_out (Nz, 1024B box)               ← ACC

  ↓ ACCCVT (CUBE)
Tile<Vec> O (ColMajor, float, unboxed)       ← Vec

  ↓ = O / l_final (TEPL)
  ↓ TCVT (float→dtype)
Tile<Vec> O_cast (ColMajor, dtype)

  ↓ TSTORE (NZ→ND)
Global O (RowMajor, [kTm,vD])
```

---



## 六、Tile 类型三层的明细（从 pto_tile.hpp 定义）

```
TileLeft  = Tile<Location::Left,  ..., BLayout::ColMajor, SLayout::RowMajor, 512>
TileRight = Tile<Location::Right, ..., BLayout::RowMajor, SLayout::ColMajor, 512>
TileAcc   = Tile<Location::Acc,   ..., BLayout::ColMajor, SLayout::RowMajor, 1024>
```

| | BLayout (外层) | SLayout (内层) | 箱子大小 | 内层形状 (fp16) |
|---|---|---|---|---|
| **TileLeft** | ColMajor | RowMajor | 512B | 16×16 |
| **TileRight** | RowMajor | ColMajor | 512B | 16×16 |
| **TileAcc** | ColMajor | RowMajor | 1024B | 16×16 |
| **Vec (unboxed)** | Row/ColMajor | NoneBox | — | 1×1 |

### 分形装箱的含义

以 Left（512B 箱，SLayout=RowMajor）为例：tile 首先被切分成 512 字节的物理块（box），每个 box 内部又按 RowMajor 切分成 16×16 的子块。在这个小 box 内部元素是连续存放的，但 box 之间的遍历顺序由 BLayout（ColMajor）决定。

这跟"裸的"行主序/列主序不是一回事——它是**两层嵌套的分形排列**，类似矩阵乘硬件中的 NC/NR 分形格式，但 PTO 用自己的 SLayout/BLayout 命名体系。

### 为什么需要 ACCCVT？

Acc tile 的装箱尺寸是 1024B（vs Left/Right 的 512B），且物理布局完全不同。逐元素处理单元（TEPL）只认 Vec（无装箱）格式，所以 `ACCCVT` 负责把 Acc 的 1024B 分形布局解出来，重组为平坦的 Vec ColMajor。这也是为什么 `ACCCVT` 走 CUBE 通路——解包需要矩阵乘引擎的数据交换能力。

---

## 七、TMATMUL 与 TMATMUL_ACC 的区别

`TMATMUL` 和 `TMATMUL_ACC` 的区别只有一个：**ACC 是否参与输入**。

| | TMATMUL(acc, A, B) | TMATMUL_ACC(acc, A, B) |
|---|---|---|
| **语义** | `acc = A × B`（覆盖） | `acc += A × B`（累加） |
| **ACC 角色** | 只写 | 读写（既是输入也是输出） |
| **硬件指令** | `blk_matmul`（CUBE） | `blk_matmul_ac`（CUBE） |
| **使用场景** | 链式乘法的第一跳 | 后续跳，把新结果叠加上去 |

FA 代码中的典型用法：

```cpp
// P = p₀×V₀ + p₁×V₁ + p₂×V₂  (Ydim=3)
for (int y = 0; y < Ydim; y++) {
    if (y == 0)
        TMATMUL(tPV_out, p[0], V[0]);       // acc = p₀×V₀ (清空起算)
    else
        TMATMUL_ACC(tPV_out, p[y], V[y]);   // acc += p_y×V_y (累加)
}
```

**代价**：`TMATMUL_ACC` 使 ACC tile 的寄存器活跃区间从"一次乘法"拉长到"整个累加链"——因为 ACC 既是输入又是输出，必须从第一跳一直存活到最后一跳，增加了 tile 寄存器压力。这就是 SFA 两遍设计要避开它的原因：在 Pass 2 中改用 fresh `TMATMUL` + 独立 `TADD` 来替代 `TMATMUL_ACC`，降低每个稀疏循环内的峰值 tile 活跃数。

---

## 八、2D Unroll 的同步与内存划分

### 同步：不需要同步

代码虽然是嵌套循环，但本质是**单线程的 tile 指令序列**，不是多线程并行。执行顺序就是 C++ 代码写的顺序：

```
1. for y { TLOAD K[y] }           ← 加载所有 K tile
2. for x { for y { QK^T } }       ← 算完所有 Xdim×Ydim 个 score
3. for x { softmax for x }        ← 逐行 softmax
```

加上 `#pragma clang loop unroll(full)` 完全展开后，编译器看到的是**扁平的指令序列**，不存在多线程间的同步问题。硬件串行执行，前一步完了才走下一步。

### 内存划分：通过限制 Ydim 控制 score tile 总数

不是 GPU 风格"算一个扔一个，按需 reload"。这里采用的方式：**一次性算出所有 score，全部放在 tile 寄存器里**：

```cpp
tileW tW[Xdim][Ydim];               // ← Xdim×Ydim 个 tile 寄存器，全部同时存活

for (x) for (y)
    TMATMUL(..., tQ[x], tK[y]);     // 填满全部 tW[x][y]

for (x)                              // 然后逐个 x 做 softmax
    for (y)                          // 每个 x 需要 tW[x][0..Ydim-1] 全在
        softmax on tW[x][y];
```

同一个 Kb 迭代内，matmul 产出的 score 必须一个不丢地全留着，因为 softmax 的 Ydim 路归约树需要同时比较 Ydim 个 score：

```
m_new = max(m_old, rowmax(S₀), rowmax(S₁), ..., rowmax(S_Y₋₁))
```

如果 score 算一个丢一个，归约树没法跑。这和 GPU 上用 shared memory + register file 按需 spill/reload 的模式完全不同——**tile 寄存器不支持 spill**，要么全塞进去，要么编译器崩溃。

### 为什么 Ydim 只到 4？

本质上是 tile 寄存器文件的容量上限。以 Xdim=2, Ydim=4, kTm=16, kTk=32, float 为例，一个 Kb 迭代内同时存活的 tile：

| 变量 | 数量 | 单个大小 | 小计 |
|------|------|---------|------|
| `tQ` (TileLeft) | Xdim=2 | 512B+ | ~1KB |
| `tK` (TileRight) | Ydim=4 | 512B+ | ~2KB |
| `tW` (Vec, float, 16×32) | Xdim×Ydim=8 | 2048B | **16KB** |
| max/sum/O/PV/scale 状态 | 5×Xdim=10 | 各几百B | ~2KB |
| `tExpW` | Xdim×Ydim=8 | 按 dtype | ~2-16KB |

纯 score tile 就占了 16KB，加上其他 tile 后峰值轻松超过 30KB。Ydim 每翻一倍，`tW` 数量翻倍——直接用容量天花板硬截断。

---

## 九、fa_2d_unroll_pto.hpp vs sfa_pto.hpp 编码差异

除了业务逻辑（稠密 vs 块稀疏），在编码层面有五个关键差异：

### 一、Pass 数量：单遍 vs 两遍

| | 稠密 | 稀疏 |
|---|---|---|
| **遍数** | 1 | 2 |
| **QK^T 计算次数** | 每 K block 1 次 | 每 K block 2 次（一遍归约 m/l，一遍算 P×V） |
| **O 更新方式** | `tO = tO × scale + tPV`（在线 rescale） | `tO += tPV`（Pass 2 中 p 已用最终 m,l 归一化） |
| **消耗** | 无重复计算 | 重复算了一遍 QK^T |

### 二、Unroll 因素：有 vs 无

| | 稠密 | 稀疏 |
|---|---|---|
| **Xdim / Ydim** | 有（编译期常量 1/2/4，二叉树硬编码归约） | 无（每个 Q block 一次处理一个） |
| **展开方式** | `tQ[Xdim]`, `tK[Ydim]`, `tW[Xdim][Ydim]` 数组存所有 tile | 单变量 `tQ`, `tK`, `tW` |
| **softmax 归约** | `TROWMAX/TROWSUM` 多路并行归约 | 逐个 K block 在线归约，不需多路 tree reduction |

因为 CSR 稀疏各 Q block 的活跃 K 数量不固定，无法做编译期固定展开。

### 三、规避的 bug：两种不同 bug，两类不同对策

这里有两类工具链 bug，分别影响两个文件：

**Bug A：tile 寄存器溢出 → 编译器崩溃**

> "linx 后端的 `loadRegFromStackSlot` 对 tile 类寄存器命中 `UNREACHABLE`"

| | 稠密 | 稀疏 |
|---|---|---|
| **是否触发** | Ydim≤4 固定，编译期可调度，不超上限 | 稀疏循环长度不固定，一旦过长就触发 |
| **对策** | — | **两遍设计**：Pass 2 用 `TMATMUL` + `TADD` 替代 `TMATMUL_ACC`，拆短 ACC 寄存器活跃区间 |

**Bug B：广播算子的 dtype static_assert**

> "工具链 TROWEXPANDSUB/MUL、TCOLEXPANDSUB/MUL 错误比较 `src0::DType`(标量) 与 `dst::TileDType`(打包向量)，触发编译失败"

| | 稠密 | 稀疏 |
|---|---|---|
| **受影响的指令** | TROWEXPANDSUB、TROWEXPANDMUL（行广播） | TCOLEXPANDSUB、TCOLEXPANDMUL（列广播） |
| **是否包含 `template_asm.h`** | **不包含** | **包含** |
| **对策** | 直接用 pto 命名空间的 `TROWEXPANDSUB/MUL`（该 bug 在稠密场景下不触发，或编译 flag 不同） | 用 `*_TEPL` 内联汇编版本（`TCOLEXPANDSUB_TEPL`、`TCOLEXPANDMUL_TEPL`）绕过 |

### 四、softmax 归约方向：行归约 vs 列归约

因为 Bug B 的 `_TEPL` 绕过方案只提供了列广播版本（`TCOLEXPANDSUB_TEPL`、`TCOLEXPANDMUL_TEPL`），没有行广播版本的 `_TEPL`，所以 sfa_pto.hpp 被迫从行归约切换到列归约：

| | 稠密 | 稀疏 |
|---|---|---|
| **max 提取** | `TROWMAX`（逐行取 max） | `TCOLMAX`（逐列取 max） |
| **sum 归约** | `TROWSUM` | `TCOLSUM` |
| **广播减** | `TROWEXPANDSUB` | `TCOLEXPANDSUB_TEPL` |
| **广播乘** | `TROWEXPANDMUL` | `TCOLEXPANDMUL_TEPL` |

### 五、循环结构

| | 稠密 | 稀疏 |
|---|---|---|
| **内层循环** | `for (j = 0; j < Kb; j += Ydim)` — 遍历所有 K block | `for (p = kv_begin; p < kv_end; p++)` — 只遍历 CSR 活跃 K block |
| **K block 编号** | `j+y`（连续递增） | `kv_idx_ptr[p]`（通过 CSR 间接索引） |
| **内层变量声明** | 内层声明 `tileK tK[Ydim]`，作用域天然短 | Q tile 在外层作用域（因为 CSR 稀疏索引是外层变量），Pass 2 还需要重新加载 Q 来缩短活跃区间 |

### 总结：引入稀疏后对编码的真正冲击

并不是"多加两个 CSR 指针"那么简单，而是编码模型从 **"编译期固定尺寸、全展开"** 退化到了 **"运行期动态循环、按需规避工具链 bug"**。具体来说：

1. **编译期 → 运行期**：Xdim/Ydim 展开消失，编译器不再能精确分配寄存器，必须用两遍设计主动降低寄存器压力
2. **行 → 列**：因为 `_TEPL` 绕过方案只有列广播版本，softmax 被迫换方向
3. **累加 → 独立加**：`TMATMUL_ACC` 被 `TMATMUL + TADD` 替代，拆短 live range
4. **额外 include**：引入 `template_asm.h`（inline asm 绕过 bug）
5. **Q 重载**：Pass 2 重新加载 Q（`tQ2`），确保 Pass 1 的 Q 不在 Pass 2 中存活

本质上，SFA 的编码是从"利用编译器能力"变成了"与编译器能力周旋"。

---

## 十、sfa_pto.hpp 入参 kv_idx_ptr 和 kv_off_ptr 的含义

来自 `sfa_pto.hpp:14-16` 的注释：

```
CSR 风格的块级稀疏：K/V 按 kTk 行切成 Kb 块，Q 按 kTm 行切成 Qb 块。对第 i 个
Q 块仅遍历其活跃 K/V 块（由 kv_idx_ptr / kv_off_ptr 指定）：
  for p in [kv_off_ptr[i], kv_off_ptr[i+1]):  j = kv_idx_ptr[p]   (0<=j<Kb)
```

- **`kv_idx_ptr`** — 活跃 K/V block 编号列表（values），`const int*`，长度为总活跃数
- **`kv_off_ptr`** — 每个 Q block 在 `kv_idx_ptr` 中的起止偏移（offsets），`const int*`，长度为 `Qb+1`

### 实例

假设 Q 切成 4 个 block，K/V 切成 8 个 block，稀疏模式如下：

```
Q block 0 → 关注 K/V block {1, 3}
Q block 1 → 关注 K/V block {}   (空)
Q block 2 → 关注 K/V block {5}
Q block 3 → 关注 K/V block {2, 6, 7}
```

则两个指针指向的数组为：

```
kv_off_ptr = [0, 2, 2, 3, 6]     // 长度 Qb+1 = 5
kv_idx_ptr = [1, 3,  5,  2, 6, 7]
             └─Q0─┘   └Q2┘ └─Q3──┘
```

代码中 `sfa_pto.hpp:94-96` 引用：

```cpp
for (int i = 0; i < Qb; ++i) {
    const int kv_begin = kv_off_ptr[i];     // Q block i 的起始
    const int kv_end   = kv_off_ptr[i + 1]; // Q block i 的结束
    for (int p = kv_begin; p < kv_end; ++p) {
        const int j = kv_idx_ptr[p];        // 具体的 K/V block 编号
        ...
    }
}
```

对 Q block 0：`kv_off_ptr[0]=0, kv_off_ptr[1]=2`，遍历 `kv_idx_ptr[0]=1` 和 `kv_idx_ptr[1]=3`——即只关注 K/V block 1 和 3。
