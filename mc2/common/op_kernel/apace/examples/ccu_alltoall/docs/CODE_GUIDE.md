# CCU AllToAll 代码导读手册

> 目的：**降低阅读成本**——先读本文建立全局地图，再按需下钻源码。
> 覆盖：全部 4 个源文件的接口与函数讲解（1864 行源码 → 本文约 20 分钟读完）。
> 配套：参数/资源疑问全沉淀于 §2.1.1（十一槽 FAQ）与 §7（资源四层模型）——新人同类问题先查这两节。
> 前提：已了解总体设计（`DESIGN_ANALYSIS.md` §0-§4：CCU 直调双流架构 + 参数桥机制）。

---

## 0. 一图看懂数据流（先记这张图）

```
                    HOST（main.cpp）
                      │ GenerateInputData: sendBuf[j块][t] = E(src=my, dst=j, t)
                      │ InvokeAllToAll ──► AllToAllKernel（.asc）
                      ▼
        ┌──────────── streamAicore ────────────┐   ┌──────────── streamCcu ────────────┐
        │  alltoallAicoreKernel（纯AIC×8核）  │   │  CcuAllToAllMesh1DKernel（CCU核）  │
        │  block0: 写11槽参数区→写wake(ev[0])  │◄──┤  等wake(ev[0]) → 循环 commTurn 轮:  │
        │  all核: 轮询 ev[r+1]==1（round done）│   │   LoadArgs→PreSync→DoAlltoAll→     │
        └──────────────────────────────────────┘   │   PostSync→EventRecord(ev[r+1])    │
                                                     └───────────────────────────────────┘
        两流通过 GM 上共享的 Variable/Event 地址同步（无任何 host 介入）

  DoAlltoAll 换块语义（ExecuteAllToAllTransfer）:
    远端: for j≠my: ccu::Write(ch[j], dst=远端recv+my*block+r*sub, src=send+j*stride+r*sub, sub字节)
    本地: GroupCopy(recv+my*block+r*sub ← send+my*stride+r*sub)   ← j==my 的块
    屏障: EventRecord(my位) → EventWait(全rank位)
```

**一句话**：AIC block0 把 host 算好的参数写入 GM 共享内存参数区 → 唤醒 CCU → CCU 每轮读参数、
逐对换块、报告 done → AIC 轮询 done（阶段3 在此接 matmul 即成通算流水）。

---

## 1. 文件地图与依赖方向

```
main.cpp ──调用──► AllToAllKernel（.asc 尾部）
                              │ include（顺序敏感！单翻译单元）
                              ▼
alltoall_ccu_kernel_impl.asc = [宏/常量/BridgeSlot枚举/CcuMappedAddrs/A2aGoSize]
                          → alltoall_aicore_kernel_impl.h（用 BridgeSlot/CcuMappedAddrs/A2aGoSize）
                          → [namespace ops_comm_a2a: CCU kernel 全部实现]
                          → alltoallAicoreKernel（__global__）
                          → alltoall_host_impl.h（用 .asc 前文所有符号）
                          → main.cpp（最后 include，含 main）
```

| 文件 | 行数 | 角色 | 核心内容 |
|---|---|---|---|
| `src/alltoall_ccu_kernel_impl.asc` | 957 | **主文件**（唯一编译单元） | 公共定义 + CCU kernel + AIC kernel + launch 入口 |
| `src/alltoall_aicore_kernel_impl.h` | 146 | AIC 侧 device 组件 | `WriteBridgeSlot` + `A2aAicoreProcess` |
| `src/alltoall_host_impl.h` | 316 | host 编排 | `AllToAllContext` + 6 个编排函数 |
| `src/main.cpp` | 443 | 样例入口 | 数据构造/校验/多 rank 线程/main |

---

## 2. 核心数据结构（参数体系 4 个 + host 工程细节 1 个）

### 2.1 `BridgeSlot` 枚举（.asc L71）— 共享内存参数区槽位唯一事实源 ★

> **术语说明**：本工程（沿自 AG 示例）所称"参数桥"，标准说法是**共享内存参数区**
> （kernel 间 mailbox 通信模式）：AIC 与 CCU 两个 kernel 通过 GM 上一块约定好的
> Variable 区域传递参数——一方写入、另一方轮询/读取。代码符号（BridgeSlot 等）保留
> Bridge 命名以维持仓库内自洽，文档表述统一用"共享内存参数区"。
```cpp
enum BridgeSlot : uint32_t {
    SLOT_INPUT=0,       // sendBuf 地址          ┐
    SLOT_OUTPUT=1,      // recvBuf 地址          │ 前7槽：AIC写→CCU每轮读
    SLOT_TOKEN=2,       // CCU内存访问令牌       │
    SLOT_SRC_OFFSET=3,  // r*sub（源偏移）        │
    SLOT_DST_OFFSET=4,  // my*block+r*sub（宿偏移，host预乘myRank项）│
    SLOT_SLICE_SIZE=5,  // sub（本轮子段字节）    │
    SLOT_SRC_STRIDE=6,  // blockSize（块间距）    ┘
    SLOT_GO_ADDR_OFFSET=7 ... SLOT_GO_TAIL_SIZE=10,  // goSize四元组（本地拷贝loop参数）
    BRIDGE_SLOT_NUM=11
};
```
**为什么重要**：参数桥是 AIC→CCU 的唯一传参通道。槽位语义只在此定义一次，
AIC 写侧（`WriteBridgeSlot`）与 CCU 读侧（`LoadArgs`）都按枚举索引——改槽位只动这里+两个调用点。

### 2.1.1 11 个槽位参数：使用方法与常见疑问（FAQ 整理）

> 本节源自实际代码走读中的一连串追问（2025-08-29），按"是什么→怎么用→为什么这么设计"组织。
> 大多数问题都指向同一个根源：**CCU 指令集只有加法**（见 2.1.1 末尾的指令集约束）。

#### 一、参数总表（11 槽 = 三组）

| 组 | 槽# | 枚举 | 参数含义 | 值（host 算） | 变化频率 | CCU 侧用法 |
|---|---|---|---|---|---|---|
| **数据面地址** | 0 | SLOT_INPUT | sendBuf GM 地址 | aclrtMalloc 分配 | 恒定 | `src.addr`/`localSrc.addr` 基址 |
| | 1 | SLOT_OUTPUT | recvBuf GM 地址 | aclrtMalloc 分配 | 恒定 | `localDst.addr` 基址；**一参两用**：PreSync 时其 Variable 值经 channel 广播给远端（对端据此算落点） |
| | 2 | SLOT_TOKEN | CCU 内存访问令牌 | HcommCcuGetMemToken | 恒定 | `src.token`（CCU 访问 GM 必携）；同样经 PreSync 广播 |
| **通信几何** | 3 | SLOT_SRC_OFFSET | round 级源偏移：块内从哪开始拿 | `r*sub` | **per-round** | `src.addr += srcOffset`（块内定位） |
| | 4 | SLOT_DST_OFFSET | round 级目的偏移：落到对方哪 | `myRank*blockSize + r*sub` | **per-round** | `dst[j].addr += dstOffset`（所有 j 共用同一 offset） |
| | 5 | SLOT_SLICE_SIZE | **每次实际搬运的字节数** | `blockSize/commTurn` | **per-round** | `ccu::Write` 长度 + 空块保护（`CCU_IF(sliceSize != 0)`）+ goSize 输入 |
| | 6 | SLOT_SRC_STRIDE | 块间距：下一块在哪 | `blockSize` | 恒定 | 游标递推 `src.addr += srcStride`（每循环一步，跨到下一块） |
| **本地拷贝计划** | 7 | SLOT_GO_ADDR_OFFSET | 完整 256KB 块起始偏移 | CalGoSize 输出 | 恒定* | GroupCopy 主路径触发条件（非零→走 8-loop 并行） |
| | 8 | SLOT_GO_LOOP_ITER | 完整块迭代数 | CalGoSize 输出 | 恒定* | LoopParam 位域编码 |
| | 9 | SLOT_GO_LOOP_EXTEND | 尾部调度位编码 | CalGoSize 输出 | 恒定* | GroupCopy 尾部分支触发（非零→追加 2 Loop） |
| | 10 | SLOT_GO_TAIL_SIZE | 尾部剩余字节 | CalGoSize 输出 | 恒定* | 尾部 Loop0 搬运长度 |

*槽 7-10 内容跨轮相同（每轮 slice 相同→施工图相同），物理上仍写 commTurn 份。

#### 二、三个"大小"的关系（最容易混）

```
sendBuf 总量 ─÷R→ blockSize（块大小，= SRC_STRIDE 的值，host 内部量不进参数区）
              blockSize ─÷commTurn→ SLICE_SIZE（每轮实际搬运量）
                                        ├→ 远端 Write 的长度
                                        └→ CalGoSize 输入 → goSize（本地搬运施工图）
```

| 名字 | 回答的问题 | 值 |
|---|---|---|
| blockSize | 块多大？ | sendBuf/R |
| SLICE_SIZE | 每轮搬多少？ | blockSize/commTurn |
| goSize | 本地这多字节怎么分趟搬？ | CalGoSize(SLICE_SIZE) |

#### 三、OFFSET / STRIDE / SLICE 怎么配合（数值实例）

设 2 卡、sendBuf 256KB（2 块 × 128KB）、commTurn=2 → **stride=128K，slice=64K，offset 逐轮 0→64K**：

```
sendBuf:  |---64K---||---64K---||---64K---||---64K---|
          | 块0前半 | 块0后半 |  块1前半 |  块1后半  |
          | (r0搬)  | (r1搬)  |  (r0搬) |  (r1搬)   |
                       ↑                      ↑
                 round1 的 src 起点      游标 +=stride(128K) 后到达
                 （OFFSET 管"块内位置"）  （STRIDE 管"块间跳跃"）
```

**round 1 循环（myRank=0）**：`src = INPUT + 64K`（指着块0后半）→ j=0 跳过但 `src += 128K` → src 指着**块1后半** → 把 64K（slice）发给 rank1。
完整地址公式：`src(j, r) = sendBuf + j*stride + r*slice`（j=块号，r=轮号）。

**为什么 stride ≠ slice 不是 bug**：stride 度量"块起点到块起点"，slice 度量"块内取多少"——commTurn>1 时块里有一部分本轮不搬，两者必然不同。commTurn=1 时 offset=0、stride=slice=blockSize，三者退化为相等。
**当前实现边界**：stride 焊死为 blockSize（紧邻切分布局）。高阶 API 的 strideCount 与 dataCount 本可独立（跨步/有洞布局）；阶段3 矩阵 tile 消费若需跨步布局，需把 stride 独立参数化（槽位已预留，改 host 赋值即可）。

#### 四、为什么 commTurn=N 需要 N 组参数（不能 1 组）？

**表面原因**：offset 逐轮变（0→64K→128K...），1 组参数每轮读到的值相同，round1 会重复搬 round0 的数据。

**根本原因：CCU 指令集只有加法**（`ccu_operator_v1.h`：`enum class CcuArithmeticOperatorType { ADDITION, INVALID }`）：
- 循环变量 r 是 host 侧 uint32，sub 是 Variable——**无 `r*sub` 重载**（Variable 只与 Variable/Address 相加，无 operator*）
- 累加也不行：offset 是从参数区读的只读快照，CCU 无"自增写回参数区"语义
- 所以**等差数列必须由会乘法的 AIC/host 预先展开成 N 份**，CCU 每轮查表执行

**内容规律**：N 组里 8 个槽跨轮不变（input/output/token/stride/goSize×4），只有 3 个槽等差变化（srcOffset/dstOffset/sliceSize）——参数区本质是"**展开后的等差数列表**"。代价极小（AIC 一次性写 N×11 槽，微秒级），换来 CCU 循环体零计算。

**AG（AllGather 基线）同款机制且更 per-round**：AG 每轮搬 sendBuf 的不同段（round r 搬第 r 个 tileM 行广播），inputOffset/outputOffset 逐轮等差——连 stride 概念都不需要（无需游标，每轮直接新 offset）。AllToAll 每轮要重新扫所有块（每块取第 r 子段），才引入 stride+游标递推。**SLOT_SRC_STRIDE 是 AllToAll 相对 AG 参数桥的唯一增量槽**。

#### 五、指令集约束速查（"CCU 无乘法"的精确定义）

来自 `pkg_inc/hcomm/ccu/ccu_datatype_v1.h`（CANN 9.1.0-beta.3 实测）：

| 类型 | 支持的运算 | 不支持 |
|---|---|---|
| Address | `= uint64_t`（立即数加载）、`= Variable`、`+ Variable`、`+= Variable` | `*`、`-`、`+= 标量` |
| Variable | 同上 + `== uint64_t`、`!= uint64_t` | `*`、`-` |
| LocalAddr | 无运算符（纯壳 `{Address addr; Variable token;}`）——代码里 `src.addr += stride` 的 `+=` 打在成员上 | 一切 |

推论：
- `src.addr += 64;` 编译不过（+= 只收 Variable）→ **一切参与地址运算的量必须先经参数区变成 Variable**
- `localSrc.addr += srcStride` 循环 myRank 次的本质 = 用 k 次 ADDITION 硬算 k×stride
- host 标量只能"立即数赋值"进场（不参与运算的量，如 totalMask）
- 乘积项的处理分工：**host 展开烧入**（myRank×blockSize → dstOffset）vs **CCU 循环展开**（j×stride → 游标递推）

#### 六、参数→搬运逻辑全景图

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        11 个参数 → CCU 搬运逻辑（示例：2卡、每块128K、commTurn=1）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
槽0/2/3/6 ──► ① 算源起点   src = INPUT + SRC_OFFSET
                   │
槽1/4/5/6 ──► ② 逐块发送   for j: Write(通道j, 对方recv+DST_OFFSET, src, SLICE_SIZE)
                   │                       src += SRC_STRIDE  ← 游标步进
槽1/4/7-10 ──► ③ 本地搬运   GroupCopy(我的recv+DST_OFFSET,
                   │                INPUT+OFFSET+我的编号×STRIDE, goSize施工图)
（Event） ──► ④ 全员确认   EventRecord(我的bit) → EventWait(全员bit)

隐藏机关（无显式路由表）：
  块号==rank号：游标走过第 j 块 → 天然发给 rank j
  落点==我的编号：DST_OFFSET 含 我的编号×块大小 → 我的数据进对方"我的编号"格
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 2.1.2 `A2aGoSize`（.asc L101）— 槽 7-10 四件套的 host 侧载体

> 本 struct 与 §2.1.1 参数总表的"本地拷贝计划"组（槽 7-10）是**同一份数据的两副面孔**：
> host/AIC 侧它是个有字段名的 struct（本节），写进参数区后 CCU 读出来是 4 个独立 Variable
> （`ctx.goSize.addrOffset` 等，GroupCopy 逐个消费）。不是新增参数，是四件套的打包形态。

```cpp
uint64_t addrOffset;    // 完整loop块起始偏移（loopSize=256KB 的整数倍部分）  → SLOT_GO_ADDR_OFFSET
uint64_t loopIterNum;   // 完整loop块迭代数                                   → SLOT_GO_LOOP_ITER
uint64_t loopExtendNum; // 尾部扩展loop参数（位编码）                          → SLOT_GO_LOOP_EXTEND
uint64_t tailSize;      // 尾部剩余字节                                       → SLOT_GO_TAIL_SIZE
```
**为什么重要**：`CalGoSize(subBytes)` 的输出，本质是"把 sub 字节拆成整数个 256KB 大块 +
尾部小块"的搬运计划，CCU 的 `GroupCopy` 据此驱动 LoopGroup 硬件并行搬运。
**为什么单独成 struct**：host→AIC 按 `<<<>>>` 传参需要具名结构（AG 的 MatmulTilingData 先例），
免 4 个散参数穿透 kernel 签名；CCU 侧无 struct 概念，拆成 4 个 Variable 各自进槽。

### 2.2 `CcuMappedAddrs`（.asc L87）— Variable/Event 两类资源的 GM 地址袋

> **注意：startVa 与 eventVa 是两类完全不同的资源**，拼在一个结构里只为一次 kernel 传参方便
> （host 侧两笔独立申请、两个 Handle、两个配额池；CCU 侧两条消费路径）。区别见下表——
> "Variable 传参数、Event 打信号旗"，别混。

```cpp
uint64_t startVa[MAX_COMMTURN][11];  // [r][slot]：第r轮第slot个 Variable 的 GM 地址
uint64_t eventVa[MAX_COMMTURN + 1];  // [0]=wake, [r+1]=round r done（Event 的 GM 落点）
```

| | **startVa 名下：Variable**（N×11 个） | **eventVa 名下：Event**（N+1 个） |
|---|---|---|
| 本质 | 可读写数值单元——**参数信箱** | 位掩码完成信号——**信号旗** |
| 申请 | `HcommCcuVariableAlloc`（一个 varHandle 批量管 N×11 个） | `HcommCcuEventAlloc`（一个 eventHandle 批量管 N+1 个） |
| 配额池 | VARIABLE=400/DIE | EVENT=48/DIE |
| 谁写谁读 | **AIC 写**（WriteBridgeSlot）→ CCU 读（LoadArgs 查表） | **双向**：wake=AIC 写→CCU 等；round done=CCU 写（EventRecord）→AIC 轮询 |
| 装什么 | 参数数值（地址/偏移/大小，uint64） | 完成状态位（16bit 掩码；本工程单 rank 置位故 AIC 查 ==1） |
| 角色 | "干活的说明书"（告诉 CCU 怎么搬） | "干完没的信号灯"（报告进度） |

**eventVa 的编号错位**：`[0]=wake` 被 AIC→CCU 的开工令占了，round r 的 done 住 `[r+1]`——
同一块 Event 池两个方向混用，靠编号分区，故数组总长 commTurn+1。

**与第三类资源 CKE 的界线**（易混，完整三条线路）：
- **Variable**：传参数，AIC→CCU 单向，GM 信箱
- **Event**：跨流打旗（AIC⇄CCU 同卡两流），GM 落点，AIC 靠 ReadGmByPassDCache 自旋轮询
- **CKE**：跨卡打旗（CCU⇄CCU），**通道硬件寄存器，不在 GM 上**（随通道 desc.notifyNum 隐式
  配置，无显式 Alloc）——AIC 指令集碰不到它
> ⚠️ 常见误解："CKE 是 AIC 和 CCU 互写状态的资源"——错。AIC↔CCU 互通走 GM（Event/Variable），
> AIC 的 API 里没有任何 Notify 系调用；CKE 只出现在 CCU kernel 的跨卡 NotifyRecord/Wait 里。

**为什么重要**：CCU 的 Variable/Event 资源在 GM 中各有一个实际地址；host 分配时取出地址塞进这个
结构，经 kernel 参数传给 AIC——AIC 才知道往哪写参数、轮询哪个地址。**它是双流同步的物理基础**。
（补充：CCU kernel 本身不消费此结构——它经 kernelArg 的 varHandle/eventHandle 以 Variable/Event
抽象访问同名资源；本结构只服务 host→AIC 传地址。结构体是本工程自定义的普通 C++ 结构，
布局可自由设计，CCU 强制的仅是"地址值 + 编号对应"这层物理契约。）

### 2.3 `CcuKernelArgAllToAllMesh1D`（.asc L225）— CCU kernel 参数块
```cpp
ChannelHandle channels[16]; // 到每个远端rank的点到点通道（CCU_MAX_RANK_SIZE=16）
uint32_t channelCount;      // = rankSize-1
uint64_t rankSize; uint32_t rankId;
CcuVariableHandle varHandle; uint32_t varNum;    // 参数桥Variable资源（commTurn×11）
CcuEventHandle eventHandle; uint32_t eventNum;   // Event资源（commTurn+1）
uint32_t commTurn;          // CCU kernel 内部循环次数
```
**为什么重要**：`<<<schd, insHandle, streamCcu>>>` launch 时传给 `__ccu_host__` kernel 的
唯一实参；CCU 侧一切资源（通道/参数桥/事件）都从这里拿。命名对齐 CANN `CcuKernelArgAlltoAllMesh1D`。

### 2.4 `AllToAllContext`（alltoall_host_impl.h L37）— host 侧编排状态（工程细节，非 kernel 机制）

> **定位澄清（2025-08-29 评审）**：纯 host 侧结构，CCU/AIC kernel **完全不感知**。它不是数据
> 构造（那是 sample 的 GenerateInputData），而是**通信资源生命周期的管理载体**：
> `PrepareAllToAll` 填它 → `AllToAllKernel` 取它 launch → `ReleaseAllToAllContext` 按它释放。
> 真实算子场景它依然存在（调用方从 sample 换成框架），只是工程实现细节——关心 kernel 间
> 通信机制的读者可跳过本节。
```cpp
schd/insHandle/kernelArg/mappedAddrs   // 交给两个kernel launch 的东西
srcOffsets/dstOffsets/sliceSizes[16]    // per-round 偏移（host算好，拷到device给AIC读）
d_srcOffsetsAddr 等 3 个                // device侧地址
token/inputSize/blockSize/srcStride     // 通信参数
rankId/rankSize/commTurn
goSize                                  // A2aGoSize
```

---

## 3. 函数逐个讲解（按执行顺序）

### 3.1 入口链（host 侧，main.cpp）

| 函数 | 行 | 讲解 |
|---|---|---|
| `main` | L340 | 解析命令行（`precision/perf [elemsPerBlock] [commTurn] [loop]`，非法参数→usage+非0退出）→ aclInit → 每设备一个线程跑 `Sample` → join 后聚合各线程 `ret`：**任一 rank FAIL → 进程退出码非 0**（run.sh set -e 依赖） |
| `Sample` | L238 | 单 rank 全流程：SetDevice → HcclCommInitRootInfo → 建双流（streamAicore/streamCcu）→ 分配等量 send/recvBuf → `GenerateInputData` → 循环 `InvokeAllToAll`（precision 1 次）→ `VerifyAndDumpRecv` → 释放 |
| `GenerateInputData` | L105 | **输入数据构造（测试向量生成器）**：host 临时缓冲上双重循环，`sendBuf[j][t] = A2aElemPattern(myRank, j, t)`——每块每元素都不同，memcpy 到 device |
| `A2aElemPattern` | L75 | 溯源编码：`srcRank<<12 \| dstBlock<<8 \| (t^(t>>8))&0xFF`。位型不解释为 FP16 数值（纯字节搬运验证）；xor 混合使块内偏移 δ≡0 mod 256 类错位也可检出 |
| `InvokeAllToAll` | L130 | **调用一次完整 AllToAll**：`AllToAllKernel`（launch 两流）→ 同步双流 → `ReleaseAllToAllContext`（一次完整通信生命周期） |
| `VerifyAndDumpRecv` | L153 | **校验**：recvBuf 读回 host → 双重循环逐元素比对 `A2aElemPattern(r, myRank, t)`（AllToAll 语义 `recv_i[r]=send_r[i]`）→ PASS/FAIL + 首错定位（位型反解 srcRank/dstBlock）→ 落地 `output/<rank>/npu_out.bin` → 返回 0/1 |

### 3.2 launch 入口（.asc 尾部）

| 函数 | 行 | 讲解 |
|---|---|---|
| `AllToAllKernel` | L916 | **双流 launch 编排**：`ValidateAllToAllParams`（check，skip 则直接返回）→ `PrepareAllToAll`（prepare）→ `CcuAllToAllMesh1DKernel<<<schd, insHandle, streamCcu>>>`（一次 launch，kernel 内循环 commTurn 轮）→ `alltoallAicoreKernel<<<8, 0, streamAicore>>>`。CCU 先 launch（它会先睡等 wake，不怕 AIC 晚起） |

### 3.3 host 编排（alltoall_host_impl.h，check 与 prepare 分离）

| 函数 | 行 | 讲解 |
|---|---|---|
| `ValidateAllToAllParams` | L97 | **check 阶段**（不触碰 CCU 资源）：指针/dtype/commTurn/整除链校验 + rank 查询 + 切分尺寸计算；sendCount==0 → 显式 `skip=true` 返回（取代旧的"早退 SUCCESS + kernelArg null"隐式契约） |
| `PrepareAllToAll` | L293 | **prepare 阶段**（前置：校验已通过）：纯编排 ~30 行，依次调下面 4 个 helper → 设 `schd={1,0,0x01}`。**不做 launch** |
| `AcquireChannelsAndIns` | L127 | 调 `.asc` 的 `AcquireCcuChannels`（每远端 rank 一条 UBC_CTP 通道）+ `CreateCcuInsWithEnabledDies`（跨 2 DIE 建资源池）+ `HcclCommAssignCcuIns` 绑定 |
| `AllocBridgeResources` | L149 | `HcommCcuVariableAlloc` 分 commTurn×11 个 Variable、`HcommCcuEventAlloc` 分 commTurn+1 个 Event，`GetAddr` 取各资源在 GM 中的地址填 `mappedAddrs`；构造 `kernelArg` |
| `ComputeAndCopyOffsets` | L187 | **AllToAll 地址数学核心**：`CalGoSize(subBytes)` 算 goSize；`srcOffset[r]=r*sub`、`dstOffset[r]=myRank*blockSize+r*sub`（**myRank 项 host 预乘烧入**——CCU 侧无标量×Variable 运算）；三个数组 memcpy 到 device |
| `ReleaseAllToAllContext` | L258 | 释放 3 个 device offset 数组 + 销毁 CCU 实例（与分配严格配对） |

### 3.4 AIC kernel（alltoall_aicore_kernel_impl.h + .asc L868；原 a2a_bridge_kernel.h，2025-08-29 更名）

| 函数 | 行 | 讲解 |
|---|---|---|
| `alltoallAicoreKernel` | .asc L868 | `__global__ __aicore__` 入口。**首行 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY)`**（函数体无 Cube/Vector 指令，必须手动声明类型，否则 ld.lld 推导失败——编译修复2 的教训）→ `InitSocState` → 转发 `A2aAicoreProcess` |
| `WriteBridgeSlot` | alltoall_aicore_kernel_impl.h L48 | 一行 helper：往 `startVa[r][slot]` 写 value（绕 DCache 直写 GM）。槽位语义不在调用点重复——**见 BridgeSlot 枚举** |
| `A2aAicoreProcess` | alltoall_aicore_kernel_impl.h L69 | 两步：① block0 逐 round 写 11 槽参数桥（`WriteGmByPassDCache`+`DataSyncBarrier`）→ 写 wake（eventVa[0]=1）；② 所有核循环 commTurn 次轮询 `eventVa[r+1]==1`（`ReadGmByPassDCache`+`DataSyncBarrier`+`Nop<800>` 降频）。**为什么 block0 写**：Variable 区是 GM 共享内存，一个核装弹即可；**为什么写完所有轮才 wake**：CCU 唤醒后各轮自动串行，参数必须先备齐 |

### 3.5 CCU kernel（.asc，namespace ops_comm_a2a 内）

| 函数 | 行 | 讲解 |
|---|---|---|
| `CcuAllToAllMesh1DKernel` | L770 | **CCU 主入口**（`__ccu_host__`）。流程对齐 CANN `Algorithm()` 五步：InitResource（取 channel 上的 output/token Variable + events/vars 数组）→ `EventWait(events[0])` 等 wake（循环外一次）→ 循环 commTurn：LoadArgs（按 BridgeSlot 读 11 槽）→ PreSync（`WriteVariableWithNotify` 把本 rank 的 output/token 通知远端 + `NotifyWait` 等远端的）→ `ExecuteAllToAllTransfer` → PostSync（NotifyRecord/Wait 确保传输彻底完成）→ `EventRecord(events[r+1])` 报 done |
| `ExecuteAllToAllTransfer` | L663 | **AllToAll 语义核心**（76 行，全工程最关键函数）。① 构造 src（远端循环用起点=input+srcOffset）与 localSrc（**独立累加循环** `+=srcStride` 递推 myRank 次——本 rank 自留块，与远端循环互不复用）；② 每 rank 算 dst（`output[j]+dstOffset`，dstOffset 已含 myRank 项）；③ `CCU_IF(sliceSize)` 内循环：`j≠my` 则 `ccu::Write`（sendBuf 第 j 块→远端 recvBuf 第 my 块），每迭代 `src.addr+=srcStride` 递推；④ 平级 `GroupCopy(localDst, localSrc)` 本地块；⑤ EventRecord/Wait 全 rank 屏障 |
| `GroupCopy` | L568 | 本地拷贝驱动：按 goSize 分两支——`addrOffset≠0`（有完整 256KB 块：LoopGroup 并行 8 loop）与 `loopExtendNum≠0`（尾部：2 个 Loop 处理 residual+memSlice）。**Loop Entity 每次重建**（lambda 按值捕获当前地址——只建一次会让 round1 复用 round0 的地址快照，AG 踩过的坑） |
| `InitGroupCopyResources` | L524 | 首次调用分配 moConfig/moRes（8 Event + 64 CcuBuffer）；每次调用重建 Loop Entity（见上） |
| `CalGoSize` | L480 | 纯数学：把 subBytes 拆成 m 个 256KB 块 + n 个 32KB 片 + p 尾巴，输出 A2aGoSize 四元组（无分支副作用，可单测） |
| `AcquireCcuChannels` | L351 | 对每个远端 rank：`HcclRankGraphGetLinks` 查物理链路 → 选 UBC_CTP 协议 → `HcclChannelAcquire` 拿通道句柄【AG 原样复用】 |
| `CreateCcuInsWithEnabledDies` | L305 | 遍历 2 个 DIE 建 resDesc（配额 ADDRESS=400/LOOP=16/CCU_BUF=128/VARIABLE=400/EVENT=48/THREAD=2）→ `HcommCcuInsCreate`【AG 原样复用】 |
| 位域编码 6 函数 | L403-459 | `SetBits/GetMaxLoopIterNum/GetLoopParam/GetParallelParam/GetOffsetParam`：CCU 硬件 64bit 寄存器的位段打包（AG 原样复用，`SetBits(end)` 注意 end 是最高位序号：`SetBits(7)=0xFF`） |
| `MakeLocalCopyConfig` | L461 | LoopGroup 机器配置单一装配出处（host/device 两处共用，防手工不一致） |

---

## 4. 三个"为什么"（读代码时最容易卡住的地方）

1. **为什么 dstOffset 在 host 预乘 myRank，而 srcStride 在 CCU 侧递推？**
   dst 是 Variable+Variable 加法（CCU 支持，AG 同款写法）；src 的 rankId 项若预乘则要
   host 传一个"每 rank 不同的 src 基址"——参数桥是静态的做不到，所以用循环递推
   （`src.addr += ctx.srcStride`，每 rank 递推 myRank 次）。

2. **为什么 CCU kernel 先 launch 却后执行？**
   `EventWait(events[0])` 让 CCU 睡到 AIC block0 写完参数桥+wake。launch 顺序无所谓，
   同步语义保证执行顺序。这也是双流解耦的关键。

3. **为什么本地块要"独立累加构造 localSrc"而不复用远端循环的 src？**
   理论上 j 循环递推到 myRank 时 src 恰好停在本 rank 块。但本地拷贝走 `GroupCopy`
   （LoopGroup 硬件原语，异步），远端走 `ccu::Write`——两者并行发出后 src 还要继续
   递推给 j+1 用，不能让 GroupCopy 引用会被继续修改的变量。独立构造彻底解耦。

---

## 5. 阅读路线建议（比本文更省力的源码下钻顺序）

| 顺序 | 文件:函数 | 预计 |
|---|---|---|
| ① | main.cpp 的 `A2aElemPattern` + `GenerateInputData` + `VerifyAndDumpRecv`（数据怎么造、怎么验） | 10 min |
| ② | alltoall_host_impl.h 的 `ComputeAndCopyOffsets`（地址数学——全方案核心公式） | 5 min |
| ③ | .asc 的 `BridgeSlot` 枚举 + `CcuAllToAllMesh1DKernel`（CCU 主循环骨架） | 10 min |
| ④ | .asc 的 `ExecuteAllToAllTransfer`（换块语义三差异的落点） | 10 min |
| ⑤ | alltoall_aicore_kernel_impl.h 全文（最短，参数区预写+轮询） | 5 min |
| ⑥ | 需要时再查：GroupCopy/CalGoSize（本地拷贝细节）、AcquireCcuChannels/CreateCcuInsWithEnabledDies（资源细节，AG 原样复用可不读） | 按需 |

---

## 6. 常量速查

| 常量 | 值 | 含义 |
|---|---|---|
| `MAX_COMMTURN` | 16 | 通信轮数上限（Variable 配额约束：16×11=176<400） |
| `CCU_MAX_RANK_SIZE` | 16 | rank 数上限（channels 数组维度） |
| `AIC_BLOCK_NUM` | 8 | AIC 桥 kernel 核数（Blaze-free 固定值，阶段3 恢复 platform 查询） |
| `CCU_MS_SIZE`×`CCU_MEMSLICES_PER_LOOP` | 4KB×8=32KB | 单 loop 搬运量（memSlice） |
| `CCU_LOCAL_COPY_LOOP_COUNT`×memSlice | 8×32KB=256KB | loopSize（CalGoSize 的块基准） |
| `DIE_ID` | 1 | Variable/Event 分配所在 DIE |
| `OUTPUT_XN_ID/TOKEN_XN_ID/CKE_IDX_0/POST_SYNC_ID` | 1/2/0/3 | channel notify 槽位（PreSync/PostSync 用） |

---

## 7. CCU 资源全景说明（"参数多"的根源与分类记忆法）

> CCU 编程的参数量确实大，但它们分属**四个层次**，按层记忆就不乱：
> **实例（Ins）→ 通道（Channel）→ 参数桥（Variable/Event）→ 搬运原语（Buffer/Loop）**。
> 下表自上而下是"资源分配"的顺序，也是代码里出现的顺序。

### 7.1 四层资源模型总览

```
第1层 CCU 实例 CcuInsHandle          （host 创建，一切资源的容器）
  │  CreateCcuInsWithEnabledDies: 跨2个DIE建资源池, 每DIE配额见 §7.2
  │  HcclCommAssignCcuIns: 绑到 HCCL comm
  │
  ├─ 第2层 通道 ChannelHandle[R-1]    （点到点通信管道）
  │    AcquireCcuChannels: 每远端rank一条 UBC_CTP 协议通道
  │    ├─ 数据面: ccu::Write（跨卡搬运）
  │    └─ CKE 通知寄存器 ×notifyNum 组（随通道创建隐式配置，非显式 alloc——
  │         desc.notifyNum 申购，仅 CCU 两侧可见，GM 无地址）
  │         两级寻址: 组号(notifyIdx) + 组内bit(mask)；本工程 3 组只用第 0 组
  │         的 bit1/2/3（output/token/PostSync），每组 16bit = uint16 mask 宽度
  │    → ccu::Write / WriteVariableWithNotify / NotifyRecord / NotifyWait 都在通道上执行
  │
  ├─ 第3层 参数桥 Variable ×(commTurn×11) + Event ×(commTurn+1)
  │    HcommCcuVariableAlloc / HcommCcuEventAlloc（host 分配）
  │    GetAddr 取 GM 地址 → CcuMappedAddrs（AIC 写参数 / CCU 读参数 / 双向事件）
  │    device 侧由 ccu::Array<ccu::Variable/Event> 句柄访问
  │
  └─ 第4层 搬运原语 CcuBuffer×64 + Loop×2 + LoopGroup（CCU kernel 内部按需构造）
       InitGroupCopyResources: 8 Event + 64 CcuBuffer 一次性分配
       GroupCopy: Loop/LoopGroup 每轮重建（地址快照坑位）
```

### 7.2 host 侧资源 API（hccl/hcomm 系，分配/查询类）

| API（前缀 HcommCcu*） | 层 | 用在哪 | 干什么 |
|---|---|---|---|
| `HcommCcuInsResDescCreate/SetNum/Destroy` | 1 | CreateCcuInsWithEnabledDies | 构造资源描述符：每 DIE 声明 6 类配额 |
| `HcommCcuQueryRemainResDesc` | 1 | 同上 | 探测 DIE 可用性（CCU_E_UNAVAIL 则跳过该 DIE） |
| `HcommCcuInsCreate` | 1 | 同上 | 创建 CCU 实例（返回 insHandle，launch 时用） |
| `HcclCommAssignCcuIns` | 1 | AcquireChannelsAndIns | 把实例绑到 HCCL comm（comm 维度生命周期管理） |
| `HcommCcuInsDestroy` | 1 | ReleaseAllToAllContext | 销毁实例 |
| `HcclRankGraphGetLinks` | 2 | AcquireCcuChannels | 查询到某远端 rank 的物理链路列表 |
| `HcclChannelAcquire` | 2 | 同上 | 按链路拿通道句柄（ENGINE=CCU，协议=UBC_CTP） |
| `HcommCcuVariableAlloc / GetAddr` | 3 | AllocBridgeResources | 分配参数区 Variable 并取其 GM 地址 |
| `HcommCcuEventAlloc / GetAddr` | 3 | 同上 | 分配 Event（位掩码型信号）并取其 GM 地址 |
| `HcommCcuGetMemToken` | 3 | PrepareAllToAll | 取 GM 内存访问令牌（CCU 访问 sendBuf/recvBuf 需携带） |

**6 类配额**（`CCU_DEFAULT_RES`，每 DIE）：ADDRESS=400 / LOOP=16 / CCU_BUF=128 / VARIABLE=400 / EVENT=48 / CCU_THREAD=2。
本工程用量校验：Variable 最多 16×11=176<400 ✓，Event 最多 17<48 ✓，CcuBuf 64<128 ✓，Loop 2<16 ✓。

### 7.3 device 侧资源 API（ccu:: 系，CCU kernel 内使用）

| API | 层 | 本工程用法 | 语义 |
|---|---|---|---|
| `ccu::Array<T>` | 3/4 | `ccu::Array<ccu::Variable>(handle, num)` | 句柄数组包装（kernelArg 里的 handle → 可索引对象） |
| `ccu::GetResByChannel<ccu::Variable>` | 2→3 | 取远端 rank 的 output/token Variable | **通道上寄生的资源**：把本 rank 的 Variable 值写到通道上，远端就能 Get 到（PreSync 的机制基础） |
| `ccu::Variable` | 3 | 参数桥的每个"槽位" | 一个可读写的数值单元，在 GM 中有实际地址（AIC 也能访问——共享内存参数区的物理基础）；支持 `+=` 算术 |
| `ccu::Event` | 3 | wake/round done/本地拷贝完成 | **16bit 位掩码信号**：EventRecord(evt, mask) 置位、EventWait(evt, mask) 等位——mask 决定"谁/哪些 rank" |
| `ccu::LocalAddr {addr, token}` | 4 | src/localSrc/localDst | 本地 GM 地址 + 令牌（CCU 访问 GM 必须带 token） |
| `ccu::RemoteAddr {addr, token}` | 4 | dst[j] | 远端 GM 地址 + 令牌（ccu::Write 的目标） |
| **CKE 通知寄存器** | 2 | `desc.notifyNum=3` 随通道隐式申购；`NotifyRecord/Wait(ch, 组号, mask)` 两级寻址 | 通道硬件寄存器（**无 GM 地址、无显式 Alloc**——与 Variable/Event 的获取方式根本不同）；本工程用组 0 的 bit1(OUTPUT_XN_ID)/bit2(TOKEN_XN_ID)/bit3(POST_SYNC_ID) |
| `ccu::Write(ch, dst, src, len, evt, mask)` | 2 | 换块：sendBuf 第 j 块 → 远端 recvBuf | **跨卡数据搬运**：经通道把 src 处 len 字节写到远端 dst；evt+mask 用于完成同步 |
| `ccu::LocalCopy(buf, addr, len, evt, mask)` | 4 | GM→CcuBuffer / CcuBuffer→GM 两跳 | 本地搬运一跳（本地拷贝经 CCU Buffer 中转） |
| `ccu::CcuBuffer` | 4 | 64 个（8 loop × 8 interleave） | CCU 片上搬运缓冲（本地拷贝的中转站） |
| `ccu::Loop(loopParam, body)` | 4 | 每轮重建 ×2 | 可重入的循环体（body 是 ccu::Func 闭包，捕获地址按值） |
| `ccu::LoopGroup(paraCfg, offsetCfg, count, loops)` | 4 | 完整块 8 loop / 尾部 2 loop | **并行 loop 调度器**：count 个 Loop 并行跑，offsetCfg 控制 buffer 交错 |
| `ccu::WriteVariableWithNotify(ch, var, xnId, cke, mask)` | 2 | PreSync：output/token 告知远端 | 把 Variable 值写到通道上并触发 notify（远端可 Get + 收到通知） |
| `ccu::NotifyRecord(ch, cke, mask)` | 2 | PostSync | 在通道 notify 寄存器上置位 |
| `ccu::NotifyWait(ch, cke, mask)` | 2 | PreSync/PostSync | 等通道 notify 的指定位到达 |

### 7.4 一轮通信的"资源使用时序"（把上面串起来）

```
[host] Ins创建→通道获取→Variable/Event分配→token→offsets拷device ──► 双流launch
[AIC ] block0: 写 startVa[r][11槽] → eventVa[0]=1(wake)
[CCU ] EventWait(ev[0])                       ← 第3层 Event：等唤醒
        ┌─ 循环 r = 0..commTurn-1 ─────────────────────────────────┐
        │ LoadArgs: vars[r*11 + SLOT_*] 读11槽                      ← 第3层 Variable
        │ PreSync: WriteVariableWithNotify(output/token)+NotifyWait ← 第2层 通道notify
        │   （经 GetResByChannel 拿到远端 output[j]/token[j]）        ← 第2→3层
        │ ExecuteAllToAllTransfer:
        │   远端: ccu::Write(ch[idx], dst[j], src, slice, ev, 1<<j)  ← 第2层通道搬运
        │   本地: GroupCopy → LoopGroup{LocalCopy×2跳}               ← 第4层 Buffer/Loop
        │   屏障: EventRecord(ev, 1<<my) → EventWait(ev, 全rank位)   ← 第3层 Event
        │ PostSync: NotifyRecord/Wait(POST_SYNC_ID)                  ← 第2层 通道notify
        │ EventRecord(events[r+1])  → AIC 轮询到 done               ← 第3层 Event
        └────────────────────────────────────────────────────────────┘
[host] 双流sync → ReleaseAllToAllContext（销毁 Ins，通道/参数区资源随之回收）
```

### 7.5 最容易混淆的四组概念

| 易混点 | 区分 |
|---|---|
| **Event vs Notify vs CKE** | Event 是**共享内存参数区层**的 16bit 位掩码信号（在 GM 中有实际地址，AIC 也能轮询——wake/done 用的就是它）；Notify 是**通道层**的寄存器信号（只有 CCU kernel 两侧能碰——PreSync/PostSync 用）。**CKE = 通道上通知寄存器的组号**：一条通道配 `CHANNEL_NOTIFY_NUM=3` 组（channel desc 的 `notifyNum` 字段），本工程全用第 0 组（`CKE_IDX_0`），组内再按 **mask 位**区分用途（bit1=OUTPUT_XN_ID、bit2=TOKEN_XN_ID、bit3=POST_SYNC_ID）。（历史注：该文件变量名原为 `waitCke`——AG 沿袭的误植，2025-08-29 检视已更名 `doneVal`，消除与通道 CKE 的混淆。）跨流（AIC↔CCU）用 Event，跨卡 kernel 间用 Notify（带 CKE 组号） |
| **Variable vs 普通 uint64** | Variable 是 CCU 运行时可读写的数值单元（在 GM 中有实际地址），AIC 写共享内存参数区就是在写它的 GM 地址；CCU 代码里 `Variable += 标量` 是合法算术。host/AIC 的普通变量没有这个能力 |
| **CcuBuffer vs GM** | CcuBuffer 是 CCU 片上缓冲（本地拷贝必须两跳中转：GM→Buffer→GM），跨卡 Write 则直接 GM→远端 GM 不经 Buffer |
| **通道数 vs 句柄数** | `ChannelHandle` 是本卡到每个其他卡的**点到点双向通道**，每卡持 `rankSize-1` 个句柄（`AcquireCcuChannels` 跳过自身）。**每对卡 (i,j) 之间只有一条通道**，双方句柄指向同一条通道的两端视角——所以全系统总通道数 = **C(R,2) = R×(R-1)/2**（R=2→1 条，R=4→6 条，R=16→120 条），而非 R×(R-1)（那是句柄总数，差 2 倍）。双向性代码佐证：同一通道上 `ccu::Write`（我→远端）与 `NotifyWait`（我←远端）并存，PreSync 双方互写互等。此模型即 CANN 命名 "**Mesh1D**" 的全互联一维 mesh 拓扑（无环无树无中继）；R 很大时通道数平方增长，需考虑 NHR 类拓扑，当前 R≤16 够用 |

### 7.6 为什么参数这么多？（设计视角）

CCU 是**独立于 AICore 的通信微架构**，host 无法直接喂参数给 CCU kernel（不像 AIC kernel 走
`<<<>>>` 参数传递）——所以必须用"host→AIC→GM 参数桥→CCU"四级接力，参数桥的 11 槽 × commTurn
轮就是这么来的。通道/Buffer/Loop 则是 CCU 硬件的原生资源模型（对齐 HCCL server 内部用法），
配额制是为了多算子共享 CCU 时不互相踩踏。**记住 §7.1 的四层图，任何参数都能对号入座。**

---

## 8. 全文档导览（本文怎么用）

```
新人上手（第一次接触本工程）
  ① README.md        5min   —— 工程定位/编译运行/与 AG 差异速览
  ② 本文 §0-§1       5min   —— 数据流图 + 文件地图（建立全局感）
  ③ 本文 §2.1-2.2    8min   —— 参数体系：槽位/十一槽FAQ/参数袋（核心中的核心）
  ④ 按需查 §3        —— 遇到具体函数再来查卡片

深入机制（读代码卡住时）
  ⑤ 本文 §2.1.1      —— 参数疑问 90% 能自答（OFFSET/STRIDE/N组参数/指令集约束）
  ⑥ 本文 §4          —— 三个"为什么"（dstOffset预乘/CCU先launch/localSrc独立构造）
  ⑦ 本文 §7          —— 资源四层模型 + 易混概念（Event/Notify/CKE、通道数vs句柄数）

维护改造（阶段3 或后续优化）
  ⑧ DESIGN_ANALYSIS.md   —— 方案全档：语义差异/搬运映射表/风险清单/§8.1 未来优化点
  ⑨ CODE_REVIEW.md       —— 两轮检视的历史记录（改什么之前先看踩过什么坑）
  ⑩ 本文 §6              —— 常量速查
```

## 修订记录

| 日期 | 内容 |
|---|---|
| 2025-08-29 v1 | 初版：§0-§6（数据流图/文件地图/数据结构/函数讲解/三问/阅读路线/常量） |
| 2025-08-29 v2 | +§7 CCU 资源全景（四层模型/资源API/时序/易混概念）；§7.5 增通道数vs句柄数、Event/Notify/CKE |
| 2025-08-29 v3 | §2.1.1 十一槽 FAQ 大改（参数总表/三大小关系/OFFSET-STRIDE实例/N组参数必然性/指令集约束/搬运全景图）；§2.2 补 Variable vs Event 两类资源区分+三线路界线；§2.1.2 A2aGoSize 并入参数体系；§2.4 AllToAllContext 降级为工程细节并加定位澄清；行号/行数全量刷新 |
