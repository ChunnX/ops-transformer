# CCU AllToAll 方案分析与实施计划

> 状态：**阶段1（编码及编译验证）已完成** ✅（2025-08-29，用户服务器编译通过）
> 待推进：阶段2 精度验证（用户 NPU 服务器上板，用例矩阵见 §7）。
> 基线工程：`../../ccu_all_gather_matmul/`（相对本文件 docs/ 目录；下称 AG 示例）。
> 已确认决策：① 三阶段推进（阶段1 编码+交付编译 / 阶段2 精度验证 / 阶段3 下一步演进）；
> ② 均匀 AllToAll（strideCount 型）；③ dtype 默认 **FP16**；④ 参数桥 11 槽（"参数桥"=AIC/CCU kernel
> 间传参的 GM 共享内存参数区，沿自 AG 的称呼，详见 CODE_GUIDE.md §2.1 术语说明）；⑤ AIC kernel 保留
> per-round done 轮询；⑥ **agent 不在本机/远程执行任何编译动作**，编码完成即通知用户编译；
> ⑦ **代码策略：直接搬 CANN `ccu_kernel_all_to_all_mesh1d.h` 的骨架 + AG 已上板验证的实现模式，先快速跑通再优化**。

---

## 0. 核心目标

**一句话：把 `ccu_all_gather_matmul` 已上板验证的"CCU 直调通信 + AIC 轮询"通算融合范式复刻到 AllToAll，
产出一个通信下发权完全握在自己 kernel 手里、可按 round 与计算细粒度流水化的 AllToAll 通信基座，
为后续 AllToAll+计算（MoE dispatch/combine、A2AV+GMM 类）融合算子打底。**

| 层 | 内容 |
|---|---|
| 最终指向 | AllToAll 通信与 AIC 计算在同一算子内按 round overlap，支撑 MoE 类通算融合算子 |
| 直接交付 | `examples/ccu_alltoall`：与 AG 同架构的 CCU 直调 AllToAll（`__ccu_host__` kernel + `ccu::Write`/`GroupCopy`/Event/Notify + 参数桥 + wake/轮询） |
| 参考定位 | `hccl_communicator.h` 是**溯源入口**：沿它找到 CANN 内部与 AG 同源的 `CcuKernelAlltoAllMesh1D`，其骨架**直接搬用**（§4）；高阶 API 本身**不是**调用目标 |

**为什么不直接用高阶 `Hccl::AlltoAll`**：高阶路径的通算编排被限定在 HCCL server 的任务队列模型内
（Prepare/Commit/Wait 编排、63 个 handle 上限、server 步调）；CCU 直调则把 round 切分、事件同步、
参数桥全部收进自己的两个 kernel（streamCcu 通信 + streamAicore 计算），per-round 精确握手。

**边界**：不调用 `Hccl::AlltoAll`/`AlltoAllV` 作为通信实现；不引入 SHMEM/UDMA 路线；阶段3 计算侧仅占位。

**达成判定**：
1. **语义正确**：等量换块，`recvBuf` 第 r 块 == rank r 的 `sendBuf` 第 myRank 块（溯源编码数据可定位任何错位）；
2. **架构同款**：通信全在 `__ccu_host__` CCU kernel 内，资源模型与 AG 一致；
3. **流水可用**：commTurn≥2 多轮事件桥验证通过（阶段3 overlap 的前提，阶段2 验证）；
4. **工程闭环**：独立目录，用户服务器可编译，precision/perf 双模式。

---

## 1. AllToAll 与 AllGather 语义对比

### 1.1 语义定义

**AllGather**：每个 rank 把**同一个**本地 slice 发给**所有** rank；recvBuf 按
`recv[r] = send_r`（r = 来源 rank）拼接，recvBuf = rankSize × sendBuf（膨胀 R 倍）。

**AllToAll**（均匀，strideCount 型）：每个 rank 把 sendBuf 切成 R 块，**第 j 块只发给 rank j**；
rank i 收到所有 rank 的各自的第 i 块，按**来源 rank** 拼接：`recv_i[r] = send_r[i]`。
recvBuf 与 sendBuf **等量**（数据守恒）。

高阶 API 语义（`AscendC::Hccl::AlltoAll`）：
> sendBuf 第 j 块（偏移 `j*strideCount`，长 `dataCount`）由 rank i 发给 rank j，
> 落在 rank j 的 recvBuf 偏移 `i*strideCount` 处。

### 1.2 2 rank 示意（R=2）

```
AllGather（AG 示例, R=2）                AllToAll（目标, R=2）
rank0 send: [A0]                         rank0 send: [A0|A1]   (A1 → rank1)
rank1 send: [B0]                         rank1 send: [B0|B1]   (B0 → rank0)

rank0 recv: [A0|B0] ← 膨胀 2x            rank0 recv: [A0|B0]   (B0 = rank1 的第 0 块)
rank1 recv: [A0|B0]                      rank1 recv: [A1|B1]   (A1 = rank0 的第 1 块)
                                         等量交换，每个 rank 收到的都不同
```

### 1.3 对 CCU 传输原语的影响（核心差异表）

| 维度 | AllGather（AG 现状） | AllToAll（目标） |
|---|---|---|
| 单轮内每次 `ccu::Write` 的 src | **固定**：`input + inputOffset` | **逐 dst rank 变化**：`input + j*srcStride + srcOffset` |
| 远端 dst 地址 | 远端 `output + outputOffset`（统一值） | 远端 `output + **myRank***stride + dstOffset`（发送方 rankId 决定落点） |
| 本地拷贝（GroupCopy） | `sendBuf+inputOff → recvBuf+outputOff` | `sendBuf+myRank*stride → recvBuf+myRank*stride`（src 侧也含 rankId 项，独立累加构造） |
| 参数桥槽位 | 10 槽 | **11 槽**（新增 srcStride；srcOffset/dstOffset 替代 inputOffset/outputOffset） |
| 数据量 | recvBuf = R × sendBuf | recvBuf = sendBuf |
| dst 落点计算位置 | host 侧算好单值 | CCU 侧用 kernelArg.rankId 参与 |

**一句话**：AllGather 是"一份源广播、落点由轮次决定"；AllToAll 是"逐对换块、src 与 dst 都由
rankId 决定"。CCU 原语集完全够用，差异全部体现在**地址计算与参数桥布局**。

---

## 2. AG 示例代码复用分析

### 2.1 可直接复用（约 70%）

| 模块 | AG 位置 | 复用方式 |
|---|---|---|
| `CreateCcuInsWithEnabledDies` | `.asc` L249 | 原样复制 |
| `AcquireCcuChannels` | `.asc` L302 | 原样复制 |
| 位域编码工具 6 个 | `.asc` L360-481 | 原样复制 |
| `CalGoSize` | `.asc` L400 | 原样复制 |
| `InitGroupCopyResources` + `GroupCopy` | `.asc` L496-627 | 原样复制（含 Loop Entity 每轮重建坑位） |
| WriteVariableWithNotify + NotifyWait 参数同步 | `.asc` L812-822 | 原样复制 |
| Post-sync（NotifyRecord + NotifyWait） | `.asc` L830-835 | 原样复制 |
| EventRecord/EventWait 屏障 | `.asc` L705-708 | 原样复制 |
| 双流 + block0 参数桥预写 + wake/直轮询 | `matmul_fusion_kernel.h` | 结构复用，去 Blaze |
| Host 编排框架 | `alltoall_host_impl.h` | 结构复用，字段改 alltoall |
| 多 rank 框架、precision/perf 双模式 | `main.cpp` | 框架复用，数据/校验重写 |
| 构建系统 | `CMakeLists.txt`/`run.sh` | 复制改 target，Blaze-free 裁剪 |

### 2.2 需改写

| 模块 | 改动 |
|---|---|
| `ExecuteAllGatherTransfer` → `ExecuteAllToAllTransfer` | ① src 逐 rank 递推；② dst 含 myRank×stride；③ 本地拷贝双偏移 |
| 参数桥 | 10 → 11 槽，`CcuMappedAddrs.startVa` 维度同步改 |
| Host offsets | allgather 公式作废，按 alltoall 语义重算（§6.5） |
| dtype 校验 | INT8 → **FP16** |
| 数据准备/校验 | 溯源编码，元素级可区分（§6.2） |

### 2.3 不可复用（allgather 语义强绑定，阶段3 另议）

`MatmulProcess`（Blaze 链）、MXFP8 双流、M 轴 tile 轮切公式。

---

## 3. 高阶 API → CCU 底层追踪（CANN 包证据链）

```
[高阶模板层] mc2_templates/communication/
  ├─ communicator/hccl_communicator.h   HcclCommunicator 模板（任务队列编排）
  └─ primitives/hccl_primitives.h       MC2AlltoAllPrimitives → hccl->AlltoAll<false>(...)
                  │
[device API 层] x86_64-linux/asc/include/adv_api/hccl/hccl.h (L157)
  └─ AscendC::Hccl::AlltoAll<commit>(sendBuf, recvBuf, dataCount, dataType, strideCount, repeat)
                  │
[HCCL server 指令层] .../alg_ccu_context/all_to_all/ccu_instruction_all_to_all_mesh1d.h
  └─ CcuTaskArgAllToAllMesh1D{inputAddr, outputAddr, sliceSize, token, srcOffset, dstOffset, srcStride}
     指令类型 CCU_ALLTOALL_MESH_1D_DIRECT
                  │
[CCU kernel 层（直接搬用对象）] .../ops/all_to_all_v/template/ccu/kernel/
  ├─ ccu_kernel_all_to_all_mesh1d.h    CcuKernelAlltoAllMesh1D（本方案主参考）
  └─ ccu_kernel_all_to_all_mesh2die.h  CcuKernelAllToAllMesh2Die（旁证：inputSliceStride/outputSliceStride 建模）
```

**关键事实（如实说明）**：CANN 包内 `ccu_kernel_all_to_all_mesh1d.h` **只有类声明头文件**，
`Algorithm()/DoAlltoAll()` 实现体编译在 libhccl.so 中。因此"直接搬"的实际形态是：
**搬其骨架（TaskArg 参数四元组、成员变量布局、Algorithm 五步流程），血肉用 AG 示例中
已上板验证的同款 ccu 原语实现填充**——AG 本身就是从隔壁 all_gather 同款头文件
（`ccu_kernel_all_gather_mesh1d_mem2mem.h`，类名一致）搬出来的成功先例，路径已验证。

---

## 4. CANN 代码直接搬运映射表（用户已认可，快速跑通策略）

| CANN `CcuKernelAlltoAllMesh1D` 成员/方法 | 本工程 `ccu_alltoall` 对应物 | 搬运方式 |
|---|---|---|
| `CcuTaskArgAlltoAllMesh1D{inputAddr, outputAddr, sliceSize, token, srcOffset, dstOffset, srcStride}` | 参数桥 11 槽（每字段一个 Variable） | **参数语义直接搬**（含四元组递推含义） |
| `srcAddr_[R]`（LocalAddr 数组）/ `dstAddr_[R]`（RemoteAddr 数组）/ `myDst_` | `ExecuteAllToAllTransfer` 内 src/dst[rankIdx]/localDst | 结构照搬，写法用 AG 已验证原语 |
| `Algorithm() = InitResource → LoadArgs → PreSync → DoAlltoAll → PostSync` | CCU kernel 主循环：InitResource（channel/vars/events）→ 每轮 LoadArgs（读 11 槽）→ PreSync（WriteVariableWithNotify + NotifyWait）→ DoAlltoAll（ExecuteAllToAllTransfer）→ PostSync（NotifyRecord/Wait）→ EventRecord | **流程骨架直接搬**，映射到 AG 双流结构 |
| `channels_` / `event_` / `groupOpSize_` / `token_` | `ctx.arg->channels` / `ctx.event` / `ctx.goSize` / `ctx.token` | AG 已验证实现 |
| `repeatNum_` + `flag_`（repeat 重复执行） | 暂不搬（commTurn 多轮机制已覆盖；阶段3 若需 repeat 语义再加） | 简化 |
| mesh2die 版 `inputSliceStride/outputSliceStride` 分离建模 | 阶段1 合并为单 `srcStride`（均匀语义二者相等）；字段位已预留 | 旁证参考 |

**结论**：CCU kernel 的类结构、方法划分、参数集**逐项对齐 CANN 原版**；实现细节（ccu::Write /
GroupCopy / Event / Notify 的调用写法、Loop Entity 重建、post-sync 顺序）**逐行对齐 AG 已验证代码**。
两个来源都是跑通过的代码，本工程只做"换块语义"的地址计算替换——这是风险最小的快速跑通路径。

---

## 5. 阶段划分与推进协议

### 5.1 三阶段总表

| 阶段 | 内容 | 交付物 | 完成标志 | 编译/运行 |
|---|---|---|---|---|
| **阶段1：编码** | 全部代码落地 + 静态自查 + 编译指引 | `.asc`/`alltoall_aicore_kernel_impl.h`/`alltoall_host_impl.h`/`main.cpp`/`CMakeLists.txt`/`run.sh`/`README.md` | 编码完成，飞书通知用户；用户编译报错回传后由 agent 修复直至编译通过 | **agent 不执行编译**；用户在自有服务器编译 |
| **阶段2：精度验证** | 上板跑通 + 语义校验 + 修复迭代 | 运行日志 + 校验结果 + npu_out.bin | precision 模式 PASS（commTurn=1 与 2） | 用户在 NPU 服务器执行，回传结果 |
| **阶段3：下一步演进** | 通算融合方案（计算侧选型/双流/流水切法） | 演进方案文档 | 方案评审通过 | 无 |

### 5.2 阶段1 子步骤

| 子步骤 | 内容 | 状态 |
|---|---|---|
| 1a | 工程骨架：CMakeLists（Blaze-free）+ run.sh + README | ✅ |
| 1b | `alltoall_host_impl.h`：AllToAllContext + PrepareAllToAll（11 槽、FP16 校验、offsets、CalGoSize、token） | ✅ |
| 1c | `alltoall_aicore_kernel_impl.h`（AIC 参数区预写 + wake + done 轮询，无 Blaze；原名 alltoall_aicore_kernel_impl.h，2025-08-29 更名）+ `alltoall_ccu_kernel_impl.asc`（CCU kernel 全量 + launch） | ✅ |
| 1d | `main.cpp`：溯源编码数据 + 断言校验 + precision/perf 双模式 | ✅ |
| 1e | 静态自查（符号一致性/include/槽位索引/ASC 语法对齐 AG） | ✅ |
| 1f | 用户服务器编译验证（两轮报错修复后通过） | ✅ 2025-08-29 |

**阶段1 编译问题记录**（详见 §5.2.1）：
1. `ACLCHECK` 宏返回 `aclError`，用于 `HcclResult` 返回函数导致类型不兼容（commit `1a80ff575`，含用户手改的 `aclrtFinalize`→`aclFinalize` API 名修正）
2. 桥 kernel 无 Cube/Vector 指令，ld.lld 无法自动推导 kernel 类型 → 手动声明 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY)`（commit `1e013df40`）

### 5.2.1 编译问题与修复记录

| # | 报错 | 根因 | 修复 | commit |
|---|---|---|---|---|
| C1 | `alltoall_host_impl.h:205 cannot initialize return object of type HcclResult with an lvalue of type aclError` | `ACLCHECK` 宏内部 `return ret`（aclError），被错误用于 `HcclResult` 返回函数（重构时误替换 AG 原版的 aclRet 局部变量模式） | `ComputeAndCopyOffsets`/`ReleaseA2aContext`/`RunA2aOnce`（后更名 InvokeAllToAll）共 11 处改回 `aclRet` 局部变量 + `return HCCL_E_INTERNAL`；审计确认剩余 ACLCHECK 均在 int 返回函数（合法） | `1a80ff575` |
| C2 | `ld.lld: error: the type of kernel function alltoallAicoreKernel auto derivate failed, please manually add the function type attribute` | 桥 kernel 函数体仅含 GM 轮询读写 + Nop，无 Cube/Vector 指令，编译器无法自动推导 AIC/AIV 类型（AG 的 matmul kernel 靠 Blaze Cube 指令自动推导，无此问题） | kernel 体首行加 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);`（语法依据：AG phase2 用过同款宏 MIX_AIC_1_1 + CANN broadcast_sch.h 的 AIV_ONLY 先例；枚举全集 AIV_ONLY/AIC_ONLY/MIX_*/AICORE） | `1e013df40` |

### 5.3 飞书通知协议（每阶段完成必达）

每阶段完成 → `~/feishu-bot/send-feishu.sh` 推送**卡片**（标题 + 完成摘要 + 按钮 + 输入框）→
短时等待回复 → 按回答分支行动。阶段1 完成卡按钮：
`{"✅ 我去编译，回头反馈":"user_build","❰ 有问题先说":"discuss","⏸ 暂停":"pause"}`。
约束：同一时刻仅一个挂起回调；超时视为已送达，等用户异步回复。

---

## 6. 阶段1 详细设计（FP16、单流、均匀 AllToAll）

### 6.1 交付文件清单

```
examples/ccu_alltoall/
├── CMakeLists.txt              # 工程根：AG 复制后 Blaze-free 裁剪，target=ccu_alltoall
├── README.md                   # 编译/运行/校验指引（面向用户服务器）
├── docs/
│   └── DESIGN_ANALYSIS.md      # 本文档
├── scripts/
│   └── run.sh                  # source 环境 + CCU_SCHED + cmake + build（用户服务器执行）
└── src/                        # 单翻译单元：.asc 经相对 include 引入其余三个文件
    ├── alltoall_ccu_kernel_impl.asc # CCU kernel（__ccu_host__）+ AIC 桥 kernel + launch 入口
    ├── alltoall_aicore_kernel_impl.h     # AIC 侧：参数区预写 + wake + per-round done 轮询（无 Blaze）
    ├── alltoall_host_impl.h         # AllToAllContext + PrepareAllToAll
    └── main.cpp         # FP16 数据准备/校验 + precision/perf 双模式 + main
```

### 6.2 数据模型（FP16 字节流，溯源编码）

- R = rankSize（2 卡起步，CCU_MAX_RANK_SIZE=16 上限）
- 每 rank sendBuf 均分 R 块：`blockSize = perRankSize / R`（整除校验）；`stride = blockSize`
- **溯源编码数据（每个 slice、每个元素都不同，用户指定的精度分析友好设计）**：
  每个 uint16 元素位型 = `srcRank<<12 | dstBlock<<8 | tPos`，其中
  `tPos = (t ^ (t>>8)) & 0xFF`（块内位置摘要；非线性混合使"Δ 为 256 整数倍"的
  块内错位亦可检出——裸 `t&0xFF` 对 commTurn 子段边界类偏移是盲区）。
  纯字节搬运验证：CCU Write/GroupCopy 均为字节级拷贝，dtype 仅用于 SIZE_TABLE
  取字节宽度，位型不解释为 FP16 数值（阶段3 叠加 matmul 时另行设计数值型数据）
- 发送侧：rank i 的 sendBuf 第 j 块元素 t = `Encode(src=i, dst=j, t)`
- 期望（AllToAll 语义 `recv_i[r] = send_r[i]`）：rank i 的 recvBuf 第 r 块元素 t
  = `Encode(src=r, dst=i, t)`，∀r, ∀t
- 诊断：actual 位型直接读出 srcRank/dstBlock（错误数据实际来自谁/本想去谁），
  可分类错误类型：src 混淆 / 块混淆 / 块内偏移 / 数据损坏

### 6.3 CCU kernel：ExecuteAllToAllTransfer

```
输入（参数桥 round r 组，11 个 Variable）：
  input / output / token / srcOffset / dstOffset / sliceSize / srcStride / goSize 四元组
  （槽位语义的单一事实源：src/ 的 BridgeSlot 枚举——实现见 LoadArgs / WriteBridgeSlot）

1) 远端传输：for j in 0..R-1, j != myRank:
     src.addr    = input + srcOffset + j * srcStride              // 差异①：逐 rank 递推 src
     dst[j].addr = output_var[j] + myRank * srcStride + dstOffset // 差异②：发送方 rankId 决定落点
     ccu::Write(channels[chId(j)], dst[j], src, sliceSize, event, mask=1<<j)
2) 本地拷贝：
     src = input + srcOffset + myRank * srcStride                 // 差异③：本地块 src 含 rankId 项
     dst = output_var[myRank] + myRank * srcStride + dstOffset
     GroupCopy(dst, src, goSize)
3) 屏障：EventRecord(event, 1<<myRank) → EventWait(event, (1<<R)-1)
```

### 6.4 参数桥布局（11 槽）

```
[0]=input(sendBuf) [1]=output(recvBuf) [2]=token
[3]=srcOffset [4]=dstOffset [5]=sliceSize [6]=srcStride [7..10]=goSize 四元组（槽位定义见 BridgeSlot 枚举）
Variable 总数 = commTurn*11（commTurn=16 时 176 < 配额 400 ✓）
Event：eventVa[0]=wake + eventVa[r+1]=round r done，共 commTurn+1 个
```

### 6.5 Host offsets（commTurn 轮按块内均分）

```
blockSize 再按 commTurn 均分为 sub：
  round r: srcOffset[r]=dstOffset[r]=r*sub, sliceSize=sub, srcStride=blockSize
commTurn=1 时退化为全零偏移 + sliceSize=blockSize（最简形态）
```

### 6.6 CMake Blaze-free 裁剪

删除：OPS_TENSOR_ROOT 检查与 include 路径、`platform tiling_api register` 链接库、
platform_ascendc 相关 include（AIC 桥 kernel 用固定核数 8，阶段3 恢复 platform 查询）。
保留：ASC 语言、`--npu-arch=dav-3510`、CANN include/lib64、hcomm/asccomm_ccu_dataplane 等。

### 6.7 AIC 桥 kernel（无 Blaze）

复用 AG `AicDirectPollProcess` 骨架，删 MatmulProcess/MatmulTilingData：
block0 逐 round 写 11 槽参数桥（WriteGmByPassDCache + DataSyncBarrier）→ 写 wake →
all blocks 循环 commTurn 次轮询 eventVa[r+1]==1（ReadGmByPassDCache + Nop<800>）。
offset/sliceSize 经 `__gm__ const uint64_t *` 数组传入（host 预拷 device，与 AG 同法）。

---

## 7. 阶段2：精度验证（用户 NPU 服务器）

| 用例 | 参数 | 验证点 |
|---|---|---|
| 2a-1 | R=2, FP16, commTurn=1, 默认 blockSize | 单轮全量换块语义（差异①②③地址计算） |
| 2a-2 | R=2, FP16, commTurn=1, 非整 blockSize（触发 goSize 尾部） | GroupCopy 尾部路径 |
| 2b-1 | R=2, FP16, commTurn=2 | 多轮事件桥 + 轮间参数隔离 |
| 2c 可选 | R=4 / 更大 blockSize | 扩展 rank、配额余量 |

precision 模式：host 读回 recvBuf 逐块断言 + 首错定位（rank/块/偏移/期望 vs 实际）+ bin 落地。
环境：NPU ≥2 卡 + CANN + `HCCL_OP_EXPANSION_MODE=CCU_SCHED`；agent 负责日志分析与修复迭代。

---

## 8. 阶段3：演进占位（另行评审）

流水切法候选：A. 块内子段轮转（§6.5 推广，CCU 零改造）/ B. 段内按 rank 轮转。
计算侧候选：拼接大 A 普通 matmul（Blaze 全复用，CMake 恢复点加回）/ MoE GMM（`hccl_a2av_op.h` 参考）。
接口预留：AllToAllContext 双流字段占位；参数桥结构不变。

### 8.1 未来优化点（记录备查，不落方案——2025-08-29 用户指示）

**本地块零拷贝**：CCU 发现 `j == myRank` 时直接跳过本地搬运，本地块留在 sendBuf 原地，
下游按"块→地址"映射直读（远端块读 recvBuf、本地块读 sendBuf）。语义上本地数据本来就在
本卡 GM，搬运只为凑出物理连续的 recvBuf。
- 收益：每轮省 sliceSize 字节的两跳中转（GM→CCU Buf→GM）+ LoopGroup 调度延迟；
  约省每轮 1/R 的 CCU 工作量（R=2 时 ~1/3，R=16 时 ~6%）
- 连带简化：参数桥 11 槽 → 7 槽（goSize 四件套仅服务本地搬运）、CalGoSize/GroupCopy/
  InitGroupCopyResources 约 250 行设备侧代码可删
- 代价：recvBuf 契约从"一块连续内存"变"R 块地址表"（下游需地址映射）；
  阶段1 逐块断言的验证闭环失效（本地块位置为垃圾）
- 前置条件：与阶段3 计算侧路线强相关——GMM 路线天然支持（每组独立地址），
  拼接大 A 路线（连续布局刚需）不兼容；高阶先例：CANN allto_allv_quant_grouped_mat_mul
  的本地 expert 数据即零拷贝直读
- 参考对话：2025-08-29（用户："CCU 发现是本卡直接略过搬运，下游自己找正确地址"）

---

## 9. 风险点清单

| # | 风险 | 说明 | 缓解 |
|---|---|---|---|
| P1 | 参数桥 11 槽改动 | AG 自定义常量，仓库内自洽 | 无外部依赖 |
| P2 | Variable 配额 | commTurn*11 ≤ 400 | 已验算 |
| P3 | dst 落点含发送方 rankId | Variable+标量混合算术 | AG 有先例；阶段2 专项验收 |
| P4 | GroupCopy 复用边界 | src 含 rankId 偏移，Loop Entity 每轮重建 | AG P14 已验证 |
| P5 | dtype FP16 | SIZE_TABLE=2 | 查表；FP16 整数精确（<2048）已核算 |
| P6 | 校验盲区 | 退化为本地自拷可能漏过 | 溯源编码数据设计 + 逐元素断言 |
| P7 | CCU 性能不可采集 | 沿用 AG 结论 | 阶段2 不以 perf 为标准 |
| P8 | agent 无法编译验证 | 用户服务器不可访问 | 静态自查清单 + 编译指引 + 报错回传修复闭环；Blaze-free 降低依赖面 |
| P9 | 飞书回调单并发 | 同一时刻一个挂起 | 每阶段一卡，轮询校验 title |

---

## 10. 已确认决策记录

| # | 决策项 | 结论 |
|---|---|---|
| 1 | dtype | **FP16** |
| 2 | commTurn 策略 | 1 跑通后立即验证 2 ✓ |
| 3 | 参数桥 | 11 槽 ✓ |
| 4 | AIC per-round 轮询 | 保留 ✓ |
| 5 | 阶段3 计算侧 | 另行评审 ✓ |
| 6 | 阶段完成通知 | 飞书卡片 + 询问下一步 ✓ |
| 7 | 阶段结构 | 三阶段 ✓ |
| 8 | 编译执行方 | **用户自有服务器，agent 不编译**，编码完成即通知 ✓ |
| 9 | 代码策略 | **直接搬 CANN `ccu_kernel_all_to_all_mesh1d.h` 骨架 + AG 已验证实现，先跑通后优化** ✓ |
