# 第三阶段详细设计：通算并行流水 + AIC 直轮询

> 隶属：CCU_ALLGATHER_MATMUL_PLAN.md 第三阶段
> 目标：commTurn 次 AllGather + commTurn 次 matmul 通算 overlap，AIC 直轮询去 AIV 同步桥
> 方案：Host 仅 1 次 CCU launch，CCU kernel 内部循环 commTurn 次 AllGather
> 前置条件：第二阶段已验证通过（commit `24f13d937`，2025-08-17，2 卡 BF16 精度 PASS）

---

## 1. 核心思路

Host 侧只 launch **1 次** CCU kernel + **1 次** MIX kernel。CCU kernel 内部循环 commTurn 次 AllGather，每轮通过独立 done Event 通知 AIC。AIC 一次性预写所有 round 参数，循环 commTurn 次（轮询 done → matmul）。

### 与第二阶段对比

| | 第二阶段 | 第三阶段 |
|---|---|---|
| CCU launch 次数 | 1 次（全量 AllGather） | 1 次（kernel 内循环 commTurn 次） |
| CCU kernel | 单次 AllGather | 循环 commTurn 次 AllGather |
| 同步桥 | AIV 写参数桥+唤醒+轮询 | AIC 直轮询（去 AIV） |
| matmul | 1 次（全量 M） | commTurn 次（分 tile，与通信 overlap） |
| CKE Event | 2 个（wake/done） | commTurn+1 个（1 wake + commTurn done） |
| Variable | 1 组（10 个） | commTurn 组（各 10 个，AIC 一次性预写） |
| 通算并行 | 无（串联） | 有（round d+1 通信与 round d 计算并行） |

---

## 2. Case 定义

| 参数 | 值 | 说明 |
|---|---|---|
| rankSize | 2 | 两卡 |
| M | 512 | 每 rank A 分片行数 |
| tileM | 256 | 每 round AllGather 的行数 |
| commTurn | 2 | AllGather 分 2 round（= M / tileM） |
| K | 256 | |
| N | 256 | |
| dtype A/B | FP16 | |
| dtype C | BF16 | |

每 rank sendBuf = 512×256 FP16 = 256KB，分 2 tile 各 128KB。commTurn < 16（MAX_COMMTURN 上限）。

---

## 3. 双流时序

```
streamCcu:    [CCU kernel launch（1 次，内部循环 commTurn 次）]
                 │ round 0: EventWait(events[0]) → 读 vars[0..9] → AllGather → EventRecord(events[1])
                 │ round 1: （串行自动进入）→ 读 vars[10..19] → AllGather → EventRecord(events[2])
                 │ ...

streamAivAic: [MIX kernel launch]
                 AIC block 0: 一次性预写 commTurn 组参数桥 + 写 wake(events[0]=1)
                 AIC 所有核:
                   for (d = 0; d < commTurn; d++):
                     轮询 events[d+1]==1（CCU round d done）
                     matmul tile d  ←── 与 CCU round d+1 通信并行
```

### overlap 点

AIC 一次性预写所有 round 参数 + 唤醒 CCU round 0 后，CCU 在 kernel 内部串行循环：round d 完成 EventRecord 后自动进入 round d+1（不需 EventWait）。AIC 轮询到 round d done 后开始 matmul tile d，此时 CCU 正在 round d+1 通信——两者并行。

---

## 4. CKE Event（commTurn + 1）

### 4.1 为何 2 个 CKE 不够

所有 round 共用同一个 done 地址（events[1]）时，matmul 耗时很短的 AIC 算完 tile d 去轮询 events[1]，读到的可能是 round d 残留的 1（CCU round d+1 还没完成通信），**数据未就绪就开始计算**。

### 4.2 分配

- **wake 只需 1 个**（events[0]）：AIC 写 1 唤醒 CCU round 0。CCU kernel 内部串行循环，round d 完成后自动进入 round d+1，不需要再唤醒。
- **done 需 commTurn 个**（events[1..commTurn]）：每 round 完成后 CCU 写独立的 done 地址，AIC 轮询当前 round 的 done，不会读到残留值。无需重置。

| Event | 地址 | 用途 | 谁写 | 谁读 |
|---|---|---|---|---|
| events[0] | eventVa[0] | wake（AIC→CCU，仅 round 0） | AIC 写 1 | CCU EventWait（仅 round 0） |
| events[1] | eventVa[1] | round0 done | CCU EventRecord | AIC 轮询 |
| events[2] | eventVa[2] | round1 done | CCU EventRecord | AIC 轮询 |
| ... | ... | ... | ... | ... |

```cpp
constexpr uint32_t CKE_NUM = commTurn + 1;  // 1 wake + commTurn done
```

### 4.3 CCU kernel Event 索引

```cpp
for (uint32_t d = 0; d < commTurn; d++) {
    if (d == 0) ccu::EventWait(events[0]);      // 仅 round 0 等 AIC 唤醒
    // ... 读 vars, AllGather ...
    ccu::EventRecord(events[d + 1]);             // 通知 AIC round d done
}
```

Host 分配 1 组 Event（commTurn+1 个），1 个 eventHandle 管理全部。

---

## 5. CCU kernel 改动

### 5.1 kernelArg 新增字段

```cpp
struct CcuKernelArgAllGatherMesh1DMem2Mem {
    // ... 原有字段不变 ...
    uint32_t commTurn;  // CCU kernel 内部循环次数
};
```

varHandle 为 1 个，管理 commTurn × HCOMM_HOST_TASK_ARG_NUM 个 Variable。CCU kernel 内部按 round 偏移读取。

### 5.2 kernel 主体加循环

原逻辑（第二阶段）：
```
EventWait(events[0])
读 vars[0..9]
WriteVariableWithNotify / NotifyWait
ExecuteAllGatherTransfer
NotifyRecord / NotifyWait
EventRecord(events[1])
```

改为：
```
for (d = 0; d < commTurn; d++) {
    if (d == 0) EventWait(events[0])        // 仅 round 0 等 AIC 唤醒
    读 vars[d*10 .. d*10+9]                  // 每 round 读该 round 对应的 Variable 组
    WriteVariableWithNotify / NotifyWait
    ExecuteAllGatherTransfer
    NotifyRecord / NotifyWait
    EventRecord(events[d+1])                 // 通知 AIC round d done
}
```

---

## 6. AIC 直轮询设计（去 AIV 同步桥）

### 6.1 AIV 分支

```cpp
if ASCEND_IS_AIV {
    return;  // AIV 空跑，MIX_AIC_1_1 配对要求
}
```

### 6.2 AIC 分支（commTurn 通用）

```cpp
if ASCEND_IS_AIC {
    // block 0 一次性预写所有 round 的参数桥 + 唤醒
    if (GetBlockIdx() == 0) {
        for (uint32_t d = 0; d < commTurn; d++) {
            WriteTaskArgs(vars[d*10], sendBuf, recvBuf, token,
                          inputOffset[d], outputOffset[d], sliceSize, goSize...);
        }
        DataSyncBarrier();
        WriteEvent(eventVa[0], 1);   // 唤醒 CCU round 0（唯一 wake）
        DataSyncBarrier();
    }

    // 循环：纯轮询 + matmul，不插任何 Variable 写入，AIC 全部用于计算
    for (uint32_t d = 0; d < commTurn; d++) {
        while (ReadEvent(eventVa[d+1]) != 1) { Nop<800>(); }
        MatmulProcess(recvBuf + d*tileABytes, weightBuf, outputBuf + d*tileCBytes, tilingData);
    }
}
```

### 6.3 参数桥写入

AIC block 0 一次性预写 commTurn 组 Variable（每组 10 个）。每组 vars[3](inputOffset) / vars[4](outputOffset) / vars[5](sliceSize) 不同，其余各组相同。预写完成后循环里不再写 Variable。

### 6.4 inputOffset/outputOffset

Host 拷贝 inputOffsets[commTurn]/outputOffsets[commTurn] 到 device GM，AIC 读取（commTurn 通用，参数数量不固定）。

```
inputOffset[d]  = d * sliceSize
outputOffset[d] = d * rankSize * sliceSize + rankId * sliceSize
```

recvBuf 布局：[r0t0 | r1t0 | r0t1 | r1t1 | ...]

---

## 7. Host 编排

### 7.1 改动点

1. CCU kernel launch 仍为 1 次（kernel 内部循环 commTurn 次）
2. kernelArg 新增 `commTurn` 字段
3. 分配 1 组 Variable（commTurn × 10 个）+ 1 组 Event（commTurn + 1 个）
4. 构造 inputOffsets/outputOffsets 数组，拷贝到 device GM
5. MIX kernel 签名增加 commTurn + d_inputOffsets/d_outputOffsets

### 7.2 Host 流程

```
1. PrepareAllGatherMatmul：
   - 分配 channel/ins
   - 分配 Variable（commTurn × 10 个，1 个 varHandle）
   - 分配 Event（commTurn + 1 个，1 个 eventHandle）
   - 构造 inputOffsets[commTurn]/outputOffsets[commTurn]
   - 拷贝到 device GM
   - kernelArg.commTurn = commTurn
2. CCU kernel launch（1 次，streamCcu）
3. MIX kernel launch（1 次，streamAivAic）
```

---

## 8. matmul tiling

每 round matmul 的 M = rankSize * tileM = 2 * 256 = 512。

```cpp
MatmulTilingData tilingData{};
tilingData.m = rankSize * tileM;  // 512
tilingData.n = N;                 // 256
tilingData.k = K;                 // 256
// baseM/baseN/baseK/mL1/nL1/kL1 同第二阶段
```

所有 round tiling 相同，只是 A/C GM 地址偏移不同。

---

## 9. 数据与校验

### 9.1 数据填充

- **A**（sendBuf）：FP16，每 rank 前 256 行 1.0(0x3C00) 后 256 行 2.0(0x4000)
- **B**（weightBuf）：FP16，全 1.0(0x3C00)
- **C**（outputBuf）：BF16

### 9.2 Golden

C = A @ B，B 全 1.0 时 C[i][j] = K × A[i][0]

- tile0 行（A=1.0）：C = 256.0
- tile1 行（A=2.0）：C = 512.0

AllGather 后 A = [rank0 tile0 | rank1 tile0 | rank0 tile1 | rank1 tile1]，C 前半 256.0 后半 512.0。

### 9.3 Python 校验

```python
import numpy as np
raw = np.fromfile("output/0/npu_out.bin", dtype=np.uint16)
c = np.frombuffer((raw.astype(np.uint32) << 16).tobytes(), dtype=np.float32)
assert np.all(c[:131072] == 256.0)   # tile0（rankSize*tileM*N = 2*256*256 = 131072 元素）
assert np.all(c[131072:] == 512.0)   # tile1
```

---

## 10. 文件改动

| 文件 | 改动 |
|---|---|
| `ccu_all_gather_matmul_kernel.asc` | CCU kernel 加 commTurn 循环 + kernelArg 加 commTurn + CcuMappedAddrs 单组数组化 + AllGatherMatmulCustom 1 次 CCU launch |
| `matmul_fusion_kernel.h` | AicDirectPollProcess commTurn 通用循环（预写 + 轮询 + matmul） |
| `host_launcher.h` | PrepareAllGatherMatmul 分配 commTurn×10 Variable + commTurn+1 Event + inputOffsets/outputOffsets |
| `sample_main.cpp` | M=512, tileM=256, 数据填充调整 |

---

## 11. 风险

| 编号 | 风险 | 状态 |
|---|---|---|
| P9 | CKE Event 扩展（commTurn+1） | ✅ 已闭环（1 wake + commTurn done，无需重置） |
| P10 | AIC 写 CCU Variable 地址空间 | ✅ 上板验证通过 |
| P11 | AIC 用 WriteGmByPassDCache/ReadGmByPassDCache/Nop | ✅ 上板验证通过 |
| P14 | CCU kernel 循环内 channel NotifyRecord/NotifyWait 多次调用 | ✅ 上板验证通过 |

---

## 12. 上板验证记录（2025-08-18）

### 12.1 精度验证结果

2 卡（rankSize=2），M=512，tileM=256，commTurn=2，K=N=256，FP16→BF16。

- tile0（前 131072 元素）：全 256.0 ✅
- tile1（后 131072 元素）：全 512.0 ✅

### 12.2 编译问题修复

| 问题 | 根因 | 修复 | 文件 |
|---|---|---|---|
| `unknown type name 'CcuMappedAddrs'` | include 顺序前向引用 | CcuMappedAddrs 前移到 include 之前 | `.asc` |
| `no matching function for call to MatmulProcess` | `GM_ADDR` 与 `__gm__ void*` 类型不匹配 | MatmulProcess 参数改 `__gm__ void*`，内部两步 cast | `matmul_fusion_kernel.h` |
| `ACLCHECK 返回 aclError 不能转 HcclResult` | PrepareAllGatherMatmul 返回 HcclResult，ACLCHECK 返回 aclError | 手动检查 aclError + 返回 HCCL_E_INTERNAL | `host_launcher.h` |

### 12.3 运行时精度问题修复

#### tile1 本 rank 数据缺失（全 0）

**现象**：tile0 全 PASS；tile1 每张卡都是本 rank 数据为 0、peer rank 数据正确：
- 0 卡 tile1：前 65536=0（rank0 缺失），后 65536=512（rank1 正确）
- 1 卡 tile1：前 65536=512（rank0 正确），后 65536=0（rank1 缺失）

**根因**：`InitGroupCopyResources` 中 Loop Entity 的 lambda 按值捕获 `loopSrc`/`loopDst`/`loopLen`。原代码仅在首次调用时创建 Loop Entity（`if (!IsLoopEntityRegistered)`），round 1 复用 round 0 的地址快照，导致本地拷贝 `GroupCopy` 写到 round 0 的位置而非 round 1。

- `GroupCopy`（本地拷贝）受影响 → 本 rank tile1 数据未写到正确位置 → 0
- `ccu::Write`（远程传输）不受 lambda 影响 → peer rank tile1 数据正常 → 512

**修复**：去掉 `if (!IsLoopEntityRegistered)` 条件，每次调用都 erase 旧 Loop Entity 并重新创建，确保 lambda 捕获当前 round 的地址。

**commit**：`377d17e8b`

### 12.4 P10/P11/P14 验证结论

| 风险 | 结论 |
|---|---|
| P10 | AIC 写 CCU Variable 地址空间 ✅ 正常工作，AIC block 0 预写 commTurn 组参数桥成功 |
| P11 | AIC WriteGmByPassDCache/ReadGmByPassDCache/Nop ✅ 轮询 CKE done 正常，通信完成后 AIC 正确感知 |
| P14 | CCU kernel 循环内 channel NotifyRecord/NotifyWait ✅ 多次调用正常，但 GroupCopy 的 Loop Entity 需每轮重建（lambda 捕获问题） |
