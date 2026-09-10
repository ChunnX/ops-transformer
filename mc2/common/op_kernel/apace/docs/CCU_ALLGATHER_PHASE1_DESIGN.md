# 第一阶段详细设计：x1 AllGather（CCU 通信 + AIV 同步桥）

> 隶属：CCU_ALLGATHER_MATMUL_PLAN.md 第一阶段
> 目标：CCU AllGather 通信跑通 + 精度验证（极简 case，不泛化）
> 同步架构：CCU 通信 + AIV 同步桥（comm_sync：参数桥+唤醒CCU+轮询，无 UB 数据搬运）
> 参考：`02_allgather_add/main.asc`，通信侧整体复用

### 阶段定位

| 阶段 | 步骤 | 目标 | 核验内容 | 同步架构 |
|------|------|------|---------|---------|
| **一（本设计）** | Step 1 | CCU AllGather 通信跑通 | 不崩溃、不卡死（不关注精度） | CCU + AIV 同步桥 |
| **一（本设计）** | Step 2 | 通信精度验证 | recvBuf 拼接顺序与内容正确 | （同上） |
| 二 | — | AllGather + Matmul 串联端到端精度 | C = concat(A) @ B，BF16 rtol/atol=1e-3 | + CrossCoreSetFlag 唤醒 AIC |
| 三 | — | 通算并行流水 + AIC 直轮询 | 两次 allgather + 两次 matmul overlap | CCU + AIC 直轮询（去 AIV 桥） |

> 本设计覆盖第一阶段 Step 1 + Step 2。极简 case：2 卡，M=256, K=256, FP16。

---

## 0. 实现演进记录

### 0.1 编译验证里程碑

| 日期 | commit | 状态 | 说明 |
|------|--------|------|------|
| 2025-08-17 | `f7c94cf15` | ✅ 编译通过 | Host 侧 uint16_t + bin 输出，无 half 软浮点依赖 |
| 2025-08-17 | `f7c94cf15`(amend) | ✅ 运行验证 PASS | 2 卡 AllGather 通信正确，recvBuf 前 65536==1.0 后 65536==2.0，两卡一致 |

### 0.2 关键设计变更（开发中踩坑 → 修正）

**变更 1：verify_copy → comm_sync（去掉 UB 数据搬运）**

原始设计照搬参考样例 `vector_add`（AIV 做参数桥 + 唤醒 + 轮询 + Adds 计算 + UB 搬运）。开发中发现：
- 第一阶段只验证通信，无 AIV 计算（matmul 第二阶段才引入，且走 AIC不经 UB）
- verify_copy 的 UB 搬运引入了 UB 容量风险（recvCount=131072 half=256KB 超 248KB 上限）和 MTE2→MTE3 同步问题

**修正**：AIV kernel 改为 `comm_sync`——只做参数桥 + 唤醒 CCU + 轮询，通信完成后 AIV 直接结束。Host 从 recvBuf 读回数据校验。

**变更 2：computeBuf 去除**

原始设计有 computeBuf（验证拷贝输出目标）。comm_sync 不做数据搬运后，computeBuf 无存在意义，整体去除。Host 直接从 recvBuf 读回数据。

**变更 3：Host 侧 half → uint16_t（编译踩坑）**

原始设计 Host 侧用 `half` 类型填充数据和打印。编译报错：
```
ld.lld error: undefined symbol: __truncsfhf2    __extendhfsf2
```
这两个符号是 LLVM compiler-rt 的 FP16 软浮点函数。`half`（`__fp16`）在 Host 侧的 `static_cast<float>(half)` / `static_cast<half>(float)` / `sizeof(half)` 都会拉入这些符号。参考样例 02_allgather_add 全程用 `float` 不触发此问题。

**尝试方案**（失败）：CMakeLists 加 `--rtlib=compiler-rt` → 链接失败，环境无此库。

**最终方案**（对齐 apace ST）：Host 侧全程 `uint16_t` 承载 FP16 数据：
- `sizeof(uint16_t) == sizeof(half) == 2`，`aclrtMemcpy` 按字节搬运不关心类型语义
- 填数据用 FP16 bit pattern 直接写入（1.0=0x3C00, 2.0=0x4000）
- 不打印数值，结果落 `output/rank{id}/npu_out.bin`，用 Python 校验
- `sendSize`/`recvSize` 用字面量 `2`（不用 `sizeof(half)`）

> **第二阶段预判**：matmul 输出 BF16 也无 `<<` 重载，同样用 `uint16_t` 承载 + bin 输出 + Python 校验。

---

## 1. 极简 case 定义

| 参数 | 值 | 说明 |
|------|-----|------|
| rankSize | 2 | 两卡 |
| M | 256 | 每 rank A 分片行数 |
| K | 256 | A 列数（= B 行数，第一阶段不涉及 B） |
| dtype | FP16 | `sizeof = 2 bytes`（Host 侧用 uint16_t 承载） |
| inputSize | `M*K*2 = 131072 bytes (128KB)` | 每 rank 发送字节数 |
| recvSize | `rankSize * inputSize = 256KB` | AllGather 聚合后总字节 |
| sendBuf | `aclrtMalloc(128KB)` | 本 rank 输入 |
| recvBuf | `aclrtMalloc(256KB)` | AllGather 拼接结果 |

**recvBuf rank-major 布局**（CCU AllGather 后）：
```
recvBuf: [rank0 的 128KB 数据 | rank1 的 128KB 数据]
           offset 0              offset 128KB (= inputSize * rankId)
```

---

## 2. 文件结构

```
apace/examples/ccu_all_gather_matmul/
├── ccu_all_gather_matmul_kernel.asc   # 单文件：CCU kernel + AIV comm_sync + Host 编排 + main
├── CMakeLists.txt                      # 逐字节对齐 02_allgather_add/CMakeLists.txt
├── run.sh                              # 编译+运行+设置 CCU_SCHED 环境变量
└── README.md
```

---

## 3. CCU 通信 kernel（整体复用 02_allgather_add）

### 3.1 复用清单（逐函数）

| 函数 | 是否改动 | 说明 |
|------|---------|------|
| `CcuKernelArgAllGatherMesh1DMem2Mem` | 不变 | CCU kernel 参数结构 |
| `AllGatherMesh1DMem2MemContext` | 不变 | CCU 运行态上下文 |
| `CcuMappedAddrs` | 不变 | 跨流同步地址桥（eventVa[2]+startVa[10]） |
| `CalGoSize` | 不变 | 大块分片循环参数计算 |
| `InitGroupCopyResources` | 不变 | GroupCopy 资源初始化 |
| `GroupCopy` | 不变 | 本地拷贝（经 CCU Buffer 中转） |
| `ExecuteAllGatherTransfer` | 不变 | AllGather 核心：ccu::Write 远端 + GroupCopy 本地 + Event 同步 |
| `CcuAllGatherMesh1DMem2MemKernel` | 不变 | CCU kernel 入口（`__ccu_host__`） |
| `AcquireCcuChannels` | 不变 | 获取 CCU 通信通道 |
| `CreateCcuInsWithEnabledDies` | 不变 | 创建 CCU 实例 |

**结论**：CCU 通信 kernel **逐字节复用**，零改动。

### 3.2 CCU kernel 流程（复用，仅展示流程）

```
1. 解析参数：channels / rankId / rankSize / varHandle / eventHandle
2. 从 channel 获取远端 output/token 变量
3. EventWait(events[0])          ← 阻塞等 AIV 写好参数
4. 从 Variable 槽读取本 rank 参数（vars[0..9]）
5. WriteVariableWithNotify + NotifyWait  ← 与远端 rank 交换 output/token
6. ExecuteAllGatherTransfer():
   - ccu::Write 把本 rank 分片发到所有远端 rank 的 recvBuf 对应位置
   - GroupCopy 本 rank 分片拷到本地 recvBuf
   - EventRecord(event, 1<<rankId) + EventWait(event, totalMask) 同步
7. post-sync (NotifyRecord/NotifyWait)
8. EventRecord(events[1])         ← 通知 AIV 通信完成
```

---

## 4. AIV 同步桥 kernel（comm_sync）

### 4.1 职责

1. **参数桥**：写 taskArgs[0..9] → GM（通信参数送 CCU Variable 槽）
2. **唤醒 CCU**：写 ckeAddr0=1
3. **轮询等待**：读 ckeAddr1==1

**不做数据搬运**（无 UB 分配、无 DataCopy、无 GlobalTensor/LocalTensor）。通信完成后 AIV 直接结束，Host 从 recvBuf 读回数据。

### 4.2 相对参考样例 vector_add 的改动

| 参考样例 vector_add | 本方案 comm_sync | 理由 |
|---------------------|-------------------|------|
| 参数桥 + 唤醒 + 轮询 | 参数桥 + 唤醒 + 轮询 | **完全一致** |
| Adds 计算（Vector） | 删除 | 第一阶段无 AIV 计算 |
| UB 分配（srcLocal + dstLocal） | 删除 | 无计算无搬运，不需 UB |
| DataCopy（GM→UB→GM） | 删除 | 同上 |
| MTE2_V / V_MTE3 流水同步 | 删除 | 无 MTE2/MTE3 操作 |
| 模板参数 `<T, maxCount>` | 删除（非 template） | 无 UB 分配不需模板 |
| dstGm / count 参数 | 删除 | 无数据搬运 |

### 4.3 comm_sync 签名

```cpp
__global__ __vector__ void comm_sync(
    __gm__ void *src,               // sendBuf（参数桥用）
    __gm__ void *recvBuf,           // recvBuf（参数桥用）
    CcuMappedAddrs mappedAddrs,     // 跨流同步地址桥
    uint64_t token,
    uint64_t currentRankSliceInputOffset,   // = 0
    uint64_t currentRankSliceOutputOffset,  // = inputSize * rankId
    uint64_t sliceSize,                     // = inputSize
    uint64_t goSize0, goSize1, goSize2, goSize3)
```

> 参数桥 + 唤醒 + 轮询三段逻辑与参考样例 vector_add L654-689 **逐字节一致**。

---

## 5. Host 编排（AllGatherCustom 入口）

### 5.1 相对 02_allgather_add 的改动点

1. 函数名 `AllGatherAddCustom` → `AllGatherCustom`
2. **去掉 `computeBuf` 参数**（无验证拷贝输出）
3. dtype 校验：`FP32 only` → `FP16 + FP32` 白名单
4. sendCount 校验：移除 `!= 256` 硬校验
5. AIV kernel 调用 `vector_add<float, ...>` → `comm_sync`（无模板参数）
6. `taskArgs` 组装逻辑**不变**（参数桥字段含义不变）

### 5.2 流程

```
1. 参数校验（FP16/FP32 白名单 + sendCount==0 早退）
2. rankId/rankSize 查询
3. rankSize==1 退化处理（HcommLocalCopyOnThread）
4. AcquireCcuChannels → CreateCcuInsWithEnabledDies → HcclCommAssignCcuIns
5. 分配 Variable(10) + Event(2)，获取 GM 地址 → mappedAddrs
6. 组装 kernelArg
7. HcommCcuGetMemToken
8. CalGoSize
9. 组装 taskArgs[0..9]（参数桥）
10. 下发 CCU kernel：CcuAllGatherMesh1DMem2MemKernel<<<schd, insHandle, streamCcu>>>
11. 下发 AIV kernel：comm_sync<<<1, nullptr, streamAiv>>>(...)
```

> 步骤 1-10 与参考样例完全一致（除校验放开），步骤 11 的 kernel 名和参数变了。

---

## 6. main() / Sample()（复用 02_allgather_add thread 模式）

### 6.1 方案

采用 `std::thread` 多线程模式，与参考样例完全对齐。每设备一个线程，同进程内 rootInfo 指针共享。

### 6.2 相对 02_allgather_add 的改动点

1. **sendCount**：固定 256 → `M*K = 65536`
2. **数据类型**：Host 侧 `float` → `uint16_t`（承载 FP16，避免 half 软浮点符号）
3. **输入填充**：`float(device)+1` → FP16 bit pattern（rank0=0x3C00=1.0, rank1=0x4000=2.0）
4. **size 计算**：`sizeof(float)` → 字面量 `2`
5. **去掉 computeBuf**：无验证拷贝输出
6. **结果输出**：进程内打印数值 → 落地 `output/rank{id}/npu_out.bin`（不打印数值）
7. **调用**：`AllGatherAddCustom` → `AllGatherCustom`

### 6.3 Host 侧数据填充（uint16_t + FP16 bit pattern）

```cpp
uint16_t *tmpHostBuff = static_cast<uint16_t *>(hostBuf);
// FP16 bit pattern: 1.0=0x3C00, 2.0=0x4000（rank0 填 1.0, rank1 填 2.0）
uint16_t fillBits = (device == 0) ? 0x3C00 : 0x4000;
for (uint16_t *p = tmpHostBuff; p < tmpHostBuff + sendCount; ++p) {
    *p = fillBits;
}
```

### 6.4 结果输出（bin 文件，不打印数值）

```cpp
ACL_CHECK(aclrtMemcpy(resultBuff, recvSize, recvBuf, recvSize, ACL_MEMCPY_DEVICE_TO_HOST));
std::string outputDir = "output/" + std::to_string(device);
system(("mkdir -p " + outputDir).c_str());
std::ofstream ofs(outputDir + "/npu_out.bin", std::ios::binary);
ofs.write(static_cast<const char *>(resultBuff), recvSize);
```

---

## 7. 测试与验证

### 7.1 验证方式

结果落地 `output/rank{id}/npu_out.bin`，用 Python 校验：

```python
import numpy as np
M, K, RANK_NUM = 256, 256, 2
recvCount = M * K * RANK_NUM  # 131072

for rank in range(RANK_NUM):
    data = np.fromfile(f"output/{rank}/npu_out.bin", dtype=np.float16)
    assert np.all(data[:M*K] == 1.0), f"rank{rank} 前半不等于 1.0"
    assert np.all(data[M*K:] == 2.0), f"rank{rank} 后半不等于 2.0"
print("[PASS]")
```

### 7.2 Step 1 / Step 2 验收

**Step 1（跑通）**：
- 程序正常退出，不崩溃、不卡死
- `output/0/npu_out.bin` 和 `output/1/npu_out.bin` 生成（各 256KB）
- ✅ 验证通过（2025-08-17）

**Step 2（精度）**：
- `output/{rank}/npu_out.bin` 前 65536 元素 == 1.0（rank0 数据）
- `output/{rank}/npu_out.bin` 后 65536 元素 == 2.0（rank1 数据）
- 两卡 bin 文件内容相同
- ✅ 验证通过（2025-08-17）

### 7.3 run.sh

```bash
#!/bin/bash
set -e
source ${ASCEND_HOME_PATH}/set_env.sh
export HCCL_OP_EXPANSION_MODE=CCU_SCHED
mkdir -p build && cd build
cmake .. && make -j
./ccu_all_gather_matmul
```

---

## 8. CMake 构建（逐字节对齐 02_allgather_add）

第一阶段不含 matmul，无需 blaze/tensor_api 依赖。CMake 逐字节复用参考样例，仅改可执行文件名和源文件名。

- include 路径 7 条
- 链接库 6 个：hcomm, asccomm_ccu_dataplane, c_sec, ascendcl, acl_rt, ascendc_runtime
- 编译选项：-Werror, -fstack-protector-strong, -fPIE, -O2
- 架构：dav-3510
- **无额外链接**（Host 侧 uint16_t 不依赖 FP16 软浮点库）

> 第二阶段引入 Blaze 时再追加 blaze/tensor_api include 和必要链接库。

---

## 9. 风险点

| 风险 | 影响 | 缓解 | 状态 |
|------|------|------|------|
| `HCCL_OP_EXPANSION_MODE=CCU_SCHED` 未设 | CCU kernel 不执行 | run.sh 中 export | ✅ 已处理 |
| Host 侧 half 软浮点符号缺失 | 编译报错 | uint16_t 承载 + bin 输出 | ✅ 已解决 |
| UB 容量超限 | 编译/运行失败 | comm_sync 不做 UB 搬运 | ✅ 已规避 |
| `HcommCcuGetMemToken` 地址/大小不匹配 | 通信数据错误 | inputSize 按字节传（M*K*2） | ✅ 已验证 |
| AIV 轮询不退出 | 程序卡死 | ckeAddr1==1 能读到 | ✅ 已验证 |
