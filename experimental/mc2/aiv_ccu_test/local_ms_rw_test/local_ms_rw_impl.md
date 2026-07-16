# 本地读写 MS 测试场景 —— 实现方案

> 对应 `设计文档.md` 3.1 节"本地读写 MS"，采用 `<<<>>>` 直调方式，对标 `quant_all_reduce` 工程结构。

---

## 一、需求分析

### 1.1 目标

通过 `<<<>>>` 直调，在 Kernel 侧将数据写入本地 MemorySlice（MS），测量写带宽。数据量 4KB → 8KB → 16KB → ... → 6MB。

### 1.2 关键背景

- 每个 CCUA 的 MS 空间 = 1.5MB，4 个 CCUA 共 6MB
- 交织粒度 = 4KB（1536 个 4KB 块，编号 MS0 ~ MS1535）
- 交织规则：每 8 个 4KB 块为一组，组内连续写同一 CCUA，组间轮转 CCUA0→1→2→3

### 1.3 设计文档要求摘要

| 项 | 要求 |
|----|------|
| Host 侧 | 通过 `GetTlvHandle` + `HrtGetCcuMemInfo` 获取 4 个 MS 地址（MSAddr0~3） |
| Kernel 入口 | `Aivccu_COMM<<<>>>(mode0, mode1, datasize, MSAddr0, MSAddr1, MSAddr2, MSAddr3, timeaddr)` |
| mode0 | 读/写/读写混合（当前先实现写） |
| mode1 | 交织/不交织 |
| 不交织场景 | 按 CCUA0→1→2→3 顺序连续写 |
| 交织场景 | 按 MS 地址交织算法，先 CCUA0(MS0~MS7)，再 CCUA1(MS8~MS15)... |
| 计时 | 打点获取 starttime/endtime，计算耗时 |
| 数据量 | 4KB → 8KB → 16KB → ... → 6MB |

---

## 二、文件规划

| 文件 | 作用 | 对标现有文件 |
|------|------|-------------|
| `local_ms_rw.h` | `<<<>>>` 调用函数声明 | `quant_all_reduce.h` |
| `local_ms_rw.cpp` | Kernel 入口 + `<<<>>>` 封装 | `quant_all_reduce.cpp` |
| `kernel/local_ms_rw_kernel.h` | Kernel 核心逻辑（DataCopy/交织/计时） | `kernel/quant_all_reduce_mte_one_shot.h` |
| `test_local_ms_rw.cpp` | Host 侧测试代码 | `test_quant_all_reduce.cpp` |
| `run_local_ms_rw.sh` | 编译运行脚本 | `run.sh` / `start.sh` |
| `CMakeLists.txt` | 修改：新增 kernel 源文件 + test 目标 | 现有 CMakeLists.txt |

---

## 三、Host 侧设计（`test_local_ms_rw.cpp`）

### 3.1 整体流程

```
aclInit → aclrtSetDevice → aclrtCreateContext → aclrtCreateStream
  │
  ▼
GetTlvHandle(devLogicId)           ← 获取 TLV 句柄
  │
  ▼
HrtGetCcuMemInfo(tlvHandle, udieIdx=0, msBitmap, memInfoList, count=4)
  → 提取 4 个 MS 地址: MSAddr0(CCUA0), MSAddr1(CCUA1), MSAddr2(CCUA2), MSAddr3(CCUA3)
  │
  ▼
aclrtMalloc(srcDataGM, dataSize)   ← 分配源数据 GM，填充测试数据
aclrtMalloc(timeGM, 16 bytes)      ← 分配计时结果 GM (start_ns + end_ns)
  │
  ▼
aivccu_comm_demo(mode0, mode1, dataSize, MSAddr0..3, timeAddr, blockDim, stream)
  │
  ▼
aclrtSynchronizeStream → 读取 timeGM → 计算耗时 → 打印带宽
  │
  ▼
释放资源
```

### 3.2 MS 地址获取

```cpp
#include "hccp_tlv_hdc_manager.h"  // HccpTlvHdcManager
#include "orion_adapter_hccp.h"     // CcuMemInfo, HrtGetCcuMemInfo, CCU_MEMTYPE_MS_BLOCK0~3

// 1. 获取 tlvHandle
auto tlvHandle = Hccl::HccpTlvHdcManager::GetInstance().GetTlvHandle(rankId);

// 2. 构造 MS bitmap（仅申请 4 个 MS_BLOCK）
uint64_t msBitmap = (1ULL << 32) | (1ULL << 33) | (1ULL << 34) | (1ULL << 35);

// 3. 查询 MS 地址
Hccl::CcuMemInfo memInfoList[4];
Hccl::HrtGetCcuMemInfo(tlvHandle, 0 /*udieIdx*/, msBitmap, memInfoList, 4);

// 4. 提取地址
uint64_t MSAddr[4] = {memInfoList[0].memVa, memInfoList[1].memVa,
                      memInfoList[2].memVa, memInfoList[3].memVa};
```

> **说明**：`memInfoList` 返回顺序与 bitmap 中 bit 位序一致，即 MS_BLOCK0→1→2→3，对应 CCUA0→1→2→3。

### 3.3 与 quant_all_reduce 的区别

| 方面 | quant_all_reduce | local_ms_rw |
|------|-----------------|-------------|
| 通信域 | 需要 HcclCommInitClusterInfo | **需要**，GetTlvHandle 依赖 HDC 基础设施 |
| TilingData | 需要 mc2Context + Tiling | **不需要**，直接传参数 |
| 输入数据 | 从文件读取 bin | Host 生成测试 pattern |
| 多卡 | ranksize 多卡 | **单卡**即可 |
| 依赖 | hccl, hcomm | hccl(仅 GetTlvHandle), hcomm |

### 3.4 命令行参数

```
./test_local_ms_rw \
    --rank_id 0 \
    --mode0 0          \   # 0=写, 1=读, 2=读写混合
    --mode1 1          \   # 0=不交织, 1=交织
    --data_size 4194304    # 数据量(字节)，4KB~6MB
```

---

## 四、Kernel 侧设计

### 4.1 入口签名

```cpp
extern "C" __global__ __aicore__ void Aivccu_COMM(
    uint32_t mode0,       // 0=写, 1=读, 2=读写混合（当前先实现写）
    uint32_t mode1,       // 0=不交织, 1=交织
    uint64_t dataSize,    // 数据量（字节），4KB~6MB
    GM_ADDR msAddr0,      // CCUA0 MS 地址
    GM_ADDR msAddr1,      // CCUA1 MS 地址
    GM_ADDR msAddr2,      // CCUA2 MS 地址
    GM_ADDR msAddr3,      // CCUA3 MS 地址
    GM_ADDR timeAddr      // 计时结果 GM（start_ns + end_ns）
);
```

### 4.2 核心逻辑流程

```
1. KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)
2. 获取 aivId = GetBlockIdx()
3. 分块计算：
     totalBlocks   = dataSize / 4096
     blocksPerCore = totalBlocks / blockDim
     myStartBlock  = aivId * blocksPerCore
     myBlocks      = (aivId == blockDim-1) ? totalBlocks - myStartBlock : blocksPerCore
4. 初始化 UB Buffer（4KB tile，32B 对齐）
5. 写模式：
   a. Duplicate(dataLocal, pattern)     ← UB 填充测试数据
   b. startTime = GetTimeStamp()
   c. for each myBlock:
        - 计算目标地址（交织/非交织）
        - DataCopy(msGlobalTensor[offset], dataLocal, 4096)
   d. PipeBarrier<PIPE_ALL>()
   e. endTime = GetTimeStamp()
   f. core 0 将 startTime/endTime 写入 timeAddr
```

### 4.3 MS 地址交织算法

```cpp
constexpr uint32_t BLOCK_4K = 4096;
constexpr uint32_t GROUP_SIZE = 8;         // 8 blocks per group
constexpr uint32_t CCUA_NUM = 4;
constexpr uint32_t BLOCKS_PER_CCUA = 384;  // 1.5MB / 4KB = 384

// === 交织模式 ===
// 每 8 个 4KB 块为一组，组间轮转 CCUA0→1→2→3
__aicore__ inline void GetInterleavedAddr(uint64_t blockIdx, const GM_ADDR msAddrs[4],
                                          uint32_t& ccuaIdx, uint64_t& offset)
{
    uint32_t group = blockIdx / GROUP_SIZE;        // 第几个 8-block 组
    ccuaIdx = group % CCUA_NUM;                     // CCUA 轮转
    uint32_t localGroup = group / CCUA_NUM;         // 在本 CCUA 内的第几组
    uint32_t posInGroup = blockIdx % GROUP_SIZE;    // 组内位置
    offset = (localGroup * GROUP_SIZE + posInGroup) * BLOCK_4K;
}

// === 非交织模式 ===
// 连续写满 CCUA0(1.5MB) 再写 CCUA1，以此类推
__aicore__ inline void GetSequentialAddr(uint64_t blockIdx, const GM_ADDR msAddrs[4],
                                         uint32_t& ccuaIdx, uint64_t& offset)
{
    ccuaIdx = blockIdx / BLOCKS_PER_CCUA;
    offset = (blockIdx % BLOCKS_PER_CCUA) * BLOCK_4K;
}
```

### 4.4 交织映射示例

| 4KB 块序号 | 交织→CCUA | 交织→偏移 | 非交织→CCUA | 非交织→偏移 |
|-----------|----------|----------|------------|------------|
| 0~7 | CCUA0 | 0~28KB | CCUA0 | 0~28KB |
| 8~15 | CCUA1 | 0~28KB | CCUA0 | 32~60KB |
| 16~23 | CCUA2 | 0~28KB | CCUA0 | 64~92KB |
| 24~31 | CCUA3 | 0~28KB | CCUA0 | 96~124KB |
| 32~39 | CCUA0 | 32~60KB | CCUA0 | 128~156KB |
| ... | ... | ... | ... | ... |
| 384~391 | CCUA0 | 1536~1564KB | CCUA1 | 0~28KB |

### 4.5 计时方案

采用设备侧 `GetSystemCycle()` 打点，频率 1000MHz，1 cycle = 1 ns，时间差即为纳秒数。

```cpp
uint64_t startCycle = GetSystemCycle();
// ... DataCopy 循环 ...
PipeBarrier<PIPE_ALL>();
uint64_t endCycle = GetSystemCycle();
uint64_t elapsedNs = endCycle - startCycle;  // 直接就是纳秒

// core 0 将 startCycle / endCycle 写入 timeAddr
if (GetBlockIdx() == 0) {
    timeGm_.SetValue(0, startCycle);
    timeGm_.SetValue(1, endCycle);
}
```

Host 侧读取 `timeAddr` 后计算：

```cpp
uint64_t startCycle, endCycle;
aclrtMemcpy(&startCycle, 8, timeGM, 8, ACL_MEMCPY_DEVICE_TO_HOST);
aclrtMemcpy(&endCycle, 8, timeGM + 8, 8, ACL_MEMCPY_DEVICE_TO_HOST);
uint64_t elapsedNs = endCycle - startCycle;       // 纳秒
double bandwidthGBs = (double)dataSize / elapsedNs; // GB/s
```

### 4.6 UB Buffer 规划

```
UB (248KB, DAV_3510/950) 分配：
  dataBuf (TBuf<VECCALC>): 4KB   ← DataCopy 的源/目的缓冲区
  总占用: 4KB（极小，无 Double Buffer 需求）
```

### 4.7 Kernel 核心代码骨架

```cpp
template <typename T>
class LocalMsRw {
public:
    __aicore__ inline void Init(uint32_t mode0, uint32_t mode1, uint64_t dataSize,
                                GM_ADDR msAddrs[4], GM_ADDR timeAddr, TPipe* pipe)
    {
        mode0_ = mode0;
        mode1_ = mode1;
        dataSize_ = dataSize;
        timeGm_.SetGlobalBuffer((__gm__ uint64_t*)timeAddr);
        for (int i = 0; i < 4; i++) {
            msGm_[i].SetGlobalBuffer((__gm__ T*)msAddrs[i]);
        }
        pipe->InitBuffer(dataBuf_, BLOCK_4K);
        totalBlocks_ = dataSize / BLOCK_4K;
        uint32_t aivId = GetBlockIdx();
        uint32_t blockDim = GetBlockNum();
        blocksPerCore_ = totalBlocks_ / blockDim;
        myStartBlock_ = aivId * blocksPerCore_;
        myBlocks_ = (aivId == blockDim - 1) ? (totalBlocks_ - myStartBlock_) : blocksPerCore_;
    }

    __aicore__ inline void Process()
    {
        LocalTensor<T> dataLocal = dataBuf_.Get<T>();
        Duplicate<T>(dataLocal, (T)1, BLOCK_4K / sizeof(T));  // 填充测试数据

        uint64_t startCycle = GetSystemCycle();

        for (uint32_t i = 0; i < myBlocks_; i++) {
            uint64_t blockIdx = myStartBlock_ + i;
            uint32_t ccuaIdx;
            uint64_t offset;
            if (mode1_ == 1) {
                GetInterleavedAddr(blockIdx, ccuaIdx, offset);
            } else {
                GetSequentialAddr(blockIdx, ccuaIdx, offset);
            }
            DataCopy(msGm_[ccuaIdx][offset / sizeof(T)], dataLocal, BLOCK_4K / sizeof(T));
        }

        PipeBarrier<PIPE_ALL>();
        uint64_t endCycle = GetSystemCycle();

        if (GetBlockIdx() == 0) {
            timeGm_.SetValue(0, startCycle);
            timeGm_.SetValue(1, endCycle);
        }
    }

private:
    uint32_t mode0_, mode1_;
    uint64_t dataSize_;
    GlobalTensor<T> msGm_[4];
    GlobalTensor<uint64_t> timeGm_;
    TBuf<TPosition::VECCALC> dataBuf_;
    uint32_t totalBlocks_, blocksPerCore_, myStartBlock_, myBlocks_;
};
```

---

## 五、CMakeLists.txt 修改

```cmake
# 新增 kernel 源文件
file(GLOB KERNEL_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/quant_all_reduce.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/local_ms_rw.cpp        # ← 新增
)

# 新增 test 目标
add_executable(test_local_ms_rw
    ${CMAKE_CURRENT_SOURCE_DIR}/test_local_ms_rw.cpp
)
target_compile_options(test_local_ms_rw PRIVATE
    -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall -Werror
)
target_compile_definitions(test_local_ms_rw PRIVATE
    SOC_VERSION="${SOC_VERSION}"
)
target_include_directories(test_local_ms_rw PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${HCOMM_SRC_PATH}                                # ← 新增：hcomm src 内部头文件
)
target_link_libraries(test_local_ms_rw PRIVATE
    host_intf_pub
    ascendc_kernels
    tiling_api
    platform
    pthread
    hccl
    hcomm
)
install(TARGETS test_local_ms_rw
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
```

---

## 六、数据量测试矩阵

| dataSize | 4KB 块数 | 交织覆盖 CCUA | 非交织覆盖 CCUA | 单核迭代次数(64核) |
|----------|---------|-------------|---------------|------------------|
| 4KB | 1 | CCUA0 | CCUA0 | 1 |
| 8KB | 2 | CCUA0 | CCUA0 | 1 |
| 16KB | 4 | CCUA0 | CCUA0 | 1 |
| 32KB | 8 | CCUA0 | CCUA0 | 1 |
| 64KB | 16 | CCUA0,1 | CCUA0 | 1 |
| 128KB | 32 | CCUA0,1,2,3 | CCUA0 | 1 |
| 256KB | 64 | CCUA0,1,2,3 | CCUA0 | 1 |
| 512KB | 128 | CCUA0,1,2,3 | CCUA0 | 2 |
| 1MB | 256 | CCUA0,1,2,3 | CCUA0 | 4 |
| 2MB | 512 | CCUA0,1,2,3 | CCUA0 | 8 |
| 4MB | 1024 | CCUA0,1,2,3 | CCUA0,1,2,3 | 16 |
| 6MB | 1536 | CCUA0,1,2,3 | CCUA0,1,2,3 | 24 |

---

## 七、关键风险与待确认项

| 风险点 | 说明 | 应对 |
|--------|------|------|
| **MS 地址可访问性** | MS 地址是 CCU VA，需确认 AIV 可通过 GlobalTensor 直接 DataCopy | 先写最小用例（4KB）验证 |
| **hcomm 头文件路径** | `hccp_tlv_hdc_manager.h` 和 `orion_adapter_hccp.h` 在 hcomm src 内部，非公开 include | 需在 CMakeLists 中添加 hcomm src 路径，或使用 `extern "C"` 声明 |
| **DataCopy 到 MS** | MS 地址非普通 GM，可能需要特殊权限或映射 | 参考 quant_all_reduce 中对 Win 区地址的 DataCopy 用法（同为 VA 地址） |
| **32B 对齐** | 4KB = 4096B，天然 32B 对齐 | 无风险 |
| **GetBlockNum 可用性** | 需确认 ascend950pr 上获取总核数的 API | 若不可用，Host 侧传入 blockDim |

---

## 八、实现顺序

1. **先验证 Host 侧 MS 地址获取**：写最小 Host 程序调用 `GetTlvHandle` + `HrtGetCcuMemInfo`，打印 4 个 MS 地址和大小
2. **再验证 Kernel 侧 DataCopy**：写最简 Kernel，4KB 写入 MSAddr0，Host 侧读回验证
3. **补全交织/非交织逻辑**：扩展到任意 dataSize
4. **加入计时**：设备侧 + Host 侧双重计时
5. **全量测试**：4KB → 6MB 扫描，输出带宽报告
