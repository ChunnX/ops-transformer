# WriteThenRead 模式实现方案

> 在现有 Kernel 基础上新增"先写后读"模式（`MODE_WRITE_THEN_READ = 3`），假设数据可一次搬运到 MS，无需分块循环。写完加 Barrier，再读回，打点外做数据校验。
>
> **数据设计**：用递增模式填充（`dataLocal[i] = i & 0xFF`），每个位置值不同，可检测零值、脏数据、地址偏移、部分写入等问题，验证读写链路真正打通。

---

## 一、整体流程

```
Process():
  ┌── 打点外：准备数据 ──────────────────────┐
  │ FillData(dataLocal)  填充递增模式          │
  │   dataLocal[i] = i & 0xFF                  │
  └─────────────────────────────────────────┘
  ┌── 打点内 ────────────────────────────────┐
  │ startCycle = GetSystemCycle()             │
  │                                           │
  │   DataCopy(MS, dataLocal)   一次写全部    │
  │                                           │
  │   PipeBarrier<PIPE_ALL>()   确保写完       │
  │                                           │
  │   DataCopy(readLocal, MS)   一次读全部    │
  │                                           │
  │   PipeBarrier<PIPE_ALL>()   确保读完       │
  │                                           │
  │ endCycle = GetSystemCycle()               │
  └─────────────────────────────────────────┘
  ┌── 打点外：校验 ──────────────────────────┐
  │ DataCopy(readLocal, MS)    重读一次       │
  │ 逐元素 GetValue 比较:                      │
  │   readLocal[i] == (i & 0xFF) ?            │
  │ 结果写入 timeGm[2]                         │
  └─────────────────────────────────────────┘
  写入 startCycle, endCycle 到 timeGm
```

---

## 二、改动详情

### 2.1 新增枚举值

```cpp
enum RwMode : uint32_t {
    MODE_WRITE = 0,
    MODE_READ = 1,
    MODE_READ_WRITE = 2,
    MODE_WRITE_THEN_READ = 3,  // 新增：先写后读
};
```

### 2.2 Init 调整

- 0 核搬全部，去掉多核分块逻辑
- UB buffer 大小改为 `dataSize`（一次搬运，不再固定 4KB）
- 数据直接写到 CCUA0（`msGm_[0]`），不跨 CCUA

```cpp
// 0 核搬全部
if (aivId_ > 0) return;

// buffer 大小 = dataSize，一次搬完
pipe->InitBuffer(dataBuf_, dataSize);
pipe->InitBuffer(readBuf_, dataSize);

// 不分块，一次搬全部
totalElem_ = dataSize / sizeof(T);
```

### 2.3 新增 `WriteThenReadMode()` 方法

`dataLocal` 和 `readLocal` 在打点外获取，避免 `Get<T>()` 开销计入计时：

```cpp
// 成员变量持有，打点外获取
LocalTensor<T> dataLocal_;
LocalTensor<T> readLocal_;

// Init 中获取
dataLocal_ = dataBuf_.Get<T>();
readLocal_ = readBuf_.Get<T>();

__aicore__ inline void WriteThenReadMode()
{
    // === 写阶段：一次写全部 ===
    DataCopy(msGm_[0], dataLocal_, totalElem_);  // UB → MS

    PipeBarrier<PIPE_ALL>();  // 确保写完

    // === 读阶段：一次读全部 ===
    DataCopy(readLocal_, msGm_[0], totalElem_);  // MS → UB

    PipeBarrier<PIPE_ALL>();  // 确保读完
}
```

### 2.4 新增 `VerifyResult()` 方法（打点外校验）

```cpp
__aicore__ inline void VerifyResult()
{
    verifyResult_ = 0;  // 0=通过, 1=失败

    DataCopy(readLocal_, msGm_[0], totalElem_);  // 重读一次
    for (uint32_t i = 0; i < totalElem_; i++) {
        if (readLocal_.GetValue(i) != static_cast<T>(i & 0xFF)) {
            verifyResult_ = 1;
            return;
        }
    }
}
```

### 2.5 `Process()` 调整

- 新增 `FillData()` 方法填充递增模式（打点前）
- switch 里加 `case MODE_WRITE_THEN_READ`
- 打点后仅新模式调用 `VerifyResult()`
- 去掉 `if (aivId_ == 0)`，直接写 timeGm

```cpp
__aicore__ inline void Process()
{
    // 打点外：填充递增模式数据
    FillData();

    // 打点内
    uint64_t startCycle = GetSystemCycle();

    switch (mode0_) {
        case MODE_WRITE:           WriteMode();          break;
        case MODE_READ:            ReadMode();           break;
        case MODE_READ_WRITE:      ReadWriteMode();      break;
        case MODE_WRITE_THEN_READ: WriteThenReadMode();  break;
        default: break;
    }

    PipeBarrier<PIPE_ALL>();
    uint64_t endCycle = GetSystemCycle();

    // 打点外：校验（仅新模式）
    if (mode0_ == MODE_WRITE_THEN_READ) {
        VerifyResult();
    }

    // 写结果
    timeGm_.SetValue(0, startCycle);
    timeGm_.SetValue(1, endCycle);
    timeGm_.SetValue(2, verifyResult_);
}
```

### 2.6 新增 `FillData()` 方法

用递增模式填充 dataLocal，每个位置值不同，确保能检测出零值、脏数据、地址偏移、部分写入等问题：

```cpp
__aicore__ inline void FillData()
{
    for (uint32_t i = 0; i < totalElem_; i++) {
        dataLocal_.SetValue(i, static_cast<T>(i & 0xFF));
    }
}
```

### 2.6 其他 TODO 处理

| # | 位置 | 处理 |
|---|------|------|
| 1 | test:171 | 删注释，rankId 即 deviceId |
| 2 | kernel:84 | 0 核搬全部，去掉多核分块逻辑 |
| 3 | kernel:132 | Duplicate 已移到 Process 打点前 |
| 4 | kernel:176 | 保留现有 ReadWriteMode 不动 |
| 5 | kernel:183 | 仅新模式做校验 |
| 6 | kernel:208 | 去掉 `if (aivId_ == 0)` |

### 2.7 新增成员变量

```cpp
LocalTensor<T> dataLocal_;   // 打点外获取，持有引用
LocalTensor<T> readLocal_;   // 打点外获取，持有引用
uint32_t totalElem_{0};      // 总元素个数 = dataSize / sizeof(T)
uint32_t verifyResult_{0};   // 校验结果：0=通过, 1=失败
```

---

## 三、UB 容量约束

一次搬运要求 `dataSize` 能放进 UB：

| 场景 | dataSize 上限 | 原因 |
|------|-------------|------|
| 两个 buffer（dataBuf + readBuf） | ~120KB | `248KB / 2 = 124KB`，扣 TPipe 开销 |
| 单 buffer（复用） | ~240KB | 但校验需要两个 buffer 同时存在 |

**当前阶段先测试 4KB ~ 120KB 范围**，后续大数据量再分块。

---

## 四、Host 侧改动 (`test_local_ms_rw.cpp`)

### 4.1 新增 dataSize 上限校验

在 `aclInit` 之前加校验，避免 UB 溢出：

```cpp
constexpr uint64_t MAX_DATA_SIZE = 120 * 1024; // 120KB，留余量给 TPipe 开销
if (g_dataSize > MAX_DATA_SIZE) {
    LOG_PRINT("[ERROR] dataSize=%lu 超过单次搬运上限 %lu bytes，请减小 data_size\n",
              g_dataSize, MAX_DATA_SIZE);
    return -1;
}
```

### 4.2 timeGm 扩到 24 字节

```cpp
// 原：ret = aclrtMalloc(&timeGm, 16, ACL_MEM_MALLOC_HUGE_FIRST);
// 改：
ret = aclrtMalloc(&timeGm, 24, ACL_MEM_MALLOC_HUGE_FIRST); // 3 x uint64_t
```

### 4.3 读取计时和校验结果

```cpp
// 原：
// uint64_t timeHost[2];
// ret = aclrtMemcpy(timeHost, 16, timeGm, 16, ACL_MEMCPY_DEVICE_TO_HOST);
// ...
// uint64_t startCycle = timeHost[0];
// uint64_t endCycle = timeHost[1];

// 改：
uint64_t timeHost[3];
ret = aclrtMemcpy(timeHost, 24, timeGm, 24, ACL_MEMCPY_DEVICE_TO_HOST);
CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy timeGm failed. ret: %d\n", ret); return ret);

uint64_t startCycle = timeHost[0];
uint64_t endCycle = timeHost[1];
uint64_t verifyResult = timeHost[2];
```

### 4.4 打印校验结果

在性能结果输出中新增校验结果打印：

```cpp
// 在 "带宽" 行之后、"======" 行之前新增：
if (g_mode0 == 3) {
    if (verifyResult == 0) {
        LOG_PRINT("  数据校验:   通过\n");
    } else {
        LOG_PRINT("  数据校验:   失败\n");
    }
}
```

### 4.5 删除 test:171 todo 注释

```cpp
// 删除：// 【todo】应该传deviceid
```

---

## 五、使用方式

```bash
# 先写后读模式 (mode0=3)，非交织 (mode1=0)，4KB 数据
./test_local_ms_rw --mode0 3 --mode1 0 --data_size 4096

# 64KB 数据
./test_local_ms_rw --mode0 3 --mode1 0 --data_size 65536

# 120KB 数据（接近单次搬运上限）
./test_local_ms_rw --mode0 3 --mode1 0 --data_size 122880
```

---

## 六、改动文件清单

| 文件 | 改动 |
|------|------|
| `kernel/local_ms_rw_kernel.h` | 新增枚举值、`WriteThenReadMode()`、`VerifyResult()`、`FillData()`；Process 重写；Init 改为按 dataSize 分配 buffer、打点外获取 dataLocal_/readLocal_；去掉多核分块；删 6 个 todo |
| `test_local_ms_rw.cpp` | dataSize 上限校验；timeGm 16→24 字节；读取 verifyResult 并打印；删 todo 注释 |
