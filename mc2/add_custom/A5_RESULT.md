# A5 (950PR) RegisterV2 零拷贝穿刺结果

**日期**: 2026-06-04
**环境**: 950PR + CANN 9.1.0 (dav-3510)
**测试代码**: `add_custom.asc` (纯 RegisterV2 路径, 无降级)

## 测试流程回顾

```
[Host] aclrtMallocHost(4) → RegisterV2(MAPPED) → GetDevicePointer → syncDev
[Host] Launch OP1 (add_custom_with_sync)
[Kernel] add 计算 → block0: *(volatile __gm__ uint32_t*)syncDev = 32768
[Host] while(*syncHost == 0) {}  ← 零拷贝自旋
```

## 结果

**失败 — kernel 运行期崩溃，零拷贝通路不可用**

```
=== [1/4] Generating test data ===
=== [2/4] Building (dav-3510) ===
   编译成功
=== [3/4] Running on NPU ===
   [INFO] Polling syncHost (zero-copy) ...   ← 卡住, 死等
```

### 日志关键证据 (`950.log`)

| 行号 | 内容 | 含义 |
|------|------|------|
| 1343 | `aclrtHostRegisterV2Impl: MemSize=4, flag=2` | RegisterV2(MAPPED) API 调用成功 |
| 1571 | `Task run failed, stream_id=61, errType=0x1 (task exception)` | kernel 运行异常 |
| 1587 | `core id=0, error code=264` | block0 崩溃, 错误码 264 (向量指令异常) |
| 1594 | `[vector core exception], retCode=0x31` | Vector Core 硬件异常 |
| 1598 | `args(3)=0x40000000000` | 第 4 个入参 = syncDev (mapped host 地址) |
| 1615 | `fault kernel_name=_Z20add_custom_with_syncPhS_S_S_j` | 崩溃 kernel = add_custom_with_sync |

### 崩溃根因

```
args[0] = 0x100000017000   (x, Device GM)        ← 正常
args[1] = 0x10000001f000   (y, Device GM)        ← 正常
args[2] = 0x100000027000   (z, Device GM)        ← 正常
args[3] = 0x40000000000    (sync, Mapped Host)   ← 崩溃!

kernel 前 159 行 add 计算访问 Device GM → 正常
最后一行 *(volatile __gm__ uint32_t*)syncDev = 32768
                                ↑ 0x40000000000
Vector Core 无法对这段 device-mapped host 地址执行写操作
→ 硬件抛出 vector core exception, 整个 kernel 被标记为异常
```

## 结论

| 项目 | A3 (910_93) | A5 (950PR) |
|------|:-----------:|:----------:|
| `aclrtMallocHost` | ✅ ret=0 | ✅ ret=0 |
| `RegisterV2(MAPPED)` | ✅ ret=0 | ✅ ret=0 |
| `GetDevicePointer` | ✅ 有效地址 | ✅ 有效地址 (`0x40000000000`) |
| kernel add 计算部分 | ✅ 正常 | ✅ 正常 |
| kernel 写 mapped 地址 | ✅ 通过 | ❌ Vector Core 异常 |
| Host 零拷贝轮询 | ✅ `while(*syncHost==0){}` | ❌ 死等 |
| 两段式 + 动态 alloc | ✅ 全链路 | ❌ 中断于 OP1 |

**950PR 上 `aclrtHostRegisterV2(MAPPED)` API 调用成功且拿到有效 device 端地址，但 NPU Vector Core 硬件不支持对 mapped host 地址执行写入操作。dispatch 方案在 950PR 上不能依赖 RegisterV2(MAPPED) 零拷贝路径，须走 device mem + D2H 替代。**
