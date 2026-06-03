# A3/A5 RegisterV2 零拷贝穿刺对比报告

> 测试代码: `add_custom.asc` (纯 RegisterV2 路径, 无降级)

---

## 1. A3 (910_93) 运行结果

**环境**: 910_93 + CANN 9.1.0 (dav-2201)

### 终端输出

```
[INFO] syncHost=0x12c0014e1000  syncDev=0x3fff8200000
[INFO] Launch OP1 ...
[INFO] Polling syncHost (zero-copy) ...
[INFO] OP1 done, size=32768 bytes
[INFO] RegisterV2(bufB PINNED) host=0xaaaaeef53990 size=32768
[INFO] Launch OP2 ...
[INFO] Done.

>>> Verifying output/output_op1.bin vs output/golden.bin
error ratio: 0.0000
>>> Verifying output/output_op2.bin vs output/golden.bin
error ratio: 0.0000
all test pass!
```

### 各步骤状态

| 步骤 | 状态 | 说明 |
|------|:---:|------|
| `aclrtMallocHost` | ✅ | host-pinned 申请成功 |
| `RegisterV2(MAPPED)` | ✅ | ret=0, flag=2 |
| `GetDevicePointer` | ✅ | syncDev=0x3fff8200000 |
| OP1 launch + add 计算 | ✅ | 8 核正常完成 |
| kernel 写 `syncDev` | ✅ | `*(volatile __gm__ uint32_t*)syncDev = 32768` |
| Host 零拷贝轮询 | ✅ | `while(*syncHost==0){}` 自旋退出 |
| 读到 size | ✅ | 32768 bytes |
| 动态 malloc + RegisterV2 | ✅ | OP2 buffer 分配成功 |
| OP2 执行 | ✅ | 正常完成 |
| OP1/OP2 精度 | ✅ | error ratio 0.0000 / 0.0000 |

---

## 2. A5 (950PR) 运行结果

**环境**: 950PR + CANN 9.1.0 (dav-3510)

### 终端输出

```
=== [3/4] Running on NPU ===
[INFO] Polling syncHost (zero-copy) ...   ← 卡住, 死等
```

### 日志关键证据 (`950.log`)

| 行号 | 内容 | 含义 |
|------|------|------|
| 1343 | `aclrtHostRegisterV2Impl: MemSize=4, flag=2` | RegisterV2(MAPPED) 调用成功 |
| 1571 | `Task run failed, stream_id=61, errType=0x1 (task exception)` | kernel 运行异常 |
| 1587 | `core id=0, error code=264` | block0 崩溃, error code 264 |
| 1594 | `[vector core exception], retCode=0x31` | Vector Core 硬件异常 |
| 1598 | `args(3)=0x40000000000` | 第 4 入参 = syncDev (mapped host 地址) |
| 1615 | `fault kernel_name=_Z20add_custom_with_syncPhS_S_S_j` | 崩溃 kernel = add_custom_with_sync |

### 崩溃根因

```
args[0] = 0x100000017000   (x,  Device GM)   ← 正常
args[1] = 0x10000001f000   (y,  Device GM)   ← 正常
args[2] = 0x100000027000   (z,  Device GM)   ← 正常
args[3] = 0x40000000000    (sync, Mapped Host)  ← 崩溃!

kernel add 计算部分 (前 159 行) 访问 Device GM → 正常
最后一行 *(volatile __gm__ uint32_t*)syncDev = 32768
                 ↑ 地址 0x40000000000
Vector Core 无法对 device-mapped host 地址执行写入
→ 硬件抛出 vector core exception
→ 整个 kernel 被标记为异常
→ Host while(*syncHost==0){} 死等
```

---

## 3. 对比结论

| 项目 | A3 (910_93) | A5 (950PR) |
|------|:-----------:|:----------:|
| `aclrtMallocHost` | ✅ | ✅ |
| `RegisterV2(MAPPED)` | ✅ ret=0 | ✅ ret=0 |
| `GetDevicePointer` | ✅ 0x3fff8200000 | ✅ 0x40000000000 |
| kernel add 计算 | ✅ | ✅ |
| kernel 写 mapped 地址 | ✅ | ❌ Vector Core 异常 |
| Host 零拷贝轮询 | ✅ | ❌ 死等 |
| 两段式 + 动态 alloc | ✅ | ❌ 中断于 OP1 |
| 精度验证 | ✅ 0.0000 | ❌ 未执行到 |

**950PR 上 `aclrtHostRegisterV2(MAPPED)` API 成功且拿到有效 device 地址, 但 NPU Vector Core 硬件不支持对 mapped host 地址执行写入, 零拷贝通路不可用。dispatch 方案在 950PR 上须走 device mem + D2H 替代路径。**
