# A3/A5 零拷贝 sync 写入穿刺对比报告

> 测试代码: `add_custom.asc` (aclrtHostRegister + MAPPED)
> 测试规格: shape=[8,2048], float16, 8 blocks, tileNum=8

---

## 1. 最终实现（写法3 — 通过）

```c
if (AscendC::GetBlockIdx() == 0) {
    AscendC::LocalTensor<half> syncLocal = outQueueZ.AllocTensor<half>();        // 借主 pipe 的 outQueueZ UB
    AscendC::Duplicate(syncLocal, (half)0.0, 16);                               // 32B 清零(避免残留)
    AscendC::LocalTensor<uint32_t> syncLocalU32 = syncLocal.ReinterpretCast<uint32_t>();
    syncLocalU32.SetValue(0, totalLength * sizeof(half));                        // 写动态值到 UB
    AscendC::GlobalTensor<half> syncGm;
    syncGm.SetGlobalBuffer((__gm__ half*)sync, 16);                              // 绑定 syncDev 地址
    AscendC::DataCopy(syncGm, syncLocal, 16);                                    // MTE 引擎搬 UB → mapped host
    outQueueZ.FreeTensor(syncLocal);                                             // 归还 UB
}
```

### 核心设计要点

| 要点 | 说明 |
|------|------|
| 不复用新 TPipe | 新 TPipe 与主 pipe 的 UB 分配冲突，必须复用主 `pipe` 的 `outQueueZ` |
| `Duplicate` 清零 | `AllocTensor` 拿到的 UB 有计算循环残留，DataCopy 打包垃圾→A5 输出乱码 |
| `ReinterpretCast` + `SetValue` | 用 `totalLength * sizeof(half)` 动态计算，不写死常量 |
| DataCopy 16 half | 与 host 侧 `sizeof(uint32_t)*8`=32B 对齐，无越界 |
| MTE 引擎而非 Vector Core | 950PR 的 Vector Core 不支持 mapped host 地址标量写，MTE 引擎可以 |

---

## 2. 三种 sync 写入方案对比

| 写法 | 原理 | A3 (910_93) | A5 (950PR) |
|------|------|:---:|:---:|
| `*(volatile __gm__ uint32_t*)sync = val` | Vector Core 标量 store | ✅ | ❌ VC 异常 |
| `st_dev(val, (__gm__ uint32_t*)sync, 0)` | Vector Core 标量 store（参照 SendJFSDoorBell） | ✅ | ❌ VC 异常 |
| `DataCopy` 复用 outQueueZ（MTE 引擎） | MTE 引擎从 UB 搬 half → mapped host | ✅ | ✅ |

---

## 3. A5 (950PR) 写法 1/2 崩溃日志

```
Task run failed, stream_id=61, errType=0x1 (task exception)
core id=0, error code=264
[vector core exception], retCode=0x31
args(3)=0x40000000000         ← syncDev (mapped host 地址)
fault kernel=add_custom_with_sync
```

根因: 950PR Vector Core 硬件不支持对 mapped host 地址（DVA 0x40000000000）执行标量 store。

---

## 4. 踩坑记录

| 问题 | 现象 | 根因 | 修复 |
|------|------|------|------|
| 新建 TPipe | A3 编译/运行失败 | 两个 TPipe 的 InitBuffer 冲突 | 复用主 pipe 的 outQueueZ |
| `ReinterpretCast` + `SetValue` 无清零 | A5 sync 值乱码 (1253722702) | UB 残留垃圾被 DataCopy 打包输出 | 先 `Duplicate` 清零 |
| `-(half)0.0` | A5 编译失败 | aicore 函数禁止 half 精度运算 | `(half)(-0.0f)` float 域取反 |
| DataCopy 16 half + host 4B 分配 | A5 输出文件损坏 (629M 元素) | mapped host 越界 28B | host 扩 32B |
| `pipe_barrier(PIPE_ALL)` | 多余 | 循环结束 + kernel 退出自动同步 | 删除 |
| `outQueueZ` buffer 扩展 | 多余 | 循环后队列已空 | 还原原始大小 |
| `#include <string>` 缺失 | A5 bisheng 编译失败 | `<fstream>` 未隐式引入 `std::string` | 显式 `#include <string>` |

---

## 5. 平台差异总结

| 项目 | A3 (910_93) | A5 (950PR) |
|------|:-----------:|:----------:|
| `aclrtMallocHost` | ✅ | ✅ |
| `RegisterV2(MAPPED)` | ✅ ret=0 | ✅ ret=0 |
| `GetDevicePointer` | ✅ 0x3fff8200000 | ✅ 0x40000000000 |
| kernel add 计算 | ✅ | ✅ |
| Vector Core 写 mapped host | ✅ 支持 | ❌ 硬件不支持 |
| MTE DataCopy 写 mapped host | ✅ 支持 | ✅ 支持 |
| Host 零拷贝轮询 | ✅ | ✅ |
| 精度验证 | ✅ 0.0000 | ✅ 0.0000 |

**结论**: 950PR 上 Vector Core 标量存不支持 mapped host 地址写入，必须通过 **MTE DataCopy 通路**。门铃寄存器（MMIO）走 non-cacheable 硬件通路与普通 host-registered 内存的 PCIe 映射路径不同，`st_dev` 无法绕过这一限制。
