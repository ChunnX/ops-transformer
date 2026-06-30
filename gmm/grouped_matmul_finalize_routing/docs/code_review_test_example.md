# 代码审查：`test_aclnn_grouped_matmul_finalize_routing.cpp`

> 审查日期：2026-06-30  
> 问题表现：本地正常跑，线上环境卡住

---

## 目录

- [1. 致命问题：双重释放（double-free）](#1-致命问题双重释放double-free)
- [2. 关键问题：设备重置前未释放 RAII 资源](#2-关键问题设备重置前未释放-raii-资源)
- [3. 条件反转：`CHECK_RET` 语义错误](#3-条件反转check_ret-语义错误)
- [4. 修复总结](#4-修复总结)
- [5. 附录：修正后的完整资源释放时序](#5-附录修正后的完整资源释放时序)

---

## 1. 致命问题：双重释放（double-free）

### 现象

同一块 device 内存 / 同一个 `aclTensor` 句柄被释放了两次：

1. 第 280–303 行（原代码）：显式调用 `aclDestroyTensor(x)` 和 `aclrtFree(xDeviceAddr)`  
2. 函数返回时：`std::unique_ptr` 析构函数再次调用 `aclDestroyTensor(x)` 和 `aclrtFree(xDeviceAddr)`

### 根因

代码同时使用了两种资源管理模式，相互冲突：

```cpp
// 模式 A：RAII（第 192-193 行）
std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> xTensorPtr(x, aclDestroyTensor);
std::unique_ptr<void, aclError (*)(void *)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);

// ...业务逻辑...

// 模式 B：手动释放（第 280、293 行，原代码）
aclDestroyTensor(x);      // ← 第一次释放 tensor 句柄
aclrtFree(xDeviceAddr);  // ← 第一次释放 device 内存
// main() 返回 → unique_ptr 析构 → 第二次释放 ← 未定义行为
```

`aclrtFree` 对 device 内存的双重释放是**线上卡死的最可能原因**。本地环境的内存管理器可能允许 double-free 无害通过（或静默忽略），但线上环境的内存管理器在检测到已释放指针时可能进入死锁或无限等待。

### 涉及资源（共 22 个资源）

| 资源类型 | 变量 | 数量 |
|---------|------|------|
| `aclTensor*` | x, w, scale, bias, offset, pertokenScale, groupList, sharedInput, logit, rowIndex, out | 11 |
| device addr (void*) | xDeviceAddr, wDeviceAddr, scaleDeviceAddr, biasDeviceAddr, offsetDeviceAddr, pertokenScaleDeviceAddr, groupListDeviceAddr, sharedInputDeviceAddr, logitDeviceAddr, rowIndexDeviceAddr, outDeviceAddr | 11 |

---

## 2. 关键问题：设备重置前未释放 RAII 资源

### 现象

即使修复了第 1 节的双重释放，仅靠 unique_ptr 在 `main()` 返回时自动析构，会使 `aclrtFree` 和 `aclDestroyTensor` 在 `aclrtResetDevice` / `aclFinalize` **之后**执行。

### 执行时序（错误）

```
// 改前代码的执行顺序
line 285  aclDestroyIntArray(tuningConfig);
line 286  aclrtFree(workspaceAddr);        // workspace 手动释放 ✓
line 287  aclrtDestroyStream(stream);       // 销毁流 ✓
line 288  aclrtResetDevice(deviceId);       // ★ 重置设备
line 289  aclFinalize();                    // ★ 终结 ACL
line 290  return 0;
          // → 栈展开，unique_ptr 析构
          // → aclrtFree(xDeviceAddr)  ← 设备已重置！UB！
          // → aclDestroyTensor(x)     ← ACL 已终结！UB！
```

CANN ACL 接口规范要求的清理顺序：

```
1. aclDestroyTensor       (释放 tensor 描述符，host 操作)
2. aclrtFree              (释放 device 内存，需要设备上下文)
3. aclrtDestroyStream     (销毁流)
4. aclrtResetDevice        (重置设备)
5. aclFinalize             (终结 ACL runtime)
```

违反此顺序在某些环境中可能静默通过，但在严格配置下可能导致 `aclrtFree` 内部等待超时。

---

## 3. 条件反转：`CHECK_RET` 语义错误

### 问题代码（第 247 行）

```cpp
aclIntArray *tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
CHECK_RET(tuningConfig == nullptr, -1);
```

### `CHECK_RET` 宏定义

```cpp
#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)
```

宏在 `!cond` 为 true 时触发。代入 `cond = (tuningConfig == nullptr)`：

```
!(tuningConfig == nullptr)  →  tuningConfig != nullptr
```

所以实际语义是：**"如果 tuningConfig 分配成功（非空），返回 -1"**——完全反了。

### 还有第二个问题：漏了 `return` 关键字

`-1;` 只是一个值为 -1 的表达式语句，不是 `return -1;`，所以即使条件触发，也**不会退出函数**。

### 影响

`tuningConfig` 的错误检查实际上是一个空操作。如果 `aclCreateIntArray` 分配失败，`tuningConfig` 为 `nullptr`，传入算子 API 会导致空指针解引用。

---

## 4. 修复总结

### 变更说明

| 位置 | 修改 | 目的 |
|------|------|------|
| 第 247 行 | `tuningConfig == nullptr` → `tuningConfig != nullptr`，补上 `return` | 修正错误检查逻辑 |
| 第 279–303 行（原） | 删除所有手动 `aclDestroyTensor` / `aclrtFree` | 消除双重释放 |
| 第 279–300 行（新） | 在 `aclrtResetDevice` 前添加所有 unique_ptr 的 `.reset()` | 确保资源在设备重置前释放 |

### 修正后的执行时序

```
// 新代码执行顺序
xTensorPtr.reset();           // → aclDestroyTensor(x)      ✓ host 操作
xDeviceAddrPtr.reset();       // → aclrtFree(xDeviceAddr)   ✓ 设备上下文完好
wTensorPtr.reset();
wDeviceAddrPtr.reset();
...  // 全部 22 个 unique_ptr 在此释放

aclDestroyIntArray(tuningConfig);
aclrtFree(workspaceAddr);
aclrtDestroyStream(stream);
aclrtResetDevice(deviceId);  // → 此时所有 device 内存已释放 ✓
aclFinalize();               // → 此时所有 tensor 句柄已释放 ✓
return 0;                    // → unique_ptr 已 empty，析构为 no-op
```

---

## 5. 附录：修正后的完整资源释放时序

```
Step 1: 显式 .reset() 所有 unique_ptr（执行 22 次释放）
  ├─ aclDestroyTensor(x)         释放在 host 上分配的 tensor 描述符
  ├─ aclrtFree(xDeviceAddr)      释放在 device 上分配的内存
  ├─ aclDestroyTensor(w)
  ├─ aclrtFree(wDeviceAddr)
  ├─ ...（其余 9 对同理）
  └─ aclDestroyTensor(out) + aclrtFree(outDeviceAddr)

Step 2: aclDestroyIntArray(tuningConfig)    释放 host 侧的整数数组

Step 3: aclrtFree(workspaceAddr)            释放 workspace

Step 4: aclrtDestroyStream(stream)          销毁 stream

Step 5: aclrtResetDevice(deviceId)          ● 设备重置——此时所有 device 内存已释放

Step 6: aclFinalize()                       ● ACL 终结——此时所有句柄已释放

Step 7: return 0 → 栈展开
         所有 unique_ptr 已是 nullptr，析构函数为空操作 ✓
```

---

## 参考

- 文件路径：`gmm/grouped_matmul_finalize_routing/examples/test_aclnn_grouped_matmul_finalize_routing.cpp`
- CANN ACL API 文档：`aclrtMalloc` / `aclrtFree` 要求成对调用且在设备重置前完成
