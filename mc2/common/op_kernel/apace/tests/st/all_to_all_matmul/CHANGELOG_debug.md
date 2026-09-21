# AllToAllMatmul 非量化 ST 调试改动清单

- **日期**: 2026-09-20
- **目标文件**: `mc2/common/op_kernel/apace/kernel/matmul/batch_matmul/all_to_all_mm_kernel.h`
- **工程**: ops-transformer apace AllToAllMatmul 非量化 HComm 场景

---

## 1. `all_to_all_mm_kernel.h`（kernel 核心）

**改 4 处**（唯一改动项，其余逻辑未动）：

| 行号 | 原代码 | 新代码 |
|------|--------|--------|
| 354 | `AscendC::PipeBarrier<PIPE_FIX>();` | `AscendC::PipeBarrier<PIPE_ALL>();` |
| 365 | `AscendC::PipeBarrier<PIPE_FIX>();` | `AscendC::PipeBarrier<PIPE_ALL>();` |
| 401 | `AscendC::PipeBarrier<PIPE_FIX>();` | `AscendC::PipeBarrier<PIPE_ALL>();` |
| 408 | `AscendC::PipeBarrier<PIPE_FIX>();` | `AscendC::PipeBarrier<PIPE_ALL>();` |

> **起因**：`PIPE_FIX` 在 dav-3510 上非法导致编译报错（cannbot 分析：`pipe_barrier` 内建只接受 [4,6]，`PIPE_FIX=10` 超范围）。
> **过程**：曾先试 `PIPE_MTE2`（编译通过但运行 AICORE 异常 507015），最终使用 `PIPE_ALL`（编译通过、无 AICORE 异常，但输出全 0）。

---

## 2. `.../tests/st/all_to_all_matmul/src/apace_a2a_kernel_launcher.h`（测试 launcher）

**改 3 处**：

```
KERNEL_TYPE_PURE_AIC  →  KERNEL_TYPE_MIX_AIC_1_1  →  KERNEL_TYPE_AIC_ONLY
```

> **起因**：`KERNEL_TYPE_PURE_AIC` 宏不存在（编译报 `undeclared identifier`）。
> **过程**：先改 `MIX_AIC_1_1`（量化版 URMA 同款），但非量化 hcomm 引入 blaze 的 `block_mmad_matmul_basic.h`，MIX 类型会编译 AIV 路径触发 pipe_barrier 非法；最终用 `AIC_ONLY`（纯 AIC，hcomm 实现是 cube-only）。

---

## 3. `.../tests/st/all_to_all_matmul/scripts/gen_data.py`

**改 1 处**（30 行）：

```python
# 改前
out.view(np.float32).tofile(os.path.join(output_dir, "cpu_output.bin"))
# 改后
out.numpy().tofile(os.path.join(output_dir, "cpu_output.bin"))
```

> **起因**：`out` 是 PyTorch Tensor，不能调用 numpy 的 `.view(np.float32)`，报 `TypeError: view() received an invalid combination of arguments`。

---

## 4. `.../tests/st/all_to_all_matmul/src/main.cpp`（改动最多）

### 4.1 新增 include

```cpp
#include "tiling/hccl/hccl_tiling.h"           // Mc2CcTilingConfig
#include "tiling/platform/platform_ascendc.h"   // 平台核数获取
```

### 4.2 新增 3 个 extern "C" 声明

```cpp
extern "C" HcclResult HcclGetCommName(HcclComm comm, char *commName);
extern "C" HcclResult HcomGetCommHandleByGroup(const char *group, HcclComm *commHandle);
extern "C" HcclResult HcclAllocComResourceByTiling(HcclComm comm, void *stream, void *mc2Tiling, void **commContext);
```

### 4.3 `exchanger.Close()` 后新增通信上下文创建（约 228-252 行）

```cpp
// 获取通信域名称并初始化 MC2 tiling
char hcomName[128] = {0};
HCCL_CHECK(HcclGetCommName(comm, hcomName));

AscendC::Mc2CcTilingConfig mc2CcTilingConfig(hcomName, 8, "AlltoAll=level0:fullmesh;level1:pairwise");
mc2CcTilingConfig.SetCommEngine(3);
mc2CcTilingConfig.GetTiling(tilingData.mc2InitTiling);
mc2CcTilingConfig.GetTiling(tilingData.mc2CcTiling);

HcclComm commHandle;
HCCL_CHECK(HcomGetCommHandleByGroup(hcomName, &commHandle));
void *mc2Context = nullptr;
HCCL_CHECK(HcclAllocComResourceByTiling(commHandle, stream, &tilingData, &mc2Context));
```

### 4.4 释放前新增

```cpp
aclrtFree(mc2Context);
```

### 4.5 `usedCoreNum` 改为平台自动获取

```cpp
// 改前
uint32_t usedCoreNum = (envCoreNum != nullptr) ? static_cast<uint32_t>(std::stoul(envCoreNum)) : 40U;
// 改后（含环境变量优先）
const char *envCoreNum = std::getenv("USED_CORE_NUM");
uint32_t usedCoreNum = 0;
if (envCoreNum != nullptr) {
    usedCoreNum = static_cast<uint32_t>(std::stoul(envCoreNum));
} else {
    usedCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
}
```

> 实测 950PR 上 `GetCoreNumAic()` 返回 **28**（非 910B 的 40）。

### 4.6 新增 `[DIAG]` 诊断代码（非最终交付内容）

- dump workspace（通信输出）和 output 的前几个值 / 非零计数
- 将 workspace 填满哨兵值 `0x3F00`（≈0.5f fp16），判断通信是否真的写入

### 4.7 ⚠️ 临时诊断改动（当前仍保留，验证后应还原）

```cpp
tilingData.localMatmul = 0  →  2   // 走 LOCAL 分支验证 kernel 逻辑
```

---

## 5. `.../tests/st/all_to_all_matmul/CMakeLists.txt`

**改 2 处**：

```cmake
# 改前
set(KERNEL_OPTS "-xasc" "--npu-arch=dav-3510" "-DASC_DEVKIT_MAJOR=9" "-w" "-O3")
set(COMMON_LIBS dl platform tiling_api ascendcl runtime hccl stdc++)

# 改后
set(KERNEL_OPTS "-xasc" "--npu-arch=dav-3510" "-DASC_DEVKIT_MAJOR=9" "-D_GLIBCXX_USE_CXX11_ABI=0" "-w" "-O3")
set(COMMON_LIBS dl platform tiling_api register ascendcl runtime hccl stdc++ pthread)
```

> 与参考实现 `quant_matmul_reduce_scatter/ubmem` 对齐。

---

## 6. `all_to_all_matmul_hcomm_impl.h`

**无改动**（逻辑完整，等待 host 侧提供合法的 `mc2InitTiling`/`mc2CcTiling` 和通信上下文）。

---

## 7. ⚠️ 待还原的临时诊断代码

| 位置 | 当前值 | 应还原为 |
|------|--------|----------|
| main.cpp:171 | `localMatmul = 2` | `localMatmul = 0` |
| main.cpp DIAG 块 | 保留 | 删除 |
| workspace 哨兵填充 | 保留 | 删除 |

---

## 8. 当前遗留问题

| 现象 | 结论 |
|------|------|
| 编译 | ✅ 通过（`PIPE_ALL` + `AIC_ONLY` 后） |
| 通信 | ❌ workspace 哨兵值 `1.75` 完好未变 → AlltoAll 未写入 workspace |
| 计算 | ❌ output 全 0，即使 localMatmul=2 走 LOCAL 分支也不产出 |
| 运行 | ⚠️ `aclError:507015`（AICORE 异常）+ `free(): invalid pointer` |

**核心结论**：换 `PIPE_ALL`、`AIC_ONLY`、补通信上下文后，**mmad 可能根本没执行**。需要先定位这个根因（核数/通信上下文/地址越界），再决定是否回退或重构 atomic 方案。

---

## 9. 参考实现

- `quant_matmul_reduce_scatter/ubmem/src/main.cpp`（hcomm 上下文创建流程）
- `experimental/mc2/quant_all_reduce/test_quant_all_reduce.cpp`（Mc2CcTilingConfig 用法）
- ops-tensor `blaze/gemm/block/block_mmad_matmul_basic.h`（非量化 mmad）
- ops-tensor `blaze/gemm/block/block_mmad_matmul_basic_split_k.h`（非量化 splitK mmad，含 `PipeBarrier<PIPE_FIX>`，内部 atomic 管理）
