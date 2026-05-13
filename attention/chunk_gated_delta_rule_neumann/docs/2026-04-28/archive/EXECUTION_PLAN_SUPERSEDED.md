# 执行计划：接入 Neumann 三步倍增法

## 失败处理规则

- 精度不过时，必须先在当前优化点上正向定位和修复；不能把回滚到 PASS baseline 当成任务完成。
- 不能在失败候选上继续叠加下一项优化。需要恢复 baseline 时，只能作为明确放弃候选或重建干净工作面的动作，并记录失败现象、根因假设和下一步正向修复入口。


## 当前问题

| 问题 | 位置 | 状态 |
|------|------|------|
| chunkSize = 64 未改 | tiling.cpp:113 | ❌ |
| INVERSE_SHAPE = 32 未改 | stage1.h:32 | ❌ |
| Neumann 函数已实现但未接入主流程 | stage1.h ProcessOneChunk() | ❌ |
| InverseCompute() 未写 L_powWsGm_ / INVWsGm_ 初值 | stage1.h InverseCompute() | ❌ |
| 后续 AICProcess 仍读 attnWsGm_ 而非 INVWsGm_ | stage1.h ProcessOneChunk() AIC | ❌ |

---

## 修改步骤

### Step 1：常量修改

**tiling.cpp:113**
```cpp
- int64_t c = 64;
+ int64_t c = 16;
```

**stage1.h:32**
```cpp
- constexpr uint64_t INVERSE_SHAPE = 32;
+ constexpr uint64_t INVERSE_SHAPE = 16;
```

---

### Step 2：InverseCompute() 新增写 L_powWsGm_ 和 INVWsGm_

AIV 在 InverseCompute() 中，除了写 attnWsGm_（I-L），还需要：
- 将 L（即 attnUbFloat_，Muls 之前的值）写入 L_powWsGm_
- 将 I-L（inverseLocal_）写入 INVWsGm_

```cpp
// 在 Muls 之前，先把 L 写到 L_powWsGm_
DataCopyOutFp32(halfChunkSize_, chunkSize_, chunkSize_, L_powWsGm_[subOffset_ * chunkSize_]);

// 原有逻辑：inverseLocal_ = I - L，写到 attnWsGm_
// 新增：同时写到 INVWsGm_
DataCopyOutFp32(halfChunkSize_, chunkSize_, chunkSize_, INVWsGm_[subOffset_ * chunkSize_]);
```

注意：DataCopyOutFp32 会 DeQue 并 FreeTensor，所以需要两次 EnQue 或改用 DataCopyPad 直接写。
实际做法：在 EnQue 之后，DataCopyOutFp32 写 attnWsGm_ 之后，再 EnQue 一次写 INVWsGm_，
或者在 DataCopyOutFp32 内部改为写两个目标。

最简方案：在 DataCopyOutFp32(attnWsGm_) 之后，重新 AllocTensor + DataCopy + EnQue + DataCopyOutFp32(INVWsGm_)。

---

### Step 3：ProcessOneChunk() 插入 Neumann 三步

**同步协议**（新增 4 个 flag：0x1/0x2/0x3/0x4）：

```
AIC 侧（CrossCoreWaitFlag(0x7) 之后）：
  Round 1:
    NeumannLpowCompute()          // L² = L @ L
    NeumannMmulCompute()          // tmp = INV @ L²
    CrossCoreSetFlag(0x1)         // 通知 AIV：tmp 已就绪
    CrossCoreWaitFlag(0x2)        // 等待 AIV：INV 已更新
  Round 2:
    NeumannLpowCompute()          // L⁴ = L² @ L²
    NeumannMmulCompute()          // tmp = INV @ L⁴
    CrossCoreSetFlag(0x3)
    CrossCoreWaitFlag(0x4)
  Round 3:
    NeumannLpowCompute()          // L⁸ = L⁴ @ L⁴
    NeumannMmulCompute()          // tmp = INV @ L⁸
    CrossCoreSetFlag(0x1)         // 复用（Round 1 已消费）
    CrossCoreWaitFlag(0x2)
  // 改读 INVWsGm_ 而非 attnWsGm_
  AICProcess(INVWsGm_, gBKWsGm_, outKCumdecayGm_, ...)
  CrossCoreWaitFlag(0x6)
  AICProcess(INVWsGm_, vBetaWsGm_, outVInnerGm_, ...)

AIV 侧（CrossCoreSetFlag(0x7) 之后）：
  Round 1:
    CrossCoreWaitFlag(0x1)
    NeumannAddCompute()           // INV = INV + tmp
    CrossCoreSetFlag(0x2)
  Round 2:
    CrossCoreWaitFlag(0x3)
    NeumannAddCompute()
    CrossCoreSetFlag(0x4)
  Round 3:
    CrossCoreWaitFlag(0x1)        // 复用
    NeumannAddCompute()
    CrossCoreSetFlag(0x2)
  GBKCompute()
  CrossCoreSetFlag(0x6)
```

---

### Step 4：验证

```bash
# 编译
cd /mnt/workspace/gitCode/cann/ops-transformer
bash build.sh --pkg --soc=ascend910b --ops=chunk_gated_delta_rule_neumann -j16

# 精度测试
cd attention/chunk_gated_delta_rule_neumann/tests/pytest
bash test_run.sh single
```

精度通过标准：max_re_rate < 5，avg_re_rate < 1.5，rmse_rate < 1.5，smra_rate < 2。
