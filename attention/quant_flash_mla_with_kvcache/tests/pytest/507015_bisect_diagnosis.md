# 507015 Device Error 二分定位过程

## 背景

func_rdv 批跑结果：3 passed, 25 failed。所有失败均为 507015 device error。
- 通过的用例：Nq=6/12/24, QS=1 (gS1Size <= 64, 单个 gS1 loop)
- 失败的用例：从 Nq=96, QS=1 (gS1Size=96 > 64, 2个 gS1 loop) 开始级联失败

## 定位过程

### 阶段1：二分注释 FA 计算流程

从后往前逐层注释 C1/V1/C2/V2，确认 507015 由 FA 计算流程中的代码引起（全注释后无507015）。

### 阶段2：单独运行 Nq=96 用例

恢复所有代码，将 Nq=96 参数填入 golden 脚本单独运行 → **Pass，无 507015**。
说明 Nq=96 用例本身没问题，是批跑时级联导致。

### 阶段3：组合测试定位触发用例

| 组合 | 000002 | 000004 | 结果 |
|------|--------|--------|------|
| 000004 单独 | - | 跑 | PASSED |
| 000000 + 000004 | 不跑 | 跑 | PASSED |
| 000001 + 000004 | 不跑 | 跑 | PASSED |
| 000001 + 000002 | 跑 | 不跑 | PASSED |
| **000002 + 000004** | **跑** | **跑** | **000004 507015** |
| 000000 + 000002 + 000004 | 跑 | 跑 | 000004 507015 |

### 阶段4：尝试 synchronize + empty_cache

在用例前后添加 `torch.npu.synchronize()` + `torch.npu.empty_cache()` → **无效**，507015 仍发生。

## 根因结论

**000002（Nq=24, QS=1, KVS=1024, LSE=True, BnNBsD）的 kernel 执行触发 aicore exception（507015），但异常异步延迟到后续用例 000004 的 synchronize 时才上报。**

- 000002 单独运行时 synchronize 能通过（异常可能延迟上报）
- 000002 的 kernel 执行在硬件层面触发了异常
- `synchronize` 和 `empty_cache` 无法修复已损坏的 aicore 状态
- 这是 **kernel bug**，不是测试脚本问题

## 已知问题用例

已将 000002 加入 `FAIL_CASES`，可通过 `pytest -m fail` 单独运行已知失败用例。

## 下一步

需要在算子 kernel 代码中定位 000002 场景（Nq=24, QS=1, gS1Size=24, LSE=True）下的 aicore 异常根因。
关键特征：gS1Size=24 <= mBaseSize(64)，只有1个 gS1 loop，但 LSE=True。
