# unitFlag 导致 Cube matmul 死锁/timeout

## 现象

完成 AIC/AIV 分工骨架后，加回真实 Bmm1 且带 `unitFlag=3` 时（E3 阶段）：
kernel **timeout（120s）**，Cube 通路跑不通。

## 根因

Mmad 指令带 `unitFlag` 在 A5 MIX 模式下与 CrossCore 同步机制冲突，导致 AIC 卡死。

## 修复

**文件**：`matmul_adapter.h:210-211`

移除 Mmad 的 unitFlag（改为注释说明不设 unitFlag）。

## 验证（E4）

去掉 unitFlag 后，只保留 AIC-Bmm1 诊断版重测：

- ✅ 无 timeout（E3 曾有 120s timeout）
- ✅ 无 AI Core Error
- ✅ Total Task 1, success 1, failed 0
- 精度比对失败（预期：Vec1/Bmm2/Vec2 还是骨架）

**结论**：去掉 unitFlag 后 Cube matmul 通路打通，AIC 能正常执行 Bmm1 并写 GM workspace，CrossCore 同步正常，死锁解决。

> 这是 A5 分核改造能继续往下走的第一个关键突破点。
