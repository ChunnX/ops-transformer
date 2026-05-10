# Stage6: Execute 读端优化 — 实验结论

**日期**: 2026-05-10
**分支**: stage6_read_db
**环境**: 4xAscend950PR, CANN 9.0.0, world_size=4

---

## 实验矩阵

| 方向 | 精度 | 性能 vs stage3 | 结论 |
|------|------|---------------|------|
| TQueBind 读端 DB | 0/192 PASS | — | 不可行 |
| x/scale EnQue 重排 | 192 PASS | QRS +12% | 劣化 |
| WriteStatus 复位外提 | 192 PASS | 持平 | 已提交 |
| Leader 模式状态同步 | 192 PASS | 持平 | 已提交 |

---

## 关键经验

1. TQueBind DeQue 后必须接 MTE DataCopy，不能接纯 Vector 计算
2. scale 数据量极小（64B），与 x 重叠 MTE 搬运无收益
3. `SyncAll()` 不能加模板参数，裸调用才对；`SyncAll<false>`/`SyncAll<true>` 导致 507035
4. Leader 模式需 tiling 侧 `SetScheduleMode(1)`（batch mode）
5. 纯 AIV 算子无需 `ASCEND_IS_AIC` 保护（AIC=0）
