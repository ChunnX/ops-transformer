# 2026-05-02 进度与问题

## 已完成

- 复核当前默认实现：`C64_1616_FP32_CVInverse`，host `chunkSize=64`。
- 明确当前目标仍是 raw MMAD fused 路线，不回退到长期 `MatmulImpl` 方案。
- 复核当前源码开关，确认多个 fallback 是为了定位问题临时打开，不是最终状态。
- 确认 2026-05-01 已完成的关键修复仍成立：raw FP32 `L1->L0A/L0B` 使用 `LoadData3DParamsV2<float>` 后，小范围 tail/compose smoke 已 PASS。
- 已回退 HF32 临时实验，重新构建安装后复测关键 case；失败形态仍为 Neumann `out` FAIL、`state` PASS。
- 为恢复精度，将 `RAW_MMAD_USE_MATMULIMPL_VINNER_FINAL_FALLBACK` 改为 `true`，`INV @ v_beta -> outVInnerGm_` 暂回 `MatmulImpl`。
- 重新构建并安装双算子包成功。
- 关键失败 case `B=2,S=160,nk=8,nv=32,dk=dv=64,seed=40` 已恢复 PASS，Neumann out/state `max_abs=9.762645e-04/4.881620e-04`。
- 全量三方精度 `CGDR_GOLDEN_CHUNK_SIZE=64 bash tests/pytest/test_run.sh compare` 从算子根目录验证 `20 passed in 6.18s`。
- 修复 `tests/pytest/test_run.sh` 的工作目录和退出码透传，避免从算子根目录运行时假通过。

## 进行中

- 当前代码以精度通过为优先状态，默认 `vinner final` 已回退到 `MatmulImpl`。
- 当前源码与文档一致：`RAW_MMAD_USE_MATMULIMPL_VINNER_FINAL_FALLBACK = true` 已落地，不再只是文档记录。
- raw `RawMmad64ByN()` final 路径仍需后续单独定位，重点仍是 `INV @ v_beta -> outVInnerGm_`。
- 如继续追 fused 性能，需要重新打开 `RAW_MMAD_USE_MATMULIMPL_VINNER_FINAL_FALLBACK=false` 并从关键 case 开始复测。

## 阻塞项

- `RAW_MMAD_USE_MATMULIMPL_VINNER_FINAL_FALLBACK=false` 时，raw `outVInnerGm_` 路径仍无法通过关键 multi-head/multi-batch case；当前通过 MatmulImpl fallback 绕过。
- raw final 写到 `tmpWsGm_` 再 fallback 覆盖真实输出仍出现 `out` FAIL，说明 `tmpWsGm_` 可能被 Stage3 或后续链路复用，当前隔离实验不够干净。
- HF32 已回退；raw final 失败曾表现为 `CPU vs ChunkGDRNeumann out max_abs=1.037243e-01` 到 `1.584216e-01` 区间，state 仍 PASS。

## 风险

- 当前文件保留大量 debug/fallback 常量，容易把“临时定位状态”误判为最终 fused 状态；每次测试前必须记录开关组合。
- `tmpWsGm_` / `INVWsGm_` 不是安全 scratch，写入会污染 fallback 或 Stage3 消费路径。
- `state` PASS 不代表 `outVInnerGm_` 正确；Stage3 对 `vInner` 更敏感，仍需单独验证 final `INV @ v_beta`。
- 当前精度 PASS 依赖 `vinner final` fallback，不代表 raw final fused 已修复。

## 下一动作

- 若目标是继续性能/fused 调试：新增或确认不参与 Stage2/Stage3 的安全 scratch，重新打开 raw `vinner final` 后隔离写回副作用。
- 若目标是提交精度修复：保留当前 fallback 配置，并在提交说明中明确 raw final 仍未完成。
- 后续逐步关闭 fallback 的顺序仍建议为：`vinner final`、`kcum final`、compose、power、base inverse。

## 交接点

- 可直接从当前源码继续，不需要重做 2026-05-01 的 L1/L0 布局排查。
- 下一步优先级：
  1. 不使用 `tmpWsGm_` 做 debug scratch。
  2. 为 raw final 增加安全 scratch 或直接修 `RawMmad64ByN()` 写回/同步。
  3. 重新打开 raw `vinner final` 后，先让关键 case 通过。
  4. 再关闭 `kcum final`、compose、power、base inverse fallback。
  5. 每轮最终运行全量 `CGDR_GOLDEN_CHUNK_SIZE=64 bash tests/pytest/test_run.sh compare`。
