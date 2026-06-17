# GOAL — 总体目标

> 总体目标导向文档，牵引核心方向不偏离，保证跨会话目标一致。

## 项目背景

ops-transformer 项目中的 GMM（Grouped MatMul）算子已支持 Atlas A2/A3 系列产品的 A16W4(INT4) 伪量化数据流。现需将该数据流扩展至 **Ascend 950PR/Ascend 950DT** 平台，补充 Ascend950 场景下的 A16W4 weight ND 数据流支持。

当前 Ascend 950 的 arch35 `weight_quant_basic_block` 体系已有 A16W8 多多多的实现（`GMMWeightQuantBasicController`），本任务参照该实现，适配 A16W4 pergroup 场景。

**范围说明**：本任务仅关注 **tiling 和 kernel** 部分的实现。Aclnn 通路（API checker、infershape）由其他人完成。

## 总体目标

为 GMM 算子在 **Ascend 950** 平台上实现 A16W4(INT4) weight ND 数据流的 **tiling + kernel** 部分：

1. **Tiling**：为 A16W4 pergroup ND 场景生成正确的 tiling 参数（`GMMWeightQuantParam.groupSize` 等）
2. **Kernel**：在 arch35 `weight_quant_basic_block` 体系中适配 A16W4 pergroup 计算逻辑
3. **参照实现**：`GMMWeightQuantBasicController`（A16W8 多多多），大部分代码可能不需要修改
4. **支持两种 tensor 输入模式**：
   - 单单单：x 单tensor、weight 单tensor、out 单tensor（groupType=0）
   - 多多多：x 多tensor、weight 多tensor、out 多tensor（groupType=-1）
5. **weight 支持转置和非转置**
6. **group_size 仅支持 32/64/128/256**，K 轴为 group_size 的倍数
7. **兼容现有 A2/A3 场景**：不破坏已有功能

## 实现方案要点

- **Tiling 策略**：参考 GMM A16W8 多多多策略，x: `256*256` L1切分，weight: `256*256`，kL1=256
- **Kernel 参照**：`GMMWeightQuantBasicController`，用多多多计算方式，同时实现单单单
- **Cube 计算**：主要流程与 A16W8 多多多方案一致
- **Vec 反量化**：写 NK/KN 两份功能实现代码，参考 wqbmmv2 的 weight ND int4 反量化方法（`AntiquantComputeW4NKPerGroup` 和 `AntiquantComputeKNPerGroup`）
- **UB 划分**：参考 GMM A16W8 实现，暂时不修改
- **地址偏移**：每个基本块计算完成后地址更新按 int4 计算字节偏移，字节大小需要减半

## 约束与成功标准

### 硬约束
- **不做** Aclnn 通路（API checker、infershape）— 由其他人负责
- 不修改已有 A2/A3 场景的代码逻辑和测试用例
- weight 格式为 ND（非 NZ）
- group_size 严格限制为 32/64/128/256
- K 轴必须为 group_size 的整数倍
- antiquant_scale shape: `(e, k/gs, n)`
- antiquant_offset shape: `(e, k/gs, n)`（可选）

### 成功标准
- [x] 算子规格文档（spec/gmm_a16w4nd_op.md）完成 ✅
- [x] 代码走读文档完成（GMM A16W8 + wqbmmv2 ND 反量化） ✅
- [x] SRS 算子开发实现方案文档完成（tiling data、tiling 计算、L1/UB 空间划分、kernel 实现方案、核心代码适配点） ✅
- [x] Tiling 计算：正确生成 A16W4 ND pergroup 场景的 tiling 参数 ✅（37/37 UT 通过）
- [x] Kernel 实现：A16W4 ND pergroup 数据流在 Ascend 950 上编译通过 ✅（0 errors, 42/42 审查通过）
- [ ] 单元测试（UT）：覆盖 tiling 和 kernel 的正确性（需 NPU 硬件环境）
- [ ] 兼容性验证：已有 950 测试全部通过，无回归（需 NPU 硬件环境）

## 相关链接

- [实现计划](./PLAN.md)
- [用户下一步计划](./NEXT_STEPS.md)
- [检查清单](./CHECKLIST.md)
- [变更记录](./CHANGELIST.md)
- [原始需求](./raw/spec/gmm_a16w4_950_requirement.md)
- [接口规格](../gmm/grouped_matmul/docs/aclnnGroupedMatmulV5.md)
