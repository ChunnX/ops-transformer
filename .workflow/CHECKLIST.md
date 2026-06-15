# CHECKLIST — 检查清单

> 各阶段的质量检查项。仅关注 tiling + kernel。
>
> **审查要求**: 使用 subagent 在独立上下文中客观检查。发现问题必须诚实报告。

## 审查原则

- 基于事实，不基于假设
- 可验证，可复现
- 明确标注：✅ 通过 / ❌ 不通过 / ⬜ 不适用

## 阶段一：代码走读

### 走读完整性
- [ ] `GMMWeightQuantBasicController` 完整流程记录（Init → Process → SetMKN → SetGmAddr → ComputeBasicBlock）
- [ ] `SetGmAddr` 中 perchannel 地址偏移公式已明确
- [ ] pergroup 与 perchannel 的地址偏移差异已分析
- [ ] `WqmmConfig` 配置项及 `antiQuantType` 影响已记录
- [ ] `int4b_t` weight 特殊处理已记录
- [ ] Tiling 中 `groupSize` 设置逻辑已明确

### 设计方案
- [ ] 设计方案覆盖 Controller/BasicBlock/Tiling 各层修改点
- [ ] pergroup 地址偏移公式正确
- [ ] 转置/非转置处理方案明确
- [ ] 兼容性方案明确（不影响已有 A16W8）

## 阶段二：Tiling

### Tiling 参数
- [ ] `GMMWeightQuantParam.groupSize` 正确设置（32/64/128/256）
- [ ] mList/kList/nList 正确填充
- [ ] singleX/singleWeight/singleY 正确标识
- [ ] groupType/groupListType 正确传递
- [ ] tiling UT 通过

### Tiling UT
- [ ] 覆盖 group_size=32/64/128/256
- [ ] 覆盖转置/非转置
- [ ] 覆盖单单单/多多多
- [ ] 已有 tiling UT 无回归

## 阶段三：Kernel

### Controller 适配
- [ ] `SetGmAddr` 正确处理 pergroup antiquantScale 偏移（`+= G_i * N_i`）
- [ ] `SetGmAddr` 正确处理 pergroup antiquantOffset 偏移
- [ ] 转置/非转置场景均正确
- [ ] 不影响已有 A16W8 perchannel 逻辑

### BasicBlock/VecCompute 适配
- [ ] pergroup 反量化计算正确
- [ ] scale/offset 按 VF tile 粒度搬运，单缓冲不超过 1KB
- [ ] `bTrans=true` 时 scale/offset 固定按 `[G,N]` 输入处理，NK VF 使用 `g*nTileAlign+n` 地址公式
- [ ] `groupSize=32/64` 的 NK VF 使用 broadcast + select 构造跨 group 的 scale/offset
- [ ] `hasAntiQuantOffset=true` 时 PER_GROUP 路径正确搬运 offset 到 UB
- [ ] INT4 weight unpack 和 cast 正确
- [ ] 各 group_size (32/64/128/256) 计算正确
- [ ] WqmmConfig 配置正确

### Kernel UT
- [ ] 覆盖转置/非转置
- [ ] 覆盖各 group_size
- [ ] 覆盖单单单/多多多
- [ ] FLOAT16 和 BFLOAT16 覆盖
- [ ] 已有 kernel UT 无回归

## 阶段四：集成验证

### 兼容性
- [ ] Ascend 950 已有 UT 全部通过
- [ ] 无编译告警新增

## 通用检查项

### 代码规范
- [ ] 文件命名与注释匹配
- [ ] Header guard 与文件名匹配
- [ ] 无遗留 TODO/FIXME 未处理
- [ ] 代码风格与现有代码一致

### 功能验证
- [ ] 编译通过
- [ ] 测试通过
- [ ] 不影响已有功能
