# Commit Message Convention

本项目所有 commit 必须遵循以下格式。

## 格式

```
<type>[optional scope]: <subject>

[optional body]

[optional footer]
```

## Type

| Type | 用途 | 示例 |
|------|------|------|
| feat | 新功能 | `feat: 支持 MXFP8 场景下 D=256` |
| fix | Bug 修复 | `fix: D256 workspace 分配不足` |
| test | 仅测试相关 | `test: 添加 KVS=256 用例` |
| revert | 回退 | `revert: 回退 splitD 修改` |
| refactor | 重构（不改变行为） | `refactor: 抽取 BMM2 切分逻辑` |
| style | 格式调整 | `style: clang-format 格式化` |
| docs | 文档 | `docs: 补充 D256 设计说明` |
| perf | 性能优化 | `perf: BMM2 结果写 UB 减少搬运` |
| chore | 杂项（CI/构建） | `chore: 裁剪非 Mxfp8 模板实例化` |

## Subject（标题行）

- 中文描述，不超过 72 字符
- 不以句号结尾
- 首字母不要求大写
- 简洁说明"做了什么"，不解释"为什么"（为什么放 body）

## Body（可选）

- 解释"为什么"和"改了什么"
- 与 subject 之间空一行
- 每行不超过 72 字符

## Footer（推荐填写）

测试结果信息（fix/test 类 commit 应填写）：

### 必填字段

- **编译**: OK / 失败(错误摘要)
- **改动前基线**: 用例名 + 通过率 + 错误特征
- **改动后实测**: 用例名 + 通过率 + 错误特征变化

### 推荐字段

- **ErrorAxis**: 失败集中在哪个轴(T/N/D)、first_fail 位置
- **D128 回归**: 100% pass 无回归 / 回归失败(摘要)
- **pytest 输出**: 逐 case 的 pass/fail/skip 和百分比

### 完整示例

```
Test:
- 编译: OK
- 改动前基线: D256 QS128 25.10% pass, T=32..127全错
- 改动后实测: D256 QS128 38.20% pass, T=32..63修复, T=64..127仍有错
- ErrorAxis: axis0(T) first_err=64, axis2(D) range=[0,255]全对
- D128 回归: 100% pass, 无回归
- pytest 输出:
  TND_B1_QS128_KVS128_D256_SP3: 38.20% pass (49/128)
  TND_B1_QS128_KVS128_D128_SP3: 100% pass (128/128)
```

### 最小示例（测试还没跑完时）

```
Test:
- 编译: OK
- 改动前基线: D256 QS128 25.10% pass
- Status: pending A5 test
```

Co-authored 标记（agent 生成时）：

```
Co-Authored-By: Claude <noreply@anthropic.com>
Co-Authored-By: OpenCode <noreply@opencode.ai>
```

## 示例

```
fix: D256 单子轮 PScale 双拷贝步长不足导致重叠

单子轮路径(actSingleLoopS2Size<=s2SplitSize)拷贝2份PScale到L1，步长用
s2BaseSizeCur。D128时s2BaseSizeCur=256=单次拷贝量，两份不重叠；
D256时s2BaseSizeCur=128<256，两份重叠[128,256)并溢出槽，破坏MM2读取的PScale。

按pScaleDataLen与单次拷贝量决定份数：D64/D128拷2份，D256只拷1份。

Test:
- 改动前基线: 1.66% pass
- 改动后实测: 13.68% pass
- Status: 部分修复，主因待定位
```

```
test: 加 D128 QS128 镜像用例用于证伪根因

Co-Authored-By: Claude <noreply@anthropic.com>
```

## Scope（可选）

- 用英文，短横线分隔（如 `d256-bmm2`、`kv-rms-norm`）
- 跨模块改动加 scope，单模块内改动可省略
- 当前 D256 分支工作默认不加 scope

## Revert 规范

- subject 格式：`revert: <原 commit subject 摘要>`
- **必须**在 body 说明回退原因和回退后的状态
- 不要用 `git revert` 自动生成的 `Revert "..."` 格式，改为手写 `revert:` 格式
- 不要用 `Reapply` 作为 type，用 `fix: reapply <原 commit 摘要>`

## 调试探针 / 插桩

- 临时插桩（PRINTF/DumpTensor/dump）用 `test: 加 ... 探针` 或 `chore: 加 ... 插桩`
- 不要用 `debug:` 作为 type（不在合法列表里）
- 探针 commit 应该在问题定位后 revert 掉，不要留在主干

## Co-Authored-By（MANDATORY）

AI agent 生成的 commit **必须**加 footer 声明身份：
```
Co-Authored-By: Claude <noreply@anthropic.com>
Co-Authored-By: OpenCode <noreply@opencode.ai>
```
人工写的 commit 不加。目的是 `git log` 时能区分人工和 AI 提交。

## 规则

1. 每个 commit 只做一件事，不要把多个不相关修改混在一个 commit
2. 不要写 "update" / "修改" / "update code" 这种无意义 subject
3. revert 的 subject 用 `revert: 回退 <原commit的subject>`
4. type 必须从合法列表中选，不要自创新 type
5. body 必须说清"为什么这么改"，不仅是"改了什么"：
   - **根因分析**: 什么现象 → 推断什么根因 → 证据是什么
   - **方案选择**: 为什么选方案 A 而非 B/C（尤其是试错后）
   - **D128 兼容性**: D128 为什么不受影响（如果涉及 D256 修改）
6. revert 的 body 必须说明回退原因（不能只写 `This reverts commit xxx`）：
   - 原方案为什么失败
   - 回退后的预期状态
   - 下一步打算做什么
