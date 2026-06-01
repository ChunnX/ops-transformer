# scatter_kv_cache_with_k_scale TTK 测试结果报告（第1轮）

## 1. 测试概览

| 测试类型 | 总用例数 | 通过 | 失败 | 通过率 | 状态 |
| -------- | -------- | ---- | ---- | ------ | ---- |
| 黑盒测试 | 1        | 0    | 1    | 0.0%   | ❌ 阶段1 冒烟失败，终止 |
| 白盒测试 | —        | —    | —    | —      | ⏭️ 跳过（黑盒阶段1 失败） |
| 网络测试 | —        | —    | —    | —      | ⏭️ 跳过（CSV 不存在） |
| **合计** | **1**    | **0** | **1** | **0.0%** | |

> ⚠️ 黑盒阶段1 冒烟测试（--ti=0）精度验证失败，按照 op_ttk_skill 规定立即终止整个 TTK 测试。

## 2. 失败用例列表

| 用例ID | 测试类型 | 失败模式 | bin_precision | dyn_precision | 失败原因 | 备注 |
|--------|----------|----------|---------------|---------------|----------|------|
| test_001 | blackbox | 精度误差 | 100.0%,100.0%,12.5% | 100.0%,100.0%,12.5% | k_scale_out 输出全部为 1.0 而非预期 scale factor | 前两个 FP8 输出精度 100% |

## 3. 精度分析

### 3.1 分输出精度

| 输出 | dtype | 精度 | 状态 |
|------|-------|------|------|
| k_cache_out（inplace） | float8_e5m2 | 100.0% | ✅ |
| v_cache_out（inplace） | float8_e5m2 | 100.0% | ✅ |
| k_scale_out | float32 | 12.5% | ❌ |

### 3.2 k_scale_out 详细差异

实际输出全部为 1.0，预期值分布如下：

| 预期值 | 实际值 | Diff |
|--------|--------|------|
| -56.697 | 1.0 | 1.018 |
| 47.239 | 1.0 | 0.979 |
| -1.248 | 1.0 | 1.801 |
| -100.000 | 1.0 | 1.010 |
| -98.454 | 1.0 | 1.010 |
| 100.000 | 1.0 | 0.990 |
| 61.633 | 1.0 | 0.984 |

- **最大 diff**：1.801311（远超容忍阈值 0.01）
- **容忍判断**：TOLERATE_FAIL

### 3.3 门禁验证

| 验证步骤 | 结果 |
|----------|------|
| 步骤 A（BIN_GOLD/DYN_GOLD 全线 100%） | ❌ 第三个输出仅 12.5% |
| 步骤 B（max_diff ≤ 0.01） | ❌ max_diff=1.801311 |

## 4. 算子包信息

| 项目 | 内容 |
|------|------|
| 目标 SoC | ascend950（Ascend950PR_9579） |
| 产物路径 | `${OPS_REPO}/build_out/cann-ops-math-custom_linux-x86_64.run` |
| 安装路径 | `/home/developer/Ascend/cann-9.1.0-beta.1/opp/vendors/custom_math/` |
| vendor 注册 | `/home/developer/Ascend/cann-9.1.0-beta.1/opp/vendors/config.ini`（load_priority=custom_math） |
| 安装状态 | ✅ 已安装（手动解压方式绕过安装脚本验证失败） |
| 安装备注 | .run 安装脚本因符号不匹配（`AscendString::GetString`）验证失败，改用 dd+tar 手动解压 |

## 5. 失败归因

> ⚠️ 以下归因供主 agent 决策参考，op_test 未修改任何代码。

| 类别 | 问题描述 | 失败原因 | 归因 |
|------|----------|----------|------|
| 算子代码 | k_scale_out 输出全部为 1.0，非预期的 scale factor | 算子的 k_scale_out 计算/写入逻辑未正确实现。FP8 cache（inplace）的 scatter 操作正确（精度 100%），但 scale 输出始终为 1.0，可能是：(1) k_scale_out 计算分支未执行或未被正确调用；(2) scale 值计算逻辑有 bug；(3) 输出内存被初始化为 1.0 且未被覆写 | **op_dev** |

## 6. 运行环境

| 项目 | 内容 |
|------|------|
| CANN 版本 | 9.1.0-beta.1 |
| NPU 型号 | Ascend950PR_9579 |
| NPU ID | 0，健康状态 OK |
| TTK 版本 | 3.0.0（Rev: 1771b48） |
| 测试模式 | ASCEND_ONBOARD |

## 7. 日志文件

| 日志类型 | 路径 |
|----------|------|
| 黑盒冒烟日志 | `/tmp/scatter_kv_cache_with_k_scale_blackbox_stage1_smoke.log` |
| TTK 比对日志 | `/workspace/ops-test-kit/ttk-compare.log` |
| 黑盒结果 CSV | `/workspace/gen_simt_code/outputs/scatter_kv_cache_with_k_scale/test_case/03_scatter_kv_cache_with_k_scale_blackbox_tse_result.csv` |

## 8. 总结

- **整体通过率**：0.0%
- **黑盒通过率**：0.0%（1/1 失败）
- **白盒通过率**：未执行
- **网络通过率**：未执行（CSV 不存在）
- **测试结论**：❌ **测试失败**

黑盒冒烟测试中，inplace 的 FP8 cache scatter 操作精度正常（100%），但 k_scale_out（float32）输出全部为 1.0 而非预期的 scale factor，最大 diff 达 1.80，远超容忍阈值。

**建议**：主 agent 应将此问题反馈给 op_dev，重点排查 k_scale_out 的计算和写入逻辑。
