# attn_res 正反向算子测试方法与当前任务状态

## 1. 背景

- **正向算子**：`block_attention_residuals`（`attention/block_attention_residuals/`），输出 `hidden` + 保存给反向的中间量 `inv_norm`、`probs`
- **反向算子**：`attention_res_grad`（`attention/attention_res_grad/`），输入 `grad_output` + `inv_norm` + `probs`，输出 `grad_prefix_sum`、`grad_block_res`、`grad_proj_weight`、`grad_norm_weight`
- **torch_extension 绑定**：[attn_res.py](../../torch_extension/cann_ops_transformer/ops/attn_res.py) 通过 `torch.autograd.Function` 将正反向绑定为单一可微算子，用户只需调用正向 + `.backward()` 即得全部梯度
- **平台**：Ascend 910B（`ascend910b`），算子包通过 `build_install_attn_res.sh` 一键构建安装

## 2. 测试方法

### 2.1 测试脚本

[test_attn_res_bound.py](test_attn_res_bound.py) — 正反向精度测试（NPU float32 autograd 标杆）。

### 2.2 四种测试模式（`--mode`）

| 模式 | 含义 | 对比项 |
|------|------|--------|
| `bound` | 绑定接口一次调用（fwd+bwd） | fwd.hidden + 4 梯度 vs golden |
| `fwd` | 单独测正向 | fwd.hidden / fwd.inv_norm / fwd.probs vs golden |
| `bwd` | 单独测反向（手动两段式：fwd 算子产出 inv_norm/probs 后手动调 attention_res_grad） | 4 梯度 vs golden |
| `vs` | 同输入对照：手动两段式 vs 绑定自动反向 | 互相比对 + 各自 vs golden（12 项） |

### 2.3 参数说明

| 参数 | 默认 | 说明 |
|------|------|------|
| `--B/--N/--H` | 8 / 4 / 7168 | 张量形状（B 为 batch，N 为 block 数，H 为隐藏维） |
| `--seed` | 0 | 随机种子（**seed=0 为问题复现种子**） |
| `--dtype` | bf16 | bf16 / fp16（当前重点为 bf16） |
| `--grad-scale` | 1.0 | 反向输入梯度放大倍数，用于暴露数值问题（1000 可复现） |
| `--show-points` | - | PASS 时也打印最大误差点及前后 2 点 |
| `--log` | 脚本目录 test_attn_res_bound.log | 每次 compare 结果同时打印终端并追加保存到日志 |

### 2.4 精度体系

- 复用仓内 [precision_compare.py](tests/pytest/precision_compare.py) 的 `data_compare`，与 [attenres_lite.py](tests/pytest/attenres_lite.py) 一致
- 阈值：diff / pct / max 三阈值均为 BF16 MERE `1/128`（2^-7）
- 判定：`result == "success"` 且通过率满足阈值即 PASS

### 2.5 golden 实现

- 标杆为 **NPU float32 autograd**：`v = cat(block_res, prefix_sum.unsqueeze(1))` → variance → rsqrt → softmax → matmul 得 hidden，再 `hidden.backward(grad_output)` 得 4 梯度，数学等价于 `attenres_lite.apply_attn_res`
- inv_norm/probs golden 同步返回，用于 fwd 模式比较保存给反向的中间量

### 2.6 运行方式

```bash
# 环境准备（顺序固定：先 conda 后 CANN）
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate kdy
source /workspace/z00889690/Ascend_env/cann-9.1.0-beta.3/set_env.sh
source <安装路径>/vendors/custom_transformer/bin/set_env.bash   # 默认 /data/kdy/gentest/custpkg

# 运行
python3 test_attn_res_bound.py --dtype bf16 --H 7168 --seed 0 --mode bound
python3 test_attn_res_bound.py --dtype bf16 --H 7168 --seed 0 --mode vs --grad-scale 1000
```

## 3. 已确认的测试结果

| 场景 | 结果 |
|------|------|
| BF16 H=7168，bound/fwd/bwd × seed 0/1/2 | **全部 PASS，max_rel_err=0** |
| vs 模式（grad_scale=1） | 12 项全 PASS |
| **grad_scale=1000 seed=0（bwd/bound/vs）** | **修复后全部 PASS（见 §4.5）** |
| grad_scale=1000 seed=1/2（bwd） | 全部 PASS（偶发） |

## 4. 反向 kernel 精度问题定位（已完成）

### 4.1 现象

反向输入梯度放大 1000 倍后，`grad_block_res` / `grad_proj_weight` / `grad_norm_weight` 相对误差飙升至 ~100%（通过率 82.9%~99.5%），且 **seed=0 复现、seed 1/2 通过**（偶发，与数据分布相关）。

### 4.2 已排除

- ✅ golden 无误：autograd golden 与手写推导结论一致；BF16 原始幅度下全 PASS
- ✅ 绑定封装无误：`vs.manual_vs_bound` 4 项 max_rel_err=0，手动两段式与绑定自动反向位级一致
- ✅ 误差不在 torch 层：两条调用路径对 golden 产生完全相同误差，全部来自 kernel 内部

### 4.3 关键观察

- `grad_prefix_sum` 恒 PASS（含 grad_scale=1000）——它是 per-token 直出（`fb2[:, N, :]`），**不经过跨核 reduce**
- 所有失败项均依赖跨核 reduce：
  - `grad_proj_weight` / `grad_norm_weight`：`Σ sc·v` 全量求和（[ReduceAndWriteWeights](../attention_res_grad/op_kernel/attention_res_grad.h#L452-L514)）
  - `grad_block_res`：列块相关求和

### 4.4 定位过程（PRINTF / DumpTensor 插桩）

按"每核 2 token、B=1 先调通、H=7168 固定"策略，通过 `[ARG_DBG]` plog 打印 kernel 中间量逐步比对 golden：

- **VAddr 偏移 bug（根因一）**：`VAddr()` 按 `(batch*numBlocks_+blk)` 元素偏移寻址，漏乘 `hiddenSize_`，blk≥1 全读错 GM 地址。修复后 batch0 的 gp/scalar/gs/sumSWV/gradInvRms/varScale 与 bf16 手写 golden 完全一致
- **MTE3→MTE2 halfBuf 竞态（根因二）**：`ReduceAndWriteWeights()` 写 grad_norm_weight（MTE3 读 halfBuf）后，下一段立即 MTE2 覆写同一 halfBuf 读 norm_weight，无 `MTE3_MTE2` 同步，grad_norm_weight 被污染成 grad_proj_weight 内容。加 `SetFlag/WaitFlag(MTE3_MTE2)` 修复
- **inv_norm/probs 被 BF16 量化（根因三）**：正向 `block_attention_residuals` 输出 FP32 的 inv_norm/probs，反向绑定却先 cast 成主 dtype（BF16）再喂 kernel。softmax 反向存在灾难性消减（gp/scalar~1e4，差~1e-2），probs 的 BF16 量化误差直接导致 gsw 符号翻转（对比：op vs bf16-golden 100% vs op vs fp32-golden 仅 15.18%、4901/7168 符号翻转）

### 4.5 修复与最终结果

三处修复：
1. [attention_res_grad.h](../attention_res_grad/op_kernel/attention_res_grad.h) `VAddr()` 补乘 `hiddenSize_`
2. [attention_res_grad.h](../attention_res_grad/op_kernel/attention_res_grad.h) `ReduceAndWriteWeights()` 加 MTE3→MTE2 同步
3. **inv_norm/probs 输入改为 FP32**：
   - [attention_res_grad_def.cpp](../attention_res_grad/op_host/attention_res_grad_def.cpp) `inv_norm`/`probs` DataType → `{DT_FLOAT, DT_FLOAT}`
   - [attention_res_grad.cpp（torch 绑定）](../../torch_extension/cann_ops_transformer/ops/csrc/attention_res_grad.cpp) 不再 cast 到主 dtype，直通 FP32（非 float 时防御性 `to(kFloat)`）
   - [attention_res_grad.h](../attention_res_grad/op_kernel/attention_res_grad.h) kernel 内 `probsGm`/`invNormGm` 改 `GlobalTensor<float>`，CopyInToken 直接搬 FP32 到 float buffer，去掉 half 缓冲与 Cast

最终回归（调试宏已全部关闭，重新构建安装）：
- bwd B=1/B=2 × seed 0/7 × grad_scale=1000：**全 PASS，pct=100%，max_rel_err=0**
- bound 全一输入：全 PASS
- mid 模式（中间量一致性 + 两组反向互比 + vs golden）：14 项全 PASS

## 5. B=16000 大 batch 测试与 golden 随 B 漂移（已完成）

### 5.1 现象

grad_scale=1000 下 B=16000（bwd 模式）4 梯度全 FAIL，通过率 ~99.94%~99.9997%，个别元素 max_rel=1.28~1.94：

| 梯度 | pct | max_rel |
|------|-----|---------|
| grad_prefix_sum | 99.999621% | 1.2759 |
| grad_block_res | 99.999715% | 1.2772 |
| grad_proj_weight | 99.944196% | 0.01938 |
| grad_norm_weight | 99.958147% | 0.0226 |

### 5.2 定位过程

- dump B=16000 全量数据定位误差最大 batch：**batch=4966**（grad_prefix_sum col=5012：npu=-0.001221 vs gold=0.001020，符号翻转）
- 该 batch 单独以 B=1 执行：**未复现**（kernel vs golden max_rel=0.0039 < 阈值 0.0078 → PASS），故按既定策略无需 kernel 加打印
- 进一步验证 golden 标杆随 B 漂移：
  ```
  golden 漂移   (g_ps1 vs g_psB): diff>0=6964/7168 max_rel=1.8359 翻转=1
  kernel 漂移  (op_ps1 vs op_psB): diff>0=0/7168     max_rel=0.0000 翻转=0
  B=1    kernel vs golden:  max_rel=0.0039 翻转=0   ← PASS
  B=16000 kernel vs golden: max_rel=1.8359 翻转=1   ← FAIL
  ```

### 5.3 结论

B=16000 FAIL 为 **golden 标杆（torch autograd）随 B 批次大小浮点实现漂移** 造成的伪差：

- kernel 输出确定、不随 B 漂移（max_rel=0，翻转=0）
- golden 的 reduce 归约顺序随 B 变化（torch 顺序求和 vs kernel half-interval fold），B=1 时两实现数值一致，B 增大后 golden 自身漂移、个别小值元素符号翻转（grad_scale=1000 放大）
- kernel 在 B=1 即验证通过，无需 kernel 加打印修改

### 5.4 调试代码清理（本次已完成）

kernel 侧调试代码全部删除并重新构建安装回归通过：

- 删除 [attention_res_grad.h](../attention_res_grad/op_kernel/attention_res_grad.h) 中全部调试宏（ARG_DEBUG_DUMP / ARG_DEBUG_GM / ARG_DBG2）、`DebugWriteGm()`、18 个调试成员变量（dbgBuf/dbgGp_/dbgScalar_/dbgGs_/dbgProbBlk_ 等）及 14 处 `#if` 调试块（PrecomputeGradProbs / SoftmaxBackwardN1 / ComputeGradV / ReduceAndWriteWeights / Process）
- grep 确认 `ARG_DEBUG|ARG_DBG|dbgBuf|dbgGp|dbgScalar|DebugWriteGm|PRINTF|DumpTensor` 无残留
- 重新构建安装（`bash build.sh --pkg --ops=block_attention_residuals,attention_res_grad --soc=ascend910b` → 安装到 /data/kdy/gentest/custpkg）后回归：
  - bwd B=2 × grad_scale=1000：4 梯度全 PASS，pct=100%，max_rel_err=0
  - mid 模式（grad_scale=1000）：14 项全 PASS
