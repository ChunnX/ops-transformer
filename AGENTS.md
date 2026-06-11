# Compressor 算子仓协作指南

本文件适用于 `ops-transformer` 仓库，当前优先范围为 `experimental/attention/compressor/`。同时遵循上层工作区 `/home/jinxh/deepseek-v4/AGENTS.md`。

## 目录职责

- `op_host/compressor_def.cpp`：公开输入、输出、属性和 SoC 注册。
- `op_host/compressor_proto.cpp`：shape/type inference。
- `op_host/arch22/`：A2/A3 host tiling；当前 DeepSeek-V4 目标主要是 A3。
- `op_host/arch35/`：950/A5 host tiling。
- `op_kernel/arch22/`：A3 kernel，包含普通与 full-load/perf 路径。
- `op_kernel/arch35/`：950/A5 kernel，包含普通与 full-load 路径及 VF 实现。
- `op_kernel/compressor.cpp`：按 SoC、tiling key 分派 kernel。
- `tests/pytest/`：CPU golden、单 case 和批量 case。
- `README.md`：公开算子契约与调用示例。

## 修改规则

1. 先确认目标 SoC、layout、ratio、head dim 和 template id，再定位具体分支。
2. 修改 `CompressorTilingData`、template key 或枚举时，同步检查对应 host 和 kernel 使用点。
3. arch22 与 arch35 不做机械复制。公共语义必须一致，硬件相关切分和 VF/Cube 策略允许不同。
4. 保留 state cache 的原地更新语义；特别检查 stride、block table 映射和 cache mode。
5. 新增 fast path 必须有明确适用条件和稳定 fallback，不能让非目标 shape 落入错误模板。
6. 影响数值归约顺序时，分别验证输出 `cmp_kv` 和更新后的 `state_cache`。
7. OpDef、README、Torch 适配层和模型调用中的 dtype/shape 必须一致；发现差异先记录并修复契约，不依赖隐式转换。

## 性能分析最小记录

每个实验至少保存：

- SoC 与软件版本。
- git commit 和未提交 diff。
- B/T/S/H/D、ratio、coff、layout、start position、有效序列长度。
- tiling key、template id、block dim、workspace、是否 full load。
- warmup/迭代次数、均值或中位数、P50/P90（可获取时）。
- 精度阈值及 `cmp_kv`、state cache 是否通过。

没有 profiler 或重复测量数据时，不以代码复杂度推断性能收益。

## 测试矩阵

至少覆盖以下主干 case：

| 场景 | layout | D | ratio | coff |
|---|---|---:|---:|---:|
| C4A prefill | TH | 512 | 4 | 2 |
| C4Li prefill | TH | 128 | 4 | 2 |
| C128A prefill | TH | 512 | 128 | 1 |
| C4A decode | BSH | 512 | 4 | 2 |
| C4Li decode | BSH | 128 | 4 | 2 |
| C128A decode | BSH | 512 | 128 | 1 |

补充边界：batch 1/多 batch、跨 block 边界、`seqused=0`、非整 ratio、不同 `start_pos`、TH 非等长 batch、state cache 第 0 维非连续（支持的平台）。

## 构建与测试

```bash
# A3
bash build.sh --pkg --experimental --soc=ascend910_93 --ops=compressor

# 950/A5
bash build.sh --pkg --experimental --soc=ascend950 --ops=compressor

# 已安装对应自定义算子包后运行单算子测试
cd experimental/attention/compressor/tests/pytest
bash test_run.sh single
```

构建和 NPU 测试依赖完整 CANN 环境。若环境不满足，只做可执行的静态检查，并在结果中列出未完成项。

## 提交边界

- 算子内核、host tiling 和本仓测试属于本仓提交。
- Torch schema/eager/Meta/GE converter 与模型调用属于 `cann-recipes-infer` 提交。
- 两仓提交说明应互相引用同一任务或实验编号，确保接口变更可追踪。
