# AttnResFwd Torch 验证环境配置

本文档汇总 AttnResFwd 算子通过 `torch_extension` / `cann_ops_transformer` 做 NPU 精度验证的环境配置与常用命令。内容来自历史对话中已验证可用的配置。

## 1. 前置条件

| 组件 | 要求 | 说明 |
|------|------|------|
| Python | **3.11** | conda 环境 `kdy311` |
| PyTorch | 2.10.0 | `kdy311` 已安装 |
| torch_npu | 2.10.0.post2 | CANN 9.x 需 torch_npu ≥2.10；py3.8 的 `kdy` 仅 torch-npu 2.4，**不可用** |
| CANN | 默认 `cann-9.2.0-weekly.20260715.01-20260715` | 可通过 `CANN_PKG` 切换 |
| 自定义 OPP | `custom_transformer` | 含 `AttnResFwd` kernel，须先编译安装 |
| NPU | Ascend 910B | 测试前自动选空闲卡 |

> **注意**：ACLNN example 测试（`test_run.sh prod`）与 Torch 测试（`test_run.sh torch`）使用同一套 CANN + custom OPP，但 Torch 路径**必须**使用 `kdy311`。

## 2. 一键环境加载

```bash
cd /data/kdy/cann/ops-transformer/attention/attn_res_fwd/tests/pytest

# Step 1: conda（默认 kdy311）
source env_kdy.sh

# Step 2: CANN + custom OPP + PYTHONPATH
source cann_env.sh

# 可选：手动指定 NPU 卡（默认 npu-smi 自动选空闲卡）
export DEVICE_ID=6
```

### 2.1 env_kdy.sh

激活 conda 环境，默认 `kdy311`：

```bash
CONDA_ENV_NAME="${CONDA_ENV_NAME:-kdy311}"
source /data/miniconda3/etc/profile.d/conda.sh
conda activate kdy311
```

### 2.2 cann_env.sh

加载 CANN toolkit 与自定义算子包：

```bash
# 默认 CANN 9.2 weekly；可切换 beta.3：
# export CANN_PKG=cann-9.1.0-beta.3-20260722

source /data/kdy/cann/pkg/${CANN_PKG}/ascend-toolkit/set_env.sh
source /data/kdy/cann/custompkgbase/vendors/custom_transformer/bin/set_env.bash

export PYTHONPATH="/data/kdy/cann/ops-transformer/torch_extension:${PYTHONPATH}"
export ASCEND_GLOBAL_LOG_LEVEL=4
export ASCEND_SLOG_PRINT_TO_STDOUT=0
```

加载后关键变量示例（CANN 9.2 weekly）：

| 变量 | 典型值 |
|------|--------|
| `ASCEND_HOME_PATH` | `/data/kdy/cann/pkg/cann-9.2.0-weekly.../cann-9.2.0` |
| `ASCEND_OPP_PATH` | `.../cann-9.2.0/opp` |
| `ASCEND_CUSTOM_OPP_PATH` | `.../custompkgbase/vendors/custom_transformer:...` |
| `ATTN_RES_FWD_CUSTOM_SETENV` | `.../custom_transformer/bin/set_env.bash` |
| `PYTHONPATH`（前缀） | `.../ops-transformer/torch_extension` |

## 3. 编译安装自定义算子包

每次修改 `op_kernel` 后须重新编译安装：

```bash
cd /data/kdy/cann/ops-transformer

bash build.sh --pkg --soc=ascend910b --ops=attn_res_fwd -j16

build_out/cann-ops-transformer-custom_linux-aarch64.run \
  --install-path=/data/kdy/cann/custompkgbase --quiet --force
```

安装完成后重新 `source cann_env.sh`。

## 4. Torch 扩展文件布局

```
torch_extension/
├── cann_ops_transformer/
│   ├── ops/
│   │   ├── attn_res_fwd.py          # OpBuilder + Python API
│   │   └── csrc/attn_res_fwd.cpp    # ACLNN wrapper (aclnnAttnResFwd)
│   └── op_builder/builder.py        # include_paths 仅加 aclnn/aclnnop
└── ...

attention/attn_res_fwd/tests/pytest/
├── env_kdy.sh                       # conda kdy311
├── cann_env.sh                      # CANN + OPP + PYTHONPATH
├── npu_device.py                    # npu-smi 自动选卡
├── attn_res_fwd_torch_runner.py     # 隔离 OpBuilder JIT 加载
├── attn_res_fwd_golden.py           # CPU golden + 精度比较
├── test_attn_res_fwd_torch.py       # Torch 精度 pytest
├── test_attn_res_fwd_paramset.py    # 用例表
└── test_run.sh                      # 统一入口
```

### 调用方式

```python
from cann_ops_transformer.ops.attn_res_fwd import attn_res_fwd

hidden = attn_res_fwd(prefix_sum, block_residual, proj_weight, norm_weight, norm_eps=1e-5)
```

测试中使用 `attn_res_fwd_torch_runner.get_attn_res_fwd_fn()` 隔离加载，避免 import 全量 `cann_ops_transformer.ops`（触发 torch 2.1 兼容问题）。

## 5. 运行测试

```bash
cd /data/kdy/cann/ops-transformer/attention/attn_res_fwd/tests/pytest
source env_kdy.sh && source cann_env.sh

# Torch 完整精度（CI + 历史失败 + resident）
bash test_run.sh torch

# 仅生产 shape（T=544, N=12, H=7168）
python -m pytest test_attn_res_fwd_torch.py::test_attn_res_fwd_torch_production -v -s

# 正向 H 扫描（T=1, N=12, seed=0）
bash test_run.sh sweep

# 或按 H 递增顺序、遇错即停（推荐 bisect 入口）：
python forward_sweep_h.py

# ACLNN example L0（不经过 Torch）
bash test_run.sh prod
```

### 5.1 正向 H 扫描（精度 bisect 辅助）

从小 shape 递增 H，定位首个 FAIL 的 hidden size：

```bash
unset DEVICE_ID
export FORCE_AUTO_DEVICE=1
python forward_sweep_h.py
```

最近一次扫描结果（seed=0, T=1, N=12）：

| H | 分支 | MERE | 结果 |
|---|------|------|------|
| 64 | RESIDENT | 2.07 | FAIL（首个精度失败） |
| 128 | RESIDENT | 0.71 | FAIL |
| 256 | RESIDENT | 2.27 | FAIL |
| 512 | RESIDENT | 0.032 | FAIL |
| 1024 | RELOAD | 0.23 | FAIL |
| 2048 | RELOAD | 0.0001 | **PASS** |
| 4096 | RELOAD | — | CRASH（507035 UB 未对齐） |

> H=2048 偶发 PASS 而 H=4096 CRASH，说明 bisect 标量 Softmax 路径在不同 H 下行为不一致，需继续 kernel 侧修复。

### 5.2 精度阈值

- BF16 MERE 阈值：`2^-7 = 0.0078125`
- Golden 实现：`attn_res_fwd_golden.apply_attn_res_cust`
- 比较工具：同目录 `precision_compare.py`

## 6. 常见问题

### JIT 编译头文件冲突

**现象**：`ACL_FLOAT8_*` 等枚举未定义。

**原因**：`OpBuilder.include_paths()` 同时加入 CANN 与 torch_npu 的 `acl/inc`，导致 `acl_base.h` 重复定义。

**修复**：`builder.py` 中 CANN 侧仅加入 `include/aclnn` 与 `include/aclnnop`，acl 基类头使用 torch_npu 自带路径。

### GE 初始化失败（set_device）

**现象**：`torch.npu.set_device()` 报 GE 相关错误。

**排查**：
1. 确认已 `source cann_env.sh`（custom OPP 路径正确）
2. 确认 custom 包已安装且与 CANN 版本匹配
3. 尝试 `export DEVICE_ID=<空闲卡>`

### NPU 直接 randn(bf16) 失败

测试在 CPU 生成随机张量后 `.to(npu)`，避免 custom OPP 环境下 NPU 端 bf16 randn 异常。

### 切换 CANN 版本

```bash
export CANN_PKG=cann-9.1.0-beta.3-20260722
source cann_env.sh
```

## 7. 相关文档

- 算子 Torch API：`torch_extension/cann_ops_transformer/docs/zh/attn_res_fwd.md`
- 开发规范：`torch_extension/cann_ops_transformer/docs/torch_extension_guidelines.md`
- 迭代计划：`attention/attn_res_fwd/docs/PLAN.md`

## 单 shape 精度 + Profiler

见 [ATTENRES_LITE.md](./ATTENRES_LITE.md)（`python attenres_lite.py` / `bash test_run.sh lite`）。
