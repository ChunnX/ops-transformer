# AttnResFwd 单算子测试脚本 `attenres_lite.py`

单 shape 多实现：**精度比对**（vs golden）+ **Ascend Profiler 性能采集**。  
仓内路径：`attention/attn_res_fwd/tests/pytest/attenres_lite.py`。

---

## 1. 前置依赖

| 组件 | 要求 |
|------|------|
| Python | conda `kdy311`（PyTorch 2.10 + torch_npu 2.10） |
| CANN | 默认 `cann-9.2.0-weekly...`（可用 `CANN_PKG` 覆盖） |
| 自定义 OPP | 已安装 `custom_transformer`（含 `AttnResFwd` kernel） |
| 适配插件 | `ops-transformer/torch_extension`（`cann_ops_transformer`） |
| 精度工具 | 同目录 `precision_compare.py` |

---

## 2. 编译安装自定义算子（OPP）

改 `op_kernel` 后必须重装：

```bash
cd /data/kdy/cann/ops-transformer
bash build.sh --pkg --soc=ascend910b --ops=attn_res_fwd -j8
bash build_out/cann-ops-transformer-custom_linux-aarch64.run \
  --install-path=/data/kdy/cann/custompkgbase --quiet --force
```

装完后重新 `source cann_env.sh`。

---

## 3. 适配插件怎么用（Torch / ACLNN 桥）

适配插件把 PyTorch Tensor 接到 `aclnnAttnResFwd`，代码在：

| 文件 | 作用 |
|------|------|
| `torch_extension/cann_ops_transformer/ops/attn_res_fwd.py` | OpBuilder + Python API |
| `torch_extension/cann_ops_transformer/ops/csrc/attn_res_fwd.cpp` | ACLNN C++ wrapper |
| `tests/pytest/attn_res_fwd_torch_runner.py` | 单算子隔离 JIT 加载（`attenres_lite` 走此路径） |

### 3.1 推荐：挂 PYTHONPATH + 首次调用 JIT

```bash
cd /data/kdy/cann/ops-transformer/attention/attn_res_fwd/tests/pytest
source env_kdy.sh
source cann_env.sh   # 已设置 ASCEND_*、custom OPP、PYTHONPATH=.../torch_extension
```

业务代码：

```python
from cann_ops_transformer.ops.attn_res_fwd import attn_res_fwd
# 或仓内隔离加载：
# from attn_res_fwd_torch_runner import get_attn_res_fwd_fn
# attn_res_fwd = get_attn_res_fwd_fn()

out = attn_res_fwd(prefix_sum, block_residual, proj_weight, norm_weight, 1e-5)
```

- **首次调用**会 JIT 编译生成 `.so`（需 `ninja` + 已 `source` CANN/`ASCEND_HOME_PATH`）。
- 改了 `.cpp` / `builder.py` 后清 JIT 缓存或换进程再跑即可重编。

### 3.2 可选：打 wheel 安装

```bash
cd /data/kdy/cann/ops-transformer/torch_extension
source ../attention/attn_res_fwd/tests/pytest/env_kdy.sh
source ../attention/attn_res_fwd/tests/pytest/cann_env.sh
python3 -m pip install -r requirements.txt
python3 -m build --wheel -n
python3 -m pip install dist/*.whl --force-reinstall --no-deps
```

### 3.3 调用约定（BF16）

| 参数 | Shape |
|------|-------|
| `prefix_sum` | `[T, H]` |
| `block_residual` | `[T, N, H]` |
| `proj_weight` | `[1, H]` |
| `norm_weight` | `[H]` |
| `norm_eps` | 标量，默认 `1e-5` |
| 输出 | `[T, H]` |

`attenres_lite` 里实现名 **`attn_res_fwd_cann`** 即走该适配插件。

更多环境细节见 [TORCH_VERIFY.md](./TORCH_VERIFY.md)、插件说明见  
`torch_extension/cann_ops_transformer/docs/zh/attn_res_fwd.md`。

---

## 4. 跑 `attenres_lite`

### 4.1 一键

```bash
cd /data/kdy/cann/ops-transformer/attention/attn_res_fwd/tests/pytest
bash test_run.sh lite
# 等价于：source env + cann 后 python attenres_lite.py
```

### 4.2 常用参数

```bash
source env_kdy.sh && source cann_env.sh

# 默认：T=544 N=12 H=7168；baseline=apply_attn_res_cust；
# 对比 Attn_Res_Fwd_MMVersion + attn_res_fwd_cann；精度+性能
python attenres_lite.py

# 仅精度
python attenres_lite.py --no-perf

# 仅性能
python attenres_lite.py --no-check

# 指定 shape / 实现
python attenres_lite.py --num-tokens 8 --num-blocks 1 \
  --compare apply_attn_res_cust attn_res_fwd_cann

# 列出实现
python attenres_lite.py --list

# 选卡
export DEVICE_ID=7
# 或强制自动选空闲卡
export FORCE_AUTO_DEVICE=1
```

### 4.3 内置实现一览

| 名称 | 含义 |
|------|------|
| `apply_attn_res` / `apply_attn_res_cust` | Torch 参考（默认 golden 为 cust） |
| `Attn_Res_Fwd*` / `*_MMVersion` / `*_FAVersion` | NPU 组合算子拆解参考 |
| **`attn_res_fwd_cann`** | **自定义融合算子（适配插件）** |

精度门控：`data_compare`，阈值 BF16 MERE `2^-7`。

### 4.4 输出

默认写到 `tests/pytest/profiler_output/attenres_lite_<timestamp>/`：

- `single_shape_compare.csv` — 精度 / 耗时汇总
- `<impl>/.../kernel_details.csv` — Profiler kernel 明细

---

## 5. 推荐工作流

1. 改 kernel → 第 2 节重装 OPP  
2. 改适配层（`.cpp` / `.py`）→ 第 3 节 JIT 或 wheel  
3. `bash test_run.sh lite` 或 `python attenres_lite.py` 看精度与相对加速比  
4. 批量用例再用 `bash test_run.sh torch` / `grid_tn`
