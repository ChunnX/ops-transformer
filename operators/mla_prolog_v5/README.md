# MlaPrologV5（独立算子工程）

> 从 [ops-transformer](https://gitcode.com/cann/ops-transformer.git) 仓 `attention/mla_prolog_v3`
> 快速迁移的独立自定义算子工程，仅保留 **A2（ascend910b / 910B 系列）** 代码，
> 并打通 pybind（Python 接口）调用流程。

## 产品支持情况

| 产品 | 是否支持 |
|:----------------------------|:-----------:|
| Atlas A2 训练系列产品/Atlas A2 推理系列产品（910B，含 910B3） | √ |
| 其他（950/arch35 等） | ×（已按迁移要求裁剪） |

## 功能说明

推理场景 Multi-Head Latent Attention（MLA）前处理算子，四路计算：

- $c^Q = \alpha_q \cdot \mathrm{RmsNorm}(x \cdot W^{DQ})$，可选输出 `query_norm`
- $q^N = (c^Q \cdot W^{UQ})\ \mathrm{per\text{-}head} \cdot W^{UK}$，$q^R = \mathrm{ROPE}(c^Q \cdot W^{QR})$
- $c^{KV} = \alpha_{kv} \cdot \mathrm{RmsNorm}(x \cdot W^{DKV}) \rightarrow$ 原地写入 kvCache
- $k^R = \mathrm{ROPE}(x \cdot W^{KR}) \rightarrow$ 原地写入 krCache

详细公式、参数、量化模式与约束说明与源仓 v3 一致（见源仓
`attention/mla_prolog_v3/README.md`，本工程不再重复维护全量文档）。

## 工程结构

```
mla_prolog_v5/
├── build.sh                # 构建入口：bash build.sh --soc=ascend910b [--install] [--torch]
├── CMakeLists.txt          # npu_op_package 独立构建（无需源仓构建框架）
├── op_kernel/              # kernel 入口 + vendored arch22 头 + tiling data 定义
├── op_host/                # def / infershape / tiling（vendored 自 mla_prolog_v3 与 mla_prolog）
├── op_api/                 # aclnnMlaPrologV5WeightNz 外层 + 手写 L0（独立工程无源仓 aclnnInner 生成框架）
├── torch_extension/        # pybind 扩展（csrc + CMake）+ Python 接口 mla_prolog_v5.py
├── examples/               # C++ aclnn 直调示例
├── tests/pybind/           # Python 接口测试（NPU 实跑 + CPU golden 比对）
└── docs/                   # REQUIREMENTS / DESIGN / LOG / aclnnAPI 文档
```

## 快速开始

### 1. 编译算子包

```bash
cd operators/mla_prolog_v5
bash build.sh --soc=ascend910b --install      # 编译 + 安装 custom_opp 包
source ~/Ascend/opp/vendors/mla_prolog_v5_custom/bin/set_env.bash   # 或按安装提示设置环境
```

### 2. 编译 pybind 扩展

```bash
bash torch_extension/build.sh                 # 产出 torch_extension/build/libmla_prolog_v5_ext.so
```

### 3. Python 接口调用

```python
import torch, torch_npu
torch.npu.config.allow_internal_format = True  # 允许 FRACTAL_NZ 等 internal format
import sys; sys.path.insert(0, "operators/mla_prolog_v5/torch_extension")
from mla_prolog_v5 import mla_prolog

# 权重需转 FRACTAL_NZ：torch_npu.npu_format_cast(t.to("npu").to(torch.bfloat16), 29)
query, query_rope, _, query_norm, _ = mla_prolog(
    token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr,
    rmsnorm_gamma_cq, rmsnorm_gamma_ckv, kv_cache, kr_cache,
    rope_sin=rope_sin, rope_cos=rope_cos,          # 成对传入开启 RoPE，均 None 关闭
    cache_index=cache_index, cache_mode="PA_BSND",  # PA_BSND / PA_NZ / PA_BLK_* / BSND / TND
    query_norm_flag=True)
```

### 4. 运行测试

```bash
python3 tests/pybind/test.py
```

当前在 910B3 上 5/5 用例 PASS（全部输出 query / query_rope / query_norm / kv_cache /
kr_cache 与 CPU golden 精确比对，覆盖 TND、PA_BSND、3 维输入、do_rope 开关、scale 校正）。

### 5. C++ aclnn 直调示例

```bash
cd examples && mkdir -p build && cd build && cmake .. && make
export LD_LIBRARY_PATH=~/Ascend/cann-9.1.0/opp/vendors/mla_prolog_v5_custom/op_api/lib:$LD_LIBRARY_PATH
./test_aclnn_mla_prolog_v5
```

## 关键输入布局契约（测试时务必注意）

以下契约由 kernel 实现决定并经 NPU 实测标定（详见 `tests/pybind/golden.py` docstring 与
`issues/issue_20260829_mm-split-multihead_01.md` 关闭结论）：

1. **`weight_uq_qr` 列布局为按头交错**：`[Qc_h(D) | Qr_h(Dr)] × N`（**不是** `[Wuq | Wqr]`
   分离拼接）。构造方式：`torch.cat([wuq(Hcq,N,D), wqr(Hcq,N,Dr)], dim=2).view(Hcq, N*(D+Dr))`。
2. `weight_dkv_kr` 列布局为**分离**：`[W_dkv(Hckv) | W_kr(Dr)]`。
3. `rope_sin`/`rope_cos` 形状 (T, Dr)（3 维输入为 (B,S,Dr)），**前后半重复**布局
   （标准 RoPE 频率表形态：`sin = cat([s_half, s_half])`）。
4. RoPE 输出（`query_rope`/`kr_cache`）为 **[前半分量 | 后半分量] 布局**：
   `out[j] = x[2j]*cos[j] - x[2j+1]*sin[j]`，`out[Dr/2+j] = x[2j]*sin[j] + x[2j+1]*cos[j]`。
5. PA 模式下未使用的 cache slot 保持原值不写入（校验初始标记时须用 bf16 可精确表示的值）。

## 与源仓（attention/mla_prolog_v3）的差异

| 项 | 说明 |
|---|---|
| 命名 | `MlaPrologV3`→`MlaPrologV5`，`aclnnMlaPrologV3/V4WeightNz`→`aclnnMlaPrologV5WeightNz`（语义取 V4：含 doRope 参数） |
| 裁剪 | kernel 删除 arch35（`__CCE_AICORE__==310`）分支；def 删除 ascend950 配置；op_graph/fallback 未迁移（aclnn 直调不需要） |
| 依赖 | v1（attention/mla_prolog）的 arch22 kernel 头、tiling/infershape/tiling_check、common 头（data_copy_transpose/ops_err）全部 vendor 进本工程，include 路径本地化，源仓零改动 |
| 新增 | 手写 L0 层（源仓由仓级框架生成 aclnnInner）；torch_extension 独立 CMake 构建（不依赖源仓 cann_ops_transformer 包） |
| 适配 | ① L0 对 torch 路径 ND 张量做 storage shape 归一化（详见 op_api/mla_prolog_v5.cpp 文件头）；② attr 中 float 型参数显式转 float（修复 rmsnorm 输出全 0 的关键适配） |
