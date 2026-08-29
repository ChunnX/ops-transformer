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

当前在 910B3 上 5/5 用例 PASS（rmsnorm 链路数值精确：query_norm / kv_cache；
其余分层验证详见测试脚本与 `issues/`）。

### 5. C++ aclnn 直调示例

```bash
cd examples && mkdir -p build && cd build && cmake .. && make
export LD_LIBRARY_PATH=~/Ascend/cann-9.1.0/opp/vendors/mla_prolog_v5_custom/op_api/lib:$LD_LIBRARY_PATH
./test_aclnn_mla_prolog_v5
```

## 已知限制（910B3，20 核）

详见 `issues/issue_20260829_mm-split-multihead_01.md`：
1. 多头（N≥2）时 `query`/`query_rope` 的 head≥1 输出与 golden 不一致（vendored kernel
   的 mm3/mm4 多核分块在 20 核下的行为，源仓代码原样迁移未改动，待 kernel 级调试）；
2. rope 输出的存储布局与标准 interleaved 假设不同（pos=0 语义已验证正确）；
3. PA 模式下未使用的 cache slot 也会被写入。

## 与源仓（attention/mla_prolog_v3）的差异

| 项 | 说明 |
|---|---|
| 命名 | `MlaPrologV3`→`MlaPrologV5`，`aclnnMlaPrologV3/V4WeightNz`→`aclnnMlaPrologV5WeightNz`（语义取 V4：含 doRope 参数） |
| 裁剪 | kernel 删除 arch35（`__CCE_AICORE__==310`）分支；def 删除 ascend950 配置；op_graph/fallback 未迁移（aclnn 直调不需要） |
| 依赖 | v1（attention/mla_prolog）的 arch22 kernel 头、tiling/infershape/tiling_check、common 头（data_copy_transpose/ops_err）全部 vendor 进本工程，include 路径本地化，源仓零改动 |
| 新增 | 手写 L0 层（源仓由仓级框架生成 aclnnInner）；torch_extension 独立 CMake 构建（不依赖源仓 cann_ops_transformer 包） |
| 适配 | ① L0 对 torch 路径 ND 张量做 storage shape 归一化（详见 op_api/mla_prolog_v5.cpp 文件头）；② attr 中 float 型参数显式转 float；③ tiling mm3 分块按头边界对齐（910B3 20 核适配，见 op_host/mla_prolog_tiling.cpp 注释） |
