# SageAttention2（INT8 QK + FP8 PV）

本目录实现 Atlas 950 上的 SageAttention2 前向路径。输入 Q/K/V 为 FP16 或 BF16，QK 使用细粒度 INT8
量化，Softmax 概率和 V 使用 FP8 E4M3 参与 PV。实现同时提供自动 PyTorch 入口、显式 PyTorch 入口和
`aclnnSageAttention2` 两段式接口。

## 支持范围

- 硬件：Atlas 950；目标软件版本 CANN 9.2.0-beta.1。
- layout：`HND [B,H,S,D]`、`NHD [B,S,H,D]`。
- dtype：FP16、BF16，Q/K/V 必须一致。
- `0 < D <= 128`，内部补齐到 64 或 128。
- MHA/GQA，要求 `Hq % Hkv == 0`。
- Q/K 量化粒度：`per_warp`、`per_thread`。
- PV 累加：`fp32`、`fp32+fp32`、`fp32+fp16`。
- 可选 causal、K/V smoothing 和 FP32 LSE。

不支持 backward、dropout、任意 attention mask、varlen、PagedAttention、KV cache 和 FP16 PV。
causal 要求 `Sq == Sk`。`smooth_v` 仅在 `pv_accum_dtype="fp32"` 时生效，其余模式在 Python 层告警并忽略。

## 代码结构

```text
sageattention2/
├── op_host/                  # 原型、shape/type infer、校验、tiling 和 workspace
├── op_kernel/                # Stats/Quant super-kernel 与 CATLASS Attention 主循环
├── op_api/                   # aclnn 公共头文件
├── torch_ops_extension/      # torch dispatcher 和上游兼容 Python API
├── tests/                    # 代表性 smoke 与全 tiling-key 构建清单
└── docs/                     # 阶段自测报告
```

Kernel 使用一次 mixed-core 启动完成三个设备阶段：

1. AIV 按 channel 计算 K mean、V mean/absmax；
2. AIV 将 Q/K 量化为 INT8，将 V 量化为 FP8，并转换为 CATLASS 连续 BSND 物理排布；
3. AIC/AIV 通过 CATLASS Ascend 950 `BlockMmadTla` 完成 INT8 QK 和 FP8 PV，score 在 UB 内应用细粒度
   scale、causal mask 和在线 Softmax，不写出完整 attention 矩阵。

两个阶段边界使用 `SyncAll<false>()`，Host 强制 batch-mode 并启动全部 AIC。主循环使用双 stage CATLASS
搬运，P 留在 L1，score/O accumulator 留在 UB；Q/K/V 量化结果和 compact scale 只占线性 GM workspace。

## PyTorch API

```python
from sageattention2 import sageattn, sageattn_qk_int8_pv_fp8_asc

out = sageattn_qk_int8_pv_fp8_asc(
    q, k, v,
    tensor_layout="HND",
    is_causal=False,
    qk_quant_gran="per_thread",
    pv_accum_dtype="fp32+fp16",
    smooth_k=True,
    smooth_v=False,
)
```

`sageattn` 在 NPU 上直接转发到同一实现。`return_lse=True` 时返回 `(out, lse)`，其中 LSE shape 为
`[B,Hq,Sq]`、dtype 为 FP32。

自动入口支持以下环境变量作为默认值；函数 kwargs 始终优先：

- `SAGEATTN_NPU_PV_ACCUM_DTYPE`：`fp32`、`fp32+fp32` 或 `fp32+fp16`；
- `SAGEATTN_NPU_QK_QUANT_GRAN`：`per_warp` 或 `per_thread`；
- `SAGEATTN_NPU_SMOOTH_K`、`SAGEATTN_NPU_SMOOTH_V`：`0/1`、`false/true`、`off/on`。

为兼容上游调用，`attn_mask=None` 会被接受并忽略；非空 `attn_mask` 会明确报错，其他未知 kwargs 同样报错。

构建扩展：

```bash
cd experimental/attention/sageattention2/torch_ops_extension
python setup.py build_ext --inplace
```

构建 Ascend 950 全 tiling-key OPP 包：

```bash
source /home/developer/Ascend/cann-9.2.0-beta.1/set_env.sh
cd /path/to/ops-transformer
SAGE_KEYS=$(tr -d '\n' < experimental/attention/sageattention2/tests/tiling_keys.txt)
bash build.sh --make_clean
bash build.sh --experimental --ops=sageattention2 --soc=ascend950 \
  --tiling_key="${SAGE_KEYS}" -j8 --pkg
```

## aclnn API

接口声明见 `op_api/aclnn_sage_attention2.h`：先调用 `aclnnSageAttention2GetWorkspaceSize` 获取 executor
和 workspace，再在同一 stream 上调用 `aclnnSageAttention2`。`smScale <= 0` 表示使用 `1/sqrt(D)`。

## 依赖版本

实现使用 `3rdparty/catlass` 的 Ascend 950 TLA FlashAttention 组件，仓库子模块需固定到
`74e34a822dee604feece97c97d24096b94f6ed87` 或接口兼容的新版本。

## 测试

完整测试矩阵作为独立附件保存，不提交到代码仓；仓内测试范围和附件索引见
[`tests/README.md`](tests/README.md)。本阶段在 Atlas 950PR、
CANN 9.2.0-beta.1 上已完成：

- 145 个唯一 tiling key 全量构建与安装；
- 480/480 正向配置 shape、dtype、NaN/Inf 与 LSE 检查；
- 380/380 功能用例、6/6 负向参数用例；
- 14/14 大形状独立进程检查；
- 20/20 历史 AICore ECC 触发形状压力回归。

阶段报告见 [`docs/stage_delivery_report_20260824.md`](docs/stage_delivery_report_20260824.md)。

## 当前阶段

当前版本已完成源码、接口、全 tiling-key 构建、功能矩阵、边界、有限值/LSE 与运行稳定性收尾，可作为
后续精度和性能工作的稳定基线。本阶段经任务方确认暂不关闭以下两项门禁：

1. CUDA `sageattn_qk_int8_pv_fp8_cuda` 同配置 golden 的 L-A/L-B 精度验收；
2. PyTorch/aclnn 两条路径相对 FIA 的 P-01～P-07 性能门禁。

因此文档只陈述已验证的功能和稳定性结果，不宣称最终精度或性能达标。
