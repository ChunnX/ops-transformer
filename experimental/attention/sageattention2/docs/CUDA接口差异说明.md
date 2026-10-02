# SageAttention2 与 CUDA 默认接口（sageattn_qk_int8_pv_fp8_cuda）差异说明

本文件对照开源 SageAttention（`sageattention/core.py`，PyPI `sageattention==2.2.0`）的 FP8 PV 默认路径，
逐项说明本算子（`sageattn_qk_int8_pv_fp8_asc` / `sageattn` / `aclnnSa2Attention2`）的接口映射、量化语义、
数值口径与行为差异。参数命名、类型、默认值与返回值以任务书要求逐项对齐开源接口，函数名 `_cuda` → `_asc`。

## 一、接口映射

- `sageattn(q, k, v, tensor_layout="HND", is_causal=False, sm_scale=None, return_lse=False, **kwargs)`：
  自动分发入口，NPU 上固定分发到 FP8 PV 路径；`pv_accum_dtype` / `qk_quant_gran` 可经 kwargs 或环境变量
  `SAGEATTN_PV_ACCUM_DTYPE` / `SAGEATTN_QK_QUANT_GRAN` 覆盖（[../README.md](../README.md) 第二节）。
- `sageattn_qk_int8_pv_fp8_asc(...)`：与开源 `sageattn_qk_int8_pv_fp8_cuda` 参数列表、默认值、返回值逐项对齐
  （`qk_quant_gran="per_thread"`、`pv_accum_dtype="fp32+fp16"`、`smooth_k=True`、`smooth_v=False`、
  `return_lse=False`）。
- `aclnnSa2Attention2`：两段式 aclnn 入口，语义与 PyTorch 路径一致（[../op_api/aclnn_sage_attention2.h](../op_api/aclnn_sage_attention2.h)）。
- 不导出开源 `*_sm90` / `*_cuda` 同名符号；不提供 FP16 PV、varlen、triton 路径。

## 二、量化语义

- QK 量化（INT8）：`qk_quant_gran` 支持 `per_thread`（head_dim 维每 16 列一个 scale）与 `per_warp`
  （每 token 一个 scale）两种粒度，均为真实实现并逐配置与开源数值语义对齐；两种粒度结果可区分并有
  独立测试断言。**非法取值严格报错（ValueError），不做静默回退。**
- V 量化（FP8 E4M3）：每 32 token × channel 一组（MX 分组），组 scale 为 E8M0（2 的幂，向上取整，
  全零组防护）；`fp32+fp16` 档平衡因子等超参对齐开源两级累加口径。
- P 码：概率域按平衡因子封顶（P ≤ κ），无逐元素乘 448 与 FP16 舍入饱和路径。

## 三、两级累加（pv_accum_dtype）

- 支持 `fp32`（全程 FP32）、`fp32+fp32`（低精度 MMA 累加定期归约 FP32）、`fp32+fp16`（定期归约
  FP16 buffer，**默认验收档**）三档，逐档检验、互不混用。
- **非法取值严格报错（ValueError），不静默回退。**

## 四、smooth_k / smooth_v 与 LSE

- `smooth_k=True`（默认）：K 沿序列维减逐通道均值；GQA 下均值按 `repeat_interleave` 组广播，
  LSE 输出含平滑校正补偿。
- `smooth_v`：仅 `pv_accum_dtype="fp32"` 档生效（序列维减均值 + 输出侧补偿）；`fp32+fp32` /
  `fp32+fp16` 档 warning 后忽略（对齐开源实现）。
- LSE：语义对齐开源 `core.py` 返回路径（`lse / 1.44269504 + lse_correction * sm_scale`，即
  `QK^T · scaling` 各行 logsumexp）；本实现在自然域闭合该等价式（`ln(denom) + m' + s·(q·μ)`），
  shape `[B, Hq, Sq]`、dtype FP32。

## 五、shape / dtype 与约束

- 输入 FP16 / BF16 同 dtype 同 device，末维连续（`stride(-1)==1`）；`Hq % Hkv == 0`；
  `is_causal` 仅 `Sq == Sk`；非因果支持 `Sq != Sk`。
- head_dim：原始 D ≤ 128；D < 64 补至 64、(64, 128) 补至 128、D > 128 或 D ≤ 0 报错；输出裁回 `head_dim_og`。
- 不支持 attention mask（非 None 报错）、dropout、varlen、autograd；仅前向。
- 平台：仅注册 Ascend 950（aclnn 侧含平台校验）。

## 六、数值行为差异与保证

- 确定性：跨核归约采用单写者确定性分块树（无 atomic 加法），同输入重复调用输出与 LSE 逐位一致
  （[../tests/test_pytorch_integration.py](../tests/test_pytorch_integration.py) 钉为契约用例）。
- 概率域封顶：P ≤ κ 构造性杜绝 E4M3 饱和穿透；极值分数域（±100、极差 > 150）、深度负分数、
  FP16 折叠上界（≤ 65504）有专项数值契约用例（[../tests/sa2_numeric_conformance.cpp](../tests/sa2_numeric_conformance.cpp)）。
- 宿主可复现回归：CPU 参考实现（[../python/sageattention2/reference.py](../python/sageattention2/reference.py)）与内核宿主差分桥可脱离 NPU
  运行全量数值断言。

## 七、验证状态（宿主 / 真机分列）

- 已验证（宿主 CPU，2026-10-02）：接口契约、参数校验、pad/裁剪、LSE 语义（含 GQA 广播与因果）、
  gran × accum 全组合、重复调用逐位一致、torch.compile(eager, fullgraph) 与 Meta/FakeTensor
  （[../tests/test_pytorch_integration.py](../tests/test_pytorch_integration.py)）、C++ 数值契约 985,726 checks 0 失败。
- 待验证（真机 NPU，挂 M3/M4/M5）：NPU 执行冒烟、AscendOpTest 精度、对 GPU 开源 L-A golden 对照、
  P-01~P-07 双路径性能。
