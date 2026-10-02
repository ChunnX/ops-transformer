# SageAttention2 PyTorch 接入说明

本文件说明 PyTorch 适配层的安装路径、dispatcher 注册面、图模式范围与使用约束。

## 一、安装与加载

- 方式 A（源码直用）：`PYTHONPATH=<本目录>/python` 后 `import sageattention2`，零编译依赖（CPU 分发
  路径可独立运行宿主回归）。
- 方式 B（扩展桥）：`pip install .`；首次 NPU 调用时经 `torch.utils.cpp_extension` JIT 编译
  `[../python/sageattention2/csrc/attention.cpp](../python/sageattention2/csrc/attention.cpp)`（需 C++17 编译器），后续走 PyTorch 扩展缓存。
- 方式 C（OPP + ACLNN）：`bash ../build_opp.sh && bash ../install_opp.sh` 安装自定义算子包，
  `ASCEND_CUSTOM_OPP_PATH` 指向包根且须在导入 torch_npu 前设置；csrc 桥经 dlopen 解析
  `aclnnSa2Attention2` / `aclnnSageAttention2` 两段式符号下发。
- 依赖：torch（≥ 2.x，含 torch.library 稳定接口）、torch_npu（NPU 路径必需）。

## 二、dispatcher 注册面

公开入口 `sageattn` / `sageattn_qk_int8_pv_fp8_asc` / `replace_sdpa` 均经
`torch.ops.sageattention2.forward`（torch.library 注册）下发，schema：

```text
sageattention2::forward(Tensor q, Tensor k, Tensor v, *,
    str tensor_layout="HND", bool is_causal=False, str qk_quant_gran="per_thread",
    float? sm_scale=None, str pv_accum_dtype="fp32+fp16", bool smooth_k=True,
    bool smooth_v=False, bool return_lse=False) -> (Tensor, Tensor?)
```

- 设备分发：CPU 实现（宿主参考）与 PrivateUse1 实现（NPU）；AutogradPrivateUse1 同注册（推理前向）。
- Meta / FakeTensor：`register_fake` 提供校验与 shape/dtype 推导（含 LSE `[B, Hq, Sq]`），
  参数校验在 fake 域即拦截。
- 契约断言：schema kwonly 参数序与默认值逐项对齐任务书 §2.3
  （[../tests/test_pytorch_integration.py](../tests/test_pytorch_integration.py)::test_schema_contract 钉为契约用例）。
- 工具辅助：提供 `create_qkv` 与 `create_attention_inputs` 工厂函数，支持以字典参数直接构造 Q/K/V 张量与算子传参，消除多处重复的 `torch.randn` 样板代码。

## 三、图模式范围（已验证 / 未验证分列）

- 已验证（宿主 CPU，2026-10-02）：`torch.compile(backend="eager", fullgraph=True)` 全图捕获，
  算子级（自定义 op 单节点）与入口级（`sageattn` python 包装层）均无 graph break，输出与 eager 逐位一致。
- 未验证：inductor / 其它后端编译、NPU 侧图模式（torch_npu 图捕获）——真机阶段补充验证口径（挂 M3）。

## 四、自动分发与环境变量

- `sageattn` 在 NPU 上固定分发到 `sageattn_qk_int8_pv_fp8_asc`（FP8 PV 路径）。
- 覆盖项：`SAGEATTN_PV_ACCUM_DTYPE`（fp32 / fp32+fp32 / fp32+fp16）、`SAGEATTN_QK_QUANT_GRAN`
  （per_warp / per_thread）；显式 kwargs 优先于环境变量，环境变量优先于默认值。
- `replace_sdpa(**overrides)`：上下文内替换 `torch.nn.functional.scaled_dot_product_attention`
  （拒绝 attn_mask / dropout），退出恢复原实现。

## 五、使用约束

- 输入 FP16 / BF16 同 dtype 同 NPU device，末维连续；GQA 要求 `Hq % Hkv == 0`；causal 仅 `Sq == Sk`。
- 非法参数（layout / gran / accum / head_dim 越界 / sm_scale 非有限 / attn_mask 非 None /
  未声明 kwargs）一律显式抛错，不静默回退。
- 输出与 q 同 dtype 同布局；`return_lse=True` 时返回 `(out, lse)`，lse 为 FP32 `[B, Hq, Sq]`。

## 六、验证状态（宿主 / 真机分列）

- 已验证（宿主 CPU）：本说明第二、三、四、五节全部断言（[../tests/test_pytorch_integration.py](../tests/test_pytorch_integration.py)，
  2026-10-02，torch 2.14.1 CPU）。
- 待验证（真机 NPU）：PrivateUse1 路径下同断言、多流并发与重复调用一致性真机域复验（挂 M3）。
