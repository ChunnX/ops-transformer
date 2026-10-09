# CausalConv1dBackward

`CausalConv1d`（因果一维深度卷积，prefill 模式，bias=None、冷启动）的反向算子，用于训练反向传播。给定上游梯度 `grad_y` 及前向输入 `x`、`weight`，计算两路全量梯度 `grad_x` / `grad_weight`。与正向 `mamba/causal_conv1d` 成对，四层命名同构（REG_OP / OpDef / aclnn / torch）。

## 数学定义

前向未保存 pre-activation，反向在算子内部重算。权重序与前向完全一致（**weight[W-1] 配当前 token x[t]，weight[0] 配最老 token x[t-(W-1)]**，vllm causal_conv1d_fn 同约定），正反向共用同一份 `[W, D]` 权重张量，无需翻序：

```
前向： pre[t]   = Σ_{k=0..W-1} weight[k]·x[t-(W-1-k)]·m(t, t-(W-1-k))
       y[t]     = Act(pre[t])
反向： dout[t]  = grad_y[t]·Act'(pre[t])        （Act' 内部重算 pre；none 时 ≡1）
       grad_x[s]      = Σ_{j=0..W-1} weight[W-1-j]·dout[s+j]   （右边界截断）
       grad_weight[k] = Σ_t dout[t]·x[t-(W-1-k)]               （左边界冷启动 pad 0）
```

序列/文档边界由 `query_start_loc` 承担：每段是一个独立的因果卷积反向（段内左补零 = 冷启动，段尾右补零），掩码从数据流中消失。`silu'(z) = σ(z)·(1 + z·(1-σ(z)))`。中间计算与累加全程 fp32，输出 cast 回输入 dtype 一次（grad_weight 禁止半精度累加）。

## 支持范围

| 项 | 值 |
|----|----|
| dtype | float16 / bfloat16 |
| shape | grad_y/x `[B, L, D]`（固定 batch）或 `[cu_seq_len, D]`（变长 + query_start_loc）；weight `[W, D]`，W∈{2, 3, 4}；`(dim × dtypeSize) % 32 == 0`（32B 对齐） |
| 属性 | activation_mode ∈ {silu, none}，默认 silu |
| 语义 | prefill 反向（bias=None、冷启动）；不支持 decode/update；不输出 convStates 梯度 |
| 芯片 | Ascend 950PR&950DT（ascend950） |
| 确定性 | 默认确定性实现（按通道维切核、核内完整归约、FP32 累加） |

## 接口

- aclnn：[aclnnCausalConv1dBackward](./docs/aclnnCausalConv1dBackward.md)（两段式）

  ```cpp
  aclnnCausalConv1dBackwardGetWorkspaceSize(gradY, x, weight, queryStartLocOptional,
                                            activation, gradXOut, gradWeightOut,
                                            &workspaceSize, &executor);
  aclnnCausalConv1dBackward(workspace, workspaceSize, executor, stream);
  ```

- PyTorch：`causal_conv1d_backward(grad_y, x, weight, query_start_loc=None, *, activation="silu") -> (grad_x, grad_weight)`

  支持非连续 weight（可直接传 `[C,1,W]` 参数的 `transpose(0,1)` 视图）；输出恒为连续张量。
- 图模式：`graph_convert_causal_conv1d_backward.py` 提供 GE converter，绑定经 `torch.ops` 分发
  （dynamo 可 trace）；`tests/test_aclgraph.py` 提供 NPUGraph 捕获重放 + torchair npugraph_ex
  两层真机验证（5/5 PASS）。

## 实现

- 切分：沿 D（通道）维切 `baseDim=128` tile，核间按 tile 轮转；每个核独占若干完整通道，grad_weight 在核内完整累加，无需跨核 reduce（确定性、无原子、无 workspace 归约）。
- token 轴按 `query_start_loc` 划段，段内 `T_CHUNK=64` 行分块 + tail 机制（跨 chunk 边界的 dx 行延迟补算），段边界即冷启动左边界。
- 向量核：**RegBase 规范写法**（参照 cannbot-skills `ascendc-regbase-best-practice`）——
  `__simd_vf__` 入口 + `asc_vf_call` 调用 + `AscendC::Reg::*` 全限定 + UpdateMask 引用消耗
  语义；`arch35/causal_conv1d_backward_vf_compute.h` 三段 VF：PassA（x 寄存器滑动窗，每 token
  仅 load 1 行新 x；z 重算+silu' 全寄存器完成；grad_weight 寄存器累加）、PassB（Mul+MulAddDst
  链）、Tail（行搬移）。d0 按 64-lane 补齐行距。UB 占用 ~245KB→~90KB。msprof 实测主 kernel **464.2us（0.1133 us/token），较 membase 版 1054.8us 加速 2.27x**
  （VF 融合链 + 寄存器累加 + fp32 平面直读）；端到端 0.52ms（0.128 us/token），较框架
  旧解析反向 10.11ms 加速 **19.4x**。

## 测试

```bash
# CPU golden 自测（52 用例：W∈{2,3,4} × packed documents × 边界场景 × fp32/fp16/bf16，
# 被测参考实现 ↔ positions 版参考前向 autograd，尺度相对误差阈值 fp32 1e-5 / fp16 5e-3 / bf16 2e-2）
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 tests/golden.py

# NPU ST（bf16/fp16，10 用例：3D / 2D+qsl / 3D+qsl / activation=none / W=2 / 全零梯度 /
# 尾 tile D=4160；需先构建安装算子包：bash build.sh --ops=causal_conv1d_backward --soc=ascend950）
cd tests/st_cpp && bash run.sh --real
```

torch 绑定：schema/Meta 推理、csrc JIT 编译、NPU 端到端调用、框架侧（torchtitan-npu §九
接入 + 自带单测 35/35 + 端到端对拍）均已验证通过（grad_weight 与 golden 逐位一致）。

## 框架接入（torchtitan-npu）

```python
from cann_ops_transformer.ops import causal_conv1d_backward

channels = x.shape[-1]
weight_wc = weight[:, 0, :].transpose(0, 1)      # [C,1,W] → [W,C]，非连续视图直接传
query_start_loc = _query_start_loc(positions)     # 与正向共用同一份缓存
grad_x, grad_weight_wc = causal_conv1d_backward(
    grad_output.reshape(-1, channels), x.reshape(-1, channels), weight_wc,
    query_start_loc, activation="silu")
grad_weight = grad_weight_wc.transpose(0, 1).unsqueeze(1)   # [W,C] → [C,1,W]
```
