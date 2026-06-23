# 后续性能优化入口

## 优先优化对象选择

对用户当前 Ascend 910B B3 环境，建议优先级：

1. `agent/ascendc-operators.md` 中 `910B CI` 为“是”的主线算子。
2. 有 `examples/` 且能通过 `build.sh --run_example` 生成二进制的算子。
3. 有 `opkernel` UT、pytest 或 ST 资产的算子。
4. 业务真实使用且可提供 shape/dtype/format 的算子。

未在 910B CI 列表覆盖的源码目录不是不能优化，但要先做支持性验证。

## kernel 侧入口

重点路径：

```text
<domain>/<op_name>/op_kernel/
<domain>/<op_name>/op_kernel/arch22/
```

优先关注：

- `__aicore__` kernel 入口。
- `AscendC::TQue`、`TBuf`、`GlobalTensor`、`LocalTensor` 的使用。
- GM 到 UB/L1/L0 的搬运模式。
- double buffer、pipeline、event 同步。
- `DataCopy`、`LoadData`、`Mmad`、`PipeBarrier` 等高频操作。
- vector 指令、cast、quant/dequant、softmax、activation、reduce 的融合程度。
- `blockIdx` 到数据分片的映射。
- tail、mask、对齐和边界处理。
- workspace 使用和中间 tensor 生命周期。

对 `ascend910b`，先确认是否存在 `arch22` 特化。没有 `arch22` 目录时，再看通用实现和条件编译。

## tiling 侧入口

重点路径：

```text
<domain>/<op_name>/op_host/*tiling*
<domain>/<op_name>/op_kernel/*tiling_data*
<domain>/<op_name>/op_kernel/*tiling_key*
```

优先关注：

- shape/dtype/format 到 tiling key 的选择。
- block dim、tile size、base block、expert/group 分桶策略。
- workspace 估算。
- 多 kernel 分支的选择条件。
- `ascend910b` 与其他 SoC 的差异分支。
- small shape、long sequence、large batch、非对齐 shape 的特殊路径。

性能优化常见风险：

- host tiling 和 kernel tiling data 结构不一致。
- 新增 tiling key 后 op info 或 binary compile 未覆盖。
- 单个 shape 优化导致其他 shape 回退或越界。

## 编译参数入口

`build.sh` 中与性能调试相关的参数：

- `--soc=ascend910b`
- `--ops=<op_name>`
- `--bisheng_flags=<flags>`
- `--kernel_template_input=<path>`
- `--oom`
- `--op_debug_config`
- `--ops-compile-options`
- `--jit`

后续做性能对比时，必须记录每次构建参数。编译参数变化可能导致 binary 差异，不能和源码改动混在一起判断。

## profiling 入口

主线方式：

```bash
bash build.sh --pkg --soc=ascend910b --ops=<op_name> --vendor_name=custom -j16
bash build.sh --run_example <op_name> eager cust --vendor_name=custom --soc=ascend910b
msprof op ./test_aclnn_<op_name>
```

分析优先级：

- 首先看目标 kernel 的 `Task Duration(us)`。
- 再看 `Block Dim` 是否符合预期。
- 再看 ArithmeticUtilization 中计算、搬运、等待等指标。
- 如果多 kernel 组成一个算子，要分清瓶颈 kernel，而不是只看总耗时。
- MC2 类算子要区分计算 kernel 和通信时间。

## 各算子域优化关注点

### attention

- softmax 数值稳定和流水。
- Q/K/V layout、seq length、head dim、KV cache。
- prompt/incremental/fused infer 场景的分支选择。
- sparse/block sparse 的索引和访存局部性。
- quant attention 的 cast、scale、dequant 融合。

### ffn

- matmul 与 activation 融合。
- batch worker 调度。
- quant 或 normalize 与 GEMM 的衔接。

### gmm

- group/expert 分布不均。
- routing 后的 token layout。
- small group 的 kernel launch 和空转。
- swiglu/quant/dequant 融合。

### mc2

- 计算和通信的重叠。
- HCCL 环境、多卡拓扑和 rank 配置。
- all-reduce/all-gather/reduce-scatter/all-to-all 与 matmul 的切分边界。
- workspace 和中间 buffer 复用。

### moe

- token permute/unpermute 的访存连续性。
- topk 和 routing 的分桶策略。
- expert tokens 分布偏斜。
- small expert batch 性能。

### posembedding

- rope/rotary 的向量化。
- sin/cos cache 访问。
- kvcache 写入连续性。
- rms norm 与 rope/cache 的融合。

## 推荐优化闭环

1. 从真实业务或 profiling 中拿到目标 shape/dtype/format。
2. 用 `--ops=<op_name>` 建立最小 build。
3. 跑对应 example，确认能触发目标 kernel。
4. 用 `msprof op` 建立 baseline。
5. 阅读 `op_host` tiling，确认目标 shape 的 tiling key。
6. 阅读 `op_kernel` 目标分支，确认瓶颈位置。
7. 做小步修改，每次只改变一个主要因素。
8. 跑 UT/example/pytest，确认精度和功能。
9. 重复 profiling，记录同一输入下的多次结果。
10. 最后扩大 shape 和回归范围。
