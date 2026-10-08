# LowLatencyCombine

从 DeepEP `csrc/kernels/combine/low_latency` 迁入的 Ascend950 低时延归并算子。依赖 [LowLatencyDispatch](../low_latency_dispatch/README.md) 的服务器去重协议和 routing handle；复用同一个 SHMEM 运行时、通信参数和双 bank 状态。

## 构建

按 [共享运行时说明](../common/deepep/README.md) 准备 CANN 9.2.0 和固定提交的 SHMEM SDK，从仓库根目录运行：

```bash
bash experimental/mc2/low_latency_combine/build.sh
export PYTHONPATH="$PWD/experimental/mc2/low_latency_combine/output/python:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$SHMEM_ROOT/lib:${LD_LIBRARY_PATH:-}"
```

此入口同时构建并安装 Dispatch + Combine，安装目录可直接使用，不需先安装另一份 Python 包。单独的 Dispatch 入口仍只编译 Dispatch；调用 Combine 时会给出缺少构建组件的明确提示。无需新增 Git 子模块。

## 调用约束

`buffer.combine(expert_output, handle)` 返回 `(output, None, event)`。输入必须是同一 NPU 上连续 BF16/FP16、形状 `[handle.num_expanded_tokens, handle.hidden]` 的专家输出；handle 必须来自同一 buffer 的 payload dispatch。

Combine 在设备内应用 handle 保存的源路由权重，因此专家输出不能预先乘权重；不接受另传 topk_weights、bias 或额外的调度/stream 选项。handle 可以重复用于 Combine，调用仍需按所有 rank 相同顺序串行执行。完整生命周期与进程组要求同 Dispatch。

## 验证

```bash
python3 -m pytest experimental/mc2/low_latency_dispatch/tests experimental/mc2/low_latency_combine/tests
cmake -S experimental/mc2/low_latency_combine \
      -B experimental/mc2/low_latency_combine/build/host-tests -DMC2_HOST_ONLY_TESTS=ON
cmake --build experimental/mc2/low_latency_combine/build/host-tests
ctest --test-dir experimental/mc2/low_latency_combine/build/host-tests --output-on-failure
```

设备回环示例 `examples/roundtrip.py` 将未修改的 Dispatch 输出作为 identity expert 输出，与原输入比对，并重复 Combine 三次。使用 torchrun 在每台机器启动 8 个 worker；总 EP 为 16/32/64/128。设置所有 rank 相同且不与 rendezvous 冲突的 `DEEPEP_SHMEM_ENDPOINT`。

```bash
torchrun --nnodes=2 --nproc-per-node=8 --node-rank=0 \
    --master-addr=<首节点IP> --master-port=29500 \
    experimental/mc2/low_latency_combine/examples/roundtrip.py
# 另一节点使用 --node-rank=1，其余参数保持一致。
```

CPU 布局/接口测试不能替代 Ascend950 的设备编译、多机精度与性能验收。
