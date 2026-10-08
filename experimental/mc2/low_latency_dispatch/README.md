# LowLatencyDispatch

从 DeepEP `csrc/kernels/dispatch/low_latency` 迁入的 Ascend950 低时延分发算子。保留 64 AIV 的发送/接收划分、服务器去重协议、双 bank 控制区和 GM tiling ABI；Host 绑定与 Python 封装一并迁入。

## 支持范围

- EP16/32/64/128，每台服务器 8 个连续 rank；每个 rank 固定分配 **8 GiB** SHMEM workspace。
- 每卡 token 容量 `< 256`；具体 H/K/E 组合继续由上游布局和 UB 容量检查判定。
- 输入 BF16/FP16，或 `(FP8 E4M3FN, scales)`；scales 形状 `[B, ceil(H/128)]`，支持 INT32 packed E8M0 或 FP32。
- topk_idx 为连续 INT64 `[B,K]`，topk_weights 为连续 FP32 `[B,K]`，所有输入必须在 buffer 所属 NPU。
- 返回 `(expanded_x, None, None, handle, event)`；输出未加权，路由权重保存在 handle 中供 Combine 使用。
- 不支持缓存 dispatch、额外 stream 调度选项和高吞吐回退。不支持的形状/选项在启动内核前报错。

## 构建与调用

先按 [共享运行时说明](../common/deepep/README.md) 准备固定版本依赖，然后从仓库根目录执行：

```bash
bash experimental/mc2/low_latency_dispatch/build.sh
export PYTHONPATH="$PWD/experimental/mc2/low_latency_dispatch/output/python:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$SHMEM_ROOT/lib:${LD_LIBRARY_PATH:-}"
export DEEPEP_SHMEM_ENDPOINT=tcp://192.0.2.1:16230  # 替换为首节点地址和空闲端口
```

每个 torchrun worker 设置 `torch.npu.set_device(LOCAL_RANK)` 并初始化 HCCL/Gloo 进程组后：

```python
from ops_transformer_mc2 import ElasticBuffer

buffer = ElasticBuffer(group=group, num_bytes=8 << 30, explicitly_destroy=True)
try:
    expanded, _, _, handle, event = buffer.dispatch(
        x, topk_idx=topk_idx, topk_weights=topk_weights,
        num_experts=num_experts, num_max_tokens_per_rank=x.shape[0],
        do_cpu_sync=True,
    )
    event.current_stream_wait()
finally:
    buffer.destroy()
```

`do_cpu_sync=True/None` 且有进程组时，通过该组归约接收数量并返回精确行数；`False` 或无进程组时返回容量行数。无进程组调用需显式传入 `rank/world_size`，调用方保证各 rank 的输入和调用顺序一致。

CPU 测试：`python3 -m pytest experimental/mc2/low_latency_dispatch/tests`。设备示例见 `examples/dispatch.py`。CPU 测试通过不代表设备编译、精度或性能验收通过。

无需 CANN 的 C++ 布局与运行时所有权检查：

```bash
cmake -S experimental/mc2/low_latency_dispatch \
      -B experimental/mc2/low_latency_dispatch/build/host-tests -DMC2_HOST_ONLY_TESTS=ON
cmake --build experimental/mc2/low_latency_dispatch/build/host-tests
ctest --test-dir experimental/mc2/low_latency_dispatch/build/host-tests --output-on-failure
```
