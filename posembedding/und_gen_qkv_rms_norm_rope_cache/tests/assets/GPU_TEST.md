# GPU 参考测试

`gpu_reference.py` 使用 PyTorch 在输入所在设备上独立计算 Q/K RMSNorm、MRoPE 和 KV Cache 写入，不调用 CPU golden 的计算函数。`golden.py` 的三条 TestSpec 均通过 `third_party["torch"]` 接入此实现；普通 CPU golden 比对仍可单独使用。

## GPU 单机测试

在安装 CUDA 版 PyTorch 的 GPU 机器上，将 `golden.py` 和 `gpu_reference.py` 放在同一目录后执行：

```bash
python gpu_reference.py --device cuda:0
```

共 8 个确定性用例，覆盖两种 head 配置、仅 und、混合 und/gen、重复 cat 索引、仅 gen、单轴/三轴 positions，以及非连续 slot 写入。与 CPU golden 比较 Q 和已写入的 K/V，另行检查未写入 cache 保持不变、V 精确一致，以及输出的 shape/dtype/device。

无 GPU 时可用 `--device cpu` 验证计算逻辑，但这不代表 CUDA 测试通过。默认 CUDA 不可用时明确报错，不自动回退 CPU。

## TTK XPU 三方交叉校验

按 ops-test-kit 的 `ttk/remote/server/README.md` 将 server 目录部署为 GPU 机器上的 `xpu_server` 包，在其父目录启动：

```bash
python -m xpu_server.xpu_server --port 9090 --bind 0.0.0.0
```

在 NPU worker 创建 `ttk.conf.yaml`，将 `GPU_HOST` 替换为 GPU 机器地址：

```yaml
remote:
  endpoints:
    - host: GPU_HOST
      port: 9090
frameworks:
  torch:
    xpu:
      torch_lib: cuda
      profiler: builtin
```

在已配置 CANN 且安装了目标算子的 ops-test-kit 环境中，设置 `ASSETS` 为本目录的绝对路径，执行现有 kernel CSV：

```bash
python3 -m ttk kernel -b=release \
  -i "$ASSETS/../st/arch35/ttk_kernel_und_gen_qkv_rms_norm_rope_cache_st.csv" \
  --plugin "$ASSETS/golden.py" --config ttk.conf.yaml \
  --compare cross_check --provider torch
```

TTK 同步 `golden.py`，远端首次调用时按缺失依赖同步同目录的 `gpu_reference.py`。输入沿用现有 CSV 和 customize_inputs，输出顺序为 `(q, k_cache, v_cache)`。

可将 `--compare cross_check` 换成 `--xpu-perf` 采集远端性能并保留默认精度判据。该实现是 eager PyTorch 多算子参考，包含中间张量分配，不能作为融合 CUDA kernel 的性能基线。RoPE 使用普通 FP32 乘加，与 CPU golden 模拟 FMA 的舍入可能不同；三方误差和 GPU 实际结果需要通过实跑判断。
