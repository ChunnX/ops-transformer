# MegaMoE

从 DeepEP `experiments/megamoe`（`7c4b276e91fc6df0c1bc5141ae45272e1af4c2d0`）迁入的 Ascend950 实验性通信计算融合算子。保留 Dispatch、GMM1、SwiGLU/量化、GMM2、Combine 融合实现，包含 W4A8 Prefill/Decode、共享专家和可选设备 timer。Python 入口为 `ops_transformer_mc2.megamoe`。

## 目录

- `csrc/`：保留上游相对布局的 Host/Torch 绑定、AscendC 内核、CATLASS GMM 和 epilogue。
- `python/ops_transformer_mc2/megamoe/`：公开 API、权重转换、模型配置、CPU Golden、benchmark 和 timer 解析。
- `cmake/`、`build.sh`：独立直调构建；共享 SHMEM 运行时见 [common/deepep](../common/deepep/README.md)。
- `tests/`：迁入的 CPU 精度参考、API、StoreGroup、timer、性能统计与边界检查。

## 依赖和编译

要求 Linux、Ascend950、CANN 9.2.0、Python >=3.11、CMake >=3.20、pybind11 和相互匹配的 Torch/torch_npu。先按 [共享说明](../common/deepep/README.md) 准备 SHMEM SDK，再从仓库根目录准备 CATLASS：

```bash
cmake -S experimental/mc2/common/deepep/dependencies \
      -B experimental/mc2/common/deepep/build/dependencies -DMC2_FETCH_CATLASS=ON
# 离线源码可追加 -DMC2_SHMEM_SOURCE=... -DMC2_CATLASS_SOURCE=...
export CATLASS_ROOT="$PWD/third_party/mc2_deepep/catlass"
bash experimental/mc2/mega_moe/build.sh
# 如需设备打点：bash experimental/mc2/mega_moe/build.sh -DDEEPEP_MEGAMOE_TIMER=ON
export PYTHONPATH="$PWD/experimental/mc2/mega_moe/output/python:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$SHMEM_ROOT/lib:${LD_LIBRARY_PATH:-}"
python3 -c 'from ops_transformer_mc2.megamoe import SymmBuffer'
```

CATLASS 固定为 `2b4bd7cbf7e67ebab23d7d5a99d472c83f90d9a1`，与仓库其他算子使用的版本不同，独立放在 `third_party/mc2_deepep/catlass`，不会修改公共 CATLASS 缓存。原 Git 子模块转为固定提交 FetchContent；不拉入 DeepEP 本身。直接指定 `CATLASS_ROOT` 时仍校验提交，避免误用默认 v2.0.0。Tensor API 从 CANN 的 common 包查找；布局不同时传 `-DMC2_MEGAMOE_COMMON_INCLUDE_DIR=/path/to/common`，该目录须含 `tensor_api/tensor.h` 或 `tensor_api/include/tensor_api/tensor.h`。

此入口安装共享运行时、Dispatch/Combine 与 MegaMoE 到同一个 `output/python`，无需 DeepEP wheel，也不自动安装系统 Python 包。原仓库的 SSH 部署及 wheel 构建脚本由此构建入口和显式 torchrun 命令替代。

## 调用与设备验证

输入、输出、权重布局、约束及接口签名见 [API.md](API.md)。`SymmBuffer` 支持 EP2～128；W4A8 Decode/Prefill 根据实际 B 选择，所有 rank 同序、同 B 调用。通过 `StoreGroup` 使用 torchrun 已有 TCPStore 进行主机协调，设备通信用 SHMEM。每个 rank 先选择自己的 NPU，所有 rank 最后显式销毁 buffer 和 StoreGroup。

```bash
# 所有节点使用同一值，端口必须与 rendezvous 不同。
export DEEPEP_SHMEM_ENDPOINT=tcp://192.0.2.1:16230
torchrun --nnodes=2 --nproc-per-node=8 --node-rank=0 \
    --master-addr=192.0.2.1 --master-port=29500 \
    -m ops_transformer_mc2.megamoe.example --input fp8
# 其他节点修改 node-rank；示例只验证有限输出，精度验收使用下方 Golden/benchmark。
```

设备精度和性能测试保留独立模块入口，可以先 `--help` 查看完整参数：

```bash
# 在各节点上为本机 rank 准备 CPU Golden；示例为首节点 rank 0-7。
python3 -m ops_transformer_mc2.megamoe.golden --config smoke_ep16 \
    --ep-size 16 --ranks 0-7 --input fp8 --golden-dir ./benchmark-golden
torchrun --nnodes=2 --nproc-per-node=8 --node-rank=0 \
    --master-addr=192.0.2.1 --master-port=29500 \
    -m ops_transformer_mc2.megamoe.benchmark --config smoke_ep16 --mode accuracy \
    --input fp8 --golden-dir ./benchmark-golden --output-dir ./benchmark-results
```

复制到目标环境后，应覆盖 B=255/256/257/512、共享专家 S=0/1/2、重复调用和 `256 → 257 → 256` 切换，并检查 EP128 Decode 的 N192 路径。CPU 测试与构建成功均不能代替多机精度和性能验收。

## CPU 验证

```bash
python3 -m pip install pytest numpy
python3 -m pytest experimental/mc2/mega_moe/tests
cmake -S experimental/mc2/mega_moe -B experimental/mc2/mega_moe/build/host-tests \
      -DMC2_HOST_ONLY_TESTS=ON
cmake --build experimental/mc2/mega_moe/build/host-tests
ctest --test-dir experimental/mc2/mega_moe/build/host-tests --output-on-failure
```

其中真实 TCPStore/torchrun 测试需要 CPU PyTorch，未安装时会显式跳过。其余 CPU 测试不需要 NPU。
