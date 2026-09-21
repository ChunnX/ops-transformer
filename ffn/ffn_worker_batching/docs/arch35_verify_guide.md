# arch35 三个 verify 命令

三个脚本分别覆盖设备协议、Torch eager 接口和 Torch 完整图执行。它们使用真实设备指针，不依赖 TEST_MAGIC 或 TTK 内部接口；需要 Ascend950 与匹配当前算子的自定义 OPP。以下命令从仓库根目录运行。

| 命令 | 调用路径与作用 | 最新已有验证结果 |
| --- | --- | --- |
| `python ffn/ffn_worker_batching/tests/st/arch35/verify_sync_flag_state.py` | ctypes 直接调用 ACL runtime 和 `aclnnFfnWorkerBatching`，绕过 Torch。通过真实设备分配和跨调用上下文，检查等待、跳过空批、游标回绕、同步收齐、全 mask、槽位复用、增量写 ids 及保留其它 ready 项。用于定位底层接收/清理协议问题。 | 9 个状态场景通过；见 [状态报告](arch35_sync_flag_validation.md) |
| `python ffn/ffn_worker_batching/tests/st/arch35/verify_torch_interface.py` | 通过 Python 公共接口、dispatcher、C++ wrapper 下发 ACLNN。检查六种 dtype 的 eager 输出字节、shape/dtype/stride、索引、计数、上下文状态、非默认流和延迟发布；检查原生及 uint8 Meta/FakeTensor、非法参数和 schema 可变性。 | 98 项通过：64 项 NPU、24 项元数据、9 项拒绝、1 项 schema；见 [接口报告](arch35_torch_validation.md) |
| `python ffn/ffn_worker_batching/tests/st/arch35/verify_torch_graph.py` | 复用 eager 脚本的 Context 与逐字节 golden，用 `torch.compile(backend="npugraph_ex", fullgraph=True, dynamic=False)` 验证完整图及 replay。六种 dtype 的基础用例开启 `mx_output_uint8=True`，检查调度组合、全 mask、singleton、连续消费；另检查同图两次消费、丢弃首次输出和外部延迟发布。 | 46 项通过；见 [图报告](arch35_torch_graph_validation.md) |

状态脚本不依赖 Torch 扩展，主要验证 FP16 协议状态，不能替代六种 dtype 的 Torch 或图测试。eager 脚本不调用 `torch.compile`，不能证明入图成功。图脚本指定 `fullgraph=True`，断图会报错；其中同图双调用、延迟发布用例使用 FP16，六种 dtype 均覆盖各自的连续 replay。

## 环境和构建

本机已验证环境为 Python 3.12、Torch 2.9.0+cpu、torch_npu 2.9.0.post8、CANN 9.2.0、Ascend950DT_9574。下面复用已有 OPP 安装目录；换环境时替换为匹配源码、包含预编译 kernel 的 OPP，不能仅用 JIT 算子包。

```bash
cd /home/chenpengyi/ops-transformer
source /home/chenpengyi/Ascend/cann/set_env.sh
source /tmp/ffn_refactor_verify_opp/vendors/custom_transformer/bin/set_env.bash
export PATH=/home/chenpengyi/.conda/envs/py312-cpy/bin:$PATH
```

状态脚本只需要上述 ACL 库环境及 NumPy，动态加载 `libascendcl.so`、`libnnopbase.so`、`libcust_opapi.so`。运行两个 Torch 脚本前还需安装最新扩展；首次真实调用会 JIT 编译 wrapper，需要编译工具和 ninja：

```bash
cd /home/chenpengyi/ops-transformer/torch_extension
TORCH_EXTENSION_OPS=ffn_worker_batching TORCH_EXTENSION_VENDOR=ffn_mx_graph \
  python setup.py bdist_wheel --dist-dir /tmp/ffn_mx_graph_wheels
python -m pip install --no-deps --upgrade --target /tmp/ffn_mx_graph_install \
  /tmp/ffn_mx_graph_wheels/*.whl
export PYTHONPATH=/tmp/ffn_mx_graph_install:$PYTHONPATH
export FFN_TORCH_PACKAGE=cann_ops_transformer_ffn_mx_graph
export MAX_JOBS=4
cd /home/chenpengyi/ops-transformer
```

`FFN_TORCH_PACKAGE` 仅供验证脚本选择导入包，dispatcher 命名空间仍为 `cann_ops_transformer`。若安装完整默认包，可不设置该变量。

## 执行和结果

建议按设备协议 → eager 接口 → 图模式顺序，在独立进程中串行运行：

```bash
timeout 90 python ffn/ffn_worker_batching/tests/st/arch35/verify_sync_flag_state.py \
  > /tmp/ffn_verify_state.log 2>&1
timeout 180 python ffn/ffn_worker_batching/tests/st/arch35/verify_torch_interface.py \
  > /tmp/ffn_verify_eager.log 2>&1
timeout 600 python ffn/ffn_worker_batching/tests/st/arch35/verify_torch_graph.py \
  --output /tmp/ffn_verify_graph_results.json > /tmp/ffn_verify_graph.log 2>&1
```

前两个脚本将 JSON 场景列表打印到标准输出，没有 `--output` 参数；图脚本打印逐项 PASS 和 JSON 汇总，并可通过 `--output` 保存结果。命令退出码应为 0，再核对完整场景数量；`timeout` 的 124 表示超过外层等待时间，不代表算子返回协议错误。首次 JIT 较慢时可增加外层时间。

这些结果验证功能、输出字节和状态语义，不是性能基准，也不覆盖 TorchAir GE、动态 shape、多卡或任意 MX 下游算子链。图模式的 uint8 是原始编码：MX scale 与 FP4 y 不进行数值转换，FP8 y 保留原生 dtype；默认 eager 仍返回原生类型。
