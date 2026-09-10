# arch35 Torch 接口验证报告

日期：2026-09-10。在当前 `ffn_worker_batching` arch35 实现上新增 Python schema/dispatcher/Meta、C++ ACLNN wrapper 和中文接口文档。没有修改 host tiling、kernel 或公共 Torch 桥接代码。最新 uint8 过渡方案回归 **98/98 通过**，结果见 [最新 eager/Meta 结果](arch35_torch_mx_results.json)。初版 86 项结果及当时的源码、wheel 哈希保留为 [历史验证产物](arch35_torch_results.json)，不代表当前产物。

## 实现核对

- API：`ffn_worker_batching`，schema、Python、C++、pybind 名称一致，默认属性与 ACLNN 一致，返回 8 个输出。
- `schedule_context` 标为 `Tensor(a!)`，反映 RECV 对 polling_index 的原地更新；文档明确二级指针缓冲的所有权、生命周期及流依赖。
- 输出申请及 ACLNN workspace/下发均处于输入设备的 DeviceGuard 内；仅允许 Ascend950。
- 复用公共 `ACLNN_CMD`，通过已安装 OPP 的 `aclnnFfnWorkerBatching` 下发；FP4 使用原生 `float4_e2m1fn_x2`，Torch shape 为 `[Y,H/2]`。
- `[1,1]` FP4 单字节输出显式使用 `(2,1)` stride，避免公共转换层将行维误判为打包轴；仍为连续单元素张量，Meta 与实际输出 stride 一致。
- 此报告记录 eager 接口验证；后续已增加 npugraph_ex 图模式验证，详见 [图模式报告](arch35_torch_graph_validation.md)。六种 dtype 均支持固定形状 npugraph_ex 图，MX 需 `mx_output_uint8=True`；仍不支持 GE 后端和反向。

- `mx_output_uint8=True` 时，C++ 仍以原生类型下发 ACLNN，返回前将 MX scale 和 FP4 y 重解释为 uint8；FP8 y 不变。Meta 返回一致的 shape/dtype/stride，默认 False 保持原生 eager 输出。该选项不传给设备算子。

## 环境

- Ascend950DT_9574，设备 0；CANN `/home/chenpengyi/Ascend/cann-9.2.0`。
- Python 3.12；PyTorch 2.9.0+cpu；torch_npu 2.9.0.post8，NPU 可用。
- Python：`/home/chenpengyi/.conda/envs/py312-cpy/bin/python`；补装 JIT 必需的 ninja。
- OPP：`/tmp/ffn_refactor_verify_opp/vendors/custom_transformer`，复用上一轮全量验证且源码匹配的 arch35 包。
- 最终 wheel：`/tmp/ffn_mx_graph_wheels/`；隔离安装目录：`/tmp/ffn_mx_graph_install/`。
- 使用仓库支持的选择性打包模式，Python 包名 `cann_ops_transformer_ffn_mx_graph`，Torch dispatcher 命名空间仍为 `cann_ops_transformer`；无需为发现机制增加手写注册。

## 结果

| 验证项 | 结果 |
| --- | --- |
| 选择性 wheel 构建、隔离安装、自动发现 | 通过 |
| C++ wrapper JIT 编译、链接及实际调用 | 通过 |
| 真实 NPU 计算/状态用例 | 64/64 通过 |
| Meta + FakeTensor（六种 dtype × 原生/uint8 两种表示） | 24/24 通过 |
| 非法 dtype/属性/shape/上下文拒绝 | 9/9 通过 |
| schema 上下文可变性 | 1/1 通过 |
| 合计 | 98/98 通过，无跳过 |

NPU 用例由以下场景组成：六种 token_dtype × H={2,66} × `(need_schedule,sync_flag)={(0,0),(0,1),(1,0),(1,1)}` 共 48 条；六种 dtype 全 mask 共 6 条；六种 dtype 的单 token 最小 shape 共 6 条；FP4 持续上下文/保留其它 ready 项/仅重置 flag 的槽位复用 1 条；非默认 stream 1 条；同步与异步延迟发布各 1 条。

所有 NPU 输入采用真实设备地址，不使用 TEST_MAGIC；token 行额外加入 16B padding，验证源 HS 步长不等于有效载荷长度的情况。校验输出 shape/dtype/device、有效 token/scale 的原始字节、四类索引、完整 group_list、actual_token_num、完整上下文和描述符终态，以及 token 数据未改写。随机字节输入覆盖低精度编码，不进行浮点近似比较。

延迟发布用例在消费流启动后等待 200ms，确认该流尚未完成，再在独立生产流发布 flag：同步用例先有一个 ready、后补齐其余三个；异步用例从全空开始，只发布一个即可返回。200ms 是测试观察窗口，不是算子超时。

初版 86 项覆盖原生 eager 接口；最新 wheel 在此基础上新增 12 项 uint8 Meta/FakeTensor 检查，完整回归 98 项通过。真实 NPU eager 用例仍验证默认原生输出；uint8 返回的实际执行和字节正确性由 46 项图测试覆盖。

本轮为单 NPU Torch 接口验证，未进行多设备 DeviceGuard 切换、多机 URMA 联调、图模式或性能验收。此前独立算子的 33 条 UT、211 条 ST、9 条状态测试仍见 [底层算子验证报告](arch35_sync_flag_validation.md)，不计入本次 98 项。

## 复现

构建与环境设置统一见 [三个 verify 命令指南](arch35_verify_guide.md)。准备完成后，在仓库根目录执行：

```bash
timeout 180 python ffn/ffn_worker_batching/tests/st/arch35/verify_torch_interface.py
```

完整包安装时使用 `import cann_ops_transformer`，无需设置 `FFN_TORCH_PACKAGE`。选择性打包只隔离 Python 安装，未替换公共 ACLNN 执行路径。

最新运行日志：`/tmp/ffn_mx_graph_eager.log`；构建日志：`/tmp/ffn_mx_graph_build.log`；安装日志：`/tmp/ffn_mx_graph_install.log`。
