# arch35 Torch 图模式设计与验证

日期：2026-09-10。六种 token_dtype 均支持固定形状 `torch.compile(backend="npugraph_ex", fullgraph=True, dynamic=False)` 推理图；MX（3/4/5）需开启 `mx_output_uint8=True`。Ascend950 实测图模式 **46/46**、eager/Meta 回归 **98/98** 通过。当前不支持 TorchAir GE 后端和反向。

## 输出表示与执行路径

`mx_output_uint8=False` 默认保持原生 eager API。开启时，MX `dynamic_scale` 返回 uint8，FP4 `y` 返回 `[Y,H/2]` 打包 uint8，FP8 `y` 仍为原生 FP8；0/1/2 不受影响。该选项贯通 Python、dispatcher schema、Meta 和 C++，不增加 ACLNN 属性。

- [schema 与 Meta](../torch_extension/ffn_worker_batching.py)：`Tensor(a!) schedule_context` 声明上下文更新；Meta 仅按静态属性推导八个输出的 shape/dtype/stride，包括 uint8 表示与 FP4 singleton `(2,1)` stride。
- [C++ wrapper](../torch_extension/csrc/ffn_worker_batching.cpp)：在 DeviceGuard 内分配原生输出、workspace 并下发 ACLNN；随后用 dtype view 将 MX scale 和 FP4 y 以 uint8 返回。共享同一存储，不做数值转换，ACLNN 描述符仍为原生类型。
- [图测试](../tests/st/arch35/verify_torch_graph.py)：使用 fullgraph=True，任何断图都会失败；复用 [eager golden](../tests/st/arch35/verify_torch_interface.py) 的真实指针上下文，逐字节核验有效 token/scale，并核验元数据、索引、计数及上下文/描述符状态。

执行顺序为 Dynamo 追踪 → Meta 推导 → 可变输入函数化 → 捕获 ACLNN 任务 → replay 内核 → 上下文更新回写。设备端 kernel、tiling 和清理协议不变；每次 replay 仍读取实时 ready 状态、处理选中项并推进游标，不增加图内主机轮询。

图外可通过 `scales.view(torch.float8_e8m0fnu)`、`y.view(torch.float4_e2m1fn_x2)`（仅 FP4）恢复类型，不能用 `.to(dtype)` 代替。图内下游需支持字节表示，本次验证不代表任意原生 MX 算子链可入图。

## 环境与产物

- Ascend950DT_9574，设备 0；Python 3.12、Torch 2.9.0+cpu、torch_npu 2.9.0.post8、CANN 9.2.0。
- OPP：`/tmp/ffn_refactor_verify_opp/vendors/custom_transformer`，匹配当前设备实现。
- wheel：`/tmp/ffn_mx_graph_wheels/`；隔离安装：`/tmp/ffn_mx_graph_install/`；包名 `cann_ops_transformer_ffn_mx_graph`。
- 构建/安装日志：`/tmp/ffn_mx_graph_build.log`、`/tmp/ffn_mx_graph_install.log`。
- 图日志：`/tmp/ffn_mx_graph_verify.log`；[图结果](arch35_torch_graph_mx_results.json)。
- eager 日志：`/tmp/ffn_mx_graph_eager.log`；[98 项结果](arch35_torch_mx_results.json)。

## 验证范围

| 场景 | 数量 | 核验重点 |
| --- | --- | --- |
| 六种 dtype × 四组 need_schedule/sync_flag | 24 | NORM 忽略 sync_flag；同步收齐；异步跳过空批 |
| 六种 dtype 全 mask | 6 | 零有效 token 仍清理并更新状态 |
| 六种 dtype singleton | 6 | 最小 shape、FP4 单字节打包和 stride |
| 六种 dtype 连续 replay | 6 | 跨批消费、游标回绕、只重置 flag 不复活旧 ids |
| FP16 同图两次调用 | 2 | 保留/丢弃首次输出均正确消费两批数据 |
| FP16 延迟发布 | 2 | replay 等待实时 flag，独立生产流发布后完成 |
| 合计 | 46 | 全部 fullgraph=True，零跳过 |

六种 dtype 基础用例均设置 `mx_output_uint8=True`，FP16 双调用用例同时保留默认参数路径。连续 replay 每次执行后立即校验，避免图输出存储复用干扰。延迟发布在首次成功执行后保持指针不变、重置上下文，并确认消费流在 200ms 观察窗口内尚未完成，再发布同步或异步 flags。

## 历史探针与当前边界

原生 MX 输出曾在 `decompose_auto_functionalized` 触发 `auto_functionalized_v2 was not removed`：本机 PyTorch 对 FP4/E8M0 的图节点处理有限制。关闭 `input_inplace_pass` 后另有 pybind 序列化错误。最新方案保留默认后端配置，通过 uint8 输出绕过该阻塞；`mx_output_uint8=False` 的原生 MX 图路径仍不列入支持范围。

[原生 MX 探针](arch35_torch_graph_mx_probes.json)、[原图结果 25 项](arch35_torch_graph_results.json) 与 [原产物哈希](arch35_torch_graph_artifacts.json) 仅为历史证据，不代表最新源码/wheel。原 eager 86 项见 [历史结果](arch35_torch_results.json)。

GE 探针曾选择 CANN 内置旧封装，缺少 sync_flag，使 tiling 收到错误属性；独立 OPP 路径另遇到原型发现失败。当前没有注册 GE Converter，npugraph_ex 通过不代表 GE 通过。

持有上下文及其指向的全部设备缓冲，保持地址有效、同一上下文串行消费，图外生产者负责发布与同步。框架无法推导二级裸指针背后的任意读写依赖。计数 `.item()` 放在图外，长期保存输出须在下次 replay 前复制并同步。不要增加会提前消费状态的 warmup 预跑。当前未验证动态 shape、多卡、URMA 联调、跨进程缓存或性能收益。

## 复现

环境、最新 wheel 构建及三个 verify 命令统一见 [验证命令指南](arch35_verify_guide.md)。完整 API 与示例见 [Torch 接口文档](../../../torch_extension/cann_ops_transformer/docs/zh/ffn_worker_batching.md)。
