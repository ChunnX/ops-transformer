# arch35 syncFlag 验证报告

验证日期：2026-09-10。执行该轮验证时的基线提交为 `231489eb5b`，验证对象为该提交叠加当前工作区的即时清理修订及 `LoadSelectedMicroBatch` 函数提取。需求、流程和核心代码解读见 [需求设计](arch35_sync_flag_design.md) 与 [架构说明](arch35_architecture.md)。本报告记录函数提取后的重新验证结果，以下数据均来自本轮重新构建和执行。

## 本次修改与检查结论

按已确认的生产者约定，异步场景也不会覆盖本次正在消费的数据。同步与异步 RECV 共用 prepare 阶段的清理逻辑：选中行 ids 搬入 `flatIds` 后，将原 ids 写为 `MASK_SENTINEL`、flag 写为 0；未选中行仅在 workspace 中填 sentinel，原描述符保持不变。移除 `ProcessRecv` 的 release 参数、gather 后的额外 `SyncAll` 和第二次 prepare 调用。

保留 `MTE3_V`，确保 ids 搬出读取 UB 完成后才覆盖 UB；保留清理后的 `MTE3_MTE2`，确保 DMA 完成后才能复用缓冲。ready 快照发布与各计算阶段原有的跨核同步保持有效。排序使用私有 ids 副本，gather 仍读取原始 token_data；两者的依赖已分别补充注释。

`sync_flag=1` 仅在 `need_schedule=1` 生效：空轮继续扫描，发现首个非空 micro batch 快照即处理快照内就绪 session。`sync_flag=0` 继续等待入口 micro batch 的 A 个 flag 全部就绪；`need_schedule=0` 保持原路径。本轮 kernel/tiling 修改范围仅为 arch35，属性、TilingData、workspace 布局和 tiling key 未再变更。

## 环境与产物

- NPU：Ascend950DT_9574，设备 0，V120。
- CANN：`/home/chenpengyi/Ascend/cann-9.2.0`。
- Python：`/home/chenpengyi/.conda/envs/py312-cpy/bin/python`。
- 测试框架：`/home/chenpengyi/ops-test-kit`，基线 `7136cc9`。
- 本轮统一安装包：`/tmp/ffn_refactor_verify_opp/vendors/custom_transformer`。

[产物清单](arch35_sync_flag_artifacts.json) 记录本轮 7 个 kernel 源文件、2 个 host 源文件及 6 个 kernel 二进制的 SHA256。安装包内 7 个 kernel 源文件与本轮验证时的工作区逐字节一致。Kernel、ACLNN 和真实地址状态测试均使用此安装包。

## 验证结果

| 验证项 | 结果 |
| --- | --- |
| Ascend950 完整二进制包构建 | 通过，六种 token dtype 对应 kernel 重新编译 |
| Host tiling / InferShape / InferDataType UT | 33/33 通过 |
| Kernel ST（新构建包的预编译二进制上板执行） | 106/106 通过，内存越界检查全部通过 |
| ACLNN ST（新构建包，全量单轮执行） | 105/105 通过 |
| 真实地址、持续上下文状态测试 | 9/9 通过 |

最终全量运行无失败或跳过。211 条 ST 的用例名、属性均与仓库输入 CSV 逐项核对，结果见 [逐用例结果](arch35_sync_flag_results.csv)。Kernel 使用 `-d false -b release`，结果列为 `bin_precision` / `bin_oob_result`；本轮未执行 TTK 动态编译模式，编译验证由前述完整包构建完成。

ST 覆盖六种 token_dtype 的部分 ready、末尾 session ready、全部 ready、有效 token 数为零、M=3 跳过空 micro batch，以及 need_schedule=0 时 syncFlag=1；包含 A=1024 的部分 ready FP16 与末尾 ready FP4 边界。未就绪描述符保留合法 ids，以检查是否误读。

真实地址测试绕过内嵌偏移测试模式，使用持续设备上下文。每次调用校验 token、四类索引、group_list、actual_token_num、完整描述符、polling_index 及 token_data。无数据时延迟 200ms 后检查流仍未完成，再发布单 flag；同步对照则先发布一个、延迟发布其余三个，确认仍需等待全部 session。

[状态测试结果](arch35_sync_flag_state_results.json) 记录 9 个场景，输出 token 数依次为 `9, 9, 18, 9, 0, 36, 2, 1, 0`。新增的三个连续场景仅增量写入设备上的指定 ids/token，随后发布 flag：复用槽位恰好输出两个 token，并保留另一 micro batch 的 ready 描述符；下一次调用消费保留项，输出一个 token；最后只发布 flag、不写 ids，输出零 token。这直接验证 ids 清理没有遗漏，且未选中描述符没有被误清。

验证范围为单 NPU、单算子功能与协议；未执行 attention_to_ffn_v2 多机 URMA 端到端联调，未进行性能提升验收。

## 测试框架诊断耗时处理

前一轮即时清理验证的首次 Kernel 全量运行因 TTK `get_loaded_so_path` 调用 `lsof -p PID` 查询已加载动态库路径而明显变慢。进程检查确认耗时位于诊断日志路径，当时已中断该轮。本轮直接使用已确认的诊断查询替代方式，全量执行两种 ST。

最终两轮 ST 临时将该诊断查询优先改为读取 `/proc/self/maps`，保留原 lsof 后备路径；运行、golden、精度比较、越界检查和性能采集逻辑均未修改。完成后已逐字节恢复 `ttk/utilities/file_utils.py`，并确认测试框架工作区干净。临时修改前后副本保存在 `/tmp/ffn_refactor_verify_ttk_before.py` 和 `/tmp/ffn_cleanup_ttk_file_utils_patched.py`，供核查。复现时如遇相同耗时，可据此定位诊断调用，算子不依赖该修改。

## 复现命令

构建、安装及 Host UT：

```bash
source /home/chenpengyi/Ascend/cann/set_env.sh
export PATH=/home/chenpengyi/.conda/envs/py312-cpy/bin:$PATH
cd /home/chenpengyi/ops-transformer
bash build.sh --pkg --soc=ascend950 --ops=ffn_worker_batching -j8
mkdir -p /tmp/ffn_refactor_verify_opp
bash build_out/cann-ops-transformer-custom_linux-x86_64.run --quiet --install-path=/tmp/ffn_refactor_verify_opp
bash build.sh --ophost_test --soc=ascend950 --ops=ffn_worker_batching -j8
source /tmp/ffn_refactor_verify_opp/vendors/custom_transformer/bin/set_env.bash
```

ST 在 TTK 目录依次执行：

```bash
cd /home/chenpengyi/ops-test-kit
python -m ttk kernel -d false -b release \
  -i /home/chenpengyi/ops-transformer/ffn/ffn_worker_batching/tests/st/arch35/test_ffn_worker_batching_arch35.csv \
  --plugin /home/chenpengyi/ops-transformer/ffn/ffn_worker_batching/tests/assets/golden.py \
  --pc 4 --proc-timeout 300 --dump-on-fail -o /tmp/ffn_refactor_verify_kernel.csv
python -m ttk aclnn \
  -i /home/chenpengyi/ops-transformer/ffn/ffn_worker_batching/tests/st/arch35/test_aclnn_ffn_worker_batching_arch35.csv \
  --plugin /home/chenpengyi/ops-transformer/ffn/ffn_worker_batching/tests/assets/golden.py \
  --pc 4 --proc-timeout 300 --dump-on-fail -o /tmp/ffn_refactor_verify_aclnn.csv
timeout 90 python /home/chenpengyi/ops-transformer/ffn/ffn_worker_batching/tests/st/arch35/verify_sync_flag_state.py
```

ACLNN 接口沿用基线提交中 `layerNum` 后增加 `syncFlag` 的签名，调用方使用新头文件重新编译。

## 原始证据

- 构建、安装、Host UT：`/tmp/ffn_refactor_verify_build.log`、`/tmp/ffn_refactor_verify_install.log`、`/tmp/ffn_refactor_verify_ut.log`。
- 最终 Kernel：`/tmp/ffn_refactor_verify_kernel.log`、`/tmp/ffn_refactor_verify_kernel.csv`。
- 最终 ACLNN：`/tmp/ffn_refactor_verify_aclnn.log`、`/tmp/ffn_refactor_verify_aclnn.csv`。
- 状态测试：`/tmp/ffn_refactor_verify_state.log`；持久化结果及源码/二进制哈希见本目录链接。

原始日志和安装包位于本机临时目录；仓库内保留逐用例摘要、状态结果和产物清单。另已完成 Python 语法、文档本地链接和 `git diff --check` 检查。


## 函数提取的专项检查

`RunPrepare` 在原异步分支、发布屏障之后调用 `LoadSelectedMicroBatch(readyWs, ctx, pipe)`。函数保持 GM 读取、`MTE2_S`、本核 ctx 赋值和 `pipe.Reset()` 的顺序；原函数块提取对比已通过，跨核快照布局不变。

此前函数提取后的编译已确认 6 个 kernel 二进制与即时清理版本一致；按本次验证要求，仍从当前源码重新构建、独立安装，并实际重跑全部 Host UT、Kernel/ACLNN ST 和状态测试。上述结果均为本轮实测，不以二进制一致性替代上板验证。

## 当前提交与验证记录的对应关系

当前提交为 `20d40e5ed0`。重新核对产物清单中的 9 份 host/kernel 源文件 SHA256，当前工作区与该轮全量验证的源码全部一致。提交标识变化不替换实际执行测试时的基线记录；上述构建、33 条 UT、211 条 ST 与 9 条状态测试仍对应当前实现。此次文档核对只修改说明文字，没有修改源码或重复运行上板测试。

## Torch 接口与最新图模式方案

Torch 扩展支持六种 token_dtype 的 eager、Meta/FakeTensor 及固定形状 npugraph_ex 推理图。
MX 图模式使用 `mx_output_uint8=True`：E8M0 scale 和 FP4 打包输出以 uint8 返回，FP8 token 保持原生类型；
默认 False 保留原生 eager dtype。该参数仅控制 Torch 输出表示，ACLNN 仍使用原生类型描述符，
设备端存储布局、scale 分组、sync_flag 和描述符清理协议不变。
最新图测试 46/46、eager/Meta 回归 98/98 通过；这些是 Torch 层验证，不替换本报告的底层阶段记录。
详见 [图模式报告](arch35_torch_graph_validation.md) 与 [三个 verify 命令指南](arch35_verify_guide.md)。
