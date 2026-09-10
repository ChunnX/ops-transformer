# arch35 低精度扩展验证报告

验证日期：2026-09-08。算子仓基线 b54822324a，验证对象为当时低精度扩展后的源码；TTK 基线 7136cc9。

> 本文及配套 CSV 为低精度扩展阶段的历史验证记录，保留原日期、命令、包路径和结果，不作为最新工作区产物说明。包含异步调度与 `LoadSelectedMicroBatch` 的当前实现验证见 [最新全量验证报告](arch35_sync_flag_validation.md)。

## 结果

| 验证项 | 结果 |
| --- | --- |
| Ascend950 完整二进制算子包构建 | 通过 |
| Host tiling / InferShape / InferDataType UT | 32/32 通过 |
| Kernel ST | 68/68 通过（原有 20 + 新增 48） |
| ACLNN ST | 67/67 通过（原有 19 + 新增 48） |
| Kernel 大 scale 分块补充复测 | 6/6 通过，已合入上面的 68 条最终结果 |

无失败或跳过用例。最终 CSV 中的属性与对应设备结果逐一核对，共 135 条唯一 ST 结果，见 [逐用例结果](arch35_low_precision_results.csv)。

三种新类型均覆盖 need_schedule=0/1、H=2 最小行、H=66 尾组、多核搬运、部分/全部 mask、单专家、大 token 分块和大 scale 分块。大 scale 用例的最终 H=4194306，scale 行 131073 字节，超过单个 UB 搬运块；全量 kernel 首轮的大 scale 用例较小，因此以补充复测结果替换相同用例名的首轮结果。

Host UT 覆盖六组类型推导、三种新类型的尾组输出 shape、FP4 奇数 H 拒绝、非法 token_dtype、以及 Ascend910B/Ascend910_93 拒绝 3/4/5。旧 arch35 UT 中已失效的 A2 tiling 结构断言已更新为当前 arch35 ABI 和分段排序语义，保留相应场景。

## 环境和构建

- NPU：Ascend950DT_9574，设备 0，V120。
- CANN：`/home/chenpengyi/Ascend/cann-9.2.0`。
- Python：3.12.14；PyTorch：2.9.0+cpu；numpy：1.26.4。
- 测试框架：`/home/chenpengyi/ops-test-kit`。
- 自定义包安装目录：`/tmp/ffn_mx_opp`，通过 ASCEND_CUSTOM_OPP_PATH 选取。

```bash
source /home/chenpengyi/Ascend/cann/set_env.sh
export PATH=/home/chenpengyi/.conda/envs/py312-cpy/bin:$PATH
cd /home/chenpengyi/ops-transformer
bash build.sh --ophost_test --soc=ascend950 --ops=ffn_worker_batching -j8
bash build.sh --pkg --soc=ascend950 --ops=ffn_worker_batching -j8
mkdir -p /tmp/ffn_mx_opp
bash build_out/cann-ops-transformer-custom_linux-x86_64.run --quiet --install-path=/tmp/ffn_mx_opp
source /tmp/ffn_mx_opp/vendors/custom_transformer/bin/set_env.bash
```

ACLNN 使用完整二进制包，不能以仅 `--jit` 的包代替。测试的 gather 头文件与工作区逐字节一致，SHA256 为 `cfaf05d7f18c59345ef97fd525943cedd36fb634c15795a3ed01dd5a606e603e`。

## ST 复现

依次运行，避免在同一 TTK 目录并发启动多个测试命令：

```bash
cd /home/chenpengyi/ops-test-kit
python -m ttk kernel \
  -i /home/chenpengyi/ops-transformer/ffn/ffn_worker_batching/tests/st/arch35/test_ffn_worker_batching_arch35.csv \
  --plugin /home/chenpengyi/ops-transformer/ffn/ffn_worker_batching/tests/assets/golden.py \
  --pc 4 --proc-timeout 300 --dump-on-fail -o /tmp/ffn_mx_kernel_all.csv
python -m ttk aclnn \
  -i /home/chenpengyi/ops-transformer/ffn/ffn_worker_batching/tests/st/arch35/test_aclnn_ffn_worker_batching_arch35.csv \
  --plugin /home/chenpengyi/ops-transformer/ffn/ffn_worker_batching/tests/assets/golden.py \
  --pc 4 --proc-timeout 300 --dump-on-fail -o /tmp/ffn_mx_aclnn_all.csv
```

原始日志和结果保留在 `/tmp/ffn_mx_build_binary.log`、`/tmp/ffn_mx_ut_final.log`、`/tmp/ffn_mx_kernel_all.{log,csv}`、`/tmp/ffn_mx_kernel_scale.{log,csv}`、`/tmp/ffn_mx_aclnn_all.{log,csv}`。

## 接口约束

新增支持仅限 arch35；FP4 要求 H 为偶数。每 32 个逻辑元素对应一个 E8M0 scale，尾组向上取整，scale 紧跟整行 token 数据。详细存储协议和 UB 规划见 [需求设计](arch35_low_precision_design.md)。本次验证功能和位模式一致性，未设定性能提升目标。

## Torch 接口与最新图模式方案

Torch 扩展支持六种 token_dtype 的 eager、Meta/FakeTensor 及固定形状 npugraph_ex 推理图。
MX 图模式使用 `mx_output_uint8=True`：E8M0 scale 和 FP4 打包输出以 uint8 返回，FP8 token 保持原生类型；
默认 False 保留原生 eager dtype。该参数仅控制 Torch 输出表示，ACLNN 仍使用原生类型描述符，
设备端存储布局、scale 分组、sync_flag 和描述符清理协议不变。
最新图测试 46/46、eager/Meta 回归 98/98 通过；这些是 Torch 层验证，不替换本报告的底层阶段记录。
详见 [图模式报告](arch35_torch_graph_validation.md) 与 [三个 verify 命令指南](arch35_verify_guide.md)。
