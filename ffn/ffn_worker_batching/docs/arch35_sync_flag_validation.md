# arch35 syncFlag 验证报告

验证日期：2026-09-08。需求与实现说明见 [需求设计](arch35_sync_flag_design.md)。

## 环境

- NPU：Ascend950DT_9574，设备 0，V120。
- CANN：`/home/chenpengyi/Ascend/cann-9.2.0`。
- Python：`/home/chenpengyi/.conda/envs/py312-cpy/bin/python`。
- 测试框架：`/home/chenpengyi/ops-test-kit`，TTK 基线 `7136cc9`。
- 最终自定义包：`/tmp/ffn_sync_final_opp/vendors/custom_transformer`。

## 验证结果

| 验证项 | 结果 |
| --- | --- |
| Ascend950 完整二进制包构建 | 通过 |
| Host tiling / InferShape / InferDataType UT | 33/33 通过 |
| Kernel ST（动态编译并上板执行） | 106/106 通过，内存越界检查全通过 |
| ACLNN ST | 105/105 通过（首轮 103 + 边界补测 2） |
| 真实地址、持续上下文状态测试 | 6/6 通过 |

无失败或跳过用例。Kernel 为原有 68 条 + 新增 38 条，ACLNN 为原有 67 条 + 新增 38 条。共 211 条 ST 的用例名与属性已逐项核对，见 [逐用例结果](arch35_sync_flag_results.csv)。

状态测试：无数据等待后单 flag 发布、跳过空 micro batch 并保留其它描述符、cursor 回绕后部分接收、已消费槽位复用、ready 但无有效 token、同步模式等待全部 session，全部通过。

新增 ST 覆盖六种 token_dtype 的部分 ready、仅末尾 session ready、全部 ready、ready 但有效 token 数为零、M=3 跳过空 micro batch，以及 need_schedule=0 时 syncFlag=1。另有 A=1024 的部分 ready FP16 和末尾 ready FP4 用例。未就绪描述符保留正常 expert_ids，防止通过预先 mask 未就绪数据掩盖误读。

独立状态测试使用真实设备地址，不依赖内嵌偏移测试模式。每次调用检查输出 token/四类索引/group_list/actual_token_num、完整 FfnDataDesc、polling_index 和未改写的 token_data。无数据时启动 kernel，延迟 200ms 后检查 stream 仍未完成，再发布 ready flag，验证 kernel 返回。同步对照用例预置一个 ready，再延迟发布其余三个，验证仍需等待全部 session。

本次验证为单算子功能与协议验证；未执行 attention_to_ffn_v2 的多机 URMA 端到端联调，未设性能提升验收目标。生产者仍需遵守先写数据再置 flag、槽位复用以及同一上下文消费者串行的约定。

## 复现

构建及安装：

```bash
source /home/chenpengyi/Ascend/cann/set_env.sh
export PATH=/home/chenpengyi/.conda/envs/py312-cpy/bin:$PATH
cd /home/chenpengyi/ops-transformer
bash build.sh --pkg --soc=ascend950 --ops=ffn_worker_batching -j8
mkdir -p /tmp/ffn_sync_final_opp
bash build_out/cann-ops-transformer-custom_linux-x86_64.run --quiet --install-path=/tmp/ffn_sync_final_opp
bash build.sh --ophost_test --soc=ascend950 --ops=ffn_worker_batching -j8
source /tmp/ffn_sync_final_opp/vendors/custom_transformer/bin/set_env.bash
```

ST（依次执行，不在同一 TTK 工作目录并行启动命令）：

```bash
cd /home/chenpengyi/ops-test-kit
python -m ttk kernel \
  -i /home/chenpengyi/ops-transformer/ffn/ffn_worker_batching/tests/st/arch35/test_ffn_worker_batching_arch35.csv \
  --plugin /home/chenpengyi/ops-transformer/ffn/ffn_worker_batching/tests/assets/golden.py \
  --pc 4 --proc-timeout 300 --dump-on-fail -o /tmp/ffn_sync_kernel_all_result.csv
python -m ttk aclnn \
  -i /home/chenpengyi/ops-transformer/ffn/ffn_worker_batching/tests/st/arch35/test_aclnn_ffn_worker_batching_arch35.csv \
  --plugin /home/chenpengyi/ops-transformer/ffn/ffn_worker_batching/tests/assets/golden.py \
  --pc 4 --proc-timeout 300 --dump-on-fail -o /tmp/ffn_sync_aclnn_all_result.csv
timeout 90 python /home/chenpengyi/ops-transformer/ffn/ffn_worker_batching/tests/st/arch35/verify_sync_flag_state.py
```

ACLNN 使用自动生成的新接口：`layerNum` 后增加 `syncFlag`，调用方必须使用本次头文件重新编译。

## 产物一致性

功能验证完成时，最终安装包的 arch35 全部 kernel 源文件与当时工作区逐字节一致。以下 SHA256 记录该次实际构建测试的源码版本：

- `ffn_worker_batching_apt.cpp`：`19025b87698c3da97e017d95ef45ef92344ec0278939d1e82a264c0f45652dd5`
- `ffn_wb_a5_prepare.h`：`e7ddc37a521276dc65e48b3df3cc4d3f3c9ac403fa72c4b9d9647b8b07da61d8`
- `ffn_worker_batching_arch35_tiling_def.h`：`da216c362f8da71fe1066f37596da56ac2d66ec0d4c92b42d0eb86bfc97b5152`

首轮 ACLNN 使用 `/tmp/ffn_sync_opp`；最终构建调整了同步/NORM 路径的 UB 预算以保留原规划，并整理注释和缩进。两包的 6 个 kernel `.o` 逐字节一致。最终 host 由 UT 和最终包 kernel 全量 ST 验证，最终 ACLNN 包由 A=1024 补测及真实地址状态测试验证。

原始日志：`/tmp/ffn_sync_build_final.log`、`/tmp/ffn_sync_ut_final.log`、`/tmp/ffn_sync_kernel_all.log`、`/tmp/ffn_sync_aclnn_all.log`、`/tmp/ffn_sync_aclnn_max.log`、`/tmp/ffn_sync_state_final.log`。

## 后续注释与设计说明更新

功能验证后补充了 host、kernel、测试代码注释，并扩充了 [设计文档](arch35_sync_flag_design.md) 的流程、内存布局、同步关系和核心代码解读。C/C++ 非注释 token 及 Python AST 对比确认执行逻辑未变，文档本地链接和 diff 格式检查通过；本轮未重复执行上板测试。以上源码哈希和 ST 结果保留为上一轮实际验证记录，不将注释更新后的源码称为与旧安装包逐字节一致。
