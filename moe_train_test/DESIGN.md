# Dispatch / Combine 训练测试框架设计报告

## 1. 范围

本目录是训练接口测试框架，所有文件扁平放在 `/home/zerozill/.openclaw/projects/moe_train_test`。

实现约束：

- 外部接口统一调用 `ElasticBuffer.dispatch()` 和 `ElasticBuffer.combine()`。
- 配置变量尽量使用训练接口暴露名：`num_tokens`、`hidden`、`num_topk`、`num_experts`、`num_max_tokens_per_rank`。
- Golden 生成只做 CPU 纯模拟，不调用 `torch.distributed`、HCCL 或 torch 通信。
- Combine 输出必须完全一致；判定使用 byte/tensor exact equal，不使用阈值。
- 失败诊断保留 `toolss.py`，但 `toolss` 只用于打印差异，不改变 exact 判定。
- 保留单算子、性能、单跑和 dispatch->combine 连跑模式；不支持图模式。
- Dispatch 输入支持 bfloat16/float16 tensor 和 fp8 tuple：`(x, scales)`，`scales.shape == (real_num_tokens, dim_scales)`；bf16/fp16 tensor 不带 scales；fp8 tuple 的 `scales` 支持 float32 和 fp8_e8m0，其中 fp8_e8m0 在 CPU golden 中按参考脚本使用 uint8 存储编码值。
- `is_variable_bs=True` 时，输入先按配置 `num_tokens` 生成完整数据，再按每 rank 的实际 token 数取 prefix slice；不引入 decode 的 mask 参数，变长只体现在每 rank 输入 tensor 第 0 维，`num_max_tokens_per_rank` 仍为上界。
- `is_random_scales_dtype=True` 且 `x_dtype` 为 fp8 时，基于固定 `random_seed` 在 float32/fp8_e8m0 中选择 case 级实际 scales dtype，并按实际 dtype 推导 `effective_dim_scales`；关闭时使用输入 `scales_dtype` 和 `dim_scales`。
- 非 cache 模式下，`dispatch_with_topk_weights` 控制 dispatch 是否传入 `topk_weights`；cache 模式下正式 dispatch 固定不传且不比较 `topk_weights`。该 weight 属于 dispatch 后的中间计算路径。`combine_with_topk_weights` 控制 combine 是否独立传入 expanded topk weights，单跑 combine 会按 handle 的 expanded 行数生成自己的 combine weight，和 dispatch 是否传 weight 无关。

接口语义以 `/home/zerozill/workspace/ops-transformer-skdsn-metamon_` 详设为准。Ascend decode V2 文档和代码仅用于理解参考脚本数据格式，包括 `expandXOut`、`assistInfoForCombineOut`、`expertTokenNumsOut`、`epRecvCountsOut` 与 combine 配套关系：

- https://www.hiascend.com/document/detail/zh/Pytorch/2600/apiref/torchnpuCustomsapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_moe_distribute_dispatch_v2.md
- https://www.hiascend.com/document/detail/zh/Pytorch/2600/apiref/torchnpuCustomsapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_moe_distribute_combine_v2.md
- 本地 CANN 参考：`/home/zerozill/.openclaw/projects/ops-transformer/mc2/moe_distribute_dispatch_v2`、`/home/zerozill/.openclaw/projects/ops-transformer/mc2/moe_distribute_combine_v2`
- 本地 torch 适配参考：`/home/zerozill/.openclaw/projects/op-plugin/op_plugin/ops/opapi/MoeDistributeDispatchV2KernelOpApi.cpp`、`MoeDistributeCombineKernelV2OpApi.cpp`

## 2. 文件组织

- `moe_train_test.py`：训练测试驱动，保留参考脚本多进程初始化、启动、回收结构。
- `golden.py`：CPU-only golden 生成、feature/handle 数据结构和 CPU-only feature 操作。
- `compare.py`：dispatch/combine 结果对比、exact tensor/feature 判定、toolss 失败诊断和格式化打印。
- `cache.py`：可复用 golden cache。
- `test_golden.py`：TDD UT。
- `toolss.py`：参考脚本同款精度诊断工具。
- `README.md`、`.gitignore`、`DESIGN.md`：使用说明、忽略规则和设计报告。

## 3. Golden 语义

### Dispatch

输入：

- `x_per_rank[rank]`: tensor 或 `(x, scales)`。
- `topk_idx_per_rank[rank]`: `[real_num_tokens, num_topk] int32`。
- `topk_weights_per_rank[rank]`: `[real_num_tokens, num_topk] float32`，作为 dispatch 输入候选；cached 预热 dispatch 始终传入非 None weight，正式 cached dispatch 固定不传。

CPU 模拟流程：

1. 遍历每个 `src_rank/token_idx/topk_pos`。
2. `expert == -1` 跳过。
3. `dst_rank = expert // num_local_experts`，`local_expert = expert % num_local_experts`。
4. 对同一 token 发往同一 `dst_rank` 的多 topk 做 rank-level dedup，生成 `dst_buffer_slot_idx`。
5. receiver 侧按固定顺序输出：`local_expert -> src_rank -> slot -> token_idx -> topk_pos`。
6. 生成 `recv_x`、可选 `recv_topk_weights`、`recv_src_metadata[A,4]`，四列依次为 `src_rank/token_idx/topk_pos/slot_idx`。
7. 构造 `GoldenHandle`：`dst_buffer_slot_idx`、`recv_src_metadata`、`num_recv_payloads_per_rank`、`num_recv_tokens_per_expert`、`topk_idx`、`num_experts`、`expert_alignment`、`num_max_tokens_per_rank`、`num_expanded_tokens`。

`A_payload` 和 `A_expand` 分开处理：

- `A_payload = sum(num_recv_payloads_per_rank)`，表示 rank-level dedup payload。
- `A_expand = sum(num_recv_tokens_per_expert)`，表示 expanded assignment 行数。

`do_cpu_sync=True` 时 `num_expanded_tokens == A_expand`；`do_cpu_sync=False` 或 cached dispatch 时按 `ep_world_size * num_max_tokens_per_rank * min(num_topk, num_local_experts)` 上界分配。

### Combine

Combine 只依赖 expanded 输入和 `handle.recv_src_metadata`。expanded 输入在进入 combine 前按参考脚本统一构造：dispatch `recv_x` 为 fp8 tuple 时先丢弃 `scales`，主 tensor 基于固定 `random_seed` 转为 fp16/bf16；dispatch `recv_x` 已为 fp16/bf16 时保持原 tensor。单跑 combine 的 golden 和 NPU 入参、连跑 combine 的实际入参都使用这套转换。

1. 遍历每个接收 rank 的 expanded rows，读取 metadata `[src_rank_id, src_token_idx, topk_pos]` 并收集反向映射。
2. 按 `(src_rank_id, src_token_idx, topk_pos)` 排序后累加，对齐参考脚本 `middle_idx` 反排序后的 `token_id_global -> top_id` 顺序。
3. 将对应 `x` 先累加到 float32 accumulator，全部累加结束后再 cast 回 combine 输入 `x` 的 fp16/bf16 dtype。
4. 如果传入 expanded topk weights，则 scatter 到 `combined_topk_weights[src_token_idx, topk_pos]`。

训练详设中的 combine 是纯 sum；不在算子内乘 router weight。

## 4. 和 Decode 参考脚本 Golden 的变量映射

参考脚本：`/home/zerozill/.openclaw/projects/pta-moe-test/moe_operator_test_v3v4_noq.py`。

本框架只复用 decode 脚本中 EP-only、无 shared expert、无 TP、无 mask、无量化通信的可比子集。参考脚本实际调用的 decode V2 数据里，`assistInfoForCombineOut`、`epRecvCountsOut` 等按官方说明是 dispatch/combine 成对使用的内部辅助信息，不适合跨版本当业务语义依赖；训练接口改为 `EPHandle` 承载路由元数据。

可直接比较的变量：

| 训练测试变量 | 参考脚本变量 | 对比方式 |
| --- | --- | --- |
| `x_per_rank` | `x_list[tp_id].chunk(ep_world_size)` | 输入一致性基础，不单独输出比较。 |
| `topk_idx_per_rank` | `expert_ids_list[tp_id].chunk(ep_world_size)` | 输入一致性基础，不单独输出比较。 |
| `dispatch_goldens[rank].recv_x` | `gen_dispatch_golden()` 中 `moe_dispatch()` 返回的 `expanded_x`，再按 `golden_moe_token` split 并按目标 rank concat 后的 `golden_expand_x[rank]` | 有效行完全一致。fp8 tuple 同时比较主 tensor 和 scales。 |
| `dispatch_goldens[rank].recv_topk_weights` | 由 `sorted_col_idx` 索引全局 `topk_weights.flatten()` 得到的 expanded weights | 非 cache 且传入 weight 时有效行完全一致；未传 weight 分支应为 `None`；cached 正式 dispatch 固定不传且不比较。 |
| `dispatch_goldens[rank].handle.recv_src_metadata[:, 0]` | `sorted_col_idx // num_topk` 按每 rank token 长度 prefix offset 反解 | 对应 `src_rank_id`。 |
| `dispatch_goldens[rank].handle.recv_src_metadata[:, 1]` | `sorted_col_idx // num_topk` 按每 rank token 长度 prefix offset 反解 | 对应 `src_token_idx`。 |
| `dispatch_goldens[rank].handle.recv_src_metadata[:, 2]` | `sorted_col_idx % num_topk` | 对应 `topk_pos`。 |
| `dispatch_goldens[rank].handle.recv_src_metadata[:, 3]` | rank-level dedup 生成的 slot | 对应 `slot_idx`，仅 cached dispatch 模式参与对比。 |
| `dispatch_goldens[rank].handle.num_recv_tokens_per_expert` | `golden_moe_token[rank * num_local_experts : (rank + 1) * num_local_experts]` | 完全一致。 |
| 从 `handle.recv_src_metadata` 和 `num_recv_tokens_per_expert` 重建的 per-expert src-rank 累计计数 | `golden_ep_recv_count_list[tp_id][rank]` | 完全一致，用于校验 source rank 分布。 |
| `combine_goldens[rank].combined_x` | `gen_combine_golden()` 中 `middle_idx_list == sorted_col_idx` 反排序后的纯 MoE combine 结果，且 `expert_scales == 1` | 完全一致。参考脚本默认会乘 `expert_scales`，训练 combine 不做该乘法，因此只比较 unit-scale 语义。 |

不能直接比较但会按训练详设独立校验的变量：

| 训练测试变量 | 参考脚本变量 | 原因 |
| --- | --- | --- |
| `handle.dst_buffer_slot_idx` | `golden_expand_idx` / `assistInfoForCombineOut` | 参考脚本 assist info 是 decode combine 辅助索引；训练 handle 是 rank-level dedup payload slot，语义不同。训练 UT 按详设 dedup 规则独立生成 reference。 |
| `handle.num_recv_payloads_per_rank` | `epRecvCountsOut` / `golden_ep_recv_count` | 训练字段统计 dedup payload；decode 字段统计 expanded token 收发计数，语义不同。 |
| `recv_topk_idx` | 无有效训练输出 | 训练详设固定 `recv_topk_idx is None`，runner 显式校验。 |
| upper-bound tail | decode precise 输出 | `do_cpu_sync=False` 会上界分配；只比较有效行，tail 只在打开 `check_invalid_padding=True` 时检查测试 golden 哨兵。 |

## 5. 对比策略

Dispatch：

- 用实际 `handle.recv_src_metadata` 反查原始 `x/scales/topk_weights`，不依赖算子对同一专家内 token 顺序稳定。
- `recv_src_metadata` 默认排序后比较；`strict_dispatch_metadata_order=True` 时要求原始位置也一致。
- `recv_x`、`recv_topk_idx is None`、`recv_topk_weights` 和完整 `handle` 均纳入结果；dispatch 未传 topk weights 时 `recv_topk_weights` 必须为 `None`。`recv_src_metadata` 第四列 `slot_idx` 仅在 cached dispatch 模式比较，非 cached 模式只比较前三列。
- 输出按 `rank` 分组，每张卡内按接口返回顺序 `recv_x -> recv_topk_idx -> recv_topk_weights -> handle.*` 打印变量结果和展开。
- `handle.*` 内按 `GoldenHandle` 定义顺序打印：`dst_buffer_slot_idx -> recv_src_metadata -> num_recv_payloads_per_rank -> num_recv_tokens_per_expert -> topk_idx -> num_experts -> expert_alignment -> num_max_tokens_per_rank -> num_expanded_tokens`。

Combine：

- `combined_x` 完全一致。
- `combined_topk_weights` 在传入 expanded topk weights 时完全一致；未传时必须为 `None`。
- 输出按 `rank` 分组，每张卡内按 `combined_x -> combined_topk_weights` 的固定顺序打印变量结果和失败展开。
- 每个 compare detail 末尾输出 ASCII 汇总表，逐行展示 `card/item/result`，同一 `rank` 的 `card` 列只在分组中间行居中显示、rank 间用横线分隔，`PASS` 使用绿色，`FAIL` 使用红色，空输出显示灰色 `-`；continue 连跑分别输出 `continue.dispatch` 和 `continue.combine` 表。

## 6. 性能模式

- 跑前清理 `Profiling_Result_train_dispatch`、`Profiling_Result_train_combine`、`Profiling_Result_train_continue`。
- profiler schedule 使用 `active = loop // repeat - warmup`，与参考脚本一致。
- 跑后扫描最新 `op_summary*`，按 30 秒窗口过滤历史文件。
- Dispatch 统计 `MoeEpDispatch`、`MoeEpDispatchEpilogue`，Combine 统计 `MoeEpCombine`。
- Dispatch 额外输出 `MoeEpDispatchTotal`，按同一轮样本将 `MoeEpDispatch + MoeEpDispatchEpilogue` 相加后再统计。
- 与参考脚本一致，每个最新 rank `op_summary*` 文件内每类算子只取最后 `performance_tail_count=50` 条样本；再汇总计算 `mean_us/min_us/max_us`，打印统计样本数 `count` 和解析到的总样本数 `total_count`。
- 性能循环保留参考脚本的大 `torch.exp` 重型算子：第 2 轮对 `10000 x 10000` fp16 NPU tensor 执行 100 次 `torch.exp`。
- `is_save_result_to_csv=True` 时自动写 `csv_result/{dispatch|combine|continue}_performance_result.csv`，表头与参考脚本一致，为 `op_type,case_name,mc2_shared_average,mc2_moe_average,mc2_shared_min,mc2_moe_min,recordtime`。训练接口没有 shared/moe rank 分类，CSV 中两个 average 字段均写该 `op_type` 的 `mean_us`，两个 min 字段均写该 `op_type` 的 `min_us`。
- 精度结果在主流程汇总后自动写 `csv_result/moe_distribute.csv`，格式与参考脚本一致，每行 `current_case_name,tensor_name,res_output`，不额外写表头；`res_output` 来自实际 `toolss.data_compare_np` 返回值，不用框架布尔结果合成。

## 7. 控核模式

- `is_full_core=True` 时不限制当前 stream 核数。
- `is_full_core=False` 时使用 `core_num` 控制核数，并派生 `aic_core_num=core_num`、`aiv_core_num=2 * core_num`。
- 每个 NPU 子进程在第一次调用 dispatch/combine 前执行 `torch.npu.set_stream_limit(torch.npu.current_stream(), aic_core_num, aiv_core_num)`，并打印当前单算子模式使用的 aic/aiv 核数。

## 8. Golden Cache

`cache.py` 可独立复用：

- 缓存目录：`golden_cache/<case_name>/`。
- tensor 保存为 `.bin`，`meta.json` 保存 shape、dtype、case 参数。
- 记录 `moe_train_test.py`、`golden.py`、`cache.py`、`compare.py`、`toolss.py` 的 MD5。
- 参数或 MD5 变化不复用；支持禁用复用和强制重生成。
- `is_reuse_golden` 控制读取已有 cache，`is_force_generate_golden` 控制强制重生成，`is_save_golden` 控制重生成后是否写 cache，默认不保存。

## 9. UT 覆盖

`test_golden.py` 覆盖：

- 多组 `num_tokens/hidden/num_topk/num_experts/world_size`。
- `is_variable_bs=True` 下按 `num_tokens` 完整生成后 prefix slice，且每 rank 输入长度可不同。
- `topk_idx == -1` 无效路由。
- Dispatch 带/不带 `topk_weights`，Combine 独立带/不带 expanded topk weights，cached 分支，以及 continue cached 预热。
- `do_cpu_sync=True` precise 和 `do_cpu_sync=False` upper-bound。
- bfloat16、float16、`float8_e4m3fn`、`float8_e5m2` tuple。
- fp8 scales 路由、`dim_scales` 非固定、float32/fp8_e8m0 scales，以及固定 seed 的随机 scales dtype 选择。
- decode `expanded_x/sorted_col_idx/golden_moe_token/golden_ep_recv_count/middle_idx` 对照。
- rank-level dedup payload 与 expanded token 数不同的场景。
- combine 纯 sum 和 topk weight scatter。
- cache roundtrip、fp8 cache、参数失效、MD5 失效、禁用复用、强制重生成。
- runner dispatch/combine 对比函数、profiling parser、性能参数校验。

当前未在本环境完成真实 NPU/HCCL 跑通验证；NPU 部分需在具备 `torch_npu` 和 `npu_ops_transformer` 的环境运行。
