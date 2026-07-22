# moe_train_test

训练场景 `ElasticBuffer.dispatch()` / `ElasticBuffer.combine()` 测试框架。

## 文件

- `moe_train_test.py`：多进程 NPU 算子测试驱动。
- `golden.py`：CPU-only dispatch/combine golden 生成。
- `compare.py`：dispatch/combine 结果对比、exact 判定、toolss 失败诊断与格式化打印。
- `cache.py`：golden bin 缓存读写，按参数和脚本 MD5 失效。
- `test_golden.py`：CPU golden、缓存、runner 对比和 profiling 解析 UT。
- `toolss.py`：沿用参考脚本的精度诊断工具。
- `DESIGN.md`：设计说明和 decode golden 变量映射。

## 运行

CPU UT：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest test_golden.py
```

只生成/校验 golden 和 cache，不跑 NPU：

```bash
MOE_EP_TRAIN_CPU_ONLY=1 PYTHONDONTWRITEBYTECODE=1 python moe_train_test.py
```

真实 NPU 测试修改 `moe_train_test.py` 顶部参数后运行：

```bash
python moe_train_test.py
```

`x_dtype` 支持 `bfloat16`/`bf16`、`float16`/`fp16` 和两种 fp8：`fp8_e4m3fn`、`fp8_e5m2`。bf16/fp16 输入为普通 tensor 且不带 scales；fp8 输入按接口要求生成为 `(x, scales)` tuple，`scales_dtype` 支持 `float32` 和 `fp8_e8m0`；本地 CPU golden 中 `fp8_e8m0` 用 `uint8` 存储编码值。

`is_random_scales_dtype=True` 且 `x_dtype` 为 fp8 时，脚本会基于固定 `random_seed` 在 `float32` 和 `fp8_e8m0` 中选择本 case 的实际 scales dtype；关闭时使用输入的 `scales_dtype`。

默认 `dim_scales` 由 `scales_dtype` 推导：fp32 scales 使用 `ceil(hidden / 128)`，`fp8_e8m0` scales 使用 `ceil(hidden / 32)` 后向上偶数对齐。`is_random_scales_dtype=True` 时，生成 scales 会按随机得到的实际 dtype 重新推导 `effective_dim_scales`。

`is_variable_bs=True` 时，脚本先按 `num_tokens` 为每张卡生成完整输入，再为每张卡取 prefix slice 作为实际输入；至少一张卡保留 `num_tokens` 行。training 接口不传 mask，变长只体现在 `x/topk_idx/topk_weights` 以及 fp8 `scales` 的第 0 维，`num_max_tokens_per_rank` 继续作为上界。

非 cache 模式下，`dispatch_with_topk_weights` 控制 dispatch 是否传入 `topk_weights`；cache 模式下正式 dispatch 固定不传且不比较 `topk_weights`。该 weight 用于 dispatch 后的中间计算路径，`combine_with_topk_weights` 控制 combine 是否独立传入 expanded topk weights。两个开关互不依赖，单跑 combine 会按 handle 的 expanded 行数生成自己的 combine weight。

单跑 combine 和 dispatch->combine 连跑都会按参考脚本构造 combine 输入：如果 dispatch `recv_x` 是 fp8 tuple，则丢弃 `scales`，主 tensor 基于固定 `random_seed` 随机选择并转换为 fp16/bf16 后传给 combine；如果已经是 fp16/bf16，则保持原 tensor。combine golden 与 NPU 入参使用同一转换结果。

`dispatch_with_handle=True` 时，每个 NPU 子进程会先在同一个 `ElasticBuffer` 上执行一次非 cached dispatch 获取真实 `EPHandle` 并初始化 window，再把该 handle 传入正式 cached dispatch；单跑 dispatch 和 dispatch->combine 连跑都使用这套预热流程，不会直接把 CPU golden handle 传给 cached dispatch。预热 dispatch 始终传入非 None `topk_weights`，正式 cached dispatch 固定不传且不比较 `topk_weights`。

结果对比默认按 `rank` 分组、按固定变量顺序输出 compare detail。PASS 和 FAIL 都会输出对应的 toolss result 返回值和 toolss detail；没有 tensor 可交给 toolss 的结构化输出会打印 skipped 说明。`handle.dst_buffer_slot_idx` 仅在 cached handle 模式按 raw 值 exact 比较，非 cached dispatch 跳过该字段；`handle.recv_src_metadata` 第四列 `slot_idx` 也仅在 cached handle 模式比较，非 cached dispatch 只比较前三列。每个 compare detail 末尾会输出 ASCII 汇总表，同一 `rank` 的 `card` 列只在分组中间行居中显示，绿色 `PASS` 表示通过，红色 `FAIL` 表示失败，空输出显示灰色 `-`。

`is_save_result_to_csv=True` 时，主进程会在结果汇总后自动保存 CSV 到 `csv_result/`：

- `moe_distribute.csv`：逐变量精度结果，格式与参考脚本一致，每行 `current_case_name,tensor_name,res_output`，不额外写表头；`res_output` 来自实际 `toolss.data_compare_np` 返回值，不用框架布尔结果合成。
- `{dispatch|combine|continue}_performance_result.csv`：性能结果，表头与参考脚本一致，为 `op_type,case_name,mc2_shared_average,mc2_moe_average,mc2_shared_min,mc2_moe_min,recordtime`。

`is_reuse_golden` 控制是否读取已有 golden cache，`is_force_generate_golden` 控制是否强制重新生成，`is_save_golden` 控制重新生成后是否写入 golden cache，默认不保存。

性能模式设置 `is_performance = 1`。脚本会在运行前清理本目录下 `Profiling_Result_train_*`，运行后解析最新 `op_summary*`，按参考脚本方式对每个最新 rank 文件内每类算子取后 `performance_tail_count=50` 条样本统计并输出性能 CSV。Dispatch 分别输出 `MoeEpDispatch`、`MoeEpDispatchEpilogue` 和两者逐轮相加的 `MoeEpDispatchTotal`。

控核场景设置 `is_full_core = False` 并调整 `core_num`。脚本会派生 `aic_core_num = core_num`、`aiv_core_num = 2 * core_num`，在 NPU 子进程运行算子前对当前 stream 调用 `torch.npu.set_stream_limit(...)`。
