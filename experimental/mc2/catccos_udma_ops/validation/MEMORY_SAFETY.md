# UDMA 算子 NaN 与内存安全诊断

`run_memory_safety.sh` 在普通精度测试之外启用四层诊断：

1. 每个 case 开始前清空 PyTorch NPU 缓存，并将当前进程可分配 HBM（默认保留 2 GiB）写成 BF16 NaN；释放后不执行 `empty_cache`，让后续 tensor 复用这些脏块。
2. 每次 kernel launch 前，将约 400 MiB SHMEM 数据 workspace 写成 `0xFF`。末尾 4096 字节同步区不被修改。
3. `x`、`weight` 两侧放置 NaN red zone；route 和可复用输出两侧放置精确 canary。可复用输出 payload 每次 launch 前写成 NaN，从而同时发现越界写和未完整写。
4. 检查输入非有限值、输入抽样指纹变化、每个输出的 NaN/Inf 计数及首批坐标，并额外运行 host-list route 路径。

默认 manifest 同时覆盖对齐均衡 case 和非对齐、非均衡、零 token expert case。后者保持所有 rank 都有通信负载，但让各 rank 的 `sum(global_tokens)` 不同，用来覆盖整网最容易暴露问题的动态行数与尾块路径。

整网复现时还可以设置 `CATCCOS_UDMA_VALIDATE_ROUTES=1`。它会在 Python facade 中将 device route 同步回 Host，检查非负/uint32 范围、输入输出行数和 route 总和。该开关只用于定位，不能用于性能测试。

## 运行

重新构建并安装 wheel（新增了 `_poison_udma_workspace` 诊断 raw op），然后执行：

```bash
bash scripts/run_memory_safety.sh
```

在关注的大 shape 上运行：

```bash
bash scripts/run_memory_safety.sh \
  --cases validation/cases/alltoallv_grouped_mat_mul.yaml \
  --case-id model_case_1 \
  --hbm-reserve-gib 4
```

结果重点看：

- `rank_<n>/memory_safety.json`：HBM/workspace 污染、输入 red zone、输入突变和输出 canary 总表；
- 每条输出路径的 `report.json`：显式 `FAIL_ACTUAL_NONFINITE` / `FAIL_MEMORY_SAFETY`；
- `nonfinite_values.csv`：首批 NaN/Inf 的 row、column、expert、source rank；
- `top_mismatches.csv`：普通精度误差分布。

`--hbm-reserve-gib` 必须给 CANN、Golden、输出 clone 和通信运行时留下空间。大 shape 若 OOM，应增大该值，而不是减小。不能安全覆盖驱动、其他进程或 SHMEM 已占用的 HBM，因此“可分配 HBM 污染”不能单独证明物理相邻；紧邻 payload 的 red zone 是确定性的补充。

## msSanitizer（补充手段）

```bash
bash scripts/run_mssanitizer.sh --case-id mem_a2av_gmm_tail_zero_experts_ep2
```

默认启用 `memcheck,initcheck`，可用 `MSSANITIZER_TOOLS=memcheck` 覆盖。该工具适合发现本地 GM/MTE 越界和未初始化访问；当前交付固定的 SHMEM 版本不保证所有 UDMA peer 访存都能被插桩，因此它不能替代 workspace poison、red zone 与多 rank 精度检查。

## 如何缩小问题范围

- 只有 workspace poison 后出现 NaN，且 output canary 完整：优先排查通信 stage 未写、flag 生命周期、零 token expert 和尾轮流水。
- output payload 仍有 NaN，但 canary 完整：优先排查未完整写或越界读。
- output canary 被改写：属于确定性越界写，按 `prefix/suffix_corrupted_elements` 判断方向。
- input red zone 或指纹变化：kernel/通信写错输入地址，或存在 storage alias。
- 仅 device route 或仅 host-list route 失败：排查 fallback route 前缀计算与 `MoeRouteMetadata` 路径差异。
- 不启用污染也出现 NaN：先检查整网上游输入/weight/grad、各 rank route 总和及调用顺序是否一致。
