# engram_test 修改设计文档（待审核）

> 状态：**设计稿，未实施**。所有代码改动待用户审核本方案后执行。
> 关联：《engram_test_review.md》（检视报告，含问题编号 F1-F6/H1-H7/M1-M9，本文沿用该编号）

---

## 0. 结论速览

| 议题 | 结论 |
|------|------|
| 修改基线 | **在当前 worktree（examples/mc2/engram_test/）上修改**，不以原目录为基线；原目录作为上游参考，部分实现需回移（见 §2） |
| 82208000 语义 | 你的理解（总 vocab）**正确且应采纳**；但两版现有代码均实现为"每 rank 行数"（总数再 ×4），必须改（M1） |
| 随机数构造慢 | 根因 = 语义 ×4 放大 + fp32 整表中间 + 整表 clone + worker 整块临时 → 峰值内存 >2.2TB，worker 被 OOM kill 后 `Pool.map` 挂死；重构后 large2 预期 **NVMe <1min / NFS <5min**（M2） |
| local/sliced 复用数据 | **可行且推荐**：按缓存键复用；论证结论：**不会使 fetch 计时"运行过快"**，只缩短 prep 墙钟（M3） |
| 显存 OOM | 表在 host 侧不占 HBM；单次输出（16M×576 fp8 ≈ 9.0GiB）合法；主因是 npugraph_ex 21 个档位的图私有内存池累积。方案 = 用例侧预检 + 图池逐档释放 + PTA 侧输出预算拦截（M5） |

---

## 1. 修改基线意见：当前 worktree vs /home/qianlong/workspace/cann/engram_test

**建议：在当前 worktree 上修改。** 理由：

1. worktree 版已是原版的功能超集：fetch_mode(local/sliced)、prep_only/run_only 两段式、qrun 排队、并行数据生成、fetch_token_num 广播均只存在于 worktree 版；以原目录为基线需把这些改动重新搬运。
2. worktree 与被测算子代码（mc2/engram_fetch、PTA elastic_buffer.py）同仓同分支，测试与代码可联动评审、随 PR 演进；e2e 例子（engram_fetch_e2e）也在旁，便于对照语义。
3. 原版仓库（wuchenhao123/engram_test）保留为上游个人仓库。原版独有的资产（grad_test.py 训练链路、analyse.py 带宽分析、fp8_test.py 独立 MXFP8 参考）建议作为 P2 移植项（见 §9），而非修改基线。
4. 两版的公共缺陷（×4 语义、无退出码、queue 无超时、forkserver 全局变量、seed 失效、永真式 golden、compare_write 注释）继承自原版——无论以哪边为基线都要修，工作量相同。

**需要从原版回移的实现（纳入本次修改）**：

| 项 | 原版位置 | 说明 |
|----|---------|------|
| 健壮整数转换 `to_int` | 原版 run_engram.sh:141-154 | 处理 NaN/空串/nan/none/空格并报列名明确的错误；worktree 版(:144)是回归 |
| `result_queue.get(timeout=...)` | 原版 grad_test.py:1349 | 防 F2 挂死的现成思路 |
| psutil CPU 亲和性 | 原版 engram_test.py:59-72, 493-508 | NUMA 亲和对 C2C 取数带宽有实际影响；回移并**参数化**（原版硬编码 288-383/96-191） |

---

## 2. M1 语义修正：input_tensor_shape = 总 vocab

### 2.1 语义核实（回答"82208000 原始语义是否为我理解"）

- **你的理解**：`[[82208000, 576]]` 中 82208000 = 全局 vocab 总行数；local 模式 = 一份全量表映射到每张卡；sliced 模式 = 均分为 4 份映射到 4 张卡。**与 e2e 例子（test_engram_fetch_e2e.py `--num-entries`，"全局表 entry 总行数，需能被 rank 数整除"）及 local 模式物理约束一致，采纳为准。**
- **现有代码（两版相同）**：`total = num_entries_per_rank * world_size`（worktree py:390，原版 py:242）→ 把 shape[0] 当**每 rank 行数**，全局表再 ×4。这不是你想要的语义，且是 large 用例内存/耗时爆炸的第一根因。
- 物理量级佐证（local 模式窗口为 host 侧内存，fp8、world_size=4）：

| case | shape[0] | 按"总 vocab"（正确）host 窗口/卡 | 按"×4 语义"（现状）host 窗口/卡 |
|------|---------|------------------------------|------------------------------|
| small 5,138,000 | 2.84 GB | 11.4 GB | 不可能（超 RAM） |
| medium2 20,552,000 | 11.36 GB | 45.4 GB | 勉强 |
| large2 82,208,000 | 44.1 GiB | 176.4 GiB | 不可行 |

### 2.2 改动内容

1. `parse_cli_args`：变量改名 `num_entries_per_rank → num_entries_total`；`total = n`（不再乘 world_size）。
2. 整除校验：sliced 模式要求 `total % world_size == 0`，否则报错退出（data.md 六档 vocab 均可被 4 整除，已核对：5138000/4=1284500 … 82208000/4=20552000，1024/4=256 ✓）。
3. `write_golden = storage.chunk(world_size)` 逻辑不变（本来就按 total 切）；local 写全量 `storage` 不变。`get_engram_storage_size_hint` 按 write_inputs.shape[0] 推导，自动适配。
4. 迁移兼容：新增 `--shape_semantics {total,per_rank}`，默认 `total`；旧 xlsx 表若按 per_rank 语义构造，需在表上加列或转换（README 注明 breaking change）。
5. **data.md 连带修正（已拍板 2026-09-21）**：语义修正后 small(5.14M)/medium(10.3M) 的 16,777,216 档位超过 vocab。**决策：保留大档位，不判失败**。依据：numTokens > vocab 是真实 LLM 场景——Prefill 阶段 embedding 取数行数 = batch×seq_len，长上下文（128K ctx）batch≥2 即超 vocab（主流模型 vocab 128K~152K）；token 分布服从 Zipf，重复 token 天然大量存在；算子侧对 numTokens 仅校验 ≥0（tiling:188-190），训练路径的去重输出（grad_unique/unique_local_entry）正为重复 index 设计。**约束是 index 值域 ∈ [0, vocab)，而非数量 ≤ vocab**。实现：indices 有放回采样自 [0, vocab)；`num_token > vocab` 时打 WARNING 注明"重复采样，对应 prefill 长上下文/大 batch 场景"，报告记录该档位为 repeated-sampling。

---

## 3. M2 随机数生成重构（性能）

### 3.1 根因定量（为什么"一上午构造不出来"）

现状链路（worktree py:292-319 `_parallel_randn` → `gen_storage_write_golden`）对 [[82208000,576]]（×4 语义下 total=328,832,000 行，189.4G 元素）：

| 步骤 | 内存开销 |
|------|---------|
| ① fp32 整表 /dev/shm | 757.7 GB |
| ② 40 个 worker 各自 `np.random.standard_normal((count,576))` 整块临时（18.9GB/个） | 峰值再 +757.7 GB |
| ③ `torch.frombuffer(...).clone()` 整表复制 | +757.7 GB |
| ④ `.to(fp8)` | 读 757.7 + 写 189.4 GB |
| ⑤ `torch.save` 189.4 GB → run 阶段 `torch.load` 再 189.4 GB | 磁盘 + RAM 双份 |

峰值内存需求 >2.2TB。实际后果：/dev/shm 不足则直接 OSError；勉强够时 worker 被 OOM killer 杀死，`mp.Pool.map` 不感知 worker 死亡 → **永久挂死**（"一上午"的现象解释）。即使不 OOM，总内存流量 ~3.4TB，也需小时级。

三个独立缺陷叠加：**(a) 语义 ×4（M1 修复）；(b) fp32 整表中间 + 无谓 clone；(c) worker 临时块=整块而非小分块。**

### 3.2 新方案：分块直写最终 dtype 的 mmap 缓存文件

废弃 `/dev/shm + frombuffer + clone + torch.save/.pt` 链路，改为：

```
prep 阶段:
  cache_dir/prep_cache/{key}.storage.bin   # 直接按最终 dtype 落盘 (np.memmap, mode='w+')
  cache_dir/prep_cache/{key}.sf.bin        # sf 表同样
  cache_dir/prep_cache/{key}.meta.json     # shape/dtype/seed/use_fp8/sf_dtype/gen_version
  key = sha1(json{shape, dtype, use_fp8, sf_dtype, seed, gen_algo_version})   # fetch_mode 不参与

  worker(i):  rows [r0,r1) 内按 block_rows(如 8192 行 ≈ 4.7M 元素) 循环:
     rng = np.random.default_rng(SeedSequence(seed, spawn_key=(i,)))   # 由 --seed 派生, 可复现
     blk = rng.standard_normal((block_rows, hidden))          # fp32 小块 ≈ 18.9MB, 常驻 L2/L3
     mm[r0+off : r0+off+block_rows] = blk 转换后               # fp8: torch.from_numpy(blk).to(fp8);
                                                              # bf16/fp16: 同法; int: rng.integers 直写
  峰值临时 = workers × 19MB ≈ 0.8GB（对比现状 757GB）

run 阶段:
  parent: np.memmap(mode='r') → torch.from_numpy → (fp8 时 .view(float8_e4m3fn)) → chunk(world_size)
  fork 子进程直接继承（见 M6-3），无 pickle/无共享内存搬运
  engram_write 完成后对 mmap 页 madvise(MADV_DONTNEED) 归还 page cache（可选开关）
```

要点：
- **不再经过 fp32 整表**：转换在小块内完成，最终 dtype 直接落盘；无 clone。
- **不依赖 /dev/shm**：mmap 文件在普通文件系统，无 50% RAM 上限。
- **RNG 升级**：`np.random.Generator(PCG64)` 替代 legacy `np.random.seed/standard_normal`，SIMD 路径约快 2-4×，且支持 SeedSequence 派生（F6 修复：--seed 真正生效）。
- **sf 表**：18 packs，rand 小块 → e8m0 位运算（`bits>>23 & 0xFF`）逐块转换，或 fp32 直存。
- 保留 `--data_pattern {random,arange}` 可选开关：`arange`（行号派生模式填充）近零成本，供纯 perf 冒烟用；**默认 random**（随机内容对地址类 bug 的暴露能力不可替代，golden 位级比对仍全程有效）。

### 3.3 预期耗时（语义修正后，fp8，hidden=576）

| case | 最终数据量 | RNG(40核, ~20G elem/s) | 磁盘写 | prep 总计（NVMe ~3GB/s / NFS ~300MB/s） |
|------|-----------|----------------------|--------|----------------------------------------|
| small 5.14M 行 | 2.84 GB | <1s | 1s | **~2s / ~10s** |
| medium2 20.55M 行 | 11.4 GB | ~1s | 4s | **~5s / ~40s** |
| large1 41.1M 行 | 22.7 GB | ~1.5s | 8s | **~10s / ~80s** |
| large2 82.2M 行 | 44.1 GiB | ~2.5s | 15s | **~20s / ~2.5min** |

对比现状：large2 = 挂死（>一上午）。

### 3.4 索引生成链路同步修复（M4，同属性能）

- `gen_balanced_indices` 返回 `np.ndarray(int32)`，删除 `.tolist()`；`indices_npu = torch.from_numpy(arr).to(device)`；golden 用 `torch.from_numpy(arr).long()` 直接 `index_select`。消除 16M token 的 Python list 中转（数百 MB 对象 + 分钟级耗时）。
- indices 改为 per-rank 确定性种子：`default_rng(SeedSequence(seed, spawn_key=(rank, size_idx)))` → 可复现，且 local/sliced 配对用例生成**相同 indices**，两条用例性能可直接对比。
- `world_size` 作为参数显式传入（配合 M6-3 根治 F5）。

---

## 4. M3 local/sliced 数据复用（同 shape 配对用例）

### 4.1 设计

- 缓存键不含 `fetch_mode`（见 §3.2），data.md 中 `perf_local_X` 与 `perf_sliced_X`（同 shape/dtype/fp8/sf/seed）自动命中同一份缓存文件。
- `prep_only`：命中且 meta 校验通过 → 打印 `[PREP] cache hit, skip generation`，直接退出 0。
- runner：不再删除 prep 文件；`prep_cache/` 目录由 `--purge_prep_cache` 显式清理（或跑批结束后统一清理非当日文件）。run_engram.sh:283-286 的删除逻辑改为仅清 legacy `.pt` 路径。
- 半程开销对比：perf_vocab 11 条用例 5 对配对 → 生成次数 11 → 6，large2 只生成一次；engram_tensor 6 条 3 对配对 → 6 → 3。

### 4.2 "后跑的用例被缓存 → 运行过快"论证（结论：不影响 fetch 计时）

计时对象是 fetch kernel 从 **engram 窗口（host 侧驱动注册内存）** 取数。论证：

1. **窗口内容与数据来源无关**：无论数据来自现场 RNG 还是缓存文件，`engram_write`（memcpy 进窗口）完成后，窗口内容逐字节相同、页常驻 host RAM。kernel 经 C2C/PCIe 读窗口，测得时延只取决于窗口侧带宽/延迟，与"数据是怎么来的"无关。
2. **page cache 只加速 prep**：缓存文件命中使 `np.memmap` 首读更快（页已在 cache）——这只缩短 prep 墙钟，不进入 fetch 计时区段。
3. **NPU 侧无跨用例延续**：每条用例是独立进程，设备重新初始化；表不在 HBM，NPU L2 相对 44GB 表可忽略且 kernel 间自然失效。
4. **窗口页 first-touch 两种情况完全一致**：窗口都是本进程新分配 + memcpy 首写。
5. 唯一附带效应：host RAM 中 page cache 占用（large2 ≈ 44GiB）——用 §3.2 的 madvise(MADV_DONTNEED) 在 write 后归还，规避与 local 模式 4×44GiB 窗口的 RAM 叠加。

结论：**复用安全**；如需额外严谨，可在正式出数时用 `--purge_prep_cache` 复跑一遍对照（预期差异在噪声内）。

---

## 5. M5 显存 OOM：根因分析与拦截方案

> 2026-09-21 修订：结合《oom.md》实测数据（Ascend950DT 91.75GB/卡，CANN 9.2.0，is_graph=1 + dynamic=False），本节从"假设"升级为"实测确认"，并修正预算公式。

### 5.1 HBM 占用定量（推理路径，已核实代码）

- engram 表在 **host 侧窗口**（num_cpu_bytes），**不占 HBM**（1x/4x 实测峰值拟合可证：1x 峰值 19,569MB ≈ Σ全档 fetched = 33.55M×576 = 19.3GB；4x 峰值 79,768MB ≈ 33.55M×2304 = 77.3GB——峰值≈Σ图池输出+零头，表本身不在其中）；
- 推理路径 workspace 恒为 16MB（engram_fetch_tiling.cpp:630 else 分支；aclnn_common.h 按 GetWorkspaceSize 返回值**精确**分配，无放大）——**修正 oom.md §3.5 第 3 行"workspace 上限≈72GiB"的说法**：16x 大档位 OOM 与 workspace 无关，预算公式无需计入巨量 workspace（graph 模式下每图 16MB 进池，可忽略）；
- 输出由调用方分配：`fetched`（T×H×1B，elastic_buffer.cpp:1894）+ `fetched_sf`（T×⌈H/32⌉×1B，elastic_buffer.py:893）+ indices/mul（各 T×4B）。

### 5.2 根因（实测确认）

oom.md 实测：16x（[[321125,9216]]）token=8,388,608 时单次 `fetched` = 8,388,608×9216×1B = **72.0GiB**，卡上仅 13.03GiB free → 4 rank 同时 OOM。此时"已分配 76.94GiB" ≈ sf(~0.1GB) + **前 20 档 graph pool 累积**（Σ fetched(16..4,194,304) = 8,388,592×9216 ≈ 72GiB，数量级吻合）。

两个叠加因素：
1. **单次分配随 H 放大**：`fetched = T×H×elem`，16x(H=9216) 在 T≥8M 时单次即 72~144GiB，任何卡都放不下（91.75GB 卡的理论单次上限 T_max ≈ free×0.8/9216 ≈ 7.7M）；
2. **图池累积**：`dynamic=False` 使 21 档各捕获一张图，图私有池随图对象存活不释放（recompile_limit=256 全保留）——即使 1x/4x 也已在 OOM 边缘（4x 峰值 79.8GB / 91.75GB），属"未爆雷"。

### 5.3 三层拦截（修订版）

**L1 用例侧（必做）**：
- 每个 token 档位执行前预算检查（修订公式，不含大 workspace）：`need = T×H×elem + T×⌈H/32⌉×sf_elem + T×4×2 + 16MB + 2GiB 固定余量`，与 `torch.npu.mem_get_info()[0]`（free）× 0.8 比较；不足则 `[SKIP] size=... need X GiB > free Y GiB` 跳过该档位（不算 FAIL），**报告必须汇总 skipped 列表**，避免"全绿"掩盖跳档。
- **graph 模式逐档位释放**：每档位测完后 `torch._dynamo.reset()`（销毁已编译图及其私有池）+ `del fetched` + `torch.npu.empty_cache()`。注意：oom.md §4.4 只提 `empty_cache()`——**不够**，empty_cache 仅回收空闲缓存块，图池内存随图对象存活，必须 dynamo.reset 销毁图。副作用 = 下档位重新编译，由 warmup 吸收，不进计时区段。
- 按 91.75GB 卡预估各新用例可测档位（M5 落地后自动 SKIP 生效）：1x(H=576) 全 21 档可测；4x(H=2304) 全 21 档可测（16M 档单次 38.7GiB）；16x(H=9216) 仅到 4M 档（36GiB），8M/16M 档必 SKIP。
- M1 语义修正后，超 vocab 的档位按 §2.2-5 处理。

**L2 C++ 层（采纳 oom.md §4.1，落点从 Python 改为 csrc，单一权威点，已拍板）**：
- `mc2/common/torch_extension/csrc/elastic_buffer.cpp`：在 :1894（infer `fetched`）、:1915（train `fetched`）、:1960-1961（train `gradUnique`/`uniqueLocalEntry`）各分配点之前，`aclrtGetMemInfo(ACL_HBM_MEM)` 取 free，`TORCH_CHECK(need < free×0.8)`，错误信息含 need/free/numTokens/hidden/dtype 与"减小 numTokens/hiddenSize"建议；
- **无开关，直接拦截**（拍板结论：不增加环境变量，检查始终生效）；
- 不在 elastic_buffer.py 重复实现（oom.md §4.2 的 Python 版与 §4.1 的 C++ 版二选一即可，C++ 层同时覆盖 PTA 全部调用方与训练分配点）。

**L3 op host 侧（可选，低优先级）**：tiling 无法感知空闲 HBM 且输出非其分配；可选加"numTokens×hiddenBytes ≤ 设备 HBM 总量"静态荒谬值上限。默认不做。

**不采纳/降级的 oom.md 建议**：
- §4.3 分块 fetch（`engram_fetch_chunked`）→ 列为 **P2 可选**：多轮 kernel 启动 + `torch.cat` 额外整块拷贝会使性能口径失真，**不适合 perf 用例**；仅适合功能性大 fetch 场景；
- §4.4 `dynamic=True` 共享图 → 备选评估项：动态图会改变静态 shape 的性能口径（padding/回退），默认仍逐档静态图+释放。

### 5.4 数据修正项（新发现，与 M1 关联）

1. **321125 不能被 4 整除**：sliced 模式的 `num_entries_per_rank` 是全局→owner 均匀映射的单一属性（tiling:509-510 已核实），非整除分片会导致映射错误 → 已拍板改为 `[[321128, 9216]]`（÷4=80282，总量 2.96GB 不变）；local 16x 一并改为 321128 以维持 local/sliced 同 shape 缓存配对复用。**engram_test_cases.xlsx 已更新并重跑自检通过**。
2. oom.md 历史 OOM 项处置：HCCL 端口冲突 → 已由 M6 `--master_port` 按 case 递增覆盖；ninja 编译 OOM（exit 137）与多用户 stream 耗尽（507899）→ 列入 README 运维提示（预编译、清残留进程、清 stale lock），非代码改动。

---

## 6. M6 框架正确性修复清单（随本次一并实施）

| 编号 | 修复 | 方案 |
|------|------|------|
| F1 | 失败退出码 | `__main__` 末尾 `sys.exit(0 if all_success else 1)`；runner 汇总失败用例并使批跑最终退出码非 0 |
| F2 | 父进程挂死 | `result_queue.get(timeout=600)`；`Empty` 异常时按 "crashed" 兜底标记该 rank；`proc.join()` 后检查 `exitcode`；子进程保证恰好 put 2 条消息（finally 中补齐，替代现 except 双 put 重复消息） |
| F5/M6-3 | forkserver 全局变量 | **start method 改回 `fork`**：父进程全程不初始化 NPU（设备操作全在子进程），fork 安全；子进程直接继承父进程已解析的 CLI 值与 storage（顺带消除巨型张量经 torch.mp 共享内存搬运的开销）。保留 `--start_method {fork,forkserver}` 开关兜底；fork 路径下断言 `torch.npu` 未初始化 |
| H1 | `<br>` 残留 | runner `_clean()` 增加 `.replace('<br>',' ').replace('<br/>',' ').replace('<br />',' ')`；字段内 `\|` 替换为 `;` 防 IFS 截断 |
| H5 | PROF_TMP_DIR 无效 | 二选一（建议前者）：py 读取 `PROF_TMP_DIR` 作为 profiling 输出根目录覆盖默认；或 runner 删除该机制。同时 profiling 输出绝对路径化（`$PWD/profiling_data/...`），消除 qrun cwd 依赖 |
| H6 | runner set -e 死代码 | `"${cmd_args[@]}" --prep_only || prep_ret=$?` 模式改写 prep/qrun 两处；单用例失败记入失败清单后 `continue`，不中断整批 |
| H7 | read_data 错误处理 | `if ! data_output=$(read_data); then exit 1; fi` |
| M7 | to_int 回移 | 从原版 run_engram.sh 移植健壮整数转换 |
| M8 | CPU 亲和性回移 | `set_cpu_affinity` 参数化：`--cpu_map "0-3:288-383;4-7:96-191"`（默认关闭自动探测，README 给出探测方法）；子进程内生效 |
| H3/H4 | write 校验/永真式 | 删除 egram_cpu 内永真式"Golden verification"打印；`compare_write` 默认注释保留，新增 `--verify_write` 开关启用（host 窗口指针读取比对） |
| M5 | 错误消息丢弃 | 消息统一为 `("write"|"fetch", ok, detail)` 三段式，父进程分类打印，无静默丢弃 |
| M6 | 端口冲突 | `--master_port` CLI + runner 按 case 序号传 `50008+idx` |
| 其他 | 清理 | 删除未用 import（pandas/time）；dead code（num_token_per_rank/gen_indices 旧版）删除；loop_time/repeat_time/warmup_time 增加 CLI 参数（默认 50/1/0） |

**H2（import 路径）**：改为 `from cann_ops_transformer import ElasticBuffer`（与 e2e 一致，新旧 wheel 布局均可导入）；README 首节写明依赖：须先构建安装本仓 `torch_extension`（`pip install` 本分支 wheel），并给出验证命令 `python -c "from cann_ops_transformer import ElasticBuffer"`。

---

## 7. 涉及文件与接口变化汇总

| 文件 | 变化 |
|------|------|
| examples/mc2/engram_test/engram_test.py | M1/M2/M3/M4/M5-L1/M6 全部；预计 -400/+450 行 |
| examples/mc2/engram_test/run_engram.sh | H1/H5/H6/H7/M7 + 缓存目录/端口/失败汇总；~±80 行 |
| examples/mc2/engram_test/data.md | 语义列说明 + §2.2-5 档位裁剪（方案 A/B 待拍板） |
| examples/mc2/engram_test/engram_test_cases.xlsx（已生成） | 批跑数据源：engram_tensor(6 条 OOM 分析用例) + perf_vocab(data.md 11 条) + 说明页；已通过字体/内容/runner 解析链路三重自检 |
| examples/mc2/engram_test/README.md（新增） | 依赖安装、xlsx 转换约定（`<br>`→换行）、缓存机制、CPU map 配置、qrun 共享目录约定、运维提示（ninja 预编译/残留进程/stale lock，见 oom.md 历史事件） |
| examples/mc2/engram_test/oom.md（用户补充） | OOM 实测分析，建议一并入库 |
| mc2/common/torch_extension/csrc/elastic_buffer.cpp | M5-L2 输出预算拦截（aclrtGetMemInfo + TORCH_CHECK，~+30 行，独立 commit；覆盖 :1894/:1915/:1960-1961 三个分配点） |

CLI 新增：`--shape_semantics`、`--prep_cache_dir`、`--purge_prep_cache`、`--data_pattern`、`--cpu_map`、`--master_port`、`--loop_time`、`--verify_write`、`--start_method`；`--prep_data.pt` 旧文件路径废弃。

## 8. 验证方案（每项改动的验收标准）

| 项 | 验证方法 | 通过标准 |
|----|---------|---------|
| V1 语义 | 跑 functional 用例（1024×576），打印 total/per-rank shape | total=1024、sliced 每 rank 256、local 每 rank 1024 |
| V2 生成性能 | large2 prep 计时 + `df -h /dev/shm`、`free -g` 监控 | NVMe <1min / NFS <5min；/dev/shm 占用≈0；无 OOM/挂死 |
| V3 复用 | 配对 case 第二条日志 + 与 purge 后复跑对比 | 出现 `cache hit`；fetch 档位耗时差异 ≤5%（噪声内） |
| V4 OOM | large2 全 21 档位 graph 模式 + `npu-smi` HBM 曲线 | 全档位通过或明确 SKIP 日志；HBM 无阶梯累积 |
| V5 退出码 | 人为篡改 golden 一行制造 fetch fail | runner 汇总 FAIL、退出码非 0、后续用例继续执行 |
| V6 挂死 | `kill -9` 一个 rank 子进程 | 父进程 600s 超时后标记 crash、正常退出 |
| V7 fork | ws=4 运行时子进程打印实际 world_size | 子进程内 world_size==4（非 8） |
| V8 seed | 同参数跑两次对比 storage/indices 哈希 | 两次一致；不同 seed 结果不同 |
| V9 PTA 拦截 | 构造超预算 num_tokens 调 engram_fetch | 抛 RuntimeError 且信息含 need/free/num_tokens |

## 9. 后续增强（P2，不在本次范围）

1. **grad_test.py 移植**（补齐训练链路覆盖：with_grad、engram_fetch_grad、反向 golden、train+local 拒绝负向用例——即检视报告覆盖缺口 1）；
2. analyse.py / fp8_test.py 移植；
3. data.md 增加 dtype 矩阵用例（bf16/fp16/fp32/fp8e5m2）与 memcpy 二次 write 更新可见性用例；
4. op host 侧 L3 静态上限校验（视 L1/L2 效果决定）。

## 10. 实施顺序（建议 3 个独立 commit）

1. **C1 框架修复+语义**：M1 + M4 + M6 全部 + runner 修复 + README → 功能用例回归（V1/V5/V6/V7）；
2. **C2 数据生成重构+缓存**：M2 + M3 → V2/V3/V8；
3. **C3 显存拦截**：M5-L1（用例侧）+ M5-L2（PTA，独立 commit 于 mc2/）→ V4/V9。

预估总工作量 2-3 人日。

## 11. 风险清单

| 风险 | 缓解 |
|------|------|
| fork 与 NPU 初始化冲突（若父进程未来触碰 NPU） | `--start_method forkserver` 兜底保留；fork 前断言未初始化 NPU |
| local large2 每 rank 44GiB pinned 窗口 ×4 = 176GiB，驱动 pinned 内存上限 | 先在 medium2（45GB 总窗口）验证；README 注明需检查驱动 pinned 上限参数 |
| npugraph_ex 逐档 reset 后首跑含编译开销 | warmup 已有；计时区段从 warmup 后开始（现 schedule 已如此） |
| torch_npu 无 `mem_get_info` | L2 自动降级 no-op；L1 可用 `npu-smi` 解析兜底 |
| NFS 上 mmap 写性能/一致性 | 建议 prep_cache 指向本地 NVMe；NFS 只读共享 |
| 旧 xlsx 表 per_rank 语义迁移 | `--shape_semantics` 开关 + README 迁移说明（默认 total 需你拍板） |
| data.md 16M 档位超 vocab | §2.2-5 方案 A/B 待你拍板 |

---

**决策点（已全部拍板，2026-09-21）**：
1. 修改基线采纳"当前 worktree"（§1）—— **OK**；
2. `--shape_semantics` 默认值取 `total`（§2.2-4）—— **OK**；
3. 超 vocab 档位 —— **保留 + 告警，不判失败**。结论依据（已核实）：numTokens > vocab 是真实 LLM 场景——Prefill 阶段 embedding 取数行数 = batch×seq_len，长上下文 batch≥2 即超 vocab（如 128K ctx、vocab 100M~129K 级，句子由词表组合、允许重复 token）；算子侧对 numTokens 仅校验 ≥0（tiling:188-190），训练路径的去重输出（grad_unique/unique_local_entry）正为重复 index 设计。**约束是 index 值域 ∈ [0, vocab)，而非数量 ≤ vocab**。实现：indices 有放回采样自 [0, vocab)；`num_token > vocab` 时打 WARNING"重复采样，对应 prefill 长上下文/大 batch 场景"，报告标注 repeated-sampling（§2.2-5）；
4. L2 拦截 —— **无开关直接拦截**（不增加环境变量，TORCH_CHECK 始终生效，§5.3）；
5. 321125 → 321128 —— **OK**（engram_test_cases.xlsx 已同步更新；local 16x 一并改为 321128，以维持 local/sliced 同 shape 缓存配对复用，local 模式功能不受影响）。
