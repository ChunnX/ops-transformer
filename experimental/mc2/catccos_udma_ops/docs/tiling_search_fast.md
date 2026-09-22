# 缩短 Tiling 搜索时间

当前搜索每个 case 复用常驻 torchrun、Gloo/SHMEM、输入、权重、路由和输出。
`batch-size` 是一次提交多少个候选，不是在同一组 NPU 上并行执行这些候选。
每个候选默认调用次数为：

```text
repeats * (align_iterations + iterations) = 3 * (3 + 3) = 18
```

256 个候选共执行 4608 次。若平均一次 26 ms，仅算调用时间就需要约 120 秒。
24,192 个候选以这个速度约需 189 分钟。因此优先减少候选数和初筛采样次数。
搜索 CLI 的采样参数覆盖 YAML 中的 performance 采样参数；不会使用 YAML 的
50 iterations。warmup 默认为每个 session 10 次。

## 推荐流程：随机初筛、邻域搜索、最终复测

下面从子项目根目录运行，使用仓库中存在的两个示例 case。客户的 case ID、EP
和 nproc-per-node 必须一起替换。单节点四卡时，case 的 world_size 也必须是 4。

```bash
set -euo pipefail
output_root="$PWD/validation_results/tiling_fast_$(date +%Y%m%d_%H%M%S)"
common=(
  --cases validation/cases/alltoallv_grouped_mat_mul.yaml
  --cases validation/cases/grouped_mat_mul_all_to_allv.yaml
  --case-id a2av_gmm_model_case_1
  --case-id a2av_gmm_model_case_2
  --nnodes 1 --nproc-per-node 8 --master-addr 127.0.0.1
  --master-port 2000 --udma-master-port 2001
  --timeout 1200 --batch-size 64
)

# 1. 广泛初筛，保留默认配置；每配置 3 次调用。
bash scripts/search_tiling.sh "${common[@]}" \
  --strategy random --seed 2026 --max-candidates 512 \
  --align-iterations 1 --iterations 2 --repeats 1 \
  --output-dir "$output_root/coarse"

# 2. 保留 Top-16 和默认配置，并探索每个 seed 的单维变化；每配置 9 次调用。
bash scripts/search_tiling.sh "${common[@]}" \
  --refine-from "$output_root/coarse" --top-k 16 --refine-neighbors \
  --strategy random --seed 2026 --max-candidates 256 \
  --align-iterations 1 --iterations 2 --repeats 3 \
  --output-dir "$output_root/local"

# 3. 只复测 Top-16 和默认配置，不扩展邻域；每配置 65 次调用。
bash scripts/search_tiling.sh "${common[@]}" \
  --refine-from "$output_root/local" --top-k 16 --max-candidates 0 \
  --align-iterations 3 --iterations 10 --repeats 5 \
  --output-dir "$output_root/final"
```

使用 `final/best_tilings.json`。每阶段每个 case 都会重新测量默认配置，阶段间
不合并计时样本。`max-candidates=0` 在最终阶段表示测完全部入围者，不会扩展成
全空间穷举。候选少于预算时自然全部测量。

| 阶段 | 每 case 候选上限 | 每候选调用数 | 调用数上限（不含 warmup） |
|---|---:|---:|---:|
| 默认穷举示例 | 24,192 | 18 | 435,456 |
| 随机初筛 | 512 | 3 | 1,536 |
| 邻域搜索 | 256 | 9 | 2,304 |
| 最终复测 | 17（16 + 默认） | 65 | 1,105 |

上述三阶段最多 4,945 次调用，约为示例穷举的 1/88；这不是实测加速比。
总时长还取决于输入准备、进程启动、不同 tiling 的耗时和设备运行状态。
初筛只有一个计时样本，噪声大时可提高到 `--repeats 2`，或提高 top-k/候选预算。
最终复测能改善入围者的排名稳定性，不能找回初筛中已被淘汰的配置。

## 利用已运行的穷举结果

停止旧搜索后，可直接将它的输出目录用于 `--refine-from`，从邻域搜索或最终
复测阶段开始。只读取已完整提交且通过校验的 `batch_*.json` 和恢复记录；不依赖
可能截断的 results.jsonl/summary.csv。只写了搜索计划的 dry-run 目录没有可用测量。
多节点时传入对应的 `node_N` 目录，各节点必须具有一致的已提交结果和搜索参数。

每个新阶段使用新目录；同一阶段中断后可加 `--resume`。精搜源的候选和分数作为
来源信息写入新 manifest，恢复时仍检查搜索计划。精搜期间及其 resume 期间不要
继续写入源搜索目录，否则新出现的更优 seed 会改变计划并被拒绝。

## 哪些空间可以缩减

当前网格已过滤编译模板、workspace、UB、worker 数和调度范围等硬约束，并做两种
硬件等价去重：AllToAllV→GMM 的 export_core_count=0 与全核数等价；GMM→AllToAllV
中产生相同完整 M 分组数的 comm_interval 等价。这类删除不损失有效配置。

初筛随机抽样和邻域搜索属于启发式缩减，不保证保留全网格最优值：

- 不要只把 `exhaustive --max-candidates` 改小。它按笛卡尔积顺序取前缀，容易只覆盖
  一个 MMAD 模板或某些早期取值；随机初筛能避免这种固定前缀偏差。
- 邻域一次改变一个搜索维度，MMAD 的 M0/N0/K0 作为整体切换；其他维度尝试网格中
  所有取值。多个优胜 seed 提供多个起点，但仍可能漏掉必须同时改变多个参数才改善的组合。
- `--space` 可指定更紧的通信块、stage_rows、copy rows、swizzle 和 worker 档位。
  在没有当前 shape 的测量证据时，不应直接固定 MMAD、workspace_stages 或 m_split。
- 邻域也受 `--space` 控制。未指定维度保留默认网格，合法默认配置和 seed 即使位于
  自定义网格外也会保留。预算不足以容纳 seed 和默认配置时会报错，避免悄悄丢掉优胜者。
- 若还有预算，可以从邻域结果再进行一轮邻域搜索，或使用不同随机 seed 的独立初筛
  检查结果是否稳定；比较不同搜索时保持 workload、软件构建、设备和拓扑一致。

GMM→AllToAllV 默认只有 2×9×7=126 个原始网格组合，合法性检查后通常更少；加上
网格外默认配置时可能增加一个。因此有约 24,000 个候选时，先检查 case 的 `operator`
和 `output_all_to_allv`，不要根据 case ID 的前缀判断算子方向。带导出的 AllToAllV→GMM
默认原始网格为 30,240 个；32 AIC 且资源约束允许全部组合时，等价去重后为 24,192 个。

## 确认实际瓶颈

各批次 checkpoint 的 `phase_timings_s` 包含每个 rank 的 enqueue、device_wait 和
measurement；session report 包含输入准备和 warmup 等耗时。enqueue 期间设备也可能
运行，驱动队列满时 enqueue 还可能等待设备，因此不能把 enqueue 全部视为 Python 开销。
先对比实际采样参数、总调用次数和 batch wall time，再决定是否需要进一步改执行路径。

把 batch 从 256 加大不会减少 kernel 调用数；小预算搜索可用 64，让进度和断点更及时，
减少失败时需要隔离重试的候选数。同一组四卡上的一个 collective 配置占用所有四个 rank，
不要把 rank 当作四个独立候选执行器。

此流程仍然是 performance-only；CPU 测试不证明 Ascend 实机性能或正确性，性能入围配置
如需部署，应使用已有的独立精度验证流程。
