# A3 EP16 model generalization

本目录保存模型规格的官方配置快照、固定版本 URL 和 SHA-256，以及确定性用例生成脚本。

当前 `rawmanual` 包已通过全部 293 项主精度和额外 15 项边界、2 项 INT8 连续变输入检查，证据位于 `results/rawmanual_precision` 和 `diagnostics/rawmanual_extra_precision.json`。rawmanual 的 64 项性能配对已完成，见 `results/rawmanual_complete_matrix`，两项 dispatch 回退仍在处理；`results/fix2_first_pass` 是仍有回退的历史候选，不能作为当前收益。最新状态见 [EP16 报告](../../../docs/A3_EP16_GENERALIZATION_REPORT.md)。

模型配置只提供 hidden size、路由专家数和 topK。每 rank token 数、BF16/FP16 输入、INT8 通信模式、掩码和偏载均为测试选择；这些用例不等同于完整模型推理，不包括专家 GEMM、模型权重量化或模型的共享专家计算。

按上级 README 准备 ep_test 的独立副本，对照 `ep_test_source_manifest.json` 中原始源码 SHA-256 后，一次应用 `ep_test_complete.patch`。该完整补丁包含当前 ABI、INT8 双标杆、cyclic 路由、有效元数据校验、结果持久化、逐 rank 校验、rank 0 库加载来源及诊断扩展；已在独立目录验证应用后与实测框架逐字节一致。不要在同一副本叠加历史分步补丁。

cyclic 路由按 rank 交错、再轮转本地专家，保证每个 token 的专家 ID 唯一，rank 与专家任务数最大相差 1。总任务数不可整除时，不宣称完全均衡。

生成命令：

    python generate_cases.py --output <output-directory>

主要 dispatch 与 round-trip 用例显式设置 `comm_alg=fullmesh_v2`，默认算法的大 B 兼容用例单独保存。不能仅通过 `.o` 文件名判断分支，因为同一二进制包含多组 tiling key。

生成 293 个精度用例（dispatch 143、combine 142、round trip 8）和 64 个性能用例。精度主体覆盖 B=1/8/32/128/256；另含 B=17/33/127/255 的 FP16、随机、偏载、掩码形状，H=1024/8192、K=1/12/16、E=16/1024 边界，以及默认通信算法下的 B=257/512。显式 fullmesh_v2 的 A3 dispatch 上限为 B=256、K=12；K16 使用默认算法单列，不能将默认算法的大 B 用例当作 fullmesh_v2 覆盖。

Mixtral-8x7B 的官方路由专家数为 8，不能满足当前 EP16 专家均分约束，保留来源但不伪造为 16 专家模型用例。模型自带共享专家也不计入 routed experts。

每个精度用例执行 10 次、在重复调用间做 rank barrier，验证最后一次输出；性能用例执行 100 次，保留每 rank 最后 50 个目标 kernel 的设备时间，汇总 16 个 rank。性能路径不包含精度 barrier。必须保留真实二进制来源和完整精度结果，缺失、超时或环境失败不能计为通过。

实现与实测结果以最终泛化报告为准；生成用例数量本身不代表完成了硬件验证。

性能结果可使用上级比较脚本，它同时支持早期矩阵与本目录的用例清单：

    python ../compare_performance.py --baseline <reference-performance-dir> --optimized <candidate-performance-dir> --cases <output-directory>/cases/cases.json --output <comparison-directory>

本轮对照使用 `hccl_per_case=0`，实际 HCCL_BUFFSIZE 统一配置为 8192 MiB；此模式下配置文件的固定值优先于用例表中继承模板的 2048 MiB 字段。实际生效值逐 rank 记录在初始化日志，覆盖 Kimi-K2 大 B 等窗口需求较高的模型。精度失败或缺少完整 16 rank profiler 数据时，比较脚本会拒绝生成收益结果。原始 fullmesh_v2 基线存在 BS 分核专家权重漏写，须先应用 `diagnostics/reference_correctness.patch` 构建正确性修复参考包；该补丁不包含计数合并、token 分核或 combine 优化。

大小规格切换诊断由 `generate_transition_cases.py --output <directory>` 生成。完整框架补丁已包含诊断支持；仅对 `transition_poison_` / `transition_delayed_` 前缀的诊断用例启用特殊数据和 rank 0 延迟，不改变主矩阵。配置 `hccl_per_case=0` 和 8192 MiB，16 项按表中顺序、每项只调用一次并验证，避免循环的后续结果掩盖首次切换错误。记录 worker 与 HCCL 名称以确认通信域复用。诊断不纳入性能统计。

变输入压力用例使用 `stress_alternating_` 前缀，在无调用间 barrier 的条件下连续运行 32 次，按正、正、负、负变换输入，使每个窗口半区复用时输入都发生变化。保存所有输出，逐次检查相对于首次输出的符号对称性，并用原 CPU 双标杆验证首次输出。该测试不计入性能收益。

额外 15 项 combine 边界用例中，最后的 H8192/K16/E1024/B256 需 8589 MiB 窗口，必须单独配置 9216 MiB 运行精度；前 14 项使用 8192 MiB。该扩展不纳入固定 8 GiB 的主性能矩阵。

隔离 OPP 必须保留 built-in 的真实目录结构。本机 CANN host loader 将目录软链接判为不存在，导致基础 tiling 库漏加载；应复制完整 built-in 与其余标准资源，单独创建空 vendors/config.ini，只通过 ASCEND_CUSTOM_OPP_PATH 指定待测私有 vendor。性能仍采用原 AllReduce/Exp 预热，未采用省去 AllReduce 的失败诊断方案。

`combine_quant_stream_stress` 另含两项 INT8 通信连续变输入检查：H2048/E512/K10，BF16/B256 和 FP16/B17；每项每 rank 连续调用 32 次，沿用相同符号对称性和首次 CPU 双标杆检查。该组不计入 293 项主精度用例，也不计入性能统计。

为满足仓库单个新增文件 500 KiB 限制，较大的 JSON 精度记录使用紧凑格式，性能详情按 16 个形状拆分。`results/fix2_first_pass/performance_details.json` 是索引，按 `parts` 顺序读取各 `path` 并连接数组可恢复全部 64 条原记录；每部分包含 SHA-256。拆分前后已逐对象核对，未删除逐 rank 数据。`rank_accuracy.csv` 的空浮点指标表示精确/布尔校验，说明见同目录 `rank_accuracy_notes.txt`。

提交版补丁增加许可证前言，并将只有一个空格的上下文行写成 Git 支持的空行；`git apply` 可直接应用。`patch_archive_manifest.json` 分别保存原始实验补丁与提交版 SHA-256、前言/尾注行数和原上下文空行位置，可逐字节恢复原补丁。各安装包清单的 `patch_sha256` 仍指原始实验补丁。已实际验证提交版 rawmanual 补丁还原当前业务源码，完整框架补丁还原全部 7 个实测文件。官方配置只规范文件结尾换行：模型清单 `sha256` 对应原始下载字节，`archived_sha256` 对应提交文件。设备观察记录以 `contiguous_repeat_device_observations.json` 的 `observations` 数组归档，逐条内容与原 JSONL 一致。
