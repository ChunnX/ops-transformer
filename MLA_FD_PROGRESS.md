# SMLA / MQSMLA / QSMLA FD 修复与测试进展

更新时间：2026-09-15 UTC。当前分支：`qsmla`。

## 当前结论

SMLA 正式修复已应用并通过设备验证：普通 Vec1 使用独立 2048 字节临时区，避免覆盖 Vec2 的活跃输出累加值；普通 Vec2 写回等待与 MQSMLA/QSMLA 对齐。FP16/BF16 各 320 个 tiling key 编译通过，104 次输出与 52 次返回 LSE 检查全部通过。

MQSMLA、QSMLA 的普通 FD slot 修正和三个算子的 `supportFd_ = true` 保留。MQSMLA 同组精度回归通过；QSMLA 使用实际 FD 分区 golden 后通过，原普通 golden 的分区量化差异仍存在。三算子 batch 共 48 通过、12 不适用，明细与原基线相同。

三算子共享回归覆盖选定的 8 组逻辑用例；SMLA 诊断扩展到 18 项，覆盖不均匀 head 分割和非 FD 短序列，并增加 FP16/BF16、LSE 开关组合。源码修复、诊断脚本及验证摘要作为同一组 FD 修复变更保存。

测试产物根目录：`/tmp/mla_fd_shared_20260915/`。下文未展开的测试文件路径均相对此目录；原 8 组输入、golden 脚本、安装包和日志仍保存在本机 `/tmp` 中，未纳入仓库，换机器或清理临时目录后不能直接复现。新增 7 项 SMLA 诊断的生成参数、复现脚本和结果摘要已保存到仓库，见后续定位节。

验证基线（应用本组修复前的 HEAD）：82ae4f6b84b4bec22ca521635212dbb3585a96b1。测试日期：2026-09-15 UTC。设备：Ascend950DT_9584；CANN 9.2.0。

## 已落地的源码修改

- MQSMLA、QSMLA：普通 FD 的 `s2SplitIdx` 在每个有效跨核任务进入 S2 循环前分配一次；batch 一致性仍按 reduction block 分配。避免普通 FD 循环内重复递增，与 metadata 连续 workspace slot 的含义对齐。
- SMLA、MQSMLA、QSMLA metadata：`supportFd_` 默认全部为 `true`。
- SMLA：Vec1 使用独立 `fdLseTmpUb`；普通 Vec2 使用 `StageVec2PartialOAndWait`；共享 helper 明确 32 行、512 个 float 的临时区容量并保留原偏移。
- 仓库共 7 个头文件修改。仓库原有 golden 未修改。

| 源码文件 | 变更 |
|---|---|
| [MQSMLA kernel](attention/mixed_quant_sparse_flash_mla/op_kernel/arch35/mixed_quant_sparse_flash_mla_csa_kernel.h) | 普通 FD 在 S2 循环外分配 slot；保留 batch 分支的分配时机 |
| [QSMLA kernel](attention/quant_sparse_flash_mla/op_kernel/arch35/quant_sparse_flash_mla_csa_kernel.h) | 与 MQSMLA 相同的 slot 修正 |
| [MQSMLA metadata](attention/mixed_quant_sparse_flash_mla_metadata/op_kernel_aicpu/mixed_quant_sparse_flash_mla_metadata_aicpu.h) | `supportFd_ = true` |
| [QSMLA metadata](attention/quant_sparse_flash_mla_metadata/op_kernel_aicpu/quant_sparse_flash_mla_metadata_aicpu.h) | `supportFd_ = true` |
| [SMLA metadata](attention/sparse_flash_mla_metadata/op_kernel_aicpu/sparse_flash_mla_metadata_aicpu.h) | `supportFd_ = true` |
| [SMLA vector](attention/sparse_flash_mla/op_kernel/arch35/sparse_flash_mla_csa_block_vector_arch35.h) | 普通 Vec1 独立临时区；Vec2 写回后等待 |
| [FD helper](attention/sparse_flash_mla/op_kernel/arch35/common/flash_decode.h) | 广播临时区容量常量与生命周期说明 |


## 正式 SMLA 修复与验证（2026-09-15）

### 实现和资源检查

普通 Vec1 增加 `fdLseTmpUb`，占 2048 字节；Vec2 改用共享的带等待写回 helper。
Batch 的缓冲区分配未改变。LSE 对外关闭时仍分配普通 FD 所需的 max/sum 临时区。

按当前 FA 静态布局核算，普通路径最大 UB 用量 218624 字节，距离平台
253952 字节上限还有 35328 字节；batch 最大 220672 字节，余量 33280 字节。
覆盖五种模式与 LSE 开关，FP16/BF16 的这些缓冲区元素宽度相同。

### 编译

- SMLA FP16、BF16 **各 320 个 tiling key 全部编译通过**：四种 Q/KV 布局组合、
  五种稀疏模式、split-G/head/batch/vec-S2-address 四个布尔维度。
- MQSMLA 重编 48 个、QSMLA 重编 24 个相关模板，产出二进制 SHA256 与旧包相同，
  共享 helper 的具名常量替换未改变这两组编译产物。
- 对每个二进制核对了 JSON 记录的 SHA256、模板数量、去重和 SMLA 完整 key 集合，
  并核对正式修复源码哈希。没有宣称 MQSMLA/QSMLA 已完成完整模板编译。

正式安装包：`formal_fix/install/`。构建记录和 UB 明细：
`formal_fix/build_summary.json`、`formal_fix/source_hashes.json`、
`formal_fix/ub_layout_audit.json`。全模板编译通过不等同于全模板设备运行覆盖。

提交钩子随后仅调整了 SMLA 调用的换行缩进、Python 格式和历史补丁的空白上下文。
已验证 C++ 非空白源码一致、两个 Python 脚本的 AST 一致，补丁格式化前后作用于基线产生相同源码。
`formal_results.json` 的 `source_sha256` 保留实际编译/测试版本，`formatted_source_sha256`
记录提交版本；对应核验记录在 `pre_commit_formatting_verification` 中。

### 正式包设备精度结果

| SMLA 数据类型 / 返回 LSE | 原 8 项输出 | 扩展 18 项输出 | 返回的 LSE |
|---|---:|---:|---:|
| BF16 / 开 | 8/8 | 18/18 | 26/26 |
| BF16 / 关 | 8/8 | 18/18 | 未请求 |
| FP16 / 开 | 8/8 | 18/18 | 26/26 |
| FP16 / 关 | 8/8 | 18/18 | 未请求 |

合计 **104 次输出检查、52 次返回 LSE 检查全部通过**，沿用原比较器和精度阈值。
每组扩展诊断包含 15 个实际 FD 用例和 3 个实际非 FD 用例。覆盖原 2+2 / 3+1
分区对照、单行长序列、B/Q 变化、heads=1/7/31/33/63/65/127/128，
以及 KV=1/127/128 短序列。原五种稀疏模式由原 8 项回归覆盖。

52 组 LSE 开关对照的输出按字节比较全部相同，FA/FD 任务表也相同。
metadata 张量容量为 1024 个 int32，`SmlaMetadata` 只定义前 900 项；
任务表比较仅覆盖这 900 项，不把未定义的尾部存储用于一致性判断。

### 关联算子与 batch 回归

- MQSMLA：普通 FD 输出/LSE **8/8**；QSMLA：实际 FD 分区参考输出/LSE **8/8**。
- QSMLA 原普通 golden 仍为输出 0/8、LSE 8/8，与原先分区量化差异一致；未改动 golden 或阈值。
- 三算子 batch 各 **16 通过、4 不适用**，合计 **48 通过、12 不适用**；
  每个 case/mode 的精度、输出/LSE MD5、变换一致性及不适用原因均与 seed=42 原基线完全相同。

### 性能对照

对照包是已通过精度验证的独立临时区候选（未增加 Vec2 写回等待），正式包增加等待。
在编译结束后使用同一批 BF16 输入测试，排除首次加载与扩展编译时间。
首轮逐次调用的 NPU event 计时受主机派发开销影响、波动明显，不能作为内核性能结论。
随后用 NPU 图捕获 100 次调用，预热后每组重放 5 次、共采样 5 组，按调用数折算耗时。
每轮重放后核对输出和 LSE，均通过。

| 用例 | 独立临时区候选（μs） | 正式修复（μs） | 中位数变化 |
|---|---:|---:|---:|
| `boundary_swa_256` | 15.388 | 15.273 | -0.75% |
| `boundary_swa_8192` | 23.915 | 23.944 | +0.12% |
| `load_4096_swa_tnd_pa` | 15.503 | 15.507 | +0.02% |
| `load_5120_ori_sparse_bsnd_pa` | 32.016 | 32.119 | +0.32% |
| `load_8192_hca_tnd_tnd` | 23.019 | 23.101 | +0.36% |

KV=256 的首次正式包采样波动较大，中位数高 9.26%；仅对此项增加到 100 次预热重放、
每组 20 次重放重新对照，上表使用这次复核值，其余四项使用原图重放结果。
五项最终中位数差异为 −0.75%～+0.36%，这些用例中未观察到稳定退化；
不据此宣称所有形状、数据类型均无性能影响。图重放也不等同于单个 device kernel 的独立 profiling。

原始采样保存在 `formal_fix/benchmark_{scratch,formal}.json`、
`formal_fix/benchmark_graph_{scratch,formal}.json`、
`formal_fix/benchmark_steady_{scratch,formal}.json`；脚本位于同目录。
仓库 `formal_results.json` 保留全部原始采样和最终对照，不只记录筛选后的中位数。

### 正式验证复现与产物

- 本机完整设备回归：`bash /tmp/mla_fd_shared_20260915/formal_fix/validate.sh`。
  QSMLA 已先行独立回归；其结果位于同一 `formal_fix/ordinary/qsmla` 和 `formal_fix/consistency/qsmla`。
- 可重新生成的 18 项诊断脚本、运行选项与结果说明见
  [fd_diagnosis/README.md](attention/sparse_flash_mla/tests/pytest/fd_diagnosis/README.md)。
- 仓库保存的正式数值摘要、编译 key/哈希、UB 明细、比较脚本哈希：
  [formal_results.json](attention/sparse_flash_mla/tests/pytest/fd_diagnosis/formal_results.json)。
- 原 8 项结果：`formal_fix/{bfloat16,float16}_{lse,no_lse}/smla/`；
  扩展诊断：`formal_fix/diagnostic/{bfloat16,float16}_{lse,no_lse}/`；
  batch：`formal_fix/consistency/{smla,mqsmla,qsmla}/`。
- MQSMLA 普通结果：`formal_fix/ordinary/mqsmla/`；QSMLA FD reference：
  `fd_reference/formal_fix/ordinary/qsmla/`。

## 用例和判据

三个算子使用相同的 8 项逻辑用例清单（`*_manifest.json`）、seed 42、输入范围 [-2, 2]；D=512、N2=1、sink 和 LSE 均开启。各算子保留自己的数据格式与量化方式，因此并非三个算子的输入字节完全相同。

| 用例 | B×Q | ori/cmp 长度 | N1 | Q/KV 布局 |
|---|---:|---:|---:|---|
| CSA | 2×2 | 1639/409，residual=3 | 64 | BSND/BSND |
| SWA | 2×2 | 4096/0 | 64 | TND/PA_BBND |
| HCA | 2×2 | 6554/1638，residual=2 | 64 | TND/TND |
| ORI_SPARSE | 2×2 | 5120/0 | 128 | BSND/PA_BBND |
| ORI_CMP_SPARSE | 2×2 | 3277/819，residual=1 | 128 | TND/PA_BBND |
| SWA 边界三项 | 1×1 | 256、4096、8192 / 0 | 64 | BSND/BSND |

普通测试在 metadata 和 kernel 调用处断言 deterministic level=0，并从设备返回的 metadata 解析确认每项实际进入 FD。精度使用各算子原有 `check_result`，没有放宽阈值。MQSMLA TND 的 golden LSE 是 [T,1,N]，NPU 是 [1,T,N]；普通测试只消除单例轴差异，batch runner 仅在读取 CPU golden 时将单例轴调整到公共比较器要求的位置。

## 历史普通 FD 结果（正式修复前）

| 算子/版本 | 输出通过 | LSE 通过 | 实际 FD |
|---|---:|---:|---:|
| MQSMLA 修改前 slot 逻辑 | 2/8 | 3/8 | 8/8 |
| MQSMLA 当前修改 | 8/8 | 8/8 | 8/8 |
| QSMLA 当前修改，原普通 golden | 0/8 | 8/8 | 8/8 |
| QSMLA 当前修改，FD 分区 golden | 8/8 | 8/8 | 8/8 |
| SMLA 修复前源码 | 3/8 | 8/8 | 8/8 |
| SMLA 历史独立 LSE 临时缓冲区候选 | 8/8 | 8/8 | 8/8 |

MQSMLA 前后使用同一批保存的输入；仅更换 MQ 内核 slot 逻辑。修改前通过的是 256、4096 边界输出；8192 边界的 LSE 虽通过阈值，最大绝对误差仍为 0.02039，不能等同于 LSE 完全正确。

QSMLA 的 FD 分区 golden 根据实际 metadata 的每行 FA 分区，分别执行本地 online softmax / HIF8 P 量化，再进行 FD 合并。与普通 golden 的量化分区不同会改变 P×V 输出，而 LSE 不经过 P 的 HIF8 量化，解释了输出失败、LSE 通过。新 reference 是与实际执行分区一致的精度参考，不是放宽普通 golden 的阈值；其结果也不能证明所有未覆盖输入均正确。实现见 `fd_reference.py`。

SMLA 修复前的三个单行边界输出通过，五个多行用例失败；最大绝对误差分别为 CSA 1.59375、ORI_CMP 6.0625、SWA 3.2578125、ORI_SPARSE 5.5390625、HCA 4.63671875。按实际 FD 分区重算 golden 后仍为输出 3/8、LSE 8/8，不能用 golden 分区差异解释。

## 原始 Batch 一致性基线

5 个多行用例 × reorder、split、token-split、shape-change 四种模式；actual deterministic level=3。每个算子 16 项通过、4 项不适用，共 48 通过、12 不适用。通过项同时满足原 batch golden 精度和变换前后输出/LSE 逐 bit 相同。

各算子的 4 个不适用项相同：CSA token-split 和 shape-change 改变 ori mask 窗口；SWA token-split 改变窗口；SWA shape-change 需要 33 个 PA block、输入只映射 32 个。原 runner 用 InvalidTransformError 记录这些拒绝，本报告将其列为不适用，不当作通过。

## 历史构建与复现边界（定位阶段）

定位阶段成功编译了覆盖这 8 项测试所需的模板组合及 batch 开关分支，未完成全模板矩阵编译。模板筛选仅发生在 build 生成目录，源码模板声明没有改动。

有效测试使用 `/tmp/mla_fd_shared_20260915/install` 内的独立包。增量打包原先残留仅有 QSMLA 的注册配置，已依据三算子的实际二进制重新生成该独立包的 binary_info / 每算子配置及 ops-info，并重新完成有效测试。build/build_out 的初次 .run 不作为本报告的可复现包。

早期缺少注册配置的 SMLA 尝试、MQ LSE 布局适配失败的尝试均单独保存为 `*_attempt`，不计入有效数值结果。

复现普通 FD：`bash /tmp/mla_fd_shared_20260915/run.sh mqsmla`（将参数改为 smla / qsmla）。
复现 batch：同一命令增加 `--consistency`。
修改前 MQ 对照：`MLA_SHARED_INSTALL=/tmp/mla_fd_shared_20260915/install_before bash /tmp/mla_fd_shared_20260915/run.sh mqsmla --tag before`。

证据：`fixed/*/summary.json`、`before/mqsmla/summary.json`、`fd_reference/fixed/*/summary.json`、`consistency/*/summary.json`；完整输入和输出保存在相应目录的 .pt 中。

## SMLA 额外定位实验：增加等待未解决

静态代码发现普通 FD 的 `StageLegacyVec1Lse` 使用 `stage2OutBufs.tensor` 作为 max/sum 广播临时缓冲区，而普通 Vec2 使用 `StageVec2PartialO` 写回后没有立即等待。为了验证“增加写回等待即可解决”这一假设，仅在 build 生成头文件中将该调用替换为 `StageVec2PartialOAndWait`，编译到独立 `install_smla_wait`；候选 diff 保存为 `smla_wait_candidate.patch`。

同一组 8 个输入实测仍为输出 **3/8**、LSE **8/8**，五个失败用例的最大误差未降低。这项实验未验证 DMA 等待缺失是根因，候选不是有效修复，未应用到仓库。该实验结束时尚未定位根因，后续定位结论见下一节。

证据：`smla_wait/smla/summary.json`、`run_smla_wait.log`、`build_smla_wait.log`。候选复现：`MLA_SHARED_INSTALL=/tmp/mla_fd_shared_20260915/install_smla_wait bash /tmp/mla_fd_shared_20260915/run.sh smla --tag smla_wait`。

前一轮结束时，完整模板头文件已恢复到 build 生成目录，相关源码复制/内核编译标记和旧注册配置输出已失效，后续构建会重新生成。`git diff --check` 通过；`source_hashes.json` 与最终 5 个源码修改哈希一致。`provenance.json` 保存 HEAD、源码与各独立安装包二进制哈希，`build_cleanup.json` 保存清理明细。未提交或推送。

## SMLA 后续定位：Vec1 覆盖活跃的 Vec2 累加值

已用原包重新复现输出 3/8、LSE 8/8（`locate_baseline/smla/summary.json`）。错误发生在 FD 合并前的 partial O 生成阶段。

- `StageLegacyVec1Lse` 用 `stage2OutBufs.tensor` 广播 max/sum；`ProcessVec2` 用同一缓冲区跨 S2 块累加输出。
- 流水线先运行下一块 Vec1，再运行上一块 Vec2。某核处理三块时，最后一块 Vec1 的广播会覆盖第 0 块的累加值，随后第 1 块 Vec2 读取了被覆盖的数据。
- 本组每个向量核处理 32 行，广播覆盖缓冲区前 2048 字节；D=512 时恰好是一行 FP32 输出。N1=64 的失败诊断仅 head 0、32 有大误差，与该地址范围吻合。
- 增加 DMA 等待无法保存被覆盖的活跃累加值，因此之前的等待候选无效。

缩小对照使用 BF16、BSND/BSND、SWA 全窗口、N1=64、N2=1、D=512，开启 sink/LSE，seed=42、deterministic level=0。

| 对照 | 原包输出最大误差 / 状态 | 独立临时区候选 |
|---|---|---|
| 同一份 B=1、Q=1、KV=512 输入，人工 FA 分区 2+2 块 | 0.03125 / Pass | Pass |
| 同一份输入，人工 FA 分区 3+1 块 | 1.625 / Failed | Pass |
| 默认 metadata，B=1、Q=1、KV=8192 | 0.03125 / Pass | Pass |
| 默认 metadata，B=1、Q=1、KV=12288 | 2.40625 / Failed | Pass |
| 默认 metadata，B×Q=1×2、KV=4096 | 0.03125 / Pass | Pass |
| 默认 metadata，B×Q=2×1、KV=4096 | 0.03125 / Pass | Pass |
| 默认 metadata，B×Q=2×2、KV=4096 | 3.2578125 / Failed | Pass |

以上两版 LSE 均为 7/7。人工分区只用于隔离每核工作量，完整无重叠地覆盖输入；单行默认 metadata 也能复现，说明多行不是必要条件。

CPU 按源码顺序模拟覆盖后，与原包的 3+1 错误输出比较达到 100% 通过，最大差异 0.03125、平均差异约 1.43e-6。此预测只验证错误机制，不是正确性 golden。

候选在 build 生成头文件中新增普通路径独立临时区，编译到 `install_smla_scratch/`。原 8 项输出均恢复通过，五个原失败用例的最大误差均降至 0.03125。普通包、原输入和原 golden 均未改变；生成头文件和完整模板声明已恢复。仍只编译选定 BF16 模板及 batch/vec-S2-address 分支，未做全模板构建。

候选另完成 SMLA batch 一致性回归：显式 seed=42、actual deterministic level=3，16 项通过、4 项不适用；完整精度与逐 bit 比较明细与原基线一致。首轮候选 batch 脚本未显式设置变换 seed，已修正环境并重跑，最终结果取重跑产物 `smla_scratch_consistency/smla/summary.json`。

仓库已保存[独立诊断说明、参数、脚本和候选补丁](attention/sparse_flash_mla/tests/pytest/fd_diagnosis/README.md)，七项诊断输入可重新生成；[结果摘要](attention/sparse_flash_mla/tests/pytest/fd_diagnosis/results.json)包含源码/二进制哈希。设备结果及原 8 项完整输入仍保存在 `/tmp`：`smla_scratch/smla/`、`smla_diagnosis_portable/{baseline,scratch}/`；候选版本证据见 `smla_diagnosis_portable/provenance.json`。这些是正式修复前的定位记录，SMLA 内核现已修改。


## 复现命令与产物索引

以下命令在当前机器、现有 CANN 和 `/tmp` 产物仍可用的前提下执行；重跑会更新对应结果目录。

```bash
# 历史基线普通 FD，默认使用正式 SMLA 修复前的独立安装包
bash /tmp/mla_fd_shared_20260915/run.sh smla
bash /tmp/mla_fd_shared_20260915/run.sh mqsmla
bash /tmp/mla_fd_shared_20260915/run.sh qsmla

# Batch 一致性，其他算子替换最后的算子参数
bash /tmp/mla_fd_shared_20260915/run.sh mqsmla --consistency

# 根据已保存的 metadata / 输入重算 FD 分区 golden
python3 /tmp/mla_fd_shared_20260915/fd_reference.py qsmla --tag fixed
python3 /tmp/mla_fd_shared_20260915/fd_reference.py smla --tag fixed
```

| 产物（相对测试根目录） | 内容 |
|---|---|
| `generate.py`、`*_manifest.json`、`inputs/` | 共享逻辑用例定义和各算子输入 |
| `prepare_batch.py`、`batch_inputs/` | Batch 一致性输入与原 batch golden |
| `run.sh`、`run.py` | 环境设置、执行及精度比较入口 |
| `fd_reference.py` | 按实际 FA 分区生成普通 FD 精度参考 |
| `fixed/{smla,mqsmla,qsmla}/summary.json` | 正式 SMLA 修复前的普通 FD 结果 |
| `before/mqsmla/summary.json` | MQSMLA 修改前对照结果 |
| `fd_reference/fixed/{smla,qsmla}/summary.json` | FD 分区 golden 对照结果 |
| `consistency/{smla,mqsmla,qsmla}/summary.json` | Batch 一致性各模式明细 |
| `smla_wait/smla/summary.json`、`smla_wait_candidate.patch` | 未解决问题的 SMLA 等待候选及实测结果 |
| `install/`、`install_before/`、`install_smla_wait/` | 历史基线、MQ 对照、SMLA 等待候选独立安装包 |
| `source_hashes.json`、`provenance.json`、`build_cleanup.json` | 源码/二进制版本证据和构建目录恢复记录 |

## 后续待办

1. SMLA 正式修复、UB 审核及扩展诊断已经落地；后续将本机原 8 组共享回归脚本与输入生成流程整理进仓库测试。
2. 整理 QSMLA 的 FD 分区 golden，使普通 FD 与 batch 一致性参考的适用范围明确；当前实现和输入仍在 `/tmp`，尚未集成到仓库测试。
3. 为四个不适用的 batch 变换构造语义保持、PA 容量足够的补充用例，补齐这部分覆盖。
4. 补充 MQSMLA/QSMLA 的完整模板构建及三算子更广泛的设备运行覆盖，再整理可提交的源码与测试产物。
