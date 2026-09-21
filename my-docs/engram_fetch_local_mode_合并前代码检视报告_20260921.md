# engram-fetch-local-mode 分支合并前代码检视报告

- 检视日期：2026-09-21（v2 修订）
- 检视基线：**origin/master（86f856ca4，09-18 19:06）**。`git merge-base origin/master HEAD = 86f856ca4`，即本分支相对 origin/master 是干净的 fast-forward（领先 22 个 commit），PR 实际增量 = **33 文件，+3501/-277**。
- **基线勘误（v1 → v2）**：v1 误用本地 `master`（ded611f93，与本分支分叉于 33273bb9d/09-16）作对比基线，并把分支自 09-16 以来的累计演进（含已进上游的内容）当作了本 PR 增量；v1 报告中"分支携带 744 文件/约 7 万行夹带"的警告对 origin/master **不成立**，本节予以更正。commit 提交信息仅用于圈定 engram 相关范围与佐证演进意图（如 grad local 分支先加后删），功能结论均以最终代码为准。

## 一、改动概览（相对 origin/master 的实际增量）

| 模块 | 内容 |
| --- | --- |
| engram_fetch 正向 | 新增 `fetch_mode=1` 本地全量表模式（仅推理）：`engram_fetch_mode_def.h`（新）、`EngramFetchInferBase` 推理公共基类（新，自 arch35.h 抽取）、`EngramFetchLocalArch35` 本地 kernel（新）、tiling 增加 fetch_mode 校验与 LOCAL_MODE tiling key、op_api 增加第 18 个入参 fetch_mode 及校验 |
| PTA（torch_extension） | `ElasticBuffer(fetch_mode=...)` 及 torch op schema 增参；local 模式校验（sf 行数、INT32_MAX 上限按全量表）、`fetch_wait` 短路、context 构建跳过跨卡通道获取（不依赖 RDMA） |
| engram_fetch_grad | **仅注释清理**（本 PR 增量）；SortLib/A2A 状态机/unique 向量化等主体已在 origin/master（见下） |
| 示例 | 新增 `engram_fetch_e2e`（端到端验证，质量高）与 `engram_test`（个人压测脚本，问题多，另含 xlsx/review/oom 文档） |
| UT | op_api UT 补 fetch_mode 矩阵；tiling UT 未同步 |

**注（已在上游、不属于本 PR 增量，v1 概览曾误并入）**：grad 侧 SortLib 引入（2241e2418）、A2A 状态机重构、unique 向量化 + chunk 列切分、训练正向 staging + `WriteWithNotifyNbi` 重写（b1fe27905）、推理同步修复（1e878a0cd）、通信域复用（7a3e6f73a）等均已包含在 origin/master（86f856ca4 的祖先）中。

## 二、注释整改（本次已直接修改并提交）

本轮已按"删除无用注释、关键注释改简洁英文"原则完成清理，共涉及 21 个文件（v2 追加 PTA 侧 2 个文件、5 处）：

1. **正向算子新增中文注释全部英译**：`aclnn_engram_fetch.cpp`、`engram_fetch_mode_def.h`、`engram_fetch_tiling.cpp`（2 处）、`engram_fetch_tiling_key.h`、`engram_fetch_utils.h`、`engram_fetch_infer_base_arch35.h`、`engram_fetch_local_arch35.h`、`engram_fetch_arch35.h` 文件头。
2. **UT 注释**：`test_aclnn_engram_fetch.cpp` 4 处中文注释英译，并修正 1 处错误描述（`training_local_rejected` 用例实际由 op_api `CheckFetchMode` 拒绝，原注释误写为"被 tiling 拒绝"）。
3. **Grad/SortLib**：
   - `engram_fetch_grad_tiling_data.h` / `engram_fetch_grad_arch35.h` / `engram_fetch_grad_unique.h` 新增字段注释英译；补 `chunkElems` 字段注释。
   - `sort_lib.h` 大块 doxygen 重写为英文，**修正 3 处事实错误**：排版错乱（`[kernel]` 并入上一行）、引用本仓库不存在的 host API（`SortTilingCompute/IsInt32Safe/DCACHE_SIZE`）、改为指向 `engram_fetch_grad_tiling.cpp` 实际实现。
   - `sort_lib_core.h` 六段 workspace 布局契约注释：去掉 emoji、修正指向不存在文件 `op_host/arch35/sort_lib_tiling.h` 的错误引用、六段说明英译；删除 `ClearWorkSpace() // MUST be before InitBuffer` 与实际调用位置矛盾的错误注释、删除 `ParserTilingData` 中 5 条复述字段名的注释。
   - `sort_lib_one_core.h` / `sort_lib_params.h` / `sort_lib_util.h` 中文注释英译，删除复述性注释。
   - `engram_fetch_grad_arch35.h` Timeout 枚举 2 处注释更新为现存函数名（原注释引用已被本 PR 删除的 `SendGradRemote/RecvGradFromPeers`）。
4. **示例脚本**：
   - `engram_test.py`：删除 3 段注释掉的死代码（旧 `gen_balanced_indices` 约 24 行、dummy matmul 预热、引用私有 API 的 `compare_write` 调用）及 1 处残缺旁白注释。
   - `run_engram.sh`：删除"新增 N 个"类 changelog 注释（4 处）。
   - `run_example.sh`：删除 1 处自明注释。
   - `engram_fetch_e2e/README.md`：修正 2 处与实现不符的示例（`--fetch-mode local --phase train` 改为 `--phase infer` 并加说明；SUMMARY 输出措辞与脚本对齐）。
5. **PTA（v2 补充）**：`elastic_buffer.py`（3 处）与 `csrc/elastic_buffer.cpp`（4 处）PR 新增中文注释英译。

未动的部分：master 上既有文件的中文注释（如 `engram_fetch_arch35.h:140` 的 ScatterByRank UB 预算注释、grad tiling cpp 既有中文注释）保持原样，不在本 PR 整改范围。

## 三、功能完善性检视

### 3.1 engram_fetch 正向（本 PR 增量，总体可用，无阻塞问题）

| # | 位置 | 问题 | 级别 |
| --- | --- | --- | --- |
| F-1 | （已上游）`engram_fetch_tiling.cpp` SetWorkSpace 训练分支 | `wsTokenStaging = totalRecv × hiddenBytes` 随规模线性膨胀（如 totalRecv=32k、hiddenBytes=8KB 时达 256MB）。staging 按"全局 recv 行号"一次性铺满，避免写完成追踪但 workspace 与输出同量级；建议评估按 chunk 环形复用 | 建议（上游跟进） |
| F-2 | （已上游）`engram_fetch_train_arch35.h` LocalReadTableAndSendRemote | 每 chunk 一次 `WriteWithNotifyNbi`（数据+notify 一次下发），替代旧 notifyGran 合并；理论净简化，建议跑多卡训练确认性能无回退 | 建议（上游跟进） |
| F-3 | `engram_fetch_local_arch35.h` GatherRange | `indicesLocal.GetValue(i)` 逐元素标量读 + 每行一次 `LocalCopySlice`（GM→UB→GM 中转）；行数大时标量开销占比高，可考虑向量化收集行基址后批量 DataCopy（参考 train 侧 `GatherRowsToStaging` 的按 tile 聚合写法） | 建议 |
| F-4 | `engram_fetch_local_arch35.h` GatherRange | `entry` 越界（<0 或 ≥numEntriesPerRank_）静默跳过，`fetched` 对应行保持未初始化；与 sliced 模式行为一致性需确认（sliced 的 LocalFetchTokens 同样不校验）。若框架契约是"indices 全部合法"，建议在 tiling 或 op_api 侧加一次显式说明 | 次要 |
| F-5 | `engram_fetch_tiling.cpp` 推理分支 | 推理（含 local）workspace 仍为固定 `SYSTEM_NEED_WORKSPACE`，local 模式实际不使用 workspace（kernel 侧 `(void)workspaceGM`），一致 ✓ | 确认项 |
| F-6 | workspace/kernel 布局对齐 | `wsIndicesReadyFlag`（host `AlignTo`）与 kernel `Ceil(...)×UB_ALIGN` 对齐方式一致；staging 段 kernel 侧 `stagingRows_ = totalRecv` 与 host 公式一致 ✓；`GatherRowsToStaging` 中 staging 行号 = `rdispl+myStart+totalSent` 全局唯一、跨核不重叠 ✓ | 确认项 |
| F-7 | `engram_fetch.cpp` 入口 | LOCAL_MODE 模板分支参数序（`fetched, sfTable, fetchedSf, ...`）与 `EngramFetchLocalArch35::Init` 签名一致 ✓ | 确认项 |

### 3.2 PTA torch_extension（本 PR 增量，v2 补充检视）

| # | 位置 | 问题 | 级别 |
| --- | --- | --- | --- |
| P-1 | `elastic_buffer.py` / `csrc/elastic_buffer.cpp` | fetch_mode 值契约三处同步（Python / csrc 常量 / op 侧 mode_def.h）已用注释锚定 ✓；`ElasticBuffer.__init__` 对 fetch_mode 类型、取值、与 with_grad 互斥做了校验 ✓；local 模式下 sf 行数校验、`num_entries ≤ INT32_MAX`（按全量表）分支正确 ✓ | 确认项 |
| P-2 | `elastic_buffer.cpp` Build | local 模式跳过 `GetHcclCommResource`/通道获取，仅记录本 rank 表基址 `virtualAddrList[rankId]`——与 kernel 侧 `commBuffer[rankId]` 取基址一致 ✓；注意 local 模式仍创建了 HCCL communicator（仅不取通道），单卡/无 RDMA 环境部署时 communicator 建立本身的依赖需在文档说明 | 次要 |
| P-3 | `elastic_buffer.py` `_wait` | local 模式短路 `engram_fetch_wait`（无在途 RDMA 读，kernel 完成即数据可见）✓；aclgraph 场景下该短路是否影响图捕获一致性建议补一条 e2e 用例（`--is_graph` 路径目前只在 engram_test 覆盖） | 建议 |
| P-4 | 注释 | 本轮已将 Python/csrc 侧 PR 新增中文注释英译（5 处，随 v2 commit） | 已处理 |

### 3.3 engram_fetch_grad（主体已上游，本 PR 仅注释清理；以下为上游跟进项，不构成本 PR 合并门槛）

| # | 位置 | 问题 | 级别 |
| --- | --- | --- | --- |
| G-1 | tiling `SetTilingData`（kSortLibFixedBytes/kBytesPerElem=21 等）vs kernel `InitUbBuffers` | **host/kernel UB 校验缝隙**：kernel 要求 `uniqueBufSize ≥ max(scatterPeak, COMM_BUF_BYTES=128KB)`，host 只校验 `minUniqueNeed/chunkMinNeed`（最小约 86KB）。`availableForPool ∈ [~86KB, 128KB)` 时 host 放行、kernel `ascendc_assert` 陷阱（release 裁剪断言后为 UB 越界）。目标平台 UB 充裕不触发，但缝隙真实存在，host 侧应补齐 128KB 下限校验 | 强烈建议（上游跟进） |
| G-2 | `Process()`（原 `ZeroGradUnique` 调用被删） | **输出语义变更**：`gradUniqueOut` 中未被本批写到的行从"确定性 0"变为未初始化内存；`ZeroGradUnique`（unique.h:52/316）成为死代码。若下游契约是"仅 [0, numUnique) 行有效"则无碍，但需确认契约，并删除或恢复死函数 | 强烈建议（上游跟进） |
| G-3 | tiling 六段 workspace 公式（256LL/4LL/2LL 裸魔数）与 sortlib kernel 六段布局 | 两处手工镜像、无常量锚点；本应承担契约说明的注释还指向不存在的文件（本次已修正注释）。建议把六段布局提升为 host/kernel 共享 constexpr（如 `sort_lib_params.h` 增加 `SortWorkspaceBytes()`），双侧同源 | 强烈建议 |
| G-4 | `sort_lib_core.h` LookbackGlobal（while(true) 轮询） | 无超时/看门狗；旧实现所有自旋均有 60s TimeoutCheck。上游 tile 状态异常时从"可诊断 trap"退化为"静默挂死" | 强烈建议 |
| G-5 | 排序轮数 | 旧实现按 entry 值域自适应 1~3 轮，新 SortLib 恒 4 轮（int32 全值域）；典型 EP 配置（numEntriesPerRank ≤ 64K）排序工作量 ×4/3~×4，需性能数据确认或补回轮数裁剪（tiling 侧已知 numEntriesPerRank） | 强烈建议 |
| G-6 | `p.isSingleCore = 0` 恒置 0（arch35.h:1392） | SortLib 单核快路径（127 行）整条为死代码，小 numRecv 场景（≤8192）从 0 SyncAll 退化为 4 轮×2 次全核 SyncAll；旧 `ProcessSingleCore` 能力丢失 | 次要（性能） |
| G-7 | A2aDriveLoop 超时只在全局零进展时触发 | 个别 send unit 被对端 credit 饿死而其他 unit 有进展时，看门狗不触发；旧实现每个等待点独立 TimeoutCheck，覆盖面回退 | 次要 |
| G-8 | unique.h entryBuf 槽位别名 | `AccumListUb()` 与 `DirectFlagUb()` 同址重叠、`Run/RunLen` 复用旧 RECV_INDEX 槽，仅靠调用图互斥保证正确，L97 布局注释未更新；建议补别名说明或静态断言 | 次要 |
| G-9 | 溢出与守卫 | `RADIX_SORT_NUM×tileId` 三处 uint32 乘法理论上 numRecv > ~1.07G 时回绕（理论）；`p.tileCount` 除法无 `numTileData==0` 守卫（依赖 host/kernel 同包）；`recvIdx = aivId_-numSendCores_` 单核时下溢（后续比较兜底）。均低概率，建议顺手加断言 | 次要 |
| G-10 | 死代码 | V1 packed 分支不可达、`A2aChunkSendStep` 第二个 `PrefetchCreditCounter` 死分支、`flushPending_` 恒假等待分支、`engram_fetch_grad_utils.h` 三个死常量（SORT_TMP_*）、`ZeroGradUnique`（见 G-2） | 次要 |
| G-11 | `SetScheduleMode(1)` | 按库文档要求启用（与 moe_ep_dispatch 先例一致）✓；但库文档警告多流/多算子并发不独占核可能死锁，部署侧需确认无同流并发 | 确认项 |
| G-12 | chunk 列切分模式 | hiddenBytes > 64KB 从"直接报错"变为"chunk 回落支持"，host `fullRowFits` 与 kernel `chunkElems_` 分支对应正确 ✓；但 `ChunkColumnPass` 每 chunk 重复全量 entry 标量走查 + companion GM 拷贝，numChunks 大时冗余放大 | 确认项+性能建议 |

### 3.4 示例

| # | 位置 | 问题 | 级别 |
| --- | --- | --- | --- |
| E-1 | `engram_test.py:60` | **硬编码内网 IP `141.61.94.223`**，不应入库，改环境变量/CLI 传入 | **阻塞** |
| E-2 | `run_engram.sh:278` | **硬编码排队工具 `qrun --emp q00983190`（个人工号）**，需参数化并在缺失时本地执行 | **阻塞** |
| E-3 | `run_engram.sh` | `set -euo pipefail` 与"失败继续跑下一条"逻辑冲突：`:186` 的 `if [[ $? -ne 0 ]]` 永不执行，`:268/:278` 任一失败直接终止整批；应改 `if ! cmd; then` 形式 | **阻塞** |
| E-4 | `engram_test` 目录 | 无版权头、无 README、未被 `examples/README.md` 索引、依赖 `data.md` 之外的 xlsx 输入；与 e2e 示例功能重叠。建议：补齐后再合入，或本 PR 不合入该目录 | **阻塞（流程）** |
| E-5 | `engram_test.py` | 恒假条件（:59 `0 if server_num > 1 else 0`、:879 len 恒等判断且报错不退出）、:809-810 异常分支连续两次 `queue.put`（挤占消息槽位）、:805 `print(f"traceback.print_exc()")` 调试残留、裸 except、未用导入（List/pandas/time）、未用全局与函数（num_allocated_qps、gen_indices、compare_write 等）、int8 默认范围 `-128..128` 会触发 numpy ValueError、aclgraph×profiling 四分支约 90 行×4 重复、私有深层导入路径 | 强烈建议 |
| E-6 | `run_engram.sh` | `python` 应为 `python3`/`PYTHON` 变量；脚本依赖 CWD 相对路径（应学 run_example.sh 用 SCRIPT_DIR）；heredoc 内插文件路径进 Python 源码（应经环境变量）；`seed=0` 被当作"未设置"落回默认 1 | 建议 |
| E-7 | `engram_fetch_e2e` | 整体质量高可合入；本次已修 README 两处不一致（L105 local+train 示例、L162 SUMMARY 输出）；另 `run_example.sh:20` 无 `set -e` 属有意（失败收集后继续），建议加一行说明 | 已处理/建议 |
| E-8 | 生成产物 | `result.csv`、`*_prep_data.pt`、`profiling_data/` 等落在仓库工作区且未被 .gitignore 覆盖 | 建议 |

## 四、UT 覆盖评估

| 测试面 | 现状 | 结论 |
| --- | --- | --- |
| 正向 op_api UT（test_aclnn_engram_fetch.cpp） | 新增 5 个用例：local 成功、sf 链路、local+sf、train+local 拒绝、非法 fetch_mode 拒绝；原有 nullptr/训练用例已同步新签名；dtype 矩阵覆盖 BF16/FP8-E4M3/E8M0 | **覆盖良好** |
| 正向 tiling UT（test_engram_fetch_tiling.cpp） | **零 fetch_mode 用例**：无 local 模式 tilingKey 断言（`GET_TPL_TILING_KEY(ENGRAM_FETCH_LOCAL_MODE)`）、无 train+local tiling 拒绝、无非法 fetch_mode tiling 拒绝；param 结构也未加 fetchMode 字段 | **缺口，建议补** |
| Grad op_api UT | 已随"local 仅推理"重构同步（42 增 5 删） | 覆盖良好 |
| Grad tiling UT | 19 个既有用例未覆盖 sortlib 新增 4 个 tiling 字段与六段 workspace 公式（该缺口随 sortlib 已存在于上游）；`SetScheduleMode(1)` 是否被 UT 框架 mock 未知，需实际跑通 | 缺口（上游跟进，本 PR 建议顺带跑通） |
| Kernel 侧 | 两侧均无 op_kernel UT（仓库内 fused_gdn_gating 等有先例）；kernel 行为主要靠 e2e 示例覆盖 | 缺口，可跟进 |
| 建议 | 补：① forward tiling local/training-local/invalid-fetch_mode 用例（fetch_mode 为本 PR 引入，属本 PR 责任）；② grad tiling 对六段 workspace 与 kernel 段偏移一致性的断言（防 G-3 再犯）；③ sortlib `numRecv ≤ totalRecv`、`numTileData > 0` 断言 | — |

## 五、合并前 Checklist（按优先级，v2 修订）

1. **阻塞（本 PR）**：处理 E-1/E-2（内网 IP、工号出库）、E-3（set -e 失效）、E-4（engram_test 目录去留）。
2. **必须（本 PR）**：实际跑通 engram_fetch / engram_fetch_grad 全部 tiling UT 与 op_api UT（尤其确认 SetScheduleMode mock）；补 forward tiling 的 fetch_mode 用例（UT 缺口①）。
3. **强烈建议（上游跟进，不阻塞本 PR）**：G-1（host 补 128KB UB 下限校验）、G-2（确认 gradUniqueOut 未初始化行的下游契约并处理 ZeroGradUnique 死代码）、G-3（workspace 公式共享常量化）、G-4（lookback 看门狗）、G-5（排序轮数性能确认）、F-1/F-2（staging 内存特征与训练性能回归）。
4. **建议**：F-3/F-4（local 模式向量化、越界语义说明）、P-2/P-3（PTA 部署依赖与 aclgraph 用例）、G-6~G-10（死代码与守卫）、E-5/E-6/E-8（engram_test 清理）。
5. 注释整改已随 commit 完成（见第二节清单；v2 追加 PTA 侧 5 处英译）。
6. ~~确认分支携带的 700+ 非 engram 文件是否经其他 PR进 master~~（v1 误判，已撤销：相对 origin/master 本分支为纯 fast-forward，PR 增量仅 33 文件）。
