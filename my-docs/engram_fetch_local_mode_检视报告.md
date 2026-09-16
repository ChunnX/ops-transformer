# engram_fetch local 模式（fetch_mode=1）分支检视报告

- **检视对象**：分支 `engram-fetch-local-mode`（基线 `ca0aebf45` → HEAD `581882847`，共 2 个提交）
- **检视范围**：
  - 设计文档 `my-docs/engram_fetch_设计文档.md`（352 行）
  - 代码变更 17 个文件（+990 / -113）：op_host / op_tiling / op_kernel / op_api / PTA(torch_extension) / UT
- **检视与验证日期**：2026-09-16
- **总体结论**：✅ **通过（附 1 项上游既有问题移交、4 项低风险改进建议）**。arch35 (ascend950) 编译验证通过、kernel 实例 4+2 全覆盖、op_api UT 15/15 通过、代码实现与设计文档一致、默认行为（fetch_mode=0）后向兼容无回归风险。

---

## 1. 编译验证（arch35 / ascend950）

### 1.1 环境

| 项 | 值 |
|---|---|
| CANN | 9.2.0（`~/Ascend/cann-9.2.0`，ASC_DEVKIT/HCOMM 9.2） |
| 编译器 | bisheng（CANN 自带）+ ccache |
| 硬件 | A3 (ascend910_93)——仅编译，无 950 运行环境 |
| 工作区 | `/mnt/workspace/gitCode/cann/ops-transformer` |

### 1.2 编译命令（可复现）

```bash
cd /mnt/workspace/gitCode/cann/ops-transformer
bash build.sh --pkg --soc=ascend950 --module=mc2 \
    --ops=engram_fetch,engram_fetch_grad,engram_fetch_wait -j16
```

> 环境备注：首次构建因 `third_party/ops-tensor` 子模块 `asc-devkit`（`include/tensor_api`）网络克隆 RPC 超时失败，属网络环境问题、非代码问题。解决方法（已执行，后续构建可直接复现）：
> ```bash
> cd third_party/ops-tensor && git submodule update --init --recursive --depth=1
> ```
> 与设计文档 §11.4 的编译环境备注不同：本次验证在 **CANN 9.2.0 原生环境**下完成，**无需 asc-comm 开发补丁**（`MakeBatchHandle/BatchCommit` 等批量 Hcomm API 已进入 9.2.0 主线）。建议同步更新设计文档 §11.1/§11.4 的环境描述。

### 1.3 编译结果

| 项 | 结果 |
|---|---|
| 退出状态 | ✅ exit=0 |
| 产物 | ✅ `build_out/cann-ops-transformer-custom_linux-aarch64.run` |
| 包内 kernel 二进制 | ✅ 19 个 `.o`（engram_fetch 5 / engram_fetch_grad 9 / engram_fetch_wait 5） |
| op_host/op_api 库 | ✅ 编译链接通过（含 fetch_mode attr 与新 aclnn 签名） |

### 1.4 kernel 实例覆盖核验（与设计 §7.1/§11.2 对照）

| 算子 | dtype 覆盖 | 每 dtype kernel 实例 | 核验方法 |
|---|---|---|---|
| EngramFetch | 5（bf16/fp16/fp32/fp8e4m3/fp8e5m2） | 4：`_0` DEFAULT / `_1` TRAIN / **`_2` LOCAL / `__3` TRAIN_LOCAL**（新增） | 解析 `build/binary/ascend950/bin/engram_fetch/EngramFetch_*.json` 的 `kernelList` |
| EngramFetchGrad | 3（bf16/fp16/fp32，fp8 无 grad 符合设计） | 2：`_0` DEFAULT / **`_1` LOCAL**（新增） | 解析 `EngramFetchGrad_*.json` |
| EngramFetchWait | 5 | 1（wait 与 fetch_mode 无关） | 同上 |

✅ 与设计文档一致：tiling-key 派生的 4 个 fetch 内核实例 + 2 个 grad 内核实例全部生成。

---

## 2. UT 验证（op_api）

### 2.1 命令（可复现）

```bash
bash build.sh --opapi_test --noexec --soc=ascend950 --module=mc2 \
    --ops=engram_fetch,engram_fetch_grad -j16
export LD_LIBRARY_PATH=$PWD/build:$LD_LIBRARY_PATH
./build/tests/ut/framework_normal/op_api/transformer_op_api_ut --gtest_filter='*Engram*'
```

### 2.2 结果：✅ 15 / 15 全部通过

| 套件 | 用例数 | 结果 |
|---|---|---|
| AclnnEngramFetchTest | 9 | 全过（含新增 `inference_local_success`、`training_local_success`、`invalid_fetch_mode`） |
| AclnnEngramFetchGradTest | 6 | 全过（含新增 `local_mode_success`） |

说明：
- UT 二进制链接成功，本身验证了 `extern "C"` 签名（含 `fetchMode` 参数）与库符号一致；
- `TestGetWorkspaceSizeWithNNopbaseInner` 在编译期注入的 ascend950 平台 mock 下执行，**真实覆盖了 host 侧全链路**：aclnn 参数透传 → tiling（`fetch_mode` 越界校验、`SetTilingKey` 派生、local workspace=SYSTEM_NEED_WORKSPACE）→ 参数校验；
- 本机为 A3 硬件，`invalid_fetch_mode`（fetch_mode=2 被 `[0,1]` 校验拒绝）在本机真实验证通过；
- **优于设计文档 §11.2 记录的结果**（文档记录 9 项通过 + 6 项因 SoC 不匹配失败；本次 15/15 全过，success 用例在 950 平台 mock 下 GetWorkspaceSize 全部成功）。kernel 级功能（LOCAL/TRAIN_LOCAL 数据面）仍需 950 真机验证，见 §6 行动项。

---

## 3. 设计文档检视（`my-docs/engram_fetch_设计文档.md`）

| 检视项 | 结论 |
|---|---|
| 结构完整性（背景→架构→既有设计→新需求→方案→接口→约束→验证） | ✅ 完整，逻辑闭环 |
| LLM 背景 / weight streaming 动机（§1） | ✅ 准确、自洽 |
| 既有 fetch_mode=0 设计描述（§2-5）与代码 | ✅ 一致（EngramCommContext、五阶段训练内核、窗口分区等均与 `engram_fetch_*_arch35.h` 对应） |
| local 模式设计（§6-7）与实现 | ✅ 一致（见 §4 核对表） |
| 接口变更（§8） | ✅ aclnn 签名追加 `fetchMode`、attr `fetch_mode` OPTIONAL Int 均已实现；attr 顺序（fetch: index 5 / grad: index 2）与 def/tiling 常量对齐 |
| PTA 适配（§9） | ✅ 一致（含 wait 短路、INT32 上限分模式、sf 行数分模式校验） |
| 边界与约束（§10） | ✅ 与 tiling 校验一致（fetch_mode∈[0,1]、hidden 128 对齐、rankSize[2,1024]） |
| 验证计划（§11） | ⚠️ §11.1/11.4 的编译环境描述（CANN 9.1.0 + asc-comm 补丁 + `~/tmp/cann950` 副本）已过时——CANN 9.2.0 原生即可编译（本次实证）；§11.2 的 UT 结果已被本次 15/15 覆盖，建议更新 |

**文档质量总评**：高。新增需求→方案→实现的映射清晰，正确性论证（grad 退化上下文、entry 偏移、maxValue 收缩）到位，验证计划含本机环境限制的诚实说明。主要改进点为环境备注的时效性。

---

## 4. 代码检视明细

### 4.1 设计↔实现核对表（全部 ✅）

| # | 设计要点 | 实现位置 | 核对结果 |
|---|---|---|---|
| 1 | `fetch_mode` attr（OPTIONAL Int，追加末位） | `engram_fetch_def.cpp:46`、`engram_fetch_grad_def.cpp:54` | ✅ |
| 2 | fetch_mode ∈ {0,1} 校验（fetch/grad 均有） | `engram_fetch_tiling.cpp:672-678`、`engram_fetch_grad_tiling.cpp:834-842` | ✅ |
| 3 | tiling key 4 实例派生（infer/train × sliced/local） | `engram_fetch_tiling.cpp:572-584` + `engram_fetch_tiling_key.h` | ✅ |
| 4 | grad tiling key 2 实例（DEFAULT/LOCAL） | `engram_fetch_grad_tiling.cpp:724-731` + `engram_fetch_grad_tiling_key.h` | ✅ |
| 5 | local fetch workspace = SYSTEM_NEED_WORKSPACE | `engram_fetch_tiling.cpp:587-591` | ✅ |
| 6 | 推理 local：表基址 `commBuffer[rankId]`，越界 entry 跳过 | `engram_fetch_local_arch35.h:110-115, 186-189` | ✅ |
| 7 | 推理 local：sf gather（fp8 路径） | `engram_fetch_local_arch35.h:150-168`（numSfPacks>0 才启用） | ✅ |
| 8 | 训练 local：6 输出契约（恒等 perm / [me]=T counts / recvLocalEntry=indices / numRecv=T） | `engram_fetch_local_arch35.h:363-445` | ✅ |
| 9 | 训练 local：零 SyncAll、零窗口、写者唯一（0 号核写 counters） | `engram_fetch_local_arch35.h:447-467` | ✅ |
| 10 | grad local：entryOffset=0（localEntry=entry） | `engram_fetch_grad_arch35.h:79-84`、`engram_fetch_grad_unique.h:641-644` | ✅ |
| 11 | grad local：排序 maxValue=numEntriesPerRank（local 即全表行数，不乘 rankSize） | `engram_fetch_grad_arch35.h:1127-1136` | ✅ |
| 12 | grad local：跳过 UnsortGrad/ExchangeGrad，`recvGradGM_=gradFetchedGM_` | `engram_fetch_grad_arch35.h:1264-1310` | ✅ |
| 13 | grad tiling：local 模式放开 `rankSize*numEntriesPerRank≤INT32_MAX`（仅校验自身） | `engram_fetch_grad_tiling.cpp:588-596` | ✅ |
| 14 | grad local：保留 SyncAll<true>（coreStart/segCount 可见性），numRecv 读取方式与 DEFAULT 一致（DCCI+GetValue） | `engram_fetch_grad_arch35.h:1270-1275, 1296` | ✅ |
| 15 | PTA：`fetch_mode` 参数校验（int + 值域）、schema/pybind 默认值 0 向后兼容 | `elastic_buffer.py:575-591, 620`、`elastic_buffer.cpp:1310-1330` | ✅ |
| 16 | PTA：推理 local wait 短路（不发 aclnnEngramFetchWait） | `elastic_buffer.py:900-908` | ✅ |
| 17 | PTA：csrc local context 构建跳过通道获取/远端地址交换（无 RDMA 依赖），with_grad 仍查 HCCL buffer size | `elastic_buffer.cpp:774-796` | ✅ |
| 18 | PTA：EngramWrite INT32 上限分模式（local：行数≤INT32_MAX） | `elastic_buffer.cpp:1450-1460` | ✅ |
| 19 | PTA：sf 表行数校验分模式（local：sf==storage 行数） | `elastic_buffer.py:798-814` | ✅ |
| 20 | recvLocalEntry 容量：local 写 T ≤ numMaxTokensPerRank ≤ maxR(分配) | csrc `at::zeros({maxR})` + tiling 校验 | ✅ 无越界 |

### 4.2 正确性专项分析（✅ 无阻塞问题）

1. **数据竞争**：train-local 各输出写者唯一（fetched/perm/recvLocalEntry 按 token 区间分片到核；counters/numRecv 仅 0 号核），无跨核共享写，无需 SyncAll——设计 §7.3 的不变量成立。
2. **事件同步链**（train-local `GatherRange`）：indices MTE2→S 读取（`MTE2_S`）、perm V→MTE3（`V_MTE3`）、recvLocalEntry MTE2→MTE3（`MTE2_MTE3`）、每 batch 尾 `MTE3_S` 收口——依赖顺序正确。
3. **grad 退化上下文自洽性**：fetch_train_local 产出（恒等 perm、[me]=T、recvLocalEntry=indices、numRecv=T）与 grad local 分支的假设逐项吻合；T=0 时 fetch 直接 return、numRecv 保持 csrc `at::zeros` 初值 0，grad 走 `WriteNumUniqueZero`——空输入路径正确。
4. **workspace 闲置区域**：grad local 复用 DEFAULT workspace 布局（gradSorted/recvGrad 闲置）——符合设计 §7.5 的"避免 tiling 双路径"取舍，无害。
5. **默认路径零改动**：fetch_mode=0 时 tiling key / workspace / kernel 分支与基线完全一致（`if constexpr` 编译期裁剪 + attr 缺省 nullptr 判定），后向兼容。

### 4.3 问题清单

#### 🟠 P2-1【上游既有问题，非本分支引入，建议另行修复】

`mc2/engram_fetch/op_host/op_tiling/arch35/engram_fetch_tiling.cpp:545`

```cpp
tilingData.rankSize = static_cast<uint32_t>(sendCountsOutShape->GetStorageShape().GetDim(0));
```

- **问题**：csrc 侧（`elastic_buffer.cpp:1522`）分配 sendCountsOut 形状为 `(rankSize * 8,)`（32B 步长契约），此处直接取 dim0 导致 `tilingData.rankSize = 8W`（真实 rankSize 的 8 倍）。对照 grad tiling（`engram_fetch_grad_tiling.cpp:271-277`）已正确做 `dim0 / SEND_COUNTS_ALIGN_FACTOR`。
- **影响**：仅 DEFAULT 训练模式的 workspace 计算使用该值（`SetWorkSpace` 的 `numRanks`），窗口布局按 8W rank 计算导致 workspace 偏大约 8 倍（浪费内存，无功能错误——kernel 侧 rankSize 从 GM comm context 读取真实值）。**本分支 local 模式不受影响**（local workspace 固定 SYSTEM_NEED_WORKSPACE）。
- **验证**：基线 `ca0aebf45` 同位置代码相同，确认非本分支引入。
- **建议**：单独提交修复 `dim0 / 8` 并同步加 shape 校验；与本分支解耦。

#### 🟡 P3 级（低风险改进建议，不阻塞合入）

| # | 位置 | 描述 | 风险 |
|---|---|---|---|
| P3-1 | `engram_fetch_local_arch35.h:319`（train-local） | `numTokens_` 为 uint32，`tilingData.numTokens`(int64) 超 UINT32_MAX 截断。既有 TRAIN_MODE kernel 同风格，且 `num_max_tokens_per_rank` attr 校验实际限值 | 低 |
| P3-2 | `engram_fetch_local_arch35.h:329-332` | ping-pong 双缓冲两次 `FetchEventID(MTE2_MTE3/MTE3_MTE2)` 若返回同一 event id，双缓冲退化为串行等待（性能而非正确性） | 低 |
| P3-3 | `test_aclnn_engram_fetch.cpp`（training_local_success） | sendCountsOut 用 `{2}` 而非真实的 `{16}`（grad UT 用了 `{16}`），通过校验但形状真实性可改进 | 低 |
| P3-4 | `elastic_buffer.py` schema 变更 | 4 个 torch op schema 追加 `int fetch_mode` 对已序列化旧 graph 不兼容（eager 无影响）；设计 §10"完全后向兼容"表述可补充此限定 | 低 |

---

## 5. 检视结论

| 维度 | 结论 |
|---|---|
| 编译（arch35/ascend950） | ✅ 通过（run 包 + 19 .o；4+2 kernel 实例全覆盖、5/3 dtype） |
| UT（op_api host 链路） | ✅ 15/15 通过（含 local/invalid fetch_mode 用例） |
| 设计文档 | ✅ 与实现一致、质量高；环境备注需更新（见行动项 2） |
| 代码正确性 | ✅ 无阻塞问题；1 项上游既有问题（P2-1）移交 |
| 后向兼容 | ✅ fetch_mode 缺省 = 0，aclnn/PTA/内核行为与基线一致 |
| kernel 数据面功能 | ⏳ 需 950 真机验证（本机 A3 无法执行，与设计 §11.1 说明一致） |

---

## 6. 后续行动项（可执行）

1. **【本分支合入前】无阻塞项**——P2-1 为上游既有问题，不阻塞本分支。
2. **【文档】更新设计文档 §11.1/§11.4**：编译环境改为 "CANN ≥9.2.0 原生环境（无需 asc-comm 补丁）"，§11.2 UT 结果更新为 15/15（950 平台 mock 下 GetWorkspaceSize 全链路通过）。
3. **【950 真机功能验证】** 按设计 §11.3 执行：
   - mode=1 推理：各卡写同一全量表 → 随机 indices fetch → 与 CPU golden 逐行比对（含 sf/fp8 路径）；
   - mode=1 训练：fetched 比对 + grad 回传 → gradUnique/uniqueLocalEntry 与 scatter-add 参考比对；断言 perm 恒等、sendCounts/recvCounts 仅 [me]=T、numRecv=T；
   - 回归：mode=0（不传 fetch_mode）与基线一致；
   - 边界：空 indices、单 entry 表、全越界 indices、大表（行数接近 INT32_MAX）。
4. **【上游修复】P2-1**：`engram_fetch_tiling.cpp:545` 改为 `dim0 / SEND_COUNTS_ALIGN_FACTOR`，并补 shape 与 rankSize 一致性校验（可参照 grad tiling 实现）。
5. **【可选优化】P3-1~P3-4**：见 §4.3 表格。

---

*报告生成：CANNWork 检视代理 · 2026-09-16 · 基于分支 `engram-fetch-local-mode` @ `581882847`*
