# Baseline: mla-static-tensor-refactor Todo 1 基线快照

> 用途：flash_mla_with_kvcache 静态 tensor 改造（行为零变化）前的改动前基线。
> 生成时间：2026-08-25 02:31 UTC（Todo 1「基线快照 + 环境确认」）
> 工作目录：/workspace/Code/flash_mla/ops-transformer（无 worktree，直接主仓；build/build_out gitignore，.omo/ 未跟踪属正常）

## 1. Recovery hash（必填）

```
b2d3abaa903a21d74c81bdefc5a641adb8abdf59
```

分支 `feat/flash_mla_with_kvcache`，HEAD 提交 = `refactor(flash_mla_with_kvcache_metadata): simplify interface, derive batch from cache_seqlens`。
Todo 2 大波恢复点 = 此 hash（`git reset --hard b2d3abaa9...` 或 `git checkout -- attention/flash_mla_with_kvcache`）。

## 2. 工作区净度

`git status --porcelain` 输出（完整）：

```
?? .omo/
```

仅 `.omo/` 未跟踪（含本基线文件与 notepads，属预期）；build/build_out 在 .gitignore 无残留。产品代码零改动。

## 3. 编译基线（全量）

- 命令：`rm -rf build build_out && bash build_mla.sh 2>&1 | tee /tmp/t1_build.log`（**全量**；build_mla.sh 脚本内部强制 -j20，实际并行度 = 20）
- 退出码：**0**（PIPE_EXIT=0；日志尾 `[build_mla] SUCCESS`，含 CPack run 包生成与 CRC/SHA256 行）
- 产物清单（**实测路径修正**：本 build 布局下 `build/binary/` 已不存在，内核 .o 不持久保留在 build 树内；真实验证链 = `.run` 内嵌 + 已安装 custom_opp）：
  - `.run` 包：**唯一真实路径 `build_out/cann-ops-transformer-custom_linux-x86_64.run`** = **6547675 B**（= 6.24 MiB）；`build/` 下无 .run
  - kernel `.o` 证据（.run 内嵌，`sh ...run --extract` 解包验证）与已安装 custom_opp 一致，各 2 个：
    - `FlashMlaWithKvcache_d11cc332fbed1b93524f5b5a4b634f08.o` (+ .json)
    - `FlashMlaWithKvcache_e76bc2b10bf573b91e21de09b0aa527a.o` (+ .json)
    - 内嵌路径：`packages/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/ascend950/flash_mla_with_kvcache/`
    - 已安装路径：`/home/developer/Ascend/custom_opp/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/ascend950/flash_mla_with_kvcache/`（同 hash）
  - Todo 2/3/5 编译门的内核产物验证一律以「build_out/.run 再生（mtime 晚于源改动）+ `--extract` 内嵌 2 个 .o/.json + 已安装 custom_opp .o mtime 更新」为准，**不再使用 `find build -name '*.o'`**（build 树不驻留 .o）。
- **构建警告计数：10**（`grep -c "warning:" /tmp/t1_build.log` = 10）。去重后分布（全部为 host 侧源，无 kernel 侧）：
  - `-Wclass-memaccess` ×1：`flash_mla_with_kvcache_tiling.cpp:169`（memset 非平凡类型 FlashMlaWithKvcacheTilingData）
  - `-Wunused-parameter` ×2：`flash_mla_with_kvcache_metadata_check.h:139`（maxSeqlenQ）、`seq_len_checker.cpp:213`（faInfo）
  - `-Wundef` ×2：`flash_mla_with_kvcache_template_tiling_key.h:53`（`__CCE_AICORE__` 未定义 eval 0；host 侧两个 include 路径各报一次）
  - `-Wshadow` ×4：`flash_mla_with_kvcache_tiling_data.h:61/69`（bnStride/n2Stride，两个 include 路径各报一次）
  - `-Wunused-macros` ×1：`build/autogen/updateproto/flash_mla_with_kvcache_proto.cpp:17`
  - 共 10 行。Todo 5「无新增」比对基准 = **10**。

## 4. build_mla.sh --incremental 支持情况

- `grep -n incremental build_mla.sh` 命中（行 27-28）：
  - `#   bash build_mla.sh --incremental   # 增量编译，复用 build/output 目录`
- 结论：**支持** `--incremental` 参数。
- **本任务基线拒绝使用 --incremental**（MUST DO ⑤）：learnings 已知陷阱——增量可能静默跳过 kernel 重编、report 0 退出码但产物陈旧。Todo 2 波后编译门同样以全量 `bash build_mla.sh` 为准（Todo 5 终检也是全量非 incremental）。

## 5. UT 基线

- 命令：`bash build.sh -u --ophost --ops=flash_mla_with_kvcache --soc=ascend950 -j20 2>&1 | tee /tmp/t1_ut.log`
- 退出码：**0**（PIPE_EXIT=0）
- 通过/失败计数：**0 通过 / 0 失败 / 0 用例**（无任何 gtest 用例被编译或执行；日志 0 个 "Built target"/"Scanning dependencies"，CMake 明确打印 `_OpsTestUt_UTestCaseLibraries and _OpsTestUt_UTestAclnnCaseLibraries both empty`）
- 原因（如实记录，非故障）：提交 **a3377fc25 `test: remove flash_mla_with_kvcache/tests (pytests/assets/ut)`** 已从仓库移除该算子全部测试资产：
  - `git ls-files attention/flash_mla_with_kvcache/tests/` = 空
  - `find tests -iname '*flash_mla*'` = 空；`git ls-files tests/ | grep -i mla` = 空（tests/ut 仅通用框架 framework_normal，与算子无关）
  - 算子 `tests/` 物理目录仅剩 `pytests/`（core/test_cases/tools/utils 的 `__pycache__`）与 `assets/`（impl 的 `__pycache__`），**.py 源文件全部不存在**
- 历史参照（**非当前基线**，供 Todo 4 对照）：mla-align-flash-attn 时代（tests 资产尚在）基线 = **36P/7F**，7 个失败统一根因 **KEY-1 漂移**：host prefill 路径 tilingKey 恒 **3672322**（bit21=1）vs CSV/UT 固化期望 **1575170**（bit21=0）；key 派生 workspace 随 key 走（3672322↔34078720，1575170↔16777216）；decode 路径与期望一致。当前已无 tests/ut 资产，**7 个失败用例名无法复跑提取**（learnings 备案：未编造清单，仅记录根因）。
- 对 Todo 2/4 的含义：UT 门当前为空套件，改造后验收将退化为「编译门 0 + 资产缺失如实记录」，不再有 36P/7F 可比对。

## 6. 行为门 before 产物

- `build/baseline/attn_out_before.csv`（pytests result_csv）：**未生成**。原因：pytests harness 源文件不存在（`attention/flash_mla_with_kvcache/tests/pytests/test_flash_mla_with_kvcache.py` 缺失，仅剩 .pyc 缓存，见第 5 节 a3377fc25），无法执行 `--result_csv`。
- examples 输出（`test_aclnn_flash_mla_with_kvcache.cpp`）：**未跑**。原因：行为门需双算子包 flash_mla_with_kvcache + flash_mla_with_kvcache_metadata，而 custom_opp 现装包仅含 flash_mla_with_kvcache（`/home/developer/Ascend/custom_opp/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/ascend950/` 下只有 flash_mla_with_kvcache/），且 Todo 1 为记录型任务不重装包。examples 行为门整体留到 Todo 4 与 after 同轮跑。
- 综合：行为门 before 未采集（环境资产缺失），Todo 4 需以「编译期 + host 侧证据 + 重新安装双算子包后同轮 before/after」策略降级执行。

## 7. pytests functional_stc / functional_nc 状态

- **未跑（资产缺失）**，非"无 NPU"：`npu-smi info` 确认有 NPU（**Ascend950PR**, Health OK, 单卡空闲, 无并发进程）。
- 原因：functional_stc/functional_nc 用例源（pytests/test_cases/*.py + assets 生成链 gen_cases/spec/impl 的 .py）已被 a3377fc25 移除，只剩 `__pycache__`（pyc 文件在 Python 3 下无源不可被 pytest 收集/import）；无 harness 可执行。
- 残留 pyc 模块名（佐证曾有资产）：`compare/golden/graph/inputs/gen_cases/spec/functional_stc/functional_nc/functional_redline_train/functional_redline_infer/functional_redline_train_tnd/performance_redline_train/performance_redline_infer` 等。
- 历史参照（非当前基线）：mla-align-flash-attn 时代 pytests 基线 10P/4F（bf16 三挂 1.6724%/1.6724%/1.8467% + DECODE_LSE 0.5127% 贴线 0.5%）。

## 附：本基线环境事实（Todo 4 前置参考）

- CANN：`/home/developer/Ascend/cann-9.1.0`（CMake 确认 ASCEND_CANN_PACKAGE_PATH）；SOC=ascend950（DAV_3510，日志提示用 Ascend 950PR/DT 硬件，环境为 950PR）
- 已装 custom_opp：`/home/developer/Ascend/custom_opp/vendors/custom_transformer/` 含 flash_mla_with_kvcache（proto/op_api/op_impl + kernel bin d11cc332/e76bc2b1 与构建产物 hash 同名一致）与 flash_mla_with_kvcache_metadata 的 proto 头；kernel bin 仅 flash_mla_with_kvcache（metadata 未装）
- examples 源存在：`attention/flash_mla_with_kvcache/examples/test_aclnn_flash_mla_with_kvcache.cpp`（行为门目标，Todo 4 用）
- 并发：单卡单进程环境，无并发编译窗口；上板结果不受编译噪声影响