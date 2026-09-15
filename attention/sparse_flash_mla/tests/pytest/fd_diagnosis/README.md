# SMLA 普通 FD 输出错误定位与修复

## 已复现的触发条件

测试日期：2026-09-15。NPU：Ascend950DT_9584，CANN 9.2.0。
使用 BF16、N1=64、N2=1、D=512、BSND/BSND、SWA 全窗口，开启 sink 和 LSE，
seed=42、输入范围 [-2, 2]、deterministic level=0。比较器沿用仓库 `check_result`。

固定 B=1、Q=1、KV=512 的同一份输入，仅改变 metadata 的 FA 分区：

| FA 分区（每块 128 token） | 输出 | 最大绝对误差 | LSE |
|---|---|---:|---|
| 2+2 | Pass | 0.03125 | Pass |
| 3+1 | Failed | 1.625 | Pass |

该对照只重写 FA/FD 任务表；两种分区均完整、无重叠地覆盖全部 KV，FD 合并两个 slot。
人工分区用于隔离每核工作量，不代表 metadata 调度器的默认选择。

默认 metadata 也有单行失败用例：B=1、Q=1、KV=12288，输出最大绝对误差 2.40625。
KV=8192 的单行用例通过。固定 KV=4096 时，B×Q=1×2、2×1 通过，2×2 失败。
因此，多行不是必要条件；应检查输出累加值仍存活时的缓冲区复用。

## 源码解释

修复前的 `sparse_flash_mla_csa_block_vector_arch35.h` 中：

- `StageLegacyVec1Lse` 将 `stage2OutBufs.tensor` 传给 `StageVec1Lse` 作广播临时区。
- `ProcessVec2` 也用 `stage2OutBufs.tensor` 保存跨 S2 块累加的输出。
- `common/flash_decode.h::StageVec1Lse` 把 max 和 sum 分别广播到临时区的
  `[0, 1024)`、`[1024, 2048)` 字节（本组每个向量核 32 行）。
- SWA 和 CSA 的流水线均先调用下一块的 Vec1，再调用上一块的 Vec2。

以一个核处理三块为例：

| 顺序 | Vec1 | Vec2 |
|---|---|---|
| 1 | 第 0 块 | 尚未处理 |
| 2 | 第 1 块 | 第 0 块，初始化输出累加值 |
| 3 | 第 2 块，最后一块，广播 max/sum 覆盖输出累加值 | 第 1 块，读取已被覆盖的累加值 |
| 4 | 流水线排空 | 第 2 块，归一化并写出错误 partial O |

这里发生的是活跃累加值被覆盖；仅等待上一笔 DMA 写回完成不能保存该值。
LSE 的 max/sum 本身仍正确，所以 LSE 可以通过。
在 N1=64、D=512 的失败对照中，大误差只出现在 head 0、32，正好对应两个向量核
各自输出累加缓冲区的前 2048 字节。

还用 CPU 显式模拟了这次覆盖：先计算三块的 max/sum，再按上述顺序用最终广播值覆盖
第 0 块输出累加值的 head 0、32，继续更新第 1、2 块并正常进行 FD 合并。
这个“故意损坏”的 CPU 预测与旧内核输出比较达到 100% 通过，最大差异 0.03125，
平均差异约 1.43e-6。它用于验证错误机制，不能用作正确性 golden。

## 复现

`reproduce.py` 和 `base_params.json` 可以重新生成全部七项诊断输入，不依赖原先保存的 `.pt`。
需要可用的 CANN 环境、SMLA Python 扩展以及启用 FD 的自定义算子安装包。
`--package-root` 可选，用于指定 `cann_ops_transformer` Python 包目录。

```bash
source /home/panjie/cann_ins/cann-9.2.0/set_env.sh
export ASCEND_CUSTOM_OPP_PATH=/tmp/mla_fd_shared_20260915/install/vendors/custom_transformer
export TORCH_EXTENSIONS_DIR=/tmp/mla_fd_shared_20260915/extensions
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MAX_JOBS=4 SMLA_BATCH_CONSISTENCY=off
python attention/sparse_flash_mla/tests/pytest/fd_diagnosis/reproduce.py \
  --artifact-dir /tmp/mla_fd_shared_20260915/smla_diagnosis_portable \
  --tag baseline \
  --package-root /tmp/mla_fd_shared_20260915/python/cann_ops_transformer
```

脚本将输入缓存到 `inputs/`，将输出、metadata 和逐 head 误差保存到 `<tag>/`。
更换安装包后使用不同 tag，可用完全相同的输入进行前后对照。
运行前应检查安装包与源码版本；以上本机安装路径不能代替源码构建步骤。

验证覆盖机制的 CPU 预测：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python \
  attention/sparse_flash_mla/tests/pytest/fd_diagnosis/predict_corruption.py \
  /tmp/mla_fd_shared_20260915/smla_diagnosis_portable/baseline/swa_b1_q1_kv512_parts3_1.pt
```

## 历史隔离候选

`dedicated_lse_scratch.patch` 为普通 FD 增加独立的 2048 字节临时区，
使 Vec1 广播不再覆盖 Vec2 累加值。此文件保留定位阶段的最小候选；
仓库现已应用正式修复，不应再应用这个历史补丁。

候选已编译到本机 `/tmp/mla_fd_shared_20260915/install_smla_scratch/`：

| 回归 | 原包 | 独立临时区候选 |
|---|---|---|
| 原 8 项普通 FD 输出 | 3/8 | 8/8 |
| 原 8 项 LSE | 8/8 | 8/8 |
| 本目录 7 项诊断输出 | 4/7 | 7/7 |
| 本目录 7 项诊断 LSE | 7/7 | 7/7 |
| Batch 一致性（seed=42、level=3） | 16 通过、4 不适用 | 16 通过、4 不适用 |

原五个失败用例的输出最大绝对误差均降至 0.03125。候选只改变广播临时区，
没有调整输入、golden 或精度阈值。更换上述 `ASCEND_CUSTOM_OPP_PATH` 并使用
`--tag scratch` 可复跑诊断。构建时仅筛选了生成目录中的 BF16 模板（六种布局/模式组合，
包含 batch 和 vec-S2-address 分支）；生成头文件和完整模板声明均已恢复，尚未完成全模板构建。

Batch 回归同时满足原 golden 精度与变换前后输出/LSE 逐 bit 相等；seed=42 的完整比较明细
与旧基线一致。首轮候选 batch 调用未显式设置变换 seed，后续已显式设置为 42 并重跑，
这里记录的是重跑结果。四项不适用仍是原来的 mask 窗口变化或 PA 容量不足。

`results.json` 保存上述数值结果、逐 head 误差、CPU 覆盖预测及源码/二进制哈希。
完整设备产物位于 `/tmp/mla_fd_shared_20260915/` 下的 `smla_scratch/smla/`、
`smla_diagnosis_portable/{baseline,scratch}/`、`smla_scratch_consistency/smla/`。

## 正式源码修复

普通路径分配独立的 `fdLseTmpUb`（512 个 float，共 2048 字节），
`StageLegacyVec1Lse` 使用它广播 max/sum，避免覆盖 Vec2 活跃累加值。
该分配不依赖调用者是否要求返回 LSE，因为 FD 合并本身需要 max/sum。
Batch 路径继续使用已有临时区。

普通 Vec2 改用 `StageVec2PartialOAndWait`，与 MQSMLA、QSMLA 的写回完成后再复用
缓冲区的约束一致。共享 helper 的 sum 偏移仍为 256 个 float，只改为具名常量；
新增编译期检查保证 SMLA 每个向量 tile 的行数不超过临时区容量。

`reproduce.py` 新增以下可组合的验证选项：

- `--dtype float16`：FP16 输入；默认 BF16。
- `--no-lse`：关闭对外返回 LSE，核对内部 FD staging 仍正确。
- `--edge-cases`：在原七项之外增加 heads=1/7/31/33/63/65/127/128 的 FD 用例，
  以及 KV=1/127/128 的非 FD 用例，共 18 项；断言 metadata 中 FD 是否实际启用。
- `--check`：输出或所请求的 LSE 精度不通过时以非零退出码结束。

正式包诊断命令（需要先构建当前源码）：

```bash
export ASCEND_CUSTOM_OPP_PATH=/tmp/mla_fd_shared_20260915/formal_fix/install/vendors/custom_transformer
python attention/sparse_flash_mla/tests/pytest/fd_diagnosis/reproduce.py \
  --artifact-dir /tmp/mla_fd_shared_20260915/formal_fix/diagnostic \
  --package-root /tmp/mla_fd_shared_20260915/python/cann_ops_transformer \
  --tag bfloat16_lse --dtype bfloat16 --edge-cases --check
```

使用不同 tag 并组合上述 dtype 和 LSE 选项，可复跑四组诊断。

### 构建与资源边界

正式 SMLA FP16/BF16 各 320 个 tiling key 全部编译通过；MQSMLA/QSMLA 分别
重编 48/24 个相关模板，二进制 SHA256 与原包相同。核对了完整 SMLA key 集合、
产物 JSON 哈希与源码哈希。全模板编译不能代替所有模板的设备运行验证。

新增临时区占 2048 字节。按现有静态布局核算，普通路径最大使用 218624 字节 UB，
平台容量 253952 字节，余量 35328 字节；batch 布局未改变，最大 220672 字节。

### 正式设备验证结果

| 组合 | 原 8 项输出 | 扩展 18 项输出 | 返回的 LSE |
|---|---:|---:|---:|
| BF16 / LSE 开 | 8/8 | 18/18 | 26/26 |
| BF16 / LSE 关 | 8/8 | 18/18 | 未请求 |
| FP16 / LSE 开 | 8/8 | 18/18 | 26/26 |
| FP16 / LSE 关 | 8/8 | 18/18 | 未请求 |

合计 104 次输出检查、52 次返回 LSE 检查全部通过；52 组开关对照的输出按字节比较相同。
未修改精度阈值。三算子 batch 共 48 通过、12 不适用，详细结果与 seed=42 原基线完全相同。
MQSMLA 普通 FD 8/8 通过；QSMLA 实际 FD 分区参考 8/8 通过，其原普通 golden 分区差异仍存在。

[formal_results.json](formal_results.json) 保存正式结果、编译 key/哈希及 UB 明细；
历史 [results.json](results.json) 保留定位阶段对照。完整设备输入和输出、编译日志及本机
共享回归脚本仍在 `/tmp/mla_fd_shared_20260915/formal_fix/`，不随仓库迁移。
这轮完成了 SMLA 完整模板编译，但设备测试只覆盖上述输入；四个不适用的 batch 变换尚未补齐。

### 性能复核

在五项固定 BF16 用例上，对比“独立临时区、未增加等待”候选与正式包。
逐次调用计时波动大，改用 NPU 图捕获 100 次调用后重复重放、取五组中位数；
KV=256 首轮波动较大，单独增加预热和重放次数复核。最终耗时变化为 −0.75%～+0.36%，
这些用例中未观察到稳定退化。每轮重放后的输出/LSE 精度均通过。
这不代表所有输入无性能影响，也不是单个 device kernel 的独立 profiling。

原始逐次调用、图重放、KV=256 复核采样均保存在 `formal_results.json` 的 `benchmarks` 中，
最终对照在 `performance_comparison` 中；完整方法和逐项耗时见仓库根目录 `MLA_FD_PROGRESS.md`。

### 提交格式化与版本记录

提交钩子仅调整换行、缩进及补丁空白。C++ 非空白源码和 Python AST 均保持一致，
历史补丁格式化前后产生相同源码；无需改变设备验证结论。
`formal_results.json` 保留实际验证版本的 `source_sha256`，并增加提交版本的
`formatted_source_sha256` 与 `pre_commit_formatting_verification`。
历史 `results.json` 的哈希仍指定位阶段原始文件；补丁及脚本的新旧哈希映射见正式记录。
