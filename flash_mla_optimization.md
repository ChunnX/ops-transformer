# Flash MLA 当前优化方案

> 当前状态更新（2026-09-10）：按用户要求撤回 metadata 的 28 核物理任务重排，恢复使用 aicIdx 写入 FA metadata。随后按用户要求撤回 PV N512/K32，恢复此前 N256/K64 实现；保留 Q 三块常驻和首次 Vec1 ICache 预取。下文含重排的性能数据为历史实验结果，不能作为撤回后的性能数据；撤回后已重新编译安装并测量，最新结果见文末。

记录日期：2026-09-08。适用目录：`attention/flash_mla_with_kvcache/op_kernel`，目标平台为 Ascend 950PR / arch35。

当前保留的实现是 **静态 LocalTensor + 显式 Mutex 同步 + L0A 三块 Q 常驻缓存**。K 按 `128+96+128+96+128=576` 分块，缓存三个 128 宽分块。`mla_pref.py` 同批各三次采样中，相比上一版 112+128 缓存，内核耗时中位数从 **660.587 μs 降到 654.446 μs，下降 0.93%**。

## 1. 当前保留的改动

### 移除管理器和公共矩阵计算封装依赖

- 移除目标内核中的 `BufferManager`、`BuffersPolicyDB` 等 `BuffersPolicy*` 用法，以及公共 `matmul.h` 的直接依赖。
- L0A/L0B 使用固定地址的 `LocalTensor<uint8_t>`，按字节偏移选择子区域，再通过 `.template ReinterpretCast<Q_T/KV_T>()` 获取计算所需类型的张量视图。
- `Mm1MatmulK`、`Mm2MatmulN` 直接调用 `LoadData`、`Mmad` 和 `Mutex::Lock/Unlock`。
- 删除 Cube 侧旧 policy 初始化、释放调用及入口中的 `fa_base_matmul::idCounterNum` 重置。

`ReinterpretCast` 只改变访问同一块内存时的元素类型，不搬运数据，也不做数值转换。

### 在 L0A 常驻三个 128 宽 Q 分块

第一阶段 `Q × Kᵀ` 的 K 维按 `128+96+128+96+128=576` 顺序分块，仍为五次 Mmad。

1. Q 块首轮搬运全部五个分块，将第 0、2、4 块各缓存到一个 16 KiB 槽。
2. 同一 Q 块后续轮次只搬运两个 96 宽分块，复用最后一个临时槽。
3. 切换 Q 时通过 `runInfo.isFirstS2Loop` 刷新缓存，各核/section 的首轮也刷新。
4. 第二阶段 `P × V` 只使用最后一个临时槽，不覆盖 Q 缓存。

满 M=64、FP16/BF16 时，缓存量从 `64×(112+128)×2=30 KiB` 增至 `64×128×2×3=48 KiB`。首轮之后每轮 Q 的 L1→L0A 搬运从 **42 KiB 降到 24 KiB，减少 42.86%**；GM 读取量不因此减少。

缓存块与临时块交替计算，为单临时槽的复用提供计算重叠机会；最后一个分块常驻，使临时槽能在最后一次 mm1 计算期间供 mm2 搬入 P。实际重叠仍受流水调度和 Mutex 等待影响，未用时间线单独量化。

保留 `#pragma unroll`。K 的连续元素顺序不变，但分块边界改变，需要重新验证数值误差。

在维持完整 P 分块 `64×128×2=16 KiB` 和当前 M 分块的条件下，剩余 48 KiB 已全部用于 Q 缓存。完整 Q 为 72 KiB，不能全部放进 64 KiB L0A；进一步增加常驻量需要连同 P/M 分块一起调整。

## 2. 内存布局

L0A 总容量仍为 64 KiB：

| 槽位 | 字节偏移 | 容量 | 用途 | Mutex ID |
|---|---:|---:|---|---:|
| 0 | 0 | 16 KiB | 缓存 Q 第 0 块，K 范围 [0,128) | 11 |
| 1 | 16384 | 16 KiB | 缓存 Q 第 2 块，K 范围 [224,352) | 12 |
| 2 | 32768 | 16 KiB | 缓存 Q 第 4 块，K 范围 [448,576) | 13 |
| 3 | 49152 | 16 KiB | 两个 96 宽 Q 分块及 P 复用 | 7 |

移除 L0A 临时槽游标；L0B 继续独立双缓冲。

单个临时槽能够容纳最大的 Q/P 分块：`64 × 128 × 2 = 16384` 字节。代码保留了容量 `static_assert`。

| 其他区域 | 布局 |
|---|---|
| L0B | 2 × 32 KiB，Mutex ID 为 9/10，独立双缓冲游标 |
| L0C | 2 × 128 KiB，保留原有计算与 Fixpipe 同步 |
| L1 Q | 1 × 72 KiB |
| L1 KV/P | 3 × 144 KiB 环形缓冲 |

本次 Q 缓存优化没有增加 L0A 总容量，没有调整 L1、L0C 或 UB 布局。

## 3. 当前同步方式

需要搬运的 L0A 分块执行：

```text
Lock<PIPE_MTE1> → LoadData(L1→L0A) → Unlock<PIPE_MTE1>
Lock<PIPE_M>    → Mmad            → Unlock<PIPE_M>
```

已经有效的 Q 缓存分块跳过 `PIPE_MTE1` 的锁和搬运，但仍在每次计算时配对执行 `Lock<PIPE_M>` / `Unlock<PIPE_M>`。下一 Q 刷新缓存时，`PIPE_MTE1` 通过同一槽位的 Mutex 等待旧计算完成，防止覆盖尚未消费完的数据。

第二阶段中，P 只搬运一次，L0A 的计算锁覆盖全部四个 N 分块，最后一次 `Mmad` 后释放；L0B 每个 N 分块分别同步。

**当前没有保留“缓存计算锁跨整个 Q 块持有”的实验，也没有将缓存初始化/释放独立移到 Q 块边界。**

## 4. 性能实测

### 测量条件

- 硬件：Ascend 950PR，采样记录的 block 数为 28 AIC / 56 AIV。
- 工具链：CANN 9.1.0，`msprof --aic-mode=task-based`。
- 脚本：仓库根目录的 `mla_pref.py`。
- 参数：BF16，B=1，Q 长度=8，Q 头数=96，KV 头数=1，KV 长度=128000，D=576，DV=512。
- 布局：Q/输出为 TND，KV 为 PA_NZ；分页大小 128；mask_mode=3；不返回 LSE。
- 每次脚本运行采集一次 MLA 调用，脚本在调用前执行其 L2 刷新流程；各版本独立采样三次。
- 比较 `OP Type=FlashMlaWithKvcache` 的设备侧 `Task Duration(us)`，不包含 metadata 算子或主机调用开销。

### 当前三块缓存与上一版 112+128 缓存对比

| 指标 | 上一版，中位数 | 当前三块缓存，中位数 | 变化 |
|---|---:|---:|---:|
| 内核总耗时 | 660.587 μs | 654.446 μs | **−0.93%** |
| AIC MTE1 | 474.496 μs | 448.872 μs | −5.40% |
| AIC Scalar | 245.522 μs | 224.508 μs | −8.56% |
| AIV Scalar | 181.290 μs | 182.119 μs | +0.46% |
| AIC MTE2 | 519.297 μs | 519.450 μs | +0.03% |
| AIC MAC | 632.253 μs | 614.280 μs | −2.84% |

| 总耗时原始值 | 第一次 | 第二次 | 第三次 |
|---|---:|---:|---:|
| 112+128 缓存版，同批重测 | 660.652 μs | 660.436 μs | 660.587 μs |
| 当前三块缓存版 | 653.864 μs | 654.890 μs | 654.446 μs |

上述指标是各自的三次中位数。不同流水之间存在重叠，流水耗时不能相加，也不能用搬运字节的减少比例直接推导总性能收益。

历史阶段的三次采样中位数如下，供追溯基线；它们不是本轮的同批对照：

| 历史版本 | 内核总耗时 | AIC Scalar |
|---|---:|---:|
| 最初使用管理器/policy 的版本 | 681.310 μs | 312.954 μs |
| 静态 Mutex、无 Q 缓存 | 683.929 μs | 232.229 μs |
| 首版 112+112 Q 缓存 | 668.780 μs | 253.118 μs |
| 上一轮 112+128 Q 缓存 | 661.113 μs | 245.576 μs |

首版 Q 缓存相对无缓存 Mutex 版本的历史收益为 2.21%；上一轮 112+128 相对重测的 112+112 版改善 1.18%；本轮 0.93% 以重测的 112+128 版为基线。

以上是该用例的三次采样结果，不代表所有形状都有同样收益，也没有进行统计显著性检验或全形状性能扫描。

## 5. 构建和精度验证

- 最终方案全量编译、打包成功。
- 100 条现有冒烟用例：**100 PASS / 0 FAIL / 0 SKIP**。
- 6 条补充长 KV 用例：**6 PASS / 0 FAIL / 0 SKIP**。覆盖三种分页布局、Q 长度 4/8、batch 1/8，KV 长度最高 400000。
- 沿用现有 smoke 输出最大绝对误差容差：FP16 为 0.001，BF16 为 0.008；没有声称通过更严格的容差。
- 补充长 KV 用例的输出最大绝对误差均为 0.0009765625，记录到的 LSE 最大绝对误差不超过 0.0000019073。现有测试的 PASS 判定依据输出误差，LSE 误差另行记录。
- `git diff --check` 通过，目标目录没有管理器/policy、公共 `matmul.h` 直接引用或旧 `HardEvent` 用法。
- 最终临时安装目录中的 Cube 头文件与工作区源码逐字节一致。

现有 `test.py` 固定按 decode Q 长度 1、分页大小 16 构造输入，而且要求用例显式提供 `return_softmax_lse`。补充长 KV 测试使用临时副本 `/tmp/flash_mla_qcache_longkv_test.py`，按用例的 Q 长度和分页大小构造输入，并为缺失的 LSE 开关设置默认值；没有修改仓库原测试脚本。

## 6. 已测试但未保留的方案

以下实验在上一轮 112+112 缓存优化阶段进行，不能直接等同于本轮三个缓存块交替计算的实现。

| 实验 | 实测结果 | 处理 |
|---|---|---|
| M=64、S2=128 满块参数特化及短循环展开 | 单次总耗时 683.215 μs，没有显示明显收益 | 撤回该特化方案 |
| Q 缓存配合 K 分块 `128+128+128+96+96` | 单次总耗时 675.959 μs，慢于当时的 112+112 缓存方案 | 恢复 `4×112+128` |
| 缓存计算锁从 Q 首轮持有到末轮 | 三次中位数 668.071 μs；AIC Scalar 236.117 μs | 总耗时差异仅约 0.11%，撤回 |
| 缓存初始化、加锁、释放移到 Q 块边界 | 三次中位数 667.761 μs；AIC Scalar 246.504 μs | 总耗时差异仅约 0.15%，撤回 |

后两种同步实验均通过 100+6 条精度测试，但总耗时改善仍在此前采样波动范围内，未确认新加速，因此没有保留。

### 三块 Q 缓存基础上的搬运顺序实验

在提交 `d14044028` 基础上，分别测试以下两项调整；保持 K 分块、48 KiB Q 缓存及 L0B 双缓冲不变。各版本用 `mla_pref.py` 独立采样三次。

| 版本 | 总耗时三次值（μs） | 总耗时中位数 | AIC Scalar 中位数 |
|---|---|---:|---:|
| 当前实现，同批重测 | 654.145 / 655.219 / 653.348 | 654.145 μs | 224.293 μs |
| mm1 先搬 B，再等待/搬 A | 656.146 / 655.814 / 654.468 | 655.814 μs | 214.217 μs |
| mm2 先搬首块 V，再等待/搬 P | 655.114 / 654.600 / 654.776 | 654.776 μs | 219.775 μs |

两项修改的 Scalar 均下降，但内核总耗时分别增加约 0.26% / 0.10%，没有显示加速，均撤回。三次样本不足以判定微小差异的统计显著性。mm2 实验通过 100 条冒烟和 6 条长 KV 精度测试；mm1 实验因没有性能收益，未追加精度回归。最终源码及安装包恢复到实验前版本。

明细：`/tmp/flash_mla_border_comparison.json`；mm2 精度日志：`/tmp/flash_mla_border_mm2_test.log`、`/tmp/flash_mla_border_mm2_longkv_test.log`。下一步需要分析流水等待时间线，或对小 M 形状单独验证完整 Q 常驻；本轮尚未实施这两项。

### 启动预取、N=256 与临时 K=128/64 实验

在 `d14044028` 基础上单独测试三种方案，各用 `mla_pref.py` 采样三次，不插入打点。

| 版本 | 总耗时三次值（μs） | 中位数（μs） | 相对基线 |
|---|---|---:|---:|
| 基线，同批重测 | 653.834 / 654.347 / 654.441 | 654.347 | — |
| 启动时先搬当前 Q/KV，再预取下一块 KV | 654.362 / 652.495 / 655.621 | 654.362 | +0.002% |
| mm2 N=256+256，整个 L0B 单缓冲 | 887.176 / 887.805 / 886.138 | 887.176 | +35.58% |
| K=128+128+128+64+128，缓存第 0/2/4 块 | 664.203 / 665.168 / 663.618 | 664.203 | +1.51% |

三项均未显示总耗时收益，全部撤回。启动顺序调整基本持平，未追加精度测试。N=256 与 K=128/64 两项各通过 100 条冒烟和 6 条长 KV 测试，沿用原有 smoke 容差。

N=256 将 mm2 的 Mmad 从四次减为两次，但整个 64 KiB L0B 在两次计算间复用，使用 Mutex 9/10 同时锁住两个半槽，失去 B 双缓冲。该实现的实测退化说明仅减少 Mmad 次数不足以改善当前用例。K=128/64 保持缓存量、搬运量、双缓冲和 Mmad 次数不变，但 AIC MAC 中位数由 613.683 增至 623.062 μs；具体内部原因未由指令级数据确认。

数据明细：`flash_mla_pipeline_analysis/next_experiments.json`（副本源自 `/tmp/mla_next_comparison.json`）。日志前缀为 `/tmp/mla_start_*`、`/tmp/mla_n256_*`、`/tmp/mla_k12864_*`。最终内核恢复到 `d14044028` 并重新构建、安装。

## 7. 代码位置

- [Cube 内核](attention/flash_mla_with_kvcache/op_kernel/arch35/flash_mla_with_kvcache_block_cube.h)：静态布局、Mutex ID、Q 缓存、`Mm1MatmulK`、`Mm2MatmulN`。
- [主 Kernel](attention/flash_mla_with_kvcache/op_kernel/arch35/flash_mla_with_kvcache_kernel.h)：任务调度及删除 Cube 旧事件初始化/释放调用。
- [算子入口](attention/flash_mla_with_kvcache/op_kernel/flash_mla_with_kvcache.cpp)：删除旧管理器计数器重置。
- [性能脚本](mla_pref.py)：固定长 KV 性能用例。

## 8. 复现命令

在仓库根目录、已配置 CANN 环境的终端中执行：

```bash
# 全量构建
bash build.sh --pkg --soc=ascend950 \
    --ops=flash_mla_with_kvcache,flash_mla_with_kvcache_metadata -j20

# 安装到本次验证使用的临时目录
bash build_out/cann-ops-transformer-custom_linux-x86_64.run \
    --quiet --install-path=/tmp/flash_mla_perf_opt_validation
source /tmp/flash_mla_perf_opt_validation/vendors/custom_transformer/bin/set_env.bash

# 使用新目录，独立采样三次
MLA_PROFILE_ROOT=$(mktemp -d /tmp/flash_mla_qcache_repro.XXXXXX)
for run in 1 2 3; do
    msprof --output="${MLA_PROFILE_ROOT}/run_${run}" \
        --aic-mode=task-based python3 mla_pref.py
done

# 100 条冒烟精度测试
python3 test.py --repo-root /workspace/Code/perf/ops-transformer-testkit --num 100

# 6 条长 KV 精度测试：依赖本次保留的临时测试副本
python3 /tmp/flash_mla_qcache_longkv_test.py \
    --repo-root /workspace/Code/perf/ops-transformer-testkit \
    --cases /workspace/Code/perf/ops-transformer-testkit/test_cases/flash_mla_with_kvcache/mla_longkv.py \
    --num 6
```

`test.py` 默认用例文件为 `/tmp/opencode/gen_1000_b777.py`。上述测试集、临时脚本和日志依赖本机环境，迁移到其他机器时需一并准备；性能脚本和内核源码位于本仓库。

## 9. 本次验证记录

以下临时文件便于在本机追溯；核心数值已经记录在本文中：

- 本轮三块缓存与 112+128 对比明细：`/tmp/flash_mla_ksearch_comparison.json`。
- 上一轮 112+112 与 112+128 对比明细：`/tmp/flash_mla_cache112128_comparison.json`。
- 首版 Q 缓存历史性能明细：`/tmp/flash_mla_qcache_perf_comparison.json`。
- 后续同步实验明细：`/tmp/flash_mla_qcache_sync_experiments.json`。
- 当前方案 100 条冒烟日志：`/tmp/flash_mla_ksearch_alt3_test.log`。
- 当前方案 6 条长 KV 日志：`/tmp/flash_mla_ksearch_alt3_longkv_test.log`。
- 当前方案构建日志：`/tmp/flash_mla_ksearch_alt3_build.log`。
- 当前方案安装日志：`/tmp/flash_mla_ksearch_alt3_install.log`。


## 2026-09-09：首次 Vec1 指令预取

在既有PV N512/K32与物理核重排基础上，ProcessVec1首次进入（loop==0，每FA section一次）增加ICachePreLoad(7)，计算、缓冲和同步不变。9.2固定BF16/H96/Q8/KV128000的A→B→A中位耗时为645.782→646.961→645.501μs，支持约1–1.5μs小收益，当前MFU约87.56%，未达到90%。固定长用例独立FP32/BF16 golden零失败、11形状同输入旧输出回归逐位一致。实际流水图仍有1–2μs启动空档，不能把收益解释成全部空档已消除。完整证据见flash_mla_mfu90_analysis.md及flash_mla_mfu90_results/vicache_*。


## 撤回核重排与 N512 后的当前方案性能（2026-09-10）

当前方案为 Q 三块常驻 + PV N256/K64 + 首次 Vec1 ICachePreLoad(7)，metadata 恢复原 aicIdx 映射。已在 CANN_0828 下重新编译并安装至 /tmp/mla_mfu_opt，未修改 mla_pref.py。固定 BF16、B1、H96、Q8、KV128000、冷 L2、三次独立进程采样为 650.161 / 650.155 / 651.448 μs，中位数 650.161 μs，沿用 378.47 TFLOPS 峰值口径，MFU 86.9292%。距90%阈值627.978 μs约22.183 μs。MAC中位数613.151 μs、MTE1 449.515 μs、MTE2 520.157 μs；各流水时间相互重叠，不能直接相加。与最初654.108 μs历史基线相比约少3.947 μs（0.60%），非同批交替对照，不能据此精确归因单项收益。输出采集与独立golden的Q/K哈希相同，按abs>0.0001且rel>0.0078125标准对FP32/BF16 golden均0/393216失败。结果见comparison.json中n256_noperm_icache及n256_noperm_icache_precision.json。
