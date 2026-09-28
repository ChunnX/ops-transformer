# 非量化 All-to-All MatMul 调试记录（2026-09-24）

> 本文档记录 `all_to_all_mm_kernel.h`（非量化 all-to-all MatMul）在 ST 测试中遇到的问题、
> 排查过程与当前结论。对象算子：`mc2/common/op_kernel/apace/kernel/matmul/batch_matmul/`，
> 基于量化版 `all_to_all_qbmm_mx_kernel.h` 移植，使用 apace 本地 fork
> `block_mmad_matmul_basic_splitk.h`。

---

## 1. 背景与测试框架

- **被测算子**：非量化 all-to-all MatMul，三种模式：
  - 模式1（localMatmul=0, REMOTE）：单次 launch，L0C 跨 rank 累加，末段单次 fixpipe
  - 模式2（localMatmul=1, DEFERRED_SYNC）：self 先算驻留 L0C → wait → 远程累加 → 单次 fixpipe
  - 模式3（localMatmul=2）：LOCAL launch（覆盖写+bias）→ REMOTE launch（跳过 self，原子累加）
- **ST 路径**：`tests/st/all_to_all_matmul/`，8 个 case（cases.csv），多 rank 真实通信。
- **环境**：
  - 本机：CANN 9.0.0，与 tensor_api 子模块（面向 9.1.0）不匹配 → **Kernel UT 无法编译**（所有算子都编不过，属环境问题）。
  - 另一台机器：CANN 9.1.0 / bisheng 9.3.0，ST 可跑。

---

## 2. 关键代码差异（非量化 vs 量化）

| 项 | 非量化 | 量化 |
|---|---|---|
| mmad | apace 本地 fork `block_mmad_matmul_basic_splitk.h`（`AscendC::Te::` 旧风格） | tensor 库 `block_mmad_qbmm_mx.h`（`asc::te::` 新风格） |
| splitK 语义 | fork 与**已合入上游的 basic mmad 逐行等价**（operator/Compute/CopyL0CToGM/NeedProcessBias 全一致） | 原生 splitKIdx |
| ST bias | 生成并传入（当前调试态） | **完全不传 bias**（gen_data 无 bias、kernel 不接） |

> 关键结论：**量化版 ST 从未测过 bias 路径**，不能当"bias 已通过"的参考。

---

## 3. 排查过程

### 3.1 矩阵计算问题定位
- 随机 A、B 失败；**A/B 全置 1 通过** → 全 1 掩盖"取值/寻址"类问题（不变性原理：结果只依赖乘积个数，与元素位置无关）。
- A=1,B=random 通过、B=1,A=random 通过 → 曾推断"A、B 各自单独正确，问题在配对"。
- **修正**：两次常量实验其实都"边缘性通过"——`allclose(rtol=1e-2)` 在量级 ~2686 时容差 ~26.9，实测 `abs_err max=14`（~0.5%）被放行；随机数据输出接近 0 时同一误差超过容差。**真实存在 ~0.5% 系统误差**。

### 3.2 8 case 结果（bias=0）
```
PASS  row1  mode0  rank2    FAIL  row3  mode2  rank2
PASS  row2  mode1  rank2    PASS  row4  mode0  rank4
PASS  row5  mode0  rank2    PASS  row6  mode2  rank4
PASS  row7  mode0  rank4    PASS  row8  mode1  rank2
```
- 唯一失败：**row3 = 模式3 + rank=2**。

### 3.3 模式3 rank=2 失败特征
- 输出 = **~22% 半值（单 rank 贡献 ~1000）+ 78% 精确 0**（由 mean diff≈1795、golden≈2016 反推）。
- 正确流程 LOCAL(self≈1000)→REMOTE 原子加(remote≈1000) 应得 ~2000；实际只有一段贡献，且大部分 tile 没落 GM。
- **量化版 mode3 rank=2 通过** → bug 在非量化 mmad 路径，不在 kernel 编排层。

### 3.4 机制分析（候选）
REMOTE 阶段 rank=2 时 `splitKNum=rankSize-1=1`，单次 mmad 的 `splitKIdx=0` **同时是首段和末段**：
`initCmatrix=true`（清零 L0C）+ `FINAL_ACCUMULATION`（触发原子 fixpipe）同一次调用。
- **首末合一 + 原子**：rank=4 的原子 fixpipe 跟在 `initCmatrix=false` 的完整累加后，不触发；
- **L2 一致性**：LOCAL 写 GM 与 REMOTE 原子 RMW 之间，rank=2 流量小、脏行未淘汰（该假设已被量化版通过削弱，但量化无原子差异待确认）。

判别实验（待跑）：
1. LOCAL-only（注释 `RunMatmul()`）：LOCAL 是否全矩阵写 self；
2. REMOTE 去原子：远程是否正常、原子是否问题。

### 3.5 bias 路径（新发现，更基础）
| bias | mode1 结果 |
|---|---|
| 0 | 通过（调试态，从未测 bias） |
| 随机 ±0.5 | 部分正确 + 部分 `65504/-61312/NaN` 垃圾 |
| 全 1.0 | **100% 输出精确 0** |
| 全 1.0 + printf 补丁 | rank0 全过 + rank1 97% 正确（右下角 2.58% 错 ~155） |

- **加非零 bias 连 mode1 都挂 → bias 路径坏，与 mode 无关**。
- 垃圾值 65504/NaN = L0C 被"垃圾 bias 表（BT）"初始化（`needBias` 时 `initCmatrix=false`，L0C 靠 BT 初始化）。
- **量化版 ST 无 bias → 从未测过该路径**。

### 3.6 bias 路径的 DBG 实验
- 补丁：`CopyL12BT` 后 `printf` dump bias L1 内容（`AscendC::print` 编译失败 → 改 `printf`；`__biasbuf__` 解引用失败 → 改普通 `float*`；`static` 局部改类成员）。
- 结果：`l1[0..7] = 0`（GM bias 是 1.0）。
- **陷阱**：AI core 上核心普通 load 读 L1 与 MTE2 写 L1 之间无 barrier 时**不保证一致**，读到旧值 0 可能是假象，不能直接证明 race。
- 待确认：**dump BT**（cube 真正读的表，走 UB 拷贝同步读），或加同步看结果是否变化。

---

## 4. 已完成的工具/文档

### verify_result.py 可视化（多次迭代）
- 加框/加点标注错误元素；
- **output-zero 分析**：全坏/未覆盖时画出"哪里是 0 / 哪里被写"；
- 全坏矩阵输出压缩（明细折叠、密度图粗化）；
- **输出规范化**：统一分节横幅 `[visual] ===== 标题 =====`、统一 3 空格缩进、修 tile 网格边框；
- **tile 坐标密度图**：`--tile MxN`（如 `--tile 256x256`）按 baseM×baseN 分块，行列标注 tile 坐标，符号 `. : o O #` 表示错误密度；
- 用法：`verify_result.py m n rank_num [base_dir] [--visual on|off] [--top N] [--tile MxN]`。

---

## 5. 代码细节澄清

### 5.1 tiling 的 L1 预算与 bias
```
remainSizeForAL1BL1 = hasBias ? (l1Size - BIAS_TABLE_NUM*4) : l1Size   // BIAS_TABLE_NUM=256 → 预留 1KB
stepKa = remainSizeForAL1BL1 / 2 / ((baseM+baseN)*baseK) / dts          // kL1 = baseK * stepKa
```
- baseM/baseN 固定 **256×256**，bias **不影响**它们，只可能让 `kL1` 略小；
- 预留 1KB **正好 = BT（bias 表）大小**（baseN×4 = 1024B）；
- mmad 每 stage 还内联 bias L1 缓冲（`nL1×4`）——**模型上 1KB 预留 vs 每 stage 1KB×2 需求失配**，但当前 shape 因 `stepKa` 整数取整留有大量富余（~128KB），未溢出；**换 shape 可能真溢出**（潜在隐患，非当前故障根因）。

### 5.2 headMSize 显式取值
- 非量化 ST 直接取 CLI/CSV 的 `head_m_size`（量化版是从 tiling 重算，导致该列失效）；
- headMSize 是**通信** M 块，与**计算** tile（mL1/nL1）无关；kernel 用 `CalcDependTileIdx` 桥接；
- 对**数值结果无影响**，只影响通信切分粒度和等待时序（越细越易暴露时序类 bug）。

### 5.3 BufferManager
- 片上 buffer 的"地址分配 + 流水线同步"管理器（`BufferManager<4,4,2>`）；
- 管 L1 A/B/bias、L0A/L0B、L0C、BT；每槽有"地址 + Event ID"，`Lock(pipe)` 提供 RAII 锁防流水线并发访问；
- `InitBias(i, base+aL1OneSize+bL1OneSize, i)` = 给 bias 在每 stage 末尾划位置。

### 5.4 bias 表（BT）
- 给 MMAD/cube 预放 bias 的专用硬件缓冲（BIAS 区），`1×N`（每列一个偏置），**必须 fp32**（指令契约 + L0C fp32 累加精度）；
- 数据流：GM bias → L1（CopyL1FromGM）→ BT（CopyL12BT）→ cube 累加。

---

## 6. 当前状态与下一步

### 已确认
- 非量化 mmad（fork）存在两个独立问题：
  1. **bias 路径**：非零 bias 连 mode1 都挂（垃圾/全 0），与 mode 无关；
  2. **模式3 rank=2**：首末合一 + 原子路径，22% 半值 + 78% 零。

### 待验证
- bias：**dump BT（拷到 UB 同步读）**确认 race，或加 `LockMte2` 同步看结果；
- 模式3 rank2：LOCAL-only / REMOTE 去原子两个判别实验；
- 当前 bias=1.0 + printf 补丁后的 2.58% 右下角差区（~155，疑似少算 K 切片）是否独立问题。

### 环境遗留
- 本机 Kernel UT 因 CANN 9.0.0 与 tensor_api 不匹配无法编译；
- rank=1 的 ST 被通信框架挡住（`HcclRankGraphGetLayers` 单 rank 无拓扑）。
