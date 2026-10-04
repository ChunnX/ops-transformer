# 需求背景（required）

## 需求来源

- 任务来源：昇腾社区开源算子开发大赛 / ops-transformer 社区算子开发任务。
- 交付目标仓库：https://gitcode.com/cann/ops-transformer ，合入路径为 `experimental/attention/sageattention2`。
- 对标参考：SageAttention2（ICML 2025，arXiv:2411.10958），GitHub 开源仓 `thu-ml/SageAttention` main 分支（`sageattention==2.2.0`），对标 `sageattn_qk_int8_pv_fp8_cuda`。
- 性能基线：ops-transformer 同仓 FIA（FusedInferAttentionScore），针对 Atlas 950PR 对标 `aclnnFusedInferAttentionScoreV5` 与 `torch_npu.npu_fused_infer_attention_score`。

## 背景介绍

### SageAttention2 算子实现优化

- Attention 计算是现代大语言模型与多模态扩散模型（如 CogVideoX、HunyuanVideo、Flux 等）计算与显存访问的核心瓶颈。
- SageAttention2 提出了 INT8 QK^T 与 FP8 PV 的混合低精度量化注意力方案，结合精细粒度量化、两级累加抑制截断误差、以及序列维均值扣减（smooth_k）消除离群值。
- 在昇腾 Atlas 950PR 架构上，基于原生 FP8 / MXFP8 计算能力与 AscendC + CATLASS 框架，实现高效、高精度的 SageAttention2 算子，显著提升推理吞吐。

### 对标实现现状分析

- 开源 GPU 实现特点：
  - 基于 Triton 与 CUDA 实现 INT8 QK^T 与 FP8 PV；
  - 量化粒度支持 per_thread（交织步长分组）与 per_warp（连续 32 token 分组）；
  - 支持 smooth_k 与 smooth_v 离群值平滑；
  - 累加模式支持 fp32、fp32+fp32、fp32+fp16。
- 昇腾 950PR 架构映射：
  - 硬件原生支持 INT8 与 FP8(e4m3) MMAD 矩阵乘指令；
  - 采用 1C2V（1 个 Cube + 2 个 Vector）异构协同架构；
  - 利用 L0C 临时累加器与 FixPipe 随路排布/精度转换硬件通道，消灭中间全量矩阵的 GM 往返。

# 需求分析（required）

## 需求描述

- 基于 AscendC + CATLASS 联合开发 SageAttention2 算子（仅前向 INT8 QK^T + FP8 PV 路径），提供统一 aclnn 接口与 PyTorch 适配层接口。
- 性能硬性门禁：在 Atlas 950PR、CANN 9.2.0-beta.1 环境下，同机同 shape 同 dtype 相对 FIA，PyTorch 与 aclnn 双路径 Speedup 均达到 1.6 倍以上（Speedup >= 1.6x）。
- 精度标准：满足《生态算子开源精度标准》，以 GPU 侧 `sageattn_qk_int8_pv_fp8_cuda` 为 L-A 主标杆（混合容差 matched_ratio >= 0.99），以 FP32 SDPA 为 L-B 辅助标杆。

## 需求拆解

1. PyTorch 层接口：`sageattn` 自动分发与 `sageattn_qk_int8_pv_fp8_asc` 显式入口，参数逐项对齐开源。
2. aclnn 两段式接口：`aclnnSageAttention2GetWorkspaceSize` 与 `aclnnSageAttention2`。
3. 量化与平滑方案：自研 Vector INT8 per_thread / per_warp 量化，smooth_k 序列维扣减均值，FP8(e4m3) V 量化（scale_max=2.25/448.0）。
4. 累加模式：支持 fp32、fp32+fp32、fp32+fp16（默认验收档），实现低精度 MMA 临时累加与定期归约。
5. 特性支撑：head_dim pad（48补到64、96补到128、大于128报错并自动裁切）、GQA（Hq 整除 Hk）、causal 掩码、`return_lse`（返回自然对数域对齐的 logsumexp）。

# 详细设计（required）

## 单一定义点清单（2026-10-03 打磨轮）

| 不变量 | 唯一定义点 | 消费方 |
|---|---|---|
| 分形粒度（M0/N0/K0、LoadData2D repeat） | 官方 catlass.hpp（sage_tile.hpp include 直引派生） | kernel 全部 MMA/搬运 |
| 行打包上限 MAX_PACK_ROWS=128 | tiling.h | kernel 窗口/host total_rows/InspectTilingPlan |
| Q/K scale 槽步长 kQScaleSlotStride=4096 / kKScaleSlotStride=2048 | tiling.h（容量域经 QScaleCount/KScaleCount constexpr static_assert 锁定） | SimMemory/设备 UB 槽寻址 |
| QK scale 定档公式 FinalizeQkScale | quant_common.h | kernel 双分支 + quant 参考实现（8 处收敛） |
| 量化组号 QScaleIndex/KScaleIndex（constexpr） | quant_common.h | 全链路 |
| V 物理槽寻址 VSlotOfToken | quant_common.h | golden/twin/kernel（3 处收敛） |
| 因果掩码哨兵 SAGE_MASK_SENTINEL | online_softmax.h | kernel/VF/twin（12 处收敛） |
| 编译域单宏 SAGE_ATTN_DEVICE_COMPILE（__CCE__ 直判）与 __aicore__ 桩 | tiling.h | 全部头文件（历史 SAGE_ATTN_KERNEL_API / __CCE_KT_TEST__ 桩域随 2026-10-04 死代码清扫并入本宏） |
| TilingData ctypes 镜像 | tests/tiling_mirror.py | test_tiling_struct/test_kernel_twin |
| FP16/BF16 位型转换（RNE） | quant_common.h | 算子/golden/benchmark（截断版已删除） |
| 数值常量（Python 侧） | error_check.py | dispatch/_ops/golden 生成器 import（本地副本已删） |
| CATLASS_ROOT 解析 | cmake/SageAttnDeps.cmake | op_kernel/op_host CMake |
| 接口布局字符串（仅 HND/NHD） | op_api 白名单报错 | op_host/proto/def 默认值 HND |
| 接口默认值常量（Python DEFAULT_*） | error_check.py | dispatch/_ops 签名默认 |
| 接口默认值常量（C++ kDefault*） | aclnn/sage_attn_proto.h | op_host tiling 缺省折叠 |
| V 量化 scale_max 分档 v_scale_max_for | dispatch.py | golden 生成器/测试填充（2.25/448 字面量已消） |
| tiling 枚举编码镜像（LAYOUT/GRAN/ACCUM） | tests/tiling_mirror.py | test_kernel_twin 等 |
| ActiveKvBlocks/SubKm 核内辅助 | op_kernel 私有方法 | 4 处重算 + 2 处 km 舍回收敛 |
| 输入位型档位归一 NormDtypeMode（M5） | op_kernel/sage_attention2_template_tiling_key.h | 核内 kDtype / host tiling / tiling key 编码共用，消费侧禁再展开三元式 |
| L1 Data+Scale 相邻摆布契约 CopyUbRowToL1 | sageattn_fp8/catlass/sage_tile.hpp | 核内 Q/K/V 三处 UB→L1 行搬（原 host 侧 PackDataAndScaleAdjacent 死链已删，2026-10-04） |
| LSE_ROW_STRIDE / MAX_GQA_RATIO | sage_resource.hpp / tiling.h | V1 子偏移、SimMemory 配额、host 上限 |

## 死代码清扫记录（2026-10-03）

以下四件为零生产消费的"已弃用策略"载体，经全仓引用图核验后删除（测试同步退役，
决策上下文以本节与本文件"同步与流水守卫"节注记存档）：

| 删除件 | 原职责 | 删除依据 |
|---|---|---|
| `sageattn_fp8/ascendc/pipeline_guard.h`（319 行） | UbTile 类型状态守卫（RES-05，编译期约束 DMA/计算状态转移） | 设备核热路径以 sync_guard.h 显式 flag 协议保序，该头零生产 include，仅 tests/test_pipeline_guard.cpp 引用 |
| `sageattn_fp8/ascendc/accum_reduce.h`（111 行） | AccumReduceController 归约节奏控制器、FixpipeConfig、QuantizeToFp16 仿真壳 | PV 归约节奏唯一定义点已收敛至 twin::AccumulatePvRow（k_step 子块域）+ 设备侧逐块 FixPipe 裸搬合并（档位直读 tiling.pv_accum_dtype）；三符号零非测试消费 |
| `tests/test_output_queue.cpp`（约 1000 行） | OutputRetireQueue（3 槽延迟成对归约 + skip-scale 变体）回归 | 队列类本体即测试文件内本地定义（"非交付热路径组件"），其 skip-scale 双锚点策略与 golden 逐块锚定强制 rescale 轨迹冲突，A7（task/12 D3）已裁定热路径弃用 |
| `tests/test_pipeline_guard.cpp` | pipeline_guard 类型状态协议测试 | 随被测载体删除 |

随动清理：op_kernel/sageattention2.cpp 与 sage_pv_mmad.hpp 的 accum_reduce.h include、
run.sh 对应编译/运行钩子、test_tiling_struct.cpp 的 AccumReduceController 用例
（编号重排）、README/PR 描述目录与同步机制条目。

## GQA 分块打包策略（2026-10-03，P7 修复）

官方 500 用例套件含 32:4（gqa_ratio=8）× 9 条，而 Q 头打包受单任务行上限
MAX_PACK_ROWS=128（tiling.h 唯一定义点）约束：4:1 × m_per_core=32 已满 128 行。
方案：单任务打包 `min(gqa_ratio, MAX_PACK_ROWS / m_per_core)` 个 Q 头，超出按
chunk 拆多任务（8:1 → 2 chunk × 4 头 × 32 行），任务空间扩为
`b × hkv × chunks × q_blks`。窗口 token 宽度恒为 m_per_core（32/64），
32-token 量化子块不被切割；Q scale 槽按全局头序（cur_g0_ + g）寻址，
GM 行寻址按窗口局部 g（基址已含 hq_first）。对 gqa ≤ 4 的既有用例
chunks=1，任务解码与调度算术逐位不变（twin 25 桶 golden 回归锁定）；
8:1 路径由 tests/test_gqa_chunk.py 的双路径位级一致性测试覆盖。
上限 gqa_ratio ≤ 8：SimMemory / GM scale 暂存条带按 8 头配额冻结
（官方套件最大 8:1）。

## 算子分析

### 数学公式

- 主计算流：
  - O = softmax(QK^T * s) * V
- 量化与平滑：
  - K 侧不做均值扣减（7 参收敛后核内无 key_mean 供给，K_hat = quant_int8(K)）；smooth_v 生效档（fp32）为 V 减均值量化 + 输出端 vm 补偿
  - Q_hat = quant_int8(Q), K_hat = quant_int8(K), V_hat = quant_fp8(V)
  - P = softmax(Q_hat * K_hat^T * scale_q * scale_k * s)
  - O = P * V_hat * scale_v
- LSE 语义与校正：
  - LSE = lse_raw / 1.44269504（exp2 域补扣 S_FP8_OFFSET 后换回自然域；7 参收敛后核内无 lse_correction 供给，`Q * k_mean * sm_scale` 校正加项不生效，该数学定义仅存于 CPU 参考轨迹与上游 CUDA）

### 支持数据类型

- 输入 Q/K/V：torch.float16、torch.bfloat16（三者 dtype 必须一致）
- 量化内部类型：Q/K 为 INT8，V 为 FP8(e4m3)
- 输出 O：与 Q 相同 dtype（FP16 或 BF16）
- 输出 LSE（当 return_lse=True 时）：Float32

### 支持形状

- HND 布局：
  - Q: [batch, num_qo_heads, qo_len, head_dim]
  - K/V: [batch, num_kv_heads, kv_len, head_dim]
- NHD 布局：
  - Q: [batch, qo_len, num_qo_heads, head_dim]
  - K/V: [batch, kv_len, num_kv_heads, head_dim]
- 维度约束：
  - head_dim 原始维度属于 (0, 128]，内部 pad 到 64 或 128；
  - num_qo_heads 能够被 num_kv_heads 整除；
  - causal 模式下要求 qo_len == kv_len。

## 算子实现

### 实现方案

#### host 侧设计

##### 分核策略

- 满核利用原则：面向 950PR 标称 32 AI Core（兼容 28 核减配档）进行满核负载均衡。
- 任务网格划分：将 (batch_size, num_kv_heads, q_blocks) 映射为一维任务流，按核均匀分配。
- GQA 聚合优化：将 Hq / Hkv 个 Q 头在 M 维聚合成批，保证单核 M 维分块达到 64 整数倍，打满 MMAD 算力。
- 双 Die 亲和调度（DieTask）：在多 Die 场景下优先保证单核任务在本 Die 内部完成，降低跨 Die 互联流量。

##### 数据分块和内存优化策略

- 片上存储预算（严格受 RES-01 资源台账校验）：
  - L1 Bank0 Q 区：常驻当前 Q tile 与 scale，上限 256KB；
  - L1 Bank1 KV 区：K（INT8）与 V（FP8）双缓冲 ping-pong 区，上限 256KB；
  - L0C 累加器：用于 QK^T（INT32）与 PV（FP32）临时乘积累加，上限 256KB；
  - V0 UB 区：输入暂存双缓冲、smooth 扣减与量化中间区、ND2NZ 重排区，上限 256KB；
  - V1 UB 区：O 主累加器（FP32，64KB 常驻）、FixPipe 接收区与 softmax exp2 结果区（64KB）、两级累加归约区（128KB），上限 256KB。
- 512B 访存对齐法则：
  - innermost 维度与分块 stride 均按 512B 整倍数对齐，触发 MTE2/MTE3 full-burst 突发读写；
  - scale 数组存储按 512B line 边界对齐，消除跨 cache line 拆包。

##### tilingkey 规划策略

- 32 位整型 tiling_key 编码（M5 起唯一定义 `./op_kernel/sage_attention2_template_tiling_key.h`）：
  - bits 20-23: input_dtype（1=fp16, 2=bf16；常量 kTplDtypeFp16/kTplDtypeBf16，归一 NormDtypeMode 单点）
  - bits 0-19: 预留恒 0——gran/accum/layout/causal/pad 不进 tiling key（位型无关路径全部编译期共享，无对应核内分派）
- M5 已落地双形态：
  - canonical 模板档位出包（ASCENDC_TPL_ARGS_DECL + GET_TPL_TILING_KEY，competitor 同款单头模式）：框架按 tiling key 直连模板实例 `sage_attention2<INPUT_DTYPE>`，零运行时位型分派；
  - 裸 bisheng 门禁（无 template_argument.h，回落编码 `TplTilingKey`）：核内 `SageAttnLaunch::RunByDtype` 按 `tiling.input_dtype` 在任务 prologue 一次分派到模板实例，热路径零位型分支；两形态同档位实例执行的算术逐位一致。

#### kernel 侧设计

- 1C2V 协同流水：
  - Cube 引擎：负责 INT8 MMAD（QK^T）与 FP8 MMAD（PV），由 CATLASS 矩阵乘模板驱动；
  - CATLASS 接入边界（2026-10-03 收敛）：分形常量唯一定义源为官方 cann/catlass 仓
    `<catlass/catlass.hpp>`（sage_tile.hpp include 直引派生，构建接线见
    cmake/SageAttnDeps.cmake）；LoadData2D/Fixpipe 数据通路因本算子 L1 为 ND 行布局
    + Data/Scale 相邻契约（与官方 TileCopyTla 的 zN 前置不匹配），保持 AscendC 自研，
    为任务书"按 Cube/CATLASS 微调分块"授权范围；
  - 官方组件复用清单（2026-10-03 复用度盘点）：
    - `Gemm::Tile::TileMmadTla<Ascend950>`：INT8 QK^T 与 FP8 PV 两条 MMAD 发射单点
      （sage_qk_mmad.hpp / sage_pv_mmad.hpp Compute，取代原手写 ExecuteTileMmad）；
    - `tla::MakeTensor/MakeLayout/MakeShape/MakeStride` + `Arch::PositionL0A/L0B/L0C`：
      L0 三级张量代数描述（两执行器 Compute）；
    - `Arch::Ascend950` 硬件常量（L1/L0A/L0B/L0C 容量）：HardwareResourceQuota 唯一映射源；
    - 分形/MTE 常量 `BYTE_PER_FRACTAL`（→LOADDATA2D_REPEAT_BYTES）、`BYTE_PER_BLK`
      （→32B 对齐）、`C0_NUM_PER_FRACTAL`（→Fixpipe zN 分形行粒度）；
    - `detail/alignment.hpp` 全局组件 `::RoundUp` / `::CeilDiv` / `::Min`（Data+Scale
      相邻对齐、LoadData2D repeat 计数、因果活跃块上限；标量 min 不再自造）。
  - 官方组件复用清单（2026-10-03 深化接入，全部为位元等价的 constexpr/forceinline 替换）：
    - `Arch::CrossCoreFlag` + `CrossCoreSetFlag/WaitFlag<MODE,PIPE>`
      （arch/cross_core_sync.hpp）：1C2V 14 flag 跨核同步原语委托官方薄封装单点
      （sync_guard.h；modeId=4 块内 + PIPE_S 语义不变，forceinline 直译 AscendC
      原语、指令流零变化）；
    - MMA 累加位型与 FP8 L0 寄存器位型直接显式书写（轮 5 复审后口径）：Traits
      ElementC 取朴素 `int32_t`（QK int8×int8→INT32）/ `float`（PV fp8×fp8→FP32），
      PV Compute 的 L0A/L0B 张量按 `AscendC::mx_fp8_e4m3_t` 直承载（位元等价于
      原 `Gemm::helper::ElementAccumulatorSelector` / `GetL0Element<float8_e4m3_t,
      true>` 官方表接入，2026-10-04 轮 5 撤销模板间接层；两 helper 已不在复用面）；
    - `detail/macros.hpp` 设备函数宏 `CATLASS_DEVICE_SIMD_VF`（3510 档
      `__simd_vf__ static`，与官方 950 FA softmax epilogue 同款）：op_kernel/
      sage_device_vf.h 全部 VF 原语的标记单点（2026-10-04 曾短暂 undef 重定义为
      `__aicore__ inline` 规避疑似 BiSheng 崩溃，经复审确认真因是 64 位变量移位、
      与本宏无关后恢复官方原值——本宏禁止再本地重定义）；
    - host tiling（aclnn/sage_attn_host_tiling.h）与设备核（op_kernel）全链路的
      分块计数/16-token 对齐改写为 `::CeilDiv` / `::RoundUp`（与手写
      `(a+b-1)/b` 位元一致；floor 语义的尾元素块号显式保留整除并注释禁改）；
    - 构建对齐：全部设备编译路径（run.sh kernel、
      build_opp.sh、tools/sim 设备步）统一注入官方架构选择宏
      `-DCATLASS_ARCH=3510`。
  - 官方组件复用清单（2026-10-03 第 4 轮深化接入，位元等价替换 + 重复造轮子收敛）：
    - `tla::MakeLayout/MakeShape/MakeStride` + `tla::crd2offset`（tla/layout.hpp）：
      GM 寻址布局 tla 单点 `Catlass::SageAttn::MakeGmBhsdLayout`（(B,H,S,D) 四维，
      HND/NHD 双模式，int64 元素步距——Q/K/V 输入与 O 输出共用；核内原 8 处
      `if (LAYOUT_HND)` 手写步距四元组全部删除，坐标折叠 b·sb+h·sh+s·ss 与手写式
      位元一致，test_tiling_struct Test 5 锁值）与 `MakeGmBhsLayout`（LSE/
      lseCorrection 的 (B,HQ,SQ) 稠密布局）；
    - `tla::MakeLayoutL0C`：L0C zN 分形布局官方单点（两执行器 Compute 的 layout_c，
      取代手写行主 MakeLayout 装配；TileMmadTla 显式 (m,n,k) 重载仅消费
      tensor.data()，位元零变化）；
    - 事件对/搬运样板收敛（本算子侧单点，参数化自 AscendC 原语）：核内全部
      SetFlag/WaitFlag 直配对（含原 4 个 no-op 宿主包装）收敛至
      `Catlass::SageAttn::Sync<HardEvent>` 单点；宿主 HardEvent 影子枚举编号逐项
      镜像 dav-3510 官方 kernel_event.h（消除双域枚举漂移）；3 处 UB→L1 行搬
      DataCopyParams 样板收敛至 `CopyUbRowToL1`（blockLen 计量 = 官方
      BYTE_PER_BLK）；Mmad 执行器恒 0 的 L0 基址 / L0C 源偏移 / S 槽第二半址死参
      全部删除。
  - 官方组件与微架构深度复用清单（2026-10-04 第 5 轮微架构与硬件原语深度复用）：
    - `AscendC::MicroAPI` 原生硬件融合指令（`ExpSub` / `MulDstAdd`）：
      在 `BlockMaxAndSumRowsVF`、`SoftmaxProbRowsVF`、`UpdateOnlineSoftmaxRowsVF`、
      `MergePvPartialRowsVF` 与 `FinalizeRowVF` 中全面落地；采用 `ExpSub` 将自然域
      投影与指数求和熔断为单拍硬件本征指令；严格对齐底层 `MulDstAdd(dst, src0, src1)`
      硬件语义（`dst = dst * src0 + src1`，vmadd），单拍闭环指数累加、Online 更新与
      部分和并入，消灭中间寄存器与 RAW 依赖延迟；
    - 片外 L2 Cache 暂存条带 StageStrip 通路（A7 架构闭环）：
      Vector 0 经 MTE3（`DataCopy`）将行/块/通道解析的 Q/K/V scale 写至 GM StageStrip（按核划分 64KB 槽位，在 Die-local 64MB 片上 L2 Cache 驻留），Vector 1 经 MTE2（`DataCopy`）拉入 UB，彻底拔除硬件不支持的 A1->UB 非法通路，保证 100% 硬件兼容与微架构时序保真；
    - `LoadDist::DIST_BRC_B32` UB 标量加载与全局广播随路融合：
      `QuantRowInt8VF` 直接接收 `__ubuf__ float* scale_ptr`，单条指令随路完成加载与
      64 通道广播，彻底消灭标量单元 `GetValue` 回读与独立 `Duplicate` 指令；
    - `FlashUpdateLastNew` 向量化归一化除法与末块写回融合：
      借鉴 CATLASS `FlashUpdateLastNew` 与 `LastDivNew` 模式，在 `FinalizeRowVF` 行首
      直接以 `DIST_BRC_B32` 随路加载 `d_run` 并通过 `Div` 原地求倒，彻底拔除
      `ComputeInvDenominatorVF` 独立全量遍历、`ub_inv_d` 缓冲区及 `LocalMemBar` 屏障；
    - STARS 2.0 硬件 8-Group 拓扑与双 Die 亲和调度（`DieTask` / `DieTaskEnd`）：
      闭式映射实现活跃核与任务空间的偶数双射覆盖，证明 (b,h) 块整对齐定理，确保两 Die
      各自处理独立 KV 批次，KV 数据 100% 命中本 Die 本地片上 64MB L2 Cache，跨 Die
      互联总线流量降为 0；
    - FixPipe `SPLIT_M` 硬件切分模式（`CopyL0CToUbSplitM`）：
      使能 `subBlockId = true` 与 $M > 64$ 双段发射，实现 1C2V 两个 Vector SubCore
      零跨步开销硬件负载均衡；
    - `Catlass::Arch::CrossCoreFlagWithReverse` 硬件队深 15 级防锁死安全防护：
      模 15 深度引入轻量反向确认握手，彻底杜绝单向快速核心踩爆硬件同步队列。
  - 官方组件不采用清单（性能/位级锁定裁定，均有源码级证据）：
    - `Gemm::Tile::CopyL0CToUBTla`（ascend950 特化）：基于 `FixpipeParamsC310` 参数面
      且 nSize 按 32B 粒度 RoundUp；本算子真机 L-A 验证走 `FixpipeParamsArch3510`
      精确 nSize + SPLIT_M 手动双段发射（golden 位级锁定域），不同构不替换；
    - `TileCopyTla` L1→L0A/L0B 特化：EnableIf 强约束 L1 侧 zN 分形布局，与本算子
      ND 行 + Data/Scale 相邻 L1 契约不匹配；
    - `Arch::Resource` / `LocalTensorBuffer`（TPipe InitBuffer）：官方头注释自证
      "Tpipe 初始化会插入同步接口，可能与使用者冲突"——与本核手管 14 flag 流水
      相斥，且零收益，不引入；
    - 竞品（A2 同类算子）"PV 中间结果写回 GM 再读回归并"策略：新增 2 次 GM 往返
      （写 P/部分和 + 读回归并），长序列显存带宽 +50% 量级，直接劣化对 FIA 的
      Speedup 门禁——本算子 L1↔L0↔UB 全片上闭环（零 GM 往返热路径）为其核心
      性能优势，坚决不采用。
  - V0 向量核：负责 MTE2 搬入、INT8/FP8 量化、scale 提取与 ND2NZ 排布转换（7 参收敛后 K 侧均值扣减摘除，核内 km 槽位恒空直通）；
  - V1 向量核：负责 L0C FixPipe 接收、exp2 域 online softmax、行 max 与恒逐块强锚定重缩放、两级累加归约与 LSE 累计；
- 延迟一拍软件流水：
  - Cube 计算与 Vector 计算深度交叠，消灭 Vector 成为关键路径的等待气泡；
- 同步与流水守卫：
  - 采用现役 14 flag 有界缓冲协议（`sync_guard.h` FLAG 0~13）实现核间/核内定向通知，
    按旗标族分治（2026-10-04 首拍死锁修复）：S 槽双缓冲保留进域预置/退域回收配平
    （V1 退域回收消费 Cube 预置的最后两个 S 槽许可）；KV/DOT/PARTIAL 交接旗标首拍
    缓冲区天然为空，生产者跳过首拍等待、不预置不回收（避免跨核伪预置首拍死锁）。
    （历史 RES-05 类型状态守卫 `pipeline_guard.h` 经死代码清扫删除：设备核热路径
    以显式 flag 协议保序，该头文件零生产消费，仅测试引用。）

## 支持硬件

- Atlas 950PR（SoC Version: Ascend950PR，NpuArch=3510），兼容 28 核与 32 核配置。

## 算子约束限制

- 仅支持前向推理计算，不支持反向传播。
- 仅支持 INT8 QK^T + FP8 PV 路径（不支持 FP16 PV 与纯 INT4）。
- 仅支持连续末维（`stride(-1) == 1`）。
- head_dim 原始维度属于 (0, 128]，超过 128 明确报错。
- 不支持任意 `attn_mask` 传入（非空即明确报错，因果模式走 `is_causal=True` 优化掩码）。

# 可维可测分析（required）

## 精度标准/性能标准

### 精度标准

- 遵循《生态算子开源精度标准》experimental_standard 规范；
- L-A 主标杆（GPU `sageattn_qk_int8_pv_fp8_cuda` 同配置）：
  - FP16 混合容差：rtol=2^-9, atol=2^-9, matched_ratio >= 0.99；
  - BF16 混合容差：rtol=2^-6, atol=2^-6, matched_ratio >= 0.99；
- L-B 辅标杆（FP32 SDPA）：相对误差比例 max <= 2.0, mean <= 1.2, RMSE <= 1.2；
- 25 桶全配置覆盖与 `tests/test_accumulator_probe.py` 4 项整数精确域探针保障。

### 性能标准

- 相对 ops-transformer 同仓 FIA 基线（`aclnnFusedInferAttentionScoreV5` 与 `npu_fused_infer_attention_score`）；
- P-01 至 P-07 全部 7 个基准 shape，在 aclnn 接口与 PyTorch 接口两条对比路径上，单 case Speedup 与全量几何平均 Speedup 均达到 1.6 倍以上（Speedup >= 1.6x）。

## 兼容性分析

- 新增独立算子模块，合入路径为 `experimental/attention/sageattention2/`；
- 接口与 SageAttention1/3 保持命名空间隔离，不修改既有 FIA 算子任何实现，保持 100% 独立与生态解耦。
