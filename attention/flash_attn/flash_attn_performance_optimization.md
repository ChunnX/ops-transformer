# Flash Attention 性能优化点扫描报告

> 扫描工具：cann-skills-zyh (CANNBot Skills)
> 扫描对象：`/mnt/workspace/transformer-issue/attention/flash_attn`
> 扫描日期：2026-06-01
> 参考框架：`ascendc-performance-best-practices`、`ops-profiling`、`npu-arch`、`ascendc-tiling-design`、`ascendc-api-best-practices`

---

## 一、流水线与调度优化 (Pipeline & Scheduling)

### 1.1 C1-C1-C2 流水线深度不足

| 项目 | 内容 |
|------|------|
| **现状** | 当前采用 3 级流水线（PRELOAD_N=2）：2 次 BMM1 与 1 次 BMM2 重叠 |
| **文件** | `op_kernel/arch35/flash_attn_kernel_base_gqa_template.h` |
| **问题** | 对于长序列（S2 循环次数多），3 级流水线可能不足以完全隐藏 BMM2 的 Fixpipe 延迟 |
| **建议** | 参考 `ascendc-performance-best-practices` 的 Pingpong 策略，评估将流水线深度增加到 4 级（PRELOAD_N=3），使 BMM1 的 MTE2 加载与 BMM2 的 Fixpipe 完全重叠 |
| **预期收益** | 流水线气泡减少 10-20% |

### 1.2 FlashDecode 串行 Reduce 瓶颈

| 项目 | 内容 |
|------|------|
| **现状** | FlashDecode 阶段 KV split 的 Reduce 是顺序处理的（`flash_attention_noquant_block_vec_flashdecode.h`） |
| **文件** | `op_kernel/arch35/flash_attention_noquant_block_vec_flashdecode.h` |
| **问题** | 当 S2 split 数量较多（多核分担一个 head 的 KV）时，Reduce 循环串行执行，成为瓶颈 |
| **建议** | 考虑树形 Reduce 或分层 Reduce 策略，利用多个 AIV 核并行执行部分 Reduce，再在最终核上合并 |
| **预期收益** | 高 split 场景下 Reduce 延迟降低 30-50% |

### 1.3 FlashDecode 始终开启

| 项目 | 内容 |
|------|------|
| **现状** | `flashDecodeFlag_ = true` 无条件开启（`arch35/flash_attn_tiling.cpp:130`） |
| **文件** | `op_host/arch35/flash_attn_tiling.cpp` |
| **问题** | 对于纯 Prefill 场景（S1 较大、无需 S2 split），FlashDecode workspace 仍然分配并执行空操作 |
| **建议** | 增加条件判断：仅在 S2 split 实际发生时（`s2Size > sInnerFactor * coreNum` 或类似条件）才开启 FlashDecode |
| **预期收益** | Prefill 场景减少 workspace 分配开销和无效 Reduce 计算 |

### 1.4 调度模式单一

| 项目 | 内容 |
|------|------|
| **现状** | `CalcScheduleMode()` 始终返回 `BATCH_MODE` |
| **文件** | `op_host/arch35/flash_attn_tiling.h` |
| **问题** | 不支持 `PIPELINE_MODE` 或多流重叠策略 |
| **建议** | 参考 `model-infer-multi-stream` skill，对 batch 维度引入多流并行，使不同 batch 的 Attention 可在不同 stream 上重叠执行 |
| **预期收益** | 大 batch 场景下整体吞吐提升 |

---

## 二、存储层次与 Buffer 优化 (Memory Hierarchy & Buffer)

### 2.1 L0C Buffer 压力（S1=128, S2=256 配置）

| 项目 | 内容 |
|------|------|
| **现状** | BMM1 结果大小 = 128×256×4B = 128KB，超过 L0C/4 = 64KB |
| **文件** | `op_kernel/arch35/flash_attn_block_cube_noquant_gqa.h` |
| **问题** | 迫使 L0C 使用 2-buffer 模式而非 4-buffer 模式，降低流水线深度 |
| **建议** | 参考 `ops-profiling` 的 CUBE Bound 优化：(1) 对 S1=128/S2=256 配置考虑将 S1 降为 64 以启用 4-buffer；(2) 或使用 L0C 累加模式减少 UB 中转 |
| **预期收益** | 流水线深度翻倍，Cube 利用率提升 15-25% |

### 2.2 大 D 场景 BMM1 的 K 矩阵重复加载

| 项目 | 内容 |
|------|------|
| **现状** | D>256 时 `IterateBmm1NdL1SplitK` 对 K 按 K-tile 从 GM 重复加载 |
| **文件** | `op_kernel/arch35/flash_attn_block_cube_noquant_gqa.h` |
| **问题** | 每个 K-tile 都需要一次 GM→L1 搬运，带宽消耗大 |
| **建议** | 参考 `ascendc-performance-best-practices` 的 FullLoad 策略：(1) 对 K 做 L1 双 buffer 预取，当前 K-tile 计算时预取下一个；(2) 评估 K 驻留 L1 的可行性（K 大小 vs L1 容量） |
| **预期收益** | 大 D 场景 BMM1 带宽瓶颈缓解 20-40% |

### 2.3 GM-resident BMM2 路径的多次 GM 往返

| 项目 | 内容 |
|------|------|
| **现状** | `!bmm2Write2Ub` 时，BMM2 结果经 GM 中转，Vec2 按 `subMBase=8192/dTemplateAlign64` 行分块从 GM 加载 |
| **文件** | `op_kernel/arch35/flash_attn_block_vec_noquant_gqa.h` |
| **问题** | 大 M 时产生多次 GM→UB 搬运，每次都有延迟开销 |
| **建议** | (1) 增大 UB 预算或压缩 D 对齐要求以减少分块数；(2) 参考 `ops-profiling` MTE2 Bound 优化：确保每次搬运 ≥16KB 且 512B 对齐 |
| **预期收益** | GM 往返次数减少，Vec2 延迟降低 15-30% |

### 2.4 Workspace 过度分配

| 项目 | 内容 |
|------|------|
| **现状** | Workspace 公式：`(bmm2Bytes + vec2Bytes) × 3 × coreNum + flashDecode_overhead` |
| **文件** | `op_host/arch35/flash_attn_tiling.cpp:197` |
| **问题** | × 3 系数（2 prefetch + 1 active）对所有场景统一使用；FlashDecode overhead 按 `coreNum × 2 × mSize × dSize × 4B` 计算，大 D=512 时开销巨大 |
| **建议** | (1) 根据实际 pipeline 深度动态调整系数；(2) FlashDecode workspace 仅在实际需要 S2 split 时分配 |
| **预期收益** | 减少 GM 占用，提升 L2 Cache 命中率 |

### 2.5 L2 Cache 未显式配置

| 项目 | 内容 |
|------|------|
| **现状** | 未发现 L2 CacheMode 配置或 L2 partition 设置 |
| **文件** | 全局 |
| **问题** | 参考 `ops-profiling` 的 L2 Cache 优化：缺少 CacheMode 设置可能导致频繁 L2 miss |
| **建议** | (1) 对 Q/K/V 的 GM 地址设置 L2 CacheMode；(2) 对 workspace 中间数据考虑 disable cache 或 L2 partition |
| **预期收益** | L2 命中率提升，GM 带宽压力降低 |

---

## 三、Tiling 策略优化 (Tiling Strategy)

### 3.1 sOuter/sInner 选择启发式有限

| 项目 | 内容 |
|------|------|
| **现状** | 仅 2 种配置：(64,128) 和 (32,256)，由 `AdjustSinnerAndSouter()` 简单条件分支选择 |
| **文件** | `op_host/fa_adjust_sinner_souter.h` |
| **问题** | 未考虑 GQA group size、实际可用核数、D 大小对 working set 的综合影响 |
| **建议** | 参考 `ascendc-tiling-design` 的 4 要素方法论：(1) 多核分配策略需结合 GQA group 维度；(2) UB 分区策略需考虑 D 对齐后的实际 UB 占用；(3) 增加 (64,256) 或 (32,128) 等更多配置选项 |
| **预期收益** | 更多场景可命中最优 tiling 配置 |

### 3.2 FlashDecode M 轴均衡粒度硬编码

| 项目 | 内容 |
|------|------|
| **现状** | `fdBalanceMBaseSize=8` 硬编码 |
| **文件** | `op_kernel/arch35/flash_attention_noquant_block_vec_flashdecode.h` |
| **问题** | 8 行/子任务对所有序列长度和核数不一定最优 |
| **建议** | 根据 `s1Size / aivNum` 动态计算均衡粒度，确保各 AIV 核负载均衡（差异 <10%） |
| **预期收益** | 核间负载均衡性提升 |

### 3.3 模板变体覆盖不足

| 项目 | 内容 |
|------|------|
| **现状** | 48 种 AOT 变体（4 layout × 3 kvLayout × 2 mask × 2 config） |
| **文件** | `op_kernel/arch35/flash_attn_template_tiling_key.h` |
| **问题** | config 仅 2 种（S1=128/S2=128 和 S1=64/S2=256），未覆盖 D>128 的专用 tiling |
| **建议** | 增加 D-aware 的 config 维度（如 D=128/256/512 分别优化），或引入运行时 tiling 参数化 |
| **预期收益** | 大 D 场景可获得专用优化路径 |

---

## 四、计算与指令优化 (Computation & Instruction)

### 4.1 Softmax 状态 Buffer 索引不一致风险

| 项目 | 内容 |
|------|------|
| **现状** | `softmaxExpBuf` 用 `loop % 3` 索引，`softmaxMaxBuf/SumBuf` 用 `mloop % 3` 索引 |
| **文件** | `op_kernel/arch35/flash_attn_block_vec_noquant_gqa.h` |
| **问题** | 当 S2 循环存在 skip（稀疏 mask 跳过）时，`loop` 和 `mloop` 可能不同步，导致状态错乱 |
| **建议** | 统一使用 `mloop`（实际执行计数）索引所有 softmax 状态 buffer，或引入显式状态版本号 |
| **预期收益** | 消除稀疏场景下的正确性风险，避免防御性同步开销 |

### 4.2 Mask 生成可程序化替代 GM 加载

| 项目 | 内容 |
|------|------|
| **现状** | Causal/Band mask 从 GM 加载后经 `MergeBand` SIMD 操作合并 |
| **文件** | `op_kernel/utils/attenmask_gs1.h` |
| **问题** | 标准 causal mask 模式是可预测的，从 GM 加载浪费带宽 |
| **建议** | 对标准 causal mask，在 UB 中程序化生成 mask 模式（利用行/列索引关系），避免 GM→UB 搬运 |
| **预期收益** | Mask 加载带宽降为零，Vec1 延迟降低 5-15% |

### 4.3 Vec1 尺寸特化路径可扩展

| 项目 | 内容 |
|------|------|
| **现状** | Vec1 有 4 个尺寸特化分支：EQ_128、GT_0_AND_LTE_64、GT_64_AND_LTE_128、GT_128_AND_LTE_256 |
| **文件** | `op_kernel/arch35/flash_attn_block_vec_noquant_gqa.h` |
| **问题** | `likely(s2Size==128)` 快速路径仅覆盖 S2=128，对于 S2=64/256 无专用优化 |
| **建议** | 为最常用的 S2 尺寸（64、128、256）分别编写完全展开的 VF kernel，消除循环开销 |
| **预期收益** | 热路径指令效率提升 |

### 4.4 BMM1/BMM2 的 L0 Split 策略选择

| 项目 | 内容 |
|------|------|
| **现状** | 根据 S2 大小选择 `MatmulK` 或 `MatmulN`，大 D 使用 `MatmulFull` |
| **文件** | `op_kernel/arch35/flash_attn_block_cube_noquant_gqa.h` |
| **问题** | L0 split 策略未考虑 L0A/L0B 的实际利用率和 Bank Conflict |
| **建议** | 参考 `ops-profiling` 的 Bank Conflict 优化：(1) 调整 UB 地址 padding（32B 倍数）；(2) 修改 repeatStride/blockStride 避免 Bank 冲突 |
| **预期收益** | Bank Conflict 率降低至 <5% |

---

## 五、数据搬运优化 (Data Transfer)

### 5.1 Attention Mask 搬运优化

| 项目 | 内容 |
|------|------|
| **现状** | 已有 `IsSkipAttentionmask` 跳过全零 mask tile 的优化 |
| **文件** | `op_kernel/utils/attenmask_gs1.h` |
| **问题** | 对于非全零但稀疏度高的 mask tile，仍然加载完整数据 |
| **建议** | (1) 引入稀疏度阈值，高稀疏 tile 用程序化生成替代 GM 加载；(2) 参考 `ascendc-performance-best-practices` DataCopy 优化：确保 mask 搬运 ≥512B 对齐 |
| **预期收益** | 稀疏 mask 场景带宽节省 20-50% |

### 5.2 Q 矩阵驻留优化已实现但可增强

| 项目 | 内容 |
|------|------|
| **现状** | Q 在首次 S2 循环加载到 L1 后驻留，后续 S2 迭代复用（`isFirstS2Loop` 守卫） |
| **文件** | `op_kernel/arch35/flash_attn_block_cube_noquant_gqa.h` |
| **问题** | Q 驻留策略未区分 D 大小：D>256 时 Q 占用 L1 过大，挤压 K/V 空间 |
| **建议** | D>256 时考虑 Q 分片驻留（部分 Q 驻留 L1，部分按需加载），或降低 S1 tile 大小 |
| **预期收益** | 大 D 场景 L1 利用率更均衡 |

### 5.3 输出写回的对齐与合并

| 项目 | 内容 |
|------|------|
| **现状** | `CopyOutAttentionOut` 按 layout 分别处理 BSND/BNSD/TND 的 GM 写入 |
| **文件** | `op_kernel/arch35/flash_attn_block_vec_noquant_gqa.h` |
| **问题** | 小 M tail 块可能产生非对齐的小粒度 GM 写入 |
| **建议** | 参考 `ascendc-performance-best-practices` Tail Block 优化：(1) 使用 `DataCopyPad` 对齐到 512B；(2) 合并多个小 tail 写入为一次批量搬运 |
| **预期收益** | 尾块写入带宽利用率提升 10-30% |

---

## 六、标量开销优化 (Scalar Overhead)

### 6.1 TilingData 结构体过大

| 项目 | 内容 |
|------|------|
| **现状** | `FlashAttnTilingData` 包含 `FlashAttnBaseParams` + `FlashAttnAttenMaskParams` + `FlashAttnPageAttentionParams` + `FlashAttnWorkspaceParams` + `FlashAttnMetaData`（含 `FAMetadata[36][16]` + `FDMetadata[72][16]`） |
| **文件** | `op_kernel/arch35/flash_attn_tiling_data.h` |
| **问题** | Metadata 数组按最大核数预分配（36 AIC + 72 AIV），实际核数可能远小于此；TilingData 过大会增加 Scalar 寄存器压力 |
| **建议** | 参考 `ascendc-performance-best-practices` Scalar 优化 P1/P4/P8：(1) 将 Metadata 改为动态偏移访问；(2) 将编译期常量用 `constexpr`/模板参数传递 |
| **预期收益** | TilingData 缩小 30-50%，Scalar 寄存器压力降低 |

### 6.2 ConstInfo_t 多重继承链

| 项目 | 内容 |
|------|------|
| **现状** | `ConstInfo_t<NO_QUANT>` 通过多重继承聚合 PA、LSE、Sink、PSE、TensorList、PostQuant、LeftPadding、SysPrefix 等子结构 |
| **文件** | `op_kernel/utils/flash_attn_common_def.h` |
| **问题** | 参考 Scalar 优化 P7：过大结构体（>64B）跨 cacheline 访问；P6：多级指针解引用 |
| **建议** | (1) 将热路径访问的字段提到结构体前端；(2) 冷字段（PA、Sink 等可选功能）分离到独立指针 |
| **预期收益** | 热路径 Scalar 访问延迟降低 |

### 6.3 任务环形缓冲区索引优化已实现

| 项目 | 内容 |
|------|------|
| **现状** | `taskRunInfo[PRELOAD_TASK_CACHE_SIZE=4]` 使用 `loop & 3` 位掩码索引 |
| **文件** | `op_kernel/arch35/flash_attn_kernel_base_gqa_template.h` |
| **评价** | **已优化** — 使用 2 的幂次大小 + 位运算替代取模，符合 Scalar 优化最佳实践 |

---

## 七、API 与 Host 侧优化 (API & Host-Side)

### 7.1 Dummy Tensor 重复创建/销毁

| 项目 | 内容 |
|------|------|
| **现状** | `returnSoftmaxLse==0` 时每次调用都 `aclCreateTensor` + `aclDestroyTensor` 创建/销毁 dummy tensor |
| **文件** | `op_api/aclnn_flash_attn.cpp:48-59`、`op_api/aclnn_flash_attn_inner.cpp` |
| **问题** | Host 侧 malloc/free 开销，推理场景（LSE 通常不需要）每次调用都触发 |
| **建议** | (1) 使用线程局部静态 dummy tensor 避免重复分配；(2) 或在 executor 层缓存 dummy tensor |
| **预期收益** | Host 侧开销降低，推理 QPS 提升 |

### 7.2 8 个 Optional Input 的 Dummy 分配

| 项目 | 内容 |
|------|------|
| **现状** | L0 层对 8 个可选输入各做一次 `executor->AllocTensor()` |
| **文件** | `op_api/flash_attn.cpp:50-73` |
| **问题** | 每次算子调用 8 次元数据分配 |
| **建议** | 批量分配或使用预分配的 dummy tensor pool |
| **预期收益** | Host 侧分配开销降低 |

### 7.3 softmaxLse 输出始终分配

| 项目 | 内容 |
|------|------|
| **现状** | 即使 `returnSoftmaxLse==0`，L0 层仍分配 softmaxLse 输出 tensor |
| **文件** | `op_api/flash_attn.cpp:76` |
| **问题** | 推理场景浪费 device 内存 |
| **建议** | 条件分配：仅在 `returnSoftmaxLse != 0` 时分配 |
| **预期收益** | 减少 device 内存占用 |

### 7.4 GE 图模式注册缺失

| 项目 | 内容 |
|------|------|
| **现状** | `flash_attn_proto.h` 中 `REG_OP(FlashAttn)` 被注释掉 |
| **文件** | `op_graph/flash_attn_proto.h` |
| **问题** | 算子无法在静态图模式下使用，限制了图融合优化机会 |
| **建议** | 完成 GE 图注册，使算子可参与图编译器的算子融合（如与上游 LayerNorm、下游 Add 的融合） |
| **预期收益** | 图模式下可获得跨算子融合收益 |

---

## 八、架构感知优化 (Architecture-Aware)

### 8.1 DAV_3510 架构锁定

| 项目 | 内容 |
|------|------|
| **现状** | `GetNpuInfo()` 硬拒绝非 DAV_3510 架构 |
| **文件** | `op_host/flash_attn_tiling_info_parser.cpp:128` |
| **问题** | 无法利用 DAV_3510 的新特性（FP8 Cube、SIMT、RegBase、NDDMA） |
| **建议** | (1) 参考 `npu-arch` skill：DAV_3510 有 256KB L0C（vs 128KB）、248KB UB（vs 192KB）、4KB BT（vs 1KB）；(2) 评估 RegBase (SIMD-Regbase) 编程模型用于 Vector 核的可行性 |
| **预期收益** | 充分利用 DAV_3510 硬件能力 |

### 8.2 AIC/AIV 核心比固定

| 项目 | 内容 |
|------|------|
| **现状** | `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)` 固定 1:2 比例 |
| **文件** | `op_kernel/flash_attn.cpp` |
| **问题** | 不同场景（Prefill vs Decode）对 Cube/Vector 计算量比不同 |
| **建议** | 参考 `ops-profiling` 的 AIC:AIV 比例优化：(1) Prefill（大 S1）Cube 占比高，可调为 1:1；(2) Decode（S1=1）Vector 占比高，维持 1:2 或 1:3 |
| **预期收益** | 参考案例：GroupedMatmul 通过 AIC:AIV 比例调整获得 41% 提升 |

### 8.3 Cross-Core 同步开销

| 项目 | 内容 |
|------|------|
| **现状** | AIC→AIV 数据传递使用 `SetCrossCore`/`WaitCrossCore` 通过共享 UB |
| **文件** | `op_kernel/arch35/flash_attn_kernel_base_gqa_template.h` |
| **问题** | 每次 BMM1→Vec1、BMM2→Vec2 都需要跨核同步 |
| **建议** | (1) 评估 DAV_3510 的 BufferID 同步机制替代 set/wait event；(2) 增大单次传递数据量以减少同步次数 |
| **预期收益** | 跨核同步延迟降低 |

---

## 九、正确性与防御性优化 (Correctness & Defensive)

### 9.1 int64→int32 静默截断

| 项目 | 内容 |
|------|------|
| **现状** | Layer 2 (int64) → Layer 0 (int32) 参数传递存在隐式窄化转换 |
| **文件** | `op_api/flash_attn.h` vs `op_api/aclnn_flash_attn.h` |
| **问题** | 超过 INT32_MAX 的值会被静默截断 |
| **建议** | 添加显式范围检查和错误返回 |

### 9.2 IsEmptyInput 死代码

| 项目 | 内容 |
|------|------|
| **现状** | `IsEmptyInput()` 始终返回 false，为空函数 |
| **文件** | `op_host/flash_attn_tiling.cpp:34` |
| **建议** | 实现真正的空 tensor 短路逻辑，或移除该函数 |

---

## 优化优先级总览

| 优先级 | 分类 | 优化点 | 预期收益 | 实现难度 |
|--------|------|--------|----------|----------|
| **P0** | 流水线 | 1.1 C1-C1-C2 流水线深度增加 | 高 | 中 |
| **P0** | 存储 | 2.1 L0C Buffer 4-buffer 模式 | 高 | 中 |
| **P0** | 架构 | 8.2 AIC/AIV 比例动态调整 | 高 | 低 |
| **P1** | 存储 | 2.2 大 D 场景 K 预取 | 中高 | 中 |
| **P1** | 计算 | 4.2 Mask 程序化生成 | 中 | 低 |
| **P1** | Tiling | 3.1 sOuter/sInner 启发式增强 | 中 | 中 |
| **P1** | 流水线 | 1.3 FlashDecode 条件开启 | 中 | 低 |
| **P2** | 存储 | 2.3 GM-resident BMM2 路径优化 | 中 | 高 |
| **P2** | 存储 | 2.5 L2 Cache 显式配置 | 中 | 低 |
| **P2** | 计算 | 4.4 Bank Conflict 优化 | 中 | 中 |
| **P2** | API | 7.1 Dummy Tensor 缓存 | 低 | 低 |
| **P2** | API | 7.4 GE 图模式注册 | 中 | 高 |
| **P3** | 标量 | 6.1 TilingData 结构体瘦身 | 低 | 中 |
| **P3** | 搬运 | 5.3 输出写回对齐合并 | 低 | 低 |
| **P3** | 流水线 | 1.2 FlashDecode 并行 Reduce | 中 | 高 |

---

## 附录：已有优化点（Positive Findings）

以下列出代码中已实现的性能优化最佳实践：

| 优化点 | 位置 | 说明 |
|--------|------|------|
| Q 矩阵 L1 驻留 | `flash_attn_block_cube_noquant_gqa.h` | Q 仅首次 S2 循环加载，后续复用 |
| 位掩码索引 | `flash_attn_kernel_base_gqa_template.h` | `loop & 3` 替代 `loop % 4` |
| 稀疏 Mask 跳过 | `attenmask_gs1.h` | 全零 mask tile 跳过 GM 加载 |
| NZ 格式 L1 存储 | `flash_attn_block_cube_noquant_gqa.h` | Ascend NZ 分形格式最大化 Cube 利用率 |
| AOT 模板编译 | `flash_attn_template_tiling_key.h` | 48 变体预编译消除运行时 JIT 开销 |
| 尺寸特化 Vec1 | `flash_attn_block_vec_noquant_gqa.h` | `likely(s2Size==128)` 快速路径 |
| Double Buffer Mask | `flash_attn_block_vec_noquant_gqa.h` | `attenMaskInQue[2]` 双缓冲 |
| Metadata 预计算 | `flash_attn_tiling_data.h` | AICPU 预计算 per-core 工作分配 |
| dc_preload 提示 | `flash_attn.cpp` | TND 布局的 cuSeqLens 预取到 L2 |
| Cross-Core UB 共享 | `flash_attn_kernel_base_gqa_template.h` | AIC/AIV 通过共享 UB 传递数据 |
