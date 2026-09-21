# FfnWorkerBatching 四提交需求分析

完整字段说明、执行流程和应用算例见 [arch35 算子介绍与源码导读](ARCH35_OPERATOR_GUIDE.md)。

## 1. 文档定位与版本

| 项目 | 内容 |
| --- | --- |
| 文档版本 | 1.0，2026-09-17 |
| 编写 | Codex，根据已提交实现逆向整理 |
| 状态 | 实现基线说明，供需求与设计评审；不表示已完成人工评审 |
| 基线 | `d967eeb08ceef7c394d0d5e27767e5b08b77aa58`，即本次 `HEAD~4` |
| 目标版本 | `399be5bfcc2c88e6d47c225a9ab82a1539e7dd6e` |
| 配套文档 | [详细设计](DESIGN.md)、[ACLNN V2](aclnnFfnWorkerBatchingV2.md)、[Torch API](torchapi_ffn_worker_batching.md) |

本文以四个提交最终合并后的行为为准，不把中间版本的属性类型、临时头文件或用例数量当作最终规格。外部 `opstest` 用例是补充的验收资产，不属于本仓四提交的 Git 差异。

| 顺序 | 提交 | 需求内容 |
| --- | --- | --- |
| 1 | `f2c3c7d5d9d3f12c2874140cbaf986c8af0417cd` | 异步 RECV、跨层专家编号、MXFP4/MXFP8 搬运、V1/V2 兼容封装、容量与内存边界修正、golden 和测试 |
| 2 | `b2f11facc2df2f5786cdf51e507fba27d2fb53d0` | Torch eager、Meta/FakeTensor、图模式适配、GE converter 与 API 入口 |
| 3 | `b4c5e09d0be92cf06a5483ad7bfad66aad3af2aa` | MX dtype 的 OpDef、GE 原型、推导、Ascend950 注册配置和文档补齐 |
| 4 | `399be5bfcc2c88e6d47c225a9ab82a1539e7dd6e` | `syncFlag/sync_flag` 全链路 bool 化、常量与实现整理、V2 示例、快速集精简；不是仅格式清理 |

修订记录：1.0 首次建立四提交需求、实现、验收的追溯关系。

## 2. 背景、目标与范围

Attention 与 FFN 分离部署时，多个 Attention session 将路由 token 写入 FFN worker 的预分配设备内存。FFN 侧需要把来自不同 session 的 token 按专家组织成连续块，并保留回传所需的来源索引。

原有 arch35 流程已经包含 NORM/RECV、专家排序、分段归并、group_list 和 gather。本次是在此基础上扩展调度与类型能力，并修复泛化边界，不是重新引入整套排序流水线。

目标如下：

1. 同步接收继续等待当前 micro batch 的全部 session；新增异步模式允许消费首个非空 ready 快照，避免被当前空 micro batch 持续阻塞。
2. 异步模式允许一个被选中的 micro batch 中不同 session 属于不同 layer，以跨层专家号区分其 token。
3. 在 Ascend950 上增加 FP8 E5M2、FP8 E4M3FN、FP4 E2M1 及 E8M0 block scale 的原样重排。
4. 保持 ACLNN V1 公共参数布局，通过 V2 暴露 bool 属性，并提供 Torch 与图模式入口。
5. 对大 BS、大 H、大索引空间、非对齐尾块、全 mask 等场景保持确定的输出和受控的 UB/workspace 使用。
6. 仓内保留短时快速回归，外部用例库承担更广泛的功能与形状覆盖。

范围不包括 FFN 矩阵计算、量化/反量化、通信发送、生产者调度器实现、反向传播或任意动态形状图支持。异步指“允许部分 session 就绪”，不是调用后立即返回空结果的 nonblocking 查询。

### 2.1 产品与调用范围

| 能力 | Ascend950 / arch35 | A2/A3 原有路径 |
| --- | --- | --- |
| dtype 0/1/2 与原 NORM/RECV | 保留 | 保留原注册和分发路径 |
| dtype 3/4/5 | 新增支持 | 不扩展 MX 签名 |
| `sync_flag=true` 的 ready 快照及跨层编号 | RECV 生效 | 不承诺具有该新增语义 |
| ACLNN V1 ABI | 保留 | 保留 |
| Torch extension | C++ 明确检查 SoC 名称前缀 `Ascend950` | 不支持本次 Torch 包装 |
| 实机验证结论 | 见第 8 节证据分级 | 不能由 Host UT 推导出已完成实机回归 |

## 3. 术语与符号

| 符号 | 定义 |
| --- | --- |
| `Amax, BSmax, Kmax, H` | `max_out_shape` 四个容量维度 |
| `A, M, BS, K, HS` | 设备 ScheduleContext 中的实际 session 数、micro batch 数、batch size、路由槽数、源 token 行字节步长 |
| `K` | 已包含共享专家槽，即接口描述中的 `topK+1`；本文不再额外加 1 |
| `Y` | 输出容量 `Amax*BSmax*Kmax` |
| `N` | `actual_token_num[0]`，本次有效输出 token 数 |
| `E, L, Ep` | `expert_num`、`layer_num`、异步模式每层专家数 `E/L` |
| `S` | MX 每行 scale 字节数 `ceil(H/32)` |
| `P` | RECV 私有扁平 ID 行跨度 `align_up(BS*K,8)`，单位 int32 元素 |
| ready | 描述符 `flag=1`；不等于其中存在有效 token |
| 快照 | 一次跨 session DMA 实际观察到的 flags 集合，不是所有生产者在同一时刻的原子快照 |

容量属性用于输出分配、tiling 和 workspace 预算；实际源地址与消费长度来自 context。调用方必须使运行时数据规模受容量覆盖、指针缓冲足够大、路由索引合法。Host 不读取并全面验证设备指针指向的内容。

## 4. 接口与数据契约

### 4.1 输入和属性

| 参数 | 类型、默认值 | 约束与语义 |
| --- | --- | --- |
| `schedule_context` | ND、一维 INT8 张量 | 至少 1024 字节；Torch 还要求 NPU 连续张量；前 1024 字节为公共结构 |
| `expert_num` | int64，必填 | `1..8192`；决定 group_list 容量 |
| `max_out_shape` | 4 个 int64，必填 | `1<=Amax<=1024`、`BSmax>0`、`1<=Kmax<=64`、`H>0` |
| `token_dtype` | int64，默认 0 | Ascend950 为 `0..5`，见下表 |
| `need_schedule` | int64，默认 0 | 0=NORM，1=RECV |
| `layer_num` | int64，默认 0 | `0..expert_num`；仅异步 RECV 要求正数且整除 expert_num |
| `sync_flag` / ACLNN `syncFlag` | **bool，默认 false** | 属性序号 5；NORM 不解释，同步/异步差异见 4.4 |
| `mx_output_uint8` | Torch 专属 bool，默认 false | 改变 MX 输出的 Torch 存储视图，不属于 GE/ACLNN 算子属性 |

属性乘积必须在乘法前检查 int64 溢出，`Y<=INT32_MAX`；RECV 补齐后的扁平容量也须 `<=INT32_MAX`。`H<=UINT32_MAX` 且有效行载荷必须能装入一个 512 字节对齐的 uint32 步长。不能把“支持泛化”解释为无表示范围上限。

M 不在 `max_out_shape` 中；现有接口文档给出的使用范围为 `1..64`。context 的 M、polling_index、指针与长度属于调用方协议；不能宣称 Host 已校验所有这些设备侧字段。

### 4.2 token 存储与 scale

| token_dtype | y 逻辑类型 | token 字节数 T | 每行 scale | ACLNN scale shape/type | Torch 原生 y shape/type |
| --- | --- | ---: | --- | --- | --- |
| 0 | FP16 | `2H` | 无有效载荷 | `[Y]` / FP32，占位 | `[Y,H]` / float16 |
| 1 | BF16 | `2H` | 无有效载荷 | `[Y]` / FP32，占位 | `[Y,H]` / bfloat16 |
| 2 | INT8 | `H` | 1 个 FP32，4 字节 | `[Y]` / FP32 | `[Y,H]` / int8 |
| 3 | FLOAT8_E5M2 | `H` | `S` 个 E8M0，`S` 字节 | `[Y,S]` / FLOAT8_E8M0 | `[Y,H]` / float8_e5m2 |
| 4 | FLOAT8_E4M3FN | `H` | 同上 | 同上 | `[Y,H]` / float8_e4m3fn |
| 5 | FLOAT4_E2M1 | `H/2` | 同上，仍每 32 个逻辑元素一个 scale | 同上 | `[Y,H/2]` / float4_e2m1fn_x2 |

FP4 要求 H 为偶数，每字节低半字节为前一个逻辑元素，高半字节为后一个。ACLNN 的 FP4 y 逻辑 shape 仍为 `[Y,H]`，不能把 Torch 的打包列数直接当作 ACLNN 的逻辑列数。

每个源 token 行的有效布局为 `[完整 token 字节][完整 scale 字节][可选行尾 padding]`。scale 不是按 32 元素与 token 交错存储。MX 尾组不足 32 个元素仍占一个 scale。HS 是源行步长，输出 y 与 scale 分开连续存储，不复制 HS 的 padding。

例：FP8、H=33，token=33B，scale=2B；FP4、H=34，token=17B，scale=2B。两者都从 `row_base+T` 开始读取 scale。

### 4.3 八个输出

| 顺序 | 名称 | shape / dtype | 有效内容 |
| --- | --- | --- | --- |
| 0 | y | ACLNN `[Y,H]`，dtype 见上表 | 前 N 行按专家升序重排的 token |
| 1 | group_list | `[E,2]` / int64 | 非空专家的 `[expert_id, token_count]` 紧凑升序排列，剩余行为 `[0,0]` |
| 2 | session_ids | `[Y]` / int32 | token 来自的 session |
| 3 | micro_batch_ids | `[Y]` / int32 | token 来自的 micro batch |
| 4 | token_ids | `[Y]` / int32 | 源 batch 内位置 `b` |
| 5 | expert_offsets | `[Y]` / int32 | **源路由槽位 `k`**，并非该专家输出分组内的累计行偏移 |
| 6 | dynamic_scale | 见 4.2 | 有效类型仅前 N 行；FP16/BF16 仅保证元数据 |
| 7 | actual_token_num | `[1]` / int64 | N，所有非空专家 count 的和 |

NORM 的 session/micro batch 来源于路由表；RECV 的 session 为源描述符行号，micro batch 为本次选中的 selectedMb。同专家顺序的现有 golden 使用稳定排序；验收还比较来源索引，不能仅验证专家计数或把组内顺序任意打乱。

N=0 时仍须写出 `actual_token_num=0`，并清零整个 group_list；不要求初始化 y、四个索引和 scale 的容量尾部。

### 4.4 sync_flag 行为真值表

| need_schedule | sync_flag | 等待与选择 | 专家编号 | 副作用 |
| ---: | --- | --- | --- | --- |
| 0 | false / true | 不读 ready flag，按 out_num 有效前缀 batching | 原始 ID | 不执行 RECV 描述符消费 |
| 1 | false | 始终等入口 polling_index 对应 micro batch 的全部 A 个 session ready | 原始 ID；layer_num=0 合法 | 消费该 micro batch，推进 cursor |
| 1 | true | 从入口 cursor 环形扫描 M 项，选第一次观测到至少一个 ready 的 micro batch | `layer_id*Ep+localExpert` | 仅消费快照选中的 session，推进 cursor |

异步选中后不把其他 micro batch 的 token 合并进来。扫描一整圈仍无 ready 时继续等待，没有“返回 N=0”的轮询超时接口。全 mask 但 ready 的描述符必须消费，此时可以正常返回 N=0。

### 4.5 生产者协议与生命周期

- context 内存放真实设备地址，不能把缓冲相对偏移或其他进程的历史地址直接填入指针字段。
- 描述符为 `flag:int32 + layer_id:int32 + expert_ids[BS*K]:int32`，顺序为 `[A,M]`；flag 合法值为 0/1。
- 生产者先完成 token、scale、layer_id、expert_ids 的写入，再发布 ready，并建立必要的设备/流可见性。
- 消费者使用固定快照。快照未选中的描述符即使稍后 ready，也不得在当前调用中被读取并清理。
- 选中描述符的 IDs 被写成 `INT32_MAX`，flag 清零，layer_id 保留；顺序是先复制私有 IDs，再失效源 IDs，再清 flag。
- flag 清零发生在 gather 完成之前。生产者必须遵循“不覆盖当前消费窗口内 payload”的协议，不能仅观察 flag=0 就覆盖 token_data。
- context、所有二级设备缓冲和 workspace 必须存活至执行完成；图执行/重放时仍需保持地址有效并重新准备已消费的 RECV 数据。
- 不支持多个未排序的消费者同时操作同一 context；本次实现没有跨调用互斥或事务机制。

## 5. 功能需求与验收标准

| ID | 需求 | 可观察的验收标准 | 设计章节 |
| --- | --- | --- | --- |
| FR-01 | 保持 NORM 有效前缀 | 只处理 out_num 行；容量尾部旧 ID 不影响 N、输出与统计 | DESIGN §3、§7 |
| FR-02 | 保持同步 RECV | 部分 ready 时不消费；全部 ready 后处理入口 micro batch，cursor 正确推进 | §6 |
| FR-03 | 非空快照异步 RECV | 跳过空 micro batch；仅消费同一次读取中 ready 的 session；其他行保持不变 | §6、§7 |
| FR-04 | 统一跨核选中项 | 所有输出 micro_batch_ids 和 gather 地址使用 selectedMb，不使用推进后的 cursor | §6 |
| FR-05 | 跨层专家号 | 两层相同 local ID 分到不同专家；非法 layer/local ID 被 mask；源索引不变 | §7 |
| FR-06 | MX 数据搬运 | 六 dtype 输出元数据正确；FP4 nibble、MX scale 和非 32 倍数 H 尾组按字节一致 | §9 |
| FR-07 | 统一 bool 契约 | V2 C++、OpDef、GE、tiling、Torch、CSV 使用 bool/Bool/True/False；V1 不新增入参 | §2、§10 |
| FR-08 | V1 兼容 | 保留公共签名；内部补 false；旧参数不会因插入属性而整体错位 | §2 |
| FR-09 | 排序与统计完整性 | 各段计数和 N 一致；跨段归并、跨块游程、单 token、全 mask、group_list 尾部正确 | §8 |
| FR-10 | 大索引与大步长安全 | 大于 2^24 的源位置不经 FP32 丢位；源地址乘法在首次运算前提升至 int64 | §9 |
| FR-11 | UB 与 workspace 有界 | 大 ID 行、大 token/scale 行可分块；段计数缓存有上限；过小 UB/溢出形状明确拒绝 | §4、§5 |
| FR-12 | Torch eager 与 Meta | 八输出 shape/type 一致；Meta 不读取设备 context；FP4 打包和单元素 stride 正确 | §10 |
| FR-13 | 固定形状图通路 | mutable context 保留；npugraph_ex 的 MX 用 uint8 原始视图；GE 不支持项明确拒绝 | §10 |
| FR-14 | 真实地址测试协议 | 输入生成分配二级 HBM；归档保存载荷并重建地址；比较器拒绝缺输出/shape/type 不匹配 | §11 |
| FR-15 | 快速集与泛化集分工 | 四入口用例均可发现；仓内 30+6+6+6，外部 468+242+12+12；bool 与签名对应 | §11 |

## 6. 非功能需求

### 6.1 正确性与确定性

这是重排与搬运算子。token 和有效 scale 采用二进制一致口径，不引入近似计算容差。排序 key 的浮点表示与地址解码中使用的向量运算属于内部实现，不改变载荷按字节验证的要求。

向量指令按 32/64 元素粒度读取的尾部必须显式初始化，不能以“只写有效输出”代替“全部实际读取 lane 已定义”。跨核依赖使用 SyncAll；本核队列或 `pipe.Reset()` 不替代跨核屏障。

### 6.2 性能与资源

- 快速验证目标为约 5–10 分钟，允许更短；需分别记录已有构建缓存的用例执行时间与构建耗时。
- 该时间是测试流程目标，不是 kernel latency 或吞吐 SLA。
- 本次没有建立与基线的正式性能对比，不声明加速比。
- 核数与可用 UB 从平台查询；对不同 UB 容量推导单段/单块长度，不硬编码某台设备的 UB 大小。
- 只为异步 RECV 额外分配 ready 快照；排序 workspace 随容量增长，使用前必须确认设备内存可用。

### 6.3 兼容与发布

OpDef、公共 wrapper、生成的内部 ACLNN、GE 原型、Host tiling、kernel tiling struct、Torch extension、CSV 应来自同一匹配版本。bool 迁移并不保证与中间 int64 syncFlag 版本二进制混用安全。兼容承诺针对 V1 公共 ABI 和默认行为，不涵盖任意混搭的中间构建产物。

## 7. 验收场景清单

| 场景组 | 必须覆盖的内容 | 测试层次 |
| --- | --- | --- |
| 元数据 | 6 dtype；8 输出；MX scale rank2；FP4 偶数 H/奇数 H 拒绝 | infer UT、ST、Torch Meta |
| 属性 | A/K/E 上界，shape 长度，乘积溢出，Y/补齐长度范围，context 长度，最小 UB | tiling UT |
| 调度 | NORM；同步 all-ready；异步 partial-ready；跳过空 mb；cursor 环回；全 mask ready | ACLNN/直调 ST + 状态型测试 |
| 层号 | 多层混排、非法 layer、越界/负 local ID、mask 加偏移前过滤 | golden/RECV 状态测试 |
| 分块 | prepare 大于一行 UB；多排序段；多归并轮；超 128 段计数；extract 尾块 | UT + 泛化 ST |
| gather | BS*K 非整齐除数、位置超 2^24、源步长乘积超 32 位、token/scale 超 UB | 泛化 ST |
| group_list | 单专家、稀疏专家、大专家数、跨块长游程、N=0/1、剩余行清零 | ST + 比较器 |
| 生命周期 | 真实二级地址、RECV 重置、归档恢复、图重放、未选中描述符不变 | 输入生成/状态型测试 |
| 接口 | V1/V2、直调、eager、npugraph_ex、GE 可支持子集与拒绝路径 | 各入口测试 |

静态 CSV 至少预置一个 ready；不能仅凭 CSV 通过就宣称验证了生产者延迟到达、全空等待或并发发布时序。Torch CSV 的 NORM `sync_flag=True` 验证属性传递与无副作用，不等同于 Torch 异步 RECV 功能验证。

## 8. 用例规模与验证证据

2026-09-17 本次按 CSV 实际数据行重新统计：

| 入口 | 仓内快速集 | 外部 opstest 集 |
| --- | ---: | ---: |
| ACLNN | 30 | 468 |
| Kernel 直调 | 6 | 242 |
| Torch eager | 6 | 12 |
| TorchGraph | 6 | 12 |
| 合计 | 48 | 734 |

仓内位置：[tests/st/arch35](../tests/st/arch35)。外部位置：`/home/chenpengyi/opstest/ffn/ffn_worker_batching/arch35`。外部 README 记录 ACLNN 从 936 减至 468、直调从 484 减至 242，MX generalization 已合入主 CSV，不再独立维护。精简保留覆盖维度，不等于保留完整参数笛卡尔积。

**证据层次必须区分：**

1. 本次直接核验：目标四提交源码、UT/ST 定义、CSV 数量、外部 README 的记录。
2. 外部 README 记录的历史上板结果：Ascend950DT_9574，ACLNN 468/468、直调 242/242、eager 12/12、TorchGraph 12/12，最终合计 734/734。直调首轮 241 通过，1 条输入分配返回 `ACL_ERROR_RT_CONTEXT_NULL (107002)`，单 worker 复跑通过；不能写成首轮全部通过。
3. 历史记录的首轮 1892.075 秒、复跑 33.429 秒，累计约 32 分 6 秒；快速集记录约 4 分 13 秒为各入口分次执行相加，非当前四提交整套连续实测。
4. README 引用的 `reports/20260917/ffn_worker_batching_bool_full/validation.md` 等原始报告在当前工作区未找到。因此本文只转述 README 的历史记录，不据此签署目标 HEAD 已独立重测通过的结论；也不将未取得原始证据的 UT/GE 历史数字列为已核验结果。

本次交付是文档修改，执行文档链接、代码摘录、CSV 元数据和差异检查，不重新声称完成 NPU 上板验收。正式发布需归档对应构建版本、SoC/CANN/Torch 环境、各行状态、精度/OOB 字段及复跑说明。

## 9. 已知契约差异与边界

- 现有 README 对 `expert_offsets` 的文字与 kernel `EmitTokens` 不一致；本需求采用代码实际输出的源槽位 k。
- README 的 Y“无硬上限”需要结合 Host 的 int32 容量限制理解。
- 部分 Torch 说明包含 GE 尚未通过的环境记录；converter 存在不等于全部 dtype/后端组合已验证。FP4 和 MX uint8 视图有显式 GE 拒绝路径。
- 同步/NORM 的 mask 策略沿用 `expert_id>=1000000`；调用方须保证其余有效 ID 属于 `[0,E)`。不能把异步新增的非法 ID 过滤推广为所有路径统一校验。
- 默认属性兼容不等于旧 GE 原型可直接搭配新 Host；converter 即使传 false 也检查新版 sync_flag 原型存在。
- 系统外部生产者可见性、缓冲生命周期、slot 不覆盖前提没有由一个 bool 属性自动解决。
