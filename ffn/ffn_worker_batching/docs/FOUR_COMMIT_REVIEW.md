# FfnWorkerBatching 最近四个提交检视与注释说明

检视日期：2026-09-17。结论针对 `399be5bfcc2c88e6d47c225a9ab82a1539e7dd6e`，与基线 `d967eeb08ceef7c394d0d5e27767e5b08b77aa58` 对比，并追踪关联的旧 A2/A3 实现。

本次执行内容是检视、补充解释性注释和验证注释不改变执行代码。下述功能问题尚未修复，不能把此文档或新增注释视为修复提交。工作区原有 README 修改及 DESIGN.md、REQUIREMENTS.md、ARCH35_OPERATOR_GUIDE.md 均保留。

## 1. 提交范围与整体判断

| 提交 | 主要内容 | 检视重点 |
|---|---|---|
| `f2c3c7d5d` | 异步接收、MX 数据搬运、跨层专家、容量与排序修正、V1/V2 适配 | 发布顺序、消费集合、payload 生命周期、实际长度与容量、UB 预算 |
| `b2f11facc` | Torch eager/Meta/GE 接口及说明 | 输出 dtype/shape、FP4 打包、可变 context、设备 guard、属性 ABI |
| `b4c5e09d0` | MX 类型注册与推导、接口文档 | 六组类型签名、E8M0 scale、逻辑 H、推导边界 |
| `399be5bfc` | sync_flag 全链路 int64→bool、V2 示例新增、快速集精简及代码清理 | 布尔属性 ABI、重构后协议一致性、测试是否证明所述行为 |

净变更涉及 37 个文件。总体调用链为 Torch → ACLNN V2 → 内部 ACLNN → Host tiling → arch35 context/prepare/sort/merge/extract/group/gather；GE converter 直接生成图节点，不经过 eager C++ wrapper。

主要算法改动与当前设计基本一致。已确认的新问题集中在两个 Host/接口整数边界；原有 A2/A3 路径仍有地址乘法回绕和输入长度防御缺口。异步状态回归测试与公开文档也存在缺失，不能只依据数值输出比较宣布全部行为通过。

## 2. 最近四提交引入的确定问题

### R1：Torch C++ wrapper 在下游校验前计算溢出的输出行数（P2）

- 位置：`torch_extension/csrc/ffn_worker_batching.cpp`，`const int64_t y_rows = a * bs * k`。
- 来源：`b2f11facc` 新增接口。
- 触发：`max_out_shape=[2, INT64_MAX, 1, 2]`，使用非 MX dtype、合法 context 元数据。当前维度校验只对 BS 要求正数，这组值能够到达乘法。
- 原因：`a*bs` 在调用 ACLNN 之前已经发生有符号溢出。后续 arch35 tiling 的乘法前检查不能保护 wrapper 中更早的表达式；Python 的任意精度整数也不改变 C++ 的行为。
- 后果：在给出明确参数错误前发生未定义行为，输出申请尺寸可能错误。这里不宣称已复现 NPU 错误写入。
- 建议：C++ 在任何乘法或输出分配前检查 `a <= INT64_MAX/bs`、`a*bs <= INT64_MAX/k`，并提前拒绝超过 kernel int32 索引容量的 Y；Meta 同步保持参数接受范围。优先以共享约束减少三处校验漂移。
- 验证：独立提取相同表达式，用 UBSan 执行 `2 * INT64_MAX * 1`，得到 signed integer overflow；未运行整个 Torch/NPU 接口。

### R2：MX scale 的 shape 推导先执行 `H+31`，可有符号溢出（P2）

- 位置：`op_host/ffn_worker_batching_infershape.cpp`，`(H + MXFP_SCALE_GROUP - 1) / MXFP_SCALE_GROUP`。
- 来源：`b4c5e09d0` 增加 MX shape 分支。
- 触发：`max_out_shape=[1,1,1,INT64_MAX]`，`token_dtype=3/4`。Y 的保护通过，H 只在此阶段校验为正。
- 原因：推导先于后续 tiling 的 `H <= UINT32_MAX` 校验；先加再除并不因为最终结果可表示就安全。
- 建议：与 C++ wrapper 一致改为 `H/32 + (H%32 != 0)`，并明确 InferShape 阶段应该拒绝的 H 上界。
- 验证：独立 UBSan 表达式探针报错。该问题是非法超大输入的拒绝路径缺陷，不意味着普通 H 的 MX scale shape 计算错误。

## 3. 原有实现中仍存在的问题

### R3：A2/A3 Gather 的维度乘法先在 uint32 中回绕（P2，基线问题）

- 位置：`op_kernel/ffn_wb_gather_out_all.h` 的 `strideSession`、`strideMicroBatch`、`strideBs` 计算。
- 原因：`M/BS/K/HS` 都是 uint32，右侧乘法先按 uint32 运算，赋给 int64 时丢失的高位不会恢复。
- 反例：`M=64, BS=64, K=64, HS=16384`，正确 session stride 为 `4294967296` 字节，旧表达式得到 `0`。session 大于零时会取错源地址。可令 H=8192、FP16，与 HS=16384 匹配。
- arch35 版本已在首次乘法前提升至 int64，但旧 A2/A3 文件未同步修复。
- 建议：在每条维度连乘的第一个操作数处显式提升，补地址公式边界测试。该测试可验证偏移计算，无需分配完整多 GiB 的设备数据。
- 验证：独立 C++ 探针输出 `old=0 wide=4294967296`。未运行大容量 NPU 用例。

### R4：A2/A3 入口未检查 context 至少有 1024 字节（P2，基线问题）

- Host 位置：`op_host/ffn_worker_batching_tiling.cpp::CheckInputParam`，检查 INT8、rank=1 后成功返回。
- Kernel 位置：`op_kernel/ffn_wb_get_schedule_context.h::ScheduleContextInfoCompute`，固定读取 `STRUCT_LENGTH=1024` 字节。
- 触发：直接以小于 1024 字节的一维 INT8 context 进入旧架构路径。
- 风险：拒绝输入之前读取超出声明容量的内容。新 arch35 的长度检查和 Torch wrapper 的检查都不保护直接调用旧架构入口的情形。
- 建议：旧 Host 路径补齐最小长度校验，增加 0/1/1023 字节失败、1024 字节通过的 UT。与设备 context 内二级指针是否有效属于两个不同问题。

原 A2/A3 tiling 的 `Y_=A_*BS_*K_` 也缺少 arch35 已有的乘法前溢出检查，应在统一输入边界时一并处理。

## 4. 测试和公开文档的实际缺口

### T1：异步消费状态没有可复现的仓内回归入口

`tests/assets/golden.py` 的比较器核对八个输出，不能仅凭这些输出证明：

1. polling_index 指向正确的下一项；
2. 本次选中 descriptor 的 flag/IDs 被清理、layer_id 保留；
3. 未选中或快照之后迟到的 descriptor 没被误清；
4. 下一次调用不会重复消费已处理的数据。

Torch ST 限制 `need_schedule=0`，提示使用 `verify_torch_*.py`，但仓库没有相应脚本源码。缓存 `.pyc` 不是可维护的回归入口。建议提供两次连续调用及迟到 session 的设备状态测试，并显式保留生产者“不提前复用 payload”的前提。

### T2：容量前缀与间接路由的修复缺少对应 ST

生成器固定 `out_num=A`、`session_ids=arange(A)`、`micro_batch_ids=0`。因此现有生成流程不能覆盖 `outNum=0`、`0<outNum<A`、session 重排和非零 microbatch 路由。建议让实际容量/路由成为可配置用例输入，并保证未使用尾部含有能暴露误读的不同数据。

### T3：V2 示例的 PASS 不能证明 Gather 寻址和异步功能

新增 V2 示例使用相同 token 位模式、固定 session 路由，且不核对四个索引输出；取错位置仍可能得到相同 y。示例固定 NORM、`syncFlag=false`。本次已补注释限定 PASS 范围，未改变示例逻辑。建议后续改为位置相关数据、逐个核对路由，并增加真正异步示例。

比较器已有的 dtype/shape 检查、MX 原始字节比较、group_list 全容量比较是有效的；token 只比较 actual_token_num 前缀符合当前输出契约，不应要求未定义尾部等于零。

### D1：公开文档与代码的三个具体不一致

- `docs/aclnnFfnWorkerBatching.md` 把 expertOffsets 描述为“专家分组内偏移”，实际 kernel 输出原始 topK/共享专家槽位 k。它不是组内的第几个 token。
- V1/V2 文档称 BS、Y“无硬上限，仅受内存限制”，但 arch35 明确限制 Y 和 RECV 对齐后 flatElements 不超过 INT32_MAX。应写清表示范围。
- `docs/torchapi_ffn_worker_batching.md` 引用的 `examples/torch_ffn_worker_batching_async.py`、`examples/torch_ffn_worker_batching_async_ge.py` 不存在，不能照链接复现。

以上文档问题记录在此，原文未在本次检视中重写。

## 5. 已核对、没有据此认定缺陷的关键改动

| 改动 | 为什么当前写法有必要 | 核查边界 |
|---|---|---|
| 统一 GetFlatAivIdx | DAV_3510 的 GetBlockIdx 实现已展平，再乘 task ratio 会遗漏/重复任务 | 本机 CANN dav_3510 头文件支持此结论，不推广到所有旧架构 |
| readyWs 快照 | 判断 ready 后复用同批 flag，避免不同核把迟到 session 纳入不同集合 | 不是所有 flag 的原子瞬时快照；生产者仍负责发布可见性 |
| selectedMb 与推进后的 cursor 分开 | 前者表示本次消费，后者表示下次扫描起点 | 所有核必须在发布后恢复 selectedMb |
| Prepare 中先复制 IDs 再清 descriptor | 排序使用私有 ID 副本，payload 稍后才 gather | 文档明确生产者不能只看到 flag=0 就覆盖 payload，不能忽略此前提后直接判竞态 |
| layer_id 只编码排序 key | 不同层相同 local expert 应分组隔离，原始 gather 位置不能变 | 非法 layer/local ID 在加偏移前屏蔽，避免 sentinel 相加溢出 |
| 大行分块 | 至少处理一行不等于必须把整行放进 UB | preparePerLoopElements 才是实际缓冲容量 |
| 计数缓存 | sort 段数可随 Y 增加，不应让全部段计数常驻 UB | 原计数在排序屏障后不变，可跨归并轮缓存 |
| Compare/Extract 尾部初始化 | 指令读取对齐范围可能大于有效 len | 不能以最终只写有效前缀为由忽略未初始化输入 |
| 2^24 以上位置走整数解码 | FP32 无法精确表示超界奇数位置 | 判断使用包含 RECV padding 的原始位置容量，而非有效 token 数 |
| MX 按字节搬运 | FP4 无整数 sizeof，E8M0 也不应数值转换为 uint8 | 逻辑 H、tokenRowBytes、scaleRowBytes、HS 必须区分 |
| V1/V2 分离 | 在旧 C 符号中插入 bool 会让旧调用方参数错位 | V1 固定 false，V2 暴露 bool，共用内部接口 |
| Torch FP4 singleton stride | [1,1] 默认两个 stride=1 会让转换器难以判定打包轴 | 使用 [2,1] 明确最后一维，未增加实际存储需求 |
| mx_output_uint8 | 图后端可传输不支持的原生 FP4/E8M0 存储 | 使用 view，不做 to/cast；ACLNN 内部仍使用原生描述 |
| 六列 OpDef 签名 | 各端口列表按位置构成完整类型组合 | 不可去重“看似重复”的 INT32/INT64 列；UnknownShapeFormat 也须六项 |

### 仍需实机/编译产物确认的项目

- 部分 UB 复用没有显式手工 MTE3→V event，但本算子启用了 `--cce-auto-sync=on`。没有目标汇编/IR 证据，不能只凭少一个手工 event 判定存在竞争；也不能反向承诺编译器必然覆盖每条依赖。
- `EncodeLayerExperts` 的末轮使用整寄存器加载，而队列按 32B 对齐；活跃 lane 已限制，但是否被当前设备检查工具视为跨分配对象读取，需要编译/运行验证。当前不认定为已证明的数值错误。
- TilingData 为内部平铺结构，新旧 Host/kernel 必须配套，不承诺跨版本混搭 ABI。
- 大容量整数 gather 路径增加标量计算是精度保障；没有性能数据，不宣称性能回退。

## 6. 本次详细注释覆盖

新增/修订注释覆盖 24 个代码、构建及测试文件，保留现有执行逻辑：

- Host：sync_flag 名称与布尔语义、层数解释范围、超大行的物理容量、原始段计数缓存预算、GE 逻辑 FP4 shape、MX scale 尾组。
- 注册和 ABI：六列 dtype 组合、unknown-shape 格式、内部符号生成、V2 参数布局、const 描述对象不等于设备载荷只读、两段式生命周期。
- Kernel：实际 context 与容量的边界、三个发布/消费屏障、RECV padding 实例、跨层 key 与原始索引分工、proposal payload 位模式、紧凑输出与源行 padding。
- Torch/GE：dtype 属性编号、Meta 不读设备内容、mutable schema 能表达和不能表达的副作用、FP4/E8M0 view、输出顺序与导入注册顺序。
- Golden/UT/示例：归档重定位、位模式比较、有效前缀、Host UT 的占位输出局限、缺少状态验证、同值示例数据的覆盖限制。

同时纠正了“Host 已校验设备 A/M/K”“A5 与 A2 tiling ABI 一一相同”“wsFlatIds 总是 Y 个元素”等容易误导后续修改的旧注释。已有注释足够详细的算法段落保留，不机械逐行复述语句。

## 7. 验证记录与限制

- `git diff --check` 通过。
- 24 个修改文件与 HEAD 比较：C++/头文件去注释 token 流一致，Python 去文档字符串后的 AST 一致，CMake 去注释语句一致。本次没有更改可执行逻辑。
- Ascend950 JSON 六组 y/scale dtype 与 Host/Torch 映射一致，JSON 解析通过。
- Golden AST 和四份 ST CSV 属性解析通过；CSV 行数分别为 ACLNN 30、kernel 6、Torch eager 6、Torch graph 6。这是文件有效性检查，不是 ST 执行通过。
- 两个 signed-overflow 表达式用独立 UBSan 探针复现；旧 uint32 stride 回绕用独立 C++ 探针复现。
- 独立 CPU FP32 探针对小容量 gather 使用的单层商余数纠偏进行 748,165,720 次检查（商边界及随机除数），未发现错误。源码 `/tmp/ffn_wb_float_check.c`，编译命令 `gcc -O2 -ffp-contract=off /tmp/ffn_wb_float_check.c -lm -o /tmp/ffn_wb_float_check`，运行退出码为 0。它不执行完整 gather，也不能替代 NPU 指令级验证。
- 未运行本次完整 NPU ST、CANN 编译或 Host UT，不把 `/tmp` 中旧日志当作当前结果。注释编辑以 token/AST 一致性验证；功能发现以源码和明确标注的独立探针为证据。

条款级完整记录见仓库根目录 `operators/ffn_worker_batching/ffn_worker_batching_review_summary.md`。其中按规则计数的重复项、风格项和疑点不等于独立功能缺陷数量；本报告按根因归并。
