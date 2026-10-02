# 需求背景（required）

## 需求来源

9 月社区任务：TurboQuantAttentionScore 算子开发。目标仓库 `cann/ops-transformer`，
贡献目录 `experimental/attention/turbo_quant_attention_score`。
本文按 cann-competitions 的 `04_tasks/01_community-task-2026/resources/design_template.md` 组织。

## 背景介绍

MHA/GQA 的长上下文 Key cache 占用大量显存。TurboQuant 将旋转空间中的主分量编码为少量 bit，
并用 QJL 符号向量修正残差。decode 路径直接从压缩 cache 生成分数；对齐 prefill 路径使用有界临时工作区复用解码系数，调用结束后释放。
与仓内 TurboQuant 稀疏 attention 算子不同，本接口输出稠密 attention score，且不包含 V 聚合。

# 需求分析（required）

## 需求描述

实现 A2 / CANN 9.1.0 的 Ascend C 算子和 aclnn 两阶段接口。
输入输出、dtype 和布局详见上级 README。以任务附件 `golden.py` 为编码语义基准。

## 需求拆解

1. 正确解析 2/3/4 bit 主编码以及 1 bit QJL 符号。
2. 使用 Paged KV cache，并支持 GQA、单序列 prefill 和批量 decode。
3. 通过 FP32 计算保留 golden 常量精度，不以 BF16 centroid 额外引入误差。
4. 覆盖五组指定 shape 与边界场景，记录真实 aclnn 精度、耗时和内存数据。

# 详细设计（required）

## 算子分析

### 数学公式

记 \(c_j\) 为查表得到的主分量，\(s_j\in\{-1,+1\}\) 为 QJL 符号：

$$
\mathrm{score}=\mathrm{norm}\sum_{j=0}^{127}q_jc_j+
\frac{\sqrt{\pi/2}}{128}\,\mathrm{gamma}\sum_{j=0}^{127}q^{\mathrm{QJL}}_js_j.
$$

query 的旋转及投影由上游完成。该输出不包括 softmax 缩放和 causal mask。

### 支持数据类型与形状

query 为 BF16，编码为 UINT8，norm/gamma 为匹配的 BF16 或 FP16，页表为 INT32，长度为 INT32/INT64，输出为 FP32。
head_dim、qjl_dim、block_size 均为 128；其余维度在接口约束内动态变化。

## 算子实现

### host 侧设计

OpDef 注册两个 norm/gamma dtype 签名，自动生成 aclnn API。
形状推导声明 `seq_lens` 的输入数据依赖，数据在 host 时输出最后一维为真实最大长度。
aclnn Tensor 路径的 optional 值依赖可能保留设备数据；该场景使用调用方预分配输出的最后一维，
不在 host 解引用设备指针。CPU 整数数组路径验证真实最大长度与输出形状一致。
tiling 检查 rank、形状关系、GQA 整除、batch/token 映射、属性和 dtype，再保存八个 UINT32 维度参数。
Vector 路径按设备 AIV 核数分派，tiling key 0/1 分别对应 BF16/FP16 norm/gamma，GM workspace 仍为零。
对齐 prefill 使用 key 2/3 的 MIX AIC:AIV=1:2 路径，并附带 TCubeTiling。
分派条件为单序列、128≤T≤512、T 为 32 的倍数、输出长度为正且是 256 的倍数，
同时限制矩阵 stride 不超过 INT_MAX、临时用户工作区不超过 128 MiB。
不满足条件或 tiling 失败时保留原 Vector 路径。

### Vector kernel 侧设计

每个任务处理某一 KV head 的 32 个连续 key 位置。单页为 128 个 token，因此 tile 不跨页。
prefill 每组最多八个 query，共享 tile 解码结果；同组所有 GQA head 同样复用解码结果。
核间按 task index 跨步分配，每块输出仅有一个写入者。

1. 按页表映射物理块，以带 stride 的 DataCopyPad 搬入编码、norm 和 gamma。
2. 在 UB 中将 byte 转成 FP32，再 Gather 两个相邻 byte。组合值最大为 65535，FP32 可精确表示。
3. 乘以精确的二进制倒数、向下取整和取模，得到精确量化索引；再 Gather centroid。
4. QJL 分支按同样方法解出符号。实数 query 参与点积，不使用错误的 popcount 等价替换。
5. query 转 FP32 后进行向量乘法和树形归约，分别乘 norm/gamma 并叠加修正项。
6. 写出 32 个或尾部不足 32 个分数。越过当前请求长度的元素写负无穷。

Vector prefill 每核使用 185 KiB UB；单 token decode 额外分配 2 KiB 压缩输入预取缓冲区，共 187 KiB。基础空间包含索引、128 维重复使用的二进制缩放向量、解码块、query 缓存和归约临时空间。
GQA 比例不超过 4 时一次搬入最多八个 query token 的全部同组 head，向量化归约合并及范数缩放；
单 token decode 使用 packed QJL bit 直接选择正负 query residual，保持原 FP32 归约顺序；下一 tile 的压缩 idx 在独立缓冲区预取，消费前等待 MTE2 到 Vector 的事件。
连续 key tile 复用同一 query 缓存；同组最多四个 head 的范数缩放和输出搬运批量执行。
Gather 的偏移单位仍为字节，保留 FP32 元素的四倍偏移及 byte 搬入后的 32 字节行对齐。
更大 GQA 比例走逐 head 搬运路径，避免越过 query 缓存容量。
所有向量缓冲区 32 字节对齐。跨流水使用显式 barrier；尾部搬运只访问有效行。
Vector 路径保持原 FP32 解码和归约顺序。

### 对齐 prefill：Vector 准备与 Cube 分阶段流水（2026-10-02）

输入、输出、bit packing、centroid、QJL 修正系数和测试阈值保持原定义。
Vector 每次解码 32 行，先计算主项 `centroid * norm` 和残差项 `sign * gamma * coefficient`。
每项系数拆成两个 FP16 分量：高分量为 FP16 舍入值，低分量为 FP32 残差乘以 256 后的 FP16 舍入值。
对应低分量的 query 除以 256；这减少低位系数落入 FP16 非规格化范围造成的舍入损失。
K=512 的顺序为主项高位、QJL 高位、主项低位、QJL 低位；Cube 以 FP32 累加，直接写入
`[T, Hq, maxLen]` 的最终跨 head 行 stride，不生成完整中间 score 矩阵，也不再进行 GM score 后处理。
该拆分并非逐位保留全部 FP32 信息，系数预乘、query 转换、低位缩放和归约仍存在舍入误差。
因此必须继续对完整输出执行原独立 CPU 和附件 golden 的精度检查，不能仅用“采用 FP32 累加”推导精度合格。

每个准备任务拥有一个 KV head 的 256-key 块，该块供同组所有 query head 复用。
查询准备完成并统一发布检查结果后，Vector 准备下一阶段的 key 系数，Cube 计算上一阶段。
每阶段末使用 `SyncAll<false>()` 显式跨 AIC/AIV 同步；矩阵乘对象在整个阶段循环中复用。
B 系数采用双阶段环形缓冲：阶段 w 写入之前，阶段 w-2 的全部 Cube 读取已在前一屏障完成。
每阶段 AIC 仅枚举该阶段的 key tile 与 GQA head 组合，避免每轮扫描完整序列。
query 打包跨越 b_/c_ 两段 UB；缩放 query 必须保存在独立 queryCache_/queryResidualCache_ 中。
支持 128×256×64 基础 tile 时，L1 的 A/B stepK 均取 4、depth 均取 8，总占用 384 KiB；
host 核对实际 tile 和 L1 容量后才采用该配置。

准备阶段检查真实长度、页号和数值范围；norm/gamma 或 query 非有限、超出 FP16 范围，
或者长度/页号不适合 Cube 时，通过初始化后的原子状态字发布回退信号。
所有 Cube 写入完成并同步后，AIV 使用原 Vector 路径覆盖整个输出；因此回退不会与 Cube 竞争写出。
MIX 模式的 AIV task stride 使用 `GetBlockNum() * GetTaskRation()`，避免两倍 Vector 核重复处理同一任务。

设 AIC 核数为 C，用户工作区字节数为 `(Hq*T + 4*C*256)*1024 + (1 + Hkv*maxLen/256)*512`，
另加平台 `GetLibApiWorkSpaceSize()` 返回的系统工作区。
当前 20-AIC A2 上，官方 T=512、Hq=32、Hkv=8 的 4K/32K 用例总 workspace 分别为
54,592,000 / 55,050,752 字节（含 16 MiB 系统区）；输出分别独立占用 256 MiB / 2 GiB。
32K 不再因完整 B 展开的工作区上限而退回 Vector。用户工作区 128 MiB 上限仍保留。
此处是临时计算开销，不能计入持久 KV cache 压缩比。

专项测试为 `tests/test_prefill_regression.py`：覆盖两种 norm dtype、两种长度 dtype、bits=2/3/4、
GQA=1/2/3/4/5/8、host 长度接口、异常后恢复正常和非对齐形状回退。
`test_prefill_ring_regression.py` 额外覆盖 88 个 key tile（20-AIC 时跨两个缓冲槽复用且最后一波不满）、
GQA 1/2/3/4/5/8、两种 norm dtype、两种长度 dtype、2/3/4 bit、乱序页表及三次连续调用。
`test_golden_nonfinite.py` 额外验证 golden 包装器拒绝有限参考位置上的 NaN/Inf，防止 NaN 比较静默漏检。
纯相对误差近零分母及模型验收结论仍需与组织方规则区分，混合误差自测通过不等于完成任务验收。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2（Ascend 910B） | √ |

## 算子约束限制

不支持 MLA、任意 head_dim/qjl_dim、多序列 prefill、Value 解码、softmax 或 V 聚合。
batch 大于 1 时每个请求只能有一个 query token；缺少 query 边界的接口不能推导更多映射。

# 可维可测分析

## 精度标准/性能标准

| 验收项 | 验证方式 |
| --- | --- |
| 压缩域计算正确性 | 附件 golden 与独立整数解码参考、全输出比较 |
| Decode 延迟 | aclnn 实际执行，与任务书给定三组 baseline 对照；另列主机调用耗时 |
| Key cache 压缩 | 默认 3 bit 主编码 + 1 bit 符号 + 两个 16 bit 范数，共 68 B/key，原 BF16 Key 为 256 B，压缩比约 3.765 |
| 总显存 | 单独列出输出大小、cache 和 workspace，不能将 Key 压缩比宣称为进程整体压缩比 |
| KL | 新附件提供配对原始 Q/K，可执行配对 score/softmax 比较；缩放、方向及聚合规则仍待确认。新版已删除 PPL 要求 |

## 兼容性分析

新增独立算子，不修改其他算子行为。使用仓库默认 experimental 自动发现和 aclnn 生成机制。

## 2026-09-30 任务附件更新

新版 decode 32K 用例调整为 64K；性能比较对象明确为 BF16 Paged Full-QK 原始 score，五组 baseline 依次为 126.892 / 2321.657 / 5386.388 / 627.391 / 5910.060 μs。旧性能基线与通过结论不再适用。

新增 generate_data.py 明确 NumPy PCG64、rotation seed=42、QJL seed=137。旋转矩阵为 QR 正交矩阵，QJL 矩阵为正交矩阵乘 `0.63*sqrt(128)`。行向量输入先计算 `query@H.T` 并舍入 BF16，再乘 `S.T` 并舍入 BF16。gamma 为原始 key 范数乘主量化残差范数；QJL 符号对零取正，量化边界使用严格大于，packed 数据低位优先。

新附件解决了生成 score 配对输入的矩阵与编码约定缺口，不等于独立上游编码算子已完成联测。golden 公式没有变化，`err_threshold=[0.001,0]` 的正式比较器仍未明确。KL 的缩放、方向与聚合规则仍须确认；PPL 已删除，不再列为本任务阻塞项。
