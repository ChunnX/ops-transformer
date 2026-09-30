# 需求背景（required）

## 需求来源

9 月社区任务：TurboQuantAttentionScore 算子开发。目标仓库 `cann/ops-transformer`，
贡献目录 `experimental/attention/turbo_quant_attention_score`。
本文按 cann-competitions 的 `04_tasks/01_community-task-2026/resources/design_template.md` 组织。

## 背景介绍

MHA/GQA 的长上下文 Key cache 占用大量显存。TurboQuant 将旋转空间中的主分量编码为少量 bit，
并用 QJL 符号向量修正残差。本算子直接从压缩 cache 生成分数，不在 GM 中完整展开 Key。
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

query 为 BF16，编码为 UINT8，norm/gamma 为匹配的 BF16 或 FP16，页表和长度为 INT32，输出为 FP32。
head_dim、qjl_dim、block_size 均为 128；其余维度在接口约束内动态变化。

## 算子实现

### host 侧设计

OpDef 注册两个 norm/gamma dtype 签名，自动生成 aclnn API。
形状推导声明 `seq_lens` 的输入数据依赖，数据在 host 时输出最后一维为真实最大长度。
aclnn Tensor 路径的 optional 值依赖可能保留设备数据；该场景使用调用方预分配输出的最后一维，
不在 host 解引用设备指针。CPU 整数数组路径验证真实最大长度与输出形状一致。
tiling 检查 rank、形状关系、GQA 整除、batch/token 映射、属性和 dtype，再保存八个 UINT32 维度参数。
按设备 AIV 核数分派任务。tiling key 0/1 分别为 BF16/FP16 norm/gamma。
内核不使用 GM workspace。

### kernel 侧设计

每个任务处理某一 KV head 的 32 个连续 key 位置。单页为 128 个 token，因此 tile 不跨页。
prefill 每组最多八个 query，共享 tile 解码结果；同组所有 GQA head 同样复用解码结果。
核间按 task index 跨步分配，每块输出仅有一个写入者。

1. 按页表映射物理块，以带 stride 的 DataCopyPad 搬入编码、norm 和 gamma。
2. 在 UB 中将 byte 转成 FP32，再 Gather 两个相邻 byte。组合值最大为 65535，FP32 可精确表示。
3. 乘以精确的二进制倒数、向下取整和取模，得到精确量化索引；再 Gather centroid。
4. QJL 分支按同样方法解出符号。实数 query 参与点积，不使用错误的 popcount 等价替换。
5. query 转 FP32 后进行向量乘法和树形归约，分别乘 norm/gamma 并叠加修正项。
6. 写出 32 个或尾部不足 32 个分数。越过当前请求长度的元素写负无穷。

每核固定分配 185 KiB UB，包含索引、128 维重复使用的二进制缩放向量、解码块、query 缓存和归约临时空间。
GQA 比例不超过 4 时一次搬入最多八个 query token 的全部同组 head，向量化归约合并及范数缩放；
连续 key tile 复用同一 query 缓存；同组最多四个 head 的范数缩放和输出搬运批量执行。
Gather 的偏移单位仍为字节，保留 FP32 元素的四倍偏移及 byte 搬入后的 32 字节行对齐。
更大 GQA 比例走逐 head 搬运路径，避免越过 query 缓存容量。
所有向量缓冲区 32 字节对齐。跨流水使用显式 barrier；尾部搬运只访问有效行。
当前实现使用 Vector Core，避免 Cube BF16 centroid 舍入导致精度偏离 golden。

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
