# GenericBlockSparseAttention Arch35 ATK 测试工程

## 1. 工程目标

本工程用于在 Ascend 950（Arch35）上验证 `GenericBlockSparseAttention` 的普通路径和 Flash Decoding（FD）路径。

主要覆盖：

- Metadata v8 协议模拟和自校验。
- Arch35 FD gate、cost model 与分核调度。
- FD partial O/LSE 和 stable combine 精度。
- 普通 Online Softmax 路径回归。
- BF16、FP16 和不同 topK。
- 单 Base Task、多 Base Task、空任务及 `sequsedQ` padding。
- PAGED_BBND 连续和 dim0 非连续 KV。
- ATK CV_FUSION 双标杆精度比较。

NPU 路径调用真实 Metadata 算子，并要求其 Metadata v8 输出与 Python 模拟器逐字一致；CPU Golden 使用同一模拟器生成的 `FdPlan`。

## 2. 整体架构

```text
run_atk_test.sh
        │
        ├─ TEST_MODE=normal
        │      └─ 随机普通 Case + 确定性 FD gate 对照
        │
        └─ TEST_MODE=fd
               └─ 10 类确定性 FD Case
                         │
                         ▼
generator_genericblocksparseattention.py
        │
        ├─ 生成 Q/K/V Shape
        ├─ 生成 sparseBlockIdx / sparseBlockCount Shape
        ├─ 生成 BlockTable Shape
        └─ 设置算子属性
                         │
                         ▼
ATK 创建 InputDataset
                         │
                         ▼
apply_init_tensors()
        │
        ├─ 构造合法 sparseBlockIdx / sparseBlockCount
        ├─ 构造 cuSeqLengths / seqused
        ├─ 构造 BlockTable
        └─ 调用 fd_metadata_v8.py
                         │
                         ▼
Metadata v8 + FdPlan
        │
        ├─ CPU pipeline Golden
        ├─ CPU FP32 Golden
        └─ NPU GBSA 主算子
                         │
                         ▼
ATK CV_FUSION 双标杆比较
```

## 3. 文件职责

### 3.1 `run_atk_test.sh`

测试总入口，负责：

- 加载 CANN 和自定义 OPP。
- 选择 `normal` 或 `fd` 测试模式。
- 选择物理 NPU。
- 通过 `ASCEND_RT_VISIBLE_DEVICES` 将物理卡映射为逻辑设备 0。
- 执行 Case 生成和测试。
- 支持 Case 白名单、重复稳定性测试和非连续 KV。

运行阶段会自动生成：

```text
all_op_generic_block_sparse_attention_fd.json
all_op_generic_block_sparse_attention_normal.json
.nodes_device_<physical_id>.yaml
result/
atk_output/
```

这些都是运行产物，不属于源码，可以重新生成。

### 3.2 `op_generic_block_sparse_attention.yaml`

ATK 算子定义文件，描述：

- ACLNN 接口名称。
- 输入、输出和属性。
- 支持的数据类型与 Shape。
- Case 生成数量。
- CV_FUSION 双标杆精度标准。

本工程不修改原有精度阈值。

### 3.3 `generator_genericblocksparseattention.py`

根据 `TEST_MODE` 选择 Case 生成策略。

普通模式由以下两部分组成：

```text
4 个确定性 FD gate 对照
        +
其余随机普通泛化 Case
```

确定性普通对照包括：

- `topK=1`。
- `topK=11`。
- `returnSoftmaxlse=1`。
- 9 个 Base Task，超过 FD gate。

FD 模式循环生成 `fd_case_specs.py` 中定义的确定性 FD Case。

生成器只负责 Case 的外部 Shape 和属性，不计算 Metadata，也不重复实现 FD gate。

### 3.4 `fd_case_specs.py`

保存确定性 FD Case 模板，包括：

- BF16，topK=12。
- BF16，topK=15。
- BF16，topK=16。
- FP16，topK=16。
- BF16，topK=256（FD 与普通路径容量上限一致）。
- 两个 Base Task。
- 八个 Base Task。
- `[16, 0]` 空任务。
- `sequsedQ` padding。
- 非连续 KV。

每个模板描述：

```text
dtype
Q storage/actual 数量
KV head/groupSize
topK
每个 Base Task 的实际 Block 数量
```

### 3.5 `fd_metadata_v8.py`

ATK 工程中唯一的 FD 调度实现，严格对齐 C++ Metadata Scheduler。

负责：

- Metadata v8 Header。
- Arch35 物理 AIC 数量 28。
- 实际 `sparseBlockCount` 前缀和。
- FD gate。
- Cost model。
- 最优 active AIC 数量。
- Decode Schedule。
- Combine Schedule。
- partial workspace 索引。
- config signature。
- `INT32[1024]` Metadata 序列化。
- Metadata 解码与严格自校验。

核心接口：

```python
metadata, plan = simulate_aicpu_metadata_v8(...)
```

返回结果：

```text
metadata
    = 直接传给 GBSA NPU Kernel 的 INT32[1024]

plan
    = CPU FD Golden 使用的结构化调度计划
```

### 3.6 `aclnn_genericblocksparseattention.py`

主执行和 Golden 实现，分为三个部分。

输入准备：

```text
apply_init_tensors()
    ├─ sparseBlockIdx
    ├─ sparseBlockCount
    ├─ BlockTable
    ├─ cuSeqLengths
    ├─ seqused
    └─ Metadata v8
```

CPU Golden：

```text
is_benchmark_task=False
    └─ pipeline Golden
           ├─ plan.enabled=False → 普通 Online Softmax
           └─ plan.enabled=True  → FD partial/combine

is_benchmark_task=True
    └─ FP32 数学 Golden
```

NPU 路径通过 `GenericBlockSparseAttentionInputProcess` 调用：

```text
aclnnGenericBlockSparseAttention
```

FD pipeline Golden 的计算顺序为：

```text
Decode Schedule
    ↓
每核 raw Block range
    ↓
partial O + partial LSE
    ↓
neutral partial
    ↓
Combine Schedule
    ↓
stable combine
    ↓
最终 attention output
```

普通路径和 FD 路径复用同一份 QK、Softmax 和 PV range 计算逻辑。

### 3.7 `aclnn_genericblocksparseattention_uncon.py`

非连续 KV 插件。它继承主 NPU InputProcess，只修改 K/V 的物理存储：

```text
k0, v0, k1, v1, ...
```

用于验证 PAGED_BBND dim0 非连续 stride 与 FD partial/combine 能够同时工作。

### 3.8 `nodes.yaml`

定义两个 ATK 执行节点：

```text
pyaclnn
    = 真实 NPU 算子

cpu
    = Python Golden
```

`run_atk_test.sh` 会基于该文件生成临时 nodes 配置，并统一使用逻辑设备 0。

## 4. Golden 与 Metadata 的关系

Metadata 序列化和 FD Golden 必须使用同一个 `FdPlan`：

```text
fd_metadata_v8.py
        │
        ├─ serialize → Metadata v8 → NPU Kernel
        │
        └─ FdPlan → FD pipeline Golden
```

因此工程内部不存在两套 FD gate 或两套分核算法。

真实 Metadata Scheduler 除了由 C++ Scheduler UT 验证，NPU ATK 路径还会逐字比较真实算子与 Python 模拟器输出；匹配后再验证 GBSA FD Kernel、partial workspace、combine 和输出精度。

## 5. 运行方法

进入测试目录：

```bash
cd /home/npu_user1/l00937279/ops-transformer/attention/generic_block_sparse_attention/tests/950
```

选择自定义 OPP：

```bash
export CUSTOM_OPP_PATH=/home/npu_user1/l00937279/package/package_custom_personal_wjc_msa_new_dev_llw2_c36ad0218c_atk/vendors/custom_transformer
```

### 5.1 FD 完整测试

```bash
TEST_MODE=fd DEVICE_ID=2 bash ./run_atk_test.sh all
```

### 5.2 普通路径完整测试

```bash
TEST_MODE=normal DEVICE_ID=2 bash ./run_atk_test.sh all
```

### 5.3 分阶段执行

```bash
TEST_MODE=fd DEVICE_ID=2 bash ./run_atk_test.sh generate
TEST_MODE=fd DEVICE_ID=2 bash ./run_atk_test.sh run
```

### 5.4 关键 FD Case

```bash
TEST_MODE=fd \
DEVICE_ID=2 \
ATK_WHITE_LIST='[0,23,46,68,90,112,134,156]' \
bash ./run_atk_test.sh run
```

### 5.5 非连续 KV FD

```bash
TEST_MODE=fd \
DEVICE_ID=2 \
ATK_WHITE_LIST='[178]' \
RUN_NONCONTIG=1 \
ATK_UNCON_WHITE_LIST='[178]' \
bash ./run_atk_test.sh run
```

### 5.6 重复稳定性测试

```bash
TEST_MODE=fd \
DEVICE_ID=2 \
ATK_WHITE_LIST='[46]' \
ATK_LOOP_NUMS=100 \
bash ./run_atk_test.sh run
```

`DEVICE_ID` 表示物理 NPU 卡号，脚本内部通过 `ASCEND_RT_VISIBLE_DEVICES` 将其映射为逻辑设备 0。

## 6. 工程设计原则

```text
fd_case_specs.py
    = 测什么

generator_genericblocksparseattention.py
    = Case Shape 怎么生成

fd_metadata_v8.py
    = Host 怎么判断 FD、怎么分核

aclnn_genericblocksparseattention.py
    = 按计划怎么算 Golden、怎么执行 NPU

aclnn_genericblocksparseattention_uncon.py
    = 怎么构造非连续 KV

run_atk_test.sh
    = 怎么组织整次测试

ATK CV_FUSION
    = 怎么判定精度通过
```

运行生成的 JSON、临时 nodes、Python 缓存、`result/` 和 `atk_output/` 均为可重新生成的运行产物，不应提交到版本库。
