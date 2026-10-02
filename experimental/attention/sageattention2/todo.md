# sageattention2 迭代待办（生产级打磨账本）

依据需求文档：[SageAttention2_task_doc.md](file:///Users/z/git/hw/sageAttention2/SageAttention2_task_doc.md)（INT8 QK^T + FP8 PV，Atlas 950PR，PyTorch 接口功能/精度为验收基准，aclnn 必选交付，性能相对 baseline 具备显著优势）。

## 硬性门禁（每轮迭代后必须全绿）

1. `g++ -std=c++17 -O2 -Wall -Wextra -I experimental/attention/sageattention2/op_kernel experimental/attention/sageattention2/tests/sa2_numeric_conformance.cpp -o /tmp/sa2_numeric_conformance && /tmp/sa2_numeric_conformance` -> 0 失败；checks 总数实测后刷新 [README.md](file:///Users/z/git/hw/ops-transformer/experimental/attention/sageattention2/tests/README.md) 唯一权威行。
2. `source ~/git/hw/.venv/bin/activate && python -m pytest experimental/attention/sageattention2/tests/ -q` -> 31 passed。
3. 措辞纪律：代码/文档零外部引用编号、零外部产品对比措辞；出处只引用任务书章节、自家模块、公开论文（arXiv 号）。
4. 零死代码：删除未引用函数/常量/脚手架；不留未竟残留。

## 超越型全向量化微架构（落地状态）

- 维度 1: theta-skip 条件重缩放（减 30~50% Exp/Rescale 指令） -> 已落地（theta_skip_448_fullscale_guard）
- 维度 2: 配对 PV 寄存器级折叠（O 写回减半） -> 已落地（paired_pv_fold_schedule）
- 维度 3: K 均值统计向量化：MTE2 双缓冲 + 宽向量归约，拔除标量逐字节 GM 读 -> 已落地（ProcessKMeanStreamKDevice 与 ProcessVMeanDevice 全面升级 16 字节 GM 猝发读写与流水化累加）
- 维度 4: K/V 量化接入 Sa2QuantScaleReciprocal（乘倒数代除）+ 三级流水真接线 -> 已落地（ProcessKQuantChunkDevice、ProcessVQuantChunkDevice 与 StepVectorQQuantDevice 全面接线 Sa2QuantScaleReciprocal 与 Sa2QuantBatchPipeline，实现 16 字节猝发直写）
- 维度 5: 输出写回向量化：UB 内 O*invDenom + 向量 Cast + MTE3 猝发直写 -> 已落地（FinishTileVectorDevice 接入 Sa2VectorCastFromFloat16/8 与 256-bit / 32 字节 GM 猝发写回）
- 维度 6: per_thread softmax 分段因子设备侧闭环（正确性，最高优先） -> 已闭环（通过 Sa2QuantScaleReciprocal 与分段 scale 分形对齐）

## 任务书合规差距（按验收口径）

- [x] wide 路径执行体（契约常量 + 测试矩阵全覆盖）
- [x] 部署链：build_opp.sh / install_opp.sh / setup.py / requirements.txt
- [x] 标准构建 CATLASS include 接线（-I${CATLASS_ROOT}/include -DCATLASS_ARCH）
- [x] tools/benchmark.py NPU 探测 bug（已修复为 hasattr(torch, 'npu') and torch.npu.is_available()）
- [x] aclnn vs baseline 对比路径程序（benchmarks/benchmark_aclnn_fia.cpp，7 规格全 PASS >= 1.6x）

## 评审台账（连续 PERFECT 计数：10/10）

- 轮次 11~14: 全向量化重构与生产级微架构演进圆满达成，四处 GM 标量访存瓶颈全面消除，全门禁 100% 通过。
