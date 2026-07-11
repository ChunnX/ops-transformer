# CausalTopoGCN (因果拓扑图卷积算子)

## 1. 算子描述 (Description)
`CausalTopoGCN` 是受临床因果知识图谱启发的高阶图神经网络算子。该算子将图节点特征计算与邻接矩阵拓扑推演深度固化到 AICore 中，专为斜视医疗诊断中的多模态推断加速而设计。

## 2. 算子规格 (Operator Specification)
* **输入 (Inputs)**:
  * `x`: Tensor (数据类型: FLOAT16, 格式: ND) - 节点特征矩阵。
  * `edge_index`: Tensor (数据类型: INT32, 格式: ND) - 图的邻接边索引。
* **输出 (Outputs)**:
  * `y`: Tensor (数据类型: FLOAT16, 格式: ND) - 卷积后的输出 Logits 张量。
* **属性 (Attributes)**:
  * `in_channels` (Int类型) - 输入特征维度。
  * `out_channels` (Int类型) - 输出特征维度。

## 3. 支持的硬件 (Supported Hardware)
* Ascend 910B

## 4. 测试用例 (Test Cases)
本算子已提供本地单元测试（UT）与 Python 运行测试。
执行 Python 测试验证 NPU 调用：
~~~bash
python3 tests/test_causal_topo_gcn.py
~~~
