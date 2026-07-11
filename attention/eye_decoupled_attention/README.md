# EyeDecoupledAttention (眼位解耦注意力算子)

## 1. 算子描述 (Description)
`EyeDecoupledAttention` 是专为眼科斜视诊断图神经网络（GNN）多维度特征对齐设计的定制化注意力算子。它在 Ascend NPU 上实现了高效的前向/反向计算，相比 CPU 兜底实现，显著降低了多模态医学图像特征融合的延迟。

## 2. 算子规格 (Operator Specification)
* **输入 (Inputs)**:
  * `x`: Tensor (数据类型: FLOAT16, 格式: ND) - 原始节点特征张量。
  * `pos`: Tensor (数据类型: FLOAT16, 格式: ND) - 眼位位置编码张量。
* **输出 (Outputs)**:
  * `y`: Tensor (数据类型: FLOAT16, 格式: ND) - 解耦注意力对齐后的特征张量。
* **属性 (Attributes)**:
  * `C`, `hb`, `wb`, `bn_h`, `bn_w` (Int类型) - 用于控制 Tiling 切分与注意力块维度的超参数。

## 3. 支持的硬件 (Supported Hardware)
* Ascend 910B

## 4. 测试用例 (Test Cases)
本算子已提供本地单元测试（UT）与 Python 运行测试。
执行 Python 测试验证 NPU 调用：
~~~bash
python3 tests/test_eye_decoupled_attention.py
~~~
