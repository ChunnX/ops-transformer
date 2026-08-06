# KPool Phase 1 CPU Golden

`k_pool_golden.py`包含两套实现：

- `run_k_pool_golden`：逐Batch、逐Token读取Block Table和State Cache的参考实现。
- `run_k_pool_oracle`：向量化读取Cache、分组Softmax和加权ReduceSum的独立Torch实现。

两套实现都覆盖：

```text
BSH [B,S,H]
TH  [T,H] + seq_lens[B]
K/Gate投影
可选LayerNorm
State Cache写入
Block Table寻址
Gate + APE
逐通道Softmax
BF16概率量化
加权K归约
固定容量输出和无效尾部清零
```

首版本RoPE接口保留，但传入非空`cos/sin`时显式抛出
`NotImplementedError`，不静默忽略。

设置`KPOOL_CASE_ROOT`后，`test_k_pool_manifest.py`会读取Phase 0.5归档的
Smoke、Typical、Full和Performance manifest。正常用例执行两套Oracle
bitwise比对；异常用例在接口阶段继续补充逐项拒绝验证。

本地Windows没有PyTorch/CANN运行环境，因此测试在A2/A5远端CPU Python环境中
执行。此阶段不调用NPU算子，也不声称NPU精度通过。
