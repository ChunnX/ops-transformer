# SageAttention2 PyTorch 接入规范与端到端集成指南

本文档介绍如何在 PyTorch 大模型训练与推理管线中接入和使用华为昇腾 Atlas 950PR 的 `SageAttention2` 算子。

## 一、接口入口概述

`sageattention2` 提供两套 Python 调用接口：

1. **统一入口 `sageattn`**：
   - 高级通用入口，自适应路由至 NPU 优化的 INT8 QK^T + FP8 PV 高性能计算流。
2. **底层直调入口 `sageattn_qk_int8_pv_fp8_asc`**：
   - 显式量化配置与全量控制参数入口，具备精细调节能力。
3. **无侵入替换器 `replace_sdpa`**：
   - 基于 Python 上下文管理器的无侵入式替换，将 PyTorch 标准 `F.scaled_dot_product_attention` 自动重定向至 `sageattn`。

## 二、快速上手代码示例

### 1. 基础调用示例（HND 布局与默认参数）

```python
import torch
import torch_npu  # 注册 "npu" 设备后端，昇腾环境下必须导入
from sageattention2 import create_attention_inputs, create_qkv, sageattn

# 方式 1：使用 create_qkv 统一工厂构造输入（彻底消除 3 行重复的 torch.randn 模版代码）
q, k, v = create_qkv(batch=1, n_heads=32, seq_len=4096, head_dim=128, dtype=torch.float16, device="npu")

# 执行 SageAttention2 高性能前向计算
out = sageattn(q, k, v, tensor_layout="HND", is_causal=False)
print("Output shape:", out.shape)  # [1, 32, 4096, 128]

# 方式 2：使用字典参数与 create_attention_inputs 结构化解耦（配置即代码）
spec = {
    "batch": 1, "n_heads": 32, "seq_len": 4096, "head_dim": 128,
    "dtype": torch.float16, "device": "npu", "is_causal": False,
}
qkv, op_kwargs = create_attention_inputs(spec)
out = sageattn(**qkv.as_dict(), **op_kwargs)
```

### 2. GQA 分组查询注意力与非对称序列

```python
# Query 具有 32 头，Key/Value 具有 8 头 (4:1 GQA)；Sq ≠ Sk 的非对称序列
# 借助 create_qkv 命名参数简化传参：
q, k, v = create_qkv(
    batch=1, hq=32, hk=8, sq=4096, sk=8192, d=128,
    dtype=torch.bfloat16, device="npu",
)

out = sageattn(q, k, v, is_causal=False)
assert out.shape == (1, 32, 4096, 128)
```


### 3. 使用 `replace_sdpa` 实现模型零修改加速

```python
import torch.nn.functional as F
from sageattention2 import replace_sdpa

# 在需要加速的 Transformer 层作用域中使用上下文管理器
with replace_sdpa():
    # 原生调用将自动重定向至 sageattn
    out = F.scaled_dot_product_attention(q, k, v)
```

## 三、动态 Shape 与非法边界防御

为确保生产环境运行的健壮性，算子内置了完备的防御性边界拦截机制：

1. **输入合法性前置拦截**：
   - 输入张量必须同设备且同数据类型（FP16 或 BF16）；
   - 最后一维必须在内存中连续分布（`stride(-1) == 1`），杜绝因内存重整引入隐藏错误；
   - GQA 必须满足整除关系（`Hq % Hk == 0`）；
   - 因果注意力模式（`is_causal=True`）要求输入序列为方阵（`Sq == Sk`）。
2. **特征维度超界拦截**：
   - 特征维度 $D$ 必须严格满足 $0 < D \le 128$，对于 $D \le 0$ 或 $D > 128$ 严格抛出 `ValueError`。
3. **未知传参安全检查**：
   - 底层入口对未声明的可变关键字参数（`**kwargs`）实施严格类型拦截，禁止非法参数被静默吞掉。

## 四、深度集成契约（torch.compile / FakeTensor / 确定性）

本算子按生产级 PyTorch 算子规范接入，以下契约均已有宿主用例锁定（见 [tests/test_host_assertions.py](../tests/test_host_assertions.py)）：

1. **torch.compile 兼容**：
   - 以 `torch.compile(fn, backend="eager", fullgraph=True)` 包裹调用可完整图捕获，不产生图断裂（GraphBreak）；
   - 算子对未知 kwargs 采用严格异常而非静默旁路，保证编译期语义与运行期一致。
2. **Meta / FakeTensor 支持**：
   - 抽象解释（FakeTensor/mode="fake"）下仅执行 shape/dtype 推导即返回正确元信息，可被 `torch.export` 与
     Dynamo 形状推导链安全消费。
3. **dispatcher schema 对齐**：
   - 注册 schema 与 Python 签名的参数名、默认值、返回结构逐项一致，`torch._C._jit_get_schemas_for_operator`
     断言通过。
4. **重复调用逐位确定性**：
   - 相同输入连续两次调用，输出与 LSE 均满足 `torch.equal` 逐位一致（设计依据：K 均值采用单写者确定性
     归约树，全程无 atomic 累加）。
5. **存储格式约束**：
   - 仅接受末维连续（`stride(-1)==1`）的标准稠密布局；私有/稀疏存储格式在入口即被明确拒绝，
     不做隐式拷贝兜底。
6. **环境锚点**：
   - 已验证组合登记于 README 与自测报告（N-4 项随真机环境定型更新）。

## 五、相关文件索引

- 接口实现：[python/sageattention2/core.py](../python/sageattention2/core.py)
- SDPA 替换器：[python/sageattention2/_ops.py](../python/sageattention2/_ops.py)
- 测试验证矩阵：[tests/test_functional_matrix.py](../tests/test_functional_matrix.py)
