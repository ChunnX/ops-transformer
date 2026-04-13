# torch_extension 模块分析

## 概述

`torch_extension` 是一个 **PyTorch NPU 算子扩展库**，通过 JIT（Just-In-Time）编译将 PyTorch 接口桥接到 ACLNN（Ascend C Library Neural Network）底层库，为 Ascend NPU 提供高性能算子支持。

## 主要功能

### 核心功能
1. **MOE（Mixture of Experts）分布式算子** - 支持 v2/v3 版本的 dispatch 和 combine 操作
2. **JIT 编译框架** - 动态编译 C++ 算子代码为共享库
3. **PyTorch Dispatcher 集成** - 通过 PrivateUse1 后端注册 NPU 算子
4. **图模式转换** - 支持 TorChair 静态图编译

### 支持的算子
- `npu_moe_distribute_dispatch_v2/v3` - MOE 专家分发算子
- `npu_moe_distribute_combine_v2/v3` - MOE 结果合并算子
- `update_context` - 分布式上下文更新
- `MoeDistributeBuffer` - 高层封装类，提供低延迟 MOE 接口

## 代码层次结构

```
torch_extension/
│
├── npu_ops_transformer/              # 主包
│   ├── __init__.py                   # 包入口，导入 torch/torch_npu
│   │
│   ├── ops/                          # 算子层
│   │   ├── __init__.py               # 导出所有算子接口
│   │   │
│   │   ├── [Python 前端层]
│   │   ├── moe_distribute_dispatch_v2.py    # Dispatch V2 算子
│   │   ├── moe_distribute_dispatch_v3.py    # Dispatch V3 算子
│   │   ├── moe_distribute_combine_v2.py     # Combine V2 算子
│   │   ├── moe_distribute_combine_v3.py     # Combine V3 算子
│   │   ├── deep_ep.py                       # MoeDistributeBuffer 高层封装
│   │   ├── update_context.py                # 上下文更新
│   │   │
│   │   ├── csrc/                     # [C++ 后端层]
│   │   │   ├── moe_distribute_dispatch_v2.cpp  # 调用 aclnnMoeDistributeDispatchV2
│   │   │   ├── moe_distribute_dispatch_v3.cpp  # 调用 aclnnMoeDistributeDispatchV3
│   │   │   ├── moe_distribute_combine_v2.cpp   # 调用 aclnnMoeDistributeCombineV2
│   │   │   ├── moe_distribute_combine_v3.cpp   # 调用 aclnnMoeDistributeCombineV3
│   │   │   └── update_context.cpp              # 调用 aclnnUpdateContext
│   │   │
│   │   └── graph_convert/            # [图模式转换层]
│   │       ├── graph_convert_moe_distribute_dispatch_v3.py
│   │       └── graph_convert_moe_distribute_combine_v3.py
│   │
│   ├── op_builder/                   # JIT 编译框架
│   │   └── builder.py                # OpBuilder 基类
│   │       ├── 管理 C++ 源码编译
│   │       ├── 注册算子 schema
│   │       ├── 配置编译参数
│   │       └── 缓存已编译模块
│   │
│   └── common/inc/                   # 公共头文件
│       ├── aclnn_common.h            # ACLNN 调用宏和类型转换
│       └── hccl_common.h             # HCCL 通信相关
│
├── setup.py                          # 打包配置
├── requirements.txt                  # 依赖列表
└── README.md                         # 使用文档
```

## 调用链路详解

### 完整调用栈

```
用户代码
  ↓
[1] Python 封装层 (deep_ep.py)
  MoeDistributeBuffer.npu_low_latency_dispatch()
  └─> torch.ops.npu_ops_transformer.npu_moe_distribute_dispatch_v3()
  ↓
[2] PyTorch Dispatcher 层 (moe_distribute_dispatch_v3.py)
  @impl(AS_LIBRARY, "npu_moe_distribute_dispatch_v3", "PrivateUse1")
  def npu_moe_distribute_dispatch_v3(...):
      op_module = builder.load()  # JIT 编译
      return op_module.npu_moe_distribute_dispatch_v3(...)
  ↓
[3] JIT 编译层 (builder.py)
  OpBuilder.load()
  └─> torch.utils.cpp_extension.load()
      ├─> 编译 csrc/*.cpp
      ├─> 链接 libascendcl.so, libtorch_npu.so
      └─> 返回 Python 模块
  ↓
[4] C++ 桥接层 (csrc/*.cpp)
  tensor_list npu_moe_distribute_dispatch_v3(...) {
      // 参数校验
      // 输出张量分配
      ACLNN_CMD(aclnnMoeDistributeDispatchV3, ...);
      return {expand_x, scales, ...};
  }
  ↓
[5] ACLNN 调用层 (aclnn_common.h)
  ACLNN_CMD 宏展开:
  ├─> ConvertTypes(): PyTorch Tensor → aclTensor
  ├─> aclnnXxxGetWorkspaceSize(): 查询工作空间
  ├─> aclnnXxx(): 提交 NPU 算子执行
  └─> ReleaseConvertTypes(): 释放 ACL 对象
  ↓
[6] 底层执行
  libopapi.so (ACLNN 库)
  └─> NPU 硬件执行
```

### 各层职责

#### 1. 用户调用层
```python
import npu_ops_transformer

# 创建 MOE 分布式缓冲区
buffer = npu_ops_transformer.ops.MoeDistributeBuffer(
    group=dist_group,
    ccl_buffer_size=200  # MB
)

# 执行低延迟 dispatch
expand_x, scales, idx, token_nums, recv_counts, expand_scales = \
    buffer.npu_low_latency_dispatch(
        x=input_tensor,
        topk_idx=expert_indices,
        num_experts=8,
        quant_mode=0
    )
```

#### 2. Python 封装层 (deep_ep.py)
- **MoeDistributeBuffer 类**：管理分布式上下文和 CCL 缓冲区
- **功能**：
  - 初始化 HCCL 通信组
  - 计算最优 CCL 缓冲区大小
  - 封装 dispatch/combine 调用
  - 支持动态更新通信组

#### 3. PyTorch Dispatcher 层 (算子 Python 文件)
- **OpBuilder 实例化**：定义算子 schema 和元信息
- **Meta 函数注册**：实现形状推导逻辑
- **PrivateUse1 实现**：NPU 后端的具体执行逻辑
- **图模式转换器**：TorChair 静态图支持

关键代码模式：
```python
class MoeDistributeDispatchV3OpBuilder(OpBuilder):
    def schema(self) -> str:
        return "npu_moe_distribute_dispatch_v3(...) -> (...)"

    def sources(self):
        return ['ops/csrc/moe_distribute_dispatch_v3.cpp']

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def meta_func(...):
            # 形状推导逻辑
            return output_shapes

@impl(AS_LIBRARY, builder.name, "PrivateUse1")
def npu_moe_distribute_dispatch_v3(...):
    op_module = builder.load()  # JIT 编译
    return op_module.npu_moe_distribute_dispatch_v3(...)
```

#### 4. JIT 编译层 (builder.py)
- **OpBuilder 基类**：管理算子编译生命周期
- **编译配置**：
  - 包含路径：torch_npu、CANN、公共头文件
  - 编译选项：`-fPIC`, `-O2`, `-D_GLIBCXX_USE_CXX11_ABI`
  - 链接库：`libascendcl.so`, `libtorch_npu.so`
- **缓存机制**：已编译模块存储在 `_loaded_ops` 字典

#### 5. C++ 桥接层 (csrc/*.cpp)
- **参数校验**：维度检查、范围验证
- **输出分配**：根据 Meta 函数逻辑预分配输出张量
- **ACLNN_CMD 宏调用**：统一的 ACLNN 算子调用接口

示例：
```cpp
tensor_list npu_moe_distribute_dispatch_v3(...) {
    // 1. 参数校验
    TORCH_CHECK(x.dim() == 2, "x should be 2D");

    // 2. 计算输出形状
    int64_t a = calculate_output_size(...);

    // 3. 分配输出张量
    at::Tensor expand_x = at::empty({a, h}, x.options());

    // 4. 调用 ACLNN 算子
    ACLNN_CMD(aclnnMoeDistributeDispatchV3,
              context, x, expert_ids, ..., expand_x, ...);

    // 5. 返回结果
    return std::make_tuple(expand_x, scales, ...);
}
```

#### 6. ACLNN 调用层 (aclnn_common.h)
**ACLNN_CMD 宏**是核心机制，展开后执行：

1. **类型转换**：`ConvertTypes()` 将 PyTorch 类型转换为 ACL 类型
   - `at::Tensor` → `aclTensor*`
   - `at::Scalar` → `aclScalar*`
   - `at::IntArrayRef` → `aclIntArray*`

2. **工作空间查询**：调用 `aclnnXxxGetWorkspaceSize()`

3. **工作空间分配**：如果需要，分配临时缓冲区

4. **算子执行**：调用 `aclnnXxx()` 提交到 NPU Stream

5. **资源释放**：`ReleaseConvertTypes()` 释放 ACL 对象

关键特性：
- **零拷贝**：直接包装 PyTorch 存储，无需数据拷贝
- **异步执行**：通过 NPU Stream 异步提交
- **错误处理**：统一的错误检查和日志

#### 7. 底层执行
- **libopapi.so**：ACLNN 算子库，包含编译好的 NPU 算子
- **libcust_opapi.so**：自定义算子库（优先加载）
- **NPU 硬件**：最终在 Ascend NPU 上执行

## 关键技术点

### 1. 双模式支持
- **Eager 模式**：通过 `PrivateUse1` dispatch key 注册
- **图模式**：通过 TorChair converter 转换为静态图

### 2. Meta 函数机制
```python
@impl(AS_LIBRARY, self.name, "Meta")
def meta_func(x, expert_ids, ...):
    # 仅计算输出形状和类型，不执行实际计算
    bs, h = x.shape
    a = calculate_output_size(...)
    expand_x = x.new_empty((a, h))
    return expand_x, ...
```

**作用**：
- 支持 FakeTensor 模式（图编译时的形状推导）
- 支持自动微分（Autograd）
- 支持 torch.compile

### 3. 零拷贝桥接
```cpp
aclTensor* ConvertType(const at::Tensor &at_tensor) {
    return aclCreateTensor(
        at_tensor.sizes().data(),      // 形状
        at_tensor.strides().data(),    // 步长
        acl_data_type,                 // 数据类型
        const_cast<void*>(at_tensor.storage().data())  // 直接使用存储指针
    );
}
```

### 4. JIT 编译缓存
```python
class OpBuilder:
    _loaded_ops = {}  # 类级别缓存

    def load(self):
        if self.name in __class__._loaded_ops:
            return __class__._loaded_ops[self.name]

        op_module = torch.utils.cpp_extension.load(...)
        __class__._loaded_ops[self.name] = op_module
        return op_module
```

### 5. HCCL 集成
```python
# 获取 HCCL 通信组名称
group_name = group._get_backend(torch.device("npu")).get_hccl_comm_name(
    rank_id, init_comm=False
)

# 传递给 C++ 层用于集合通信
update_context(group_name, world_size, buffer_size, context)
```

### 6. 量化模式支持
```cpp
enum QuantMode {
    QUANT_MODE_NO_QUANT = 0,    // 无量化
    QUANT_MODE_STATIC = 1,       // 静态量化
    QUANT_MODE_PERTOKEN = 2,     // Per-token 量化
    QUANT_MODE_PERGROUP = 3,     // Per-group 量化
    QUANT_MODE_MX = 4,           // MX 量化
};
```

## 使用示例

### 基础用法
```python
import torch
import torch_npu
import npu_ops_transformer

# 初始化数据
x = torch.randn(1024, 4096, dtype=torch.float16).npu()
expert_ids = torch.randint(0, 8, (1024, 2), dtype=torch.int32).npu()

# 创建分布式组
import torch.distributed as dist
dist.init_process_group(backend='hccl')
group = dist.new_group()

# 创建 MOE Buffer
buffer = npu_ops_transformer.ops.MoeDistributeBuffer(
    group=group,
    ccl_buffer_size=200
)

# Dispatch
expand_x, scales, idx, token_nums, recv_counts, expand_scales = \
    buffer.npu_low_latency_dispatch(
        x=x,
        topk_idx=expert_ids,
        num_experts=8,
        quant_mode=0
    )

# ... 专家计算 ...

# Combine
output = buffer.npu_low_latency_combine(
    x=expert_output,
    topk_idx=expert_ids,
    topk_weights=weights,
    assist_info_for_combine=idx,
    ep_send_counts=recv_counts,
    num_experts=8
)
```

### 计算最优缓冲区大小
```python
buffer_size_mb = npu_ops_transformer.ops.MoeDistributeBuffer.get_low_latency_ccl_buffer_size(
    world_size=8,
    num_max_dispatch_tokens_per_rank=2048,
    hidden=4096,
    num_moe_expert=64,
    topk=2,
    comm_alg="fullmesh_v2"
)
print(f"Recommended buffer size: {buffer_size_mb} MB")
```

## 编译与安装

### 依赖要求
- OS: Linux
- Python: 3.8+
- Compiler: GCC 9.4.0+
- PyTorch: 2.6.0+
- torch_npu: 与 PyTorch 版本匹配
- CANN Toolkit: 已安装并设置 `ASCEND_HOME_PATH`

### 安装步骤
```bash
cd torch_extension

# 1. 安装依赖
pip install -r requirements.txt

# 2. 构建 wheel
python -m build --wheel -n

# 3. 安装
pip install dist/*.whl --force-reinstall --no-deps
```

## 架构优势

1. **高性能**：直接调用 ACLNN 底层算子，零拷贝数据传递
2. **易用性**：PyTorch 原生接口，无需修改用户代码
3. **灵活性**：JIT 编译支持快速迭代开发
4. **可扩展**：OpBuilder 基类简化新算子添加
5. **兼容性**：同时支持 Eager 和图模式

## 总结

`torch_extension` 模块是 CANN 生态中连接 PyTorch 前端和 Ascend NPU 硬件的关键桥梁。通过 JIT 编译、PyTorch Dispatcher 机制和 ACLNN 库的深度集成，为用户提供了高性能、易用的 NPU 算子扩展能力。

其设计充分体现了现代深度学习框架的工程实践：
- **分层架构**：清晰的职责划分
- **零拷贝优化**：最小化数据传输开销
- **动态编译**：开发效率与运行性能的平衡
- **标准化接口**：符合 PyTorch 生态规范
