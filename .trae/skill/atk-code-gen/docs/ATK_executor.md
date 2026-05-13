# ATK Executor API 参考

本文档从现有executor代码中总结ATK框架提供的接口和特性，按 **CPU实现 / NPU级联实现 / ACLNN实现** 三类场景组织。

---

## 一、框架基础

### 1.1 注册器 `register`

```python
from atk.tasks.api_execute import register
```

| 注册名前缀 | 用途 | 继承基类 |
|------------|------|---------|
| `function_` | 标杆执行器（CPU真值 + NPU小算子级联） | `BaseApi` |
| `pyaclnn_` | ACLNN执行器 | `AclnnBaseApi` |

注册名对应yaml中 `api_type` / `aclnn_api_type` 字段。

```python
@register("function_aclnn_alltoall_matmul")
class AllToAllMatmul(BaseApi): ...

@register("pyaclnn_aclnn_alltoall_matmul")
class AclnnAllToAllMatmul(AclnnBaseApi): ...
```

### 1.2 基类 `BaseApi`

```python
from atk.tasks.api_execute.base_api import BaseApi
```

**构造参数**：`task_result`

**关键属性**（从 `task_result` 获取）：

| 属性 | 类型 | 说明 |
|------|------|------|
| `self.dist_task_info` | 对象 | 分布式任务信息 |
| `self.dist_task_info.rank` | int | 当前进程rank |
| `self.dist_task_info.world_size` | int | 总进程数 |
| `self.dist_task_info.is_bm` | bool | 是否为标杆模式（NPU小算子级联） |
| `self.name` | str | 执行后端名称，`"cpu"` / `"npu"` / `"pyaclnn"` 等 |
| `self.device` | str | 设备标识，`"pyaclnn"` 等 |

**关键方法**：

| 方法 | 说明 |
|------|------|
| `__call__(self, input_data, with_output=False)` | 标杆执行入口，需重写 |
| `init_by_input_data(self, input_data)` | 输入预处理，可重写 |
| `get_cpp_func_signature_type(self)` | 返回aclnn C++函数签名，用于参数校验 |

### 1.3 基类 `AclnnBaseApi`

```python
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
```

**构造参数**：`task_result, backend`

**关键属性**：

| 属性 | 类型 | 说明 |
|------|------|------|
| `self.task_result` | 对象 | 任务结果，含 `output_info_list` |
| `self.backend` | 对象 | ACLNN后端执行器 |

**关键方法**：

| 方法 | 说明 |
|------|------|
| `init_by_input_data(self, input_data)` | 默认参数转换（torch→ctypes），返回 `(input_args, output_packages)` |
| `__call__(self)` | 默认调用 `self.backend.aclnn_x_get_workspace_size()` + `self.backend.aclnn_x()` |
| `after_call(self, output_packages)` | 输出后处理，将aclTensor转回torch tensor |
| `acl_tensor_to_torch(self, output_pack)` | 单个AclTensorStruct→torch.Tensor |
| `int4_acl_tensor_to_torch(self, output_pack)` | ACL_INT4的AclTensor→torch.int8的Tensor |
| `torch_tensor_to_acl(self, tensor, fmt)` | torch.Tensor→AclTensorStruct |
| `get_format(self, input_data, index=None, name=None)` | 指定tensor的AclFormat，默认`ACL_FORMAT_ND` |
| `get_storage_shape(self, input_data, index=None, name=None)` | 指定非连续tensor的storage_shape |

### 1.4 输入数据结构 `InputDataset`

```python
from atk.configs.dataset_config import InputDataset
```

| 属性 | 类型 | 说明 |
|------|------|------|
| `input_data.args` | `List[Any]` | 位置参数列表 |
| `input_data.kwargs` | `Dict[str, Any]` | 关键字参数字典，按yaml中name索引 |

**典型用法**：
```python
x = input_data.kwargs['x']
weight = input_data.kwargs['weight']
```

### 1.5 输出数据结构 `OutputData`

通过 `self.task_result.output_info_list` 访问，每个元素：

| 属性 | 类型 | 说明 |
|------|------|------|
| `dtype` | str | Torch数据类型字符串，如 `"torch.float32"` |
| `shape` | `List[int]` | 张量维度 |
| `stride` | `List[int]` 或 None | 步长信息 |

---

## 二、CPU实现（真值计算）

CPU实现写在 `BaseApi.__call__` 的 `if self.name == "cpu"` 分支中。

### 2.1 分支判断

```python
def __call__(self, input_data: InputDataset, with_output: bool = False):
    if self.name == "cpu":
        # CPU真值逻辑
```

### 2.2 输入获取

从 `input_data.kwargs` 按参数名取值：

```python
x = input_data.kwargs['x']
weight = input_data.kwargs['weight']
bias = input_data.kwargs['bias']
hasBias = input_data.kwargs['hasBias']
```

### 2.3 通信域配置（`init_by_input_data`）

CPU分支本身不使用通信域，但 `init_by_input_data` 在 `pyaclnn` 设备下预注入group，供后续ACLNN使用：

```python
def init_by_input_data(self, input_data: InputDataset):
    if self.device == 'pyaclnn' and dist.is_available():
        from torch.distributed.distributed_c10d import _get_default_group
        default_pg = _get_default_group()
        input_data.kwargs['group'] = default_pg
```

---

## 三、NPU级联实现（小算子级联标杆）

NPU级联实现写在 `if self.dist_task_info.is_bm` 分支中。

### 3.1 分支判断

```python
if self.dist_task_info.is_bm:
    # NPU小算子级联标杆逻辑
```

### 3.2 `torch_npu` 专用算子

从用户提供的代码和用户的要求中推测。

---

## 四、ACLNN实现

ACLNN实现写在继承 `AclnnBaseApi` 的类中，核心是 `init_by_input_data` 的参数预处理。

### 4.1 `init_by_input_data` 流程

```
1. 从input_data.kwargs提取控制参数（hasBias/isSmoothQuant等，后续需pop掉）
2. 对输入做预处理（转置、类型转换等）
3. 调用 super().init_by_input_data(input_data) 执行默认torch→ctypes转换
4. 对返回的input_args做后处理（空指针替换、int4构造等）
5. 返回 (input_args, output_packages)
```

### 4.2 默认参数转换（`super().init_by_input_data`）

框架自动完成以下转换：

| Python类型 | → ctypes类型 | 条件 |
|-----------|-------------|------|
| `torch.Tensor` | `AclTensorStruct` | 自动 |
| `int` | `ctypes.c_int64` | yaml中 `type: attr` |
| `float` | `ctypes.c_float` | yaml中 `type: attr` |
| `bool` | `ctypes.c_bool` | yaml中 `type: attr` |
| `str` | `ctypes.c_char_p` | yaml中 `type: attr` |
| `None` | `ctypes.c_void_p` | 自动 |
| `List[int]` | `AclIntArray` | yaml中 `type: attrs` |
| `List[float]` | `AclFloatArray` | yaml中 `type: attrs` |
| `List[bool]` | `AclBoolArray` | yaml中 `type: attrs` |

**返回值**：
- `input_args`：`List[ctypes | AclTensorStruct]`，严格按C++函数签名顺序
- `output_packages`：`List[AclTensorStruct]`，算子输出数据包

### 4.3 通信域处理

将torch ProcessGroup转为hccl通信名：

```python
def init_by_input_data(self, input_data: InputDataset):
    rank_id = self.dist_task_info.rank
    self.group = input_data.kwargs['group']
    input_data.kwargs['group'] = self.group._get_backend(torch.device("npu")).get_hccl_comm_name(rank_id)
```

### 4.4 空指针构造

#### 空AclIntArray指针（Optional int数组置空）

```python
from atk.tasks.backends.lib_interface.acl_wrapper import AclIntArray

AclIntArrayPtr = ctypes.POINTER(AclIntArray)
null_void_ptr = ctypes.c_void_p(None)
null_array_ptr = ctypes.cast(null_void_ptr, AclIntArrayPtr)
input_args[9] = null_array_ptr
```

**使用场景**：aclnn接口中 `Optional` 后缀的IntArray参数，如 `alltoAllAxesOptional`。

#### 空AclTensor指针（简化写法，使用TensorPtr）

```python
from atk.tasks.backends.lib_interface.acl_wrapper import TensorPtr

input_args[4] = TensorPtr()
```

> `TensorPtr()` 等价于 `get_null_tensor_pte()`，写法更简洁。
**使用场景**：aclnn接口中 `Optional` 后缀的tensor参数，如 `biasOptional` 不存在时置空。

### 4.5 INT4 Tensor构造

ACLNN侧存在 `ACL_INT4` 数据类型，但torch没有 `torch.int4`，需用 `torch.int8` 的tensor来构造。

#### 创建INT4输入tensor

```python
from atk.tasks.backends.lib_interface.acl_wrapper import nnopbase, AclFormat

input_args[0] = nnopbase.create_int4_acl_tensor(
    input_data.kwargs['x1'],
    fmt=AclFormat.ACL_FORMAT_ND,
    storage_shape=None
)
torch.npu.synchronize()
```

> **要求**：传入的 `torch.int8` tensor取值范围必须在 `[-8, 7]` 之内。

#### 创建INT4输出tensor

```python
output_data = self.task_result.output_info_list[-1]
dtype = output_data.dtype
torch_dtype = getattr(torch, dtype.replace("torch.", ""))
if output_data.stride:
    empty_tensor = torch.empty_strided(output_data.shape, output_data.stride,
                                        dtype=torch_dtype, device='npu')
else:
    empty_tensor = torch.empty(output_data.shape, dtype=torch_dtype, device='npu')
empty_tensor = empty_tensor.fill_(0)

output = nnopbase.create_int4_acl_tensor(empty_tensor, fmt=AclFormat.ACL_FORMAT_ND, storage_shape=None)
torch.npu.synchronize()

output_packages[-1] = output
input_args[-1] = output
```

### 4.6 `after_call` 输出转换

#### 默认转换

```python
def after_call(self, output_packages):
    output = []
    for output_pack in output_packages:
        output.append(self.acl_tensor_to_torch(output_pack))
    return output
```

#### 含INT4输出的转换

```python
def after_call(self, output_packages):
    if len(output_packages) == 2:
        return [self.acl_tensor_to_torch(output_packages[0]),
                self.int4_acl_tensor_to_torch(output_packages[1])]
    return [self.acl_tensor_to_torch(output_packages[0])]
```

### 4.7 `__call__` 调用

默认实现（通常无需重写）：

```python
def __call__(self):
    self.backend.aclnn_x_get_workspace_size()
    self.backend.aclnn_x()
```

### 4.8 控制参数的pop模式

yaml中定义的控制参数（如 `hasBias`、`isSmoothQuant`、`quantType`、`hasAlltoall`）在ACLNN接口中不存在，需在 `init_by_input_data` 中先提取再pop：

```python
def init_by_input_data(self, input_data: InputDataset):
    hasBias = input_data.kwargs['hasBias']
    isSmoothQuant = input_data.kwargs.get('isSmoothQuant', False)
    input_data.kwargs.pop('hasBias')
    input_data.kwargs.pop('isSmoothQuant')

    input_args, output_packages = super().init_by_input_data(input_data)

    if not hasBias:
        input_args[2] = self.get_null_tensor_pte()
    ...
```

### 4.9 输入预处理（转置等）

在调用 `super().init_by_input_data` 之前对 `input_data.kwargs` 做预处理：

```python
if input_data.kwargs['transposeX2']:
    input_data.kwargs['weight'] = input_data.kwargs['weight'].T.clone()
```

### 4.10 参数校验 `get_cpp_func_signature_type`

返回aclnn算子的C++函数签名，用于框架自动校验参数类型和顺序：

```python
def get_cpp_func_signature_type(self):
    return """aclnnStatus aclnnAlltoAllMatmulGetWorkspaceSize(const aclTensor* x, const aclTensor* weight,
                const aclTensor* biasOptional, const char* group, const aclIntArray* alltoAllAxesOptional,
                bool transposeX1, bool transposeX2,
                aclTensor* output, aclTensor* all2allOutOptional, uint64_t* workspaceSize, aclOpExecutor** executor)"""
```

### 4.11 `get_format` 指定tensor格式

重写此方法可指定特定输入tensor的AclFormat：

```python
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat

def get_format(self, input_data: InputDataset, index=None, name=None):
    if name == "weight":
        return AclFormat.ACL_FORMAT_FRACTAL_NZ
    return AclFormat.ACL_FORMAT_ND
```

### 4.12 `get_storage_shape` 指定非连续tensor的storage shape

当tensor经过切片等操作变为非连续时，需指定原始shape作为storage_shape：

```python
def get_storage_shape(self, input_data: InputDataset, index=None, name=None):
    if name is not None:
        return self.original_kwargs[name].shape
    return None
```