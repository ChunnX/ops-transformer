# AscendOps

**AscendOps** - 一个轻量级，高性能的算子开发工程模板

## 项目简介 | Introduction
AscendOps 是一个轻量级，高性能的算子开发工程模板，它集成了PyTorch、PyBind11和昇腾CANN工具链，提供了从算子内核编写，编译到Python封装的完整工具链。

## 核心特性 | Features
🚀 开箱即用 (Out-of-the-Box): 预置完整的昇腾NPU算子开发环境配置，克隆后即可开始开发。

🧩 极简设计 (Minimalist Design): 代码结构清晰直观，专注于核心算子开发流程。

⚡ 高性能 (High Performance): 基于AscendC编程模型，充分发挥昇腾NPU硬件能力。

📦 一键部署 (One-Click Deployment): 集成setuptools构建系统，支持一键编译和安装。

🔌 PyTorch集成 (PyTorch Integration): 无缝集成PyTorch张量操作，支持自动微分和GPU/NPU统一接口。

## 核心交付件 | Core Deliverables
1. `csrc/xxx/xxx_torch.cpp` 算子Kernel实现
2. `csrc/xxx/CMakeLists.txt` 算子cmake配置
3. `csrc/npu_ops_def.cpp` 注册算子接口

## 环境要求 | Prerequisites
*   Python: 3.8+
*   CANN Ascend Toolkit
*   PyTorch: 2.1.0+
*   PyTorchAdapter

## 环境准备 | Preparation

1. **安装社区版CANN包**
   
    请参考[算子调用指南](../../docs/zh/invocation/quick_op_invocation.md)的环境准备章节，安装CANN toolkit包和CANN legacy包，并配置好环境变量。
    

2. **安装torch与torch_npu包**
   
   根据实际环境，下载对应torch包并安装: `torch-${torch_version}+cpu-${python_version}-linux_${arch}.whl` 下载链接为:[官网地址](http://download.pytorch.org/whl/torch)

   安装命令如下：

    ```sh
    pip install torch-${torch_version}+cpu-${python_version}-linux_${arch}.whl
    ```

   根据实际环境，安装对应torch-npu包: `torch_npu-${torch_version}-${python_version}-linux_${arch}.whl`

   可以直接使用pip命令下载安装，命令如下：

    ```sh
    pip install torch_npu
    ```
    
    - \$\{torch\_version\}：表示torch包版本号。
    - \$\{python\_version\}：表示python版本号。
    - \$\{arch\}：表示CPU架构，如aarch64、x86_64。

## 安装步骤 | Installation

1. 进入目录，安装依赖
    ```sh
    cd fast_kernel_launch_example
    pip install -r requirements.txt
    ```

2. 从源码构建.whl包
    ```sh
    python -m build --wheel -n
    ```

3. 安装构建好的.whl包
    ```sh
    pip install dist/xxx.whl
    ```

    重新安装请使用以下命令覆盖已安装过的版本：
    ```sh
    pip install dist/xxx.whl --force-reinstall --no-deps
    ```

4. （可选）再次构建前建议先执行以下命令清理编译缓存
   ```sh
    python setup.py clean
    ```

## 开发模式构建 | Developing Mode

此命令实现即时生效的开发环境配置，执行后即可使源码修改生效，省略了构建完整whl包和安装的过程，适用于需要多次修改验证算子的场景：
  ```sh
  pip install --no-build-isolation -e .
  ```

## 使用示例 | Usage Example

安装完成后，您可以像使用普通PyTorch操作一样使用NPU算子，以groupedmatmul算子为例，您可以在`ascend_ops\csrc\grouped_matmul\test`目录下找到并执行这个脚本:

```python
import torch
import torch_npu
import ascend_ops

supported_dtypes = {torch.bfloat16}
E = 3
M = 64
K = 32
N = 32
PERGROUP = M

def generate_group_list_tensor(E, M, PERGROUP):
    group_list = torch.zeros(E, dtype=torch.int64)
    total_group = PERGROUP
    
    for i in range(E):
        if total_group <= M:
            group_list[i] = PERGROUP
            total_group += PERGROUP
        else:
            group_list[i] = PERGROUP - (total_group - M)
            break
    
    return group_list

group_list = generate_group_list_tensor(E, M, PERGROUP)

EPS = 0.001

for data_type in supported_dtypes:
    print(f"DataType = <{data_type}>")
    
    x_list = []
    x_cpu = torch.rand(M, K, dtype=data_type)
    x_list.append(x_cpu)

    weight_list = []
    weight_cpu = torch.rand(K, N, dtype=data_type)
    weight_list.append(weight_cpu)
    
    # 将数据移动到NPU
    x_list_npu = [x_i.npu() for x_i in x_list]
    weight_list_npu = [weight_i.npu() for weight_i in weight_list]
    group_list_npu = group_list.npu()
    
    # 调用groupedmatmul，提供所有必需的参数
    try:
        npu_result = ascend_ops.ops.groupedmatmul(
            x_list_npu,           # Tensor[] x
            weight_list_npu,      # Tensor[] weight
            None,                 # Tensor[]? bias (可选)
            None,                 # Tensor[]? scale (可选)
            None,                 # Tensor[]? offset (可选)
            None,                 # Tensor[]? antiquantScale (可选)
            None,                 # Tensor[]? antiquantOffset (可选)
            group_list_npu,       # Tensor? groupList
            None,                 # Tensor[]? perTokenScale (可选)
            3,                    # int splitItem
            0,                    # int groupType
            1,                    # int groupListType
            0,                    # int actType
            None                  # int[]? tuningConfigOptional (可选)
        ).cpu()
        
        print(f"Result shape: {npu_result.shape} \n",npu_result)
        
    except Exception as e:
        print(f"Error calling groupedmatmul: {e}")
    
    torch_result = None
    try:
        x = x_list[0].to(torch.float32)
        weight = weight_list[0].to(torch.float32)
        
        group_size = M // E
        remainder = M % E
        split_sizes = [group_size] * E
        if remainder > 0:
            split_sizes[-1] += remainder
        
        x_splits = torch.split(x, split_sizes, dim=0)
        
        split_results = []
        for x_split in x_splits:
            split_matmul = torch.matmul(x_split, weight)
            split_results.append(split_matmul)
        
        torch_result = torch.cat(split_results, dim=0).to(data_type)
        print(f"PyTorch Result shape: {torch_result.shape}")
        print(f"分组大小: {split_sizes}, 各分组结果形状: {[r.shape for r in split_results]}")
    except Exception as e:
        print(f"Error calculating PyTorch grouped matmul: {e}")
        continue

    try:
        if npu_result.shape != torch_result.shape:
            print(f"Shape mismatch! NPU: {npu_result.shape}, PyTorch: {torch_result.shape}")
            print("精度对比失败")
            continue
        
        npu_float = npu_result.to(torch.float32)
        torch_float = torch_result.to(torch.float32)
        
        abs_diff = torch.abs(npu_float - torch_float)
        bad_indices = torch.where(abs_diff > EPS)
        bad_values = abs_diff[bad_indices]
        
        if len(bad_values) > 0:
            print(f"\n精度对比失败！发现 {len(bad_values)} 个点位差值超过 {EPS}:")
            for idx in range(min(len(bad_values), 10)):
                i, j = bad_indices[0][idx].item(), bad_indices[1][idx].item()
                diff = bad_values[idx].item()
                npu_val = npu_float[i, j].item()
                torch_val = torch_float[i, j].item()
                print(f"点位({i}, {j}): NPU={npu_val:.6f}, PyTorch={torch_val:.6f}, 差值={diff:.6f}")
            print("精度对比失败！")
        else:
            max_diff = torch.max(abs_diff).item()
            print(f"\n精度对比通过！最大绝对误差: {max_diff:.6f} (阈值={EPS})")
    except Exception as e:
        print(f"Error comparing results: {e}")
```

最终输出包含以下信息，即为执行成功：
```bash
精度对比通过！
```


## 开发新算子 | Developing New Operators
1. 编写算子调用文件，以添加算子my_ops为例
   
    在 `csrc` 目录下添加新的算子目录 `my_ops`，在 `my_ops` 目录下添加新的算子调用文件 `my_ops_torch.cpp`
    ```c++
    __global__ __aicore__ void mykernel(GM_ADDR input, GM_ADDR output, int64_t num_element) {
        // 您的算子kernel实现
    }

    void my_ops_api(aclrtStream stream, const at::Tensor& x, const at::Tensor& y) {
        // 您的算子入口实现，在该方法中使用<<<>>>的方式调用算子kernel
        mykernel<<<blockDim, nullptr, stream>>>(x, y, num_element);
    }

    torch::Tensor my_ops_npu(torch::Tensor x, torch::Tensor y) {
        // 您的算子wrapper接口，用于向pytorch注册自定义接口
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::kHalf, at::kBFloat16, x.scalar_type(), "my_ops_npu", [&] { my_ops_api(stream, x, y); });
    }

    // PyTorch提供的宏，用于在特定后端注册算子
    TORCH_LIBRARY_IMPL(ascend_ops, PrivateUse1, m)
    {
        m.impl("my_ops", my_ops_npu);
    }
    ```

2. 在`my_ops`目录下创建`CMakeLists.txt`
   
    ```cmake
    if (BUILD_TORCH_OPS)
        # 使用您的实际算子名替换my_ops
        set(OPERATOR_NAME "my_ops")
        message(STATUS "BUILD_TORCH_OPS ON in ${OPERATOR_NAME}")
        
        set(OPERATOR_TARGET "${OPERATOR_NAME}_objects")
        set(OPERATOR_CONFIG "${OPERATOR_NAME}:${OPERATOR_TARGER}" PARENT_SCOPE)

        file(GLOB OPERATOR_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

        # Mark .cpp files with special properties
        set_source_files_properties(
            ${OPERATOR_SOURCES} PROPERTIES
            LANGUAGE CXX
            COMPILE_FLAGS "--cce-soc-version=Ascend910B1 --cce-soc-core-type=VecCore --cce-auto-sync -xcce"
        )

        add_library(${OPERATOR_TARGET} OBJECT ${OPERATOR_SOURCES})

        target_compile_options(${OPERATOR_TARGET} PRIVATE ${COMMON_COMPILE_OPTIONS})
        target_include_directories(${OPERATOR_TARGET} PRIVATE ${COMMON_INCLUDE_DIRS})
        return()
    endif()
    ```

3. 在 `csrc/npu_ops_def.cpp`中添加TORCH_LIBRARY_IMPL定义
   
    ```c++
    TORCH_LIBRARY_IMPL(ascend_ops, PrivateUse1, m) {
        m.impl("my_ops", my_ops_npu);
    }
    ```

4. （可选）在 `ascend_ops/ops.py`中封装自定义接口
    ```python
    def my_ops(x: Tensor) -> Tensor:
        return torch.ops.ascend_ops.my_ops.default(x)
    ```

5. 使用开发模式进行编译
    ```bash
    pip install --no-build-isolation -e .
    ```

6. 编写测试脚本并测试新算子
    ```python
    torch.ops.ascend_ops.my_ops(x)
    ```
