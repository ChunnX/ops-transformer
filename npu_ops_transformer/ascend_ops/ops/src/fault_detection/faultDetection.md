# fault_detection

**FaultDetection** - 服务器可用性检测算子

## 项目简介 | Introduction

FaultDetection 是一个适用于昇腾NPU集群故障检测的检测工具

## 核心特性 | Features

故障发生时可快速检测集群内可用节点，进行快速故障恢复

## 安装步骤 | Installation

1. 进入目录，从源码构建.whl包

   ```sh
   cd npu_ops_transformer
   python -m build --wheel -n
   ```

2. 安装构建好的.whl包

   ```sh
   pip install dist/xxx.whl
   ```

3. (可选)再次构建前请先执行以下命令清理编译缓存

## 使用示例 | Usage Example

您可以在`npu_ops_transformer\examples`目录下找到fault_detection的测试脚本以验证功能:

最终看到如下输出，即为执行成功：

```
tensor([1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int32)
detection run npu success
```