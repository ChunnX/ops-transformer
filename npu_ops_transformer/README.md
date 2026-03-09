# npu-ops-transformer

## 概述

npu-ops-transformer通过Pybind11为Aclnn提供Python绑定，提供了一个简化的接口利用华为Ascend Aclnn的功能。该项目无需手动配置CMake和构建C/C++可执行文件。提供对Aclnn接口的直接Python访问，无需依赖torch_npu wheel包。

## 项目特点

1. 直接通过Python访问Aclnn接口。
2. 无需手动配置CMake或C++。
3. 依赖极小：仅需CANN环境、Pytorch和torch_npu。
4. 提供预构建的wheel包，便于安装。
5. 无缝集成现有的Pytorch工作流程。

## 配置要求

* Python 3.8 或更高版本
* Pytorch 2.0 或更高版本
* torch_npu （最新兼容版本）
* CANN （最新版最佳）
* Ninja >= 1.13
* GCC （X86架构 >= 9.3.1，ARM架构 >= 10.2）

## 安装

此项目不需要安装，可直接使用；如果使用wheel打包安装，可以参考以下流程：
```
cd npu_ops_transformer
python -m build --wheel -n
pip install dist/xxx.whl
```
再次构建前请先清理编译缓存

## 目录结构
关键目录如下。
```
├── ascend_ops
│   ├── __init__.py
│   └── ops
│       ├── inc
│       │   ├── op_api_common.h
│       │   ├── op_errno.h
│       │   └── op_log.h
│       ├── __init__.py
│       └── src
│           ├── fault_detection
│           │   ├── faultDetection.md
│           │   └── detection_torch.cpp
│           ├── ...
│           └── ops_def.cpp
├── docs
│   ├── npu_ops_transformer_api_list.md
│   └── README.md
├── examples
│   ├── test_detection.py
│   └── ...
├── pyproject.toml
└── README.md
```
