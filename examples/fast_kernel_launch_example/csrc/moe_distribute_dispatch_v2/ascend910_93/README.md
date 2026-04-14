# moe_distribute_dispatch_v2

## 简介

环境部署、安装步骤等信息参考[README.md](../../../README.md)。

## 约束说明

想要完整使能功能，需要满足以下版本要求
- gcc 9.4.0+
- python 3.9+
- torch>=2.7.1
- cann 9.0.0+

该算子支持的平台为ascend910_93，默认构建环境为ascend910b，在ascend910_93构建环境下构建，请先执行命令：
```sh
export NPU_ARCH=ascend910_93
```
