# AscendC Matmul 算子开发迭代 SRS 模板

## 1. 迭代信息

| 项目 | 内容 |
| --- | --- |
| 迭代名称 | QuantGroupedMatmulInplaceAdd支持HIFP8 --Torch单算子/Torch图模式/aclnn |
| 算子类型 | QuantGroupedMatmulInplaceAdd |
| 迭代负责人 | 冯兴盼 |
| 更新时间 | 2026.04.03 |
| 迭代版本 | c11-迭代1 |

## 2. 需求背景

QuantGroupedMatmulInplaceAdd算子增加hifp8输入T-T量化场景支持盘古自定义数据类型训练

## 3. 方案简述



## 4. 输入输出规格

### 4.1 输入规格

| 输入项 | 说明 |
| --- | --- |
| 量化场景 | 动态T-T |
| dtype | x1:hifloat8 x2:hifloat8 scale1:float scale2:float y:float |
| shape | x1:(k,m) x2:(k,n) scale1:(g,1)/(g,) scale2:(g,1)/(g,) y:(g,m,n) |
| transpose | true-False |
| format | ND |

### 4.2 输出规格

| 输出项 | 说明 |
| --- | --- |
| shape | (g,m,n) |
| dtype | float |
| format | ND |

## 5. 交付标准

| 交付范围 | 当前迭代 |
| --- | --- |
| 交付通路 | Torch单算子/Torch图模式/aclnn |
| 动静要求 | 动态、动静归一 |
| 精度交付标准 | 输出fp32满足单边万分之一（单点精度误差不超过十分之一） |
| 性能交付标准 | 网络case融合后比小算子平均收益0.2*min(QuantGroupedMatmul、InplaceAdd)；泛化case平均性能不差于融合前 |
| 客户场景索引 |  |
| 相关设计文档索引 | 方案设计-模板设计 |
| 维测相关索引 |  |

## 6. 具体方案描述

### 6.1 Tiling

填写 Tiling 方案，可包括：

1. 输入约束。
2. 切分策略。
3. 命中 shape。
4. 限制说明。

![Tiling 示例](./figure/example_tiling.png)

### 6.2 Kernel

填写 Kernel 方案，可包括：

1. 执行流程。
2. 关键逻辑。
3. Buffer 和数据流。
4. 异常处理。

## 7. 软件设计和接口定义

### 7.1 软件设计

| 模块 | 说明 |
| --- | --- |
| Host | 填写 Host 职责。 |
| Tiling | 填写 Tiling 职责。 |
| Kernel | 填写 Kernel 职责。 |
| 测试 | 填写测试职责。 |

### 7.2 接口定义

| 接口项 | 说明 |
| --- | --- |
| Host 接口 | 填写 Host 接口。 |
| Tiling 接口 | 填写 Tiling 接口。 |
| Kernel 接口 | 填写 Kernel 接口。 |
| 对外接口 | 填写对外接口。 |

## 8. 外部依赖

| 依赖项 | 依赖内容 | 影响 |
| --- | --- | --- |
| CANN / AscendC 开发环境 | 填写依赖内容。 | 高/中/低 |
| 目标芯片与运行时环境 | 填写依赖内容。 | 高/中/低 |
| Matmul 相关官方能力或推荐实现方式 | 填写依赖内容。 | 高/中/低 |
| 构建与集成链路 | 填写依赖内容。 | 高/中/低 |
| 测试框架与基线数据 | 填写依赖内容。 | 高/中/低 |
| 日志与问题定位手段 | 填写依赖内容。 | 高/中/低 |

## 9. 待确认事项

1. 填写待确认事项 1。
2. 填写待确认事项 2。
3. 填写待确认事项 3。
4. 填写待确认事项 4。
5. 填写待确认事项 5。
