# RotateDynamicQuant 算子设计文档

## 基本信息
- **算子名称**：RotateDynamicQuant
- **算子功能描述**：融合旋转位置编码和动态量化的算子
- **SOC版本**：DAV_2201
- **算子仓类型**：ops-transformer
- **生成日期**：2026-03-20

## 输入参数
| 参数名 | 参数类型 | 支持的数据类型 | 支持的格式 | Shape 描述 | 说明 |
|--------|----------|----------------|------------|------------|------|
| x | REQUIRED | DT_FLOAT16, DT_BFLOAT16 | FORMAT_ND | [M, N] | 进行int4量化时，要求形状最后一维是8的整数倍，支持非连续tensor |
| rot | REQUIRED | DT_FLOAT16, DT_BFLOAT16 | FORMAT_ND | [N, N] | 两个维度大小相同，数据类型与x保持一致，支持非连续tensor |

## 输出参数
| 参数名 | 参数类型 | 支持的数据类型 | 支持的格式 | Shape 描述 | 说明 |
|--------|----------|----------------|------------|------------|------|
| yOut | REQUIRED | DT_INT8, DT_INT32 | FORMAT_ND | [M, N] | shape与输入x一致 |
| scaleOut | REQUIRED | DT_FLOAT | FORMAT_ND | [M] | 表示动态量化过程中计算出的缩放系数 |

## 属性参数
| 属性名 | 属性类型 | 默认值 | 说明 |
|--------|----------|--------|------|
| dst_type | int32_t | 0 | 指定输出yOut的数据类型，0表示DT_INT8，1表示DT_INT32 |

## dtype 组合设计

### 组合规则
- 输入 x 和 rot 数据类型必须保持一致
- 输出 scaleOut 固定为 DT_FLOAT
- 输出 yOut 数据类型由 dst_type 属性决定

### dtype 组合表
| 组合ID | x | rot | yOut | scaleOut | 说明 |
|--------|---|-----|------|----------|------|
| 0 | DT_FLOAT16 | DT_FLOAT16 | DT_INT8 | DT_FLOAT | FLOAT16输入，INT8量化输出 |
| 1 | DT_FLOAT16 | DT_FLOAT16 | DT_INT32 | DT_FLOAT | FLOAT16输入，INT32量化输出 |
| 2 | DT_BFLOAT16 | DT_BFLOAT16 | DT_INT8 | DT_FLOAT | BFLOAT16输入，INT8量化输出 |
| 3 | DT_BFLOAT16 | DT_BFLOAT16 | DT_INT32 | DT_FLOAT | BFLOAT16输入，INT32量化输出 |

### 生成配置
- **总组合数**：4
- **DataType**：
  - x: `{ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16}`
  - rot: `{ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16}`
  - yOut: `{ge::DT_INT8, ge::DT_INT32, ge::DT_INT8, ge::DT_INT32}`
  - scaleOut: `{ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT}`
- **Format**：全部使用 `ge::FORMAT_ND`

## 目录结构
```
rotate_dynamic_quant/
├── CMakeLists.txt
├── RotateDynamicQuant.md
├── op_host/
│   ├── CMakeLists.txt
│   ├── rotate_dynamic_quant_def.cpp
│   ├── rotate_dynamic_quant_infershape.cpp
│   ├── rotate_dynamic_quant_tiling.h
│   ├── rotate_dynamic_quant_tiling.cpp
│   └── op_api/
│       ├── aclnn_rotate_dynamic_quant.h
│       ├── aclnn_rotate_dynamic_quant_utils.h
│       ├── aclnn_rotate_dynamic_quant_base_handler.h
│       ├── aclnn_rotate_dynamic_quant_process.cpp
│       ├── rotate_dynamic_quant.h
│       └── rotate_dynamic_quant.cpp
└── op_kernel/
    ├── rotate_dynamic_quant.h
    └── rotate_dynamic_quant.cpp
```

## 文件说明

### op_host 目录
| 文件名 | 说明 |
|--------|------|
| rotate_dynamic_quant_def.cpp | 算子定义文件，定义输入输出参数、数据类型、格式等 |
| rotate_dynamic_quant_infershape.cpp | 形状推导实现，计算输出tensor的shape和dtype |
| rotate_dynamic_quant_tiling.h | Tiling数据结构定义 |
| rotate_dynamic_quant_tiling.cpp | Tiling计算实现 |

### op_host/op_api 目录
| 文件名 | 说明 |
|--------|------|
| aclnn_rotate_dynamic_quant.h | ACLNN公共API声明 |
| aclnn_rotate_dynamic_quant_utils.h | Handler基类、参数结构、Builder模式 |
| aclnn_rotate_dynamic_quant_base_handler.h | DAV_2201版本Handler实现 |
| aclnn_rotate_dynamic_quant_process.cpp | L2入口、Handler工厂路由 |
| rotate_dynamic_quant.h | L0算子声明 |
| rotate_dynamic_quant.cpp | L0算子实现 |

### op_kernel 目录
| 文件名 | 说明 |
|--------|------|
| rotate_dynamic_quant.h | Kernel头文件，定义Kernel类和数据结构 |
| rotate_dynamic_quant.cpp | Kernel实现入口函数 |

## 约束条件
- x的最后一维N必须是rot的两个维度大小
- rot的两个维度必须相等
- 进行int4量化时，要求x的最后一维是8的整数倍
- x和rot的数据类型必须一致

## 参考算子
无（基于empty_op_template模板生成）
