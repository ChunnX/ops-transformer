# RotateDynamicQuant 算子设计文档

## 基本信息
- **算子名称**：RotateDynamicQuant
- **算子功能描述**：融合旋转和量化需求，实现旋转位置编码后的动态量化
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
| dst_type | int32_t | 0 | 指定输出y的数据类型，0表示DT_INT8，1表示DT_INT32 |

## dtype 组合设计

### 组合规则
- x 和 rot 必须使用相同 dtype（类型一致原则）
- yOut 根据属性 dst_type 决定输出 INT8 或 INT32
- scaleOut 固定为 DT_FLOAT

### dtype 组合表
| 组合ID | x | rot | yOut | scaleOut | 说明 |
|--------|---|-----|------|----------|------|
| 0 | DT_FLOAT16 | DT_FLOAT16 | DT_INT8 | DT_FLOAT | FLOAT16输入 + INT8量化输出 |
| 1 | DT_FLOAT16 | DT_FLOAT16 | DT_INT32 | DT_FLOAT | FLOAT16输入 + INT32量化输出 |
| 2 | DT_BFLOAT16 | DT_BFLOAT16 | DT_INT8 | DT_FLOAT | BFLOAT16输入 + INT8量化输出 |
| 3 | DT_BFLOAT16 | DT_BFLOAT16 | DT_INT32 | DT_FLOAT | BFLOAT16输入 + INT32量化输出 |

### 生成配置
- **总组合数**：4
- **x DataType**: {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BFLOAT16, ge::DT_BFLOAT16}
- **rot DataType**: {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BFLOAT16, ge::DT_BFLOAT16}
- **yOut DataType**: {ge::DT_INT8, ge::DT_INT32, ge::DT_INT8, ge::DT_INT32}
- **scaleOut DataType**: {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT}
- **Format**: 全部使用 FORMAT_ND

## 目录结构
```
rotate_dynamic_quant/
├── CMakeLists.txt                              # 根目录 CMake 配置（遍历结构）
├── RotateDynamicQuant.md                       # 算子设计文档
└── op_host/
    ├── CMakeLists.txt                          # op_host 编译配置
    ├── rotate_dynamic_quant_def.cpp            # 算子定义（dtype组合）
    ├── rotate_dynamic_quant_infershape.cpp     # 形状推导
    ├── op_api/
    │   ├── aclnn_rotate_dynamic_quant.h        # 公共 API 声明
    │   ├── aclnn_rotate_dynamic_quant_utils.h  # Handler 基类、参数结构、Builder
    │   ├── aclnn_rotate_dynamic_quant_base_handler.h  # 基础 SOC 版本 Handler 实现
    │   ├── aclnn_rotate_dynamic_quant_process.cpp     # L2 入口、Handler 工厂路由
    │   ├── rotate_dynamic_quant.h              # L0 算子声明
    │   └── rotate_dynamic_quant.cpp            # L0 算子实现
    ├── op_tiling/
    │   ├── rotate_dynamic_quant_tiling.h       # Tiling 数据结构定义
    │   └── rotate_dynamic_quant_tiling.cpp     # Tiling 计算实现
    └── op_kernel/
        ├── rotate_dynamic_quant.h              # Kernel 头文件
        └── rotate_dynamic_quant.cpp            # Kernel 实现（待完善）
```

## 其他说明
- **特殊约束条件**：
  - x 的最后一维 N 必须与 rot 的两个维度相同
  - 进行 int4 量化时，要求形状最后一维是 8 的整数倍
  - 支持非连续 tensor 输入
- **参考算子**：无（仅参考插件模板）

## 待完成项
1. **L0 算子实现** (`rotate_dynamic_quant.cpp`)：需要完善实际的计算逻辑
2. **Kernel 实现** (`op_kernel/rotate_dynamic_quant.cpp`)：需要实现核心计算逻辑
   - 数据加载
   - 旋转计算（矩阵乘法）
   - 动态量化计算
   - 数据写回
3. **Tiling 策略优化**：根据实际算子特性优化 tiling 策略
