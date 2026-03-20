# RotateDynamicQuant 算子设计文档

## 基本信息
- **算子名称**：RotateDynamicQuant
- **算子功能描述**：融合旋转和量化需求，实现旋转位置编码后的动态量化
- **SOC版本**：DAV_2201
- **算子仓类型**：ops-transformer
- **生成日期**：2026-03-19

## 输入参数
| 参数名 | 参数类型 | 支持的数据类型 | 支持的格式 | Shape 描述 | 说明 |
|--------|----------|----------------|------------|------------|------|
| x | REQUIRED | DT_FLOAT16, DT_BFLOAT16 | FORMAT_ND | [M, N] | 进行int4量化时，要求形状最后一维是8的整数倍，支持非连续tensor |
| rot | REQUIRED | DT_FLOAT16, DT_BFLOAT16 | FORMAT_ND | [N, N] | 两个维度大小相同，数据类型与x保持一致，支持非连续tensor |

## 输出参数
| 参数名 | 参数类型 | 支持的数据类型 | 支持的格式 | Shape 描述 | 说明 |
|--------|----------|----------------|------------|------------|------|
| yOut | REQUIRED | DT_INT8, DT_INT32 | FORMAT_ND | [M, N] | shape与输入x一致 |
| scaleOut | REQUIRED | DT_FLOAT32 | FORMAT_ND | [M] | 表示动态量化过程中计算出的缩放系数 |

## 属性参数
| 属性名 | 属性类型 | 默认值 | 说明 |
|--------|----------|--------|------|
| dst_type | int32_t | 0 | 指定输出y的数据类型，0=DT_INT8, 1=DT_INT32 |

## dtype 组合设计

### 组合规则
1. x 和 rot 数据类型必须一致
2. scaleOut 固定为 DT_FLOAT32
3. yOut 的 dtype 由属性 dst_type 决定（0=DT_INT8, 1=DT_INT32）

### dtype 组合表
| 组合ID | x | rot | yOut | scaleOut | 说明 |
|--------|---|-----|------|----------|------|
| 0 | DT_FLOAT16 | DT_FLOAT16 | DT_INT8 | DT_FLOAT32 | FP16输入 → INT8输出 |
| 1 | DT_FLOAT16 | DT_FLOAT16 | DT_INT32 | DT_FLOAT32 | FP16输入 → INT32输出 |
| 2 | DT_BFLOAT16 | DT_BFLOAT16 | DT_INT8 | DT_FLOAT32 | BF16输入 → INT8输出 |
| 3 | DT_BFLOAT16 | DT_BFLOAT16 | DT_INT32 | DT_FLOAT32 | BF16输入 → INT32输出 |

### 生成配置
- **总组合数**：4
- **DataType**：{ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16}
- **Format**：{ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND}

## 目录结构
```
rotate_dynamic_quant/
├── CMakeLists.txt
├── RotateDynamicQuant.md          # 算子设计文档
├── op_host/
│   ├── CMakeLists.txt
│   ├── rotate_dynamic_quant_def.cpp
│   ├── rotate_dynamic_quant_infershape.cpp
│   ├── op_api/
│   │   ├── aclnn_rotate_dynamic_quant.h
│   │   ├── aclnn_rotate_dynamic_quant_utils.h
│   │   ├── aclnn_rotate_dynamic_quant_base_handler.h
│   │   ├── aclnn_rotate_dynamic_quant_process.cpp
│   │   ├── rotate_dynamic_quant.h
│   │   └── rotate_dynamic_quant.cpp
│   └── op_tiling/
│       ├── rotate_dynamic_quant_tiling.h
│       └── rotate_dynamic_quant_tiling.cpp
├── op_kernel/
│   ├── rotate_dynamic_quant.cpp
│   └── rotate_dynamic_quant.h
└── tests/
    └── ut/
        ├── op_api/
        │   └── test_aclnn_rotate_dynamic_quant.cpp
        └── op_host/
            ├── test_rotate_dynamic_quant_infershape.cpp
            └── test_rotate_dynamic_quant_tiling.cpp
```

## API 接口

### aclnnRotateDynamicQuantGetWorkspaceSize
```cpp
aclnnStatus aclnnRotateDynamicQuantGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *rot,
    int32_t dst_type,
    const aclTensor *yOut,
    const aclTensor *scaleOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);
```

### aclnnRotateDynamicQuant
```cpp
aclnnStatus aclnnRotateDynamicQuant(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## 特殊约束条件
1. 进行 int4 量化时，要求输入 x 的最后一维是 8 的整数倍
2. 支持非连续 tensor 输入
3. rot 必须是方阵 [N, N]
4. x 和 rot 的数据类型必须一致
5. yOut 的 shape 必须与 x 一致
6. scaleOut 的第一维必须等于 x 的第一维 M

## 参考算子
无（使用插件模板生成）

## 待完善项
- [ ] Kernel 实现核心计算逻辑
- [ ] Tiling 数据切分策略优化
- [ ] 单元测试用例
