# NormFlashAttention 算子设计文档

## 基本信息
- **算子名称**：NormFlashAttention
- **算子功能描述**：标准Flash Attention计算
- **SOC版本**：DAV_2201
- **算子仓类型**：ops-transformer
- **生成日期**：2026-04-28

## 输入参数
| 参数名 | 参数类型 | 支持的数据类型 | 支持的格式 | Shape 描述 | 说明 |
|--------|----------|----------------|------------|------------|------|
| query | REQUIRED | DT_FLOAT16, DT_FLOAT, DT_BF16 | FORMAT_ND | [B, S, H, D] | Query 输入 |
| key | REQUIRED | DT_FLOAT16, DT_FLOAT, DT_BF16 | FORMAT_ND | [B, S, H, D] | Key 输入 |
| value | REQUIRED | DT_FLOAT16, DT_FLOAT, DT_BF16 | FORMAT_ND | [B, S, H, D] | Value 输入 |

## 输出参数
| 参数名 | 参数类型 | 支持的数据类型 | 支持的格式 | Shape 描述 | 说明 |
|--------|----------|----------------|------------|------------|------|
| attentionOut | REQUIRED | DT_FLOAT16, DT_FLOAT, DT_BF16 | FORMAT_ND | [B, S, H, D] | Attention 输出 |

## 属性参数
无

## dtype 组合设计

### 组合规则
所有输入输出使用相同 dtype（类型一致原则）

### dtype 组合表
| 组合ID | query | key | value | attentionOut | 说明 |
|--------|-------|-----|-------|-------------|------|
| 0 | DT_FLOAT16 | DT_FLOAT16 | DT_FLOAT16 | DT_FLOAT16 | FP16 组合 |
| 1 | DT_FLOAT | DT_FLOAT | DT_FLOAT | DT_FLOAT | FP32 组合 |
| 2 | DT_BF16 | DT_BF16 | DT_BF16 | DT_BF16 | BF16 组合 |

### 生成配置（ascend910b）
- **总组合数**：3
- **DataType**：{ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16}
- **Format**：{ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND}

## 目录结构
```
norm_flash_attention/
├── CMakeLists.txt                    # 根CMakeLists（遍历子目录查找op_host/CMakeLists.txt）
├── op_host/
│   ├── CMakeLists.txt                # 编译配置
│   ├── norm_flash_attention_def.cpp  # 算子定义
│   ├── norm_flash_attention_infershape.cpp  # 形状推导
│   ├── norm_flash_attention_tiling.h  # Tiling数据结构定义
│   ├── op_api/
│   │   ├── aclnn_norm_flash_attention.h     # 公共API声明
│   │   ├── aclnn_norm_flash_attention_utils.h       # Handler基类、参数结构、Builder
│   │   ├── aclnn_norm_flash_attention_base_handler.h  # 基础SOC版本Handler实现
│   │   ├── aclnn_norm_flash_attention_process.cpp     # L2入口、Handler工厂路由
│   │   ├── norm_flash_attention.h     # L0算子声明
│   │   └── norm_flash_attention.cpp   # L0算子实现
│   └── op_tiling/
│       └── norm_flash_attention_tiling.cpp  # Tiling计算实现
├── op_kernel/
│   ├── norm_flash_attention.h        # Kernel头文件
│   └── norm_flash_attention.cpp      # Kernel实现
└── NormFlashAttention.md             # 本设计文档
```

## 其他说明
- 特殊约束条件：无
- 参考算子：无（仅参考插件模板）
