# NormFlashAttention 算子信息

## 基本信息
- **算子名称**：NormFlashAttention
- **算子功能描述**：标准Flash Attention计算
- **生成路径**：M:\code\ops-transformer_Main\attention\norm_flash_attention
- **SOC版本**：DAV_2201（默认）或其他 _____________

## 输入参数
| 参数名 | 参数类型 | 支持的数据类型 | 支持的格式 | Shape 描述 | 说明 |
|--------|----------|----------------|------------|------------|------|
| query | REQUIRED | DT_FLOAT16, DT_FLOAT, DT_BF16 | FORMAT_ND | [B, S, H, D] | Query 输入 |
| key | REQUIRED | DT_FLOAT16, DT_FLOAT, DT_BF16 | FORMAT_ND | [B, S, H, D] | Key 输入 |
| value | REQUIRED | DT_FLOAT16, DT_FLOAT, DT_BF16 | FORMAT_ND | [B, S, H, D] | Value 输入 |

**参数类型说明**：
- REQUIRED：必需参数
- OPTIONAL：可选参数
- DYNAMIC：动态输入（可变数量）

## 输出参数
| 参数名 | 参数类型 | 支持的数据类型 | 支持的格式 | Shape 描述 | 说明 |
|--------|----------|----------------|------------|------------|------|
| attentionOut | REQUIRED | DT_FLOAT16, DT_FLOAT, DT_BF16 | FORMAT_ND | [B, S, H, D] | Attention 输出 |

## dtype 支持（用于 AI 生成组合）
请为每个输入输出参数列出支持的 dtype 列表（AI 将自动生成有效组合）：

| 参数名 | 支持的数据类型 |
|--------|----------------|
| query | DT_FLOAT16, DT_FLOAT, DT_BF16 |
| key | DT_FLOAT16, DT_FLOAT, DT_BF16 |
| value | DT_FLOAT16, DT_FLOAT, DT_BF16 |
| attentionOut | DT_FLOAT16, DT_FLOAT, DT_BF16 |

**说明**：
- 只需填写每个参数支持的 dtype 列表
- AI 将自动生成所有有效的 dtype 组合初稿
- 您可以在后续步骤中确认/修改/补充组合

## 其他说明
- 特殊约束条件：_____________
- 参考算子：_____________（如需参考现有算子，请填写算子名称；**不填则仅参考插件模板**）
