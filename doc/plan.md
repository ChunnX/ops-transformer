# OP_LOGE 日志整改任务计划

## 任务范围
mc2 目录下的 OP_LOGE 调用替换为预定义的 EZ 宏。

## 任务状态

### 已完成
- [x] mc2/matmul_allto_all/op_graph/ - fallback_matmul_allto_all.cpp (5处)
- [x] mc2/allto_all_matmul/op_graph/ - fallback_allto_all_matmul.cpp (5处)
- [x] mc2/quant_grouped_mat_mul_allto_allv/op_graph/ - fallback_quant_grouped_mat_mul_allto_allv.cpp (18处)
- [x] mc2/allto_allv_quant_grouped_mat_mul/op_graph/ - fallback_allto_allv_quant_grouped_mat_mul.cpp (17处)
- [x] mc2/allto_allv_quant_grouped_mat_mul/op_host/op_tiling/arch35/allto_allv_quant_grouped_mat_mul_tiling_common.cpp (3处)
- [x] mc2/allto_allv_quant_grouped_mat_mul/op_host/op_tiling/allto_allv_quant_grouped_mat_mul_tiling_base.cpp (32处)
- [x] mc2/allto_allv_quant_grouped_mat_mul/op_host/op_tiling/arch35/allto_allv_mx_quant_grouped_mat_mul_tiling.cpp (21处)
- [x] mc2/allto_allv_quant_grouped_mat_mul/op_host/op_tiling/arch35/allto_allv_tt_quant_grouped_mat_mul_tiling.cpp (20处)
- [x] mc2/allto_allv_quant_grouped_mat_mul/op_host/allto_allv_quant_grouped_mat_mul_infershape.cpp (0处 - 无需修改，已使用EZ宏)

### 未修改（内部错误，保留 OP_LOGE）
- ACL转换失败 (x1Acl, x2Acl, x1ScaleAcl等)
- Attr/Attrs获取失败
- 函数调用失败 (Failed to get ...)
- HCCL Tiling失败
- Workspace获取失败
- Platform info检索/校验失败
- Context空指针
- ACL/API错误码
- 跨输入一致性校验 (H1, H2, BS等比较)
- 条件性校验 (permuteOut when flag等)
- 复杂shape校验

## 总计
- 修改文件数: 8
- OP_LOGE 替换次数: 121
