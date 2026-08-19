## 描述
参考 allto_all_matmul 算子的修改方式，对 quant_matmul_allto_all（matmul_allto_all）的 mxfp8 和 mxfp4 量化方式进行完整双路径适配，引入 apace hcomm 实现路径，与当前主线 pipeline 形成编译期双路径分发。

### 核心改动
1. tiling_key.h：新增 USING_APACE_IMPL 和 ISSMALLK 模板参数，更新 QUANTMODE 枚举值（KC 拆分为 FP8E5M2/FP8E4M3），MX+APACE=1 使用 AIC_ONLY 核类型
2. tiling_data.h：新增 hcommAllToAllMatmulTilingData 结构体（apace hcomm 路径专用 tiling data）
3. mx_quant_matmul_allto_all_arch35.h：新增 IsMxFp4 模板参数、rankForComm_（FP4 自动折半）、ProcessScale() 方法、all2all_out 参数
4. mx_quant_matmul_allto_all_pipeline.h（新建）：MX 量化专用 pipeline，含 scaleTransposeContext 和 ProcessScale()
5. apt.cpp：FP4/FP8 分开 dispatch，USING_APACE_IMPL 编译期分流 apace hcomm 与主线 pipeline
6. tiling_base.h/cpp：新增 usingApaceImpl_ 判断（all2all_out 为空且卡数 <= 8）、DoHcommMxQuantMMTiling/SetHcommTilingInfo/双路径 PostTiling/SetHcclTiling/GetTilingKey

### 双路径分发规则
- apace hcomm 路径：all2all_out 为空且卡数 <= 8，使用 Blaze quant matmul + HCCL 纯 C 通信（AIC_ONLY）
- 主线路径：all2all_out 非空或卡数 > 8，使用通信 + 转置 + 计算 pipeline（MIX_AIC_1_2）

两路径通过 TilingKey 的 USING_APACE_IMPL 模板参数区分，编译期分流。

## 关联的Issue
https://gitcode.com/cann/ops-transformer/issues/4380

## 测试
UT + ST + RDV

## 类型标签
- [x] 新特性
- [x] 重构
