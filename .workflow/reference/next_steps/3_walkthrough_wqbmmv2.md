# 任务 3：走读 ops-nn wqbmmv2 ND 反量化实现

## 背景

GMM A16W4 pergroup ND 的向量反量化（VF）函数需要参考 ops-nn 仓库中 `weight_quant_batch_matmul_v2` 的 ND int4 反量化实现。核心函数为 `AntiquantComputeW4NKPerGroup` 和 `AntiquantComputeKNPerGroup`。需要走读这些函数的实现，了解如何迁移到 GMM 的 `basic_block_vf_nd.h` 中。

## 任务详情

### 目标

产出走读文档，覆盖 wqbmmv2 中 weight ND int4 pergroup 反量化的完整实现，明确迁移到 GMM 的适配方案。

### 走读范围

| 文件 | 走读重点 |
|------|----------|
| `/home/sunduiyang/Code/ops-nn/matmul/weight_quant_batch_matmul_v2/op_kernel/arch35/weight_quant_batch_matmul_v2_reg_base_common.h` | **核心文件**：`AntiquantComputeW4NKPerGroup` 和 `AntiquantComputeKNPerGroup` 函数实现 |
| 同目录下 `weight_quant_batch_matmul_v2_vf.h` | VF 函数底层实现（MicroAPI 调用） |
| 同目录下 `n_first/basic_block_vf_nz.h` | NZ 格式的 pergroup 实现（作为对比参照） |

### 走读要点

#### 1. AntiquantComputeW4NKPerGroup
- 函数签名和模板参数
- NK 排布下 weight int4 的 unpack 方式
- pergroup scale/offset 的加载方式（shape 为 `(k/gs, n)` 的 ND 排布）
- 反量化计算公式的 VF 实现：`(weight + offset) * scale`
- UB buffer 的使用方式
- 循环结构（bubNLen、bubKLen 的遍历方式）

#### 2. AntiquantComputeKNPerGroup
- KN 排布与 NK 排布的差异
- weight int4 在 KN 排布下的 unpack 方式
- pergroup scale/offset 在 KN 排布下的加载方式
- 与 NK 版本的计算差异

#### 3. 迁移适配分析
- wqbmmv2 与 GMM 的代码架构差异（命名空间、模板参数、buffer 管理方式）
- 哪些代码可以直接复用
- 哪些代码需要适配（buffer 分配方式、地址计算方式、模板参数映射）
- GMM 的 `basic_block_vf_nd.h` 中已有的 ND 函数与新函数的接口对齐方式

### 产出

走读文档输出到 `.workflow/raw/docs/` 目录下，建议文件名：
- `wqbmmv2_nd_antiquant_walkthrough.md`

### 完成标准

- [ ] 完整记录 `AntiquantComputeW4NKPerGroup` 的实现逻辑
- [ ] 完整记录 `AntiquantComputeKNPerGroup` 的实现逻辑
- [ ] 明确 NK 和 KN 两种排布的差异
- [ ] 明确迁移到 GMM 的适配方案（可复用部分 + 需适配部分）
- [ ] 明确与 GMM `basic_block_vf_nd.h` 已有函数的接口对齐方式

## 任务计划

1. 走读 `weight_quant_batch_matmul_v2_reg_base_common.h` 中 `AntiquantComputeW4NKPerGroup` 实现
2. 走读 `AntiquantComputeKNPerGroup` 实现
3. 对比 NK 和 KN 两种排布的差异
4. 分析迁移适配方案
5. 产出走读文档
