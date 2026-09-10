---
iteration_count: 1
iterations:
  - id: 1
    goal: "全功能实现"
    wave1:
      a1_main:
        scope: "kernel（avg/max 双 TilingKey、fp16/fp32 input_dtype、L≤5 槽位、[K,5]/[K,4]+index）+ Host Tiling + OpDef/infershape/proto + aclnn L0 + CMakeLists（require_pypto_pro/enable_pypto_kernel）+ 根 CMakeLists 注册 detection 模块"
      a1_p:
        - task_name: "probe_gather_semantics"
          tiling_key: "TilingKey_0(avg)"
          dtype: "float32"
          memory_strategy: "single_buffer"
      b:
        scope: "UT 用例开发（tiling 分支 + op_api）与 ST golden 对拍用例"
    wave2:
      a2:
        scope: "tiling UT 全覆盖（shape/属性/双 TilingKey/L 边界/K=0）+ op_api UT"
    acceptance:
      ut: "UT 全部通过（tiling UT + op_api UT）"
      st: "L0+L1 golden 对拍通过（层级/aligned/sr/max/非方形/空 RoI/越界/退化）"
---

# roi_extractor 迭代执行计划

## 修订记录

| 版本 | 修订内容 | 修订时间 | 修订人(gitId) |
| --- | --- | --- | --- |
| v1.0 | 基于 spec.yaml 与 DESIGN.md 生成迭代计划 | 2026-09-10 | qq_52056150 |

## 全覆盖目标

| 覆盖维度 | 内容 | 来源 | 说明 |
| --- | --- | --- | --- |
| 全 dtype | fp16/fp32（input_dtype 单二进制覆盖） | spec.yaml dtype_policy | 输出与 feats 同 dtype，fp32 中间量累加 |
| shape/broadcast | L∈[1,5]、K∈[0,65536]、C≤4096、ph/pw∈[1,56]、非方形池化 | spec.yaml shape_constraints | 无广播；feats 折叠维 (L,N,C) 逐层 H/W 递减 |
| boundary/extreme | 空 RoIs、退化 RoI、越界采样、层级 clamp、batch_idx 越界报错、非法属性报错 | spec.yaml boundary_conditions / extreme_inputs | 与 UT/ST 用例一一对应 |
| precision | fp32 rtol=atol=1e-5；fp16 rtol=atol=1e-3 | spec.yaml numerical_tolerance | 对拍 NumPy 双路径 golden + torchvision 逐层参考 |
| 计算分支 | pool_mode avg/max（TilingKey bit0 双模板）、[K,5]/[K,4]+index（colOffset）、aligned 0/1 | DESIGN.md §3.2/§3.5 | UT 覆盖 TilingKey 与 colOffset 全分支 |

## 穿刺结果判定

| 状态 | 判定标准 | 后续处理 |
| --- | --- | --- |
| 成功 | probe_gather_semantics（vf.gather flat 索引 + vf.arange 缩放）编译通过且单点用例数值正确 | 复用主线实现 |
| 部分成功 | gather 语义与假设不符但逐行 gather 备选可行 | 切换备选方案（每通道行独立 1D gather），重跑单点 |
| 失败 | Vector 路径无法表达（极端情况） | 归档问题，回到方案设计修订（§3.1 技术路径重选） |

## 穿刺任务规格

| 项 | 内容 |
| --- | --- |
| 任务名 | probe_gather_semantics |
| TilingKey | TilingKey_0（pool_mode=avg） |
| Dtype | float32（最小验证面；fp16 走同一 input_dtype 二进制） |
| Shape | L=1, N=1, C=64, H=W=16, K=2, pooled 2×2, sample_num=1, aligned=True |
| Memory Strategy | single_buffer（穿刺仅验证语义，不启用双缓冲） |
| 验证点 | vf.gather 在 [Cb, R·Wc] tile 上的 flat 索引取数正确性；vf.arange+muls 生成通道基址向量正确性 |
| 对照 | NumPy golden 单点输出逐元素比对（atol=1e-5） |
