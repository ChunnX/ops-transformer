# RoiExtractor

多层级 RoI 特征提取算子（FPN 层级映射 + RoIAlign 双线性插值池化），兼容 mmdetection
`SingleRoIExtractor` 语义与遗留 GE `RoiExtractor` 属性面。

- **接口**：`aclnnRoiExtractor`（单算子两段式，feats 以 `aclTensorList` 传入，1~5 层）
- **kernel**：PyPTO Pro（`op_kernel/roi_extractor.py`，AICore Vector，950PR/910B 双端）
- **Host**：AscendC registry-invoke（OpDef / infershape / tiling），TilingData 与 kernel
  dataclass 三侧一致，TilingKey bit0 = pool_mode(avg/max)
- **dtype**：fp16 / fp32（输出与 feats 一致，插值中间量 fp32 累加）
- **属性**：`finest_scale`(56) `roi_scale_factor`(0) `spatial_scale`(ListFloat)
  `pooled_height`(7) `pooled_width`(7) `sample_num`(0) `pool_mode`("avg") `aligned`(true)
- **rois**：`[K,5]`=(batch_idx,x1,y1,x2,y2) 或 `[K,4]`+`index`([K] int32)

设计文档见 `docs/`（spec.yaml / DESIGN.md / MDE.md / PLAN.md）。

## 构建

```bash
# Host + Kernel 二进制（Ascend 950PR）
bash build.sh --ophost --opkernel --soc=ascend950 --ops=roi_extractor

# 910B：当前 CANN 9.2.0 的 PyPTO 二进制编译器不支持 ascend910b 目标
# （kernel 代码为纯 vf 路径，待 PyPTO 二进制路径支持后启用 def.cpp 的 AddConfig("ascend910b") 即可）

# UT（tiling + op_api）
bash build.sh --ophost_test --ops=roi_extractor
```

## 数值语义（与 torchvision/mmcv 对齐）

- 层级映射：`lvl = clamp(floor(log2(sqrt(area)/finest_scale + 1e-6)), 0, L-1)`
  （kernel 以等价比较式实现：`area >= (finest_scale * 2^j * roi_scale_factor)^2`）
- 采样点完全越界（`y<-1 ∨ y>H ∨ x<-1 ∨ x>W`）贡献 0（avg 计入分母）
- 边界钳制 `y'=max(y,0)`、`y1=min(y0+1,H-1)`；退化 RoI 按 `max(w,1)/max(h,1)`
- 精度：fp32 rtol=atol=1e-5；fp16 rtol=atol=1e-3（对拍 fp64 golden）
