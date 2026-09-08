# RoiExtractor (detection)

FPN RoI Extract（RoIAlignV2 语义）PyPTO-Pro 算子：对每个 RoI 按原图坐标面积开方值
选择 FPN 层级，经 `spatial_scale`（`aligned=true` 时各坐标再减 0.5）换算到该层特征图
坐标系，逐输出像素做对齐式双线性采样并按采样数平均，输出
`y[index[n] or n, c, ph, pw]`。纯 Vector 算子（无 Cube），目标平台 Ascend950PR。

- 语义依据：canndev 内置算子 RoiExtractor（TBE 动态实现），接口/attr 顺序与其 op_proto
  一致；`pooled_height`/`pooled_width` 独立使用（修复 canndev 动态版 `pool_w=pooled_h`
  笔误）。
- 接口形态：canndev `DYNAMIC_INPUT(features)`（TensorList，1~8 层）展开为 8 个 OPTIONAL
  输入 `features_0..features_7`（层数 L 经 TilingData 传递）；`index` 为 OPTIONAL int32
  输出行重映射。
- kernel：`op_kernel/roi_extractor.py`（`@pl.jit` + `datatype={"rois": "input_dtype"}`，
  fp16/fp32 双特化；TilingKey 仅 `HasIndex`）。
- 接入模式参照：`attention/dense_lightning_indexer_softmax_lse_v2/`。

## 目录结构

```
roi_extractor/
├── CMakeLists.txt            # 子目录遍历
├── op_host/
│   ├── CMakeLists.txt        # require_pypto_pro + enable_pypto_kernel + Ascend950PR_9599
│   ├── roi_extractor_def.cpp         # OpDef（8 OPTIONAL features + rois + OPTIONAL index + y）
│   ├── roi_extractor_infershape.cpp  # y = [R, C, pooled_height, pooled_width]
│   ├── roi_extractor_tiling.h/.cpp   # TilingData 填充 + SetTilingKey(HasIndex) + blockDim
│   └── config/ascend950/             # binary.json / simplified_key.ini
├── op_kernel/
│   └── roi_extractor.py      # PyPTO-Pro kernel
├── op_graph/
│   └── roi_extractor_proto.h
├── examples/
│   └── test_aclnn_roi_extractor.cpp
└── README.md
```

## 编译

```bash
bash build.sh --pkg --soc=ascend950 --ops=roi_extractor --vendor_name=custom -j16
```
