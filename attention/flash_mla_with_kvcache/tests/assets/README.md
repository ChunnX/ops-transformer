# FlashMlaWithKvcache BAND 与 sinks 回归

## BAND 窗口回归

安装同版本 MLA 主算子及 metadata custom 包、更新 torch 扩展后运行：

```bash
OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10 MAX_JOBS=10 \
python attention/flash_mla_with_kvcache/tests/assets/test_band.py
```

覆盖右对齐闭区间、FP16/BF16、分页边界、单点/无限窗口、空 batch、
整行屏蔽、FlashDecode、mode 0/3 等价性和非法参数。CPU golden 同时检查 full/tiled 两条路径。

`test_band.py` 同时覆盖 FP32 per-head sinks：mode 0/3/4、空 sinks、极大 logit、
full/tiled golden、长 KV FlashDecode，以及 FP16/BF16。CPU 验证可运行
`python test_band.py BandGoldenTest`；完整 NPU 回归需重建并加载匹配的算子包和 Torch 扩展。

## Host UT（无需 NPU）

Checker 回归已迁移到仓库正式 GTest UT，包含主算子 tiling、shape/dtype 推导及 metadata checker。
目录结构、CSV 用例和 context faker 与 `flash_attn/tests/ut` 保持一致。
构建、运行和覆盖范围见 [Host UT 说明](../ut/README.md)。
