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

## Host checker 回归（无需 NPU）

先准备已配置 MLA 的 CMake 构建目录，其中应有 `compile_commands.json` 和 `CMakeCache.txt`，
且 CANN SDK、opbase 依赖可用。从仓库根目录执行：

```bash
python3 attention/flash_mla_with_kvcache/tests/assets/run_checker_tests.py --build-dir build
```

脚本复用构建目录中的 C++ 编译参数与 CANN 路径，在临时目录编译真实 checker 和 parser，
执行后自动清理测试二进制，不修改已有构建产物。为覆盖独立检查方法，测试编译使用
`-fno-access-control`；生产编译不受影响，并强制启用测试断言。

覆盖布局组合、1–128头、分页块范围/对齐、Q/K零轴、mask存在性、sink类型/头数、
metadata空layout、LSE类型/shape和组合checker调用顺序。它不代替完整算子编译或NPU精度测试。
