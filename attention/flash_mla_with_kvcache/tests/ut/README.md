# FlashMlaWithKvcache Host UT

参照 `attention/flash_attn/tests/ut`，使用仓库的 GTest、CSV loader、tiling/infershape context faker，
运行真实 Host 实现，不依赖 NPU。metadata checker 在自身算子的 `tests/ut/op_host` 下维护。

## 用例组织

| 文件 | 验证内容 |
| --- | --- |
| `op_host/arch35/test_flash_mla_checkers_tiling.cpp` | 完整合法输入 fixture；缺失描述符；布局组合；Q 头数；block size；Q/K 所有零轴；mask、sinks、LSE 边界 |
| `op_host/arch35/test_flash_mla_with_kvcache_tiling.cpp/.csv` | 真实注册 tiling 入口，覆盖 parser → checker → tiling；FP16/BF16、12 种合法布局组合、mode 0/3/4、sinks、窗口和非法输入 |
| `op_host/test_flash_mla_with_kvcache_shape_infershape.cpp/.csv` | 输出布局、512 维输出、PA_NZ 逻辑宽度、LSE 开关、动态维度和错误 shape |
| `op_host/test_flash_mla_with_kvcache_dtype_infershape.cpp/.csv` | 输出继承 Q 的 FP16/BF16 类型；LSE 为 FP32 |
| `attention/flash_mla_with_kvcache_metadata/tests/ut/op_host/test_flash_mla_metadata_check_tiling.cpp` | metadata 的 Q 头数、布局、mask/window、固定 MLA 维度、必传输入、seqlens 类型/长度、容量边界 |

Tiling 成功用例同时检查序列化结果中的 mask mode、左右窗口、mask shape、LSE 开关和 MLA 维度。
其中 `-1` 窗口应转换成 `INT32_MAX`，mode 3 的右窗口为 0。
Sinks 由输入指针供内核读取，Host UT 检查其存在性、类型和头数，不把它当作 tiling 中的开关字段。

Q 头数遍历 `-1…129`；block size 遍历 `0…1040`，覆盖 64 个合法值及中间所有不对齐值。
循环检查使用 `SCOPED_TRACE` 标明失败参数；参数化测试由 GTest 分别报告结果。

## 编译和运行

先加载当前 CANN SDK 环境，确保 `ASCEND_HOME_PATH` 指向该 SDK，并准备仓库的 third_party 依赖。
在仓库根目录使用标准入口：

```bash
bash build.sh --ophost_test --ops=flash_mla_with_kvcache,flash_mla_with_kvcache_metadata \
  --soc=ascend950 -j15
```

如果需要保留已有 `build/` 产物，可使用独立构建目录。以下命令与本次验证方式相同：

```bash
export MLA_UT_BUILD=/tmp/flash-mla-host-ut
cmake -S . -B "$MLA_UT_BUILD" \
  -DCUSTOM_ASCEND_CANN_PACKAGE_PATH="$ASCEND_HOME_PATH" \
  -DCANN_3RD_LIB_PATH="$PWD/third_party" \
  -DASCEND_COMPUTE_UNIT=ascend950 \
  '-DASCEND_OP_NAME=flash_mla_with_kvcache;flash_mla_with_kvcache_metadata' \
  '-DTESTS_UT_OPS_TEST=flash_mla_with_kvcache;flash_mla_with_kvcache_metadata' \
  -DBUILD_OPEN_PROJECT=ON -DBUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG=ON \
  -DENABLE_BUILT_IN=OFF -DENABLE_TEST=ON -DOP_HOST_UT=ON \
  -DUT_INFERSHAPE_FLAG=ON -DENABLE_UT_EXEC=OFF -DENABLE_ASAN=OFF \
  -DENABLE_AICPU=ON -DENABLE_TILING_SINK=OFF -DCMAKE_BUILD_TYPE=Debug
cmake --build "$MLA_UT_BUILD" --target transformer_op_host_ut -j15
BUILD_PATH="$MLA_UT_BUILD" \
LD_LIBRARY_PATH="$ASCEND_HOME_PATH/lib64:${LD_LIBRARY_PATH:-}" \
  "$MLA_UT_BUILD/tests/ut/framework_normal/op_host/transformer_op_host_ut" \
  --gtest_filter='*FlashMla*' --gtest_output="xml:$MLA_UT_BUILD/flash_mla_ut.xml"
```

`BUILD_PATH` 用于加载 UT Host 库和注册算子，手动运行时必须设置。
`ENABLE_AICPU=ON` 用于让构建系统包含 metadata 算子；上述构建目标只编译 Host UT。
编译并行度不超过 15。可以进一步降低，例如 `-j8`。

## 扩展方式

- 新的 tiling、shape、dtype 组合：在对应 CSV 中增加一行，使用唯一的 `case_name`。
- CSV 类型使用 loader 定义的 `FLOAT16`、`BF16`、`FLOAT`、`INT32`、`INT8`；`FLOAT` 表示 FP32。
- 缺失输入：该输入的 shape/dtype/format 三列留空；一维空 Tensor 的 shape 填 `0`。
- checker 的新边界规则：基于合法 fixture 只改变一条约束，避免因别的错误而误通过负例。
- 主算子和 metadata 的源码变更通过 `tests/test_config.yaml` 同时选择两者的 UT。

仅两个白盒 checker 测试源文件使用 `-fno-access-control`；生产构建参数不变。
Host UT 不执行内核数值计算；BAND/sinks 的 CPU golden 和 NPU 精度回归仍在 `tests/assets/test_band.py`。
