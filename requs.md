# 需求

## 需求一

核心目标，增补算子attention/kv_quant_sparse_flash_attention和attention/quant_lightning_indexer的tests/ut （算子简称QSFA/QLI）

ut是lcov覆盖代码的的框架，现在我已经从sparse_flash_attention和lightning_indexer下面拷贝ut框架过来了，但代码我还没改，请帮我修改attention/kv_quant_sparse_flash_attention/tests/ut和attention/quant_lightning_indexer/tests/ut

以op代表算子，特此说明op/tests/ut下面的cpp文件（test_op_tiling.cpp、test_op_infershape.cpp）用例参数是严格参照op/op_host/op_def.cpp来的，包括输入tensor、输出tensor、属性。（test_op_tiling.cpp、test_op_infershape.cpp）里的参数顺序也必须与def保持一致，举例子：

```
# test_sparse_flash_attention_infershape.cpp的参数和sparse_flash_attention_def.cpp里面参数一致
        {
            {{{4, 1, 128, 512}, {4, 1, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},         // query            input0
            {{{4, 65536, 1, 512}, {4, 65536, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // key              input1
            {{{4, 65536, 1, 512}, {4, 65536, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // value            input2
            {{{4, 1, 1, 16}, {4, 1, 1, 16}}, ge::DT_INT32, ge::FORMAT_ND},              // sparse_indices   input3
            {{{4, 1024}, {4, 1024}}, ge::DT_INT32, ge::FORMAT_ND},                      // block_table      input4
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_qlist},          // actual_seq_lengths_query
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_kvlist},         // actual_seq_lengths_kv 
            {{{4096, 64, 1, 512}, {4096, 64, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // query_rope       input5
            {{{4096, 64, 1, 512}, {4096, 64, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}      // key_rope         input6
        },
        {// 输出Tensor
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},     // attention_out
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},    // softmax_max
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}     // softmax_sum
        },
        {// 属性
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0416666666666667)},
            {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        });
# test_sparse_flash_attention_tiling.cpp的参数和sparse_flash_attention_def.cpp里面参数一致
        {
            {{{4, 1, 128, 512}, {4, 1, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},         // query            input0
            {{{4, 65536, 1, 512}, {4, 65536, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // key              input1
            {{{4, 65536, 1, 512}, {4, 65536, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // value            input2
            {{{4, 1, 1, 16}, {4, 1, 1, 16}}, ge::DT_INT32, ge::FORMAT_ND},              // sparse_indices   input3
            {{{4, 1024}, {4, 1024}}, ge::DT_INT32, ge::FORMAT_ND},                      // block_table      input4
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_qlist},          // actual_seq_lengths_query
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_kvlist},         // actual_seq_lengths_kv 
            {{{4096, 64, 1, 512}, {4096, 64, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // query_rope       input5
            {{{4096, 64, 1, 512}, {4096, 64, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}      // key_rope         input6
        },
        {
            {{{4, 1, 128, 512}, {4, 1, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // attention_out
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                // softmax_max
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                                 // softmax_sum
        },
        {
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0416666666666667)},
            {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        },
```

覆盖目标是算子目录下面的op_host/op_tiling.cpp，行覆盖率要达到80%以上

验证测试的命令：

```
# 先source包的环境变量
source /home/j60100428/Ascend_0408/cann/set_env.sh
# 开启打屏error日志（设置了才会具体显示报错信息）
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=1
# 跑lcov ut覆盖
mkdir -p ./log
bash build.sh -u --ophost --ops=kv_quant_sparse_flash_attention --soc="ascend910b,ascend950" --cov 2>&1 | tee ./log/qli_ophost_ut.log
或者
bash build.sh -u --ophost --ops=quant_lightning_indexer --soc="ascend910b,ascend950" --cov 2>&1 | tee ./log/qli_ophost_ut.log
```

开启打屏error日志后你可以看到当前用例是被哪些拦截判断拦截了，通过此既构造返回GRAPH_SUCCESS的用例又构造返回GRAPH_FAILED的用例，禁止你动算子的功能代码哈，只能在attention/quant_lightning_indexer/tests/ut、attention/kv_quant_sparse_flash_attention/tests/ut和tests/test_config.yaml修改

"/home/j60100428/ops-transformer_0408/requs.md"和"/home/j60100428/ops-transformer_0408/lcov.md"是其他算子实现ut 覆盖的一些推理过程，你可以借鉴一下

把记录记到本本md文档
## 2026-04-14 实现记录

### 本轮修改目标

- 将 `attention/quant_lightning_indexer/tests/ut`、`attention/kv_quant_sparse_flash_attention/tests/ut` 中从 `lightning_indexer` / `sparse_flash_attention` 直接拷贝过来的占位用例，替换为与当前量化算子定义一致的 op_host UT。
- 严格按各自 `op_host/*_def.cpp` 的输入、输出、属性顺序重写 `infershape` 与 `tiling` 用例参数。
- 保持不修改功能代码，只落在需求允许的测试目录内。

### 已修改文件

- `attention/quant_lightning_indexer/tests/ut/op_host/test_lightning_indexer_infershape.cpp`
- `attention/quant_lightning_indexer/tests/ut/op_host/arch32/test_lightning_indexer_tiling.cpp`
- `attention/kv_quant_sparse_flash_attention/tests/ut/op_host/test_sparse_flash_attention_infershape.cpp`
- `attention/kv_quant_sparse_flash_attention/tests/ut/op_host/arch32/test_sparse_flash_attention_tiling.cpp`

### QuantLightningIndexer 处理说明

#### infershape 重写

- 原文件还是 `LightningIndexer` 的拷贝版本，存在以下问题：
  - `opName` 错误，仍写成 `LightningIndexer`。
  - 输入数量错误，缺少 `query_dequant_scale`、`key_dequant_scale`。
  - 输出数量错误，`QuantLightningIndexer` 只有一个输出 `sparse_indices`，原文件仍保留了 `sparse_values`。
  - 属性错误，缺少 `query_quant_mode`、`key_quant_mode`，并保留了旧算子的 `return_values`。
  - dtype 错误，原用例还是 BF16/FLOAT 路径，没有切到量化算子的 `INT8 + FLOAT16 scale` 组合。
- 现已改成：
  - 2 个 infershape 成功用例：`BSND/BSND`、`TND/PA_BSND`。
  - 1 个 inferdtype 用例：校验输出固定为 `DT_INT32`。

#### tiling 重写

- 原文件同样还是旧 `LightningIndexer` 的覆盖框架，参数顺序和定义文件不匹配。
- 已按 `quant_lightning_indexer_def.cpp` 的顺序改成 8 个输入：
  - `query`
  - `key`
  - `weights`
  - `query_dequant_scale`
  - `key_dequant_scale`
  - `actual_seq_lengths_query`
  - `actual_seq_lengths_key`
  - `block_table`
- 新增场景：
  - 3 个成功场景：`BSND/BSND`、`TND/TND`、`BSND/PA_BSND`。
  - 3 个失败场景：`query_quant_mode` 非法、`layout_query/layout_key` 不匹配、`sparse_count=0`。
- 成功场景没有硬编码 `tilingKey`，而是使用 `ExecuteTiling` 断言成功并检查关键结果（`tilingKey > 0`、workspace 非空），避免当前阶段因为首次补测而被未知 tiling key 阻塞。

### KvQuantSparseFlashAttention 处理说明

#### infershape 重写

- 原文件还是 `SparseFlashAttention` 的拷贝版本，存在以下问题：
  - `opName` 错误，仍写成 `SparseFlashAttention`。
  - dtype 错误，`key/value` 还是 BF16，而量化算子要求 910B 上走 `INT8`。
  - 属性缺失，缺少 `key_quant_mode`、`value_quant_mode`、`quant_scale_repo_mode`、`tile_size`、`rope_head_dim`。
  - 输出数量错误，原文件保留了 `softmax_max` / `softmax_sum`，但 `KvQuantSparseFlashAttention` 只有 `attention_out` 一个输出。
  - 输入顺序错误，原文件没有按新算子的 `key_dequant_scale` / `value_dequant_scale` / `block_table` / `actual_seq_lengths_*` 顺序组织。
- 现已改成：
  - 2 个 infershape 成功用例：`BSND/PA_BSND`、`TND/TND`。
  - 1 个 inferdtype 用例：校验输出 dtype 与 `query` 保持一致。
- shape 设计遵循 README 中的 MLA 约束：
  - `query` 末轴 `576 = 512 + 64`
  - `key/value` 末轴 `656 = 512 + 64 * 2 + 4 * 4`
  - `attention_out` 末轴 `512`

#### tiling 重写

- 已按 `kv_quant_sparse_flash_attention_def.cpp` 的输入顺序重写为 9 个输入：
  - `query`
  - `key`
  - `value`
  - `sparse_indices`
  - `key_dequant_scale`
  - `value_dequant_scale`
  - `block_table`
  - `actual_seq_lengths_query`
  - `actual_seq_lengths_kv`
- 新增场景：
  - 2 个成功场景：`BSND/PA_BSND`、`TND/TND`。
  - 3 个失败场景：`attention_mode` 非法、`sparse_block_size` 非法、输出 shape 与 MLA 输出不一致。
- 成功场景同样使用 `ExecuteTiling` 直接验证成功返回，避免初次补测阶段先被精确 `tilingKey` 回填卡住。

### 验证记录

#### 1. 构建命令

已执行：

```bash
source /home/j60100428/Ascend_0408/cann/set_env.sh
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=1
bash build.sh -u --ophost --ops=quant_lightning_indexer --soc=ascend910b --cov > log/qli_ophost_ut.log 2>&1
cmake --build build --target transformer_op_host_ut -j4 > log/qli_ophost_build.log 2>&1
```

#### 2. 构建结果

- `build.sh -u --ophost --ops=quant_lightning_indexer --soc=ascend910b --cov` 已跑通到 CMake 配置完成。
- `cmake --build build --target transformer_op_host_ut -j4` 已成功完成，说明当前仓的 op_host UT 目标整体可重新生成。
- 过程中网络依赖下载曾被沙箱拦截，放开后完成构建。

#### 3. 运行态验证现状

额外尝试执行：

```bash
build/tests/ut/framework_normal/op_host/transformer_op_host_ut --gtest_filter='*QuantLightningIndexer*'
```

当前现象：

- 二进制返回 `Running 0 tests from 0 test suites`。
- 这说明在当前仓的这次构建产物里，`transformer_op_host_ut` 没有实际注册任何 gtest 用例；也就是说，目前还存在一层仓内 UT 聚合/注册问题，导致新增 testcase 尚未在运行态被枚举出来。
- 这个现象不是我这次改动特有的问题，因为即使直接 `--gtest_list_tests`，当前二进制也没有列出任何测试集。

### 当前结论

- 已完成你要求的核心测试代码重写：QLI / QSFA 的 `infershape` 和 `tiling` 参数现在都已对齐各自 `op_def.cpp`。
- 成功/失败分支用例都已经补上，不再是原始拷贝态。
- 当前剩余阻塞不是测试文件本身，而是仓内 `transformer_op_host_ut` 目标的 testcase 聚合/注册链路没有把 gtest 用例枚举出来，导致还不能继续做你要的逐条日志打点和 `tilingKey` 回填。
- 下一步如果继续推进，建议优先排查：
  - `tests/ut/framework_normal/op_host` 的 UT 聚合逻辑是否真的把这两个算子的 tests 源文件编入目标。
  - `add_modules_ut_sources` 对这两个算子目录是否生效。
  - 生成出来的 `transformer_op_host_ut` 是否链接了对应 testcase object。

### 待继续补充

- 若后续确认 testcase 已被真正编进二进制，需要继续：
  - 跑 `QuantLightningIndexer` / `KvQuantSparseFlashAttention` 的单测过滤命令。
  - 根据打屏日志逐个把成功用例的 `tilingKey` 和必要 workspace 回填。
  - 再结合 lcov 报告针对低覆盖函数继续补失败/成功分支。



## 需求二

关于` `transformer_op_host_ut` 目标的 testcase 聚合/注册链路没有把 gtest 用例枚举出来`的问题我已修正，现在运行命令可以把 gtest 用例枚举出来了，并且我将ut的文件名更正了

现在要求(以op代指算子)：

1、根据打屏日志逐个把成功用例的 `tilingKey` 和必要 workspace 回填，保留TilingData空字符串的变量。

2、用例命令用数字标号，场景说明用注释就行

3、QSFA算字的tiling覆盖还不达标，根据验证测试命令再去逐步增补用例，QLI的覆盖率如果可以继续优化的话也可以继续写

```
# 先source包的环境变量
source /home/j60100428/Ascend_0408/cann/set_env.sh
# 开启打屏error日志（设置了才会具体显示报错信息）
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=1
# 跑lcov ut覆盖
mkdir -p ./log
bash build.sh -u --ophost --ops=kv_quant_sparse_flash_attention --soc=ascend910b --cov 2>&1 | tee ./log/qsfa_ophost_ut.log
或者
bash build.sh -u --ophost --ops=quant_lightning_indexer --soc=ascend910b --cov 2>&1 | tee ./log/qli_ophost_ut.log
```

4、你查阅QSFA和QLI的op_def.cpp文件会发现当芯片版本为ascend950时会与其他芯片版本的参数配置不一样，参考attention/flash_attention_score/tests/ut/op_host下面的配置，这里我说明一下op/tests/ut/op_host下面只针对op/op_host目录下面tiling.cpp文件和infershape.cpp，其中tiling文件的场景分两种，arch35目录对应ascend950或者tiling_v2.cpp，arch32则对应其他芯片版本，QSFA和QLI的tiling文件不同芯片都是用的同一个，但def文件中却配置了不同芯片版本的参数，可能在tiling文件中也存在只有ascend950才能走到的分支逻辑

你帮我判断一下是否有需要去像attention/flash_attention_score/tests/ut/op_host下面一样去搞arch35的UT（比如tiling有些分支限制了只有ascend950时才能跑进去），如果有需要请帮我实现吧

如果有实现，arch35命令需要换下：

```
bash build.sh -u --ophost --ops=kv_quant_sparse_flash_attention --soc=ascend950 --cov 2>&1 | tee ./log/qsfa_ophost_ut.log
或者
bash build.sh -u --ophost --ops=quant_lightning_indexer --soc=ascend950 --cov 2>&1 | tee ./log/qli_ophost_ut.log
```




### 需求二实施记录（2026-04-14）

#### 1) 是否需要新增 arch35 UT 的判断

结论：**需要**。

原因（已结合 `op_def.cpp` 和 `tiling.cpp` 实际分支验证）：

- QLI 在 Ascend950（A5）存在仅 A5 才能命中的参数/类型分支：
  - `DT_FLOAT8_E4M3FN`、`DT_HIFLOAT8` 相关约束
  - hifloat8 下 `sparse_count == 2048` 约束
- QSFA 在 Ascend950（A5）存在仅 A5 才能命中的分支：
  - key/value 仅支持 `DT_FLOAT8_E4M3FN` / `DT_HIFLOAT8`
  - `sparse_block_size == 1` 约束
  - hifloat8 下 `sparseBlockCount == 2048` 约束

因此按 `attention/flash_attention_score/tests/ut/op_host` 的模式补了 `arch35` UT。

#### 2) 本次代码改动（仅在允许目录内）

新增：

- `attention/quant_lightning_indexer/tests/ut/op_host/arch35/CMakeLists.txt`
- `attention/quant_lightning_indexer/tests/ut/op_host/arch35/test_quant_lightning_indexer_tiling.cpp`
- `attention/kv_quant_sparse_flash_attention/tests/ut/op_host/arch35/CMakeLists.txt`
- `attention/kv_quant_sparse_flash_attention/tests/ut/op_host/arch35/test_kv_quant_sparse_flash_attention_tiling.cpp`

修改：

- `attention/quant_lightning_indexer/tests/ut/op_host/arch32/test_quant_lightning_indexer_tiling.cpp`
- `attention/quant_lightning_indexer/tests/ut/op_host/arch35/test_quant_lightning_indexer_tiling.cpp`
- `attention/kv_quant_sparse_flash_attention/tests/ut/op_host/arch32/test_kv_quant_sparse_flash_attention_tiling.cpp`
- `attention/kv_quant_sparse_flash_attention/tests/ut/op_host/arch35/test_kv_quant_sparse_flash_attention_tiling.cpp`

未修改：

- `tests/test_config.yaml`
- 任意 op 功能实现代码（仅改 UT）

#### 3) 成功用例回填（tilingKey / workspace / 空 TilingData）

已把成功场景从 `ExecuteTiling(...)` 统一改为：

- `int64_t expectTilingKey = ...;`
- `std::string expectTilingData = "";`
- `std::vector<size_t> expectWorkspaces = {...};`
- `ExecuteTestCase(...)`

回填值如下：

- QLI `910b`
  - `QuantLightningIndexer_910b_tiling_0`: `tilingKey=197122`, `workspace={167903232}`
  - `QuantLightningIndexer_910b_tiling_1`: `tilingKey=570622466`, `workspace={167903232}`
  - `QuantLightningIndexer_910b_tiling_2`: `tilingKey=1090716162`, `workspace={167903232}`
- QLI `950`（arch35）
  - `QuantLightningIndexer_950_tiling_0`: `tilingKey=205860`, `workspace={167903232}`
  - `QuantLightningIndexer_950_tiling_1`: `tilingKey=1090724386`, `workspace={167903232}`
- QSFA `910b`
  - `KvQuantSparseFlashAttention_910b_0`: `tilingKey=1154`, `workspace={293928960}`
  - `KvQuantSparseFlashAttention_910b_1`: `tilingKey=1092`, `workspace={268763136}`
- QSFA `950`（arch35）
  - `KvQuantSparseFlashAttention_950_tiling_0`: `tilingKey=1154`, `workspace={293928960}`
  - `KvQuantSparseFlashAttention_950_tiling_1`: `tilingKey=1092`, `workspace={268763136}`

同时已移除临时 `DumpTilingInfo` 打印函数。

#### 4) 用例命名/注释规范落实

- 用例名全部采用数字后缀（如 `_0/_1/...`）。
- 场景说明采用注释，不在用例名中写文本描述。

#### 5) QSFA 覆盖率补齐动作

为把 `ascend910b` 的 `kv_quant_sparse_flash_attention_tiling.cpp` 拉到 `>= 80%`，新增了失败场景：

- 属性非法分支：`key_quant_mode` / `quant_scale_repo_mode` / `rope_head_dim`
- 参数存在性分支：
  - `layout_kv=TND` 缺 `actualSeqLengthsKv`
  - `layout_kv=PA_BSND` 缺 `actualSeqLengthsKv`
  - `layout_kv=PA_BSND` 缺 `blockTable`
- 布局解析分支：
  - `layout_kv` 不支持
  - `layout_query` 与非 PA `layout_kv` 不一致
  - `layout_query` 不支持

#### 6) 串行执行命令与结果（已避开并行构建）

说明：为避免并行构建互相污染（之前出现过日志错乱/无效结果），本次全部 **串行** 执行。

执行命令：

```bash
source /home/j60100428/Ascend_0408/cann/set_env.sh
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=1

bash build.sh -u --ophost --ops=quant_lightning_indexer --soc=ascend910b --cov 2>&1 | tee log/qli_910b_final.log
bash build.sh -u --ophost --ops=quant_lightning_indexer --soc=ascend950 --cov 2>&1 | tee log/qli_950_final.log
bash build.sh -u --ophost --ops=kv_quant_sparse_flash_attention --soc=ascend950 --cov 2>&1 | tee log/qsfa_950_final.log
bash build.sh -u --ophost --ops=kv_quant_sparse_flash_attention --soc=ascend910b --cov 2>&1 | tee log/qsfa_910b_final3.log
```

结果摘要：

- QLI 910b：`9/9 PASSED`
- QLI 950：`6/6 PASSED`
- QSFA 950：`6/6 PASSED`
- QSFA 910b（最终）：`17/17 PASSED`

#### 7) 覆盖率结果

- QLI 910b：
  - `quant_lightning_indexer_tiling.cpp |93.0% (384)`
- QLI 950（arch35）：
  - `quant_lightning_indexer_tiling.cpp |90.1% (384)`
- QSFA 950（arch35）：
  - `kv_quant_sparse_flash_attention_tiling.cpp |77.4% (898)`
- QSFA 910b（最终，达标）：
  - `kv_quant_sparse_flash_attention_tiling.cpp |80.2% (898)`

## 需求三
四处需要修改检查的地方(op_api不用管)：
attention\lightning_indexer\tests\ut
attention\sparse_flash_attention\tests\ut
attention\quant_lightning_indexer\tests\ut
attention\kv_quant_sparse_flash_attention\tests\ut

在这四处ut中存在一些问题：
1、int64_t actual_seq_qlist[] = {} 是零长度数组初始化，在 C++ 标准中这属于未定义行为（ISO C++ 不允许零长度数组）。虽然 GCC/Clang 作为扩展支持，但不同编译器行为可能不一致。建议改为 int64_t* actual_seq_qlist = nullptr，和 tiling_1 中的写法保持一致，也更明确地表达"无数据"的语义。
2、如果验证过程发现有些用例不能通过，暂时使用 DISABLED_ 前缀（GTest 的标准机制）来跳过而不是注释掉整个用例，这样既能保留代码又能在测试报告中清楚看到有多少用例被临时禁用。
3、所有返回GRAPH_SUCCESS的成功测试用例的 expectTilingData 都是空字符串，意味着只验证了 tiling 函数的返回状态和 tilingKey，但完全没有验证实际生成的 tiling 数据内容。这让测试的有效性大打折扣，即使 tilingKey 正确，tiling 数据中的具体参数（比如分块大小、循环次数等）出错也无法发现。建议至少对几个关键场景填充预期的 tiling 数据进行端到端验证。
4、检查这四个的ut，所有返回GRAPH_FAILED的用例清理掉诸如expectTilingKey和expectTilingData等的校验，并在用例前面TEST_F(KvQuantSparseFlashAttentionTiling, KvQuantSparseFlashAttention_910b_1)上加上拦截日志，可以借鉴attention\kv_quant_sparse_flash_attention\tests\ut\op_host\arch32\test_kv_quant_sparse_flash_attention_tiling.cpp的KvQuantSparseFlashAttention_910b_17用例
5、代码将要上库，再次检查四个ut的Copyright是否合理（不只是cpp文件，cmake文件也要检查）

修改后记得验证
```
# 先source包的环境变量
source /home/j60100428/Ascend_0408/cann/set_env.sh
# 开启打屏error日志（设置了才会具体显示报错信息）
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=1

mkdir -p ./log

# 跑lcov ut覆盖
bash build.sh -u --ophost --ops=lightning_indexer --soc="ascend910b" --cov 2>&1 | tee ./log/li_ophost_ut.log
bash build.sh -u --ophost --ops=sparse_flash_attention --soc="ascend910b" --cov 2>&1 | tee ./log/sfa_ophost_ut.log
bash build.sh -u --ophost --ops=quant_lightning_indexer --soc="ascend910b,ascend950" --cov 2>&1 | tee ./log/qli_ophost_ut.log
bash build.sh -u --ophost --ops=kv_quant_sparse_flash_attention --soc="ascend910b,ascend950" --cov 2>&1 | tee ./log/qsfa_ophost_ut.log

```
