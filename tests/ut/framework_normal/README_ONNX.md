# ONNX 插件单元测试

ONNX 插件 UT 复用 Host UT 的 GTest 入口，随 `OP_HOST_UT` 或 `UT_TEST_ALL`
启用，链接到 `transformer_op_host_ut`。

## 运行

在已配置 CANN 开发环境的 Linux 主机上，从仓库根目录执行：

```bash
bash build.sh --ophost_test --ops=moe_finalize_routing_v2,moe_gating_top_k_softmax
```

仅运行这两个插件的测试（仍会构建所选算子的 Host UT）：

```bash
GTEST_FILTER='OnnxMoeFinalizeRoutingV2PluginTest.*:OnnxMoeGatingTopKSoftmaxPluginTest.*' \
  bash build.sh --ophost_test --ops=moe_finalize_routing_v2,moe_gating_top_k_softmax
```

只编译、不执行时增加 `--noexec`。不传 `--ops` 时沿用仓库默认的 `ALL`。

## 新增插件测试

1. 在算子的 `tests/ut/framework/` 下新增 `test_*_onnx_plugin.cpp`。
2. 在该目录添加以下 `CMakeLists.txt`，并确保上层测试目录会遍历此目录：

   ```cmake
   if(UT_TEST_ALL OR OP_HOST_UT)
       add_modules_ut_sources(UT_NAME ${OP_FRAMEWORK_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})
   endif()
   ```

3. 测试文件直接包含 `../../../framework/<plugin>_onnx_plugin.cpp`，以调用文件内的
   `static` 解析函数。每个插件只在一个测试翻译单元中包含，避免重复注册。
4. 用普通 `ge::Operator` 构造源和目标，通过 `SetAttr` 准备输入，断言解析状态、
   目标属性和失败路径。使用 `TestOp` 类型，避免算子自身默认属性掩盖缺失属性场景。

公共目标在 `cmake/ut.cmake` 定义并按 `ASCEND_OP_NAME` 筛选用例，在
`tests/ut/framework_normal/op_host/CMakeLists.txt` 中通过 `--whole-archive` 链接，
防止未被显式引用的 GTest 注册对象被丢弃。

该测试目标不引入 ONNX protobuf 生成代码或 `onnx_common.h`。
测试调用真实 GE Operator API，需要支持 `ParseParamsByOperatorFn` 的 CANN 开发包。
这些用例验证插件回调的属性解析行为，不替代 ONNX 模型导入的集成测试。
