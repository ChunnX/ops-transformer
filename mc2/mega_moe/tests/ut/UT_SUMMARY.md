# mega_moe 算子 UT 测试总结报告

## 算子基本信息

| 项目 | 内容 |
|------|------|
| **算子名称** | MegaMoe |
| **算子路径** | mc2/mega_moe |
| **平台支持** | ascend950 (arch35) |
| **算子分类** | MC2 分布式 MoE 算子 |

## 测试文件清单（参考 matmul_all_reduce 结构）

```
mc2/mega_moe/tests/
├── CMakeLists.txt                          # 主测试目录入口
├── ut/
│   ├── CMakeLists.txt                      # UT 子目录入口
│   ├── op_api/
│   │   ├── CMakeLists.txt                  # op_api 构建配置
│   │   └── test_aclnn_mega_moe.cpp         # op_api 测试 (2 用例)
│   ├── op_host/
│   │   ├── CMakeLists.txt                  # op_host 入口 + infershape
│   │   └── arch35/
│   │       ├── CMakeLists.txt              # arch35 tiling 构建
│   │       └── test_mega_moe_tiling.cpp    # op_tiling 测试 (1 用例)
│   └── op_kernel/
│       ├── CMakeLists.txt                  # op_kernel AddOpTestCase
│       └── test_mega_moe.cpp               # op_kernel 测试 (1 用例)
```

## 编译与运行

```bash
# 配置 CMake
cmake -B build \
    -DENABLE_TEST=ON \
    -DUT_TEST_ALL=ON \
    -DASCEND_CANN_PACKAGE_PATH=/path/to/cann \
    -DASCEND_COMPUTE_UNIT=ascend950

# 编译
cmake --build build

# 运行测试
./build/tests/ut/framework_normal/transformer_op_api_ut
./build/tests/ut/framework_normal/transformer_op_host_ut
./build/tests/ut/framework_normal/transformer_op_kernel_ut_ascend950
```

## 测试用例统计

| 测试类型 | 文件名 | 测试数量 |
|----------|--------|----------|
| op_api | test_aclnn_mega_moe.cpp | 2 |
| op_tiling | test_mega_moe_tiling.cpp | 1 |
| op_kernel | test_mega_moe.cpp | 1 |

## 注意事项

1. **LSP错误**: 显示的错误因缺少CANN头文件，编译时自动解决
2. **旧目录清理**: 需手动删除 `tests/utest/` 和 `tests/ut/op_host/op_api/` 目录