# MegaMoe UT 测试框架总结

## ✅ 已修正的错误

### 2026-05-06 修正记录

#### 1. test_mega_moe_infershape.cpp
- ✅ 修正命名空间：`Ops::NN::AnyValue` → `Ops::Transformer::AnyValue`
- ✅ 删除错误的 `IrInstanceNum()` 参数
- ✅ 使用正确的框架：`mc2_infer_shape_case_executor.h` + `Mc2ExecuteTestCase`
- ✅ 明确列出所有10个输入和2个输出

#### 2. test_mega_moe_tiling.cpp (arch35)
- ✅ 修正命名空间：`Ops::NN::AnyValue` → `Ops::Transformer::AnyValue`
- ✅ 删除错误的 `IrInstanceNum()` 参数和 `TilingContextFaker`
- ✅ 使用正确的框架：`mc2_tiling_case_executor.h` + `TilingContextPara`
- ✅ 删除 CSV 文件，改用硬编码测试用例数组 `g_testCases[]`
- ✅ 使用参数化测试：`TEST_P` + `INSTANTIATE_TEST_CASE_P(testing::ValuesIn(g_testCases))`

#### 3. test_mega_moe.cpp (op_kernel)
- ✅ 删除错误的框架（`.IrInstanceNum()`、`InitMegaMoeTiling()`）
- ✅ 使用 CPU 仿真框架：`__CCE_KT_TEST__` + `ICPU_RUN_KF`
- ✅ 添加 placeholder 测试（硬件环境测试）
- ✅ 删除错误的 `mega_moe.cpp` 辅助文件
- ✅ 同步修正 CMakeLists.txt：
  - 修正 SocVersion：`"Ascend950"` → `"ascend950"`（小写）
  - 修正参数格式：添加 tiling 文件路径变量，参数4传递 tiling 文件列表而非测试文件

#### 4. test_aclnn_mega_moe.cpp (op_api)
- ✅ 修正参数类型：`weight1/2/weightScales1/2` 从 `aclTensor` → `vector<TensorDesc>`（TensorList）
- ✅ 使用正确的参数顺序和类型

#### 5. mega_moe_tiling_def.h
- ✅ 创建正确的 tiling 数据结构定义

#### 6. CI 配置文件同步
- ✅ 在 `tests/test_config.yaml` 中添加 mega_moe 配置（按字母顺序插入）
- ✅ 在 `tests/test_soc_config.yaml` 中添加 mega_moe ascend950 配置

---

## 📋 当前测试文件结构

```
E:/opencode/yangzeheng/ops-transformer/mc2/mega_moe/tests/ut/
├── op_api/
│   ├── CMakeLists.txt
│   └── test_aclnn_mega_moe.cpp          ✅ 已修正
├── op_host/
│   ├── CMakeLists.txt
│   ├── test_mega_moe_infershape.cpp     ✅ 已修正
│   └── arch35/
│       ├── CMakeLists.txt
│       └── test_mega_moe_tiling.cpp     ✅ 已修正（硬编码数组）
└── op_kernel/
    ├── CMakeLists.txt
    ├── mega_moe_tiling_def.h            ✅ 已修正
    └── test_mega_moe.cpp                ✅ 已修正
```

---

## ⚠️ 重要说明

### 框架使用规范

| 模块 | 正确框架 | 关键要点 |
|------|---------|---------|
| op_host infershape | `mc2_infer_shape_case_executor.h` | 不用 IrInstanceNum，用 Ops::Transformer |
| op_host tiling (arch35) | `mc2_tiling_case_executor.h` | 使用 CSV + TEST_P，添加 Mc2Hcom::MockValues |
| op_kernel | `__CCE_KT_TEST__` + CPU仿真 | 需要 Python生成数据 + bin文件 |
| op_api | `OP_API_UT` | 注意 TensorList 类型 |

### 常见错误

1. ❌ **命名空间错误**：mc2 算子必须用 `Ops::Transformer::AnyValue`
2. ❌ **IrInstanceNum**：新框架不需要此参数
3. ❌ **TensorList 类型**：op_api 中权重参数是 TensorList，不是单个 Tensor
4. ❌ **Kernel 测试框架**：不能用 `KernelRunContextFaker`，需要 CPU仿真

---

## 🔧 后续工作

### 编译验证
```bash
# 需要配置编译环境
bash build.sh --opapi_test --ophost_test --opkernel_test --ops=mega_moe --soc=ascend950
```

### 数据生成（op_kernel）
需要创建 Python 脚本生成测试数据：
- `gen_data.py` - 生成输入数据 bin 文件
- `gen_tiling.py` - 生成 tiling 数据 bin 文件

### 真实测试
op_kernel 测试需要在 ascend950 硬件环境下运行，CPU 仿真模式可能无法完整验证通信算子。

---

## 📊 测试用例统计

| 模块 | 用例数 | 状态 |
|------|-------|------|
| op_api | 4 | ✅ 已修正 |
| op_host infershape | 10 | ✅ 已修正 |
| op_host tiling | 8 (硬编码) | ✅ 已修正 |
| op_kernel | 1 (placeholder) | ⚠️ 需硬件环境 |

总计：23 个测试点