# 常见问题排查指南

## 问题 1: ERROR 561103 - Parse dynamic kernel config fail

### 错误特征

```
[ERROR] Parse dynamic kernel config fail (errno 561108)
[ERROR] ADD_TO_LAUNCHER_LIST_AICORE failed (errno 161002)
```

### 根本原因

**SoC 版本不匹配** - 编译时使用的 `--soc` 参数与实际 NPU 设备类型不一致。

### 解决方案

```bash
# Step 1: 检查实际 SoC 类型
python3 -c "import torch_npu; print(torch_npu.npu.get_device_name(0))"

# 输出对照表（参见 reference/soc_mapping.md）：
# Ascend910_93  → --soc=ascend910_93
# Ascend910_9362 → --soc=ascend910_93  
# Ascend910B1   → --soc=ascend910b

# Step 2: 使用正确的 SoC 版本重新编译
bash build.sh --pkg --soc=<正确的版本> --ops=<算子名> -j8

# Step 3: 重新安装
./build/cann-ops-transformer-custom_linux-aarch64.run --quiet --nox11
```

### 详细排查步骤

1. **启用详细日志**
   ```bash
   export ASCEND_GLOBAL_LOG_LEVEL=5
   export ASCEND_SLOG_PRINT_TO_STDOUT=1
   ```

2. **检查已安装的 kernel**
   ```bash
   ls ${ASCEND_OPP_PATH}/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/
   # 检查是否存在对应 SoC 版本的目录
   ```

3. **确认根因**
   - 如果 kernel 目录下只有 `ascend910b/`，但设备是 `Ascend910_93`
   - 则需要重新编译为 `ascend910_93`

---

## 问题 2: 头文件找不到

### 错误特征

```
fatal error: acl/acl.h: No such file or directory
fatal error: aclnnop/aclnn_${算子名}.h: No such file or directory
```

### 解决方案

检查 include 路径是否完整：

```bash
# ACL 头文件位置
ls ${ASCEND_HOME_PATH}/${ARCH_INFO}-linux/include/acl/acl.h

# 自定义算子头文件位置
ls ${ASCEND_OPP_PATH}/vendors/custom_transformer/op_api/include/aclnnop/aclnn_${算子名}.h

# 编译时必须包含所有路径：
-I ${ASCEND_HOME_PATH}/${ARCH_INFO}-linux/include
-I ${ASCEND_HOME_PATH}/${ARCH_INFO}-linux/include/aclnnop
-I ${ASCEND_OPP_PATH}/vendors/custom_transformer/op_api/include
-I ${ASCEND_OPP_PATH}/vendors/custom_transformer/op_api/include/aclnnop
```

---

## 问题 3: 库文件链接失败

### 错误特征

```
cannot find -lascendcl: No such file or directory
undefined reference to symbol 'aclCreateTensor'
```

### 解决方案

检查库路径和链接顺序：

```bash
# 库文件位置
ls ${ASCEND_HOME_PATH}/${ARCH_INFO}-linux/lib64/libascendcl.so
ls ${ASCEND_OPP_PATH}/vendors/custom_transformer/op_api/lib/libcust_opapi.so

# 必须链接的库（顺序敏感）：
-lcust_opapi -lascendcl -lnnopbase -lc_sec

# 必须设置 rpath：
-Wl,-rpath=${CUST_LIBRARY_PATH} -Wl,-rpath=${EAGER_LIBRARY_PATH}
```

---

## 问题 4: 输出全是 NaN

### 根本原因

**输入数据不合理** - 使用全 1.0 或其他不合理的输入值。

### 解决方案

使用合理的输入数据：

```cpp
// 错误示例：全 1.0 输入
std::vector<uint16_t> queryHostData(querySize, 0x3f80);  // BF16 = 1.0

// 正确示例：随机初始化 + L2 normalize（attention 算子常见）
// 在 Python 侧生成并导出：
torch.rand() → torch.nn.functional.normalize(q, p=2, dim=-1)
```

---

## 问题 5: 设备初始化失败

### 错误特征

```
aclrtSetDevice failed. ERROR: 107001
```

### 解决方案

检查可用设备：

```bash
npu-smi info                    # 检查 NPU 状态
python3 -c "import torch_npu; print(torch_npu.npu.device_count())"  # 检查可用设备数

# 使用正确的 deviceId（通常从 0 开始）
int32_t deviceId = 0;  // 或根据 torch_npu 显示的设备
```

---

## 问题 6: 找不到算子库

### 错误特征

```
cannot find -lcust_opapi
```

### 解决方案

1. **确认算子已安装**
   ```bash
   ls ${ASCEND_OPP_PATH}/vendors/custom_transformer/op_api/lib/libcust_opapi.so
   ```

2. **如果不存在，重新安装算子包**
   ```bash
   ./build/cann-ops-transformer-custom_linux-aarch64.run --quiet --nox11
   ```

---

## 通用排查流程

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: 检查设备 SoC 类型                                   │
│  python3 -c "import torch_npu; print(torch_npu.npu.get_device_name(0))"  │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 2: 检查已安装的 kernel 目录                            │
│  ls ${ASCEND_OPP_PATH}/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/  │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 3: 对比 SoC 类型与 kernel 目录                         │
│  如果不匹配 → 重新编译                                       │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 4: 检查 include 和 library 路径                       │
│  编译时是否包含所有必要路径                                  │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 5: 检查 LD_LIBRARY_PATH                               │
│  运行时是否设置了正确的库路径                                │
└─────────────────────────────────────────────────────────────┘
```