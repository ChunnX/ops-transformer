# 错误码参考

## ACL 错误码

| 错误码 | 名称 | 含义 | 常见原因 |
|-------|------|------|---------|
| 0 | ACL_SUCCESS | 成功 | - |
| 107001 | ACL_ERROR_INVALID_PARAM | 参数无效 | deviceId 错误、设备不可用 |
| 107000 | ACL_ERROR_MEMORY_ALLOC_FAILED | 内存分配失败 | 内存不足 |
| 507000 | ACL_ERROR_RT_FAILURE | Runtime 失败 | 内部错误 |

## 算子相关错误码

| 错误码 | 名称 | 含义 | 常见原因 |
|-------|------|------|---------|
| 561103 | OP_ERROR_KERNEL_INIT_FAILED | 算子初始化失败 | SoC 版本不匹配、kernel 未安装 |
| 561108 | OP_ERROR_KERNEL_CONFIG_PARSE_FAILED | kernel config 解析失败 | binary_info_config.json 格式错误 |
| 161002 | OP_ERROR_KERNEL_ADD_FAILED | kernel 添加失败 | kernel binary 缺失 |

## 详细错误分析

### ERROR 561103 - 算子初始化失败

**完整错误信息：**
```
aclnn${算子名}GetWorkspaceSize failed. ERROR: 561103
[ERROR] [KERNEL] Parse dynamic kernel config fail
```

**可能原因：**
1. SoC 版本不匹配（最常见）
2. kernel 未安装或安装路径错误
3. `ASCEND_CUSTOM_OPP_PATH` 未设置

**排查步骤：**
```bash
# 1. 检查 SoC 类型
python3 -c "import torch_npu; print(torch_npu.npu.get_device_name(0))"

# 2. 检查 kernel 安装
ls ${ASCEND_OPP_PATH}/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/

# 3. 检查环境变量
echo $ASCEND_CUSTOM_OPP_PATH
```

---

### ERROR 561108 - kernel config 解析失败

**完整错误信息：**
```
[ERROR] Parse dynamic kernel config fail (errno 561108)
```

**可能原因：**
1. `binary_info_config.json` 文件缺失或损坏
2. kernel 版本与 CANN 版本不兼容

**排查步骤：**
```bash
# 检查 config 文件
ls ${ASCEND_OPP_PATH}/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/${soc}/${算子名}/binary_info_config.json

# 查看文件内容
cat ${ASCEND_OPP_PATH}/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/${soc}/${算子名}/binary_info_config.json
```

---

### ERROR 161002 - kernel 添加失败

**完整错误信息：**
```
[ERROR] ADD_TO_LAUNCHER_LIST_AICORE failed (errno 161002)
```

**可能原因：**
1. kernel binary (*.o) 文件缺失
2. kernel 与当前 SoC 不兼容

**排查步骤：**
```bash
# 检查 kernel binary
ls ${ASCEND_OPP_PATH}/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/${soc}/${算子名}/*.o
```

---

### ERROR 107001 - 设备初始化失败

**完整错误信息：**
```
aclrtSetDevice failed. ERROR: 107001
```

**可能原因：**
1. deviceId 超出范围
2. 设备被其他进程占用
3. 驱动未安装或未启动

**排查步骤：**
```bash
# 检查可用设备
npu-smi info
python3 -c "import torch_npu; print(torch_npu.npu.device_count())"

# 检查设备状态
npu-smi info -t board
```

---

## 启用详细日志

```bash
# 设置日志级别（5 = DEBUG）
export ASCEND_GLOBAL_LOG_LEVEL=5
export ASCEND_SLOG_PRINT_TO_STDOUT=1

# 日志输出位置
export ASCEND_SLOG_PATH=/tmp/ascend_log
```

运行程序后查看日志：
```bash
ls /tmp/ascend_log/
cat /tmp/ascend_log/*.log
```

## 错误码查询方法

```bash
# 使用 ascendcl 错误码查询工具（如果可用）
python3 -c "import acl; acl.get_error_message(561103)"
```