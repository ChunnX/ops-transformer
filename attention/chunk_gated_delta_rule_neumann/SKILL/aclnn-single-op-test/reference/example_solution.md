# chunk_gated_delta_rule 实例

## 问题背景

尝试调用 `chunk_gated_delta_rule` 算子时遇到错误：

```
aclnnChunkGatedDeltaRuleGetWorkspaceSize failed. ERROR: 561103
[ERROR] Parse dynamic kernel config fail (errno 561108)
[ERROR] ADD_TO_LAUNCHER_LIST_AICORE failed (errno 161002)
```

## 根因定位过程

### Step 1: 启用详细日志

```bash
export ASCEND_GLOBAL_LOG_LEVEL=5
export ASCEND_SLOG_PRINT_TO_STDOUT=1
```

发现关键错误信息：
```
[ERROR] Parse dynamic kernel config fail
[ERROR] Please check if the kernel package is installed.
```

### Step 2: 检查设备 SoC 类型

```bash
python3 -c "import torch_npu; print(torch_npu.npu.get_device_name(0))"
# 输出: Ascend910_93
```

### Step 3: 检查已安装的 kernel

```bash
ls ${ASCEND_OPP_PATH}/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/
# 发现只有 ascend910b/ 目录，没有 ascend910_93/
```

### Step 4: 根因确认

**编译时使用 `--soc=ascend910b`，但设备是 `Ascend910_93`**

kernel 安装目录：
```
kernel/
└── ascend910b/
    └── chunk_gated_delta_rule/
        └── *.o
```

但设备需要：
```
kernel/
└── ascend910_93/
    └── chunk_gated_delta_rule/
        └── *.o
```

## 解决步骤

### 1. 使用正确的 SoC 版本重新编译

```bash
cd /mnt/workspace/gitCode/cann/ops-transformer

bash build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule -j8

# 编译成功标志：
# Built target chunk_gated_delta_rule_kernel
# Built target chunk_gated_delta_rule_oplib
```

### 2. 重新安装算子包

```bash
./build/cann-ops-transformer-custom_linux-aarch64.run --quiet --nox11

# 安装后检查：
ls ${ASCEND_OPP_PATH}/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/ascend910_93/chunk_gated_delta_rule/
# 应看到：*.o 文件和 binary_info_config.json
```

### 3. 重新编译测试程序

```bash
cd /mnt/workspace/gitCode/cann/ops-transformer/attention/chunk_gated_delta_rule

source ${ASCEND_HOME_PATH}/bin/setenv.bash

ARCH_INFO=$(uname -m)
INCLUDE_PATH="${ASCEND_HOME_PATH}/include"
EAGER_INCLUDE_OPP_ACLNNOP_PATH="${ASCEND_HOME_PATH}/${ARCH_INFO}-linux/include/aclnnop"
EAGER_LIBRARY_PATH="${ASCEND_HOME_PATH}/${ARCH_INFO}-linux/lib64"
CUST_INCLUDE_PATH="${ASCEND_OPP_PATH}/vendors/custom_transformer/op_api/include"
CUST_LIBRARY_PATH="${ASCEND_OPP_PATH}/vendors/custom_transformer/op_api/lib"

g++ examples/test_aclnn_chunk_gated_delta_rule.cpp \
    -I ${CUST_INCLUDE_PATH} -I ${CUST_INCLUDE_PATH}/aclnnop \
    -I ${INCLUDE_PATH} -I ${EAGER_INCLUDE_OPP_ACLNNOP_PATH} \
    -L ${CUST_LIBRARY_PATH} -L ${EAGER_LIBRARY_PATH} \
    -lcust_opapi -lascendcl -lnnopbase -lc_sec \
    -Wl,-rpath=${CUST_LIBRARY_PATH} -Wl,-rpath=${EAGER_LIBRARY_PATH} \
    -o test_aclnn_chunk_gated_delta_rule \
    -std=c++17 -O2
```

### 4. 运行测试

```bash
TILING_LIBRARY_PATH="${ASCEND_OPP_PATH}/vendors/custom_transformer/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64"
export LD_LIBRARY_PATH=${TILING_LIBRARY_PATH}:${CUST_LIBRARY_PATH}:${EAGER_LIBRARY_PATH}:${LD_LIBRARY_PATH}

./test_aclnn_chunk_gated_delta_rule
```

## 成功标志

```
Workspace size: 37921280 bytes     ← GetWorkspaceSize 成功
Calling aclnnChunkGatedDeltaRule...
Synchronizing stream...
Test completed successfully!       ← 算子执行成功
```

## 测试代码位置

```
/mnt/workspace/gitCode/cann/ops-transformer/attention/chunk_gated_delta_rule/examples/test_aclnn_chunk_gated_delta_rule.cpp
```

## 关键参数

参考 `tests/ut/op_kernel/test_chunk_gated_delta_rule.cpp` 默认参数：

```cpp
int64_t B = 2;      // batch size
int64_t T = 128;    // total tokens (B * seqlen)
int64_t nk = 4;     // key heads
int64_t nv = 4;     // value heads  
int64_t dk = 64;    // key dimension
int64_t dv = 64;    // value dimension
int64_t hasGamma = 1;  // 有可选 gamma 参数
std::vector<int32_t> seqlens = {64, 64};  // 每个序列长度
float scale = 1.0f;  // attention scale
```

## 算子 API 签名

```cpp
// 获取 workspace 大小
aclnnStatus aclnnChunkGatedDeltaRuleGetWorkspaceSize(
    const aclTensor* query,
    const aclTensor* key,
    const aclTensor* value,
    const aclTensor* beta,
    const aclTensor* initial_state,
    const aclIntArray* actual_seq_lengths,
    const aclTensor* g,          // 可选，nullptr 表示无
    double scale_value,
    const aclTensor* out,
    const aclTensor* final_state,
    uint64_t* workspaceSize,
    aclOpExecutor** executor
);

// 执行算子
aclnnStatus aclnnChunkGatedDeltaRule(
    void* workspaceAddr,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream
);
```

## 输入 Tensor Shape

| Tensor | Shape | Dtype |
|--------|-------|-------|
| query | (T, nk, dk) | BF16 |
| key | (T, nk, dk) | BF16 |
| value | (T, nv, dv) | BF16 |
| beta | (T,) | BF16 |
| initial_state | (B, nv, dv, dk) | BF16 |
| actual_seq_lengths | (B,) | INT32 |
| g (可选) | (T,) | FLOAT |
| out | (T, nv, dv) | BF16 |
| final_state | (B, nv, dv, dk) | BF16 |

## 总结

**最关键的一步：确认 SoC 类型并使用正确的 `--soc` 参数编译**

| 错误现象 | 根因 | 解决方案 |
|---------|------|---------|
| ERROR 561103 | SoC 版本不匹配 | `--soc=ascend910_93` 重新编译 |
| Parse kernel config fail | kernel 目录缺失 | 重新安装 `.run` 包 |
| 设备显示 Ascend910_93 | 需要 ascend910_93 kernel | 不要用 ascend910b |

这个实例展示了从错误现象到根因定位再到解决的完整流程，可作为其他算子调试的参考模板。