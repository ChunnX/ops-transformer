# CANN算子错误日志收集与分析标准化流程

## 一、概述

本文档总结了CANN算子错误日志收集与分析的标准化流程，适用于所有CANN算子的错误日志提取与分析工作。

**关键要点：**
- 主机层（op_host/op_api）和Kernel层（op_kernel）使用不同的错误宏
- Kernel层错误极少但至关重要（通常仅占0.8%-1%）
- 必须搜索所有ERROR级别的错误宏，避免遗漏
- 数据完整性验证至关重要（需考虑共存情况）
- 必须基于文件路径+行号进行去重处理

## 二、错误宏分类与识别

### 2.1 主机层错误宏（op_host/op_api）

主机层主要使用以下错误宏：

| 错误宏 | 类型 | 说明 | 示例 |
|--------|------|------|------|
| **OP_LOGE** | ERROR日志 | 主机层错误日志输出 | `OP_LOGE(context->GetNodeName(), "GetXandWeightDtype failed.")` |
| **ACLNN_ERR_PARAM_NULLPTR** | ERROR码 | API层参数空指针错误 | `CHECK_COND(x != nullptr, ACLNN_ERR_PARAM_NULLPTR, "X must not be nullptr.")` |
| **ACLNN_ERR_PARAM_INVALID** | ERROR码 | API层参数无效错误 | `CHECK_COND(shape != nullptr, ACLNN_ERR_PARAM_INVALID, "shape is null.")` |
| **ACLNN_ERR_INNER** | ERROR码 | API层内部错误 | `CHECK_RET(func() == ACLNN_SUCCESS, ACLNN_ERR_INNER)` |
| **OPS_REPORT** | ERROR报告 | 主机层错误报告 | `OPS_REPORT(context->GetNodeName(), "compileInfoPtr is nullptr.")` |
| **OPS_REPORT_CUBE_INNER_ERR** | ERROR报告 | Cube层错误报告 | `OPS_REPORT_CUBE_INNER_ERR(op_name, "X dim is not 2.")` |

**注意：**
- `OP_LOGE(ACLNN_ERR_PARAM_INVALID, ...)` 是OP_LOGE调用，应标记为OP_LOGE类型（不是ACLNN_ERR）
- `OPS_REPORT_CUBE_INNER_ERR` 属于OPS_REPORT家族，应标记为OPS_REPORT类型
- `OP_TILING_CHECK`、`OPS_ERR_IF`、`OP_CHECK_IF` 等是上述错误宏的包装宏，实际调用的是OP_LOGE或OPS_REPORT

**重要：共存情况处理**

在实际代码中，OP_LOGE 和 ACLNN_ERR 可能出现在同一行，例如：

```cpp
// 格式1：OP_LOGE 使用 ACLNN_ERR 作为第一个参数
OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Invalid parameter value %d", value);

// 格式2：CHECK_COND 后紧跟 OP_LOGE
CHECK_COND(x != nullptr, ACLNN_ERR_PARAM_NULLPTR, "X is null");
if (!x) { OP_LOGE(context->GetNodeName(), "Check failed"); }
```

**处理原则：**
- 格式1：归类为 **OP_LOGE** 类型（优先级更高）
- 格式2：两个独立的错误，分别归类（但实际很少见）

**统计影响：**
- 共存情况会导致 ACLNN_ERR 的 grep 数量大于 CSV 数量
- 验证时需要检查共存数量，并从实际数量中减去共存数量进行比较

### 2.2 Kernel层错误宏（op_kernel）

Kernel层使用Ascend C专用错误宏：

| 错误宏 | 类型 | 说明 | 示例 |
|--------|------|------|------|
| **ASCENDC_ASSERT** | 断言检查 | Kernel层编译时/运行时断言 | `ASCENDC_ASSERT(b != 0, { KERNEL_LOG(KERNEL_ERROR, "Division by zero error!"); });` |
| **KERNEL_LOG(KERNEL_ERROR)** | ERROR日志 | Kernel层错误日志（通常嵌套在ASCENDC_ASSERT中） | 见上例 |

**注意：**
- Kernel层错误极少（通常仅占1%左右），但都是关键安全检查
- Kernel层主要用于数学安全检查（除零、溢出、对齐等）
- 搜索Kernel层时只搜索 `ASCENDC_ASSERT`，不要重复搜索 `KERNEL_LOG.*ERROR`（避免重复计数）

### 2.3 不需要包含的错误宏

以下宏不应包含在ERROR日志统计中：

| 错误宏 | 类型 | 说明 | 原因 |
|--------|------|------|------|
| **OP_LOGD** | DEBUG日志 | 调试日志 | 非ERROR级别 |
| **OP_LOGI** | INFO日志 | 信息日志 | 非ERROR级别 |
| **OP_LOGW** | WARNING日志 | 警告日志 | 非ERROR级别（但可能需要单独分析） |
| **throw** | 异常抛出 | C++异常 | 通常未使用 |

## 三、数据收集标准流程

### 3.1 环境准备

```bash
# 1. 进入算子目录
cd /path/to/cann/ops-transformer/<operator_name>

# 2. 确认目录结构
ls -la
# 应包含：op_host、op_api、op_kernel等目录

# 3. 创建输出目录
mkdir -p error_analysis_output
cd error_analysis_output
```

### 3.2 主机层错误搜索（Step 1）

```bash
# 搜索所有主机层错误宏（op_host + op_api）
# 注意：使用 | 分隔多个搜索模式，需要转义

# 方法1：分别搜索每种错误宏
grep -rn 'OP_LOGE' --include='*.cpp' --include='*.h' ../op_host ../op_api > op_loge.txt
grep -rn 'ACLNN_ERR' --include='*.cpp' --include='*.h' ../op_api > aclnn_err.txt
grep -rn 'OPS_REPORT' --include='*.cpp' --include='*.h' ../op_host ../op_api > ops_report.txt

# 方法2：一次性搜索所有主机层错误宏
grep -rn 'OP_LOGE\|ACLNN_ERR\|OPS_REPORT' --include='*.cpp' --include='*.h' ../op_host ../op_api > host_errors.txt
```

### 3.3 Kernel层错误搜索（Step 2）

```bash
# 搜索Kernel层错误宏（只搜索ASCENDC_ASSERT，避免重复）
grep -rn 'ASCENDC_ASSERT' --include='*.cpp' --include='*.h' ../op_kernel > kernel_errors.txt

# 注意：不要使用以下命令（会导致重复计数）
# grep -rn 'ASCENDC_ASSERT\|KERNEL_LOG.*ERROR' --include='*.cpp' --include='*.h' ../op_kernel
# 原因：每个ASCENDC_ASSERT包含KERNEL_LOG，会匹配两次
```

### 3.4 数据整合与处理（Step 3）

使用Python脚本处理grep结果并生成CSV：

```python
import pandas as pd
import re
import subprocess

def collect_errors(operator_name, base_path):
    """收集所有错误日志"""
    
    all_errors = []
    
    # 1. 搜索主机层错误
    cmd_host = f"grep -rn 'OP_LOGE\\|ACLNN_ERR\\|OPS_REPORT' --include='*.cpp' --include='*.h' {base_path}/op_host {base_path}/op_api 2>/dev/null"
    result_host = subprocess.run(cmd_host, shell=True, capture_output=True, text=True)
    
    if result_host.stdout:
        for line in result_host.stdout.strip().split('\n'):
            if line:
                parts = line.split(':', 2)
                if len(parts) >= 3:
                    filepath = parts[0]
                    filename = filepath.split('/')[-1]
                    linenum = parts[1]
                    content = parts[2]
                    
                    # 提取错误类型（优先级很重要！）
                    # 注意：OP_LOGE 可能包含 ACLNN_ERR 作为第一个参数，所以 OP_LOGE 优先级最高
                    # OPS_REPORT_CUBE_INNER_ERR 是 OPS_REPORT 家族，需要优先判断
                    error_type = 'Unknown'
                    if 'OP_LOGE' in content:
                        error_type = 'OP_LOGE'
                    elif 'OPS_REPORT_CUBE_INNER_ERR' in content:
                        error_type = 'OPS_REPORT'
                    elif 'OPS_REPORT' in content:
                        error_type = 'OPS_REPORT'
                    elif 'ACLNN_ERR_PARAM_NULLPTR' in content:
                        error_type = 'ACLNN_ERR_PARAM_NULLPTR'
                    elif 'ACLNN_ERR_PARAM_INVALID' in content:
                        error_type = 'ACLNN_ERR_PARAM_INVALID'
                    elif 'ACLNN_ERR_INNER' in content:
                        error_type = 'ACLNN_ERR_INNER'
                    elif 'ACLNN_ERR' in content:
                        error_type = 'ACLNN_ERR'
                    
                    # 提取错误消息
                    error_msg = ''
                    # OP_LOGE
                    match = re.search(r'OP_LOGE\([^,]+,\s*"([^"]+)"', content)
                    if match:
                        error_msg = match.group(1)
                    
                    # OPS_REPORT_CUBE_INNER_ERR
                    if not match and 'OPS_REPORT_CUBE_INNER_ERR' in content:
                        match = re.search(r'OPS_REPORT_CUBE_INNER_ERR\([^,]+,\s*"([^"]+)"', content)
                        if match:
                            error_msg = match.group(1)
                    
                    # CHECK_COND等
                    if not match:
                        match = re.search(r'"([^"]+)"', content)
                        if match:
                            error_msg = match.group(1)
                    
                    all_errors.append({
                        '算子': operator_name,
                        '文件路径': filename,
                        '完整路径': filepath,
                        '行号': linenum,
                        '错误类型': error_type,
                        '错误内容': error_msg if error_msg else content[:100],
                        '完整内容': content[:200]
                    })
    
    # 2. 搜索Kernel层错误
    cmd_kernel = f"grep -rn 'ASCENDC_ASSERT' --include='*.cpp' --include='*.h' {base_path}/op_kernel 2>/dev/null"
    result_kernel = subprocess.run(cmd_kernel, shell=True, capture_output=True, text=True)
    
    if result_kernel.stdout:
        for line in result_kernel.stdout.strip().split('\n'):
            if line:
                parts = line.split(':', 2)
                if len(parts) >= 3:
                    filepath = parts[0]
                    filename = filepath.split('/')[-1]
                    linenum = parts[1]
                    content = parts[2]
                    
                    error_type = 'ASCENDC_ASSERT'
                    
                    # 提取KERNEL_LOG中的错误消息
                    error_msg = ''
                    match = re.search(r'KERNEL_LOG\(KERNEL_ERROR,\s*"([^"]+)"', content)
                    if match:
                        error_msg = match.group(1)
                    
                    all_errors.append({
                        '算子': operator_name,
                        '文件路径': filename,
                        '完整路径': filepath,
                        '行号': linenum,
                        '错误类型': error_type,
                        '错误内容': error_msg if error_msg else content[:100],
                        '完整内容': content[:200]
                    })
    
    return all_errors

# 使用示例
operator_name = 'grouped_matmul'
base_path = '/path/to/operator'

errors = collect_errors(operator_name, base_path)
df = pd.DataFrame(errors)
df.to_csv('error_logs_complete.csv', index=False)

print(f'总错误数: {len(df)}')
print(f'按错误类型:')
print(df['错误类型'].value_counts())
```

### 3.5 完整性验证（Step 4）

**重要：验证需要考虑共存情况！**

由于 `OP_LOGE(ACLNN_ERR_XXX, ...)` 格式的错误存在，同一行可能同时包含 OP_LOGE 和 ACLNN_ERR。这类错误在提取时被归类为 OP_LOGE，因此在验证 ACLNN_ERR 数量时，需要减去与 OP_LOGE 共存的行数。

```python
import pandas as pd
import subprocess

df = pd.read_csv('error_logs_complete.csv')

print('=== 完整性验证 ===')
print('\n【主机层验证】')

# 1. 验证 OP_LOGE（无需考虑共存）
csv_count = len(df[df['错误类型'] == 'OP_LOGE'])
cmd = f"grep -r 'OP_LOGE' --include='*.cpp' --include='*.h' ../op_host ../op_api 2>/dev/null | wc -l"
result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
actual_count = int(result.stdout.strip()) if result.stdout.strip() else 0
match = '✓' if csv_count == actual_count else '✗'
print(f'{match} OP_LOGE: CSV={csv_count}, 实际={actual_count}')

# 2. 验证 ACLNN_ERR_PARAM_NULLPTR（需要考虑共存）
csv_count = len(df[df['错误类型'] == 'ACLNN_ERR_PARAM_NULLPTR'])
cmd = f"grep -r 'ACLNN_ERR_PARAM_NULLPTR' --include='*.cpp' --include='*.h' ../op_host ../op_api 2>/dev/null | wc -l"
result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
actual_count = int(result.stdout.strip()) if result.stdout.strip() else 0

# 检查与 OP_LOGE 的共存数量
cmd = f"grep -r 'ACLNN_ERR_PARAM_NULLPTR.*OP_LOGE\\|OP_LOGE.*ACLNN_ERR_PARAM_NULLPTR' --include='*.cpp' --include='*.h' ../op_host ../op_api 2>/dev/null | wc -l"
result2 = subprocess.run(cmd, shell=True, capture_output=True, text=True)
coexist_count = int(result2.stdout.strip()) if result2.stdout.strip() else 0
expected_count = actual_count - coexist_count
match = '✓' if csv_count == expected_count else '✗'
print(f'{match} ACLNN_ERR_PARAM_NULLPTR: CSV={csv_count}, 实际={actual_count} (其中 {coexist_count} 条与OP_LOGE共存)')

# 3. 验证 ACLNN_ERR_PARAM_INVALID（需要考虑共存）
csv_count = len(df[df['错误类型'] == 'ACLNN_ERR_PARAM_INVALID'])
cmd = f"grep -r 'ACLNN_ERR_PARAM_INVALID' --include='*.cpp' --include='*.h' ../op_host ../op_api 2>/dev/null | wc -l"
result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
actual_count = int(result.stdout.strip()) if result.stdout.strip() else 0

# 检查与 OP_LOGE 的共存数量
cmd = f"grep -r 'ACLNN_ERR_PARAM_INVALID.*OP_LOGE\\|OP_LOGE.*ACLNN_ERR_PARAM_INVALID' --include='*.cpp' --include='*.h' ../op_host ../op_api 2>/dev/null | wc -l"
result2 = subprocess.run(cmd, shell=True, capture_output=True, text=True)
coexist_count = int(result2.stdout.strip()) if result2.stdout.strip() else 0
expected_count = actual_count - coexist_count
match = '✓' if csv_count == expected_count else '✗'
print(f'{match} ACLNN_ERR_PARAM_INVALID: CSV={csv_count}, 实际={actual_count} (其中 {coexist_count} 条与OP_LOGE共存)')

# 4. 验证 ACLNN_ERR_INNER
csv_count = len(df[df['错误类型'] == 'ACLNN_ERR_INNER'])
cmd = f"grep -r 'ACLNN_ERR_INNER' --include='*.cpp' --include='*.h' ../op_host ../op_api 2>/dev/null | wc -l"
result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
actual_count = int(result.stdout.strip()) if result.stdout.strip() else 0
match = '✓' if csv_count == actual_count else '✗'
print(f'{match} ACLNN_ERR_INNER: CSV={csv_count}, 实际={actual_count}')

# 5. 验证 OPS_REPORT
csv_count = len(df[df['错误类型'] == 'OPS_REPORT'])
cmd = f"grep -r 'OPS_REPORT' --include='*.cpp' --include='*.h' ../op_host ../op_api 2>/dev/null | wc -l"
result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
actual_count = int(result.stdout.strip()) if result.stdout.strip() else 0
match = '✓' if csv_count == actual_count else '✗'
print(f'{match} OPS_REPORT: CSV={csv_count}, 实际={actual_count}')

print('\n【Kernel层验证】')
kernel_count = len(df[df['错误类型'] == 'ASCENDC_ASSERT'])
cmd = f"grep -r 'ASCENDC_ASSERT' --include='*.cpp' --include='*.h' ../op_kernel 2>/dev/null | wc -l"
result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
actual_kernel_count = int(result.stdout.strip()) if result.stdout.strip() else 0
match = '✓' if kernel_count == actual_kernel_count else '✗'
print(f'{match} ASCENDC_ASSERT: CSV={kernel_count}, 实际={actual_kernel_count}')

print(f'\nCSV总数: {len(df)}')
```

### 3.6 去重处理（Step 5）

**重要：必须对 grep 结果进行去重！**

由于使用 `grep -rn 'OP_LOGE\|ACLNN_ERR\|OPS_REPORT'` 搜索多个模式，同一行如果包含多个错误宏（例如 `OP_LOGE(ACLNN_ERR_XXX, ...)`），会被匹配多次。因此必须基于「文件路径+行号」进行去重。

```python
seen_lines = set()  # 用于去重

for line in result.stdout.strip().split('\n'):
    if line:
        parts = line.split(':', 2)
        if len(parts) >= 3:
            filepath = parts[0]
            linenum = parts[1]
            content = parts[2]
            
            # 去重：同一行只处理一次
            unique_key = f"{filepath}:{linenum}"
            if unique_key not in seen_lines:
                seen_lines.add(unique_key)
                # 提取错误信息...
```

## 四、数据统计与分析建议

### 4.1 数据统计维度

完成错误日志提取后，建议从以下维度进行统计和分析：

**按错误类型统计：**

| 错误类型 | 说明 | 典型占比 |
|---------|------|---------|
| OP_LOGE | 主机层错误日志 | 通常占40-60% |
| ACLNN_ERR_PARAM_INVALID | 参数无效错误 | 通常占30-40% |
| OPS_REPORT | 错误报告 | 通常占5-10% |
| ACLNN_ERR_PARAM_NULLPTR | 参数空指针错误 | 通常占2-5% |
| ACLNN_ERR_INNER | 内部错误 | 通常占1-3% |
| ASCENDC_ASSERT | Kernel层断言 | 通常占0.8-1% |

**按错误来源统计：**

| 错误来源 | 说明 | 典型占比 |
|---------|------|---------|
| host (主机层) | op_host + op_api | 通常占99%以上 |
| kernel (Kernel层) | op_kernel | 通常占0.8-1% |

**按文件统计：**
- 错误数最多的文件（Top 10）
- 错误密度最高的文件（错误数/文件行数）
- 不同模块的错误分布（tiling、infershape、checker等）

### 4.2 验证通过标准

完整性验证应满足以下标准：

**主机层验证：**
- ✓ OP_LOGE: CSV数量应等于实际grep数量
- ✓ ACLNN_ERR_PARAM_NULLPTR: CSV数量 = 实际数量 - 共存数量
- ✓ ACLNN_ERR_PARAM_INVALID: CSV数量 = 实际数量 - 共存数量
- ✓ ACLNN_ERR_INNER: CSV数量应等于实际grep数量
- ✓ OPS_REPORT: CSV数量应等于实际grep数量

**Kernel层验证：**
- ✓ ASCENDC_ASSERT: CSV数量应等于实际grep数量

**验证公式：**
```
CSV数量 = grep实际数量 - 共存数量（仅适用于ACLNN_ERR）
共存数量 = grep "ACLNN_ERR.*OP_LOGE|OP_LOGE.*ACLNN_ERR" 的数量
```

### 4.3 数据质量判断标准

**高质量数据特征：**
1. 所有验证项全部通过（✓）
2. 去重处理正确执行（无重复行）
3. 共存情况正确处理（CSV数量合理）
4. 错误类型判断优先级正确

**典型数据分布：**
1. 主机层错误占比应在99%左右
2. Kernel层错误占比应在0.8-1%左右
3. OP_LOGE 和 ACLNN_ERR_PARAM_INVALID 是主要错误类型
4. 不同算子的共存数量差异较大（取决于代码风格）

### 4.4 关键发现模板

分析完成后，建议总结以下关键发现：

**错误分布特征：**
- 主机层 vs Kernel层占比
- 主要错误类型及其占比
- 共存情况的数量和比例

**文件分布特征：**
- 错误密集的文件及其错误数量
- 不同模块的错误分布特点
- 是否存在错误过多或过少的异常文件

**Kernel层特征：**
- Kernel层错误的具体类型（除零、对齐、溢出等）
- Kernel层错误的分布位置（工具函数、核心计算等）
- Kernel层错误的严重程度评估

## 五、常见问题与解决方案

### 5.1 验证失败怎么办？

如果验证失败，检查以下几点：

1. **是否进行了去重处理？**
   - 必须基于文件路径+行号去重，避免同一行被重复统计

2. **是否考虑了共存情况？**
   - OP_LOGE 和 ACLNN_ERR 可能出现在同一行
   - 验证 ACLNN_ERR 数量时需要减去共存行数

3. **错误类型判断顺序是否正确？**
   - OP_LOGE 优先级最高
   - OPS_REPORT_CUBE_INNER_ERR 需优先于普通 OPS_REPORT 判断

### 5.2 如何处理特殊情况？

**情况1：OP_LOGE 包含多个参数**

```cpp
OP_LOGE(context->GetNodeName(), "Error message with %s and %d", name, value)
```

解决方案：使用正则表达式提取第一个字符串参数
```python
match = re.search(r'OP_LOGE\([^,]*,\s*"([^"]+)"', content)
```

**情况2：ASCENDC_ASSERT 没有 KERNEL_LOG**

```cpp
ASCENDC_ASSERT(condition, { /* no KERNEL_LOG */ });
```

解决方案：即使没有 KERNEL_LOG，也应统计该 ASCENDC_ASSERT

**情况3：一行包含多个错误宏**

```cpp
CHECK_COND(x != nullptr, ACLNN_ERR_PARAM_NULLPTR, "X is null");
OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Error");
```

解决方案：根据优先级判断，归类为 OP_LOGE 类型

## 六、最佳实践建议

### 6.1 数据收集最佳实践

1. **使用统一的脚本模板**
   - 基于本指南提供的 Python 脚本模板
   - 确保错误类型判断优先级正确

2. **必须进行完整性验证**
   - 验证通过才能确认数据准确
   - 记录验证失败的详细信息

3. **保留原始 grep 结果**
   - 便于后续问题排查
   - 可以重新处理数据

### 6.2 数据分析最佳实践

1. **按文件统计错误分布**
   - 识别错误密集的文件
   - 重点关注错误最多的文件

2. **按错误类型分类**
   - 不同类型错误有不同的处理优先级
   - OP_LOGE 通常需要立即修复
   - ACLNN_ERR_PARAM_INVALID 可能需要参数校验优化

3. **区分主机层和Kernel层**
   - Kernel层错误虽然少，但都是关键安全检查
   - 要特别关注 Kernel层错误的处理逻辑

### 6.3 文档输出建议

生成的 CSV 文件应包含以下列：

| 列名 | 说明 |
|------|------|
| 算子 | 算子名称 |
| 错误来源 | host/kernel |
| 文件路径 | 完整文件路径 |
| 文件名 | 文件名（便于统计） |
| 行号 | 错误所在行号 |
| 错误类型 | 错误宏类型 |
| 错误内容 | 提取的错误消息 |
| 完整内容 | 错误行完整代码（便于定位） |

---

**适用范围**: 所有 CANN 算子错误日志提取  
**关键要素**: 错误宏识别、共存处理、去重逻辑、完整性验证  
**更新时间**: 2026-04-30
