# GMM/GMMFR 算子错误码规范化整改
# STC软件测试用例与测试报告（System Test Case）

## 1. 测试策略概述

### 1.1 测试目标

**主要目标：**
- 验证错误码规范化整改后，错误信息上报正确性
- 验证 EZ0008-EZ0017 错误码按规范上报
- 验证错误日志格式符合 CANN 错误码规范
- 验证参数名称准确传递到错误信息
- 验证整改后功能不受影响

**测试范围：**
| 算子 | 整改文件数 | 整改错误数 | 测试重点 |
| --- | ---: | ---: | --- |
| GMM | 3个重点文件 | ~341条 | A5伪量化场景错误码上报 |
| GMMFR | 1个重点文件 | ~130条 | A5伪量化场景错误码上报 |

### 1.2 测试分层

| 测试层级 | 测试类型 | 测试对象 | 验证内容 | 测试工具 |
| --- | --- | --- | --- | --- |
| **UT** | 单元测试 | op_api/op_host函数 | 错误码上报正确性、参数验证逻辑 | gtest/ttk |

### 1.3 测试设计原则

**ErrMsg整改测试设计原则：**
1. **错误场景全覆盖**：覆盖所有整改的错误类型（dtype/format/shape/dim/value等）
2. **错误码精确验证**：验证上报的EZ错误码与错误类型匹配
3. **参数名准确传递**：验证错误日志中的参数名与实际参数名一致
4. **错误信息准确性**：验证错误原因描述准确清晰
5. **反向测试为主**：重点测试异常场景的错误上报，而非正常功能

---

## 2. UT测试用例设计（Unit Test）

### 2.1 UT测试用例脑图

```
UT测试用例脑图
├─ 1. 参数验证错误测试
│  ├─ 1.1 输入shape错误（EZ0008/EZ0009/EZ0010/EZ0011）
│  │  ├─ shape不匹配 → EZ0008
│  │  ├─ shape维度错误 → EZ0009
│  │  ├─ shape错误含原因 → EZ0010
│  │  └─ shape size错误 → EZ0011
│  ├─ 1.2 输出shape错误（EZ0012/EZ0013/EZ0014/EZ0015）
│  │  ├─ 输出shape不匹配 → EZ0012
│  │  ├─ 输出shape维度错误 → EZ0013
│  │  ├─ 输出shape错误含原因 → EZ0014
│  │  └─ 输出shape size错误 → EZ0015
│  ├─ 1.3 输出格式错误（EZ0016/EZ0017）
│  │  ├─ 输出format不匹配 → EZ0016
│  │  └─ 输出dtype不匹配 → EZ0017
│  ├─ 1.4 输入format/dtype错误（旧版宏EZ0006/EZ0007）
│  │  ├─ 输入format不匹配 → EZ0006
│  │  └─ 输入dtype不匹配 → EZ0007
│  └─1.5 其他参数错误（新增错误码）
│     ├─ 参数值非法
│     ├─ 参数值范围错误
│     ├─ group参数错误
│     ├─ scale参数错误
│     └─ quant参数错误
└─ 2. 内部错误测试（新增错误码）
   ├─ 2.1 空指针错误
   │  ├─ context空指针 → EZ_INNER_NULLPTR（上报新增）
   │  ├─ tensor空指针 → EZ_PARAM_NULLPTR（评审决定）
   │  └─ executor空指针 → EZ_INNER_CREATE_EXECUTOR（上报新增）
   ├─ 2.2 运行时错误
   │  ├─ 内存分配失败 → EZ_RUNTIME_ERROR（上报新增）
   │  └─ 执行器创建失败 → EZ_INNER_CREATE_EXECUTOR（上报新增）
   └─ 2.3 CHECK_RET错误
      ├─ CHECK_RET失败 → EZ_INNER_NULLPTR（上报新增）
      └─ 内部断言失败 → EZ_INNER_NULLPTR（上报新增）
```

其中测试用例覆盖全部修改，无法构建的场景参考脑图保证ErrCode级别覆盖

### 3. 错误码说明

**重要说明：**
根据 `/home/developer/Ascend/cann/aarch64-linux/pkg_inc/op_common/log/log.h` 文件，当前 CANN 定义的标准错误码如下：

**输入验证错误码（EZ0008-EZ0011）：**
- EZ0008: 输入shape异常 - "OP {opName} input {inputName} has incorrect shape [{incorrectShape}], it should be [{correctShape}]."
- EZ0009: 输入shape dim异常 - "OP {opName} input {inputName} has incorrect shape dim [{incorrectDim}], it should be [{correctDim}]."
- EZ0010: 输入shape异常（含原因） - "OP {opName} input {inputName} has incorrect shape [{incorrectShape}], reason: {reason}."
- EZ0011: 输入shape size异常 - "OP {opName} input {inputName} has incorrect shape size {incorrectSize}, it should be {correctSize}."

**输出验证错误码（EZ0012-EZ0017）：**
- EZ0012: 输出shape异常
- EZ0013: 输出shape dim异常
- EZ0014: 输出shape异常（含原因）
- EZ0015: 输出shape size异常
- EZ0016: 输出format异常
- EZ0017: 输出dtype异常

**旧版输入验证宏（EZ0001-EZ0007）：**
- EZ0001: 输入shape异常（使用序号index）
- EZ0006: 输入format异常（使用序号index）
- EZ0007: 输入dtype异常（使用序号index）

**通用错误码：**
- EZ9999: 通用内部错误 - 用于其他未明确定义的错误类型

**注意：**
1. dtype/format相关的参数错误应使用 EZ0006/EZ0007
2. value/tensor num等其他参数错误应使用 EZ9999（通用错误码）
3. 如需新增专门的错误码（如value错误、tensor num错误），需要向 CANN 团队提交错误码定义申请

**示例1：输入shape错误**
```
[ERROR] EZ0008: OP grouped_matmul input x has incorrect shape [64,128], it should be [128,256].
```

**示例2：输入shape dim错误**
```
[ERROR] EZ0009: OP grouped_matmul input x has incorrect shape dim [2D], it should be [3D].
```

**示例3：输出dtype错误**
```
[ERROR] EZ0017: OP grouped_matmul output y has incorrect dtype DOUBLE, it should be FLOAT16.
```
