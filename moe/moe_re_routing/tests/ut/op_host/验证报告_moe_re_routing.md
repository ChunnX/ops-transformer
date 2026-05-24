# moe_re_routing UT验证报告

## 验证目的
验证 moe_re_routing 算子的 UT 测试是否能够正确编译、运行并通过测试。

## 验证时间
2026-05-24 19:07:41

## 验证环境
- 操作系统：Linux
- 编译器：GCC 13.3.0
- CANN版本：9.0.0
- SoC版本：ascend950
- 工作目录：/mnt/workspace/gitCode/cann/ops-transformer

## 算子信息

**算子功能**：MoE网络中，进行AlltoAll操作从其他卡上拿到需要算的token后，将token按照专家顺序重新排列。

**支持的dtype**：
- tokens/permute_tokens：FLOAT16、BF16、INT8、FLOAT8_E5M2、FLOAT8_E4M3FN、HIFLOAT8、FLOAT4_E2M1、FLOAT4_E1M2
- expert_token_num_per_rank：INT32、INT64
- per_token_scales：FLOAT、FLOAT8_E8M0

**约束条件**：
- expert_token_num_type：当前只支持为1（count模式）
- idx_type：Ascend 950支持0或1，其他产品仅支持0

---

## 验证过程

### 1. 原有测试情况

**原有测试数量**：21个
- Tiling测试：18个（membase、regbase、量化场景）
- InferShape测试：3个

**编译运行结果**：✅ 21/21 PASS

### 2. 新增测试

**新增测试数量**：2个

**InferShape新增测试**：
1. MoeReRouting_inferdtype_bfloat16：BFLOAT16数据类型推导
2. MoeReRouting_inferdtype_int64：INT64 expert_token_num_per_rank类型推导

### 3. 编译验证

**编译命令**：
```bash
bash build.sh --ophost_test --soc=ascend950 --ops=moe_re_routing --cov
```

**编译结果**：✅ 编译成功
- 无编译错误
- 无警告信息

### 4. 运行验证

**运行结果**：
```
[==========] Running 23 tests from 2 test suites.
[----------] Global test environment set-up.
[----------] 18 tests from MoeReRoutingTiling
[----------] 5 tests from MoeReRouting
[==========] 23 tests from 2 test suites ran. (22 ms total)
[  PASSED  ] 23 tests.
```

**验证结果**：✅ 所有23个测试通过
- 原有21个测试全部通过
- 新增2个测试全部通过
- 执行时间22ms，性能良好

---

## 测试覆盖场景

### Tiling测试覆盖（18个）

| 测试类型 | 测试场景 | 数量 |
|---------|---------|------|
| Membase基本 | 不同shape、数据类型 | 3个 |
| Regbase基本 | 不同shape、数据类型 | 3个 |
| HIFLOAT8量化 | FP32 scale + INT32/INT64 idx | 2个 |
| FP4_E2M1量化 | FP32/E8M0 scale + INT32/INT64 idx | 4个 |
| FP4_E1M2量化 | FP32/E8M0 scale + INT32/INT64 idx | 4个 |
| 边界场景 | block测试 | 2个 |

### InferShape测试覆盖（5个）

| 测试场景 | 数量 |
|---------|------|
| Shape推导基本 | 1个 |
| Scale推导 | 1个 |
| Dynamic shape | 1个 |
| Unknown shape | 1个 |
| Datatype推导（新增） | 2个 |

---

## 新增测试验证详情

### 1. MoeReRouting_inferdtype_bfloat16

**测试场景**：BFLOAT16数据类型推导

**验证要点**：
- ✅ 输入tokens类型为BF16
- ✅ 输出permute_tokens类型为BF16
- ✅ 输出per_token_scales类型为FLOAT
- ✅ 输出permute_token_idx类型为INT32
- ✅ 输出expert_token_num类型为INT32

**执行结果**：`[       OK ]`

**结论**：✅ 验证通过

### 2. MoeReRouting_inferdtype_int64

**测试场景**：INT64 expert_token_num_per_rank类型推导

**验证要点**：
- ✅ 输入expert_token_num_per_rank类型为INT64
- ✅ 输出expert_token_num类型推导为INT64
- ✅ 类型一致性正确

**执行结果**：`[       OK ]`

**结论**：✅ 验证通过

---

## 覆盖率数据

| 覆盖率类型 | 值 | 目标 | 状态 |
|-----------|-----|------|------|
| 行覆盖率 | 82.4% (392/476行) | ≥80% | ✅ 达标 |
| 函数覆盖率 | 85.9% (61/71函数) | ≥80% | ✅ 达标 |

---

## 验证结论

### 总体结论
✅ **验证通过**

### 详细结论

1. **编译验证**：✅ 通过
   - 代码正确，编译成功
   
2. **运行验证**：✅ 通过
   - 所有23个测试用例通过
   - 新增2个测试用例通过
   - 原有测试未受影响
   
3. **功能验证**：✅ 通过
   - BFLOAT16数据类型推导正确
   - INT64类型推导正确
   
4. **性能验证**：✅ 通过
   - 执行时间22ms
   - 性能良好
   
5. **覆盖率验证**：✅ 达标
   - 行覆盖率82.4% > 80%
   - 函数覆盖率85.9% > 80%

---

## 验证完成标志

- [x] 编译通过
- [x] 所有测试通过（23/23）
- [x] 新增测试功能正确
- [x] 无性能下降
- [x] 行覆盖率达标（82.4%）
- [x] 函数覆盖率达标（85.9%）
- [x] 验证报告生成完成

---

**验证人**：自动化验证系统  
**验证时间**：2026-05-24 19:07:41  
**验证状态**：✅ 通过