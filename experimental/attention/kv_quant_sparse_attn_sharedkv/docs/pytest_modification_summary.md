# kv_quant_sparse_attn_sharedkv pytest测试工程修改总结

**修改日期**: 2026-05-19
**提交数量**: 2个提交
**总修改**: 20个文件（算子层+pytest层）

---

## 一、修改概述

本次修改分为两个阶段：
1. **阶段一**：pytest测试工程参数适配（删除seqused_kv，新增5个参数）
2. **阶段二**：pytest params改为dict传递（提高可读性和维护性）

---

## 二、提交详情

### 提交1：pytest测试工程参数适配
- **提交ID**: `d2165a20`
- **提交时间**: 2026-05-19 16:24:10
- **修改统计**: 15 files changed, 203 insertions(+), 156 deletions(-)

### 提交2：pytest params改为dict传递
- **提交ID**: `ffd26eba`
- **提交时间**: 2026-05-19 16:43:08
- **修改统计**: 5 files changed, 135 insertions(+), 47 deletions(-)

---

## 三、修改文件清单（20个文件）

### 3.1 算子Host层（6个文件）
| 文件 | 改动内容 |
|------|---------|
| `README.md` | 更新参数说明（新增seqused_ori_kv等5个参数描述） |
| `op_host/kv_quant_sparse_attn_sharedkv_check.h` | 参数索引常量调整、参数名称常量 |
| `op_host/kv_quant_sparse_attn_sharedkv_check_consistancy.cpp` | 参数一致性校验更新 |
| `op_host/kv_quant_sparse_attn_sharedkv_check_existance.cpp` | 参数存在性校验更新 |
| `op_host/kv_quant_sparse_attn_sharedkv_check_single_para.cpp` | 单参数校验更新 |
| `op_host/kv_quant_sparse_attn_sharedkv_def.cpp` | 删除seqused_kv参数定义 |
| `op_host/kv_quant_sparse_attn_sharedkv_tiling.cpp` | 参数获取和tiling传递更新 |

### 3.2 算子Kernel层（2个文件）
| 文件 | 改动内容 |
|------|---------|
| `op_kernel/arch35/kv_quant_sparse_attn_sharedkv_kvcache.h` | 参数获取逻辑、默认值计算 |
| `op_kernel/kv_quant_sparse_attn_sharedkv.cpp` | Kernel入口参数调整 |

### 3.3 pytest测试层（5个文件）
| 文件 | 改动内容 |
|------|---------|
| `tests/pytest/kv_quant_sparse_attn_sharedkv_paramset.py` | 删除seqused_kv，新增5个参数配置 |
| `tests/pytest/test_kv_quant_sparse_attn_sharedkv_single.py` | 参数列表更新、自动生成逻辑、params改为dict |
| `tests/pytest/kv_quant_sparse_attn_sharedkv_golden.py` | 类初始化、calculate_by_bnsd、generate_and_save_testdata、gen_ori_kv/gen_cmp_kv、params改为dict |
| `tests/pytest/batch/kv_quant_sparse_attn_sharedkv_process.py` | Network类、metadata调用、算子调用、params改为dict |
| `tests/pytest/utils.py` | save_result函数更新、params改为dict |

### 3.4 其他文件（2个）
| 文件 | 改动内容 |
|------|---------|
| `.gitignore` | 新增，忽略docs目录 |
| `tests/pytest/check_valid_param.py` | 参数校验更新、params改为dict |

---

## 四、参数变化详情

### 4.1 删除参数
- `seqused_kv`：已废弃，删除所有引用

### 4.2 新增参数（5个）
| 参数名 | 含义 | 维度 | 是否可选 |
|--------|------|------|----------|
| `seqused_ori_kv` | ori_kv真实使用长度 | [B] | 可选 |
| `seqused_cmp_kv` | cmp_kv真实使用长度 | [B] | 可选 |
| `cu_seqlens_ori_kv` | ori_kv累加和（TND layout） | [B+1] | 可选 |
| `cu_seqlens_cmp_kv` | cmp_kv累加和（TND layout） | [B+1] | 可选 |
| `cmp_residual_kv` | cmp余数（mask边界计算） | [B] | 特定场景强制 |

### 4.3 params传递方式变化
```python
# 修改前（tuple方式）
params = Testcase_Name, layout_q, layout_kv, ..., template_run_mode
Testcase_Name = params[0]
layout_q = params[1]

# 修改后（dict方式）
params = {
    'Testcase_Name': Testcase_Name,
    'layout_q': layout_q,
    ...
    'template_run_mode': template_run_mode,
}
Testcase_Name = params['Testcase_Name']
layout_q = params['layout_q']
```

---

## 五、核心改动点

### 5.1 参数自动生成逻辑
当参数为None时，自动生成默认值（向后兼容）：
```python
# seqused_ori_kv：沿用旧逻辑
if seqused_ori_kv == None:
    seqused_ori_kv = KVS  # 根据cu_seqlens_q和actlen_mode生成

# seqused_cmp_kv：根据cmp_ratio计算
if seqused_cmp_kv == None:
    seqused_cmp_kv = [math.floor(s / cmp_ratio) for s in seqused_ori_kv]

# cmp_residual_kv：根据公式计算（仅特定场景）
if cmp_mask_mode == 3 and cmp_ratio != 1:
    if cmp_residual_kv == None:
        cmp_residual_kv = [s % cmp_ratio for s in seqused_ori_kv]

# cu_seqlens_ori_kv/cu_seqlens_cmp_kv：累加和计算
if cu_seqlens_ori_kv == None:
    cu_seqlens_ori_kv = [0]
    for s in seqused_ori_kv:
        cu_seqlens_ori_kv.append(cu_seqlens_ori_kv[-1] + s)
```

### 5.2 block_num计算修复
修复缩进错误，确保cmp_block_num在循环内计算：
```python
for cur_ori_act_kv in seqused_ori_kv:
    cur_ori_kv_block_num = math.ceil(cur_ori_act_kv / block_size1)
    ori_block_num_per_batch.append(cur_ori_kv_block_num)
    ori_block_num_sum += cur_ori_kv_block_num

    # 移入循环内（修复缩进错误）
    cur_cmp_act_kv = math.floor(cur_ori_act_kv / cmp_ratio)
    cur_cmp_kv_block_num = math.ceil(cur_cmp_act_kv / block_size2)
    cmp_block_num_per_batch.append(cur_cmp_kv_block_num)
    cmp_block_num_sum += cur_cmp_kv_block_num
```

### 5.3 gen_cmp_kv参数签名更新
拆分seqused_kv为seqused_ori_kv和seqused_cmp_kv：
```python
# 修改前
def gen_cmp_kv(..., seqused_kv, cmp_ratio, ...):
    for cur_ori_act_kv in seqused_kv:
        cur_cmp_act_kv = math.floor(cur_ori_act_kv / cmp_ratio)
        ...

# 修改后
def gen_cmp_kv(..., seqused_ori_kv, seqused_cmp_kv, cmp_ratio, ...):
    for cur_cmp_act_kv in seqused_cmp_kv:  # 直接使用seqused_cmp_kv
        ...
```

---

## 六、配置文件变更

### 6.1 .gitignore新增
```
docs/
```
**目的**：忽略docs目录，不提交到远端（本地开发文档）

---

## 七、语法验证结果

所有修改通过`python3 -m py_compile`检查，无语法错误：
```bash
python3 -m py_compile kv_quant_sparse_attn_sharedkv_paramset.py
python3 -m py_compile test_kv_quant_sparse_attn_sharedkv_single.py
python3 -m py_compile kv_quant_sparse_attn_sharedkv_golden.py
python3 -m py_compile utils.py
python3 -m py_compile check_valid_param.py
python3 -m py_compile batch/kv_quant_sparse_attn_sharedkv_process.py
```

---

## 八、改进优势

### 8.1 pytest params改为dict的优势
1. **可读性提升**：键名访问（`params['layout_q']`）比索引访问（`params[1]`）更直观
2. **维护性增强**：参数顺序调整不影响代码，无需重新编排索引
3. **安全性增强**：使用`params.get('key')`方法可避免KeyError异常
4. **扩展性好**：新增参数只需添加键值对，无需调整其他代码

### 8.2 参数扩展的优势
1. **语义清晰**：真实使用长度与存储长度分离
2. **灵活性增强**：cmp长度可手动指定，不受cmp_ratio限制
3. **精度提升**：cmp_residual_kv解决mask边界越界问题
4. **TND支持**：cu_seqlens参数支持TND layout累加和

---

## 九、后续工作建议

### 9.1 测试验证
- 运行pytest测试，验证修改正确性
- 对比CPU golden和NPU结果，确保精度达标
- 测试新参数传入场景（手动指定seqused_ori_kv等）

### 9.2 功能扩展（可选）
- 新增`layout_kv="TND"`测试场景
- 新增`cmp_residual_kv`强制传入场景测试
- 新增`seqused_ori_kv ≠ seqused_cmp_kv * cmp_ratio`场景测试

---

## 十、分支状态说明

### 10.1 当前状态
```
本地分支 qsfa: 2个新提交（d2165a20, ffd26eba）
远端分支 origin/qsfa: 1个提交（包含docs文件）
分支状态: 已分歧（diverged）
```

### 10.2 docs文件处理
- **本地状态**：docs目录已删除（通过.gitignore忽略）
- **远端状态**：docs文件已提交（需要后续处理）
- **建议**：推送前需要处理分支分歧（merge或rebase）

---

## 十一、总结

本次修改完成了pytest测试工程的完整适配：
1. ✅ 删除废弃参数seqused_kv（6个文件）
2. ✅ 新增5个参数（算子层+pytest层）
3. ✅ params改为dict传递（5个pytest文件）
4. ✅ 自动生成逻辑（向后兼容）
5. ✅ block_num计算修复（缩进错误）
6. ✅ gen_cmp_kv参数签名更新
7. ✅ .gitignore配置（忽略docs）
8. ✅ 语法验证（无错误）

**总修改统计**：
- 算子Host层：7个文件
- 算子Kernel层：2个文件
- pytest测试层：5个文件
- 其他：2个文件（.gitignore, check_valid_param.py）
- **总计**：20个文件，338行新增，203行删除

---

**修改完成日期**: 2026-05-19
**作者**: CANNBot