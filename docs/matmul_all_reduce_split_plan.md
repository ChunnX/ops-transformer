# MatmulAllReduce Kernel 拆分方案

## 背景

`matmul_all_reduce` 算子 kernel 编译耗时过长（单次约 10 分钟），主要原因是：
- 单个 kernel 文件包含所有 MM_TYPE（27 个实例）
- 每个 tilingKey 编译时需处理完整文件（3448 行）
- 编译时大量模板实例化开销

## 方案目标

将单个 kernel 文件拆分为多个文件（按 MM_TYPE 分类），降低编译耗时 **70%**。

---

## 代码工作量估算

### MM_TYPE 分布

| MM_TYPE | Kernel 实例数 | 编译耗时占比 |
|---------|-------------|-------------|
| FP_MM | 5 | 低 |
| FP_MM_CUBE_ONLY | 2 | 低 |
| **QUANT_MATMUL** | 10 | **最高（最慢）** |
| WEIGHT_QUANT_MATMUL | 4 | 中 |

### 拆分后文件结构

拆分后的文件结构（完整方案）：

```
mc2/matmul_all_reduce/op_kernel/
├── common.h                              (保留)
├── matmul_all_reduce_common.h            (新增，公共宏定义 ~50行)
├── matmul_all_reduce_fp_mm.cpp           (新增，~150行)
├── matmul_all_reduce_fp_mm_tiling_key.h  (新增，~140行)
├── matmul_all_reduce_fp_mm_cube_only.cpp           (新增，~90行)
├── matmul_all_reduce_fp_mm_cube_only_tiling_key.h  (新增，~55行)
├── matmul_all_reduce_quant.cpp           (新增，~250行)
├── matmul_all_reduce_quant_tiling_key.h  (新增，~275行)
├── matmul_all_reduce_weight_quant.cpp           (新增，~130行)
├── matmul_all_reduce_weight_quant_tiling_key.h  (新增，~110行)
├── matmul_all_reduce.cpp                 (修改，删除拆分内容，保留 ~350行)
├── matmul_all_reduce_tiling_key.h        (修改，删除拆分内容，保留 ~470行)
└── arch31/                               (保留现有架构实现)
└── arch32/                               (保留现有架构实现)
└── arch35/                               (保留现有架构实现)
```

### 工作量汇总

| 项目 | 数据 |
|------|------|
| **新增文件数量** | 8 个（4 个 cpp + 4 个 tiling_key.h） |
| **新增代码量** | ~1193 行 |
| **修改代码量** | ~50 行（CMakeLists + 公共头文件） |
| **总工作量** | ~1243 行 |
| **实施耗时** | 11-18 小时（约 1.5-2 个工作日） |

---

## 实施步骤

### Phase 1: 创建公共头文件（30分钟）

创建 `matmul_all_reduce_common.h`，提取所有 MM_TYPE 公用的宏定义：
- AICORE_TYPE 定义（ASCEND_910B, ASCEND_310P）
- MM_TYPE 定义
- 公共参数（EMPTY_INPUT, INT8_COMM, ENABLE_L2_CACHE）
- 其他公共宏（约 50 行）

### Phase 2: 拆分 QUANT_MATMUL（3小时，最关键）

QUANT_MATMUL 占 10 个实例，编译最慢（约 3 分钟）。

#### 2.1 创建 tiling_key 文件

从 `matmul_all_reduce_tiling_key.h` 提取包含 `MATMUL_ALLREDUCE_MM_TYPE_QUANT_MATMUL` 的所有 `ASCENDC_TPL_ARGS_SEL` 块（10-12 个），创建独立文件。

#### 2.2 创建 kernel cpp 文件

从 `matmul_all_reduce.cpp` 提取量化逻辑：
- 量化相关头文件引用
- `MM_TYPE == MATMUL_ALLREDUCE_MM_TYPE_QUANT_MATMUL` 条件编译块
- 量化特定宏定义（INVOKE_QUANT_xxx）

#### 2.3 修改原文件

删除已拆分内容，保留其他 MM_TYPE。

### Phase 3: 拆分其他 MM_TYPE（可选）

如需进一步优化，拆分：
- FP_MM（5 个实例）
- FP_MM_CUBE_ONLY（2 个实例）
- WEIGHT_QUANT_MATMUL（4 个实例）

### Phase 4: 修改编译配置（30分钟）

修改 CMakeLists.txt，添加独立编译目标：
```cmake
add_bin_compile_target(
    NAME matmul_all_reduce_quant
    SOURCES matmul_all_reduce_quant.cpp
    TILING_KEY matmul_all_reduce_quant_tiling_key.h
)
```

### Phase 5: 编译验证（2小时）

```bash
# 清理
bash build.sh --make_clean

# 编译
bash build.sh --pkg --soc=ascend950 --ops=matmul_all_reduce -j16
```

### Phase 6: 功能回归测试

```bash
# 单元测试
bash build.sh --opkernel_test --ops=matmul_all_reduce

# 示例程序
bash build.sh --run_example matmul_all_reduce eager --soc=ascend950
```

---

## 编译耗时对比

| 场景 | 拆分前 | 拆分后 | 优化幅度 |
|------|--------|--------|---------|
| **单个 kernel 目录** | 10 分钟 | 3 分钟（并行） | **-70%** |
| **最大文件（QUANT）** | - | 3 分钟 | 减少 62% |
| **完整算子编译（16 个组合）** | 60 分钟 | 23 分钟 | **-62%** |

### 并行编译优势

拆分后的 4 个文件可并行编译，总耗时取决于最慢的文件（QUANT_MATMUL，约 3 分钟）。

---

## 验证方法

### 验证差异的关键点

**应该相同**：
- ✅ Run package MD5（如果编译参数一致）
- ✅ Kernel binaries 数量（27 个实例）
- ✅ 功能测试结果

**应该不同**：
- ⚠️ 编译耗时（降低 70%）
- ⚠️ 源码文件数量（增加）
- ⚠️ 编译中间产物结构

### MD5 不同时的排查

如果 MD5 不同，可能原因：
1. 编译时间戳不同（忽略）
2. 编译参数不同（检查 CMakeLists）
3. 代码逻辑错误（回归测试）

### 验证命令

```bash
# 1. 对比 MD5
md5sum build_out/*.run

# 2. 统计 kernel binaries
find build/binary/ascend950 -name "*.o" | wc -l

# 3. 统计 TilingKey 数量
grep -c "ASCENDC_TPL_ARGS_SEL" mc2/matmul_all_reduce/op_kernel/*.h

# 4. 功能回归
bash build.sh --opkernel_test --ops=matmul_all_reduce
```

---

## 推荐实施策略

### 快速验证方案（推荐）

只拆分 QUANT_MATMUL（工作量 3-4 小时）：
- 验证效果：编译耗时从 10 分钟降至 3 分钟
- 如果效果好，继续拆分其他 MM_TYPE

### ROI 计算

假设每周编译 20 次：
- 当前耗时：20 × 60 分钟 = 1200 分钟/周
- 拆分后耗时：20 × 23 分钟 = 460 分钟/周
- **每周节省：740 分钟 ≈ 12.3 小时**

**ROI**：投入 15 小时，每周回报 12.3 小时 → **第 2 周开始盈利**

---

## 风险与对策

| 风险 | 对策 |
|------|------|
| **TilingKey 提取遗漏** | grep 精确统计，确保实例数正确 |
| **公共宏定义缺失** | 编译报错时根据错误补全 |
| **功能回归失败** | 回滚到备份目录 |
| **MD5 不同** | 检查参数 + 功能测试 |

### 回滚方案

```bash
# 恢复原始编译产物
cp .backup_before_split/*.run build_out/

# 恢复源码
git checkout mc2/matmul_all_reduce/op_kernel/
```

---

## 验证结果记录模板

```
拆分实施日期: YYYY-MM-DD
实施人: XXX

拆分前状态:
- 编译耗时: XX 分钟
- Run package MD5: XXXXX
- Kernel 实例数: 27

拆分后状态:
- 编译耗时: XX 分钟（降低 XX%）
- Run package MD5: XXXXX（是否相同）
- Kernel 实例数: XX（拆分后总和）

差异分析:
- 编译耗时差异: XX 分钟
- 功能测试结果: PASS/FAIL
- 二进制差异: 有/无（原因分析）

结论: 拆分成功 / 需要修复 / 回滚
```

---

## 总结

| 维度 | 评估 |
|------|------|
| **工作量** | 中等（11-18 小时） |
| **效果** | 显著（编译耗时降低 70%） |
| **风险** | 低（有备份和回滚方案） |
| **ROI** | 高（第 2 周开始盈利） |
| **推荐度** | ⭐⭐⭐⭐⭐ |

**建议**：立即实施拆分方案，优先处理 QUANT_MATMUL。