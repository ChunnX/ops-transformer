---
name: tilelang-debug-cpp-diff
description: 用 C++ kernel 代码 diff 定位 TileLang 算子运行时问题（崩溃/精度）。dump 两个配置的 kernel source → diff 找差异 → 定位 buffer overflow / auto-sync bug / 参数差异。适用场景：VEC engine crash、UB overflow、精度不一致。
---

# TileLang C++ Kernel Diff 调试方法

## 适用场景

- Kernel 在特定参数下崩溃（aicore exception），其他参数正常
- 精度不一致，怀疑编译产物有差异
- UB 溢出、auto-sync flag 缺失等编译期问题

## Step 1: Dump 编译产物

```python
import tilelang; tilelang.disable_cache()
from my_kernel import make_my_kernel

func = make_my_kernel(param1=..., param2=...)
kernel_source = func.get_kernel_source()  # 返回 C++ 源代码字符串

with open('/tmp/kernel.cc', 'w') as f:
    f.write(kernel_source)
```

## Step 2: 生成两个配置的 kernel 并 diff

```bash
diff /tmp/kernel_good.cc /tmp/kernel_bad.cc
```

## Step 3: 分析差异

### 3a. Buffer 大小与 offset

```
auto buf = ascend_ub.GetWithOffset<float>(SIZE_BYTES/4, OFFSET);
```

- `SIZE_BYTES/4` 是 float 元素数。对比两个配置的 size 是否合理。
- `OFFSET` 是 UB 起始偏移。检查是否有意外 alias（两个 buffer 同 offset）。

### 3b. 操作参数 vs buffer 容量

```
MergeSort<float>(dst, tmp, src1, src2, n1, n2);
```

- `n1`, `n2` 是 PAIRS 数（每 pair 包含 score + index 两个 float）
- dst 需要容纳 `2 * (n1 + n2)` floats
- **检查**: `dst_size >= 2 * (n1 + n2)`

```
copy_ub_to_ub<float, float, N>(dst, src, ...);
copy_gm_to_ub<float, N>(dst, src, ...);
```

- `N` 是拷贝元素数。dst/src 必须 ≥ N。

### 3c. Auto-sync flag 差异

```
SetFlag<MTE3_V>(id);
WaitFlag<MTE3_V>(id);
```

- 对比两个配置的 flag 位置和 ID
- Flag 缺失可能导致 MTE/DMA 未完成时 VEC 写入同一地址 → crash
- 常见缺失点：`copy_ub_to_gm` (DMA) 之后、`Cast` 之前

### 3d. Flag ID 冲突

- Auto-sync 分配的 flag ID 通常为 1-7
- 手动插入的 flag 应使用 >10 的 ID 避免冲突

## Step 4: 根因分类

| 症状 | 常见根因 | 修复方向 |
|------|---------|---------|
| buffer size 小于操作需求 | 合并 buffer 容量计算错误 | 增大 `T.alloc_ub` 的 size 参数 |
| 两个 buffer 同 offset | Memory planner alias | 手动调整 allocation 顺序或加 padding |
| Flag 位置不同 / 缺失 | Auto-sync bug | 加显式 `T.set_flag`/`T.wait_flag` |
| Flag ID 冲突 | 手动 flag 与 auto-sync 冲突 | 用手动 flag 使用 >10 的 ID |

## 典型案例

### 案例 1: merge_sort buffer overflow

```diff
- auto merged_ub = GetWithOffset<float>(256, 8768);   // 256 floats = 128 pairs
+ auto merged_ub = GetWithOffset<float>(640, 8768);   // 640 floats = 320 pairs

  MergeSort(merged_ub, ..., topk_a, cache_merged, 64, 256);
  //                                              64+256=320 pairs > 128 overflow!
```

修复：`merged_ub = T.alloc_ub(max(2*_TA2, 2*(TOP_K_ALIGNED + 4*_K_PER_BLOCK)), calc_dtype)`

### 案例 2: auto-sync 漏 MTE3_V flag

```diff
- Cast(output_ub, CAST_ROUND, 2048);          // 没有 sync！
+ SetFlag<MTE3_V>(7); WaitFlag<MTE3_V>(7);    // auto-sync 加了 flag
+ Cast(output_ub, CAST_ROUND, 1024);
```

修复：在 DMA 和 Cast 间加显式 `T.set_flag("MTE3", "V", 15)` + `T.wait_flag("MTE3", "V", 15)`

## 注意事项

- **先清缓存**: `for m in list(sys.modules): if 'my_module' in m: del sys.modules[m]`
- **调用 `tilelang.disable_cache()`** 避免 JIT 缓存
- **每个配置单独进程测试** 避免 NPU 脏状态影响
- 优先对比 C++ 代码，不是 TIR（C++ 更接近硬件，直接显示 buffer offset 和 flag）