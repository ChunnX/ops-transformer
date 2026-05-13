# 2026-04-30 Raw MMAD Fused 交接总结

## 失败处理规则

- 精度不过时，必须先在当前优化点上正向定位和修复；不能把回滚到 PASS baseline 当成任务完成。
- 不能在失败候选上继续叠加下一项优化。需要恢复 baseline 时，只能作为明确放弃候选或重建干净工作面的动作，并记录失败现象、根因假设和下一步正向修复入口。


## 当前任务

目标是在 `C64_1616_FP32_CVInverse` 上实现真正 Stage1 fused inverse：

```text
INV += INV @ P
```

从当前 `MatmulImpl + GM workspace + FIX/MTE2` 串联，推进到 raw `LoadData/Mmad/Fixpipe` 路径，并尽量在 AIC 内完成累加。当前优先对象是：

- `op_kernel/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP32_CVInverse.h`
- 暂不修改 Stage2/Stage3
- 精度通过后再推广到其它方案

用户补充的底层 MMAD 参考：

- `/mnt/workspace/gitCode/cann/ops-transformer/attention/common/op_kernel/arch32/fia_block_cube_nonquant_gqa.h`

## 当前代码状态

### 默认方案

当前 selector 指向：

```cpp
#include "chunk_gated_delta_rule_neumann_stage1_C64_1616_FP32_CVInverse.h"
```

位置：

- `op_kernel/chunk_gated_delta_rule_neumann_stage1.h`

当前 host tiling：

- `chunkSize = 64`
- 已新增 micro tiling：`matmulTilingFp32C16/C32/C64`
- FP16 相关 tiling 也已加入，但当前默认不是 FP16 selector

位置：

- `op_host/chunk_gated_delta_rule_neumann_tiling.cpp`
- `op_kernel/chunk_gated_delta_rule_neumann_tiling_data.h`

### Raw MMAD helper 当前状态

`RawMmadInvAdd16()` 目前被临时置为 no-op：

- `op_kernel/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP32_CVInverse.h:759`

当前代码形态：

```cpp
__aicore__ inline void RawMmadInvAdd16(uint64_t invOffset, GlobalTensor<float> powerGm)
{
    (void)invOffset;
    (void)powerGm;
    return;
    ...
}
```

这意味着：

- kernel 可返回，不再挂死
- 但 `INV += INV@P` 三轮累加没有实际执行
- 因此当前版本不是精度正确版本，最小 case 会返回但精度失败

## 已确认事实

1. 外层 `C64_1616` 流程不挂。
2. `RawMmadInvAdd16()` 改成 no-op 后，最小 case 能返回。
3. 当前挂点集中在 `RawMmadInvAdd16()` 内部 raw GM/L1/L0/MMAD/Fixpipe 链路。
4. 不是 pytest、CPU golden、外层 compose、cross-core flag 导致挂死。
5. 之前 `MatmulImpl + GM workspace` 版本能通过全量三方精度，但性能仍被 tiny MatMul 固定开销限制。

## 已尝试但未解决

以下尝试仍会挂：

- 补齐 `M_FIX` 同步
- 显式 `Mmad(dst, a, b, bias=dst)` 改成 `Mmad(dst, a, b, params)`
- 只算 `INV@P -> GM`，不做累加
- raw local buffer 从 `1024B` 扩到 `4KB`

当前判断：

- 很可能是 `GM/ND -> L1 -> L0A/L0B -> Mmad -> M_FIX -> Fixpipe` 的参数或事件链不符合 AscendC raw MMAD 要求。
- 需要对照 FIA 的成熟写法重构或二分。

## 关键约束

- 不要自行 `git push`，除非用户明确要求。
- Stage2/Stage3 暂时不要动。
- CPU golden 逻辑不能改。
- 当前精度阈值：`atol=0.08, rtol=0.08`。
- 构建 SoC：`ascend910_93`。
- 性能必须用已有 `msprof` warmup/cache flush 脚本，不能自写 benchmark。
- 每阶段精度必须全量通过；性能可先 seq sweep。

## 推荐下一步

### 1. 对照 FIA raw MMAD 链路

打开并重点看：

```bash
sed -n '1,260p' /mnt/workspace/gitCode/cann/ops-transformer/attention/common/op_kernel/arch32/fia_block_cube_nonquant_gqa.h
```

需要对齐这些内容：

- GM 到 L1 的 `DataCopy` / ND2NZ 参数
- L1 到 L0A/L0B 的 `LoadData` 参数
- `MmadParams` 字段设置
- `M_FIX`、`FIX_MTE2`、`MTE1_M` 等事件顺序
- `FixpipeParamsV220` 的 `mSize/nSize/srcStride/dstStride`

### 2. 恢复并二分 `RawMmadInvAdd16()`

建议按最小返回点分段验证：

1. 只做 `GM -> L1` 后 return。
2. 加 `L1 -> L0A/L0B` 后 return。
3. 加 `Mmad(cmatrixInitVal=true)` 后 return。
4. 加 `M_FIX + Fixpipe` 后 return。
5. 最后再加 `INV` 预载和 `cmatrixInitVal=false`/bias 累加。

每一步都用最小 case 跑，确认是否挂死。

### 3. 最小验证命令

从 repo root 构建安装：

```bash
cd /mnt/workspace/gitCode/cann/ops-transformer
bash build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16
bash build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet
```

跑最小精度 smoke：

```bash
cd /mnt/workspace/gitCode/cann/ops-transformer/attention/chunk_gated_delta_rule_neumann/tests/pytest
export LD_LIBRARY_PATH=/home/developer/Ascend/cann-9.0.0/opp/vendors/custom_transformer/op_api/lib/:${LD_LIBRARY_PATH}
export ASCEND_CUSTOM_OPP_PATH=/home/developer/Ascend/cann-9.0.0/opp/vendors/custom_transformer
CGDR_GOLDEN_CHUNK_SIZE=64 timeout 120s python3 compare_chunk_gdr_three_way.py \
  --batch-size 1 --seqlen 16 --nk 1 --nv 1 --dk 16 --dv 16 \
  --seed 123 --atol 0.08 --rtol 0.08
```

全量精度：

```bash
CGDR_GOLDEN_CHUNK_SIZE=64 bash test_run.sh compare
```

## 当前性能背景

已有结论：

- C64_1616 micro tiling 后仍慢于 normal。
- FP16 C64_1616 也没有解决主瓶颈。
- 主要瓶颈不是理论 MAC，而是：
  - `MatmulImpl` tiny MatMul 调度固定开销
  - GM workspace 往返
  - `FIX/MTE2` 串联
  - AIC scalar 时间偏高

因此 raw MMAD fused 的验证价值在于：

- 去掉 `MatmulImpl` 通用调度开销
- 减少中间 GM workspace
- 在 CO1/L0C 里直接完成 `A@B + C` 累加路径

## 风险点

- CO1/L0C 可以作为 raw `Mmad` 的 bias/C 初值，但不能作为下一轮 A/B 输入。
- 跨 Neumann round 仍需要把结果搬到 L1/TSCM/GM 后再 LoadData 到 L0A/L0B。
- 如果 raw MMAD 参数错误，表现可能是 kernel 挂死而不是立即编译失败。
- 不要在未通过全量精度前做性能结论。

## 当前未完成项

- [ ] 对照 FIA 重构或二分 `RawMmadInvAdd16()`。
- [ ] 恢复真实 `INV += INV@P`。
- [ ] 最小 smoke 精度通过。
- [ ] `CGDR_GOLDEN_CHUNK_SIZE=64 bash test_run.sh compare` 全量通过。
- [ ] seq sweep msprof 性能验证。
- [ ] 若 C64_1616 raw fused 有收益，再推广到其它方案。
