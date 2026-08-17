# V121 LSE 临时缓冲区容量修复设计

## 目标

修复 arch-920r1 路径中 `lseTmpBuff` 容量不足导致的 UB 越界，不改变 LSE 数值计算、流水顺序或 attentionOut 算法。

## 设计

`ComputeLseOutputVF` 对每 8 行输出一个 64-float 寄存器块，因此每个逻辑行在 UB 中占 8 个 float。V121 每个 Vec1 task 处理 64 行，单个 LSE 临时缓冲区需要 `64 * 8 * sizeof(float) = 2048` 字节。

在 `QBSABlockVec::InitLocalBuffer` 中使用由 `s1BaseSize` 推导的 `lseTmpBufferSize` 初始化两个临时缓冲区。保留双缓冲索引、LSE VF 和 DataCopyPad 路径不变。

## 验证

- 容量检查必须从当前 `[1024, 1024]` 失败，并在修改后确认两个缓冲区均满足 2048B。
- 检查代码差异只涉及 LSE 缓冲区容量及说明。
- 运行仓库中可用的格式、静态或编译检查；硬件精度回归若环境没有 NPU，则明确留给目标环境验证。
