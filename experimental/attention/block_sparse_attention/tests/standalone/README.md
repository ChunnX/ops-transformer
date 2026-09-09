# effRows 主机侧回归

无需 Ascend SDK，直接编译生产代码中的逐块调度辅助函数：

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Werror test_eff_rows_tile.cpp -o /tmp/test_eff_rows_tile
/tmp/test_eff_rows_tile
```

覆盖空序列、全空块、单 tile、部分有效块、多 tile、非连续原始块索引、尾块裁剪、非零 batch/head 元数据偏移，以及 QK/PV 两级预启动调度的一致性。使用不同块的 K/V scale 对照有效行参考计算，另有固定种子的 2000 个随机场景。

此测试不执行 AscendC 搬运、矩阵计算或同步指令。上卡需验证 FP16/BF16 输出、不同 innerPrecise、TND/BNSD/BSND 布局、Q/KV 矩形块及 128/256/512 KV tile，并回归 quantMode=20 与不传 effRows 的路径。
