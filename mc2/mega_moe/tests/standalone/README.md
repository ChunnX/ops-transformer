# GMM1 concat 地址检查

在仓库根目录运行，无需 CANN 或 NPU：

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Werror \
  mc2/mega_moe/tests/standalone/test_gmm1_concat_geometry.cpp \
  -o /tmp/test_gmm1_concat_geometry
/tmp/test_gmm1_concat_geometry
```

测试使用生产代码的 NZ burst 描述，模拟字节搬运，再与独立的 `[N,K]` 权重索引逐元素比较。
覆盖不同专家、多个 N tile、非零 K 偏移、短 K 尾块、目标缓冲区边界及 `SiLU(gate) * up` 配对。
它验证地址与布局，不替代设备上的 DMA、MX scale、量化精度和同步测试。
