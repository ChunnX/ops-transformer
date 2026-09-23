# KvCacheTurboQuant · 交付实现根目录

本目录 = 本单唯一写码位置（浅挂在 ascend-work/deliver/ 下）。

| 路径 | 用途 |
|------|------|
| `op_host/` | Host：def / tiling / inferShape / aclnn |
| `op_kernel/` | Device Kernel；算法核=`encode_one_core.h` |
| `tests/` | 精度 / 性能自测 |
| `golden/` | 任务书金样 CPU 复刻（已有） |
| `docs/` | 设计文档草稿 |

任务书：`../../tasks/kv_cache_turbo_quant-a3/任务书/`
参考仓：`../../refs/ops-transformer/experimental/attention/`
算法真源：任务书 `golden.py`；本目录 `golden/cpu_golden.py` 为 CPU 复刻；Kernel 同核=`op_kernel/encode_one_core.h`。

### Kernel 语义中间验证（未挂 ops-transformer 前）

```bash
cd deliver/kv_cache_turbo_quant/tests
python3 gen_selftest_matrices.py
g++ -O2 -std=c++17 -o cpu_encode_twin cpu_encode_twin.cpp && ./cpu_encode_twin
g++ -O2 -std=c++17 -o verify_kernel_encode_vs_twin verify_kernel_encode_vs_twin.cpp
./verify_kernel_encode_vs_twin
python3 compare_twin_vs_python.py
```

期望：`ALL PASS verify_kernel_encode_vs_twin` + `ALL PASS compare_twin_vs_python`。
**阻塞**：本目录尚未挂入 `ops-transformer` 构建，故无 NPU 上板 Kernel 二进制证据；下一步=按 experimental/attention 目录范式接入仓内编译再跑 aclnn smoke。
