# KvCacheTurboQuant（Encode-only）

挂载目录：`experimental/attention/kv_cache_turbo_quant`  
配方：`mse_bits∈{2,3,4}` + 1-bit QJL；`head_dim=128`；任务书存储口径约 4.25 bit / 3.76×（默认 bits=3）。

| 路径 | 用途 |
|------|------|
| `op_host/` | def / InferShape / Tiling / aclnn |
| `op_kernel/` | MIX Kernel；算法核 `encode_one_core.h` |
| `examples/` | aclnn smoke |
| `golden/` | CPU 金样形状/语义 |
| `tests/` | CPU 孪生与精度脚本（不含大体积 NPU case 数据） |
| `docs/design.md` | 设计说明 |

## 编译（示例）

```bash
bash build.sh --pkg --experimental --soc=ascend910_93 --ops=kv_cache_turbo_quant -j8
# 或验收机：--soc=ascend910b
```

## 自测摘要（本机 A3 / CANN 9.0）

- decode idx/qjl exact；§3.2 IP95 PASS  
- 五档性能 warmup+SynchronizeStream ALL ≥10×  

复现命令见 `docs/挂仓编译与NPU_smoke.md`、`docs/精度门禁_NPU_S32.md`。
