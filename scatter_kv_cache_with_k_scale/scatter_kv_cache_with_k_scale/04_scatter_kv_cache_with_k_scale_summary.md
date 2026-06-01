# scatter_kv_cache_with_k_scale 开发总结

## 编译状态：✅ 成功

## 文件清单

| 文件 | 说明 |
|------|------|
| CMakeLists.txt | 构建配置 |
| op_host/scatter_kv_cache_with_k_scale_def.cpp | 算子定义（4 dtype 组合） |
| op_host/scatter_kv_cache_with_k_scale_infershape.cpp | Shape/Dtype 推导 |
| op_host/arch35/scatter_kv_cache_with_k_scale_tiling.cpp | Tiling 实现 |
| op_kernel/scatter_kv_cache_with_k_scale_apt.cpp | 算子入口 |
| op_kernel/arch35/scatter_kv_cache_with_k_scale_simt.h | SIMT kernel |
| op_kernel/arch35/scatter_kv_cache_with_k_scale_tiling_data.h | Tiling 数据结构 |
| op_kernel/arch35/scatter_kv_cache_with_k_scale_tiling_key.h | Tiling Key |

## 编译产物

- Kernel 二进制：4 个 .o（build/binary/ascend950/bin/ascend950/scatter_kv_cache_with_k_scale_apt/）
- .run 包：build_out/cann-ops-math-custom_linux-x86_64.run

## 技术要点

- 纯访存算子，SIMT 编程模型，2048 线程
- VF 参数通过 LocalMemAllocator + UB 结构体打包，从 40×32bit 缩减至 16×32bit
- Host 端自实现 UintDiv magic/shift 计算
- FP8 数据使用 uint8 同位宽读写
