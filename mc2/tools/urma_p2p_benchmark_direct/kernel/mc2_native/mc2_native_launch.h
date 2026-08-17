// ============================================================================
// mc2_native_launch.h — mc2_native 模式的 launch 函数声明
//
// Host 侧 (test_urma_p2p_bench.cpp) 通过此头文件调用 mc2_native_bench_launch()
// 启动 kernel。
//
// 参数说明:
//   blockDim      - 核心数 (当前固定 1, 单核执行)
//   stream        - ACL stream
//   hcclContextGM - HcclCombinOpParam 在 device GM 中的地址
//                   (Host 侧 build_hccl_context 构造后 aclrtMemcpy H2D)
//   workspace     - 本端 URMA workspace 地址 (URMA 源数据缓冲区)
//   timingGm      - timing 输出 GM 地址 (8 个 int64_t, Host 侧读取打印)
//   tiling        - UrmaP2pBenchTiling 参数结构体
// ============================================================================

#ifndef MC2_NATIVE_LAUNCH_H
#define MC2_NATIVE_LAUNCH_H

#include <cstdint>
#include "../common/urma_p2p_bench_tiling.h"

void mc2_native_bench_launch(uint32_t blockDim, void *stream,
                             uint64_t hcclContextGM, uint64_t workspace,
                             uint64_t timingGm, UrmaP2pBenchTiling *tiling);

#endif
