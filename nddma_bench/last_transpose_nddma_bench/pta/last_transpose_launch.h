/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file last_transpose_launch.h
 * \brief LastTranspose NDDMA kernel 的 PTA 直调 C 接口（供 pybind 扩展调用）。
 *        由 ASC 编译的 liblast_transpose_launch.so 实现；pybind 端仅按 C ABI 声明调用。
 */

#ifndef LAST_TRANSPOSE_LAUNCH_H
#define LAST_TRANSPOSE_LAUNCH_H

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 在给定 device 指针上执行 [0,1,3,2] NDDMA 转置。
 * 输入/输出均为已在 NPU 上的 device 内存（由 PyTorch/torch_npu 分配）。
 *
 * @param xDev     输入 device 指针，shape (dim0,dim1,dim2,dim3)
 * @param yDev     输出 device 指针，shape (dim0,dim1,dim3,dim2)
 * @param dim0..3  维度；dim0 作为分核数（blockNum）
 * @param dtype    0=fp16, 1=fp32, 2=int8(b8)
 * @param isMte3   1=写回 GM（正常转置输出）；0=去掉 MTE3（纯读带宽）
 * @param extStream 外部 aclrtStream；传 nullptr 时内部自建流并同步
 * @return 0 成功，非 0 为 ACL 错误码
 */
int LaunchLastTranspose(void* xDev, void* yDev,
                         uint32_t dim0, uint32_t dim1, uint32_t dim2, uint32_t dim3,
                         uint32_t dtype, uint32_t isMte3, void* extStream);

/**
 * 自包含 host 入口：输入/输出均为 HOST 指针（CPU 张量 data_ptr）。
 * 本函数内部完成 aclInit/SetDevice（进程内仅一次）、device malloc、H2D 拷贝、
 * kernel 发射、D2H 拷回。用于 torch_npu 不支持当前 SoC（Ascend950PR 9579）时，
 * 绕开 torch_npu 设备后端，仍以 PyTorch(CPU) 张量驱动真实 NPU kernel。
 *
 * @param xHost   输入 host 指针，shape (dim0,dim1,dim2,dim3)
 * @param yHost   输出 host 指针，shape (dim0,dim1,dim3,dim2)，需已分配 numel*elemSize 字节
 * @param dim0..3 维度；dim0 作为分核数
 * @param dtype   0=fp16, 1=fp32, 2=int8(b8)
 * @param isMte3  1=正常转置输出；0=纯读带宽（输出无意义）
 * @param deviceId 目标 NPU 卡号
 * @return 0 成功，非 0 为 ACL 错误码
 */
int LaunchLastTransposeHost(const void* xHost, void* yHost,
                             uint32_t dim0, uint32_t dim1, uint32_t dim2, uint32_t dim3,
                             uint32_t dtype, uint32_t isMte3, int32_t deviceId);

#ifdef __cplusplus
}
#endif

#endif  // LAST_TRANSPOSE_LAUNCH_H
