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
 * \file nddma_broadcast_launch.h
 * \brief Broadcast NDDMA kernel 的 PTA 直调 C 接口（供 pybind 扩展调用）——3D NDDMA 方案（DESIGN §9，权威）。
 *        由 ASC 编译的 libnddma_broadcast_launch.so 实现；pybind 端仅按 C ABI 声明调用。
 *
 * 5D 四场景（AB/BA/BAB/ABA）统一为单条 3D NDDMA（dim1 唯一核内 for，dim2/dim3/dim4 为 loopInfo 三轴）。
 *   输入 shape 随 scene（广播轴长度=1），输出恒 (dim0,dim1,dim2,dim3,dim4)。
 */

#ifndef NDDMA_BROADCAST_LAUNCH_H
#define NDDMA_BROADCAST_LAUNCH_H

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 5D 广播场景直调（DESIGN §9，权威）：4 场景（AB/BA/BAB/ABA）统一为单条 3D NDDMA，
 * dim1 唯一核内 for，dim2/dim3/dim4 为 loopInfo 三轴（§8 方案 A 的 dim2-for 已作废）。
 *
 * 输入/输出均为已在 NPU 上的 device 指针（torch_npu 分配，零拷贝）。
 * 输入 shape 随 scene（广播轴长度=1，单条 3D 整块 DESIGN §9.2）：
 *   AB  (dim0,dim1,dim2,dim3,1)   BA  (dim0,dim1,dim2,1,dim4)
 *   BAB (dim0,dim1,1,dim3,1)      ABA (dim0,dim1,dim2,1,dim4)
 * 输出恒 (dim0,dim1,dim2,dim3,dim4)。
 *
 * @param xDev/yDev  device 指针
 * @param dim0..dim4 5D 维度；分核轴=dim0
 * @param scene      0=AB 1=BA 2=BAB 3=ABA
 * @param dtype      0=fp16 1=fp32 2=int8
 * @param isMte3     1=写回 GM（正常输出）；0=纯读带宽
 * @param blockNum   发射核数（核数扫描用）；传 0 → 缺省 min(dim0,56)
 * @param extStream  外部 aclrtStream；传 nullptr 内部自建并同步
 * @return 0 成功，非 0 为 ACL 错误码
 */
int LaunchNddmaBroadcast5d(void* xDev, void* yDev,
                           uint32_t dim0, uint32_t dim1, uint32_t dim2,
                           uint32_t dim3, uint32_t dim4, uint32_t scene,
                           uint32_t dtype, uint32_t isMte3,
                           uint32_t blockNum, void* extStream);

#ifdef __cplusplus
}
#endif

#endif  // NDDMA_BROADCAST_LAUNCH_H
