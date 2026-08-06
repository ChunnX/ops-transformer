/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file bitcast_proto.h
 * \brief 图模式原型定义（来自 ops-math common/inc/op_graph/op_math_proto_extend.h，供 fusion pattern stub）
 */
#ifndef BITCAST_STUB_PROTO_H
#define BITCAST_STUB_PROTO_H

#include "graph/operator_reg.h"

namespace ge {
/**
 *@brief This operation convert output dataType and shape.

 *@par Inputs:
 *The input handle must have the resource type. Inputs include:
 *x:A list of Tensor objects. One or more tensors from which
 the enqueued tensors should be taken . \n

 *@par Outputs:
 *y:A list of Tensor objects. One or more tensors from which
 the enqueued tensors should be taken . \n

 *@par Attributes:
 *type: An optional ge::DataType. It refers to the target data type of outputs . \n
 *keepdim: If True, output dims equal intput dims. otherwise output dims can append. default value is false.  \n

 *@par Third-party framework compatibility
 *Compatible with tensorflow QueueIsClosed operator.
 */
REG_OP(Bitcast)
    .INPUT(x, TensorType({DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_INT4, DT_INT8, DT_INT32, DT_UINT32, DT_UINT8, DT_INT64,
                          DT_UINT64, DT_INT16, DT_UINT16, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128, DT_QINT8, DT_QUINT8,
                          DT_QINT16, DT_QUINT16, DT_QINT32}))
    .OUTPUT(y, TensorType({DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_INT4, DT_INT8, DT_INT32, DT_UINT32, DT_UINT8, DT_INT64,
                           DT_UINT64, DT_INT16, DT_UINT16, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128, DT_QINT8, DT_QUINT8,
                           DT_QINT16, DT_QUINT16, DT_QINT32}))
    .REQUIRED_ATTR(type, Type)
    .ATTR(keep_dim, Bool, false)
    .OP_END_FACTORY_REG(Bitcast)
} // namespace ge

#endif // BITCAST_STUB_PROTO_H
