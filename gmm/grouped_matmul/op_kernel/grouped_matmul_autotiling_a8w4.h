/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul_autotiling_a8w4.h
 * \brief
 */
#ifndef ASCENDC_GROUPED_AUTOTILING_A8W4_H
#define ASCENDC_GROUPED_AUTOTILING_A8W4_H

#include "grouped_matmul_utils.h"
#include "grouped_matmul.h"

//内存设置别名
#define GlobalMem               TPosition::GM
#define UBMem                   TPosition::VECCALC
#define L1AMem                  TPosition::A1
#define L1BMem                  TPosition::B1
#define L0AMem                  TPosition::A2
#define L0BMem                  TPosition::B2
#define L0CMem                  TPosition::CO1
#define L0C2Mem                 TPosition::CO2

// 数据类型长度
#define INT2                    0.25
#define INT4                    0.5
#define INT6                    0.75
#define INT8                    1
#define INT16                   2
#define INT32                   4
#define INT64                   8
#define FP4                     0.5
#define FP6                     0.75
#define FP8                     1
#define FP16                    2
#define FP32                    4
#define FP64                    8

/* 计算过程中系数 */
#define FACTOR_2                2
#define FACTOR_4                4
#define FACTOR_8                8
#define FACTOR_16               16
#define FACTOR_64               64
#define FACTOR_22               22
#define FACTOR_7                7
#define FACTOR_128              128
#define FACTOR_4096             4096
/* 维度数值 */
#define DIM_VAL1                1
#define DIM_VAL2                2
#define DIM_VAL3                3
/* 数组下标 */
#define ARRAY_IDX0              0
#define ARRAY_IDX1              1
#define ARRAY_IDX2              2
/* 平台硬件相关参数 */
#define ELEM_BASE_ALIGN         16  // mixmum alignment value for element number in one dimension of tensor
#define RATIO_AIV2AIC           2   // ratio of number between AIV and AIC
#define COPY_BLK_BYTES          32  // bytes of data for one data move
#define MUL_BYTES               256 // bytes of mul each
#define MAT_FRAC_BYTES          512 // bytes of matrix fractal
/* 硬件同步ID */
#define LOCAL_HWEVENT_ID0       0
#define LOCAL_HWEVENT_ID1       1
#define LOCAL_HWEVENT_ID2       2
#define LOCAL_HWEVENT_ID3       3
#define LOCAL_HWEVENT_ID4       4
#define LOCAL_HWEVENT_ID5       5
#define LOCAL_HWEVENT_ID6       6
#define LOCAL_HWEVENT_ID7       7
/* Mode ID */
#define LOCAL_MODEID0           0x0
#define LOCAL_MODEID1           0x1
#define LOCAL_MODEID2           0x2
#define LOCAL_FLAGID0           0x0
#define LOCAL_FLAGID1           0x1
#define LOCAL_FLAGID2           0x2
#define LOCAL_FLAGID3           0x3
#define LOCAL_FLAGID4           0x4
#define LOCAL_FLAGID5           0x5
#define LOCAL_FLAGID6           0x6
#define LOCAL_FLAGID7           0x7

#define HardWareSyncAmount      8388608     // (2048 * 4096)
#define GmmWorkSpaceAmount      524288      // (256 * 2048)
#define SoftwareWorkSpaceEle    (64)
#define L0C_FORMAT_SIZE         4
#define DEFAULT_BASEK           1088


namespace GROUPED_MATMUL
{
using namespace AscendC;
using namespace matmul;

namespace GMMHighPerf
{
__aicore__ inline uint32_t ceilINT(uint32_t x, uint32_t y)
{
    return Ceil(x, y);
}

//8的倍数向上取整
__aicore__ inline uint32_t ceilINT_8(uint32_t num)
{
    if (num % FACTOR_8 == 0) {
        return num;
    } else {
        return (num / FACTOR_8) * FACTOR_8 + FACTOR_8;
    }
}

//16的倍数向上取整
__aicore__ inline uint32_t ceilINT_16(uint32_t num)
{
    if (num % FACTOR_16 == 0) {
        return num;
    } else {
        return (num / FACTOR_16) * FACTOR_16 + FACTOR_16;
    }
}

__aicore__ inline uint32_t ceilINT_64(uint32_t num)
{
    if (num % FACTOR_64 == 0) {
        return num;
    } else {
        return (num / FACTOR_64) * FACTOR_64 + FACTOR_64;
    }
}

struct Dim0 {
public:
    __aicore__ inline Dim0()
    {
        dim = 0;
    }

    __aicore__ inline Dim0(uint32_t in)
    {
        dim = in;
    }

    uint32_t dim;

private:
    uint32_t ele_size;
};

struct Dim1 : public Dim0 {
public:
    __aicore__ inline Dim1() : Dim0(DIM_VAL1)
    {
        s[0] = 0;
    }

    __aicore__ inline Dim1(uint32_t in) : Dim0(DIM_VAL1)
    {
        s[0] = in;
    }

    __aicore__ inline Dim1(uint32_t in, char name) : Dim0(DIM_VAL1)
    {
        s[0] = in;
        axis_name[0] = name;
    }

    __aicore__ inline Dim1(const Dim1 &d): Dim0(DIM_VAL1)
    {
        s[0] = d.s[0];
        axis_name[0] = d.axis_name[0];
    }

    __aicore__ inline uint32_t size()
    {
        return s[0];
    }

     __aicore__ inline uint32_t size_16()
    {
        return ceilINT_16(s[0]);
    }

    __aicore__ inline Dim1 Dim_16()
    {
        return Dim1(ceilINT_16(s[0]));
    }

    __aicore__ inline void clear()
    {
        s[0] = 0;
    }

    __aicore__ inline void setAxisName(char n0='0')
    {
        axis_name[0] = n0;
    }

    __aicore__ inline void copyAxisName(Dim1 &dim)
    {
        axis_name[0] = dim.axis_name[0];
    }

    __aicore__ inline void setByAxisName(uint32_t in, char name)
    {
        if (axis_name[0] == name) {
            s[0] = in;
        } else {
        }
    }

    __aicore__ inline void set(uint32_t in)
    {
        s[0] = in;
    }

    __aicore__ inline uint32_t getAxisByName(char name) const
    {
        if (axis_name[0] == name) {
            return s[0];
        }
        return -1;
    }

    __aicore__ inline bool operator == (Dim1 &in)
    {
        return s[0] == in.s[0];
    }

    __aicore__ inline void posIterator(Dim1 ori_pos, Dim1 base_tiling, Dim1 ori_vec, Dim1 &curr_vec)
    {
        uint32_t temp0 = s[0] + base_tiling.s[0];
        if (temp0 < ori_pos.s[0] + ori_vec.s[0]) {
            s[0] = temp0;
        } else {
            curr_vec.s[0] = ori_pos.s[0] + ori_vec.s[0] - s[0];
            s[0] = ori_pos.s[0];
        }
    }

    uint32_t s[DIM_VAL1];
    char axis_name[DIM_VAL1];
};

struct Dim2 {
public:
    uint32_t dim;
    __aicore__ inline Dim2()
    {
        s[0] = 0;
        s[1] = 0;
    }

    __aicore__ inline Dim2(uint32_t in0, uint32_t in1)
    {
        s[0] = in0;
        s[1] = in1;
    }

    __aicore__ inline Dim2(uint32_t in0, uint32_t in1, char name0, char name1)
    {
        s[0] = in0;
        s[1] = in1;
        axis_name[0] = name0;
        axis_name[1] = name1;
    }

    __aicore__ inline Dim2(const Dim2 &d)
    {
        s[0] = d.s[0];
        s[1] = d.s[1];
        axis_name[0] = d.axis_name[0];
        axis_name[1] = d.axis_name[1];
    }

    __aicore__ inline uint32_t size()
    {
        return s[0] * s[1];
    }

    __aicore__ inline uint32_t size_16()
    {
        return ceilINT_16(s[0]) * ceilINT_16(s[1]);
    }

    __aicore__ inline Dim2 Dim_16()
    {
        return Dim2(ceilINT_16(s[0]), ceilINT_16(s[1]));
    }

    __aicore__ inline void clear()
    {
        s[0] = 0;
        s[1] = 0;
    }

    __aicore__ inline void set(uint32_t in0, uint32_t in1)
    {
        s[0] = in0;
        s[1] = in1;
    }

    __aicore__ inline void transpose()
    {
        uint32_t temp = s[0];
        s[0] = s[1];
        s[1] = temp;

        char temp_name = axis_name[0];
        axis_name[0] = axis_name[1];
        axis_name[1] = temp_name;
    }

    __aicore__ inline void setAxisName(char n0='0', char n1='1')
    {
        axis_name[0] = n0;
        axis_name[1] = n1;
    }

    __aicore__ inline void copyAxisName(Dim2 &dim)
    {
        axis_name[0] = dim.axis_name[0];
        axis_name[1] = dim.axis_name[1];
    }

    __aicore__ inline void setByAxisName(uint32_t in, char name)
    {
        if (axis_name[0] == name) {
            s[0] = in;
        } else if (axis_name[1] == name) {
            s[1] = in;
        } else {
        }
    }

    __aicore__ inline uint32_t getAxisByName(char name)
    {
        if (axis_name[0] == name) {
            return s[0];
        } else if (axis_name[1] == name) {
            return s[1];
        }
        return -1;
    }

    __aicore__ inline bool operator == (Dim2 &in)
    {
        return s[0] == in.s[0] && s[1] == in.s[1];
    }

    __aicore__ inline void posIterator(Dim2 ori_pos, Dim2 base_tiling, Dim2 ori_vec, Dim2 &curr_vec)
    {
        uint32_t temp0 = s[0] + base_tiling.s[0];
        if (temp0 < ori_pos.s[0] + ori_vec.s[0]) {
            s[0] = temp0;
        } else {
            curr_vec.s[0] = ori_pos.s[0] + ori_vec.s[0] - s[0];
            s[0] = ori_pos.s[0];

            uint32_t temp1 = s[1] + base_tiling.s[1];
            if (temp1 < ori_pos.s[1] + ori_vec.s[1]) {
                s[1] = temp1;
            } else {
                curr_vec.s[1] = ori_pos.s[1] + ori_vec.s[1] - s[1];
                s[1] = ori_pos.s[1];
            }
        }
    }

    uint32_t s[DIM_VAL2];
    char axis_name[DIM_VAL2];
};

struct Dim3 : public Dim0 {
public:
    __aicore__ inline Dim3() : Dim0(DIM_VAL3)
    {
        s[ARRAY_IDX0] = 0;
        s[ARRAY_IDX1] = 0;
        s[ARRAY_IDX2] = 0;
    }

    __aicore__ inline Dim3(uint32_t in0, uint32_t in1, uint32_t in2) : Dim0(DIM_VAL3)
    {
        s[ARRAY_IDX0] = in0;
        s[ARRAY_IDX1] = in1;
        s[ARRAY_IDX2] = in2;
    }

    __aicore__ inline Dim3(uint32_t in0, uint32_t in1, uint32_t in2, char name0, char name1, char name2) : Dim0(DIM_VAL3)
    {
        s[ARRAY_IDX0] = in0;
        s[ARRAY_IDX1] = in1;
        s[ARRAY_IDX2] = in2;
        axis_name[ARRAY_IDX0] = name0;
        axis_name[ARRAY_IDX1] = name1;
        axis_name[ARRAY_IDX2] = name2;
    }

    __aicore__ inline Dim3(const Dim3 &d): Dim0(DIM_VAL3)
    {
        s[ARRAY_IDX0] = d.s[ARRAY_IDX0];
        s[ARRAY_IDX1] = d.s[ARRAY_IDX1];
        s[ARRAY_IDX2] = d.s[ARRAY_IDX2];
        axis_name[ARRAY_IDX0] = d.axis_name[ARRAY_IDX0];
        axis_name[ARRAY_IDX1] = d.axis_name[ARRAY_IDX1];
        axis_name[ARRAY_IDX2] = d.axis_name[ARRAY_IDX2];
    }

    __aicore__ inline uint32_t size()
    {
        return s[ARRAY_IDX0] * s[ARRAY_IDX1] * s[ARRAY_IDX2];
    }

    __aicore__ inline uint32_t size_16()
    {
        return ceilINT_16(s[ARRAY_IDX0]) * ceilINT_16(s[ARRAY_IDX1]) * ceilINT_16(s[ARRAY_IDX2]);
    }

    __aicore__ inline Dim3 Dim_16()
    {
        return Dim3(ceilINT_16(s[ARRAY_IDX0]), ceilINT_16(s[ARRAY_IDX1]), ceilINT_16(s[ARRAY_IDX2]));
    }

    __aicore__ inline void clear()
    {
        s[ARRAY_IDX0] = 0;
        s[ARRAY_IDX1] = 0;
        s[ARRAY_IDX2] = 0;
    }

    __aicore__ inline void set(uint32_t in0, uint32_t in1, uint32_t in2)
    {
        s[ARRAY_IDX0] = in0;
        s[ARRAY_IDX1] = in1;
        s[ARRAY_IDX2] = in2;
    }

    __aicore__ inline void setAxisName(char n0='0', char n1='1', char n2='2')
    {
        axis_name[ARRAY_IDX0] = n0;
        axis_name[ARRAY_IDX1] = n1;
        axis_name[ARRAY_IDX2] = n2;
    }

    __aicore__ inline void copyAxisName(Dim3 &dim)
    {
        axis_name[ARRAY_IDX0] = dim.axis_name[ARRAY_IDX0];
        axis_name[ARRAY_IDX1] = dim.axis_name[ARRAY_IDX1];
        axis_name[ARRAY_IDX2] = dim.axis_name[ARRAY_IDX2];
    }

    __aicore__ inline void setByAxisName(uint32_t in, char name)
    {
        if (axis_name[ARRAY_IDX0] == name) {
            s[ARRAY_IDX0] = in;
        } else if (axis_name[ARRAY_IDX1] == name) {
            s[ARRAY_IDX1] = in;
        } else if (axis_name[ARRAY_IDX2] == name) {
            s[ARRAY_IDX2] = in;
        } else {
        }
    }

    __aicore__ inline uint32_t getAxisByName(char name)
    {
        if (axis_name[ARRAY_IDX0] == name) {
            return s[ARRAY_IDX0];
        } else if (axis_name[ARRAY_IDX1] == name) {
            return s[ARRAY_IDX1];
        } else if (axis_name[ARRAY_IDX2] == name) {
            return s[ARRAY_IDX2];
        }
        return -1;
    }

    __aicore__ inline bool operator == (Dim3 &in)
    {
        return s[ARRAY_IDX0] == in.s[ARRAY_IDX0] && s[ARRAY_IDX1] == in.s[ARRAY_IDX1] && s[ARRAY_IDX2] == in.s[ARRAY_IDX2];
    }

    __aicore__ inline void posIterator(Dim3 ori_pos, Dim3 base_tiling, Dim3 ori_vec, Dim3 &curr_vec)
    {
        //记录下初始的坐标
        uint32_t t_pos[DIM_VAL3];
        t_pos[ARRAY_IDX0] = s[ARRAY_IDX0];
        t_pos[ARRAY_IDX1] = s[ARRAY_IDX1];
        t_pos[ARRAY_IDX2] = s[ARRAY_IDX2];
        uint32_t temp0 = s[ARRAY_IDX0] + base_tiling.s[ARRAY_IDX0];
        if (temp0 < ori_pos.s[ARRAY_IDX0] + ori_vec.s[ARRAY_IDX0]) {
            s[ARRAY_IDX0] = temp0;
        } else {
            s[ARRAY_IDX0] = ori_pos.s[ARRAY_IDX0];
            curr_vec.s[ARRAY_IDX0] = base_tiling.s[ARRAY_IDX0] - (temp0 - (ori_pos.s[ARRAY_IDX0] + ori_vec.s[ARRAY_IDX0]));

            uint32_t temp1 = s[ARRAY_IDX1] + base_tiling.s[ARRAY_IDX1];
            if (temp1 < ori_pos.s[ARRAY_IDX1] + ori_vec.s[ARRAY_IDX1]) {
                s[ARRAY_IDX1] = temp1;
            } else {
                s[ARRAY_IDX1] = ori_pos.s[ARRAY_IDX1];
                curr_vec.s[ARRAY_IDX1] = base_tiling.s[ARRAY_IDX1] - (temp1 - (ori_pos.s[ARRAY_IDX1] + ori_vec.s[ARRAY_IDX1]));

                uint32_t temp2 = s[ARRAY_IDX2] + base_tiling.s[ARRAY_IDX2];
                if (temp2 < ori_pos.s[ARRAY_IDX2] + ori_vec.s[ARRAY_IDX2]) {
                    s[ARRAY_IDX2] = temp2;
                } else {
                    s[ARRAY_IDX2] = ori_pos.s[ARRAY_IDX2];
                    curr_vec.s[ARRAY_IDX2] = base_tiling.s[ARRAY_IDX2] - (temp2 - (ori_pos.s[ARRAY_IDX2] + ori_vec.s[ARRAY_IDX2]));
                }
            }
        }

        uint32_t temp3 = t_pos[ARRAY_IDX1] + base_tiling.s[ARRAY_IDX1];
        if (temp3 > ori_pos.s[ARRAY_IDX1] + ori_vec.s[ARRAY_IDX1]) {
                curr_vec.s[ARRAY_IDX1] = base_tiling.s[ARRAY_IDX1] - (temp3 - (ori_pos.s[ARRAY_IDX1] + ori_vec.s[ARRAY_IDX1]));
        }

        uint32_t temp4 = t_pos[ARRAY_IDX2] + base_tiling.s[ARRAY_IDX2];
        if (temp4 > ori_pos.s[ARRAY_IDX2] + ori_vec.s[ARRAY_IDX2]) {
                curr_vec.s[ARRAY_IDX2] = base_tiling.s[ARRAY_IDX2] - (temp4 - (ori_pos.s[ARRAY_IDX2] + ori_vec.s[ARRAY_IDX2]));
        }
    }

    uint32_t s[DIM_VAL3];
    char axis_name[DIM_VAL3];
};

struct AxisTiling3Vector {
    uint32_t s[DIM_VAL3];
};

template <class Dim>
class UnisorShape {
public:
    __aicore__ inline  UnisorShape()
    {
        Dim dim;
        vector_val = dim;
    }

    __aicore__ inline  UnisorShape(Dim vec)
    {
        vector_val = vec;
    }

    __aicore__ inline  unsigned size()
    {
        return vector_val.size();
    }

    Dim vector_val;// vector value to ensure the space of UnisorShape
};

template <class Dim>
class Cartesian {
public:
    __aicore__ inline Cartesian()
    {
        Dim dim;
        pos = dim;
        vec = dim;
    }

    __aicore__ inline Cartesian(const Cartesian &c)
    {
        pos = c.pos;
        vec = c.vec;
    }

    __aicore__ inline Cartesian(Dim v)
    {
        vec = v;
        pos = v;
        pos.clear();
    }

    __aicore__ inline Cartesian(Dim p, Dim v)
    {
        pos = p;
        vec = v;
    }

    __aicore__ inline unsigned size()
    {
        return vec.size();
    }

    Dim pos;//start position
    Dim vec;// vector value to ensure the space of UnisorShape
};

template <class Dim>
class CartesianIterator
{
public:
    __aicore__ inline CartesianIterator(Cartesian<Dim> cartesianC, Dim tiling)
    {
        iter_tiling = tiling;
        ori_pos = cartesianC.pos;
        ori_vec = cartesianC.vec;
        offset_pos = cartesianC.pos;
        active = false;
        odd_even = false;
    }

    __aicore__ inline bool posIterator(Cartesian<Dim> &cartesianUnisorC)
    {
        odd_even = !odd_even;
        cartesianUnisorC.pos = offset_pos;
        Dim curr_vec = iter_tiling;
        offset_pos.posIterator(ori_pos, iter_tiling, ori_vec, curr_vec);
        cartesianUnisorC.vec = curr_vec;
        if (cartesianUnisorC.pos == ori_pos && active) {
            //用完了重置
            active = false;
            offset_pos = ori_pos;
            return false;
        }
        active = true;
        return true;
    }

    __aicore__ inline bool isLast()
    {
        if (offset_pos == ori_pos) {
            return true;
        } else {
            return false;
        }
    }

    __aicore__ inline bool oddEven()
    {
        return odd_even;
    }

    Dim iter_tiling;
    Dim ori_pos;
    Dim ori_vec;
    Dim offset_pos;
    bool active;
    bool odd_even;
};

/*
 * Uniform tensor for both global and local
 * Unisor必须是一块联系的地址片段，但是shape可以指定Unisor的某一个Tiling地址空间
 */
template <TPosition MemType, class Dim>
class Unisor : public UnisorShape <Dim>, public Cartesian <Dim>
{
public:
    float format_size;
    GlobalTensor<uint8_t> inputGlobal;
    TBuf<MemType> calcBuf;
    GM_ADDR gm_addr = nullptr;
    int64_t gm_datasize = 0;

    __aicore__ inline Unisor() : UnisorShape <Dim> (), Cartesian <Dim> ()
    {
    }

    __aicore__ inline Unisor(const Unisor &t):UnisorShape <Dim> (t.vector_val), Cartesian <Dim> (t.pos, t.vec), format_size(t.format_size), inputGlobal(t.inputGlobal), calcBuf(t.calcBuf)
    {
    }

    //LocalTensor
    __aicore__ inline Unisor(Dim vec, float format_size) : UnisorShape <Dim> (vec), Cartesian <Dim> (vec)
    {
        // LocalTensor要通过init()函数初始化
        this->format_size = format_size;
    }

    //GlobalTensor
    __aicore__ inline Unisor(GM_ADDR gm, Dim vec, float format_size) : UnisorShape <Dim> (vec.Dim_16()), Cartesian <Dim> (vec.Dim_16())
    {
        this->format_size = format_size;
        int64_t dataSize  = format_size * vec.size_16();
        this->gm_addr = gm;
        this->gm_datasize = dataSize;
        inputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(gm), dataSize);
    }

    __aicore__ inline Unisor(GM_ADDR gm, Dim vec, float format_size, bool isReal) : UnisorShape <Dim> (vec), Cartesian <Dim> (vec)
    {
        this->format_size = format_size;
        int64_t dataSize  = format_size * vec.size();
        this->gm_addr = gm;
        this->gm_datasize = dataSize;
        inputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(gm), dataSize);
    }

    __aicore__ inline void setCartesian(Dim p, Dim v)
    {
        this->pos = p;
        this->vec = v;
    }

    __aicore__ inline void setCartesian(Cartesian <Dim> cartesian)
    {
        this->pos = cartesian.pos;
        this->vec = cartesian.vec;
    }

    __aicore__ inline void setCartesianVec(Dim v)
    {
        this->vec = v;
    }

    __aicore__ inline void transpose()
    {
        this->pos.transpose();
        this->vec.transpose();
    }

    __aicore__ inline  void init(Dim vec, float format_size, TPipe *pipe)
    {
        // Shape和Cartesian初始化
        this->vector_val = vec.Dim_16();
        this->vector_val.copyAxisName(vec);
        this->vec = vec;
        this->pos.copyAxisName(vec);
        this->format_size = format_size;

        int64_t size = format_size * vec.size_16();
        if (size > 0) {
            pipe->InitBuffer(calcBuf, size);
        }
    }

    __aicore__ inline  void init_RealShape(Dim vec, float format_size, TPipe *pipe)
    {
        //Shape和Cartesian初始化
        this->vector_val = vec;
        this->vector_val.copyAxisName(vec);
        this->vec = vec;
        this->pos.copyAxisName(vec);
        this->format_size = format_size;

        int64_t size = format_size * vec.size();
        if (size > 0) {
            pipe->InitBuffer(calcBuf, size);
        }
    }

    template <typename T>
    __aicore__ inline LocalTensor<T> get()
    {
        return calcBuf.template Get<T>();
    }

    __aicore__ inline GlobalTensor<uint8_t> getGM()
    {
        return inputGlobal;
    }

    __aicore__ inline GlobalTensor<int32_t> getGM_32()
    {
        GlobalTensor<int32_t> tmp32tGlobalTensor;
        tmp32tGlobalTensor.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t *>(this->gm_addr), this->gm_datasize / FACTOR_4);
        return tmp32tGlobalTensor;
    }
};

class KernelBase {
public:
    TPipe *pipe;
    Dim3 splitDim;
    Dim3 block;
    uint32_t blockId;
    uint32_t pattern = 1;
    uint8_t output_type = 0;
    bool offset_enable = false;
    uint32_t nCoreAIC;
    uint32_t nCoreAIV;
    uint32_t szMemL0AB;
    uint32_t szMemL0C;
    uint32_t szMemUB;

public:
    __aicore__ inline void Init(TPipe* pipeIn)
    {
        pipe = pipeIn;
    }

    template<typename T> __aicore__ inline void NPU_Duplicate(Unisor<UBMem, Dim1> &out, const T value)
    {
        LocalTensor<uint16_t> srcLocal = out.get<uint16_t>();
        uint32_t dstShape = out.vec.s[0] / FACTOR_2;
        Duplicate(srcLocal, value, dstShape);
        PipeBarrier<PIPE_V>();
    }

    template<typename T> __aicore__ inline void NPU_Adds(Unisor<UBMem, Dim2> &out, Unisor<UBMem, Dim2> &in, const T &scalar_Value)
    {
        LocalTensor<half> outLocal = out.get<half>();
        LocalTensor<half> inLocal = in.get<half>();
        uint32_t width = in.vector_val.s[1];
        uint32_t height = in.vector_val.s[0];

        Dim2 vecOUT = out.vec;
        uint32_t width_cal = vecOUT.s[1];

        Dim2 posOUT = out.pos;
        uint32_t posOUTHeight = posOUT.s[0];
        uint32_t posOUTWidth = posOUT.s[1];

        Dim2 posA = in.pos;
        uint32_t posAHeight = posA.s[0];
        uint32_t posAWidth = posA.s[1];

        Adds(outLocal, inLocal, scalar_Value, width_cal * height);
        PipeBarrier<PIPE_V>();
    }

    template<typename T> __aicore__ inline void NPU_Muls(Unisor<UBMem, Dim2> &out, Unisor<UBMem, Dim2> &in, const T &scalar_Value)
    {
        LocalTensor<half> outLocal = out.get<half>();
        LocalTensor<half> inLocal = in.get<half>();

        uint32_t width = in.vector_val.s[1];  // in的整个R轴长度
        uint32_t height = in.vector_val.s[0]; // in的整个D轴长度

        Dim2 posA = in.pos;
        uint32_t posAHeight = posA.s[0];
        uint32_t posAWidth = posA.s[1];

        Dim2 posOUT = out.pos;
        uint32_t posOUTHeight = posOUT.s[0];
        uint32_t posOUTWidth = posOUT.s[1];

        Dim2 vecOUT = out.vec;
        uint32_t width_cal = vecOUT.s[1];  // 计算窗口 R 轴长度

        Muls(outLocal, inLocal, scalar_Value, width_cal * height);
        PipeBarrier<PIPE_V>();
    }

    template<typename T> __aicore__ inline void NPU_Muls_int32_t(Unisor<UBMem, Dim2> &out, Unisor<UBMem, Dim2> &in, const T &scalar_Value)
    {
        Dim2 vecOUT = out.vec;
        uint32_t width_cal = vecOUT.s[1];  // 计算窗口 R 轴长度

        Dim2 posOUT = out.pos;
        uint32_t posOUTHeight = posOUT.s[0];
        uint32_t posOUTWidth = posOUT.s[1];

        Dim2 posA = in.pos;
        uint32_t posAHeight = posA.s[0];
        uint32_t posAWidth = posA.s[1];

        LocalTensor<int32_t> outLocal = out.get<int32_t>();
        LocalTensor<int32_t> inLocal = in.get<int32_t>();

        uint32_t width = in.vector_val.s[1];  // in的整个R轴长度
        uint32_t height = in.vector_val.s[0]; // in的整个D轴长度

        Muls(outLocal, inLocal, scalar_Value, width_cal * height);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void NPU_Broadcast(Unisor<UBMem, Dim2> &out, Unisor<UBMem, Dim1> &in)
    {
        Dim2 vec = out.vec;
        if (vec.s[0] == 0 || vec.s[1] == 0) {
            return;
        }
        LocalTensor<float> outLocal = out.get<float>();
        LocalTensor<float> inLocal = in.get<float>();

        uint32_t width = vec.s[1];
        uint32_t height = vec.s[0];

        uint32_t dstShape[ARRAY_IDX2] = {0, 0};
        uint32_t srcShape[ARRAY_IDX2] = {0, 0};
        dstShape[ARRAY_IDX0] = height;
        dstShape[ARRAY_IDX1] = width;
        srcShape[ARRAY_IDX0] = height;
        srcShape[ARRAY_IDX1] = 1;
        AscendC::BroadCast<float, DIM_VAL2, 1>(outLocal, inLocal, dstShape, srcShape);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void NPU_SetZero(Unisor<UBMem, Dim1> &in)
    {
        LocalTensor<float> srcLocal = in.get<float>();
        uint32_t dstShape = in.vec.s[0];

        Duplicate<float>(srcLocal, 0, dstShape);
        PipeBarrier<PIPE_V>();
    }

    // 向量乘：需要A的输入是dim1的
    __aicore__ inline void NPU_VecMul(Unisor<UBMem, Dim2> &out, Unisor<UBMem, Dim2> &unisorA, Unisor<UBMem, Dim2> &unisorB)
    {
        Dim2 vec = out.vec;
        if (vec.s[0] == 0 || vec.s[1] == 0) {
            return;
        }

        Dim2 vecB = unisorB.vec;
        LocalTensor<float> outLocal = out.get<float>();
        LocalTensor<float> aLocal = unisorA.get<float>();
        LocalTensor<float> bLocal = unisorB.get<float>();

        uint32_t width = vec.s[1];
        uint32_t height = vec.s[0];

        AscendC::Mul(outLocal, bLocal, aLocal, width * height);
        PipeBarrier<PIPE_V>();
    }

    // 向量乘：需要A的输入是[1:]的
    __aicore__ inline void NPU_VecMuls(Unisor<UBMem, Dim2> &out, Unisor<UBMem, Dim2> &unisorA, Unisor<UBMem, Dim2> &unisorB)
    {
        Dim2 vec = out.vec;
        if (vec.s[0] == 0 || vec.s[1] == 0) {
            return;
        }
        Dim2 vecA = unisorA.vec;
        if (vecA.s[0] != 1) return;

        LocalTensor<float> outLocal = out.get<float>();
        LocalTensor<float> aLocal = unisorA.get<float>();
        LocalTensor<float> bLocal = unisorB.get<float>();

        uint32_t width = vec.s[1];
        uint32_t height = vec.s[0];
        uint32_t ele = MUL_BYTES / FP32;
        uint64_t mask = ele;
        // repeatTimes = 4, 一次迭代计算128个数, 共计算512个数
        // dstBlkStride, src0BlkStride, src1BlkStride = 1, 单次迭代内数据连续读取和写入
        // dstRepStride, src0RepStride, src1RepStride = 8, 相邻迭代间数据连续读取和写入
        uint8_t dstRepStride = ceilINT(width, FACTOR_8);
        uint8_t src0RepStride = ceilINT(width, FACTOR_8);
        uint16_t repeat = ceilINT(width, ele);
        for (int i = 0; i < repeat; i++)
        {
            uint32_t dstOffset = i * ele;
            uint32_t srcOffset = i * ele;
            if (i == repeat - 1)
            {
                mask = width - i * ele;
            }
            AscendC::Mul(outLocal[dstOffset], bLocal[dstOffset], aLocal[srcOffset], mask, height, { 1, 1, 1, dstRepStride, src0RepStride, 0});
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ void NPU_ReduceSum(Unisor<UBMem, Dim1> &out, Unisor<UBMem, Dim2> &in, Unisor<UBMem, Dim1> &workUnisor)
    {
        LocalTensor<float> srcLocal = in.get<float>();
        LocalTensor<float> dstLocal = out.get<float>();

        Dim2 posUB = in.pos;
        uint32_t posUBHeight = posUB.s[0];
        uint32_t posUBWidth = posUB.s[1];

        Dim2 vecUB = in.vec;

        uint32_t width = in.vec.s[1];  // k 轴长度
        uint32_t height = vecUB.s[0]; // m 轴长度

        LocalTensor<float> workLocal = workUnisor.get<float>();

        for (uint32_t i = 0; i < height; i++) {
            uint32_t srcOffset = i * width + posUBWidth;
            ReduceSum(dstLocal[i], srcLocal[srcOffset], workLocal, vecUB.s[1]);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void NPU_Add_Bias(Unisor<UBMem, Dim2> &out, Unisor<UBMem, Dim2> &bias)
    {
        Dim2 vec = out.vec;
        if (vec.s[0] == 0 || vec.s[1] == 0) {
            return;
        }
        LocalTensor<float> outLocal = out.get<float>();
        LocalTensor<float> biasLocal = bias.get<float>();

        uint32_t width = vec.s[1];
        uint32_t height = vec.s[0];

        uint32_t dstOffset = 0;
        uint32_t ele = MUL_BYTES / FP32;
        uint64_t mask = ele;
        // repeatTimes = 4, 一次迭代计算128个数, 共计算512个数
        // dstBlkStride, src0BlkStride, src1BlkStride = 1, 单次迭代内数据连续读取和写入
        // dstRepStride, src0RepStride, src1RepStride = 8, 相邻迭代间数据连续读取和写入
        uint16_t repeat = ceilINT(width, ele);
        uint8_t src0RepStride = ceilINT(width, FACTOR_8);
        uint8_t dstRepStride = ceilINT(width, FACTOR_8);
        for (int i = 0; i < repeat; i++)
        {
            if (i == repeat - 1) {
                mask = width - i * ele;
            }
            uint32_t dstOffset = i * ele;
            uint32_t srcOffset = i * ele;
            AscendC::Add(outLocal[dstOffset], outLocal[dstOffset], biasLocal[srcOffset], mask, height, { 1, 1, 1, dstRepStride, src0RepStride, 0 });
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void NPU_Load(Unisor<UBMem, Dim2> &out, Unisor<GlobalMem, Dim2> &in)
    {
        Dim2 vec = in.vec;
        if (vec.s[0] == 0 || vec.s[1] == 0) {
            return;
        }
        GlobalTensor<uint8_t> srcLocal = in.getGM();
        LocalTensor<uint8_t> dstLocal = out.get<uint8_t>();
        Dim2 pos = in.pos;

        float format_size = in.format_size;
        int ele_num = COPY_BLK_BYTES;
        int32_t width = vec.s[1] * format_size;
        uint32_t widthBlock = ceilINT(width, ele_num);
        int32_t in_width = in.vector_val.s[1] * format_size;
        int32_t posWidth = pos.s[1] * format_size;

        uint32_t height = vec.s[0];
        uint32_t in_height = in.vector_val.s[0];
        uint32_t posHeight = pos.s[0];

        //ND->ND
        uint16_t dstStride = 0;
        uint16_t srcStride = ceilINT(in_width, ele_num) - widthBlock;
        uint16_t blockCount = height;
        uint16_t blockLen = widthBlock;
        int64_t srcOffset = posHeight * in_width + posWidth;
        int64_t dstOffset = 0;
        DataCopy(dstLocal[dstOffset], srcLocal[srcOffset], { blockCount, blockLen, srcStride, dstStride});
    }

    __aicore__ void NPU_Store(Unisor<GlobalMem, Dim2> &out, Unisor<UBMem, Dim2> &in)
    {
        Dim2 vec = out.vec;
        if (vec.s[0] == 0 || vec.s[1] == 0) {
            return;
        }
        LocalTensor<uint8_t> src = in.get<uint8_t>();
        GlobalTensor<uint8_t> dst = out.getGM();
        Dim2 pos = out.pos;

        float format_size = out.format_size;
        int ele_num = COPY_BLK_BYTES;
        int32_t in_width = out.vector_val.s[1] * format_size;
        uint32_t in_height = out.vector_val.s[0];
        int32_t width = vec.s[1] * format_size;
        uint32_t height = vec.s[0];
        int32_t posWidth = pos.s[1] * format_size;
        uint32_t posHeight = pos.s[0];
        uint32_t nBlocks = ceilINT(width, ele_num);
        uint32_t mBlocks = ceilINT(height, ele_num);
        uint32_t t_nBlocks = ceilINT(in_width, ele_num);

        //ND->ND
        uint16_t blockCount = height;
        uint16_t blockLen = nBlocks;
        uint16_t srcStride = 0;
        uint16_t dstStride = t_nBlocks - blockLen + t_nBlocks;
        int64_t srcOffset = 0;
        int64_t dstOffset = posHeight * in_width + posWidth;

        DataCopy(dst[dstOffset], src[srcOffset], { blockCount, blockLen, srcStride, dstStride});
    }

    __aicore__ void NPU_And(Unisor<UBMem, Dim2> &out, Unisor<UBMem, Dim2> &a, Unisor<UBMem, Dim1> &b)
    {
        // 获取 UB 中的 Tensor
        LocalTensor<uint16_t> outLocal = out.get<uint16_t>();
        LocalTensor<uint16_t> aLocal = a.get<uint16_t>();
        LocalTensor<uint16_t> bLocal = b.get<uint16_t>();
        uint32_t width = a.vector_val.s[1] / FACTOR_2;  // a的整个R轴长度
        uint32_t height = a.vector_val.s[0]; // a的整个D轴长度

        Dim2 posA = a.pos;
        uint32_t posAHeight = posA.s[0];
        uint32_t posAWidth = posA.s[1] / FACTOR_2;

        Dim2 posOUT = out.pos;
        uint32_t posOUTHeight = posOUT.s[0];
        uint32_t posOUTWidth = posOUT.s[1] / FACTOR_2;

        Dim2 vecOUT = out.vec;
        uint32_t width_cal = vecOUT.s[1] / FACTOR_2;  // 计算窗口 R 轴长度

        for (int i = 0 ; i < height; i++)
        {
            uint32_t srcOffset = (posAHeight + i) * width + posAWidth;
            uint32_t dstOffset = (posOUTHeight + i) * width_cal + posOUTWidth;
            And(outLocal[dstOffset], aLocal[srcOffset], bLocal, width_cal);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void NPU_Add_int32_t(Unisor<UBMem, Dim2> &out, Unisor<UBMem, Dim2> &a, Unisor<UBMem, Dim2> &b)
    {
        LocalTensor<int32_t> outLocal = out.get<int32_t>();
        LocalTensor<int32_t> aLocal = a.get<int32_t>();
        LocalTensor<int32_t> bLocal = b.get<int32_t>();

        outLocal = aLocal + bLocal;
        PipeBarrier<PIPE_V>();
    }

    // GM到UB
    __aicore__ inline void NPU_Load_GMToUB(Unisor<UBMem, Dim2> &out, Unisor<GlobalMem, Dim2> &in)
    {
        Dim2 vec = in.vec;
        if (vec.s[0] == 0 || vec.s[1] == 0) {
            return;
        }
        GlobalTensor<uint8_t> srcLocal = in.getGM();
        LocalTensor<uint8_t> dstLocal = out.get<uint8_t>();
        Dim2 pos = in.pos;

        float format_size = in.format_size;
        int64_t ele_num = COPY_BLK_BYTES;
        int64_t in_width = in.vector_val.s[1] * format_size;
        uint64_t in_height = in.vector_val.s[0];
        int64_t width = vec.s[1] * format_size;
        uint64_t height = vec.s[0];
        uint64_t posHeight = pos.s[0];
        int64_t posWidth = pos.s[1] * format_size;
        uint64_t widthBlock = ceilINT(width, ele_num);

        //ND->ND
        uint16_t blockCount = height;
        uint16_t blockLen = widthBlock;
        uint16_t dstStride = 0;
        uint16_t srcStride = in_width / ele_num - widthBlock;
        int64_t srcOffset = posHeight * in_width + posWidth;
        int64_t dstOffset = 0;
        DataCopy(dstLocal[dstOffset], srcLocal[srcOffset], { blockCount, blockLen, srcStride, dstStride});
    }

    // 带起始地址间隔搬数
    __aicore__ inline void NPU_Load_GMToUB_ByStride_ByOffset(Unisor<UBMem, Dim2> &out, Unisor<GlobalMem, Dim2> &in, uint32_t offset)
    {
        Dim2 vec = in.vec;
        if (vec.s[0] == 0 || vec.s[1] == 0) {
            return;
        }
        GlobalTensor<uint8_t> srcLocal = in.getGM();
        LocalTensor<uint8_t> dstLocal = out.get<uint8_t>();
        Dim2 pos = in.pos;

        float format_size = in.format_size;
        int64_t ele_num = COPY_BLK_BYTES;
        int64_t in_width = in.vector_val.s[1] * format_size;
        int64_t width = vec.s[1] * format_size;
        uint64_t widthBlock = ceilINT(width, ele_num);
        int64_t posWidth = pos.s[1] * format_size;
        uint64_t in_height = in.vector_val.s[0];
        uint64_t height = vec.s[0];
        uint64_t posHeight = pos.s[0];

        //ND->ND
        int64_t srcOffset = offset + posHeight * vec.s[1] * format_size;
        int64_t dstOffset = 0;
        uint16_t blockCount = height;
        uint16_t blockLen = widthBlock;
        uint16_t dstStride = 0;
        uint16_t srcStride = widthBlock;
        DataCopy(dstLocal[dstOffset], srcLocal[srcOffset], { blockCount, blockLen, srcStride, dstStride});
    }

    // UB到UB
    template<class Out, class In> __aicore__ static void NPU_Cast(Unisor<UBMem, Dim2> &out, Unisor<UBMem, Dim2> &in, const AscendC::RoundMode round_mode = AscendC::RoundMode::CAST_NONE)
    {
        LocalTensor<In> srcLocal = in.get<In>();
        LocalTensor<Out> dstLocal = out.get<Out>();
        Dim2 vec = in.vec;
        if (vec.s[0] == 0 || vec.s[1] == 0) return;
        uint32_t width = vec.s[1];
        uint32_t height = vec.s[0];
        uint32_t srcOffset = 0;
        uint32_t dstOffset = 0;
        AscendC::Cast(dstLocal, srcLocal, round_mode, width * height);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void NPU_Cast_FP32ToBF16(Unisor<UBMem, Dim2> &out, Unisor<UBMem, Dim2> &in)
    {
        NPU_Cast<bfloat16_t, float>(out, in, AscendC::RoundMode::CAST_RINT);
    }

    __aicore__ inline void NPU_Cast_FP32ToFP16(Unisor<UBMem, Dim2> &out, Unisor<UBMem, Dim2> &in)
    {
        NPU_Cast<half, float>(out, in, AscendC::RoundMode::CAST_NONE);
    }

    __aicore__ inline void NPU_Cast_FP32ToFP16(Unisor<UBMem, Dim2> &in)
    {
        NPU_Cast_FP32ToFP16(in, in);
    }

    __aicore__ inline void NPU_Cast_INT32ToFP32(Unisor<UBMem, Dim2> &out, Unisor<UBMem, Dim2> &in)
    {
        NPU_Cast<float, int32_t>(out, in, AscendC::RoundMode::CAST_NONE);
    }

    // UB到GM
    __aicore__ inline void NPU_Load_UBToGM(Unisor<GlobalMem, Dim2> &out, Unisor<UBMem, Dim2> &in)
    {
        Dim2 vec = out.vec;
        if (vec.s[0] == 0 || vec.s[1] == 0) {
            return;
        }
        LocalTensor<uint8_t> src = in.get<uint8_t>();
        GlobalTensor<uint8_t> dst = out.getGM();
        Dim2 pos = out.pos;

        float format_size = out.format_size;
        int64_t ele_num = COPY_BLK_BYTES;
        int64_t width = vec.s[1] * format_size;
        uint64_t height = vec.s[0];
        int64_t posWidth = pos.s[1] * format_size;
        uint64_t posHeight = pos.s[0];
        int64_t in_width = out.vector_val.s[1] * format_size;
        uint64_t in_height = out.vector_val.s[0];
        uint64_t nBlocks = ceilINT(width, ele_num);
        uint64_t mBlocks = ceilINT(height, ele_num);
        uint64_t t_nBlocks = ceilINT(in_width, ele_num);

        //ND->ND
        uint16_t blockCount = height;
        uint16_t blockLen = nBlocks;
        uint16_t srcStride = 0;
        uint16_t dstStride = t_nBlocks - blockLen;
        int64_t srcOffset = 0;
        int64_t dstOffset = posHeight * in_width + posWidth;
        DataCopy(dst[dstOffset], src[srcOffset], { blockCount, blockLen, srcStride, dstStride});
    }

    // 从 GM 到 L1A, ND -> Nz
    __aicore__ inline void NPU_Load_GMToL1A(Unisor<L1AMem, Dim2> &out, Unisor<GlobalMem, Dim2> &in)
    {
        Dim2 vec = out.vec;
        if (vec.s[0] == 0 || vec.s[1] == 0) {
            return;
        }

        GlobalTensor<uint8_t> src = in.getGM();
        LocalTensor<uint8_t> dst = out.get<uint8_t>();

        float format_size = in.format_size;
        int64_t ele_num = COPY_BLK_BYTES;
        uint64_t in_posHeight = in.pos.s[0];
        int64_t in_posWidth = in.pos.s[1] * format_size;
        uint64_t out_posHeight = out.pos.s[0];
        int64_t out_posWidth = out.pos.s[1] * format_size;

        int64_t in_width = in.vector_val.s[1] * format_size;
        int64_t out_height = ceilINT_16(out.vector_val.s[0]);

        int64_t height = ceilINT_16(vec.s[0]);
        int64_t width = vec.s[1] * format_size;

        int64_t srcOffset = in_posHeight * in_width + in_posWidth;
        int64_t dstOffset = out_posWidth * out_height + out_posHeight * ele_num;

        AscendC::Nd2NzParams dataCopyA1Params;
        dataCopyA1Params.ndNum = 1;
        dataCopyA1Params.nValue = height;
        dataCopyA1Params.dValue = width;
        dataCopyA1Params.srcNdMatrixStride = 0;
        dataCopyA1Params.srcDValue = in_width;
        dataCopyA1Params.dstNzC0Stride = out_height;
        dataCopyA1Params.dstNzNStride = 1;
        dataCopyA1Params.dstNzMatrixStride = 0;
        AscendC::DataCopy(dst[dstOffset], src[srcOffset], dataCopyA1Params);
    }

    // 从 GM 到 L1B, ND -> Nz
    __aicore__ inline void NPU_Load_GMToL1B(Unisor<L1BMem, Dim2> &out, Unisor<GlobalMem, Dim2> &in)
    {
        Dim2 vec = out.vec;
        if (vec.s[0] == 0 || vec.s[1] == 0) {
            return;
        }

        GlobalTensor<uint8_t> src = in.getGM();
        LocalTensor<uint8_t> dst = out.get<uint8_t>();

        float format_size = in.format_size;
        int64_t ele_num = COPY_BLK_BYTES;
        int64_t in_height = in.vector_val.s[0];
        int64_t in_width = in.vector_val.s[1] * format_size;
        int64_t out_height = out.vector_val.s[0];
        int64_t out_width = out.vector_val.s[1] * format_size;
        int64_t height = vec.s[0];
        int64_t width = vec.s[1] * format_size;
        uint64_t in_posHeight = in.pos.s[0];
        int64_t in_posWidth = in.pos.s[1] * format_size;
        uint64_t out_posHeight = out.pos.s[0];
        int64_t out_posWidth = out.pos.s[1] * format_size;

        // NZ -> NZ 格式
        uint16_t blockCount = width / ele_num;
        uint16_t blockLen = height;
        uint16_t dstStride = out_height - height;
        uint16_t srcStride = in_height - height;

        uint64_t srcOffset = in_posWidth * in_height + in_posHeight * ele_num;
        uint64_t dstOffset = out_posWidth * out_height + out_posHeight * ele_num;
        DataCopy(dst[dstOffset], src[srcOffset], { blockCount, blockLen, srcStride, dstStride});
    }

    // 从 L1 到 L0A, 转格式
    __aicore__ inline void NPU_Load_L1ToL0A(TBuf<L0AMem> &dst, TBuf<L1AMem> &src, Cartesian<Dim2> dstCart, Cartesian<Dim2> srcCart, Dim2 srcShape, float srcFormat)
    {
        Dim2 vec = dstCart.vec;
        if (vec.s[0] == 0 || vec.s[1] == 0) {
            return;
        }

        LocalTensor<uint8_t> srcLocal = src.template Get<uint8_t>();
        LocalTensor<uint8_t> dstLocal = dst.template Get<uint8_t>();
        float format_size = srcFormat;
        int ele_num = COPY_BLK_BYTES;

        int32_t in_posHeight = srcCart.pos.s[0];
        int32_t in_posWidth = srcCart.pos.s[1] * format_size;

        uint32_t height = ceilINT_16(vec.s[0]);
        int32_t width = vec.s[1] * format_size;
        uint32_t in_height = ceilINT_16(srcShape.s[0]);

        uint32_t widthBlocks = width / COPY_BLK_BYTES;
        uint32_t heightBlocks = height / ELEM_BASE_ALIGN;

        uint32_t srcStrideOffset = COPY_BLK_BYTES * in_height;
        uint32_t dstStrideOffset = MAT_FRAC_BYTES;

        uint32_t srcOffset = in_posWidth * in_height + in_posHeight * ele_num;
        uint32_t dstOffset = 0;
        AscendC::LoadData2dParams loadL0AParams;
        loadL0AParams.repeatTimes = heightBlocks;
        loadL0AParams.srcStride = 1;
        loadL0AParams.dstGap = widthBlocks - 1;
        loadL0AParams.ifTranspose = false;
        for (uint32_t i = 0; i < widthBlocks; i++) {
            AscendC::LoadData(dstLocal[dstOffset], srcLocal[srcOffset], loadL0AParams);
            srcOffset = srcOffset + srcStrideOffset;
            dstOffset = dstOffset + dstStrideOffset;
        }
    }

    // 从 L1 到 L0B, 转格式
    __aicore__ inline void NPU_Load_L1ToL0B(TBuf<L0BMem> &dst, TBuf<L1BMem> &src, Cartesian<Dim2> dstCart, Cartesian<Dim2> srcCart, Dim2 srcShape, float srcFormat)
    {
        Dim2 vec = dstCart.vec;
        if (vec.s[0] == 0 || vec.s[1] == 0) {
            return;
        }

        float format_size = srcFormat;
        /* 一次传输的元素个数 */
        uint32_t ele = static_cast<uint32_t>(static_cast<int32_t>((float)COPY_BLK_BYTES / format_size));

        // INT4类型
        if (format_size - (float)INT4 <= 0.0f) {
            LocalTensor<int4b_t> srcLocal = src.template Get<int4b_t>();
            LocalTensor<int4b_t> dstLocal = dst.template Get<int4b_t>();

            uint32_t in_posHeight = srcCart.pos.s[0];
            uint32_t in_posWidth = srcCart.pos.s[1];

            uint32_t height = vec.s[0];
            uint32_t width = vec.s[1];
            uint32_t in_height = srcShape.s[0];
            uint32_t widthBlocks = width / ele;
            uint32_t heightBlocks = height / ele;
            uint32_t in_heightBlocks = in_height / ele;

            uint32_t srcOffset = in_posWidth * in_height + in_posHeight * ele;
            uint32_t dstOffset = 0;

            // 非转置
            if (vec.axis_name[0] == 'K') {
                for (uint32_t i = 0; i < widthBlocks; i++) {
                    AscendC::LoadData2dTransposeParams loadDataParams;
                    loadDataParams.srcStride = 1;
                    loadDataParams.repeatTimes = heightBlocks;
                    loadDataParams.dstGap = widthBlocks * FACTOR_4 - 1;//注意这是指原来相邻的两个方形搬到目的操作数之后的间隔大小
                    loadDataParams.dstFracGap = 0;
                    AscendC::LoadDataWithTranspose(dstLocal[dstOffset], srcLocal[srcOffset], loadDataParams);
                    srcOffset = srcOffset + ele * in_height;
                    dstOffset = dstOffset + MAT_FRAC_BYTES * FACTOR_8;
                }
            } else {//转置
                if (in_heightBlocks == heightBlocks) {
                    AscendC::LoadData2dParams loadL0BParams;
                    loadL0BParams.repeatTimes = heightBlocks * widthBlocks * FACTOR_4;
                    loadL0BParams.srcStride = 1;
                    loadL0BParams.dstGap = 0;
                    loadL0BParams.ifTranspose = false;
                    AscendC::LoadData(dstLocal[dstOffset], srcLocal[srcOffset], loadL0BParams);
                } else {
                    for (uint32_t i = 0; i < widthBlocks; i++) {
                        AscendC::LoadData2dParams loadL0BParams;
                        loadL0BParams.repeatTimes = heightBlocks * FACTOR_4;
                        loadL0BParams.srcStride = 1;
                        loadL0BParams.dstGap = 0;
                        loadL0BParams.ifTranspose = false;
                        AscendC::LoadData(dstLocal[dstOffset], srcLocal[srcOffset], loadL0BParams);
                        srcOffset = srcOffset + in_height * ele;
                        dstOffset = dstOffset + height * ele;
                    }
                }
            }
        } else if ((format_size - (float)INT8) <= 0.0f) {// INT8类型
            LocalTensor<uint8_t> srcLocal = src.template Get<uint8_t>();
            LocalTensor<uint8_t> dstLocal = dst.template Get<uint8_t>();

            uint32_t in_posHeight = srcCart.pos.s[0];
            uint32_t in_posWidth = srcCart.pos.s[1];

            uint32_t height = vec.s[0];
            uint32_t width = vec.s[1];
            uint32_t in_height = srcShape.s[0];
            uint32_t widthBlocks = width / COPY_BLK_BYTES;
            uint32_t heightBlocks = height / COPY_BLK_BYTES;

            uint32_t srcOffset = in_posWidth * in_height + in_posHeight * COPY_BLK_BYTES;
            uint32_t dstOffset = 0;

            // 非转置
            if (vec.axis_name[0] == 'K') {
                for (uint32_t i = 0; i < widthBlocks; i++) {
                    AscendC::LoadData2dTransposeParams loadDataParams;
                    loadDataParams.srcStride = 1;
                    loadDataParams.repeatTimes = heightBlocks;
                    loadDataParams.dstGap = widthBlocks * FACTOR_2 - 1;// 注意这是指原来相邻的两个方形搬到目的操作数之后的间隔大小
                    loadDataParams.dstFracGap = 0;
                    AscendC::LoadDataWithTranspose(dstLocal[dstOffset], srcLocal[srcOffset], loadDataParams);
                    srcOffset = srcOffset + ele * in_height;
                    dstOffset = dstOffset + MAT_FRAC_BYTES * FACTOR_2;
                }
            } else {
                AscendC::LoadData2dParams loadL0BParams;
                loadL0BParams.repeatTimes = heightBlocks * widthBlocks * FACTOR_2;
                loadL0BParams.srcStride = 1;
                loadL0BParams.dstGap = 0;
                loadL0BParams.ifTranspose = false;
                AscendC::LoadData(dstLocal[dstOffset], srcLocal[srcOffset], loadL0BParams);
            }
        } else if ((format_size - (float)FP16) <= 0.0f) {// FP16类型
            LocalTensor<half> srcLocal = src.template Get<half>();
            LocalTensor<half> dstLocal = dst.template Get<half>();

            uint32_t in_posHeight = srcCart.pos.s[0];
            uint32_t in_posWidth = srcCart.pos.s[1];

            uint32_t height = vec.s[0];
            uint32_t width = vec.s[1];
            uint32_t in_height = srcShape.s[0];
            uint32_t widthBlocks = width / ele;
            uint32_t heightBlocks = height / ele;

            uint32_t srcOffset = in_posWidth * in_height + in_posHeight * ele;
            uint32_t dstOffset = 0;

            // 非转置
            if (vec.axis_name[0] == 'K') {
                for (uint32_t i = 0; i < widthBlocks; i++) {
                    AscendC::LoadData2dTransposeParams loadDataParams;
                    loadDataParams.srcStride = 1;
                    loadDataParams.repeatTimes = heightBlocks;
                    loadDataParams.dstGap = widthBlocks - 1;// 注意这是指原来相邻的两个方形搬到目的操作数之后的间隔大小
                    AscendC::LoadDataWithTranspose(dstLocal[dstOffset], srcLocal[srcOffset], loadDataParams);
                    srcOffset = srcOffset + (ele * in_height);
                    dstOffset = dstOffset + MUL_BYTES;
                }
            } else {// 转置
                AscendC::LoadData2dParams loadL0BParams;
                loadL0BParams.repeatTimes = heightBlocks * widthBlocks;
                loadL0BParams.srcStride = 1;
                loadL0BParams.dstGap = 0;
                loadL0BParams.ifTranspose = false;
                AscendC::LoadData(dstLocal[dstOffset], srcLocal[srcOffset], loadL0BParams);
            }
        }
    }

    // MatMul计算, 输出到L0C
    __aicore__ inline   void NPU_Matmul(TBuf<L0CMem> &dstC, TBuf<L0AMem> &srcA, TBuf<L0BMem> &srcB, Dim3 &realShape, float srcFormat, bool isBias)
    {
        if (realShape.s[ARRAY_IDX0] == 0 || realShape.s[ARRAY_IDX1] == 0 || realShape.s[ARRAY_IDX2] == 0) {
            return;
        }

        float format_size = srcFormat;
        // INT4类型
        if ((format_size - (float)INT4) <= 0.0f) {
            LocalTensor<int4b_t> src0 = srcA.template Get<int4b_t>();
            LocalTensor<int4b_t> src1 = srcB.template Get<int4b_t>();
            LocalTensor<int32_t> dst = dstC.template Get<int32_t>();

            MmadParams mmadParams;
            mmadParams.m = (uint16_t)ceilINT_16(realShape.s[ARRAY_IDX0]);
            mmadParams.n = (uint16_t)realShape.s[ARRAY_IDX2];
            mmadParams.k = (uint16_t)realShape.s[ARRAY_IDX1];
            mmadParams.cmatrixInitVal=true;
            mmadParams.cmatrixSource=false;
            mmadParams.isBias=isBias;
            Mmad(dst, src0, src1, mmadParams);
        }
    }

    // 从 L0C 到 GM, NZ to ND
    __aicore__ inline void NPU_Load_L0CToGM_New(GlobalTensor<int32_t> dst, TBuf<L0CMem> &src, Cartesian<Dim2> dstCart, Dim2 dstShape, float dstFormat)
    {
        Dim2 vec = dstCart.vec;
        if (vec.s[0] == 0 || vec.s[1] == 0) {
            return;
        }

        // int32_t 类型
        LocalTensor<int32_t> srcLocal = src.template Get<int32_t>();

        float format_size = dstFormat;
        uint64_t out_posWidth = dstCart.pos.s[1];
        uint64_t out_posHeight = dstCart.pos.s[0];

        uint64_t height = vec.s[0];
        uint64_t height_16 = ceilINT_16(vec.s[0]);
        uint64_t width = vec.s[1];

        int64_t srcOffset = 0;
        int64_t dstOffset = out_posWidth * FACTOR_4096 * format_size;

        AscendC::FixpipeParamsV220 fixpipeParams;
        fixpipeParams.nSize = width;
        fixpipeParams.mSize = height;
        fixpipeParams.srcStride = height_16;
        fixpipeParams.dstStride = width;
        fixpipeParams.ndNum = 1;
        fixpipeParams.srcNdStride = 0;
        fixpipeParams.dstNdStride = 0;
        AscendC::Fixpipe(dst[dstOffset], srcLocal[srcOffset], fixpipeParams);
    }

    // 封装API（带Set/Wait flag）
    __aicore__ inline void npu_matmul_loadL1AToL0A(TBuf<L0AMem> &dst, TBuf<L1AMem> &src, Cartesian<Dim2> dstCart, Cartesian<Dim2> srcCart, Dim2 srcShape, float srcFormat, uint32_t M_MTE1_EventID)
    {
        WaitFlag<HardEvent::M_MTE1>(M_MTE1_EventID);
        NPU_Load_L1ToL0A(dst, src, dstCart, srcCart, srcShape, srcFormat);
        SetFlag<HardEvent::MTE1_M>(M_MTE1_EventID);
    }

    __aicore__ inline void npu_matmul_loadL1BToL0B(TBuf<L0BMem> &dst, TBuf<L1BMem> &src, Cartesian<Dim2> dstCart, Cartesian<Dim2> srcCart, Dim2 srcShape, float srcFormat, uint32_t M_MTE1_EventID)
    {
        WaitFlag<HardEvent::M_MTE1>(M_MTE1_EventID);
        NPU_Load_L1ToL0B(dst, src, dstCart, srcCart, srcShape, srcFormat);
        SetFlag<HardEvent::MTE1_M>(M_MTE1_EventID);
    }

    __aicore__ inline void npu_matmul_loadGMToL0B(Unisor<L0BMem, Dim2> &unisorB0, Unisor<L1BMem, Dim2> &unisorB1, Unisor<GlobalMem, Dim2> &unisorB, uint32_t MTE1_MTE2_EventID, uint32_t M_MTE1_EventID)
    {
        WaitFlag<HardEvent::MTE1_MTE2>(MTE1_MTE2_EventID);
        NPU_Load_GMToL1B(unisorB1, unisorB);
        SetFlag<HardEvent::MTE2_MTE1>(MTE1_MTE2_EventID);
        WaitFlag<HardEvent::MTE2_MTE1>(MTE1_MTE2_EventID);

        WaitFlag<HardEvent::M_MTE1>(M_MTE1_EventID);
        NPU_Load_L1ToL0B(unisorB0.calcBuf, unisorB1.calcBuf, Cartesian<Dim2>(unisorB0.pos, unisorB0.vec), Cartesian<Dim2>(unisorB1.pos, unisorB1.vec), unisorB1.vector_val, unisorB1.format_size);
        SetFlag<HardEvent::MTE1_M>(M_MTE1_EventID);
        SetFlag<HardEvent::MTE1_MTE2>(MTE1_MTE2_EventID);
    }

    __aicore__ inline void npu_matmul_loadGMToL0A_cacheL1(Unisor<L0AMem, Dim2> &unisorA0, Unisor<L1AMem, Dim2> &unisorA1,
        Unisor<GlobalMem, Dim2> &unisorA, uint32_t MTE1_MTE2_EventID_0, uint32_t MTE1_MTE2_EventID_1, uint32_t M_MTE1_EventID, bool NN_firstflag, bool NN_lastflag, bool firstNIter)
    {
        if (firstNIter) {
            // ######## GM -> L1A
            WaitFlag<HardEvent::MTE1_MTE2>(MTE1_MTE2_EventID_1);
            NPU_Load_GMToL1A(unisorA1, unisorA);
            SetFlag<HardEvent::MTE2_MTE1>(MTE1_MTE2_EventID_1);
            WaitFlag<HardEvent::MTE2_MTE1>(MTE1_MTE2_EventID_1);
        }
        // ######## L1A -> L0A
        WaitFlag<HardEvent::M_MTE1>(M_MTE1_EventID);
        NPU_Load_L1ToL0A(unisorA0.calcBuf, unisorA1.calcBuf, Cartesian<Dim2>(unisorA0.pos, unisorA0.vec), Cartesian<Dim2>(unisorA1.pos, unisorA1.vec), unisorA1.vector_val, unisorA1.format_size);
        if (firstNIter) {
            SetFlag<HardEvent::MTE1_MTE2>(MTE1_MTE2_EventID_1);
        }
        SetFlag<HardEvent::MTE1_M>(M_MTE1_EventID);
    }

    // loadEvenEventID是通过isEven决定的EventID！
    __aicore__ inline void npu_matmulUnisor_ping(Unisor<L0CMem, Dim2> &unisorC0, Unisor<L0AMem, Dim2> &unisorA0, Unisor<L0BMem, Dim2> &unisorB0,
            Dim3 &realShape, uint32_t loadEvenEventID, uint32_t loadEventID, uint32_t storeCEventID, bool firstKIter)
    {
        if (firstKIter) {
            WaitFlag<HardEvent::FIX_M>(storeCEventID);// 与步骤4对称
        }
        WaitFlag<HardEvent::MTE1_M>(loadEvenEventID);
        WaitFlag<HardEvent::MTE1_M>(loadEventID);
        NPU_Matmul(unisorC0.calcBuf, unisorA0.calcBuf, unisorB0.calcBuf, realShape, unisorA0.format_size, !firstKIter);
        SetFlag<HardEvent::M_MTE1>(loadEventID); // 与步骤2对称
    }

    // loadEvenEventID是通过isEven决定的EventID！
    __aicore__ inline void npu_matmulUnisor_pong(Unisor<L0CMem, Dim2> &unisorC0, Unisor<L0AMem, Dim2> &unisorA0, Unisor<L0BMem, Dim2> &unisorB0,
            Dim3 &realShape, uint32_t loadEvenEventID, uint32_t loadEventID, uint32_t storeCEventID, bool firstKIter)
    {
        if (firstKIter) {
            WaitFlag<HardEvent::FIX_M>(storeCEventID);// 与步骤4对称
        }
        WaitFlag<HardEvent::MTE1_M>(loadEventID);
        NPU_Matmul(unisorC0.calcBuf, unisorA0.calcBuf, unisorB0.calcBuf, realShape, unisorA0.format_size, !firstKIter);
        SetFlag<HardEvent::M_MTE1>(loadEventID);
        SetFlag<HardEvent::M_MTE1>(loadEvenEventID);
    }

    __aicore__ inline void npu_matmul_storeL0CToGM_New(Unisor<GlobalMem, Dim2> &unisorC, Unisor<L0CMem, Dim2> &unisorC0, uint32_t eventID, bool atomic_flag)
    {
        // int32_t 类型
        if (atomic_flag) {
            SetAtomicAdd<int32_t>();
        }

        SetFlag<HardEvent::M_FIX>(eventID);
        WaitFlag<HardEvent::M_FIX>(eventID);
        NPU_Load_L0CToGM_New(unisorC.getGM_32(), unisorC0.calcBuf, Cartesian<Dim2>(unisorC.pos, unisorC.vec), unisorC.vector_val, unisorC.format_size);
        SetFlag<HardEvent::FIX_M>(eventID);
        if (atomic_flag) {
            SetAtomicNone();
        }
    }

    __aicore__ inline void NPU_Add_float(Unisor<UBMem, Dim2> &out, Unisor<UBMem, Dim2> &a, Unisor<UBMem, Dim2> &b)
    {
        LocalTensor<float> outLocal = out.get<float>();
        LocalTensor<float> aLocal = a.get<float>();
        LocalTensor<float> bLocal = b.get<float>();

        outLocal = aLocal + bLocal;
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void NPU_Add_float(Unisor<UBMem, Dim1> &out, Unisor<UBMem, Dim1> &a, Unisor<UBMem, Dim1> &b)
    {
        LocalTensor<float> outLocal = out.get<float>();
        LocalTensor<float> aLocal = a.get<float>();
        LocalTensor<float> bLocal = b.get<float>();

        outLocal = aLocal + bLocal;
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void NPU_Load_GMToUB(Unisor<UBMem, Dim1> &out, Unisor<GlobalMem, Dim1> &in)
    {
        Dim1 vec = in.vec;
        if (vec.s[0] == 0) {
            return;
        }
        GlobalTensor<uint8_t> srcLocal = in.getGM();
        LocalTensor<uint8_t> dstLocal = out.get<uint8_t>();
        Dim1 pos = in.pos;

        float format_size = in.format_size;
        int ele_num = COPY_BLK_BYTES;
        int32_t width = vec.s[0] * format_size;
        int32_t posWidth = pos.s[0] * format_size;
        int32_t in_width = in.vector_val.s[0] * format_size;
        uint32_t widthBlock = ceilINT(width, ele_num);

        //ND->ND
        uint16_t srcStride = 0;
        uint16_t dstStride = 0;
        uint16_t blockCount = 1;
        uint16_t blockLen = ceilINT(width, COPY_BLK_BYTES);
        int64_t srcOffset = posWidth;
        int64_t dstOffset = 0;
        DataCopy(dstLocal[dstOffset], srcLocal[srcOffset], { blockCount, blockLen, srcStride, dstStride});
    }

    __aicore__ inline void NPU_Load_UBToGM(Unisor<GlobalMem, Dim1> &out, Unisor<UBMem, Dim1> &in)
    {
        Dim1 vec = out.vec;
        if (vec.s[0] == 0) {
            return;
        }
        LocalTensor<uint8_t> src = in.get<uint8_t>();
        GlobalTensor<uint8_t> dst = out.getGM();
        Dim1 pos = out.pos;

        float format_size = out.format_size;
        int ele_num = COPY_BLK_BYTES;
        int32_t in_width = out.vector_val.s[0] * format_size;
        int32_t width = vec.s[0] * format_size;
        int32_t posWidth = pos.s[0] * format_size;

        // ND->ND
        uint16_t blockCount = 1;
        uint16_t blockLen = ceilINT(width, COPY_BLK_BYTES);
        uint16_t srcStride = 0;
        uint16_t dstStride = 0;
        int64_t srcOffset = 0;
        int64_t dstOffset = posWidth;
        DataCopy(dst[dstOffset], src[srcOffset], { blockCount, blockLen, srcStride, dstStride});
    }

    __aicore__ inline uint32_t ceilINT(uint32_t x, uint32_t y)
    {
        return Ceil(x, y);
    }

    __aicore__ inline uint32_t splitBy_64(uint32_t length)
    {
        if (length <= FACTOR_64) {
            return length;
        }
        uint32_t temp = length / FACTOR_2;
        temp = ceilINT_64(temp);
        return temp;
    }

    __aicore__ inline uint32_t getMiniK_SplitBaseN(Dim2 baseCTiling, uint32_t total_k, float format_size)
    {
        int temp0 = 0;
        int temp1 = 0;

        // split_N
        temp0 = szMemL0AB / FACTOR_2 / baseCTiling.s[0] / format_size;
        temp1 = szMemL0AB / baseCTiling.s[1] / format_size;

        uint32_t temp =  temp0 > temp1 ? temp1 : temp0;
        uint32_t result = 0;
        if (temp > ceilINT_16(total_k)) {
            result = total_k;
        } else if (temp > FACTOR_16) {
            if (temp % FACTOR_16 == 0) {
                result = temp;
            } else {
                result = ceilINT_16(temp) - FACTOR_16;
            }
        } else {
            result = FACTOR_16;
        }

        return  result;
    }

    __aicore__ inline void initBlockPos_3D(AxisTiling3Vector &splitRecord)
    {
        splitDim.s[ARRAY_IDX0] = splitRecord.s[ARRAY_IDX0];
        splitDim.s[ARRAY_IDX1] = splitRecord.s[ARRAY_IDX1];
        splitDim.s[ARRAY_IDX2] = splitRecord.s[ARRAY_IDX2];

        uint32_t coreID = GetBlockIdx();
        if (g_coreType == AscendC::AIV) {
            coreID = coreID / FACTOR_2;
        }
        block.s[ARRAY_IDX2] = coreID % splitRecord.s[ARRAY_IDX2];
        uint32_t temp = coreID / splitRecord.s[ARRAY_IDX2];
        block.s[ARRAY_IDX1] = temp % splitRecord.s[ARRAY_IDX1];
        block.s[ARRAY_IDX0] = temp / splitRecord.s[ARRAY_IDX1];
    }
};

class KernelMatmul: public KernelBase {
public:
    __aicore__ inline  KernelMatmul(): KernelBase() {}

    __aicore__ inline void npu_user_defined_matmul_kernel_switch(Unisor<GlobalMem, Dim2> &unisorA, Unisor<GlobalMem, Dim2> &unisorB, Unisor<GlobalMem, Dim2> &unisorC, Unisor<GlobalMem, Dim2> &workUnisor, Unisor<GlobalMem, Dim1> &RowsumUnisor, Unisor<GlobalMem, Dim2> &biasUnisor, Unisor<GlobalMem, Dim2> &offsetUnisor, Unisor<GlobalMem, Dim1> &saUnisor, Unisor<GlobalMem, Dim2> &swUnisor, Dim2 &baseCTiling, uint32_t group_index, uint8_t kernel_index)
    {
        npu_user_defined_matmul_kernel_slave_CacheA_split_BaseN(unisorA, unisorB, unisorC, workUnisor, RowsumUnisor, biasUnisor, offsetUnisor, saUnisor, swUnisor, baseCTiling, group_index);
    }

    __aicore__ inline void npu_user_defined_matmul_kernel_slave_CacheA_split_BaseN(Unisor<GlobalMem, Dim2> &unisorA, Unisor<GlobalMem, Dim2> &unisorB, Unisor<GlobalMem, Dim2> &unisorC, Unisor<GlobalMem, Dim2> &workUnisor, Unisor<GlobalMem, Dim1> &RowsumUnisor, Unisor<GlobalMem, Dim2> &biasUnisor, Unisor<GlobalMem, Dim2> &offsetUnisor, Unisor<GlobalMem, Dim1> &saUnisor, Unisor<GlobalMem, Dim2> &swUnisor, Dim2 &baseCTiling, uint32_t group_index)
    {
        float format_size = unisorA.format_size;
        uint32_t split_k = unisorA.vec.getAxisByName('K');
        uint32_t mk = getMiniK_SplitBaseN(baseCTiling, split_k, format_size);
        bool atomic_flag = (splitDim.s[0]>1);
        uint32_t first_M = unisorA.pos.getAxisByName('M');
        uint32_t Last_M = unisorA.pos.getAxisByName('M') + unisorA.vec.getAxisByName('M');
        uint32_t first_N = unisorB.pos.getAxisByName('N');
        uint32_t Last_N = unisorB.pos.getAxisByName('N') + unisorB.vec.getAxisByName('N');
        uint32_t Base_M = baseCTiling.getAxisByName('M');
        uint32_t Base_N = baseCTiling.getAxisByName('N');
        Dim1 posN(unisorB.pos.getAxisByName('N'), 'N');
        Dim1 vecN(unisorB.vec.getAxisByName('N'), 'N');
        Dim1 posM(unisorA.pos.getAxisByName('M'), 'M');
        Dim1 vecM(unisorA.vec.getAxisByName('M'), 'M');

        uint32_t B_posK = unisorB.pos.getAxisByName('K');

        // ping pong buffer
        Unisor<L1AMem, Dim2> cacheUnisorA;
        Unisor<L1BMem, Dim2> unisorBping;
        Unisor<L1BMem, Dim2> unisorBpong;

        //L0A
        Unisor<L0AMem, Dim2> unisorA0ping;
        Unisor<L0AMem, Dim2> unisorA0pong;

        //L0B
        Unisor<L0BMem, Dim2> unisorB0ping;
        Unisor<L0BMem, Dim2> unisorB0pong;

        //L0C
        Unisor<L0CMem, Dim2> unisorC0ping;
        Unisor<L0CMem, Dim2> unisorC0pong;

        //取Base_N的一半（且是16的倍数）
        uint32_t N_length11 = splitBy_64(Base_N);
        uint32_t N_length22 = Base_N - N_length11;

        //BaseM x K
        Dim2 vecAA = Dim2(Base_M, unisorA.vec.getAxisByName('K'), 'M', 'K');

        //miniK x BaseN
        Dim2 vecMiniAA = Dim2(Base_M, mk, 'M', 'K');
        Dim2 vecMiniBB11 = Dim2(mk, N_length11, 'K', 'N');
        Dim2 vecMiniBB22 = Dim2(mk, N_length22, 'K', 'N');
        if (pattern != 1) {
            vecMiniBB11.transpose();
            vecMiniBB22.transpose();
        }
        Dim2 vecCC11 = Dim2(Base_M, N_length11, 'M', 'N');
        Dim2 vecCC22 = Dim2(Base_M, N_length22, 'M', 'N');

        if (g_coreType == AscendC::AIC) {
            //L1A
            cacheUnisorA.init(vecAA, unisorA.format_size, this->pipe);
            //L1B
            unisorBping.init(vecMiniBB11, unisorB.format_size, this->pipe);
            unisorBpong.init(vecMiniBB22, unisorB.format_size, this->pipe);

            //L0A
            unisorA0ping.init(vecMiniAA, unisorA.format_size, this->pipe);
            unisorA0pong.init(vecMiniAA, unisorA.format_size, this->pipe);
            //L0B
            unisorB0ping.init(vecMiniBB11, unisorB.format_size, this->pipe);
            unisorB0pong.init(vecMiniBB22, unisorB.format_size, this->pipe);
            //L0C
            unisorC0ping.init(vecCC11, L0C_FORMAT_SIZE, this->pipe); //这里手动指定Float32
            unisorC0pong.init(vecCC22, L0C_FORMAT_SIZE, this->pipe); //这里手动指定Float32
        }

        if (g_coreType == AscendC::AIC) {
            SetFlag<HardEvent::MTE1_MTE2>(LOCAL_HWEVENT_ID0);
            SetFlag<HardEvent::MTE1_MTE2>(LOCAL_HWEVENT_ID1);
            SetFlag<HardEvent::MTE1_MTE2>(LOCAL_HWEVENT_ID2);
            SetFlag<HardEvent::MTE1_MTE2>(LOCAL_HWEVENT_ID3);

            SetFlag<HardEvent::M_MTE1>(LOCAL_HWEVENT_ID4);
            SetFlag<HardEvent::M_MTE1>(LOCAL_HWEVENT_ID5);
            SetFlag<HardEvent::M_MTE1>(LOCAL_HWEVENT_ID6);
            SetFlag<HardEvent::M_MTE1>(LOCAL_HWEVENT_ID7);
            SetFlag<HardEvent::FIX_M>(LOCAL_HWEVENT_ID6);
            SetFlag<HardEvent::FIX_M>(LOCAL_HWEVENT_ID7);
        }

        //K轴切分
        for (uint32_t i=0; i<splitDim.s[0]; i++) {
            if (i>0) {
                atomic_flag = true;
            } else {
                atomic_flag = false;
            }

            bool isFirst_K = (i == 0);
            bool isLast_K = (i == (splitDim.s[0] - 1));
            uint32_t pos_K = i * split_k;
            uint32_t current_K = split_k;
            if (i == (splitDim.s[0] - 1)) {
                current_K = unisorA.vector_val.s[1] - (i * split_k);
            }

            uint32_t first_K = pos_K;
            uint32_t last_K = pos_K + current_K;

            //N轴迭代器
            Cartesian<Dim1> cartesianN(posN, vecN);
            Dim1 baseN(Base_N, 'N');
            CartesianIterator<Dim1> unisorIteratorN(cartesianN, baseN);
            Cartesian<Dim1> cartesianUnisorN;

            //M轴迭代器
            Cartesian<Dim1> cartesianM(posM, vecM);
            Dim1 baseM(Base_M, 'M');
            CartesianIterator<Dim1> unisorIteratorM(cartesianM, baseM);
            Cartesian<Dim1> cartesianUnisorM;

            //K轴迭代器
            Dim1 pos_KK(pos_K, 'K');
            Dim1 vec_KK(current_K, 'K');
            Cartesian<Dim1> cartesianK(pos_KK, vec_KK);
            Dim1 miniK(mk, 'K');
            CartesianIterator<Dim1> unisorIteratorK(cartesianK, miniK);
            Cartesian<Dim1> cartesianUnisorK;
            cartesianUnisorK.pos = pos_KK;
            cartesianUnisorK.vec = vec_KK;

            uint32_t count = 0;
            uint32_t write_point_count = 0;
            while (unisorIteratorM.posIterator(cartesianUnisorM)) {
                while (unisorIteratorN.posIterator(cartesianUnisorN)) {
                    Dim2 posC = Dim2(cartesianUnisorM.pos.getAxisByName('M'), cartesianUnisorN.pos.getAxisByName('N'), 'M', 'N');
                    Dim2 vecC = Dim2(cartesianUnisorM.vec.getAxisByName('M'), cartesianUnisorN.vec.getAxisByName('N'), 'M', 'N');

                    uint32_t N_length1 = splitBy_64(vecC.getAxisByName('N'));
                    uint32_t N_length2 = vecC.getAxisByName('N') - N_length1;

                    //尾块处理
                    if (vecC.getAxisByName('N') <= N_length11) {
                        N_length1=vecC.getAxisByName('N');
                        N_length2=0;
                    }

                    while (unisorIteratorK.posIterator(cartesianUnisorK)) {
                        Dim1 posK = cartesianUnisorK.pos;
                        Dim1 vecK = cartesianUnisorK.vec;
                        Dim3 realShape1(vecC.getAxisByName('M'), vecK.getAxisByName('K'), N_length1, 'M', 'K', 'N');
                        Dim3 realShape2(vecC.getAxisByName('M'), vecK.getAxisByName('K'), N_length2, 'M', 'K', 'N');

                        //GM_A矩阵坐标
                        Dim2 posA_GM = Dim2(posC.getAxisByName('M'), posK.getAxisByName('K'), 'M', 'K');
                        Dim2 vecA_GM = Dim2(vecC.getAxisByName('M'), vecK.getAxisByName('K'), 'M', 'K');
                        Cartesian<Dim2> cartesianTempA_GM1(posA_GM, vecA_GM);

                        //L1A矩阵坐标
                        Dim2 posA_L11 = Dim2(0, posK.getAxisByName('K') - first_K, 'M', 'K');
                        Cartesian<Dim2> cartesianTempA_L11(posA_L11, vecA_GM);

                        //L0A矩阵坐标
                        Dim2 posA_L01 = Dim2(0, 0, 'M', 'K');
                        Cartesian<Dim2> cartesianTempA_L01(posA_L01, vecA_GM);

                        //GM_B1矩阵坐标
                        Dim2 posB_GM1 = Dim2(posK.getAxisByName('K'), posC.getAxisByName('N'), 'K', 'N');
                        Dim2 vecB_GM1 = Dim2(vecK.getAxisByName('K'), N_length1, 'K', 'N');
                        if (pattern != 1) {
                            posB_GM1.transpose();
                            vecB_GM1.transpose();
                        }
                        posB_GM1.s[1] = posB_GM1.s[1] + B_posK;
                        Cartesian<Dim2> cartesianTempB_GM1(posB_GM1, vecB_GM1);

                        //GM_B2矩阵坐标
                        Dim2 posB_GM2 = Dim2(posK.getAxisByName('K'), posC.getAxisByName('N') + N_length1, 'K', 'N');
                        Dim2 vecB_GM2 = Dim2(vecK.getAxisByName('K'), N_length2, 'K', 'N');
                        if (pattern != 1) {
                            posB_GM2.transpose();
                            vecB_GM2.transpose();
                        }
                        posB_GM2.s[1] = posB_GM2.s[1] + B_posK;
                        Cartesian<Dim2> cartesianTempB_GM2(posB_GM2, vecB_GM2);

                        //L1_B1矩阵坐标
                        Dim2 posB_L11 = Dim2(0, 0, 'K', 'N');
                        if (pattern != 1) {
                            posB_L11.transpose();
                        }
                        Cartesian<Dim2> cartesianTempB_L11(posB_L11, vecB_GM1);

                        //L1_B2矩阵坐标
                        Dim2 posB_L12 = Dim2(0, 0, 'K', 'N');
                        if (pattern != 1) {
                            posB_L12.transpose();
                        }
                        Cartesian<Dim2> cartesianTempB_L12(posB_L12, vecB_GM2);

                        bool firstNIter = (cartesianUnisorN.pos.getAxisByName('N') == first_N);
                        bool LastNIter = (cartesianUnisorN.pos.getAxisByName('N') + Base_N >= Last_N);
                        bool firstMIter = (cartesianUnisorM.pos.getAxisByName('M') == first_M);
                        bool LastMIter = (cartesianUnisorM.pos.getAxisByName('M') + Base_M >= Last_M);
                        bool firstKIter = (posK.getAxisByName('K') == first_K);
                        bool lastKIter = (posK.getAxisByName('K') + mk >= last_K);
                        bool NN_firstflag = (firstNIter && firstKIter);
                        bool NN_lastflag = (LastNIter && lastKIter);
                        bool isFirstMN = (firstMIter && firstNIter);
                        bool isLastMN = (LastMIter && LastNIter);
                        bool postFlag = (isLast_K && lastKIter);
                        bool isEven = (count % FACTOR_2 == 0);

                        if (g_coreType == AscendC::AIC) {
                            if (isEven) {
                                unisorA.setCartesian(cartesianTempA_GM1);
                                cacheUnisorA.setCartesian(cartesianTempA_L11);
                                unisorA0ping.setCartesian(cartesianTempA_L11);
                                npu_matmul_loadGMToL0A_cacheL1(unisorA0ping, cacheUnisorA, unisorA, LOCAL_HWEVENT_ID4, LOCAL_HWEVENT_ID0, LOCAL_HWEVENT_ID4, NN_firstflag, NN_lastflag, firstNIter);
                            } else {
                                unisorA.setCartesian(cartesianTempA_GM1);
                                cacheUnisorA.setCartesian(cartesianTempA_L11);
                                unisorA0pong.setCartesian(cartesianTempA_L11);
                                npu_matmul_loadGMToL0A_cacheL1(unisorA0pong, cacheUnisorA, unisorA, LOCAL_HWEVENT_ID5, LOCAL_HWEVENT_ID1, LOCAL_HWEVENT_ID5, NN_firstflag, NN_lastflag, firstNIter);
                            }

                            unisorB.setCartesian(cartesianTempB_GM1);
                            unisorBping.setCartesian(cartesianTempB_L11);
                            unisorB0ping.setCartesian(cartesianTempB_L11);
                            npu_matmul_loadGMToL0B(unisorB0ping, unisorBping, unisorB, LOCAL_HWEVENT_ID2, LOCAL_HWEVENT_ID6);

                            if (isEven) {
                                npu_matmulUnisor_ping(unisorC0ping, unisorA0ping, unisorB0ping, realShape1, LOCAL_HWEVENT_ID4, LOCAL_HWEVENT_ID6, LOCAL_HWEVENT_ID6, firstKIter);
                            } else {
                                npu_matmulUnisor_ping(unisorC0ping, unisorA0pong, unisorB0ping, realShape1, LOCAL_HWEVENT_ID5, LOCAL_HWEVENT_ID6, LOCAL_HWEVENT_ID6, firstKIter);
                            }

                            if (lastKIter) {
                                if (isFirst_K && isFirstMN) {
                                    AscendC::CrossCoreWaitFlag(LOCAL_FLAGID7);
                                }
                                Dim2 posC1 = Dim2(1, write_point_count, 'M', 'N');
                                Dim2 vecC1 = Dim2(cartesianUnisorM.vec.getAxisByName('M'), N_length1, 'M', 'N');
                                Cartesian<Dim2> cartesianTempC1(posC1, vecC1);
                                workUnisor.setCartesian(cartesianTempC1);
                                npu_matmul_storeL0CToGM_New(workUnisor, unisorC0ping, LOCAL_HWEVENT_ID6, atomic_flag);
                            }
                        }

                        //输出1
                        if (postFlag && g_coreType == AscendC::AIC) {
                            AscendC::CrossCoreSetFlag<LOCAL_MODEID2, PIPE_FIX>(LOCAL_FLAGID4);
                        }

                        if (postFlag && g_coreType == AscendC::AIV) {
                            AscendC::CrossCoreWaitFlag(LOCAL_FLAGID4);
                            Dim2 pos_Bias(group_index, cartesianUnisorN.pos.getAxisByName('N'), 'M', 'N');
                            Dim2 vec_Bias(1, N_length1, 'M', 'N');
                            biasUnisor.setCartesian(pos_Bias, vec_Bias);

                            Dim2 pos_Offset(group_index, cartesianUnisorN.pos.getAxisByName('N'), 'M', 'N');
                            Dim2 vec_Offset(1, N_length1, 'M', 'N');
                            offsetUnisor.setCartesian(pos_Offset, vec_Offset);

                            Dim2 pos_Sw(group_index, cartesianUnisorN.pos.getAxisByName('N'), 'M', 'N');
                            Dim2 vec_Sw(1, N_length1, 'M', 'N');
                            swUnisor.setCartesian(pos_Sw, vec_Sw);

                            Dim2 posWork = Dim2(1, write_point_count, 'M', 'N');
                            Dim2 vecC1 = Dim2(cartesianUnisorM.vec.getAxisByName('M'), N_length1, 'M', 'N');
                            Cartesian<Dim2> cartesianTempWork(posWork, vecC1);
                            workUnisor.setCartesian(cartesianTempWork);

                            Dim2 posC1 = Dim2(cartesianUnisorM.pos.getAxisByName('M'), cartesianUnisorN.pos.getAxisByName('N'), 'M', 'N');
                            Cartesian<Dim2> cartesianTempC1(posC1, vecC1);
                            unisorC.setCartesian(cartesianTempC1);

                            if (offset_enable) {
                                npu_user_defined_matmul_kernel_slave_Post_Processing(unisorC, workUnisor, RowsumUnisor, biasUnisor, offsetUnisor, saUnisor, swUnisor);
                            } else {
                                npu_user_defined_matmul_kernel_slave_Post_Processing_off(unisorC, workUnisor, biasUnisor, saUnisor, swUnisor);
                            }
                        }

                        if (lastKIter) {
                            write_point_count++;
                        }

                        if (g_coreType == AscendC::AIC) {
                            // ######### GM -> L0B
                            unisorB.setCartesian(cartesianTempB_GM2);
                            unisorBpong.setCartesian(cartesianTempB_L12);
                            unisorB0pong.setCartesian(cartesianTempB_L12);
                            npu_matmul_loadGMToL0B(unisorB0pong, unisorBpong, unisorB, LOCAL_HWEVENT_ID3, LOCAL_HWEVENT_ID7);

                            if (isEven) {
                                npu_matmulUnisor_pong(unisorC0pong, unisorA0ping, unisorB0pong, realShape2, LOCAL_HWEVENT_ID4, LOCAL_HWEVENT_ID7, LOCAL_HWEVENT_ID7, firstKIter);
                            } else {
                                npu_matmulUnisor_pong(unisorC0pong, unisorA0pong, unisorB0pong, realShape2, LOCAL_HWEVENT_ID5, LOCAL_HWEVENT_ID7, LOCAL_HWEVENT_ID7, firstKIter);
                            }

                            // ######### L0C -> GM
                            if (lastKIter)
                            {
                                Dim2 posC2 = Dim2(1, write_point_count, 'M', 'N');
                                Dim2 vecC2 = Dim2(cartesianUnisorM.vec.getAxisByName('M'), N_length2, 'M', 'N');
                                Cartesian<Dim2> cartesianTempC2(posC2, vecC2);
                                workUnisor.setCartesian(cartesianTempC2);
                                npu_matmul_storeL0CToGM_New(workUnisor, unisorC0pong, LOCAL_HWEVENT_ID7, atomic_flag);
                            }
                        }

                        //输出2
                        if (postFlag && g_coreType == AscendC::AIC) {
                            AscendC::CrossCoreSetFlag<LOCAL_MODEID2, PIPE_FIX>(LOCAL_FLAGID5);
                        }

                        if (postFlag && g_coreType == AscendC::AIV) {
                            AscendC::CrossCoreWaitFlag(LOCAL_FLAGID5);
                            Dim2 pos_Bias(group_index, cartesianUnisorN.pos.getAxisByName('N') + N_length1, 'M', 'N');
                            Dim2 vec_Bias(1, N_length2, 'M', 'N');
                            biasUnisor.setCartesian(pos_Bias, vec_Bias);

                            Dim2 pos_Offset(group_index, cartesianUnisorN.pos.getAxisByName('N') + N_length1, 'M', 'N');
                            Dim2 vec_Offset(1, N_length2, 'M', 'N');
                            offsetUnisor.setCartesian(pos_Offset, vec_Offset);

                            Dim2 pos_Sw(group_index, cartesianUnisorN.pos.getAxisByName('N') + N_length1, 'M', 'N');
                            Dim2 vec_Sw(1, N_length2, 'M', 'N');
                            swUnisor.setCartesian(pos_Sw, vec_Sw);

                            Dim2 posWork = Dim2(1, write_point_count, 'M', 'N');
                            Dim2 vecC2 = Dim2(cartesianUnisorM.vec.getAxisByName('M'), N_length2, 'M', 'N');
                            Cartesian<Dim2> cartesianTempWork(posWork, vecC2);
                            workUnisor.setCartesian(cartesianTempWork);

                            Dim2 posC2 = Dim2(cartesianUnisorM.pos.getAxisByName('M'), cartesianUnisorN.pos.getAxisByName('N') + N_length1, 'M', 'N');
                            Cartesian<Dim2> cartesianTempC2(posC2, vecC2);
                            unisorC.setCartesian(cartesianTempC2);

                            if (offset_enable) {
                                npu_user_defined_matmul_kernel_slave_Post_Processing(unisorC, workUnisor, RowsumUnisor, biasUnisor, offsetUnisor, saUnisor, swUnisor);
                            } else {
                                npu_user_defined_matmul_kernel_slave_Post_Processing_off(unisorC, workUnisor, biasUnisor, saUnisor, swUnisor);
                            }

                            if (isLastMN) {
                                AscendC::CrossCoreSetFlag<LOCAL_MODEID2, PIPE_MTE3>(LOCAL_FLAGID7);
                            }
                        }
                        if (lastKIter) {
                            write_point_count++;
                        }
                        count++;
                    }
                }
            }
        }
        if (g_coreType == AscendC::AIC) {
            WaitFlag<HardEvent::MTE1_MTE2>(LOCAL_HWEVENT_ID0);
            WaitFlag<HardEvent::MTE1_MTE2>(LOCAL_HWEVENT_ID1);
            WaitFlag<HardEvent::MTE1_MTE2>(LOCAL_HWEVENT_ID2);
            WaitFlag<HardEvent::MTE1_MTE2>(LOCAL_HWEVENT_ID3);

            WaitFlag<HardEvent::M_MTE1>(LOCAL_HWEVENT_ID4);
            WaitFlag<HardEvent::M_MTE1>(LOCAL_HWEVENT_ID5);
            WaitFlag<HardEvent::M_MTE1>(LOCAL_HWEVENT_ID6);
            WaitFlag<HardEvent::M_MTE1>(LOCAL_HWEVENT_ID7);
            WaitFlag<HardEvent::FIX_M>(LOCAL_HWEVENT_ID6);
            WaitFlag<HardEvent::FIX_M>(LOCAL_HWEVENT_ID7);
        }
        this->pipe->Reset();
    }

    //INT32转Fp16,只转单核的数据 baseM * baseN
    __aicore__ inline void npu_user_defined_matmul_kernel_slave_Post_Processing(Unisor<GlobalMem, Dim2> &unisorC, Unisor<GlobalMem, Dim2> &workUnisor,
                                                                                Unisor<GlobalMem, Dim1> &RowsumUnisor, Unisor<GlobalMem, Dim2> &biasUnisor,
                                                                                Unisor<GlobalMem, Dim2> &offsetUnisor, Unisor<GlobalMem, Dim1> &saUnisor,
                                                                                Unisor<GlobalMem, Dim2> &swUnisor)
    {
    }

    //INT32转Fp16,只转单核的数据 baseM * baseN
    __aicore__ inline void npu_user_defined_matmul_kernel_slave_Post_Processing_off(Unisor<GlobalMem, Dim2> &unisorC, Unisor<GlobalMem, Dim2> &workUnisor,
                                                                                    Unisor<GlobalMem, Dim2> &biasUnisor, Unisor<GlobalMem, Dim1> &saUnisor, Unisor<GlobalMem, Dim2> &swUnisor)
    {
        //注意下面值不一定是16的倍数
        uint32_t pos_Work_M = workUnisor.pos.getAxisByName('M');
        uint32_t pos_Work_N = workUnisor.pos.getAxisByName('N');
        uint32_t write_point_count = pos_Work_N;
        uint32_t work_N = workUnisor.vec.getAxisByName('N');

        uint32_t startOffset1 = (szMemL0C / FACTOR_2) * write_point_count;
        uint32_t startOffset2 = (szMemL0C / FACTOR_2) * write_point_count + work_N * FACTOR_4;

        uint32_t single_M = workUnisor.vec.getAxisByName('M');
        if (single_M == 0) {
            return;
        }
        uint32_t current_M = ceilINT(single_M, FACTOR_2); //取一半
        bool single_aiv = (single_M < FACTOR_128);
        if (single_aiv) {
            current_M = single_M;
        }
        if (current_M % FACTOR_2 != 0) {
            current_M = current_M + 1;
        }
        uint32_t current_N = workUnisor.vec.getAxisByName('N');
        if (current_N == 0) {
            return;
        }
        uint32_t pos_M = unisorC.pos.getAxisByName('M');
        uint32_t pos_N = unisorC.pos.getAxisByName('N');
        uint32_t pos_MM = pos_M;
        uint32_t rowsum_m = ceilINT(current_M, FACTOR_2);

        uint32_t BlockIdx = GetBlockIdx();
        uint32_t SubBlockIdx = GetSubBlockIdx();
        //第二个V核，暂时不考虑尾块
        if (SubBlockIdx == 1) {
            if (single_aiv) {
                return;
            }
            pos_MM = pos_M + current_M;
            current_M = single_M - current_M;

            if (single_M == FACTOR_2) {
                return;
            }
        }
        if (current_M == 0) {
            return;
        }

        //UB内存是192KB 需要三块Half大小的内存 每块不能超过64KB Base_M * Base_N <= 32 * 1024
        uint32_t UB_Size = szMemUB - FACTOR_8 * current_N - FACTOR_4 * current_M;
        uint32_t Base_M = UB_Size / FACTOR_2 / FACTOR_4 / current_N;
        if (Base_M % FACTOR_2 != 0) {
            Base_M  = Base_M - 1;
        }

        if (Base_M > current_M) {
            Base_M = current_M;
        }
        uint32_t Base_N = current_N;

        //M轴迭代器
        Dim1 posM(pos_MM, 'M');
        Dim1 vecM(current_M, 'M');
        Cartesian<Dim1> cartesianM(posM, vecM);
        Dim1 baseM(Base_M, 'M');
        CartesianIterator<Dim1> unisorIteratorM(cartesianM, baseM);
        Cartesian<Dim1> cartesianUnisorM;

        //N轴迭代器
        Dim1 posN(pos_N, 'N');
        Dim1 vecN(current_N, 'N');
        Cartesian<Dim1> cartesianN(posN, vecN);
        Dim1 baseN(Base_N, 'N');
        CartesianIterator<Dim1> unisorIteratorN(cartesianN, baseN);
        Cartesian<Dim1> cartesianUnisorN;
        Dim2 pos_work(0, 0, 'M', 'N');
        Dim1 pos_work_d1(0, 'M');
        Dim2 vec_work(Base_M / FACTOR_2, Base_N, 'M', 'N');

        //Bias初始化
        Dim2 vec_bias(1, current_N, 'M', 'N');
        Unisor<UBMem, Dim2> UBUnisor_Bias;
        UBUnisor_Bias.init_RealShape(vec_bias, FP32, this->pipe);
        UBUnisor_Bias.setCartesian(pos_work, vec_bias);
        NPU_Load_GMToUB(UBUnisor_Bias, biasUnisor);

        //Sa初始化
        Dim1 vec_sa(Base_M / FACTOR_2, 'M');
        Unisor<UBMem, Dim1> UBUnisor_sa;
        UBUnisor_sa.init(vec_sa, FP32, this->pipe);
        Unisor<UBMem, Dim2> UBUnisor_sa_broadcast;
        UBUnisor_sa_broadcast.init_RealShape(vec_work, FP32, this->pipe);
        UBUnisor_sa_broadcast.setCartesian(pos_work, vec_work);

        //Sw初始化
        Dim2 vec_sw(1, current_N, 'M', 'N');
        Unisor<UBMem, Dim2> UBUnisor_sw;
        UBUnisor_sw.init_RealShape(vec_sw, FP32, this->pipe);
        UBUnisor_sw.setCartesian(pos_work, vec_sw);
        NPU_Load_GMToUB(UBUnisor_sw, swUnisor);
        AscendC::PipeBarrier<PIPE_MTE2>();

        //UB
        Unisor<UBMem, Dim2> UBUnisor_FP32_A1;
        UBUnisor_FP32_A1.init_RealShape(vec_work, FP32, this->pipe);
        Unisor<UBMem, Dim2> UBUnisor_FP32_A2;
        UBUnisor_FP32_A2.init_RealShape(vec_work, FP32, this->pipe);
        Unisor<UBMem, Dim2> UBUnisor_FP16;
        UBUnisor_FP16.init_RealShape(vec_work, FP16, this->pipe);

        uint32_t count = 0;
        AscendC::SetFlag<HardEvent::V_MTE2>(LOCAL_HWEVENT_ID0);
        AscendC::SetFlag<HardEvent::MTE3_V>(LOCAL_HWEVENT_ID1);
        while (unisorIteratorM.posIterator(cartesianUnisorM)) {
            while (unisorIteratorN.posIterator(cartesianUnisorN)) {
                bool isEven = (count % FACTOR_2 == 0);
                Dim2 posC = Dim2(cartesianUnisorM.pos.getAxisByName('M')-pos_M, cartesianUnisorN.pos.getAxisByName('N')-pos_N, 'M', 'N');
                Dim2 out_posC = Dim2(cartesianUnisorM.pos.getAxisByName('M') / FACTOR_2, cartesianUnisorN.pos.getAxisByName('N'), 'M', 'N');
                Dim2 vecC = Dim2(cartesianUnisorM.vec.getAxisByName('M') / FACTOR_2, cartesianUnisorN.vec.getAxisByName('N'), 'M', 'N');
                Dim1 pos_sa = Dim1((cartesianUnisorM.pos.getAxisByName('M')) / FACTOR_2, 'M');
                Dim1 vec_sa = Dim1(cartesianUnisorM.vec.getAxisByName('M') / FACTOR_2, 'M');
                Dim2 pos_sw = Dim2(0, cartesianUnisorN.pos.getAxisByName('N'), 'M', 'N');
                Dim2 vec_sw = Dim2(1, cartesianUnisorN.vec.getAxisByName('N'), 'M', 'N');

                //1、WorkSpace->UB
                workUnisor.setCartesian(posC, vecC);
                UBUnisor_FP32_A1.setCartesian(pos_work, vecC);
                UBUnisor_FP32_A2.setCartesian(pos_work, vecC);
                UBUnisor_FP16.setCartesian(pos_work, vecC);
                saUnisor.setCartesian(pos_sa, vec_sa);
                UBUnisor_sa.setCartesian(pos_work_d1, vec_sa);
                UBUnisor_sa_broadcast.setCartesian(pos_work, vecC);

                AscendC::WaitFlag<HardEvent::V_MTE2>(LOCAL_HWEVENT_ID0);
                NPU_Load_GMToUB_ByStride_ByOffset(UBUnisor_FP32_A1, workUnisor, startOffset1);
                NPU_Load_GMToUB_ByStride_ByOffset(UBUnisor_FP32_A2, workUnisor, startOffset2);
                NPU_Load_GMToUB(UBUnisor_sa, saUnisor);
                AscendC::SetFlag<HardEvent::MTE2_V>(LOCAL_HWEVENT_ID2);
                AscendC::WaitFlag<HardEvent::MTE2_V>(LOCAL_HWEVENT_ID2);
                NPU_Broadcast(UBUnisor_sa_broadcast, UBUnisor_sa);

                //A1乘16
                const int32_t sixteen = FACTOR_16;
                NPU_Muls_int32_t(UBUnisor_FP32_A1, UBUnisor_FP32_A1, sixteen);
                //A1 + A2
                NPU_Add_int32_t(UBUnisor_FP32_A1, UBUnisor_FP32_A1, UBUnisor_FP32_A2);
                AscendC::PipeBarrier<PIPE_V>();
                NPU_Cast_INT32ToFP32(UBUnisor_FP32_A1, UBUnisor_FP32_A1);
                AscendC::PipeBarrier<PIPE_V>();
                //A1反量化
                NPU_VecMuls(UBUnisor_FP32_A1, UBUnisor_sw, UBUnisor_FP32_A1);

                NPU_Add_Bias(UBUnisor_FP32_A1, UBUnisor_Bias);
                AscendC::PipeBarrier<PIPE_V>();
                // * Sa
                NPU_VecMul(UBUnisor_FP32_A1, UBUnisor_sa_broadcast, UBUnisor_FP32_A1);
                //2、FP32->FP16
                AscendC::WaitFlag<HardEvent::MTE3_V>(LOCAL_HWEVENT_ID1);
                if (this->output_type == 0) {
                    NPU_Cast_FP32ToFP16(UBUnisor_FP16, UBUnisor_FP32_A1);
                } else {
                    NPU_Cast_FP32ToBF16(UBUnisor_FP16, UBUnisor_FP32_A1);
                }
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetFlag<HardEvent::V_MTE2>(LOCAL_HWEVENT_ID0);
                AscendC::SetFlag<HardEvent::V_MTE3>(LOCAL_HWEVENT_ID3);
                AscendC::WaitFlag<HardEvent::V_MTE3>(LOCAL_HWEVENT_ID3);
                unisorC.setCartesian(out_posC, vecC);
                NPU_Load_UBToGM(unisorC, UBUnisor_FP16);
                AscendC::SetFlag<HardEvent::MTE3_V>(LOCAL_HWEVENT_ID1);
            }
        }
        AscendC::WaitFlag<HardEvent::V_MTE2>(LOCAL_HWEVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE3_V>(LOCAL_HWEVENT_ID1);
        this->pipe->Reset();
    }

    //A INT8->INT4 处理单核
    __aicore__ inline void npu_user_defined_matmul_kernel_slave_Single_A8ToA4_off(Unisor<GlobalMem, Dim2> in, Unisor<GlobalMem, Dim2> out)
    {
        //多核切分
        uint32_t BlockIdx = GetBlockIdx();
        uint32_t M_length = in.vec.s[0];
        uint32_t K_length = in.vec.s[1];
        //切分成40个Vector核计算
        uint32_t singleM = ceilINT(M_length,nCoreAIV);
        uint32_t singleK = K_length;

        //注意！！！BaseM要根据下面使用了多少个UB Unisor来计算
        uint32_t BaseM = 0;
        if (singleK != 0) {
            BaseM = (szMemUB-singleK) / singleK / FACTOR_7;
        }

        if (BaseM > M_length) {
            BaseM = M_length;
        }

        Dim2 baseTiling(BaseM, singleK, 'M', 'K');

        if (BlockIdx * singleM > M_length) {//已算完
            return ;
        }

        uint32_t current_M = singleM;
        if ((BlockIdx + 1) * singleM > M_length) {
            current_M = M_length - (BlockIdx * singleM);
        }

        if (current_M == 0) {
            return;
        }

        uint32_t posM_in = in.pos.s[0] + BlockIdx * singleM;
        uint32_t posK_in = in.pos.s[1];
        uint32_t posM_out = out.pos.s[0] + BlockIdx * singleM;
        uint32_t posK_out = out.pos.s[1];

        in.pos.setAxisName('M', 'K');
        in.vec.setAxisName('M', 'K');
        out.pos.setAxisName('M', 'K');
        out.vec.setAxisName('M', 'K');

        Unisor<GlobalMem, Dim2> inputUnisor(in);
        Unisor<GlobalMem, Dim2> outputHighUnisor(out);
        Unisor<GlobalMem, Dim2> outputLowUnisor(out);

        Dim2 posIn = Dim2(posM_in, posK_in, 'M', 'K');
        Dim2 vecIn = Dim2(current_M, singleK, 'M', 'K');
        inputUnisor.setCartesian(posIn, vecIn);

        Dim2 posHighOut = Dim2(posM_out, posK_out, 'M', 'K');
        Dim2 vecOut = Dim2(current_M, singleK, 'M', 'K');
        outputHighUnisor.setCartesian(posHighOut, vecOut);

        InternalRun_off(inputUnisor, outputHighUnisor, outputLowUnisor, baseTiling);
        this->pipe->Reset();
    }

    // Note!, this function is only for 2 singleCoreAIV. That is to say, call this function once, two AIVs compute!
    __aicore__ inline void npu_user_defined_matmul_kernel_slave_Double_A8ToA4_off(Unisor<GlobalMem, Dim2> in, Unisor<GlobalMem, Dim2> out, uint32_t a8M_length)
    {
        //多核切分
        uint32_t BlockIdx = GetBlockIdx() % FACTOR_2;
        uint32_t M_length = in.vec.s[0];
        uint32_t K_length = in.vec.s[1];

        if (in.pos.s[0] + M_length >= a8M_length) {
            M_length = a8M_length - in.pos.s[0];
        }

        //切分成 2 个Vector核计算
        uint32_t singleM = ceilINT(M_length, FACTOR_2); // must be a multiple of 8.
        uint32_t singleK = K_length;

        //注意！！！BaseM要根据下面使用了多少个UB Unisor来计算
        uint32_t BaseM = 0;
        if (singleK != 0) {
            BaseM = (szMemUB - singleK) / singleK / FACTOR_7;
        }

        if (BaseM > M_length) {
            BaseM = M_length;
        }

        Dim2 baseTiling(BaseM, singleK, 'M', 'K');

        if (BlockIdx * singleM > M_length) {//已算完
            return ;
        }

        uint32_t current_M = singleM;
        if ((BlockIdx + 1) * singleM > M_length) {
            current_M = M_length - (BlockIdx * singleM);
        }

        if (current_M == 0) {
            return;
        }

        uint32_t posM_in = in.pos.s[0] + BlockIdx * singleM;
        uint32_t posK_in = in.pos.s[1];
        uint32_t posM_out = out.pos.s[0] + BlockIdx * singleM;
        uint32_t posK_out = out.pos.s[1];

        if (posM_in >= a8M_length) {//a8 complete
            return;
        }

        in.pos.setAxisName('M', 'K');
        in.vec.setAxisName('M', 'K');
        out.pos.setAxisName('M', 'K');
        out.vec.setAxisName('M', 'K');

        Unisor<GlobalMem, Dim2> inputUnisor(in);
        Unisor<GlobalMem, Dim2> outputHighUnisor(out);
        Unisor<GlobalMem, Dim2> outputLowUnisor(out);

        Dim2 posIn = Dim2(posM_in, posK_in, 'M', 'K');
        Dim2 vecIn = Dim2(current_M, singleK, 'M', 'K');
        inputUnisor.setCartesian(posIn, vecIn);

        Dim2 posHighOut = Dim2(posM_out, posK_out, 'M', 'K');
        Dim2 vecOut = Dim2(current_M, singleK, 'M', 'K');
        outputHighUnisor.setCartesian(posHighOut, vecOut);

        InternalRun_off(inputUnisor, outputHighUnisor, outputLowUnisor, baseTiling);
        this->pipe->Reset();
    }

    __aicore__ inline void InternalRun_off(Unisor<GlobalMem, Dim2> in, Unisor<GlobalMem, Dim2> outHigh, Unisor<GlobalMem, Dim2> outLow, Dim2 &baseTiling)
    {
        uint32_t posM_in = in.pos.getAxisByName('M');
        uint32_t posM_outHigh = outHigh.pos.getAxisByName('M');
        uint32_t posM_outLow = outLow.pos.getAxisByName('M');
        Dim1 posM(0, 'M');
        Dim1 vecM(in.vec.getAxisByName('M'), 'M');
        Cartesian<Dim1> cartesianM(posM, vecM);
        Dim1 baseM(baseTiling.getAxisByName('M'), 'M');
        CartesianIterator<Dim1> unisorIteratorM(cartesianM, baseM);
        Cartesian<Dim1> cartesianUnisorM;
        cartesianUnisorM.pos.setAxisName('M');
        cartesianUnisorM.vec.setAxisName('M');

        Dim1 posK(in.pos.getAxisByName('K'), 'K');
        Dim1 vecK(in.vec.getAxisByName('K'), 'K');
        Cartesian<Dim1> cartesianK(posK, vecK);
        Dim1 baseK(baseTiling.getAxisByName('K'), 'K');
        CartesianIterator<Dim1> unisorIteratorK(cartesianK, baseK);
        Cartesian<Dim1> cartesianUnisorK;
        cartesianUnisorK.pos.setAxisName('K');
        cartesianUnisorK.vec.setAxisName('K');

        Unisor<UBMem, Dim1> maskUnisor;
        Unisor<UBMem, Dim2> unisorIn;
        Unisor<UBMem, Dim2> highUnisor;
        Unisor<UBMem, Dim2> highInt4Unisor;
        Unisor<UBMem, Dim2> lowInt8Unisor;
        Unisor<UBMem, Dim2> lowUnisor;
        Unisor<UBMem, Dim2> lowInt4Unisor;
        {
            Dim1 maskDim(baseK.getAxisByName('K'), 'K');
            maskUnisor.init_RealShape(maskDim, INT8, this->pipe);
            Dim2 vecIn(baseM.getAxisByName('M'), baseK.getAxisByName('K'), 'M', 'K');
            unisorIn.init_RealShape(vecIn, INT8, this->pipe);
            highUnisor.init_RealShape(vecIn, FP16, this->pipe);
            highInt4Unisor.init_RealShape(vecIn, INT4, this->pipe);
            lowInt8Unisor.init_RealShape(vecIn, INT8, this->pipe);
            lowUnisor.init_RealShape(vecIn, FP16, this->pipe);
            lowInt4Unisor.init_RealShape(vecIn, INT4, this->pipe);
        }

        // Step 0: Prepare a 1D mask filled with 0x0fU, which will be used to
        // extract the lower 4 bits of each int8 of 'in'.
        NPU_Duplicate<uint16_t>(maskUnisor, 0x0f0fU);

        SetFlag<HardEvent::V_MTE2>(LOCAL_HWEVENT_ID0);
        SetFlag<HardEvent::MTE3_V>(LOCAL_HWEVENT_ID0);
        SetFlag<HardEvent::MTE3_V>(LOCAL_HWEVENT_ID1);
        while (unisorIteratorM.posIterator(cartesianUnisorM)) {
            while (unisorIteratorK.posIterator(cartesianUnisorK)) {
                Dim2 posIn(cartesianUnisorM.pos.getAxisByName('M') + posM_in,
                           cartesianUnisorK.pos.getAxisByName('K'),
                           'M', 'K');
                Dim2 vecIn(cartesianUnisorM.vec.getAxisByName('M'),
                           cartesianUnisorK.vec.getAxisByName('K'),
                           'M', 'K');
                Dim2 posHighOut(cartesianUnisorM.pos.getAxisByName('M') * FACTOR_2 + posM_in * FACTOR_2,
                               cartesianUnisorK.pos.getAxisByName('K'),
                               'M', 'K');
                Dim2 posLowOut(cartesianUnisorM.pos.getAxisByName('M') * FACTOR_2 + posM_in * FACTOR_2 + 1,
                               cartesianUnisorK.pos.getAxisByName('K'),
                               'M', 'K');

                Cartesian<Dim2> cartesianIn(posIn, vecIn);
                Cartesian<Dim2> cartesianHighOut(posHighOut, vecIn);
                Cartesian<Dim2> cartesianLowOut(posLowOut, vecIn);
                in.setCartesian(cartesianIn);
                outHigh.setCartesian(cartesianHighOut);
                outLow.setCartesian(cartesianLowOut);
                // Step 1: Load a unisor of input data into 'unisorIn'.
                WaitFlag<HardEvent::V_MTE2>(LOCAL_HWEVENT_ID0);
                NPU_Load(unisorIn, in);
                SetFlag<HardEvent::MTE2_V>(LOCAL_HWEVENT_ID0);
                WaitFlag<HardEvent::MTE2_V>(LOCAL_HWEVENT_ID0);

                //
                // Higher 4 bits
                //
                // Step 2: Cast 'unisorIn' to fp16, and save the results to 'highUnisor'.
                NPU_Cast<half, int8_t>(highUnisor, unisorIn);
                // Step 3: Divide 'highUnisor' by 1/16 element-wisely to extract
                // the higher 4 bits of each number.
                const half one_over_sixteen = 0.0625;
                NPU_Muls(highUnisor, highUnisor, one_over_sixteen);
                // Step 4: Cast 'highUnisor' to int4.
                WaitFlag<HardEvent::MTE3_V>(LOCAL_HWEVENT_ID0);
                NPU_Cast<int4b_t, half>(highInt4Unisor, highUnisor, AscendC::RoundMode::CAST_FLOOR);
                SetFlag<HardEvent::V_MTE3>(LOCAL_HWEVENT_ID0);
                WaitFlag<HardEvent::V_MTE3>(LOCAL_HWEVENT_ID0);
                // Step 5: Store 'highInt4Unisor' back to global memory.
                NPU_Store(outHigh, highInt4Unisor);
                SetFlag<HardEvent::MTE3_V>(LOCAL_HWEVENT_ID0);

                //
                // Lower 4 bits
                //
                // Step 6: Extract the lower 4 bits of each int8.
                NPU_And(lowInt8Unisor, unisorIn, maskUnisor);
                SetFlag<HardEvent::V_MTE2>(LOCAL_HWEVENT_ID0);
                // Step 7: Cast 'lowInt8Unisor' to half.
                NPU_Cast<half, int8_t>(lowUnisor, lowInt8Unisor);
                // Step 8: Subtract 8 from 'lowUnisor'.
                const half neg_eight = -8.0;
                NPU_Adds(lowUnisor, lowUnisor, neg_eight);
                // Step 9: Cast 'lowUnisor' to int4.
                WaitFlag<HardEvent::MTE3_V>(LOCAL_HWEVENT_ID1);
                NPU_Cast<int4b_t, half>(lowInt4Unisor, lowUnisor);
                SetFlag<HardEvent::V_MTE3>(LOCAL_HWEVENT_ID1);
                WaitFlag<HardEvent::V_MTE3>(LOCAL_HWEVENT_ID1);
                // Step 10: Store 'lowInt4Unisor' back to global memory.
                NPU_Store(outLow, lowInt4Unisor);
                SetFlag<HardEvent::MTE3_V>(LOCAL_HWEVENT_ID1);
            }
        }
        WaitFlag<HardEvent::MTE3_V>(LOCAL_HWEVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(LOCAL_HWEVENT_ID1);
        WaitFlag<HardEvent::V_MTE2>(LOCAL_HWEVENT_ID0);
    }
};

#define MAX_GROUP_LEN 256

__aicore__ inline void dynamic_unisor_programming(GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmC, GM_ADDR gmGrpList,
                                                  GM_ADDR gmBias, GM_ADDR gmOffset, GM_ADDR gmSa, GM_ADDR gmSw,
                                                  GM_ADDR gmWorkspaceDevice, A8W4HPTiling *tiling_data,
                                                  TPipe *pipe)
{
    auto ori_in0_shape = tiling_data->ori_in0_shape;
    auto ori_in1_shape = tiling_data->ori_in1_shape;
    auto ori_out_shape = tiling_data->ori_out_shape;
    auto single_core_tiling = tiling_data->single_core_tiling;
    auto single_core_base_tiling = tiling_data->single_core_base_tiling;
    AxisTiling3Vector splitRecord{tiling_data->splitRecord[ARRAY_IDX0], tiling_data->splitRecord[ARRAY_IDX1],
                                  tiling_data->splitRecord[ARRAY_IDX2]};

    KernelMatmul op;
    op.Init(pipe);
    op.initBlockPos_3D(splitRecord);
    uint32_t numAic = tiling_data->numAic;
    uint32_t numAiv = tiling_data->numAiv;
    uint32_t pattern = tiling_data->pattern;
    op.pattern = pattern;
    op.nCoreAIC = numAic;
    op.nCoreAIV = numAiv;
    op.szMemUB = tiling_data->szUb;
    op.szMemL0AB = tiling_data->szL0A;
    op.szMemL0C = tiling_data->szL0C;
    op.pattern = pattern;
    op.output_type = tiling_data->output_type;

    uint32_t blockIdx = GetBlockIdx();//当前核ID
    uint32_t blockId = 0;

    if (g_coreType == AscendC::AIC) {
        blockId = blockIdx;
    }

    if (g_coreType == AscendC::AIV) {
        blockId = blockIdx / FACTOR_2;
    }
    op.blockId = blockId;

    //GroupMatmul参数
    uint32_t group_type = tiling_data->group_type;
    uint32_t group_num = tiling_data->group_num;

    //Group总长度
    uint32_t total_M = ori_out_shape[0];               // 16384
    uint32_t total_N = ori_out_shape[1];               // 4096
    uint32_t total_K_A = ori_in0_shape[1];             // 7168
    uint32_t total_K_B = ori_in0_shape[1] * group_num; // 512

    //单个矩阵长度
    int64_t M_Lengths[MAX_GROUP_LEN];
    int64_t N_Length = total_N;

    //单核迭代长度
    uint32_t single_M = single_core_tiling[ARRAY_IDX0];
    uint32_t single_N = single_core_tiling[ARRAY_IDX1];
    uint32_t single_K = single_core_tiling[ARRAY_IDX2];

    //Base迭代长度
    Dim2 baseCTiling(single_core_base_tiling[0], single_core_base_tiling[1], 'M', 'N');

    float format_in = tiling_data->format_in;
    float format_out = tiling_data->format_out;

    Dim2 vecC(total_M / FACTOR_2, total_N);
    Dim2 vecD(total_M, total_N);
    Dim2 vecB;

    if (pattern == 1) {//B不转置
        vecB = Dim2(total_K_B, total_N);
    } else {//B转置
        vecB = Dim2(total_N, total_K_B);
    }

    Dim2 vecA(total_M / FACTOR_2, total_K_A);
    Dim2 vecA_Work(total_M, total_K_A);
    Dim2 vec_bias(group_num, total_N);
    Dim2 vec_offset(group_num, total_N);
    Dim1 vec_sa(total_M / FACTOR_2);
    Dim2 vec_sw(group_num, total_N);
    Dim2 vecWork(single_M, single_N);
    Dim1 vecWorkRow(total_M / FACTOR_2);
    // workspaceSize的格式依次是：GMM的输出，a8转a4的空间，软同步的空间，offset的空间
    uint32_t pingOffset = (blockId * GmmWorkSpaceAmount) * INT32;
    Unisor<GlobalMem, Dim2> workUnisorPing((gmWorkspaceDevice + pingOffset), vecWork, INT32);

    uint32_t Temp_A_Offset = numAic * GmmWorkSpaceAmount * INT32;
    Unisor<GlobalMem, Dim2> inputA_workUnisor(gmWorkspaceDevice + Temp_A_Offset, vecA_Work, INT4);

    uint32_t SyncOffset = Temp_A_Offset + (ceilINT_16(total_M) * total_K_A / FACTOR_2 * sizeof(uint8_t));
    GlobalTensor<uint64_t> SyncUnisor;
    SyncUnisor.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(gmWorkspaceDevice + SyncOffset), ceilINT_16(total_M) * SoftwareWorkSpaceEle / sizeof(uint64_t));

    uint32_t rowOffset = SyncOffset + ceilINT_16(total_M) * SoftwareWorkSpaceEle;
    Unisor<GlobalMem, Dim1> RowsumUnisor((gmWorkspaceDevice + rowOffset), vecWorkRow, FP32, true);
    Unisor<GlobalMem, Dim2> inputAUnisor(gmA, vecA, INT8);
    Unisor<GlobalMem, Dim2> inputBUnisor(gmB, vecB, format_in);
    Unisor<GlobalMem, Dim2> outputCUnisor(gmC, vecC, format_out);
    Unisor<GlobalMem, Dim2> biasUnisor(gmBias, vec_bias, INT32, true);
    Unisor<GlobalMem, Dim2> offsetUnisor(gmOffset, vec_offset, FP32, true);
    Unisor<GlobalMem, Dim1> saUnisor(gmSa, vec_sa, FP32, true);
    Unisor<GlobalMem, Dim2> swUnisor(gmSw, vec_sw, FP32, true);

    uint32_t work_count = 0;

    inputAUnisor.pos.setAxisName('M', 'K');
    inputAUnisor.vec.setAxisName('M', 'K');
    inputAUnisor.vector_val.setAxisName('M', 'K');

    inputA_workUnisor.pos.setAxisName('M', 'K');
    inputA_workUnisor.vec.setAxisName('M', 'K');
    inputA_workUnisor.vector_val.setAxisName('M', 'K');

    //B不转置
    if (pattern == 1) {
        inputBUnisor.pos.setAxisName('K', 'N');
        inputBUnisor.vec.setAxisName('K', 'N');
        inputBUnisor.vector_val.setAxisName('K', 'N');
    } else {
        //B转置
        inputBUnisor.pos.setAxisName('N', 'K');
        inputBUnisor.vec.setAxisName('N', 'K');
        inputBUnisor.vector_val.setAxisName('N', 'K');
    }

    outputCUnisor.pos.setAxisName('M', 'N');
    outputCUnisor.vec.setAxisName('M', 'N');
    outputCUnisor.vector_val.setAxisName('M', 'N');

    workUnisorPing.pos.setAxisName('M', 'N');
    workUnisorPing.vec.setAxisName('M', 'N');
    workUnisorPing.vector_val.setAxisName('M', 'N');

    uint8_t required_core_num = tiling_data->required_core_num;
    uint8_t kernel_index = tiling_data->kernel_index;
    uint32_t splitTimes = tiling_data->splitTimes;
    uint32_t core_id = 0;

    GlobalTensor<int64_t> groupListGm;
    groupListGm.SetGlobalBuffer((__gm__ int64_t *)gmGrpList);

    for (uint32_t group_idx = 0; group_idx < group_num; group_idx++) {
        if (group_type == 0) { // 从m轴方向切割
        int64_t mSizeInGroup = static_cast<int64_t>(groupListGm.GetValue(group_idx));
        M_Lengths[group_idx] = mSizeInGroup * FACTOR_2;
        }
    }

    uint32_t posInOffset = 0, posOutOffset = 0;
    uint32_t pos_M_Offset = 0;
    // true：只有硬同步； false：硬 + 软同步混合;  我们可以将该变量加入到tiling_data中，这样可以由tiling自动选择是否开启软同步；default：硬 + 软同步混合
    bool hardSyncFlag = true;
    uint32_t hardSyncRounds = op.ceilINT(HardWareSyncAmount, single_M * numAic); // 硬 + 软同步混合中 硬同步的计算量
    uint32_t preComputeAmount = single_M * numAic * hardSyncRounds;

    if (hardSyncFlag) {
        preComputeAmount = total_M / FACTOR_2;
    }

    if (g_coreType == AscendC::AIV) {
        uint32_t mIn = preComputeAmount;
        mIn = mIn > (total_M / FACTOR_2) ? (total_M / FACTOR_2) : mIn;
        uint32_t mOut = mIn * FACTOR_2;
        Dim2 vecIn(mIn, total_K_A);
        Dim2 vecOut(mOut, total_K_A);
        Dim2 posIn(0, 0);
        Dim2 posOut(0, 0);

        inputAUnisor.setCartesian(posIn, vecIn);
        inputA_workUnisor.setCartesian(posOut, vecOut); //两倍关系
        Dim1 posRowsum(0);
        Dim1 vecRowsum(mIn);
        RowsumUnisor.setCartesian(posRowsum, vecRowsum);
        if (op.offset_enable) {
            return;
        } else {
            op.npu_user_defined_matmul_kernel_slave_Single_A8ToA4_off(inputAUnisor, inputA_workUnisor);
        }

        SetFlag<HardEvent::MTE3_S>(LOCAL_HWEVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(LOCAL_HWEVENT_ID0);

        posInOffset = mIn;
        posOutOffset = mOut;

        SyncUnisor.SetValue(blockIdx * INT64, posOutOffset);
        AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(SyncUnisor[blockIdx * INT64]);
        AscendC::PipeBarrier<PIPE_ALL>();

        AscendC::SyncAll<false>();
    }

    if (g_coreType == AscendC::AIC) {
        AscendC::SyncAll<false>();
    }

    if (g_coreType == AscendC::AIV) {
        AscendC::CrossCoreSetFlag<LOCAL_MODEID2, PIPE_MTE3>(LOCAL_FLAGID7);
    }

    //Cube计算 + 后处理
    for (int32_t i=0; i<group_num; i++) {
        auto M_Length = M_Lengths[i];

        uint32_t pos_M = pos_M_Offset;
        uint32_t pos_N = 0;
        uint32_t pos_K_group = i * total_K_A;
        uint32_t pos_N_group = i * total_N;

        pos_M_Offset += M_Length;

        Dim1 C0pos(pos_M, 'M'), C0vec(M_Length, 'M');
        Cartesian<Dim1> multiCoreCartesianC0(C0pos, C0vec);
        Dim1 singleCoreCTilingS0(single_M, 'M');
        CartesianIterator<Dim1> multiCoreUnisorIterator0(multiCoreCartesianC0, singleCoreCTilingS0);
        Cartesian<Dim1> multiCoreCartesianUnisorC0;

        Dim1 C1pos(pos_N, 'N'), C1vec(N_Length, 'N');
        Cartesian<Dim1> multiCoreCartesianC1(C1pos, C1vec);
        Dim1 singleCoreCTilingS1(single_N, 'N');
        CartesianIterator<Dim1> multiCoreUnisorIterator1(multiCoreCartesianC1, singleCoreCTilingS1);
        Cartesian<Dim1> multiCoreCartesianUnisorC1;

        while (multiCoreUnisorIterator0.posIterator(multiCoreCartesianUnisorC0)) {
            bool once_flag = false;
            while (multiCoreUnisorIterator1.posIterator(multiCoreCartesianUnisorC1)) {
                if ((core_id % numAic) == blockId) {
                    //前处理
                    if (g_coreType == AscendC::AIV && !once_flag && !hardSyncFlag) {
                        int mIn = single_M / FACTOR_2;
                        int mOut = single_M;
                        Dim2 vecIn(mIn, total_K_A);
                        Dim2 vecOut(mOut, total_K_A);
                        Dim2 posIn(posInOffset, 0);
                        Dim2 posOut(posOutOffset, 0);

                        inputAUnisor.setCartesian(posIn, vecIn);
                        inputA_workUnisor.setCartesian(posOut, vecOut); //两倍关系
                        Dim1 posRowsum(posInOffset);
                        Dim1 vecRowsum(mIn);
                        RowsumUnisor.setCartesian(posRowsum, vecRowsum);
                        if (op.offset_enable) {
                            return;
                        } else {
                            op.npu_user_defined_matmul_kernel_slave_Double_A8ToA4_off(inputAUnisor, inputA_workUnisor, total_M / FACTOR_2);
                        }
                        SetFlag<HardEvent::MTE3_S>(LOCAL_HWEVENT_ID0);
                        WaitFlag<HardEvent::MTE3_S>(LOCAL_HWEVENT_ID0);

                        //软同步
                        if (blockIdx % FACTOR_2 != 0) {
                        SyncUnisor.SetValue(blockIdx * INT64, posOutOffset + mOut / FACTOR_2);
                        } else {
                        SyncUnisor.SetValue(blockIdx * INT64, posOutOffset + mOut);
                        }
                        AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(SyncUnisor[blockIdx * INT64]);
                        AscendC::PipeBarrier<PIPE_ALL>();

                        once_flag = true;
                    }

                    Dim2 posA = Dim2(multiCoreCartesianUnisorC0.pos.getAxisByName('M'), 0, 'M', 'K');
                    Dim2 vecA = Dim2(multiCoreCartesianUnisorC0.vec.getAxisByName('M'), single_K, 'M', 'K');
                    inputA_workUnisor.setCartesian(posA, vecA);

                    //B不转置
                    if (pattern == 1) {
                        Dim2 posB = Dim2(pos_N_group, multiCoreCartesianUnisorC1.pos.getAxisByName('N'), 'K', 'N');
                        Dim2 vecB = Dim2(single_K, multiCoreCartesianUnisorC1.vec.getAxisByName('N'), 'K', 'N');
                        inputBUnisor.vector_val.s[0] = total_K_A;
                        inputBUnisor.setCartesian(posB, vecB);
                    } else {
                        Dim2 posB = Dim2(multiCoreCartesianUnisorC1.pos.getAxisByName('N'), pos_K_group, 'N', 'K');
                        Dim2 vecB = Dim2(multiCoreCartesianUnisorC1.vec.getAxisByName('N'), single_K, 'N', 'K');
                        inputBUnisor.setCartesian(posB, vecB);
                    }

                    Dim2 posC(multiCoreCartesianUnisorC0.pos.getAxisByName('M'), multiCoreCartesianUnisorC1.pos.getAxisByName('N'), 'M', 'N');
                    Dim2 vecC(multiCoreCartesianUnisorC0.vec.getAxisByName('M'), multiCoreCartesianUnisorC1.vec.getAxisByName('N'), 'M', 'N');
                    outputCUnisor.setCartesian(posC, vecC);
                    workUnisorPing.setCartesian(posC, vecC);
                    bool isEven = (work_count % FACTOR_2 == 0);
                    work_count++;

                    //Cube核
                    if (g_coreType == AscendC::AIC) {
                        bool sync_flag = hardSyncFlag;
                        uint8_t hit_count = 0;
                        uint64_t index = 0;
                        uint64_t aic_current_field = multiCoreCartesianUnisorC0.pos.getAxisByName('M') + single_M;
                        if (aic_current_field > total_M ) {
                            aic_current_field = total_M;
                        }
                        while(!sync_flag) {
                            for (int sync_index = 0; sync_index < numAiv; sync_index++) {
                                index = (uint64_t)SyncUnisor.GetValue(sync_index * INT64);
                                if (index >= aic_current_field) {
                                    hit_count++;
                                }
                                if(hit_count == numAiv){
                                    sync_flag = true;
                                }
                            }
                            if (sync_flag) {
                                break;
                            }

                            for (int sync_index = 0; sync_index < numAiv; sync_index++) {
                                AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(SyncUnisor[sync_index * FACTOR_64]);
                                index = (uint64_t)SyncUnisor.GetValue(sync_index * INT64);
                                if (index >= aic_current_field) {
                                    hit_count++;
                                }
                                if(hit_count == numAiv){
                                    sync_flag = true;
                                }
                            }
                        }
                    }
                    op.npu_user_defined_matmul_kernel_switch(inputA_workUnisor, inputBUnisor, outputCUnisor, workUnisorPing, RowsumUnisor, biasUnisor, offsetUnisor, saUnisor, swUnisor, baseCTiling, i, kernel_index);
                }
                core_id++;
            }
            posInOffset += single_M / FACTOR_2;
            posOutOffset += single_M;
        }
    }

    if (g_coreType == AscendC::AIC) {
        AscendC::CrossCoreWaitFlag(LOCAL_FLAGID7);
        SetFlag<HardEvent::MTE3_S>(LOCAL_HWEVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(LOCAL_HWEVENT_ID0);
        M_Lengths[0] = 0; //没有意义，只是表达有一条scalar指令
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

class GMMA4W8AutotilingCompute {
private:
    TPipe *pipe;
    GM_ADDR gmA;
    GM_ADDR gmB;
    GM_ADDR gmC;
    GM_ADDR gmGrpListOptional;
    GM_ADDR gmBias;
    GM_ADDR gmOffset;
    GM_ADDR gmSa;
    GM_ADDR gmSw;
    GM_ADDR gmWorkspaceDevice;
    A8W4HPTiling *tiling_data;

public:
    __aicore__ inline GMMA4W8AutotilingCompute(GM_ADDR addrA, GM_ADDR addrB, GM_ADDR addrC, GM_ADDR addrGrpList,
                                               GM_ADDR addrBias, GM_ADDR addrOffset, GM_ADDR addrSa, GM_ADDR addrSw,
                                               GM_ADDR addrWorkspace, A8W4HPTiling *tilingData, TPipe *p)
        : pipe(p), gmA(addrA), gmB(addrB), gmC(addrC), gmGrpListOptional(addrGrpList), gmBias(addrBias),
          gmOffset(addrOffset), gmSa(addrSa), gmSw(addrSw), gmWorkspaceDevice(addrWorkspace), tiling_data(tilingData)
    {
    }

    __aicore__ inline void Init()
    {
        gmA = GetTensorAddr<uint8_t>(0, gmA);
        gmB = GetTensorAddr<uint8_t>(0, gmB);
        gmC = GetTensorAddr<uint8_t>(0, gmC);
        gmBias = GetTensorAddr<uint8_t>(0, gmBias);
        gmOffset = GetTensorAddr<uint8_t>(0, gmOffset);
        gmSw = GetTensorAddr<uint8_t>(0, gmSw);
    }

    __aicore__ inline void Process()
    {
        dynamic_unisor_programming(gmA, gmB, gmC, gmGrpListOptional, gmBias, gmOffset, gmSa, gmSw, gmWorkspaceDevice,
                                   tiling_data, pipe);
    }
};
}

using GMMHighPerf::GMMA4W8AutotilingCompute;
}
#endif
