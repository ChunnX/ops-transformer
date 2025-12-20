/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#ifndef INCLUDE_CATLASS_ARCH_MEMORY_H
#define INCLUDE_CATLASS_ARCH_MEMORY_H

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"

namespace Catlass::Arch {

struct LocalTensorBufferBase {
public:
    template <class Element = half>
    CATLASS_DEVICE
    AscendC::LocalTensor<Element> GetBufferByByte(const uint32_t offset) const
    {
        return tensor[offset].template ReinterpretCast<Element>();
    }

protected:
    CATLASS_DEVICE
    LocalTensorBufferBase() = default;

    AscendC::LocalTensor<uint8_t> tensor;
};

template <
    class ArchTag,
    AscendC::TPosition Position
>
struct LocalTensorBuffer {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported local tensor buffer, can not find the specialization.");
};

/// Partial specialization for TPosition::A1
template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::A1> : LocalTensorBufferBase {
public:
    static constexpr AscendC::TPosition Position = AscendC::TPosition::A1;

    CATLASS_DEVICE
    LocalTensorBuffer()
    {
        tensor = AscendC::LocalTensor<uint8_t>(Position, 0, ArchTag::L1_SIZE);
    }
};

///////////////////////////////////////////////////////////

/// Partial specialization for TPosition::A2
template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::A2> : LocalTensorBufferBase {
public:
    static constexpr AscendC::TPosition Position = AscendC::TPosition::A2;

    CATLASS_DEVICE
    LocalTensorBuffer()
    {
        tensor = AscendC::LocalTensor<uint8_t>(Position, 0, ArchTag::L0A_SIZE);
    }
};

///////////////////////////////////////////////////////////

/// Partial specialization for TPosition::B1
template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::B1> : LocalTensorBufferBase {
public:
    static constexpr AscendC::TPosition Position = AscendC::TPosition::B1;

    CATLASS_DEVICE
    LocalTensorBuffer()
    {
        tensor = AscendC::LocalTensor<uint8_t>(Position, 0, ArchTag::L1_SIZE);
    }
};

///////////////////////////////////////////////////////////

/// Partial specialization for AtlasA2, TPosition::B2
template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::B2> : LocalTensorBufferBase {
public:
    static constexpr AscendC::TPosition Position = AscendC::TPosition::B2;

    CATLASS_DEVICE
    LocalTensorBuffer()
    {
        tensor = AscendC::LocalTensor<uint8_t>(Position, 0, ArchTag::L0B_SIZE);
    }
};

///////////////////////////////////////////////////////////

/// Partial specialization for AtlasA2, TPosition::C1
template <>
struct LocalTensorBuffer<Arch::AtlasA2, AscendC::TPosition::C1> : LocalTensorBufferBase {
public:
    using ArchTag = Arch::AtlasA2;
    static constexpr AscendC::TPosition Position = AscendC::TPosition::C1;

    CATLASS_DEVICE
    LocalTensorBuffer()
    {
        tensor = AscendC::LocalTensor<uint8_t>(Position, 0, ArchTag::L1_SIZE);
    }
};

///////////////////////////////////////////////////////////

/// Partial specialization for AtlasA2, TPosition::C2
template <>
struct LocalTensorBuffer<Arch::AtlasA2, AscendC::TPosition::C2> : LocalTensorBufferBase {
public:
    using ArchTag = Arch::AtlasA2;
    static constexpr AscendC::TPosition Position = AscendC::TPosition::C2;

    CATLASS_DEVICE
    LocalTensorBuffer()
    {
        tensor = AscendC::LocalTensor<uint8_t>(Position, 0, ArchTag::BIAS_SIZE);
    }
};

///////////////////////////////////////////////////////////

/// Partial specialization for TPosition::CO1
template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::CO1> : LocalTensorBufferBase {
public:
    static constexpr AscendC::TPosition Position = AscendC::TPosition::CO1;

    CATLASS_DEVICE
    LocalTensorBuffer()
    {
        tensor = AscendC::LocalTensor<uint8_t>(Position, 0, ArchTag::L0C_SIZE);
    }
};

///////////////////////////////////////////////////////////

/// Partial specialization for AtlasA2, TPosition::C2PIPE2GM
template <>
struct LocalTensorBuffer<Arch::AtlasA2, AscendC::TPosition::C2PIPE2GM> : LocalTensorBufferBase {
public:
    using ArchTag = Arch::AtlasA2;
    static constexpr AscendC::TPosition Position = AscendC::TPosition::C2PIPE2GM;

    CATLASS_DEVICE
    LocalTensorBuffer()
    {
        tensor = AscendC::LocalTensor<uint8_t>(Position, 0, ArchTag::FIXBUF_SIZE);
    }
};

///////////////////////////////////////////////////////////

/// Partial specialization for TPosition::VECIN
template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::VECIN> : LocalTensorBufferBase {
public:
    static constexpr AscendC::TPosition Position = AscendC::TPosition::VECIN;

    CATLASS_DEVICE
    LocalTensorBuffer()
    {
        tensor = AscendC::LocalTensor<uint8_t>(Position, 0, ArchTag::UB_SIZE);
    }
};

///////////////////////////////////////////////////////////

/// Partial specialization for TPosition::VECOUT
template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::VECOUT> : LocalTensorBufferBase {
public:
    static constexpr AscendC::TPosition Position = AscendC::TPosition::VECOUT;

    CATLASS_DEVICE
    LocalTensorBuffer()
    {
        tensor = AscendC::LocalTensor<uint8_t>(Position, 0, ArchTag::UB_SIZE);
    }
};

///////////////////////////////////////////////////////////

/// Partial specialization for TPosition::VECCALC
template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::VECCALC> : LocalTensorBufferBase {
public:
    static constexpr AscendC::TPosition Position = AscendC::TPosition::VECCALC;

    CATLASS_DEVICE
    LocalTensorBuffer()
    {
        tensor = AscendC::LocalTensor<uint8_t>(Position, 0, ArchTag::UB_SIZE);
    }
};

}  // namespace Catlass::Arch

#endif  // INCLUDE_CATLASS_ARCH_MEMORY_H