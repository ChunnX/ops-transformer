#pragma once
#include "kernel_operator.h"
enum class CubeFormat { ND };
namespace AscendC {
enum class TPosition { GM, VECIN };
template<TPosition, CubeFormat, class Type, bool = false> struct MatmulType { using T = Type; };
}
