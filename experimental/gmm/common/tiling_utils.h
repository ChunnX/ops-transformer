/*!
 * \file utils.h
 * \brief Common utility functions and structures for operator tiling.
 */

#pragma once

#include <cstdint>

namespace optiling {

constexpr uint32_t INVERSE_FIFTEEN = ~15;

inline uint32_t SixteenAlign(uint32_t a) {
  return a & INVERSE_FIFTEEN;
}

template <typename T>
static T AlignUp(T num1, T num2) {
  if (num2 == 0) {
    return 0;
  }
  return (num1 + num2 - 1) / num2 * num2;
}

template <typename T>
static T AlignDown(T num1, T num2) {
  if (num2 == 0) {
    return 0;
  }
  return num1 / num2 * num2;
}

template <typename T>
static T CeilDivision(T num1, T num2) {
  if (num2 == 0) {
    return 0;
  }
  return (num1 + num2 - 1) / num2;
}

template <typename T1, typename T2>
static T1 CeilDiv(const T1 n1, const T2 n2) {
  if (n1 == 0) {
    return 0;
  }
  return (n2 != 0) ? (((n1 - 1) / n2) + 1) : n1;
}


template <typename T>
static T CalcTailSize(T num1, T num2) {
  if (num2 == 0) {
    return 0;
  }
  T mod = num1 % num2;
  return mod != 0 ? mod : num2;
}

struct AiCoreParams {
  uint64_t ubSize;
  uint64_t blockDim;
  uint64_t aicNum;
  uint64_t l1Size;
  uint64_t l0aSize;
  uint64_t l0bSize;
  uint64_t l0cSize;
};

} // namespace optiling