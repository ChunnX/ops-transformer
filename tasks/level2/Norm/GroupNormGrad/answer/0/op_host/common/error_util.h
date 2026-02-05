/**
 * Minimal error_util for tiling: OP_TILING_CHECK only.
 */
#ifndef COMMON_INC_ERROR_UTIL_H_
#define COMMON_INC_ERROR_UTIL_H_

#include "log.h"

#define OP_TILING_CHECK(cond, log_func, expr) \
  do {                                        \
    if (cond) {                               \
      log_func;                               \
      expr;                                   \
    }                                         \
  } while (0)

#endif
