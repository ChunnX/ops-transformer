/*
 * compressor_grad operator declarations
 */

#ifndef OPS_H
#define OPS_H

#include <torch/extension.h>

namespace ascend_kernel {

std::vector<at::Tensor> compressor_grad(
    const at::Tensor &x,
    const at::Tensor &wkv,
    const at::Tensor &wgate,
    const at::Tensor &dCmpKv,
    const at::Tensor &softmaxScore,
    const at::Tensor &kv,
    int64_t cmpRatio,
    int64_t coff);

}  // namespace ascend_kernel

#endif  // OPS_H
