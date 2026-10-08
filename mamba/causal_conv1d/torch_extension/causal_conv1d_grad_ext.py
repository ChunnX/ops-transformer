# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ---------------------------------------------------------------------------
"""causal_conv1d 反向算子 PyTorch 扩展 + autograd 绑定。

- 编译加载：torch.utils.cpp_extension.load 编译 csrc/causal_conv1d_grad_ext.cpp，
  include/ldflags 与 cann_ops_transformer.op_builder.OpBuilder 保持一致。
- 算子调用：csrc 内通过 aclnn_common.h 的 ACLNN_CMD 宏走 aclnnCausalConv1dGrad
  两段式接口（GetWorkspaceSize + Run）；符号从 ASCEND_CUSTOM_OPP_PATH 指定的
  custom_opp/vendors/causal_conv1d_grad_custom/op_api/lib/libcust_opapi.so 动态解析。
- autograd 绑定：CausalConv1dFnAutoGrad 把前向 causal_conv1d_fn 与本反向算子串联。

运行前置条件：进程环境需设置（必须在首次调用算子前）：
  export ASCEND_CUSTOM_OPP_PATH=<repo>/operators/causal_conv1d_grad/custom_opp/vendors/causal_conv1d_grad_custom
"""
import importlib
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# 环境对齐（必须在 import torch_extension.ca l..._fn / cann_ops_transformer 之前）：
# 1. 源码仓 torch_extension 下的 cann_ops_transformer 含完整 op_builder/__init__.py，
#    site-packages 安装副本缺失该文件（`from cann_ops_transformer.op_builder import
#    OpBuilder` 会失败），因此把源码仓路径插到 sys.path 最前。
# ---------------------------------------------------------------------------
_SRC_TORCH_EXT = _HERE.parents[2] / "torch_extension"  # ops-transformer/torch_extension
if not (_SRC_TORCH_EXT / "cann_ops_transformer" / "op_builder" / "__init__.py").exists():
    raise FileNotFoundError(
        f"cann_ops_transformer source package not found under {_SRC_TORCH_EXT}"
    )
if str(_SRC_TORCH_EXT) not in sys.path:
    sys.path.insert(0, str(_SRC_TORCH_EXT))

import torch  # noqa: E402
import torch_npu  # noqa: E402
from torch.utils.cpp_extension import load  # noqa: E402

# 注意：torch_extension/__init__.py 中 `from .causal_conv1d_fn import causal_conv1d_fn`
# 会把包属性 causal_conv1d_fn 覆盖为函数，因此必须用 importlib 取模块对象。
_fwd_mod = importlib.import_module(f"{__package__}.causal_conv1d_fn")
causal_conv1d_fn = _fwd_mod.causal_conv1d_fn

# ---------------------------------------------------------------------------
# 2. mamba 副本 causal_conv1d_fn.py 的 sources() 返回 "csrc/mamba/..."，相对
#    cann_ops_transformer 包根解析；源码仓包根无 csrc/mamba/ 目录（打包安装时布局
#    重排），这里 patch 为 mamba/causal_conv1d/torch_extension/csrc 下实际存在的
#    源文件（get_absolute_paths 对绝对路径直接透传）。
# ---------------------------------------------------------------------------
_FWD_CPP = _HERE / "csrc" / "causal_conv1d_fn.cpp"
if not _FWD_CPP.exists():
    raise FileNotFoundError(f"forward csrc not found: {_FWD_CPP}")
_fwd_mod.CausalConv1dFnOpBuilder.sources = lambda self: [str(_FWD_CPP)]

# aclnn_common.h 所在目录：ops-transformer 仓 torch_extension/cann_ops_transformer/common
_ACLNN_COMMON_DIR = _SRC_TORCH_EXT / "cann_ops_transformer" / "common"
if not (_ACLNN_COMMON_DIR / "aclnn_common.h").exists():
    raise FileNotFoundError(
        f"aclnn_common.h not found at {_ACLNN_COMMON_DIR}; "
        "this module must live inside the ops-transformer source tree"
    )

_CANN_PATH = None
for _env in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
    _p = os.environ.get(_env)
    if _p and Path(_p).exists():
        _CANN_PATH = _p
        break
if _CANN_PATH is None:
    raise RuntimeError("ASCEND_HOME_PATH/ASCEND_TOOLKIT_HOME not set; source CANN env first")

_TORCH_NPU_PATH = Path(torch_npu.__file__).resolve().parent


def _preload_libopapi_global():
    """以 RTLD_GLOBAL 预加载 CANN libopapi.so（必须在首次调用反向算子前执行）。

    libcust_opapi.so 含未定义的 l0op::* 符号（如 l0op::Contiguous，定义于
    libopapi.so），且其 DT_NEEDED 不引用 libopapi.so；若进程全局作用域中没有
    libopapi.so，aclnn_common.h 对 libcust_opapi.so 的 dlopen(RTLD_LAZY) 后无法
    解析这些符号，表现为 aclnnCausalConv1dGrad 符号查找失败。
    注意：glibc 对已加载的 .so 不升级 RTLD_GLOBAL；若运行前 libopapi.so 已被
    其他组件以 RTLD_LOCAL 加载，本函数无效，需改用 LD_PRELOAD=libopapi.so。
    """
    import ctypes

    cann_lib = Path(_CANN_PATH) / "lib64" / "libopapi.so"
    if not cann_lib.exists():
        raise FileNotFoundError(f"libopapi.so not found at {cann_lib}")
    ctypes.CDLL(str(cann_lib), mode=os.RTLD_LAZY | os.RTLD_GLOBAL)


_preload_libopapi_global()


def _cxx_args():
    args = [
        "-fstack-protector-all",
        "-Wl,-z,relro,-z,now,-z,noexecstack",
        "-fPIC",
        "-pie",
        "-s",
        "-fvisibility=hidden",
        "-D_FORTIFY_SOURCE=2",
        "-O2",
        "-w",
    ]
    args.append("-D_GLIBCXX_USE_CXX11_ABI=" + ("1" if torch._C._GLIBCXX_USE_CXX11_ABI else "0"))
    return args


def _build_extension():
    return load(
        name="causal_conv1d_grad_ext",
        sources=[str(_HERE / "csrc" / "causal_conv1d_grad_ext.cpp")],
        extra_include_paths=[
            str(_TORCH_NPU_PATH / "include"),
            str(_TORCH_NPU_PATH / "include" / "third_party" / "hccl" / "inc"),
            str(_TORCH_NPU_PATH / "include" / "third_party" / "acl" / "inc"),
            str(Path(_CANN_PATH) / "include"),
            str(_ACLNN_COMMON_DIR),
        ],
        extra_cflags=_cxx_args(),
        extra_ldflags=[
            "-L" + str(Path(_CANN_PATH) / "lib64"),
            "-lascendcl",
            "-L" + str(_TORCH_NPU_PATH / "lib"),
            "-ltorch_npu",
        ],
        verbose=True,
    )


_grad_ext = None


def _get_grad_ext():
    global _grad_ext
    if _grad_ext is None:
        _grad_ext = _build_extension()
    return _grad_ext


def causal_conv1d_grad(grad_y: torch.Tensor, x: torch.Tensor, weight: torch.Tensor,
                       bias: torch.Tensor):
    """调用自定义反向算子 aclnnCausalConv1dGrad。

    Args:
        grad_y: 上游梯度 [B, T, D]
        x: 前向输入 [B, T, D]
        weight: 卷积权重 [kW, D]
        bias: 偏置 [D]

    Returns:
        (dx [B,T,D], dweight [kW,D], dbias [D])
    """
    return _get_grad_ext().causal_conv1d_grad(grad_y, x, weight, bias)


class CausalConv1dFnAutoGrad(torch.autograd.Function):
    """causal_conv1d_fn 的 autograd 绑定：forward 走已验证的前向算子，backward 走
    causal_conv1d_grad 自定义反向算子。

    约束（与反向算子能力对齐）：
      - 输入 x 为 3D 固定 batch [B, T, D]，等长序列（query_start_loc 均匀切分）
      - conv_states 冷启动全零、has_initial_state 未启用
      - bias 必须提供（反向算子需要 bias 计算 silu'）
      - activation 仅支持 "silu"
    """

    @staticmethod
    def forward(ctx, x, weight, bias, conv_states, query_start_loc, activation):
        if bias is None:
            raise ValueError("CausalConv1dFnAutoGrad requires a non-None bias "
                             "(causal_conv1d_grad needs bias to recompute silu')")
        if activation != "silu":
            raise ValueError(f"CausalConv1dFnAutoGrad only supports activation='silu', got '{activation}'")
        ctx.save_for_backward(x.detach(), weight.detach(), bias.detach())
        y = causal_conv1d_fn(
            x.detach(), weight.detach(), bias.detach(), conv_states, query_start_loc,
            activation=activation,
        )
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, weight, bias = ctx.saved_tensors
        if grad_y.dim() != 3:
            raise RuntimeError(f"grad_y must be 3D [B,T,D], got {tuple(grad_y.shape)}")
        grad_y = grad_y.contiguous()
        dx, dweight, dbias = causal_conv1d_grad(grad_y, x, weight, bias)
        return dx, dweight, dbias, None, None, None


def causal_conv1d_fn_autograd(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                              conv_states: torch.Tensor, query_start_loc: torch.Tensor,
                              activation: str = "silu") -> torch.Tensor:
    """支持梯度回传的 causal_conv1d 前向接口（kimik3 场景：3D 固定 batch + silu + 冷启动）。"""
    return CausalConv1dFnAutoGrad.apply(x, weight, bias, conv_states, query_start_loc, activation)
