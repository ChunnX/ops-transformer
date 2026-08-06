@echo off
set OPS=D:\code\ops-transformer\torch_extension\cann_ops_transformer\ops
set ROOT=D:\code\ops-transformer
set SRC_CSRC=%OPS%\csrc
set SRC_GC=%OPS%\graph_convert

REM Special: mamba/causal_conv1d - two ops sharing one dir + shared graph_convert
set DIR=%ROOT%\mamba\causal_conv1d\torch_extension
set CS=%DIR%\csrc
mkdir "%CS%" 2>nul
copy /Y "%OPS%\causal_conv1d_fn.py" "%DIR%\causal_conv1d_fn.py" >nul
copy /Y "%OPS%\causal_conv1d_update.py" "%DIR%\causal_conv1d_update.py" >nul
copy /Y "%SRC_CSRC%\causal_conv1d_fn.cpp" "%CS%\causal_conv1d_fn.cpp" >nul
copy /Y "%SRC_CSRC%\causal_conv1d_update.cpp" "%CS%\causal_conv1d_update.cpp" >nul
copy /Y "%SRC_GC%\graph_convert_causal_conv1d.py" "%DIR%\graph_convert_causal_conv1d.py" >nul
(
echo __all__ = ["causal_conv1d_fn", "causal_conv1d_update",
echo            "convert_causal_conv1d_fn", "convert_causal_conv1d_update"]
echo.
echo from .causal_conv1d_fn import causal_conv1d_fn
echo from .causal_conv1d_update import causal_conv1d_update
echo from .graph_convert_causal_conv1d import convert_causal_conv1d_fn, convert_causal_conv1d_update
) > "%DIR%\__init__.py"
echo OK: mamba/causal_conv1d (2 ops)

REM Special: mc2/common - comm_context
set DIR=%ROOT%\mc2\common\torch_extension
set CS=%DIR%\csrc
mkdir "%CS%" 2>nul
copy /Y "%OPS%\comm_context.py" "%DIR%\comm_context.py" >nul
copy /Y "%SRC_CSRC%\comm_context.cpp" "%CS%\comm_context.cpp" >nul
(
echo __all__ = ["CommContextManager"]
echo.
echo from .comm_context import CommContextManager
) > "%DIR%\__init__.py"
echo OK: mc2/common

REM Special: mc2/moe_distribute_dispatch - includes deep_ep
set DIR=%ROOT%\mc2\moe_distribute_dispatch\torch_extension
set CS=%DIR%\csrc
mkdir "%CS%" 2>nul
copy /Y "%OPS%\moe_distribute_dispatch.py" "%DIR%\moe_distribute_dispatch.py" >nul
copy /Y "%OPS%\deep_ep.py" "%DIR%\deep_ep.py" >nul
copy /Y "%SRC_CSRC%\moe_distribute_dispatch.cpp" "%CS%\moe_distribute_dispatch.cpp" >nul
copy /Y "%SRC_GC%\graph_convert_moe_distribute_dispatch.py" "%DIR%\graph_convert_moe_distribute_dispatch.py" >nul
(
echo __all__ = ["moe_distribute_dispatch", "MoeDistributeBuffer",
echo            "convert_moe_distribute_dispatch"]
echo.
echo from .moe_distribute_dispatch import moe_distribute_dispatch
echo from .deep_ep import MoeDistributeBuffer
echo from .graph_convert_moe_distribute_dispatch import convert_moe_distribute_dispatch
) > "%DIR%\__init__.py"
echo OK: mc2/moe_distribute_dispatch

REM Special: mc2/engram_fetch - elastic_buffer
set DIR=%ROOT%\mc2\engram_fetch\torch_extension
set CS=%DIR%\csrc
mkdir "%CS%" 2>nul
copy /Y "%OPS%\elastic_buffer.py" "%DIR%\elastic_buffer.py" >nul
copy /Y "%SRC_CSRC%\elastic_buffer.cpp" "%CS%\elastic_buffer.cpp" >nul
(
echo __all__ = ["ElasticBuffer", "EPHandle"]
echo.
echo from .elastic_buffer import ElasticBuffer, EPHandle
) > "%DIR%\__init__.py"
echo OK: mc2/engram_fetch

REM Special: posembedding/inplace_partial_rotary_mul_grad - filename differs from dirname
set DIR=%ROOT%\posembedding\inplace_partial_rotary_mul_grad\torch_extension
set CS=%DIR%\csrc
mkdir "%CS%" 2>nul
copy /Y "%OPS%\inplace_partial_rotary_mul_backward.py" "%DIR%\inplace_partial_rotary_mul_backward.py" >nul
copy /Y "%SRC_CSRC%\inplace_partial_rotary_mul_backward.cpp" "%CS%\inplace_partial_rotary_mul_backward.cpp" >nul
copy /Y "%SRC_GC%\graph_convert_inplace_partial_rotary_mul_backward.py" "%DIR%\graph_convert_inplace_partial_rotary_mul_backward.py" >nul
(
echo __all__ = ["inplace_partial_rotary_mul_backward",
echo            "convert_inplace_partial_rotary_mul_backward"]
echo.
echo from .inplace_partial_rotary_mul_backward import inplace_partial_rotary_mul_backward
echo from .graph_convert_inplace_partial_rotary_mul_backward import convert_inplace_partial_rotary_mul_backward
) > "%DIR%\__init__.py"
echo OK: posembedding/inplace_partial_rotary_mul_grad

echo All special cases done
