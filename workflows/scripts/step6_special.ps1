$ROOT = "D:\code\ops-transformer"
$OPS = "$ROOT\torch_extension\cann_ops_transformer\ops"
$SRC_CSRC = "$OPS\csrc"
$SRC_GC = "$OPS\graph_convert"

# Special: mamba/causal_conv1d - two ops sharing one dir + shared graph_convert
$dir = "$ROOT\mamba\causal_conv1d\torch_extension"
$csrc = "$dir\csrc"
New-Item -ItemType Directory -Path $csrc -Force | Out-Null
Copy-Item "$OPS\causal_conv1d_fn.py" "$dir\causal_conv1d_fn.py" -Force
Copy-Item "$OPS\causal_conv1d_update.py" "$dir\causal_conv1d_update.py" -Force
Copy-Item "$SRC_CSRC\causal_conv1d_fn.cpp" "$csrc\causal_conv1d_fn.cpp" -Force
Copy-Item "$SRC_CSRC\causal_conv1d_update.cpp" "$csrc\causal_conv1d_update.cpp" -Force
Copy-Item "$SRC_GC\graph_convert_causal_conv1d.py" "$dir\graph_convert_causal_conv1d.py" -Force
Set-Content -Path "$dir\__init__.py" -Value @'
__all__ = ["causal_conv1d_fn", "causal_conv1d_update",
           "convert_causal_conv1d_fn", "convert_causal_conv1d_update"]

from .causal_conv1d_fn import causal_conv1d_fn
from .causal_conv1d_update import causal_conv1d_update
from .graph_convert_causal_conv1d import convert_causal_conv1d_fn, convert_causal_conv1d_update
'@

# Special: mc2/common - comm_context (public module)
$dir = "$ROOT\mc2\common\torch_extension"
$csrc = "$dir\csrc"
New-Item -ItemType Directory -Path $csrc -Force | Out-Null
Copy-Item "$OPS\comm_context.py" "$dir\comm_context.py" -Force
Copy-Item "$SRC_CSRC\comm_context.cpp" "$csrc\comm_context.cpp" -Force
Set-Content -Path "$dir\__init__.py" -Value @'
__all__ = ["CommContextManager"]

from .comm_context import CommContextManager
'@

# Special: mc2/moe_distribute_dispatch - includes deep_ep
$dir = "$ROOT\mc2\moe_distribute_dispatch\torch_extension"
$csrc = "$dir\csrc"
New-Item -ItemType Directory -Path $csrc -Force | Out-Null
Copy-Item "$OPS\moe_distribute_dispatch.py" "$dir\moe_distribute_dispatch.py" -Force
Copy-Item "$OPS\deep_ep.py" "$dir\deep_ep.py" -Force
Copy-Item "$SRC_CSRC\moe_distribute_dispatch.cpp" "$csrc\moe_distribute_dispatch.cpp" -Force
Copy-Item "$SRC_GC\graph_convert_moe_distribute_dispatch.py" "$dir\graph_convert_moe_distribute_dispatch.py" -Force
Set-Content -Path "$dir\__init__.py" -Value @'
__all__ = ["moe_distribute_dispatch", "MoeDistributeBuffer",
           "convert_moe_distribute_dispatch"]

from .moe_distribute_dispatch import moe_distribute_dispatch
from .deep_ep import MoeDistributeBuffer
from .graph_convert_moe_distribute_dispatch import convert_moe_distribute_dispatch
'@

# Special: mc2/engram_fetch - elastic_buffer
$dir = "$ROOT\mc2\engram_fetch\torch_extension"
$csrc = "$dir\csrc"
New-Item -ItemType Directory -Path $csrc -Force | Out-Null
Copy-Item "$OPS\elastic_buffer.py" "$dir\elastic_buffer.py" -Force
Copy-Item "$SRC_CSRC\elastic_buffer.cpp" "$csrc\elastic_buffer.cpp" -Force
Set-Content -Path "$dir\__init__.py" -Value @'
__all__ = ["ElasticBuffer", "EPHandle"]

from .elastic_buffer import ElasticBuffer, EPHandle
'@

# Special: posembedding/inplace_partial_rotary_mul_grad - filename differs from dirname
$dir = "$ROOT\posembedding\inplace_partial_rotary_mul_grad\torch_extension"
$csrc = "$dir\csrc"
New-Item -ItemType Directory -Path $csrc -Force | Out-Null
Copy-Item "$OPS\inplace_partial_rotary_mul_backward.py" "$dir\inplace_partial_rotary_mul_backward.py" -Force
Copy-Item "$SRC_CSRC\inplace_partial_rotary_mul_backward.cpp" "$csrc\inplace_partial_rotary_mul_backward.cpp" -Force
Copy-Item "$SRC_GC\graph_convert_inplace_partial_rotary_mul_backward.py" "$dir\graph_convert_inplace_partial_rotary_mul_backward.py" -Force
Set-Content -Path "$dir\__init__.py" -Value @'
__all__ = ["inplace_partial_rotary_mul_backward",
           "convert_inplace_partial_rotary_mul_backward"]

from .inplace_partial_rotary_mul_backward import inplace_partial_rotary_mul_backward
from .graph_convert_inplace_partial_rotary_mul_backward import convert_inplace_partial_rotary_mul_backward
'@

Write-Output "Special cases done: causal_conv1d, mc2/common, mc2/moe_distribute_dispatch, mc2/engram_fetch, posembedding/inplace_partial_rotary_mul_grad"
