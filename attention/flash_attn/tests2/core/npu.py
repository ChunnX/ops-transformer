import torch

from core import NpuBase
from core.utils.input_spec import build_flash_attn_params

try:
    from cann_ops_transformer.ops import flash_attn, flash_attn_metadata
    _HAS_NPU = True
except ImportError:
    _HAS_NPU = False

try:
    import torchair
    from torchair.configs.compiler_config import CompilerConfig
    _HAS_TORCHAIR = True
except ImportError:
    _HAS_TORCHAIR = False


class FlashAttnGraphNetwork(torch.nn.Module):
    """torch.compile 图模式包装器。

    NOTE: forward() 参数列表须与 flash_attn 签名保持同步。
    torch.compile 要求固定输入签名, 故使用显式参数而非 **kwargs。
    """

    def forward(self, q, k, v, *, metadata, block_table=None, sinks=None,
                attn_mask=None, cu_seqlens_q=None, cu_seqlens_kv=None,
                seqused_q=None, seqused_kv=None, softmax_scale=1.0,
                mask_mode=0, win_left=-1, win_right=-1,
                max_seqlen_q=-1, max_seqlen_kv=-1,
                layout_q="BNSD", layout_kv="BNSD", layout_out="BNSD",
                return_softmax_lse=False):
        out, lse_out = flash_attn(
            q, k, v,
            block_table=block_table,
            sinks=sinks,
            attn_mask=attn_mask,
            metadata=metadata,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            seqused_q=seqused_q,
            seqused_kv=seqused_kv,
            softmax_scale=softmax_scale,
            mask_mode=mask_mode,
            win_left=win_left,
            win_right=win_right,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            layout_q=layout_q,
            layout_kv=layout_kv,
            layout_out=layout_out,
            return_softmax_lse=return_softmax_lse,
        )
        return out, lse_out


class NpuBackend(NpuBase):
    name = "npu"

    def __init__(self, device_id: int = 0):
        self._device = torch.device(f"npu:{device_id}")
        self._graph_net = None
        if _HAS_NPU:
            torch.npu.set_device(device_id)

    @property
    def device(self) -> torch.device:
        return self._device

    def is_available(self) -> bool:
        return _HAS_NPU

    def clear_cache(self):
        if _HAS_NPU:
            torch.npu.empty_cache()

    def _get_graph_net(self):
        if self._graph_net is None:
            torch._dynamo.reset()
            net = FlashAttnGraphNetwork().npu()
            config = CompilerConfig()
            config.mode = "reduce-overhead"
            config.experimental_config.aclgraph._aclnn_static_shape_kernel = True
            config.experimental_config.aclgraph._aclnn_static_shape_kernel_build_dir = "./"
            config.experimental_config.frozen_parameter = True
            config.experimental_config.tiling_schedule_optimize = True
            config.experimental_config.topology_sorting_strategy = "StableRDFS"
            npu_backend = torchair.get_npu_backend(compiler_config=config)
            self._graph_net = torch.compile(net, fullgraph=False, backend=npu_backend, dynamic=False)
        return self._graph_net

    def npu_compute(self, inputs: dict, graph_path: int = 0) -> dict:
        if not _HAS_NPU:
            raise ImportError("cann_ops_transformer 不可用, 请安装后重试")
        if graph_path != 0 and not _HAS_TORCHAIR:
            raise ImportError("torchair 不可用, graph_path!=0 需要 torchair, 请安装后重试")

        meta_kwargs, kernel_kwargs, _ = build_flash_attn_params(
            inputs, self._device, inputs)

        metadata = flash_attn_metadata(**meta_kwargs)

        if graph_path != 0:
            net = self._get_graph_net()
            out, lse = net(
                inputs["q"], inputs["k"], inputs["v"],
                metadata=metadata, **kernel_kwargs
            )
        else:
            out, lse = flash_attn(inputs["q"], inputs["k"], inputs["v"],
                                  metadata=metadata, **kernel_kwargs)
        torch.npu.synchronize()
        return {"out": out, "lse": lse}
