import torch
import torch_npu
from torch.library import impl
from cann_ops_transformer.op_builder.builder import OpBuilder
from cann_ops_transformer.op_builder.builder import AS_LIBRARY

class HostPinnedCounterOpBuilder(OpBuilder):
    def __init__(self):
        super(HostPinnedCounterOpBuilder, self).__init__("host_pinned_counter")

    def sources(self):
        return ['ops/csrc/host_pinned_counter.cpp']
    
    def schema(self) -> str:
        return "host_pinned_counter() -> bool"

    def register_meta(self):
        pass

host_pinned_counter_op_builder = HostPinnedCounterOpBuilder()
op_module_host_pinned_counter = host_pinned_counter_op_builder.load()

HostPinnedCounter = op_module_host_pinned_counter.HostPinnedCounter