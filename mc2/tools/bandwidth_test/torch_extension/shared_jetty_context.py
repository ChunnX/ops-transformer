# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

"""Operator-local shared jetty context, constructed collectively before graph capture."""

import os

import torch
from cann_ops_transformer.op_builder import OpBuilder


class SharedJettyContextOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("bandwidth_test_shared_jetty_context", category="tools")

    def sources(self):
        return ["csrc/tools/bandwidth_test_shared_jetty_context.cpp"]

    def schema(self):
        return None

    def register_meta(self):
        pass

    def include_paths(self):
        # hcomm_host.h is installed with the AscendC communication headers.
        return [os.path.join(self.cann_path, "include/ascendc/host_api"),
                os.path.join(self.cann_path, "include/ascendc/adv_api")] + super().include_paths()


_builder = SharedJettyContextOpBuilder()


class SharedJettyContextManager:
    def __init__(self, group, group_name, world_size, shared_jetty_num, shared_jetty_comm_level,
                 buffer_size, sequence):
        self.group = group
        self.ccl_buffer_size = buffer_size
        # The manager owns the Win; the communicator owns its registration and multi-channel handles.
        # Keep both owners alive while kernels or captured graphs may use these resources. The bandwidth
        # test destroys its process group before this owner is released, matching the shared-Jetty sample.
        tag = f"bandwidth_test_shared_jetty_{sequence}_level_{shared_jetty_comm_level}"
        self._manager = _builder.load().SharedJettyContextManager(
            group_name, world_size, shared_jetty_num, shared_jetty_comm_level, buffer_size, tag)
        self._manager.prepare()
        local = (shared_jetty_num, shared_jetty_comm_level, buffer_size, sequence,
                 self._manager.local_buffer_address)
        gathered = [None] * world_size
        torch.distributed.all_gather_object(gathered, local, group=group)
        if any(item[:4] != local[:4] for item in gathered):
            raise ValueError(
                "shared jetty count, communication level, buffer size and creation order must match across ranks"
            )
        self._remote_buffers = [item[4] for item in gathered]

    def create_context(self):
        return self._manager.create_context(self._remote_buffers)
