# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Operator-local shared jetty context, constructed on every group rank before graph capture."""

import os

import torch
from cann_ops_transformer.op_builder import OpBuilder


class SharedJettyContextOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("bandwidth_test_shared_jetty_context", category="tools")

    def sources(self):
        return [
            "csrc/tools/bandwidth_test_shared_jetty_context.cpp",
            "csrc/tools/bandwidth_test_shared_jetty_adapter.cpp",
        ]

    def schema(self):
        return None

    def register_meta(self):
        pass

    def include_paths(self):
        # hcomm_host.h is an AscendC/asc-comm Host API. CANN run packages place it under
        # <arch>-linux/asc/include/adv_api, while some development layouts expose include/ascendc.
        candidates = [
            os.path.join(self.cann_path, "aarch64-linux/asc/include/adv_api"),
            os.path.join(self.cann_path, "x86_64-linux/asc/include/adv_api"),
            os.path.join(self.cann_path, "asc/include/adv_api"),
            os.path.join(self.cann_path, "include/ascendc/host_api"),
            os.path.join(self.cann_path, "include/ascendc/adv_api"),
        ]
        asc_comm_paths = [
            path for path in candidates if os.path.isfile(os.path.join(path, "hcomm/hcomm_host.h"))
        ]
        if not asc_comm_paths:
            raise FileNotFoundError(
                "AscendC Host header hcomm/hcomm_host.h was not found in the CANN installation; "
                "install the matching asc-comm development headers"
            )
        return asc_comm_paths + super().include_paths()


_builder = SharedJettyContextOpBuilder()


class SharedJettyContextManager:
    def __init__(
        self,
        group,
        group_name,
        world_size,
        shared_jetty_num,
        shared_jetty_comm_level,
        buffer_size,
        sequence,
    ):
        self.group = group
        self.ccl_buffer_size = buffer_size
        # The C++ manager owns the Win; HCCL owns the registered resources and shared channels.
        # Keep both owners alive until every kernel and graph replay has completed.
        tag = f"bandwidth_test_shared_jetty_{sequence}_level_{shared_jetty_comm_level}"
        self._manager = _builder.load().SharedJettyContextManager(
            group_name,
            world_size,
            shared_jetty_num,
            shared_jetty_comm_level,
            buffer_size,
            tag,
        )

    def create_context(self):
        try:
            return self._manager.create_context()
        except Exception as creation_error:
            # HCCL exposes no public deregistration API for this registered Win. Once construction has
            # entered HCCL, tear down the poisoned communicator before releasing the C++ Win owner.
            try:
                torch.distributed.destroy_process_group(self.group)
            except Exception as destroy_error:
                self._manager._abandon_registered_window()
                self._manager = None
                raise RuntimeError(
                    "shared Jetty context construction failed and its ProcessGroup could not be destroyed; "
                    "the registered Win was intentionally retained until process exit: "
                    f"{destroy_error}"
                ) from creation_error
            self._manager = None
            raise
