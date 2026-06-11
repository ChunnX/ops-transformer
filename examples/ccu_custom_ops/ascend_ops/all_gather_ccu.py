#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Custom CCU AllGather (Ascend950PR) — Python orchestration over ACL VMM IPC (FABRIC handles).

import torch
import torch_npu
import torch.distributed as dist


class AllGatherCcu:
    def __init__(self, group):
        self.group = group
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        self.device_id = torch.npu.current_device()
        self._anchor = torch.zeros(1, dtype=torch.int32).npu()
        self._inited = False
        self.context = None

    def _setup(self, size_bytes):
        fabric, pid = torch.ops.ascend_ops.ipcAllocExport(self._anchor, size_bytes, self.device_id)
        # all_gather 128B FABRIC handles across ranks
        fab_npu = fabric.npu()
        fouts = [torch.empty_like(fab_npu) for _ in range(self.world_size)]
        dist.all_gather(fouts, fab_npu, group=self.group)
        handles_all = torch.stack(fouts).contiguous()          # [ws,128] uint8 (npu)
        # all_gather bare tgids
        pid_t = torch.tensor([pid], dtype=torch.int64).npu()
        pouts = [torch.empty_like(pid_t) for _ in range(self.world_size)]
        dist.all_gather(pouts, pid_t, group=self.group)
        peer_pids = [int(o.cpu().item()) for o in pouts]
        # authorize peers, publish, import
        torch.ops.ascend_ops.ipcSetpid(self._anchor, peer_pids)
        dist.barrier(self.group)
        self.context = torch.ops.ascend_ops.ipcImportBuild(
            self._anchor, handles_all, self.rank, self.world_size, self.device_id
        )
        dist.barrier(self.group)
        self._inited = True

    def all_gather(self, x):
        x = x.contiguous()
        n = x.numel()
        size_bytes = self.world_size * n * x.element_size()
        if not self._inited:
            self._setup(size_bytes)
        torch.ops.ascend_ops.AllGatherCcuPut(x, self.context, self.rank, self.world_size)
        torch.npu.synchronize()
        dist.barrier(self.group)
        torch.npu.synchronize()
        out = torch.ops.ascend_ops.AllGatherCcuGet(x, self.context, self.rank, self.world_size)
        torch.npu.synchronize()
        return out
