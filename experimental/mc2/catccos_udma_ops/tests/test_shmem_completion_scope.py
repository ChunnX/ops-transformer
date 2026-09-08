# Copyright (c) 2026 Huawei Technologies Co., Ltd.

"""Source guards for the fused workers' single-owner UDMA completion protocol.

NPU execution is required to validate data visibility; these checks prevent
rank-wide polling and premature teardown of the local MTE event loop.
"""

import re
from pathlib import Path


OVERLAY = (
    Path(__file__).parents[1] / "csrc" / "catccos_overlay" / "include" / "catccos"
)


def _without_comments(source):
    return re.sub(r"//[^\n]*|/\*.*?\*/", "", source, flags=re.DOTALL)


def _function_body(source, name):
    start = source.index("{", source.index(f"{name}()"))
    depth = 1
    end = start + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start + 1:end - 1]


def test_peer_owned_workers_do_not_use_rank_wide_completion():
    # SHMEM 1.6.0's generic APIs only drain local pipes/cache, while master !715
    # polls all peers. Neither replaces completion by the UDMA peer's AIV owner.
    offenders = []
    for path in OVERLAY.rglob("*.hpp"):
        source = _without_comments(path.read_text(encoding="utf-8"))
        if re.search(r"\baclshmem(?:i)?_(?:fence|quiet)\s*\(", source):
            offenders.append(str(path.relative_to(OVERLAY)))
    assert not offenders, f"Rank-wide completion from peer-owned workers: {offenders}"


def test_mid_loop_completion_keeps_local_mte_events_alive():
    source = (OVERLAY / "comm" / "block" / "comm_block_mte_udma.hpp").read_text(
        encoding="utf-8"
    )
    body = _without_comments(_function_body(source, "CompleteForSignal"))
    # Only the remote block is retired: a following local tile must still find
    # the MTE3_MTE2 events created at the start of this block loop.
    assert body.index("remoteComm.FinalizeBlockLoop()") < body.index(
        "aclshmemx_mte_quiet()"
    )
    assert "localComm." not in body
    assert "InitBlockLoop" not in body


def test_completed_gets_and_self_copies_do_not_poll_udma_again():
    kernel_dir = OVERLAY / "dgemm" / "kernel"
    get_kernel = (kernel_dir / "grouped_matmul_alltoallv_tla.hpp").read_text(
        encoding="utf-8"
    )
    get_kernel = _without_comments(get_kernel)
    completed_get = re.compile(
        r"blockRemoteCopy\.FinalizeBlockLoop\(\);\s*"
        r"aclshmemx_mte_quiet\(\);\s*aclshmemx_signal_op\("
    )
    assert len(completed_get.findall(get_kernel)) == 2

    put_kernel = (kernel_dir / "ascend950_alltoallv_grouped_matmul.hpp").read_text(
        encoding="utf-8"
    )
    self_copy = put_kernel.split("if (isLocalCopy) {", 1)[1].split(
        "// UDMA does not need the MTE communication block.", 1
    )[0]
    assert "aclshmemx_mte_quiet();" in self_copy
    assert "CompleteForSignal" not in self_copy
