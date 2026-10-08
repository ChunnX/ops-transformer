# Copyright (c) 2026, edconeone
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Combine must reject unsupported ownership/options before a device launch."""

import sys
from types import SimpleNamespace

import pytest

from ops_transformer_mc2.buffers import elastic


@pytest.fixture
def buffer(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(Tensor=type("Tensor", (), {}))
    )
    result = elastic.ElasticBuffer.__new__(elastic.ElasticBuffer)
    result._handle_token = object()
    result._runtime = SimpleNamespace(closed=True, ensure_alive=lambda: None)
    return result


def call(buffer, handle, **kwargs):
    options = dict(
        topk_weights=None,
        bias=None,
        num_compute_units=0,
        num_qps=0,
        previous_event=None,
        previous_event_before_epilogue=None,
        async_with_compute_stream=False,
        allocate_on_comm_stream=False,
    )
    options.update(kwargs)
    return buffer._combine_low_latency(None, handle, **options)


def test_foreign_handle_rejected(buffer):
    with pytest.raises(ValueError, match="this buffer"):
        call(buffer, SimpleNamespace(_buffer_token=object(), metadata_only=False))


def test_metadata_only_handle_rejected(buffer):
    with pytest.raises(ValueError, match="payload dispatch"):
        call(
            buffer,
            SimpleNamespace(_buffer_token=buffer._handle_token, metadata_only=True),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bias": object()},
        {"topk_weights": object()},
        {"num_qps": 2},
        {"num_compute_units": 64},
        {"previous_event": object()},
        {"async_with_compute_stream": True},
        {"allocate_on_comm_stream": True},
    ],
)
def test_unsupported_combine_options(buffer, kwargs):
    with pytest.raises(NotImplementedError):
        call(
            buffer,
            SimpleNamespace(_buffer_token=buffer._handle_token, metadata_only=False),
            **kwargs,
        )


def test_dispatch_only_build_reports_missing_combine(buffer, monkeypatch):
    monkeypatch.setattr(elastic, "require_native", lambda _: SimpleNamespace())
    with pytest.raises(NotImplementedError, match="low_latency_combine/build.sh"):
        buffer.combine(None, None)
