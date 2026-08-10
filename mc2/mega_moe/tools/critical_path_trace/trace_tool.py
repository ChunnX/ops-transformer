#!/usr/bin/env python3
# coding: utf-8
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Validate, audit, and render reviewable operator critical-path traces.

The tool intentionally uses only the Python standard library so it can run in
operator build/test environments without adding a package dependency.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable
import zlib


SPEC_SCHEMA_VERSION = 1
TRACE_SCHEMA_VERSION = 1
AUDIT_SCHEMA_VERSION = 1
POINT_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*_S$")
TRACE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
COMPILE_SWITCH_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")
FIELD_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
REQUIRED_HEADER_FIELDS = {
    "magic",
    "version",
    "schema_id",
    "header_words",
    "record_words",
    "record_count",
    "role_counts",
    "clock_bits",
    "repetition_count",
    "capture_capacity",
    "truncated",
    "flags",
    "rank",
    "runtime_config_hash",
}

POINT_TEXT_ZH = {
    "process_begin_S": {
        "when": "该物理 AI Block 的对应角色刚进入 MegaMoeWave::Process，尚未发出本角色的算子工作。",
        "why": "为同一 AI Block 内的 AIC、AIV0、AIV1 提供局部参考起点，并观察三个角色的启动偏差。",
        "proof": "该读时钟位于 Process 第一条正常语句之前，是本角色执行的第一条诊断指令。",
        "perturbation": "每个角色/核增加一次 32 位 GetSystemCycle 和一次 Scalar 赋值。",
    },
    "work_end_S": {
        "when": "本角色的有效工作已经完成，但尚未进入 tracing 专用的终结同步和 y 落盘。AIV 侧已等到最终 Unpermute 写回，AIC 侧的发布已越过 FIX_S。",
        "why": "把真实有效工作的尾部与 tracing rendezvous、记录写回等诊断开销分开。",
        "proof": "采样位于 routed wave、跨 rank 完成、最终 Unpermute MTE3_V 等待和控制状态恢复之后，同时早于两次 tracing rendezvous。",
        "perturbation": "终结同步之前增加一次 32 位 GetSystemCycle 和一次 Scalar 赋值。",
    },
    "dispatch_begin_S": {
        "when": "当前 wave 的第一个专家即将在 AIV1 Scalar 路径执行 SendCntCal。",
        "why": "覆盖真实的逐 wave Dispatch 预取窗口，避免用可能一次预取多个 wave 的外层循环近似。",
        "proof": "DispatchExpert 对每个专家只执行一次；首专家判断把 expertIdx 映射到当前 experts_per_wave。",
        "perturbation": "每个有效 AIV1/wave 增加一次动态边界判断和一次时钟采样。",
    },
    "dispatch_end_S": {
        "when": "当前 wave 的最后一个专家已完成 token、scale、meta-info 的 MTE3 写入和 ready 发布。",
        "why": "证明 Dispatch 何时真正释放 GMM1，并可与同一 AI Block 内的 GMM1 起点直接比较。",
        "proof": "最后一个专家复用已回收的 EVENT_ID0 增加 MTE3_S，把最终写入和发布依赖传到 Scalar 后再读时钟。",
        "perturbation": "每个有效 AIV1/wave 增加一次 MTE3→Scalar 桥接、一次时钟采样和三次独占 workspace 标量写。",
    },
    "gmm1_begin_S": {
        "when": "AIC 已观察到 Dispatch ready，并完成前序 ping-pong 依赖等待；第一个 GMM1 tile 即将发出 MMAD。",
        "why": "把 Dispatch-ready/前序依赖等待与真正的 GMM1 计算区间分开。",
        "proof": "打点位于 upstream ready 和 WaitForVector 之后；复用 EVENT_ID0 的 FIX_S 把 FIX 依赖传到 Scalar。",
        "perturbation": "每个有效 AIC/wave 增加一次条件判断、一次 FIX→Scalar 桥接和一次时钟采样。",
    },
    "gmm1_end_S": {
        "when": "当前 wave 分配给该 AIC 的所有 GMM1 FIX 工作均已完成。",
        "why": "让 GMM1 结束边界落在实际 FIX 完成，而不是 helper 返回或通知发出时刻。",
        "proof": "完整 tile 循环结束后执行 FIX_S，且只有记录过 begin 的核才采样，因此时钟不会早于最后一个 FIX。",
        "perturbation": "每个有效 AIC/wave 增加一次 FIX→Scalar 完成桥接和一次时钟采样。",
    },
    "swiglu_begin_S": {
        "when": "非空 epilogue 的 GMM1/输入依赖已经有序进入 Vector，马上开始 SwiGLU、MX 量化设置和 Vector 工作。",
        "why": "统一覆盖非预取 WaitForCube 和预取 MTE2_V 两条分支之后的真实 SwiGLU 起点。",
        "proof": "所有 routed-wave epilogue 分支都在各自 readiness wait 后进入公共 operator；V_S 再把 Vector 依赖传到 Scalar。",
        "perturbation": "每个有效 AIV0/wave 增加一次条件判断、一次 Vector→Scalar 桥接和一次时钟采样。",
    },
    "swiglu_end_S": {
        "when": "当前 wave 分配给该 AIV0 的全部 SwiGLU/MX 量化 GM 输出已经完成。",
        "why": "观察真实 AIV0 epilogue 尾部，以及它与同组 GMM2、Dispatch、Combine 的重叠。",
        "proof": "每个 epilogue 返回前已经消费 MTE3_V/MTE3_S 并发布 ready；外层 wave 返回必然晚于最后一次 GM 完成。",
        "perturbation": "每个有效 AIV0/wave 只增加一次时钟采样，不额外串行化完成路径。",
    },
    "gmm2_begin_S": {
        "when": "AIC 的 Scalar 轮询已经观察到当前 wave 的 SwiGLU-ready 计数，第一个 GMM2 MMAD 即将发出。",
        "why": "让 GMM2 区间从 ready 被观察之后开始，把发布到观察的等待留在计算区间之外。",
        "proof": "bypass GM 轮询只会在 epilogue MTE3_S 和 ready AtomicAdd 完成后返回；随后同一 Scalar 顺序直接读时钟。",
        "perturbation": "每个有效 AIC/wave 增加一次条件判断和一次时钟采样。",
    },
    "gmm2_end_S": {
        "when": "当前 wave 分配给该 AIC 的全部 GMM2 FIX 写入已完成，最后一次 per-expert ready 也已发布。",
        "why": "证明本地 GMM2 完成和 ready 发布时刻，并把 AIV1 的远端观察保留为独立 handoff。",
        "proof": "PublishGmm2Ready 在 bypass GM 写之前执行 FIX_S；外层采样位于最后一次发布之后，但不宣称 AIV1 已经观察到。",
        "perturbation": "每个有效 AIC/wave 增加一次时钟采样；复用生产路径已有的 FIX_S 发布语义。",
    },
    "gmm1_expert_begin_S": {
        "when": "当前专家的 Dispatch ready 与前序 ping-pong 依赖已满足，本核第一个 GMM1 tile 即将发出 MMAD。",
        "why": "把批专家级长框拆成逐专家区间，单独暴露同一 wave 内的 FIX 尾部和依赖等待。",
        "proof": "打点位于 upstream-ready 与 WaitForVector 之后，FIX_S 仅把 causal begin 依赖传到 Scalar。",
        "perturbation": "每个活跃专家增加一次动态状态检查、一次 FIX→Scalar causal bridge 和一次时钟采样。",
    },
    "gmm1_expert_end_S": {
        "when": "当前专家分配给本核的 GMM1 MMAD/NotifyVector 已全部发出；FIX 可能仍在尾部执行。",
        "why": "提供低扰动逐专家结束边界，并把到下一专家 causal begin 的 FIX 尾部/等待显示为独立间隙。",
        "proof": "采样紧跟 GroupMatmulWithSwigluQuant 返回，不增加逐专家 FIX_S，因此明确标记为 issue 而不是 actual。",
        "perturbation": "每个活跃专家增加一次时钟采样和三次独占 workspace 标量写；不新增完成 bridge，避免强制串行化 FIX 尾部。",
    },
    "swiglu_expert_begin_S": {
        "when": "当前专家首个非空 epilogue 已完成 GMM1/输入就绪等待，即将开始 SwiGLU 与 MX 量化。",
        "why": "避免把多个专家之间的 WaitForCube/输入等待误认为 SwiGLU 计算。",
        "proof": "所有 epilogue 分支均在 readiness wait 后进入公共 operator；V_S 再把 Vector 依赖传到 Scalar。",
        "perturbation": "每个活跃专家增加一次动态状态检查、一次 V→Scalar bridge 和一次 begin 时钟。",
    },
    "swiglu_expert_end_S": {
        "when": "当前专家在本 AIV0 上的全部 epilogue tile 已返回，量化 data/scale 已写回 GM 且 ready 已发布。",
        "why": "标出逐专家真实输出尾部，让下一专家前的等待/调度间隙可见。",
        "proof": "每个 operator 返回前都复用已有 MTE3_V/MTE3_S 完成依赖并发布 ready；helper 返回后的时钟晚于最后一个 tile。",
        "perturbation": "每个活跃专家增加一次条件判断、结束时钟和三次独占 workspace 标量写；不新增 MTE3 bridge。",
    },
    "gmm2_expert_begin_S": {
        "when": "当前专家的 SwiGLU-ready counter 已被 Scalar 观察，本核第一个 GMM2 MMAD 即将发出。",
        "why": "逐专家区分 GMM2 计算与 ready 等待，并验证 GMM2 是否近似连续向 GM 输出。",
        "proof": "bypass-GM 轮询在 ready 可见后返回，紧随其后的 Scalar 时钟无需跨流水 bridge。",
        "perturbation": "每个活跃专家增加一次动态状态检查和一次时钟采样。",
    },
    "gmm2_expert_end_S": {
        "when": "当前专家本核全部 GMM2 FIX 直写完成，且该 AIC 的 per-expert ready 已发布。",
        "why": "逐专家显示 GMM2 实际完成，并把到下一专家 begin 的 ready 等待/调度间隙独立画出。",
        "proof": "PublishGmm2Ready 内已有 FIX_S 先证明本核写回完成，再执行 bypass-GM ready 发布。",
        "perturbation": "每个活跃专家增加一次时钟采样和三次独占 workspace 标量写；复用生产路径已有 FIX_S。",
    },
    "combine_begin_S": {
        "when": "该 AIV1 拥有非空 token，且首个专家的 WaitGmm2Ready 已观察到完整 AIC fan-in；随后开始 meta-info 加载和发送。",
        "why": "显式展示 GMM2 发布到 AIV1 观察的 handoff，并保证 Combine 从完整 fan-in 之后计时。",
        "proof": "WaitGmm2Ready 只有在 MTE2_V/V_S 有序的归约结果达到全部 AIC 目标后才返回，打点紧随其后。",
        "perturbation": "每个有效 AIV1/wave 增加一次条件判断和一次时钟采样。",
    },
    "combine_end_S": {
        "when": "当前 wave 由该 AIV1 发出的全部远端 token row 已完成 MTE3 send。",
        "why": "让 Combine 在实际远端发送完成时结束，而不是停在 Scalar 发出或 drain 入队时刻。",
        "proof": "ring drain 等待所有实际使用 slot 的 MTE3_MTE2 事件，再用复用 EVENT_ID0 的 MTE2_S 把完成依赖传到 Scalar。",
        "perturbation": "每个有效 AIV1/wave 增加一次 MTE2→Scalar 桥接、一次时钟采样和三次独占 workspace 标量写。",
    },
    "a8w4_process_begin_S": {
        "when": "A8W4 MTE 主流程刚进入 Process。",
        "why": "建立同一 AI Block 三个角色的局部时间起点。",
        "proof": "该采样位于 Process 的有效工作之前。",
        "perturbation": "每核一次时钟采样。",
    },
    "a8w4_work_end_S": {
        "when": "本角色有效工作已结束、trace 写回尚未开始。",
        "why": "排除 trace 汇合与落盘自身的开销。",
        "proof": "采样位于 Unpermute/计算返回之后、trace 专用 SyncAll 之前。",
        "perturbation": "每核一次时钟采样。",
    },
    "a8w4_dispatch_begin_S": {
        "when": "AIV1 即将为当前专家执行 SendCntCal 和 token dispatch。",
        "why": "观察逐专家 Dispatch 与当前 GMM 的重叠。",
        "proof": "调用点紧邻 SendCntCal 之前。",
        "perturbation": "每专家一次时钟采样。",
    },
    "a8w4_dispatch_end_S": {
        "when": "当前专家的 Dispatch MTE3 写入已经完成。",
        "why": "给出 GMM1 输入真正可用的完成边界。",
        "proof": "MTE3_S 将写入完成依赖传给 Scalar 后采样。",
        "perturbation": "每专家一次 MTE3_S 桥接和时钟采样。",
    },
    "a8w4_gmm1_begin_S": {
        "when": "AIC 已通过上游 ready 等待，即将发出首个 GMM1 MMAD。",
        "why": "区分输入等待与 GMM1 计算。",
        "proof": "采样位于 WaitForUpstreamReady 之后、blockMmad 之前。",
        "perturbation": "每专家每活跃 AIC 一次时钟采样。",
    },
    "a8w4_gmm1_end_S": {
        "when": "当前专家分配给该 AIC 的 GMM1 tile 已完成发布。",
        "why": "给出 GMM1 的实际结束边界。",
        "proof": "每个 tile 均在 FIX_S 后发布，采样位于计算函数返回之后。",
        "perturbation": "每专家每活跃 AIC 一次时钟采样。",
    },
    "a8w4_gmm1_prologue_begin_S": {
        "when": "AIV0 即将发出当前专家首个 GMM1 权重 prologue。",
        "why": "单独观察 W4 到 W8 权重展开的占用。",
        "proof": "采样位于首个 blockPrologue 之前。",
        "perturbation": "每专家每活跃 AIV0 一次时钟采样。",
    },
    "a8w4_gmm1_prologue_end_S": {
        "when": "AIV0 已完成当前专家全部 GMM1 权重 prologue。",
        "why": "观察完整的 W4 到 W8 权重展开与 L1 写入时间。",
        "proof": "prologue 返回并回收事件环后，MTE3_S 将最终 L1 写入完成依赖传到 Scalar。",
        "perturbation": "每专家每活跃 AIV0 一次 MTE3→Scalar 桥接和时钟采样。",
    },
    "a8w4_swiglu_begin_S": {
        "when": "AIV1 已观察到首个 GMM1 输出 ready，即将加载并执行 SwiGLU。",
        "why": "把 GMM1 handoff 等待与 SwiGLU 本体分开。",
        "proof": "采样位于 ready 轮询之后、首次输出加载之前。",
        "perturbation": "每专家每活跃 AIV1 一次时钟采样。",
    },
    "a8w4_swiglu_end_S": {
        "when": "AIV1 已完成当前专家最后一次 SwiGLU 与 MX 量化输出。",
        "why": "观察完整的 SwiGLU、量化和 GM 输出时间。",
        "proof": "BlockEpilogueSwigluMxQuant 每次返回前已有 MTE3_S，最后一次调用后的 Scalar 时钟可直接作为 actual end。",
        "perturbation": "每专家每活跃 AIV1 一次时钟采样。",
    },
    "a8w4_gmm2_begin_S": {
        "when": "AIC 已观察到 SwiGLU 输出 ready，即将发出首个 GMM2 MMAD。",
        "why": "区分 SwiGLU handoff 等待与 GMM2 计算。",
        "proof": "采样位于上游 ready 等待之后、blockMmad 之前。",
        "perturbation": "每专家每活跃 AIC 一次时钟采样。",
    },
    "a8w4_gmm2_end_S": {
        "when": "当前专家分配给该 AIC 的 GMM2 tile 已完成发布。",
        "why": "给出 GMM2 的实际结束边界。",
        "proof": "每个 tile 均在 FIX_S 后发布，采样位于计算函数返回之后。",
        "perturbation": "每专家每活跃 AIC 一次时钟采样。",
    },
    "a8w4_gmm2_prologue_begin_S": {
        "when": "AIV0 即将发出当前专家首个 GMM2 权重 prologue。",
        "why": "观察 GMM2 权重展开是否落在关键路径。",
        "proof": "采样位于首个 blockPrologue 之前。",
        "perturbation": "每专家每活跃 AIV0 一次时钟采样。",
    },
    "a8w4_gmm2_prologue_end_S": {
        "when": "AIV0 已完成当前专家全部 GMM2 权重 prologue。",
        "why": "观察完整的 GMM2 W4 到 W8 权重展开与 L1 写入时间。",
        "proof": "prologue 返回并回收事件环后，MTE3_S 将最终 L1 写入完成依赖传到 Scalar。",
        "perturbation": "每专家每活跃 AIV0 一次 MTE3→Scalar 桥接和时钟采样。",
    },
    "a8w4_combine_begin_S": {
        "when": "AIV1 已观察到首个 GMM2 输出 group/tile ready，即将开始 Combine。",
        "why": "显式观察 GMM2 到 Combine 的 handoff。",
        "proof": "采样位于 GM sequence 或 group counter ready 之后。",
        "perturbation": "每专家每活跃 AIV1 一次时钟采样。",
    },
    "a8w4_combine_end_S": {
        "when": "当前专家最后一次 Combine 发送的完成依赖已到达 Scalar。",
        "why": "让区间结束于实际发送完成，而非仅发射完成。",
        "proof": "MTE2_S 消费 Combine 返回时保留的完成依赖后采样。",
        "perturbation": "每专家一次 MTE2_S 桥接和时钟采样。",
    },
    "a8w4_combine_tile_wait_begin_S": {
        "when": "AIV1 即将轮询当前 Combine tile 的 GMM2 ready sequence。",
        "why": "量化每个 tile 实际等待 GMM2 的时间。",
        "proof": "采样直接位于 gmmToEpilogueFlag 轮询之前。",
        "perturbation": "每个本核 Combine tile 一次时钟采样。",
    },
    "a8w4_combine_tile_wait_end_S": {
        "when": "当前 Combine tile 的 GMM2 ready sequence 已可见。",
        "why": "精确关闭逐 tile 等待区间。",
        "proof": "采样直接位于 ready 轮询之后。",
        "perturbation": "与 tile work begin 共用一次时钟采样。",
    },
    "a8w4_combine_tile_begin_S": {
        "when": "当前 GMM2 tile 已 ready，即将开始 GM 到 UB 搬运和 Combine。",
        "why": "将等待空泡与 Combine 工作分开。",
        "proof": "采样位于 ready 轮询之后、首次 GM 到 UB 搬运之前。",
        "perturbation": "与 tile wait end 共用一次时钟采样。",
    },
    "a8w4_combine_tile_end_S": {
        "when": "当前 Combine tile 的远端 MTE3 写出完成依赖已到达 Scalar。",
        "why": "得到逐 tile Combine 工作的实际完成边界。",
        "proof": "依次经过 MTE3_MTE2 和 MTE2_S 后采样。",
        "perturbation": "每个本核 Combine tile 一次 MTE2_S 桥接和时钟采样。",
    },
    "a8w4_gmm1_wave_wait_begin_S": {
        "when": "AIC 首次为当前专家 M-wave 检查 Dispatch ready。",
        "why": "量化 Dispatch 到 GMM1 的逐 M-wave 首次等待。",
        "proof": "采样直接位于 WaitForUpstreamReady 之前。",
        "perturbation": "每个活跃专家 M-wave 保留一次时钟。",
    },
    "a8w4_gmm1_wave_wait_end_S": {
        "when": "当前专家 M-wave 的 Dispatch 行数条件已经满足。",
        "why": "精确关闭逐 M-wave 首次等待。",
        "proof": "采样直接位于 WaitForUpstreamReady 之后。",
        "perturbation": "与 wave envelope begin 共用一次时钟。",
    },
    "a8w4_gmm1_wave_begin_S": {
        "when": "当前专家 M-wave 的首个本核 tile 即将发出 MMAD。",
        "why": "建立该 M-wave 活跃包络的起点。",
        "proof": "采样在 readiness 后、blockMmad 前。",
        "perturbation": "与 wave wait end 共用一次时钟。",
    },
    "a8w4_gmm1_wave_end_S": {
        "when": "当前专家 M-wave 最新一个本核 tile 已完成 FIX_S 发布。",
        "why": "建立该 M-wave 活跃包络的终点。",
        "proof": "采样位于每个 tile 已有 FIX_S 和 ready 发布之后，最终保留最新值。",
        "perturbation": "每个本核 GMM1 tile 一次时钟；包络可能包含 swizzle 穿插的其他 wave。",
    },
}

SPAN_LABEL_ZH = {
    "process": "角色有效工作",
    "dispatch": "Dispatch",
    "gmm1": "GMM1",
    "swiglu": "SwiGLU + MX 量化",
    "gmm2": "GMM2 + ready 发布",
    "gmm1_expert": "GMM1（逐专家 issue）",
    "swiglu_expert": "SwiGLU + MX 量化（逐专家）",
    "gmm2_expert": "GMM2 + ready 发布（逐专家）",
    "combine": "ready 汇聚 + Combine",
    "a8w4_process": "A8W4 角色有效工作",
    "a8w4_dispatch": "逐专家 Dispatch",
    "a8w4_gmm1": "逐专家 GMM1",
    "a8w4_gmm1_prologue": "GMM1 权重 prologue",
    "a8w4_swiglu": "SwiGLU",
    "a8w4_gmm2": "逐专家 GMM2",
    "a8w4_gmm2_prologue": "GMM2 权重 prologue",
    "a8w4_combine": "逐专家 Combine",
    "a8w4_combine_tile_wait": "Combine tile wait",
    "a8w4_combine_tile_work": "Combine tile work",
    "a8w4_gmm1_wave_wait": "GMM1 wave wait",
    "a8w4_gmm1_wave_work": "GMM1 wave active envelope",
}


class ContractError(RuntimeError):
    """Raised when a spec, source audit, or normalized trace is unsafe."""


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ContractError(f"file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ContractError(f"invalid JSON in {path}: {exc}") from exc


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _spec_sha256(spec: dict[str, Any]) -> str:
    return _sha256_bytes(_canonical_json_bytes(spec))


def _expected_schema_id(spec: dict[str, Any]) -> int:
    """Derive a stable uint32 id from wire and timeline semantics, not source lines."""
    trace = spec.get("trace", {})
    protocol = trace.get("protocol", {})
    protocol_contract = {
        key: protocol.get(key)
        for key in (
            "magic",
            "version",
            "word_bits",
            "endianness",
            "header_words",
            "record_words",
            "record_order",
            "record_order_description",
            "header_fields",
            "record_fields",
        )
    }
    if "fine_metadata_tail" in protocol:
        protocol_contract["fine_metadata_tail"] = protocol["fine_metadata_tail"]
    payload = {
        "trace_name": trace.get("name"),
        "roles": trace.get("roles"),
        "clock": {
            key: trace.get("clock", {}).get(key)
            for key in (
                "unit",
                "bits",
                "timestamp_pipeline",
                "scope",
                "comparable_across_records",
                "missing_value",
            )
        },
        "protocol": protocol_contract,
        "points": [
            {
                key: point.get(key)
                for key in (
                    "id",
                    "stage",
                    "edge",
                    "boundary_quality",
                    "roles",
                    "scope",
                )
            }
            | {
                "producer_pipeline": point.get("dependency", {}).get(
                    "producer_pipeline"
                ),
                "timestamp_pipeline": point.get("dependency", {}).get(
                    "timestamp_pipeline"
                ),
                "bridge": point.get("dependency", {}).get("bridge"),
            }
            for point in spec.get("points", [])
        ],
        "spans": [
            {key: span.get(key) for key in ("id", "begin", "end", "roles", "quality")}
            for span in spec.get("spans", [])
        ],
    }
    if "fine_grained_compile_switch" in trace:
        payload["fine_grained_compile_switch"] = trace["fine_grained_compile_switch"]
    return zlib.crc32(_canonical_json_bytes(payload)) & 0xFFFFFFFF


def _contains_todo(value: Any, path: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, str) and "TODO" in value.upper():
        found.append(path)
    elif isinstance(value, dict):
        for key, child in value.items():
            found.extend(_contains_todo(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_contains_todo(child, f"{path}[{index}]"))
    return found


def _parse_uint(value: Any, field: str, bits: int = 32) -> int:
    if isinstance(value, bool):
        raise ContractError(f"{field} must be an unsigned integer, not boolean")
    try:
        parsed = int(value, 0) if isinstance(value, str) else int(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{field} must be an unsigned integer: {value!r}") from exc
    if parsed < 0 or parsed >= (1 << bits):
        raise ContractError(f"{field}={parsed} does not fit uint{bits}")
    return parsed


def _resolve_record_words(
    protocol: dict[str, Any], header: dict[str, Any], location: str = "record_words"
) -> int:
    layout = protocol.get("record_words")
    if isinstance(layout, int) and not isinstance(layout, bool) and layout > 0:
        return layout
    if not isinstance(layout, dict):
        raise ContractError(f"{location} must be a positive integer or dynamic layout")
    base = layout.get("base_words")
    if not isinstance(base, int) or isinstance(base, bool) or base < 0:
        raise ContractError(
            f"{location} dynamic layout requires non-negative base_words"
        )
    terms = layout.get("terms")
    if terms is None:
        terms = [layout]
    if not isinstance(terms, list) or not terms:
        raise ContractError(f"{location}.terms must be a non-empty list")
    result = base
    for index, term in enumerate(terms):
        term_location = f"{location}.terms[{index}]"
        if not isinstance(term, dict):
            raise ContractError(f"{term_location} must be an object")
        per_count = term.get("words_per_count")
        count_field = term.get("count_header_field")
        if (
            not isinstance(per_count, int)
            or isinstance(per_count, bool)
            or per_count <= 0
            or not isinstance(count_field, str)
            or not count_field
        ):
            raise ContractError(
                f"{term_location} requires positive words_per_count and count_header_field"
            )
        count = header.get(count_field)
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ContractError(
                f"{location} cannot resolve non-negative header field {count_field!r}: {count!r}"
            )
        result += per_count * count
    return result


def _resolve_field_count(
    field: dict[str, Any], header: dict[str, Any], location: str = "record field count"
) -> int:
    count_layout = field.get("count")
    if (
        isinstance(count_layout, int)
        and not isinstance(count_layout, bool)
        and count_layout > 0
    ):
        return count_layout
    if not isinstance(count_layout, dict):
        raise ContractError(f"{location} must be a positive integer or dynamic count")
    count_field = count_layout.get("header_field")
    if not isinstance(count_field, str) or not count_field:
        raise ContractError(f"{location} dynamic count requires header_field")
    count = header.get(count_field)
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ContractError(
            f"{location} cannot resolve non-negative header field {count_field!r}: {count!r}"
        )
    return count


def _resolve_field_word(
    field: dict[str, Any], header: dict[str, Any], location: str = "record field word"
) -> int:
    word_layout = field.get("word")
    if (
        isinstance(word_layout, int)
        and not isinstance(word_layout, bool)
        and word_layout >= 0
    ):
        return word_layout
    if not isinstance(word_layout, dict):
        raise ContractError(f"{location} must be a non-negative integer or dynamic layout")
    base = word_layout.get("base_words")
    terms = word_layout.get("terms")
    if (
        not isinstance(base, int)
        or isinstance(base, bool)
        or base < 0
        or not isinstance(terms, list)
        or not terms
    ):
        raise ContractError(
            f"{location} dynamic layout requires non-negative base_words and non-empty terms"
        )
    result = base
    for index, term in enumerate(terms):
        term_location = f"{location}.terms[{index}]"
        if not isinstance(term, dict):
            raise ContractError(f"{term_location} must be an object")
        words_per_count = term.get("words_per_count")
        count_field = term.get("count_header_field")
        if (
            not isinstance(words_per_count, int)
            or isinstance(words_per_count, bool)
            or words_per_count <= 0
            or not isinstance(count_field, str)
            or not count_field
        ):
            raise ContractError(
                f"{term_location} requires positive words_per_count and count_header_field"
            )
        count = header.get(count_field)
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ContractError(
                f"{location} cannot resolve non-negative header field {count_field!r}: {count!r}"
            )
        result += words_per_count * count
    return result


def _record_words_text(
    protocol: dict[str, Any], header: dict[str, Any] | None = None
) -> str:
    layout = protocol["record_words"]
    if isinstance(layout, int):
        return str(layout)
    terms = layout.get("terms", [layout])
    formula = " + ".join(
        [str(layout["base_words"])]
        + [
            f"{term['words_per_count']} × header.{term['count_header_field']}"
            for term in terms
        ]
    )
    if header is None:
        return formula
    return f"{_resolve_record_words(protocol, header)}（{formula}）"


def _required_string(
    obj: Any, key: str, location: str, errors: list[str]
) -> str | None:
    if not isinstance(obj, dict):
        errors.append(f"{location} must be an object")
        return None
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{location}.{key} must be a non-empty string")
        return None
    return value


def _validate_spec(spec: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(spec, dict):
        return ["spec root must be an object"]
    if spec.get("schema_version") != SPEC_SCHEMA_VERSION:
        errors.append(
            f"$.schema_version must be {SPEC_SCHEMA_VERSION}, got {spec.get('schema_version')!r}"
        )

    todo_paths = _contains_todo(spec)
    if todo_paths:
        preview = ", ".join(todo_paths[:12])
        suffix = " ..." if len(todo_paths) > 12 else ""
        errors.append(f"unresolved TODO placeholders: {preview}{suffix}")

    trace = spec.get("trace")
    if not isinstance(trace, dict):
        errors.append("$.trace must be an object")
        return errors

    trace_name = _required_string(trace, "name", "$.trace", errors)
    if trace_name and not TRACE_NAME_RE.fullmatch(trace_name):
        errors.append("$.trace.name must contain only letters, digits, '.', '_' or '-'")
    compile_switch = _required_string(trace, "compile_switch", "$.trace", errors)
    if compile_switch and not COMPILE_SWITCH_RE.fullmatch(compile_switch):
        errors.append("$.trace.compile_switch must be a C/C++ identifier")
    fine_compile_switch = trace.get("fine_grained_compile_switch")
    if fine_compile_switch is not None and (
        not isinstance(fine_compile_switch, str)
        or not COMPILE_SWITCH_RE.fullmatch(fine_compile_switch)
    ):
        errors.append(
            "$.trace.fine_grained_compile_switch must be a C/C++ identifier when set"
        )

    roles = trace.get("roles")
    if (
        not isinstance(roles, list)
        or not roles
        or any(not isinstance(role, str) or not role.strip() for role in roles)
    ):
        errors.append("$.trace.roles must be a non-empty list of strings")
        roles = []
    elif len(set(roles)) != len(roles):
        errors.append("$.trace.roles must be unique")
    role_set = set(roles)

    clock = trace.get("clock")
    if not isinstance(clock, dict):
        errors.append("$.trace.clock must be an object")
        clock = {}
    for key in ("api", "unit", "timestamp_pipeline", "scope"):
        _required_string(clock, key, "$.trace.clock", errors)
    bits = clock.get("bits")
    if bits not in (32, 64):
        errors.append("$.trace.clock.bits must be 32 or 64")
    if not isinstance(clock.get("comparable_across_records"), bool):
        errors.append("$.trace.clock.comparable_across_records must be boolean")
    visualization = trace.get("visualization", {})
    if not isinstance(visualization, dict):
        errors.append("$.trace.visualization must be an object when set")
        visualization = {}
    comparison_group = visualization.get("comparison_group")
    if comparison_group is not None:
        if not isinstance(comparison_group, dict):
            errors.append("$.trace.visualization.comparison_group must be an object")
        else:
            if comparison_group.get("kind") != "shared-core-index":
                errors.append(
                    "$.trace.visualization.comparison_group.kind must be shared-core-index"
                )
            if comparison_group.get("key") != "core":
                errors.append("$.trace.visualization.comparison_group.key must be core")
            _required_string(
                comparison_group,
                "label",
                "$.trace.visualization.comparison_group",
                errors,
            )
            group_roles = comparison_group.get("roles")
            if group_roles != roles:
                errors.append(
                    "$.trace.visualization.comparison_group.roles must exactly match trace.roles"
                )
            if comparison_group.get("reference_role") not in role_set:
                errors.append(
                    "$.trace.visualization.comparison_group.reference_role must name a trace role"
                )
            if comparison_group.get("cross_group_comparable") is not False:
                errors.append(
                    "$.trace.visualization.comparison_group.cross_group_comparable must be false"
                )
    missing = clock.get("missing_value")
    if missing is not None and isinstance(bits, int):
        try:
            _parse_uint(missing, "$.trace.clock.missing_value", bits)
        except ContractError as exc:
            errors.append(str(exc))

    protocol = trace.get("protocol")
    if not isinstance(protocol, dict):
        errors.append("$.trace.protocol must be an object")
        protocol = {}
    for key in ("magic", "schema_id", "header_contract"):
        _required_string(protocol, key, "$.trace.protocol", errors)
    for key in ("magic", "schema_id"):
        if key in protocol:
            try:
                _parse_uint(protocol[key], f"$.trace.protocol.{key}")
            except ContractError as exc:
                errors.append(str(exc))
    version = protocol.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or version <= 0:
        errors.append("$.trace.protocol.version must be a positive integer")
    word_bits = protocol.get("word_bits")
    if word_bits != 32:
        errors.append("$.trace.protocol.word_bits must be 32")
    if protocol.get("endianness") not in ("little", "big"):
        errors.append("$.trace.protocol.endianness must be little or big")
    header_words = protocol.get("header_words")
    if (
        not isinstance(header_words, int)
        or isinstance(header_words, bool)
        or header_words <= 0
    ):
        errors.append("$.trace.protocol.header_words must be a positive integer")
    record_words = protocol.get("record_words")
    validation_record_header: dict[str, int] = {}
    if isinstance(record_words, dict):
        for term in record_words.get("terms", [record_words]):
            if isinstance(term, dict) and isinstance(term.get("count_header_field"), str):
                validation_record_header[term["count_header_field"]] = 3
    try:
        _resolve_record_words(
            protocol,
            validation_record_header,
            "$.trace.protocol.record_words",
        )
    except ContractError as exc:
        errors.append(str(exc))
    record_order = _required_string(
        protocol, "record_order", "$.trace.protocol", errors
    )
    if record_order and record_order not in ("role-major", "custom"):
        errors.append("$.trace.protocol.record_order must be role-major or custom")
    _required_string(protocol, "record_order_description", "$.trace.protocol", errors)

    fine_tail = protocol.get("fine_metadata_tail")
    if fine_tail is not None:
        tail_loc = "$.trace.protocol.fine_metadata_tail"
        if not isinstance(fine_tail, dict):
            errors.append(f"{tail_loc} must be an object when set")
        else:
            words_per_core = fine_tail.get("words_per_core")
            core_stride_words = fine_tail.get("core_stride_words")
            if (
                not isinstance(words_per_core, int)
                or isinstance(words_per_core, bool)
                or words_per_core <= 0
            ):
                errors.append(f"{tail_loc}.words_per_core must be a positive integer")
            if core_stride_words != words_per_core:
                errors.append(f"{tail_loc}.core_stride_words must equal words_per_core")
            for key in ("base_word_formula", "total_words_formula"):
                _required_string(fine_tail, key, tail_loc, errors)
            tail_fields = fine_tail.get("fields")
            if not isinstance(tail_fields, list) or not tail_fields:
                errors.append(f"{tail_loc}.fields must be a non-empty list")
                tail_fields = []
            occupied_tail_words: dict[int, str] = {}
            for field_index, tail_field in enumerate(tail_fields):
                field_loc = f"{tail_loc}.fields[{field_index}]"
                if not isinstance(tail_field, dict):
                    errors.append(f"{field_loc} must be an object")
                    continue
                field_name = _required_string(tail_field, "name", field_loc, errors)
                writer_role = tail_field.get("writer_role")
                if writer_role not in role_set:
                    errors.append(f"{field_loc}.writer_role must name a trace role")
                word_in_core = tail_field.get("word_in_core")
                if (
                    not isinstance(word_in_core, int)
                    or isinstance(word_in_core, bool)
                    or word_in_core < 0
                    or not isinstance(words_per_core, int)
                    or word_in_core >= words_per_core
                ):
                    errors.append(
                        f"{field_loc}.word_in_core must be inside words_per_core"
                    )
                elif word_in_core in occupied_tail_words:
                    errors.append(
                        f"{field_loc} overlaps tail word {word_in_core} used by "
                        f"{occupied_tail_words[word_in_core]}"
                    )
                elif field_name:
                    occupied_tail_words[word_in_core] = field_name
                packing = tail_field.get("packing")
                if not isinstance(packing, list) or not packing:
                    errors.append(f"{field_loc}.packing must be a non-empty list")
                    continue
                occupied_bits: set[int] = set()
                for packed_index, packed in enumerate(packing):
                    packed_loc = f"{field_loc}.packing[{packed_index}]"
                    if not isinstance(packed, dict):
                        errors.append(f"{packed_loc} must be an object")
                        continue
                    _required_string(packed, "name", packed_loc, errors)
                    bit = packed.get("bit")
                    width = packed.get("width")
                    if (
                        not isinstance(bit, int)
                        or isinstance(bit, bool)
                        or bit < 0
                        or not isinstance(width, int)
                        or isinstance(width, bool)
                        or width <= 0
                        or bit + width > 32
                    ):
                        errors.append(f"{packed_loc} must fit inside one uint32 word")
                        continue
                    for occupied_bit in range(bit, bit + width):
                        if occupied_bit in occupied_bits:
                            errors.append(f"{packed_loc} overlaps bit {occupied_bit}")
                        occupied_bits.add(occupied_bit)
                    count = packed.get("count")
                    element_bits = packed.get("element_bits")
                    if (count is None) != (element_bits is None):
                        errors.append(
                            f"{packed_loc}.count and element_bits must be supplied together"
                        )
                    elif count is not None and (
                        not isinstance(count, int)
                        or isinstance(count, bool)
                        or count <= 0
                        or not isinstance(element_bits, int)
                        or isinstance(element_bits, bool)
                        or element_bits <= 0
                        or count * element_bits != width
                        or not isinstance(packed.get("normalized_prefix"), str)
                        or not packed["normalized_prefix"]
                    ):
                        errors.append(
                            f"{packed_loc} repeated field must have positive count/element_bits, "
                            "count*element_bits=width, and normalized_prefix"
                        )

    header_fields = protocol.get("header_fields")
    if not isinstance(header_fields, list) or not header_fields:
        errors.append("$.trace.protocol.header_fields must be a non-empty list")
        header_fields = []
    header_names: set[str] = set()
    occupied_header_words: dict[int, str] = {}
    for index, field in enumerate(header_fields):
        loc = f"$.trace.protocol.header_fields[{index}]"
        if not isinstance(field, dict):
            errors.append(f"{loc} must be an object")
            continue
        name = _required_string(field, "name", loc, errors)
        if name:
            if not FIELD_NAME_RE.fullmatch(name):
                errors.append(f"{loc}.name must be a C-like identifier")
            if name in header_names:
                errors.append(f"duplicate header field name: {name}")
            header_names.add(name)
        word = field.get("word")
        width = field.get("width_words")
        if not isinstance(word, int) or isinstance(word, bool) or word < 0:
            errors.append(f"{loc}.word must be a non-negative integer")
            continue
        if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
            errors.append(f"{loc}.width_words must be a positive integer")
            continue
        if isinstance(header_words, int) and word + width > header_words:
            errors.append(f"{loc} exceeds header_words={header_words}")
        for occupied_word in range(word, word + width):
            previous = occupied_header_words.get(occupied_word)
            if previous is not None:
                errors.append(
                    f"{loc} overlaps header word {occupied_word} already used by {previous}"
                )
            elif name:
                occupied_header_words[occupied_word] = name
    missing_header_fields = sorted(REQUIRED_HEADER_FIELDS - header_names)
    if missing_header_fields:
        errors.append(
            "$.trace.protocol.header_fields lacks required fields: "
            + ", ".join(missing_header_fields)
        )
    header_field_by_name = {
        field.get("name"): field for field in header_fields if isinstance(field, dict)
    }
    for required_name in REQUIRED_HEADER_FIELDS - {"role_counts"}:
        field = header_field_by_name.get(required_name)
        if isinstance(field, dict) and field.get("width_words") != 1:
            errors.append(
                f"$.trace.protocol.header_fields field {required_name!r} must have width_words=1"
            )
    role_counts_field = header_field_by_name.get("role_counts")
    if isinstance(role_counts_field, dict) and role_counts_field.get(
        "width_words"
    ) != len(roles):
        errors.append(
            "$.trace.protocol.header_fields field 'role_counts' must have width_words "
            f"equal to len(trace.roles)={len(roles)}"
        )
    if isinstance(record_words, dict):
        for index, term in enumerate(record_words.get("terms", [record_words])):
            count_field = term.get("count_header_field") if isinstance(term, dict) else None
            if count_field is not None and count_field not in header_names:
                errors.append(
                    f"$.trace.protocol.record_words.terms[{index}].count_header_field "
                    "must name a header field"
                )

    output = trace.get("output")
    if not isinstance(output, dict):
        errors.append("$.trace.output must be an object")
        output = {}
    for key in ("carrier", "capacity_check", "write_phase", "rendezvous"):
        _required_string(output, key, "$.trace.output", errors)
    if not isinstance(output.get("overwrites_user_output"), bool):
        errors.append("$.trace.output.overwrites_user_output must be boolean")

    points = spec.get("points")
    if not isinstance(points, list) or not points:
        errors.append("$.points must be a non-empty list")
        points = []
    point_by_id: dict[str, dict[str, Any]] = {}
    for index, point in enumerate(points):
        loc = f"$.points[{index}]"
        if not isinstance(point, dict):
            errors.append(f"{loc} must be an object")
            continue
        point_id = _required_string(point, "id", loc, errors)
        if point_id:
            if not POINT_ID_RE.fullmatch(point_id):
                errors.append(f"{loc}.id must end in _S and be a C-like identifier")
            if point_id in point_by_id:
                errors.append(f"duplicate point id: {point_id}")
            else:
                point_by_id[point_id] = point
        _required_string(point, "stage", loc, errors)
        for key in ("scope", "why", "semantics", "perturbation"):
            _required_string(point, key, loc, errors)

        edge = point.get("edge")
        if edge not in ("begin", "end", "instant"):
            errors.append(f"{loc}.edge must be begin, end, or instant")
        quality = point.get("boundary_quality")
        allowed_quality = {
            "begin": {"causal", "inferred"},
            "end": {"actual", "issue", "inferred"},
            "instant": {"causal", "actual", "issue", "inferred"},
        }
        if edge in allowed_quality and quality not in allowed_quality[edge]:
            errors.append(
                f"{loc}.boundary_quality={quality!r} is invalid for edge={edge!r}"
            )

        point_roles = point.get("roles")
        if not isinstance(point_roles, list) or not point_roles:
            errors.append(f"{loc}.roles must be a non-empty list")
        elif any(role not in role_set for role in point_roles):
            errors.append(f"{loc}.roles must be a subset of $.trace.roles")

        source = point.get("source")
        if not isinstance(source, dict):
            errors.append(f"{loc}.source must be an object")
            source = {}
        source_file = _required_string(source, "file", f"{loc}.source", errors)
        if source_file:
            source_path = Path(source_file)
            if source_path.is_absolute() or ".." in source_path.parts:
                errors.append(f"{loc}.source.file must be a safe repo-relative path")
        _required_string(source, "symbol", f"{loc}.source", errors)
        placement = source.get("placement")
        if placement not in ("before", "after", "at"):
            errors.append(f"{loc}.source.placement must be before, after, or at")
        marker = _required_string(source, "marker", f"{loc}.source", errors)
        if marker and point_id and marker != f"TRACE_POINT: {point_id}":
            errors.append(f"{loc}.source.marker must equal 'TRACE_POINT: {point_id}'")
        distance = source.get("max_line_distance")
        if (
            not isinstance(distance, int)
            or isinstance(distance, bool)
            or distance < 0
            or distance > 200
        ):
            errors.append(
                f"{loc}.source.max_line_distance must be an integer in [0, 200]"
            )
        anchor = source.get("target_anchor")
        if not isinstance(anchor, dict):
            errors.append(f"{loc}.source.target_anchor must be an object")
        else:
            if anchor.get("mode") not in ("literal", "regex"):
                errors.append(
                    f"{loc}.source.target_anchor.mode must be literal or regex"
                )
            _required_string(anchor, "pattern", f"{loc}.source.target_anchor", errors)
            occurrence = anchor.get("occurrence")
            if occurrence is not None and (
                not isinstance(occurrence, int)
                or isinstance(occurrence, bool)
                or occurrence <= 0
            ):
                errors.append(
                    f"{loc}.source.target_anchor.occurrence must be a positive integer when set"
                )

        dependency = point.get("dependency")
        if not isinstance(dependency, dict):
            errors.append(f"{loc}.dependency must be an object")
        else:
            for key in (
                "precondition",
                "producer_pipeline",
                "timestamp_pipeline",
                "bridge",
                "proof",
            ):
                _required_string(dependency, key, f"{loc}.dependency", errors)

    record_fields = protocol.get("record_fields")
    if not isinstance(record_fields, list) or not record_fields:
        errors.append("$.trace.protocol.record_fields must be a non-empty list")
        record_fields = []
    timestamp_coverage = {
        (point_id, role): 0
        for point_id, point in point_by_id.items()
        for role in point.get("roles", [])
    }
    occupied_record_words: dict[str, dict[int, str]] = {role: {} for role in roles}
    clock_width_words = (
        (bits + word_bits - 1) // word_bits
        if isinstance(bits, int) and isinstance(word_bits, int) and word_bits > 0
        else None
    )
    for index, field in enumerate(record_fields):
        loc = f"$.trace.protocol.record_fields[{index}]"
        if not isinstance(field, dict):
            errors.append(f"{loc} must be an object")
            continue
        name = _required_string(field, "name", loc, errors)
        kind = field.get("kind")
        if kind not in ("timestamp", "counter"):
            errors.append(f"{loc}.kind must be timestamp or counter")
        field_roles = field.get("roles")
        if not isinstance(field_roles, list) or not field_roles:
            errors.append(f"{loc}.roles must be a non-empty list")
            field_roles = []
        elif len(set(field_roles)) != len(field_roles):
            errors.append(f"{loc}.roles must be unique")
        elif any(role not in role_set for role in field_roles):
            errors.append(f"{loc}.roles must be a subset of $.trace.roles")
        if kind == "timestamp" and name:
            point = point_by_id.get(name)
            if point is None:
                errors.append(f"{loc}.name references unknown timestamp point {name!r}")
            else:
                unsupported = [
                    role for role in field_roles if role not in point.get("roles", [])
                ]
                if unsupported:
                    errors.append(
                        f"{loc}.roles {unsupported} are not declared by timestamp point {name!r}"
                    )
                for role in field_roles:
                    key = (name, role)
                    if key in timestamp_coverage:
                        timestamp_coverage[key] += 1
        elif kind == "counter" and name and not FIELD_NAME_RE.fullmatch(name):
            errors.append(f"{loc}.name must be a C-like identifier")

        width = field.get("width_words")
        count_layout = field.get("count")
        stride = field.get("stride_words")
        validation_header = {name: 3 for name in header_names}
        try:
            word = _resolve_field_word(field, validation_header, f"{loc}.word")
        except ContractError as exc:
            errors.append(str(exc))
            word = None
        try:
            count = _resolve_field_count(field, validation_header, f"{loc}.count")
        except ContractError as exc:
            errors.append(str(exc))
            count = None
        if isinstance(count_layout, dict):
            count_field = count_layout.get("header_field")
            if count_field not in header_names:
                errors.append(f"{loc}.count.header_field must name a header field")
        integers = (
            isinstance(word, int) and not isinstance(word, bool) and word >= 0,
            isinstance(width, int) and not isinstance(width, bool) and width > 0,
            isinstance(count, int) and not isinstance(count, bool) and count > 0,
            isinstance(stride, int) and not isinstance(stride, bool) and stride > 0,
        )
        for valid, key, requirement in zip(
            integers,
            ("word", "width_words", "count", "stride_words"),
            (
                "a non-negative integer",
                "a positive integer",
                "a positive integer",
                "a positive integer",
            ),
        ):
            if not valid:
                errors.append(f"{loc}.{key} must be {requirement}")
        if not all(integers):
            continue
        if kind == "counter" and count != 1:
            errors.append(f"{loc}.count must be 1 for counter fields")
        if (
            kind == "timestamp"
            and clock_width_words is not None
            and width < clock_width_words
        ):
            errors.append(
                f"{loc}.width_words={width} cannot hold a {bits}-bit timestamp in {word_bits}-bit words"
            )
        last_end = word + (count - 1) * stride + width
        try:
            validation_record_words = _resolve_record_words(
                protocol, validation_header, "$.trace.protocol.record_words"
            )
        except ContractError:
            validation_record_words = None
        if validation_record_words is not None and last_end > validation_record_words:
            errors.append(
                f"{loc} exceeds resolved record_words={validation_record_words}"
            )
        for role in field_roles:
            role_occupied = occupied_record_words.setdefault(role, {})
            for repetition in range(count):
                start_word = word + repetition * stride
                for occupied_word in range(start_word, start_word + width):
                    previous = role_occupied.get(occupied_word)
                    if previous is not None:
                        errors.append(
                            f"{loc} overlaps {role} record word {occupied_word} already used by {previous}"
                        )
                    elif name:
                        role_occupied[occupied_word] = f"{name}[{repetition}]"
    for (point_id, role), coverage in timestamp_coverage.items():
        if coverage != 1:
            errors.append(
                f"timestamp point {point_id!r} for role {role!r} must have exactly one "
                f"record_fields mapping, found {coverage}"
            )

    spans = spec.get("spans")
    if not isinstance(spans, list) or not spans:
        errors.append("$.spans must be a non-empty list")
        spans = []
    span_ids: set[str] = set()
    for index, span in enumerate(spans):
        loc = f"$.spans[{index}]"
        if not isinstance(span, dict):
            errors.append(f"{loc} must be an object")
            continue
        span_id = _required_string(span, "id", loc, errors)
        if span_id:
            if span_id in span_ids:
                errors.append(f"duplicate span id: {span_id}")
            span_ids.add(span_id)
        _required_string(span, "label", loc, errors)
        begin = _required_string(span, "begin", loc, errors)
        end = _required_string(span, "end", loc, errors)
        if begin and begin not in point_by_id:
            errors.append(f"{loc}.begin references unknown point {begin!r}")
        if end and end not in point_by_id:
            errors.append(f"{loc}.end references unknown point {end!r}")
        span_roles = span.get("roles")
        if not isinstance(span_roles, list) or not span_roles:
            errors.append(f"{loc}.roles must be a non-empty list")
            span_roles = []
        elif any(role not in role_set for role in span_roles):
            errors.append(f"{loc}.roles must be a subset of $.trace.roles")
        quality = span.get("quality")
        if quality not in ("exact", "issue", "inferred"):
            errors.append(f"{loc}.quality must be exact, issue, or inferred")
        color = _required_string(span, "color", loc, errors)
        if color and not COLOR_RE.fullmatch(color):
            errors.append(f"{loc}.color must be a six-digit #RRGGBB value")
        if quality == "exact" and begin in point_by_id and end in point_by_id:
            begin_point = point_by_id[begin]
            end_point = point_by_id[end]
            if (
                begin_point.get("edge") != "begin"
                or begin_point.get("boundary_quality") != "causal"
            ):
                errors.append(
                    f"{loc} is exact but begin {begin!r} is not a causal begin"
                )
            if (
                end_point.get("edge") != "end"
                or end_point.get("boundary_quality") != "actual"
            ):
                errors.append(
                    f"{loc} is exact but end {end!r} is not an actual-completion end"
                )
            if any(role not in begin_point.get("roles", []) for role in span_roles):
                errors.append(
                    f"{loc}.roles are not all supported by begin point {begin!r}"
                )
            if any(role not in end_point.get("roles", []) for role in span_roles):
                errors.append(f"{loc}.roles are not all supported by end point {end!r}")

    fine_visualization = visualization.get("fine_grained")
    if fine_visualization is not None:
        fine_loc = "$.trace.visualization.fine_grained"
        if not isinstance(fine_visualization, dict):
            errors.append(f"{fine_loc} must be an object when set")
        else:
            _required_string(fine_visualization, "enabled_by", fine_loc, errors)
            if fine_visualization.get("occurrence_kind") != "expert-index":
                errors.append(f"{fine_loc}.occurrence_kind must be expert-index")
            for key in (
                "expert_count_header_field",
                "experts_per_wave_header_field",
                "wave_partition",
            ):
                value = _required_string(fine_visualization, key, fine_loc, errors)
                if key.endswith("header_field") and value and value not in header_names:
                    errors.append(f"{fine_loc}.{key} must name a header field")
            derived_gaps = fine_visualization.get("derived_idle_gaps")
            if not isinstance(derived_gaps, list) or not derived_gaps:
                errors.append(f"{fine_loc}.derived_idle_gaps must be a non-empty list")
            else:
                seen_gap_spans: set[str] = set()
                for gap_index, gap in enumerate(derived_gaps):
                    gap_loc = f"{fine_loc}.derived_idle_gaps[{gap_index}]"
                    if not isinstance(gap, dict):
                        errors.append(f"{gap_loc} must be an object")
                        continue
                    gap_span = _required_string(gap, "span", gap_loc, errors)
                    if gap_span and gap_span not in span_ids:
                        errors.append(
                            f"{gap_loc}.span references unknown span {gap_span!r}"
                        )
                    if gap_span in seen_gap_spans:
                        errors.append(f"{gap_loc}.span is duplicated")
                    elif gap_span:
                        seen_gap_spans.add(gap_span)
                    if gap.get("same_wave_only") is not True:
                        errors.append(f"{gap_loc}.same_wave_only must be true")
                    for key in ("label", "caveat"):
                        _required_string(gap, key, gap_loc, errors)
                    if gap.get("quality") != "inferred":
                        errors.append(f"{gap_loc}.quality must be inferred")
                    gap_color = _required_string(gap, "color", gap_loc, errors)
                    if gap_color and not COLOR_RE.fullmatch(gap_color):
                        errors.append(
                            f"{gap_loc}.color must be a six-digit #RRGGBB value"
                        )

    if not todo_paths:
        try:
            actual_schema_id = _parse_uint(
                protocol.get("schema_id"), "$.trace.protocol.schema_id"
            )
            expected_schema_id = _expected_schema_id(spec)
            if actual_schema_id != expected_schema_id:
                errors.append(
                    "$.trace.protocol.schema_id does not match the deterministic contract: "
                    f"actual=0x{actual_schema_id:08x}, expected=0x{expected_schema_id:08x}; "
                    "update the spec, device header, decoder, and fixtures together"
                )
        except ContractError as exc:
            errors.append(str(exc))
    return errors


def _load_valid_spec(path: Path) -> dict[str, Any]:
    spec = _load_json(path)
    errors = _validate_spec(spec)
    if errors:
        raise ContractError("invalid trace spec:\n  - " + "\n  - ".join(errors))
    return spec


def _replace_template_tokens(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, str):
        for old, new in replacements.items():
            value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [_replace_template_tokens(child, replacements) for child in value]
    if isinstance(value, dict):
        return {
            key: _replace_template_tokens(child, replacements)
            for key, child in value.items()
        }
    return value


def _cmd_init(args: argparse.Namespace) -> int:
    name = args.name.strip()
    if not TRACE_NAME_RE.fullmatch(name):
        raise ContractError(
            "--name must start with a letter/digit and use only letters, digits, '.', '_' or '-'"
        )
    output = args.output.expanduser().resolve()
    if output.exists():
        raise ContractError(f"refusing to overwrite existing spec: {output}")
    template_path = (
        Path(__file__).resolve().parent.parent / "assets" / "trace-spec.template.json"
    )
    template = _load_json(template_path)
    rendered = _replace_template_tokens(
        template,
        {
            "TODO_TRACE_NAME": name,
            "TODO_UNIQUE_U32_SCHEMA_ID": "0x00000000",
        },
    )
    schema_id = _expected_schema_id(rendered)
    rendered["trace"]["protocol"]["schema_id"] = f"0x{schema_id:08x}"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(rendered, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[OK] initialized trace spec: {output}")
    print(
        f"[INFO] template schema_id=0x{schema_id:08x}; validation will report the new "
        "deterministic id after TODO fields/layout are replaced"
    )
    print(
        "[NEXT] replace every TODO, run validate, adopt its expected schema id, then audit --phase plan"
    )
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    spec = _load_valid_spec(args.spec.expanduser().resolve())
    print(
        f"[OK] valid trace spec: {args.spec} "
        f"sha256={_spec_sha256(spec)} points={len(spec['points'])} spans={len(spec['spans'])}"
    )
    return 0


def _anchor_matches(
    lines: list[str], anchor: dict[str, Any], description: str
) -> list[int]:
    mode = anchor["mode"]
    pattern = anchor["pattern"]
    if mode == "literal":
        return [index for index, line in enumerate(lines, start=1) if pattern in line]
    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        raise ContractError(f"invalid regex for {description}: {exc}") from exc
    return [index for index, line in enumerate(lines, start=1) if compiled.search(line)]


def _select_match(matches: list[int], anchor: dict[str, Any], description: str) -> int:
    if not matches:
        raise ContractError(f"{description} did not match any source line")
    occurrence = anchor.get("occurrence")
    if occurrence is None:
        if len(matches) != 1:
            raise ContractError(
                f"{description} matched {len(matches)} lines {matches}; make the semantic anchor unique"
            )
        return matches[0]
    if occurrence > len(matches):
        raise ContractError(
            f"{description} occurrence {occurrence} exceeds {len(matches)} matches {matches}"
        )
    return matches[occurrence - 1]


def _run_git(repo_root: Path, arguments: list[str]) -> str | None:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _git_metadata(repo_root: Path) -> dict[str, Any]:
    revision = _run_git(repo_root, ["rev-parse", "HEAD"])
    branch = _run_git(repo_root, ["branch", "--show-current"])
    status_text = _run_git(repo_root, ["status", "--short"])
    full_status = status_text.splitlines() if status_text else []
    status = full_status[:200]
    diff_raw = _run_git(repo_root, ["diff", "--no-ext-diff", "--raw"])
    is_git_repository = revision is not None
    return {
        "is_git_repository": is_git_repository,
        "revision": revision,
        "branch": branch,
        "dirty": bool(full_status) if is_git_repository else None,
        "status_short": status,
        "status_truncated": len(full_status) > len(status),
        "tracked_diff_raw_sha256": (
            _sha256_bytes(diff_raw.encode("utf-8")) if diff_raw is not None else None
        ),
    }


def _source_context(
    lines: list[str], line_number: int, radius: int = 2
) -> dict[str, Any]:
    start = max(1, line_number - radius)
    end = min(len(lines), line_number + radius)
    return {
        "start_line": start,
        "end_line": end,
        "lines": [
            {"line": number, "text": lines[number - 1]}
            for number in range(start, end + 1)
        ],
    }


def _audit_point(point: dict[str, Any], repo_root: Path, phase: str) -> dict[str, Any]:
    source = point["source"]
    relative_file = Path(source["file"])
    source_path = (repo_root / relative_file).resolve()
    try:
        source_path.relative_to(repo_root)
    except ValueError as exc:
        raise ContractError(f"{point['id']}: source path escapes repo root") from exc
    if not source_path.is_file():
        raise ContractError(
            f"{point['id']}: source file does not exist: {relative_file}"
        )

    raw = source_path.read_bytes()
    try:
        source_text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ContractError(
            f"{point['id']}: source file is not UTF-8: {relative_file}"
        ) from exc
    lines = source_text.splitlines()
    target_matches = _anchor_matches(
        lines, source["target_anchor"], f"{point['id']} target anchor"
    )
    target_line = _select_match(
        target_matches, source["target_anchor"], f"{point['id']} target anchor"
    )

    marker_matches = [
        index for index, line in enumerate(lines, start=1) if source["marker"] in line
    ]
    marker_line: int | None = None
    marker_context: dict[str, Any] | None = None
    if phase == "implemented":
        marker_line = _select_match(
            marker_matches,
            {"mode": "literal", "pattern": source["marker"]},
            f"{point['id']} instrumentation marker",
        )
        distance = abs(marker_line - target_line)
        if distance > source["max_line_distance"]:
            raise ContractError(
                f"{point['id']}: marker line {marker_line} is {distance} lines from target "
                f"line {target_line}, exceeding max_line_distance={source['max_line_distance']}"
            )
        placement = source["placement"]
        if placement == "after" and marker_line < target_line:
            raise ContractError(
                f"{point['id']}: marker line {marker_line} must be after target line {target_line}"
            )
        if placement == "before" and marker_line > target_line:
            raise ContractError(
                f"{point['id']}: marker line {marker_line} must be before target line {target_line}"
            )
        if placement == "at" and marker_line != target_line:
            raise ContractError(
                f"{point['id']}: placement=at requires marker and target on line {target_line}"
            )
        marker_context = _source_context(lines, marker_line)

    return {
        "id": point["id"],
        "stage": point["stage"],
        "edge": point["edge"],
        "boundary_quality": point["boundary_quality"],
        "roles": point["roles"],
        "scope": point["scope"],
        "file": relative_file.as_posix(),
        "symbol": source["symbol"],
        "placement": source["placement"],
        "target_anchor": source["target_anchor"],
        "target_match_count": len(target_matches),
        "target_line": target_line,
        "target_context": _source_context(lines, target_line),
        "marker": source["marker"],
        "marker_match_count": len(marker_matches),
        "marker_line": marker_line,
        "marker_context": marker_context,
        "max_line_distance": source["max_line_distance"],
        "source_file_sha256": _sha256_bytes(raw),
        "why": point["why"],
        "semantics": point["semantics"],
        "dependency": point["dependency"],
        "perturbation": point["perturbation"],
    }


def _markdown_cell(value: Any) -> str:
    if isinstance(value, list):
        value = ", ".join(str(item) for item in value)
    return str(value).replace("|", "\\|").replace("\n", " ")


def _context_markdown(context: dict[str, Any] | None) -> str:
    if not context:
        return ""
    return "\n".join(
        f"{entry['line']:>6}  {entry['text']}".rstrip() for entry in context["lines"]
    )


def _audit_markdown(audit: dict[str, Any]) -> str:
    git = audit["git"]
    lines = [
        "# Trace point audit",
        "",
        f"- Status: `{audit['status']}`",
        f"- Phase: `{audit['phase']}`",
        f"- Trace: `{audit['trace_name']}`",
        f"- Repository: `{audit['repo_root']}`",
        f"- Revision: `{git.get('revision') or 'not-a-git-repository'}`",
        f"- Branch: `{git.get('branch') or 'detached-or-unknown'}`",
        f"- Dirty: `{git.get('dirty')}`",
        f"- Spec SHA-256: `{audit['spec']['sha256']}`",
        "",
    ]
    if audit.get("errors"):
        lines.extend(["## Errors", ""])
        lines.extend(f"- {error}" for error in audit["errors"])
        lines.append("")
    lines.extend(
        [
            "## Placement summary",
            "",
            "| Point | Roles | Resolved location | Target line | Placement | Boundary | Bridge | Why |",
            "|---|---|---|---:|---|---|---|---|",
        ]
    )
    for point in audit["points"]:
        resolved_line = point["marker_line"] or point["target_line"]
        lines.append(
            "| "
            + " | ".join(
                _markdown_cell(value)
                for value in (
                    point["id"],
                    point["roles"],
                    f"{point['file']}:{resolved_line}",
                    point["target_line"],
                    point["placement"],
                    point["boundary_quality"],
                    point["dependency"]["bridge"],
                    point["why"],
                )
            )
            + " |"
        )
    lines.append("")
    lines.extend(["## Point evidence", ""])
    for point in audit["points"]:
        resolved_line = point["marker_line"] or point["target_line"]
        lines.extend(
            [
                f"### `{point['id']}`",
                "",
                f"- Where: `{point['file']}:{resolved_line}` in `{point['symbol']}`; "
                f"`{point['placement']}` target line `{point['target_line']}`.",
                f"- Cardinality: {point['scope']}",
                f"- Boundary claim: `{point['boundary_quality']}` — {point['semantics']}",
                f"- Why: {point['why']}",
                f"- Dependency: {point['dependency']['precondition']}",
                f"- Pipeline proof: `{point['dependency']['producer_pipeline']}` → "
                f"`{point['dependency']['timestamp_pipeline']}` using "
                f"`{point['dependency']['bridge']}` — {point['dependency']['proof']}",
                f"- Perturbation: {point['perturbation']}",
                f"- Source SHA-256: `{point['source_file_sha256']}`",
                "",
                "```text",
                _context_markdown(point["marker_context"] or point["target_context"]),
                "```",
                "",
            ]
        )
    if git.get("status_short"):
        lines.extend(["## Working tree status", "", "```text"])
        lines.extend(git["status_short"])
        lines.extend(["```", ""])
    return "\n".join(lines)


def _cmd_audit(args: argparse.Namespace) -> int:
    spec_path = args.spec.expanduser().resolve()
    spec = _load_valid_spec(spec_path)
    repo_root = args.repo_root.expanduser().resolve()
    if not repo_root.is_dir():
        raise ContractError(f"repo root is not a directory: {repo_root}")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    audited_points: list[dict[str, Any]] = []
    errors: list[str] = []
    for point in spec["points"]:
        try:
            audited_points.append(_audit_point(point, repo_root, args.phase))
        except ContractError as exc:
            errors.append(str(exc))

    audit = {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "status": "fail" if errors else "pass",
        "phase": args.phase,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "trace_name": spec["trace"]["name"],
        "repo_root": str(repo_root),
        "git": _git_metadata(repo_root),
        "spec": {
            "path": str(spec_path),
            "sha256": _spec_sha256(spec),
            "protocol_version": spec["trace"]["protocol"]["version"],
            "schema_id": spec["trace"]["protocol"]["schema_id"],
        },
        "points": audited_points,
        "errors": errors,
    }
    json_path = output_dir / "trace_point_audit.json"
    markdown_path = output_dir / "trace_point_audit.md"
    json_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(_audit_markdown(audit), encoding="utf-8")
    if errors:
        raise ContractError(
            f"{args.phase} audit failed; reports written to {output_dir}:\n  - "
            + "\n  - ".join(errors)
        )
    print(
        f"[OK] {args.phase} audit passed: points={len(audited_points)} "
        f"json={json_path} markdown={markdown_path}"
    )
    return 0


def _timestamp_sequence(value: Any, field: str, bits: int) -> list[int | None]:
    values = value if isinstance(value, list) else [value]
    result: list[int | None] = []
    for index, item in enumerate(values):
        if item is None:
            result.append(None)
            continue
        result.append(_parse_uint(item, f"{field}[{index}]", bits))
    return result


def _validate_normalized_trace(
    trace_data: Any, spec: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not isinstance(trace_data, dict):
        raise ContractError("normalized trace root must be an object")
    if trace_data.get("schema_version") != TRACE_SCHEMA_VERSION:
        raise ContractError(
            f"normalized trace schema_version must be {TRACE_SCHEMA_VERSION}"
        )
    trace_spec = spec["trace"]
    protocol = trace_spec["protocol"]
    if trace_data.get("trace_name") != trace_spec["name"]:
        raise ContractError(
            f"trace_name mismatch: data={trace_data.get('trace_name')!r}, spec={trace_spec['name']!r}"
        )
    if trace_data.get("protocol_version") != protocol["version"]:
        raise ContractError(
            f"protocol_version mismatch: data={trace_data.get('protocol_version')!r}, "
            f"spec={protocol['version']!r}"
        )
    if _parse_uint(trace_data.get("schema_id"), "$.schema_id") != _parse_uint(
        protocol["schema_id"], "$.trace.protocol.schema_id"
    ):
        raise ContractError("schema_id mismatch between normalized trace and spec")

    metadata = trace_data.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ContractError("$.metadata must be an object when set")
    runtime_header = metadata.get("header", {})
    if not isinstance(runtime_header, dict):
        raise ContractError("$.metadata.header must be an object")

    bits = trace_spec["clock"]["bits"]
    roles = trace_spec["roles"]
    point_by_id = {point["id"]: point for point in spec["points"]}
    point_ids = set(point_by_id)
    timestamp_field_counts = {
        (role, field["name"]): _resolve_field_count(
            field, runtime_header, f"record field {field['name']} count"
        )
        for field in trace_spec["protocol"]["record_fields"]
        if field["kind"] == "timestamp"
        for role in field["roles"]
    }
    records = trace_data.get("records")
    if not isinstance(records, list) or not records:
        raise ContractError("$.records must be a non-empty list")
    seen: set[tuple[str, int]] = set()
    normalized_records: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        loc = f"$.records[{index}]"
        if not isinstance(record, dict):
            raise ContractError(f"{loc} must be an object")
        role = record.get("role")
        core = record.get("core")
        if role not in roles:
            raise ContractError(f"{loc}.role={role!r} is not declared in the spec")
        if not isinstance(core, int) or isinstance(core, bool) or core < 0:
            raise ContractError(f"{loc}.core must be a non-negative integer")
        key = (role, core)
        if key in seen:
            raise ContractError(f"duplicate role/core record: {role}/{core}")
        seen.add(key)
        label = record.get("label", f"{role}-{core:02d}")
        if not isinstance(label, str) or not label:
            raise ContractError(f"{loc}.label must be a non-empty string when set")
        timestamps = record.get("timestamps")
        if not isinstance(timestamps, dict):
            raise ContractError(f"{loc}.timestamps must be an object")
        unknown = sorted(set(timestamps) - point_ids)
        if unknown:
            raise ContractError(
                f"{loc}.timestamps contains undeclared points: {unknown}"
            )
        wrong_role = sorted(
            point_id
            for point_id in timestamps
            if role not in point_by_id[point_id]["roles"]
        )
        if wrong_role:
            raise ContractError(
                f"{loc}.timestamps contains points not declared for role {role}: {wrong_role}"
            )
        normalized_timestamps = {
            point_id: _timestamp_sequence(value, f"{loc}.timestamps.{point_id}", bits)
            for point_id, value in timestamps.items()
        }
        wrong_counts = [
            f"{point_id}={len(values)} (expected {timestamp_field_counts[(role, point_id)]})"
            for point_id, values in normalized_timestamps.items()
            if len(values) != timestamp_field_counts[(role, point_id)]
        ]
        if wrong_counts:
            raise ContractError(
                f"{loc}.timestamps repetition count disagrees with record_fields: "
                + ", ".join(wrong_counts)
            )
        counters = record.get("counters", {})
        if not isinstance(counters, dict) or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in counters.values()
        ):
            raise ContractError(f"{loc}.counters must map strings to numbers")
        normalized_records.append(
            {
                "role": role,
                "core": core,
                "label": label,
                "timestamps": normalized_timestamps,
                "counters": counters,
            }
        )

    reference = trace_data.get("reference")
    if not isinstance(reference, dict):
        raise ContractError("$.reference must be an object")
    reference_point = reference.get("point")
    occurrence = reference.get("occurrence", 0)
    if reference_point not in point_ids:
        raise ContractError("$.reference.point must name a declared point")
    if (
        not isinstance(occurrence, int)
        or isinstance(occurrence, bool)
        or occurrence < 0
    ):
        raise ContractError("$.reference.occurrence must be a non-negative integer")
    reference_normalized = {
        "role": reference.get("role"),
        "core": reference.get("core"),
        "point": reference_point,
        "occurrence": occurrence,
    }
    if trace_spec["clock"]["comparable_across_records"]:
        if reference_normalized["role"] not in roles:
            raise ContractError("$.reference.role must name a declared role")
        if (
            not isinstance(reference_normalized["core"], int)
            or isinstance(reference_normalized["core"], bool)
            or reference_normalized["core"] < 0
        ):
            raise ContractError("$.reference.core must be a non-negative integer")
    return normalized_records, reference_normalized


def _get_timestamp(
    record: dict[str, Any], point_id: str, occurrence: int, missing_value: int | None
) -> int | None:
    values = record["timestamps"].get(point_id, [])
    if occurrence >= len(values):
        return None
    value = values[occurrence]
    if value is None or (missing_value is not None and value == missing_value):
        return None
    return value


def _relative(raw: int, reference: int, bits: int) -> int:
    modulus = 1 << bits
    delta = (raw - reference) & (modulus - 1)
    if delta >= modulus // 2:
        delta -= modulus
    return delta


def _modular_span(begin: int, end: int, bits: int) -> int:
    return (end - begin) & ((1 << bits) - 1)


def _wave_expert_count(
    wave_begin: int, expert_count: int, experts_per_wave: int
) -> int:
    if wave_begin >= expert_count:
        return 0
    remaining = expert_count - wave_begin
    if experts_per_wave <= 0 or remaining <= experts_per_wave:
        return remaining
    tail = remaining - experts_per_wave
    return (
        remaining
        if tail < experts_per_wave and 2 * tail < experts_per_wave
        else experts_per_wave
    )


def _expert_wave_ids(header: dict[str, Any]) -> list[int]:
    expert_count = int(header.get("moe_experts_per_rank", 0))
    experts_per_wave = int(header.get("experts_per_wave", 0))
    result = [-1] * expert_count
    wave_begin = 0
    wave_id = 0
    while wave_begin < expert_count:
        count = _wave_expert_count(wave_begin, expert_count, experts_per_wave)
        if count <= 0:
            raise ContractError(
                "cannot derive routed-wave partition from runtime header"
            )
        for expert in range(wave_begin, wave_begin + count):
            result[expert] = wave_id
        wave_begin += count
        wave_id += 1
    return result


def _load_audit(path: Path | None, spec: dict[str, Any]) -> dict[str, Any] | None:
    if path is None:
        return None
    audit = _load_json(path.expanduser().resolve())
    if not isinstance(audit, dict):
        raise ContractError("audit root must be an object")
    if audit.get("audit_schema_version") != AUDIT_SCHEMA_VERSION:
        raise ContractError("unsupported audit schema version")
    if audit.get("status") != "pass" or audit.get("phase") != "implemented":
        raise ContractError("renderer requires a passing implemented audit")
    audit_spec = audit.get("spec")
    if not isinstance(audit_spec, dict) or audit_spec.get("sha256") != _spec_sha256(
        spec
    ):
        raise ContractError("audit was generated from a different trace spec")
    audited_ids = {point.get("id") for point in audit.get("points", [])}
    expected_ids = {point["id"] for point in spec["points"]}
    if audited_ids != expected_ids:
        raise ContractError("audit point set does not match the trace spec")
    return audit


def _json_pretty(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _render_html(
    spec: dict[str, Any],
    trace_data: dict[str, Any],
    records: list[dict[str, Any]],
    reference: dict[str, Any],
    audit: dict[str, Any] | None,
) -> str:
    trace_spec = spec["trace"]
    clock = trace_spec["clock"]
    bits = clock["bits"]
    missing_value = clock.get("missing_value")
    comparable = clock["comparable_across_records"]
    point_ids = {point["id"] for point in spec["points"]}
    missing_chinese = sorted(point_ids - set(POINT_TEXT_ZH))
    if missing_chinese:
        raise ContractError(
            "renderer lacks Chinese descriptions for points: "
            + ", ".join(missing_chinese)
        )
    role_order = {role: index for index, role in enumerate(trace_spec["roles"])}
    comparison_group = trace_spec.get("visualization", {}).get("comparison_group")
    fine_visualization = trace_spec.get("visualization", {}).get("fine_grained")
    runtime_header = trace_data.get("metadata", {}).get("header", {})
    if not isinstance(runtime_header, dict):
        raise ContractError("renderer requires metadata.header")
    expert_wave_ids = (
        _expert_wave_ids(runtime_header) if isinstance(fine_visualization, dict) else []
    )
    derived_gap_by_span = {
        gap["span"]: gap
        for gap in (
            fine_visualization.get("derived_idle_gaps", [])
            if isinstance(fine_visualization, dict)
            else []
        )
    }
    grouped_by_ai_block = (
        isinstance(comparison_group, dict)
        and comparison_group.get("kind") == "shared-core-index"
    )
    records = sorted(
        records,
        key=(
            lambda record: (record["core"], role_order[record["role"]])
            if grouped_by_ai_block
            else (role_order[record["role"]], record["core"])
        ),
    )

    if grouped_by_ai_block:
        group_roles = comparison_group["roles"]
        reference_role = comparison_group["reference_role"]
        records_by_key = {
            (record["role"], record["core"]): record for record in records
        }
        cores = sorted({record["core"] for record in records})
        row_references = {}
        for core in cores:
            missing_roles = [
                role for role in group_roles if (role, core) not in records_by_key
            ]
            if missing_roles:
                raise ContractError(
                    f"AI Block {core} lacks role records: {', '.join(missing_roles)}"
                )
            reference_record = records_by_key[(reference_role, core)]
            group_reference = _get_timestamp(
                reference_record,
                reference["point"],
                reference["occurrence"],
                missing_value,
            )
            if group_reference is None:
                raise ContractError(
                    f"AI Block {core} reference {reference_role}/{reference['point']} "
                    "is missing"
                )
            for role in group_roles:
                row_references[(role, core)] = group_reference
        axis_note = (
            f"每个 AI Block 以本组 {reference_role}/{reference['point']}"
            f"[{reference['occurrence']}] 独立归零；组内 AIC/AIV0/AIV1 可直接对齐，"
            "不同 AI Block 的绝对时间戳不比较"
        )
    elif comparable:
        reference_record = next(
            (
                record
                for record in records
                if record["role"] == reference["role"]
                and record["core"] == reference["core"]
            ),
            None,
        )
        if reference_record is None:
            raise ContractError("reference role/core record is absent")
        shared_reference = _get_timestamp(
            reference_record, reference["point"], reference["occurrence"], missing_value
        )
        if shared_reference is None:
            raise ContractError("reference timestamp is missing")
        row_references = {
            (record["role"], record["core"]): shared_reference for record in records
        }
        axis_note = (
            f"共享 {clock['unit']} 时钟；参考点为 "
            f"{reference_record['label']}/{reference['point']}[{reference['occurrence']}]"
        )
    else:
        row_references: dict[tuple[str, int], int] = {}
        for record in records:
            raw_reference = _get_timestamp(
                record, reference["point"], reference["occurrence"], missing_value
            )
            if raw_reference is None:
                raise ContractError(
                    f"local-clock record {record['label']} lacks reference point {reference['point']}"
                )
            row_references[(record["role"], record["core"])] = raw_reference
        axis_note = (
            f"每条记录以 {reference['point']}[{reference['occurrence']}] 独立归零；"
            "不同记录之间的空隙不可比较"
        )

    plotted: list[int] = []
    rendered_spans: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for record in records:
        record_key = (record["role"], record["core"])
        raw_reference = row_references[record_key]
        row_spans: list[dict[str, Any]] = []
        for span in spec["spans"]:
            if record["role"] not in span["roles"]:
                continue
            begins = record["timestamps"].get(span["begin"], [])
            ends = record["timestamps"].get(span["end"], [])
            if len(begins) != len(ends):
                raise ContractError(
                    f"{record['label']} span {span['id']} has {len(begins)} begins and {len(ends)} ends"
                )
            for occurrence, (begin, end) in enumerate(zip(begins, ends)):
                if (
                    begin is None
                    or end is None
                    or (
                        missing_value is not None
                        and (begin == missing_value or end == missing_value)
                    )
                ):
                    continue
                duration = _modular_span(begin, end, bits)
                if duration >= (1 << (bits - 1)):
                    raise ContractError(
                        f"{record['label']} {span['id']}[{occurrence}] has ambiguous/backward span {duration}"
                    )
                begin_delta = _relative(begin, raw_reference, bits)
                end_delta = begin_delta + duration
                plotted.extend((begin_delta, end_delta))
                gap_contract = derived_gap_by_span.get(span["id"])
                expert_indexed = span["id"] != "process" and occurrence < len(
                    expert_wave_ids
                )
                wave_id = expert_wave_ids[occurrence] if expert_indexed else None
                tile_expert = None
                tile_ordinal = None
                if span["id"] in {
                    "a8w4_combine_tile_wait",
                    "a8w4_combine_tile_work",
                }:
                    tiles_per_expert = int(
                        runtime_header.get("combine_tiles_per_expert_per_core", 0)
                    )
                    if tiles_per_expert <= 0:
                        raise ContractError(
                            "A8W4 tile span lacks positive combine_tiles_per_expert_per_core"
                        )
                    tile_expert = occurrence // tiles_per_expert
                    tile_ordinal = occurrence % tiles_per_expert
                    wave_id = None
                wave_expert = None
                m_wave = None
                if span["id"] in {
                    "a8w4_gmm1_wave_wait",
                    "a8w4_gmm1_wave_work",
                }:
                    waves_per_expert = int(
                        runtime_header.get("gmm1_waves_per_expert", 0)
                    )
                    if waves_per_expert <= 0:
                        raise ContractError(
                            "A8W4 wave span lacks positive gmm1_waves_per_expert"
                        )
                    wave_expert = occurrence // waves_per_expert
                    m_wave = occurrence % waves_per_expert
                    wave_id = None
                row_spans.append(
                    {
                        "span": span,
                        "occurrence": occurrence,
                        "wave_id": wave_id,
                        "tile_expert": tile_expert,
                        "tile_ordinal": tile_ordinal,
                        "wave_expert": wave_expert,
                        "m_wave": m_wave,
                        "derived_gap": None,
                        "begin_raw": begin,
                        "end_raw": end,
                        "begin": begin_delta,
                        "end": end_delta,
                        "duration": duration,
                    }
                )
        for span_id, gap_contract in derived_gap_by_span.items():
            stage_spans = sorted(
                (
                    rendered
                    for rendered in row_spans
                    if rendered["span"]["id"] == span_id
                ),
                key=lambda rendered: rendered["occurrence"],
            )
            for previous, current in zip(stage_spans, stage_spans[1:]):
                if previous["wave_id"] != current["wave_id"]:
                    continue
                gap_duration = current["begin"] - previous["end"]
                if gap_duration <= 0:
                    continue
                plotted.extend((previous["end"], current["begin"]))
                row_spans.append(
                    {
                        "span": {
                            "id": f"{span_id}_idle_gap",
                            "label": gap_contract["label"],
                            "quality": gap_contract["quality"],
                            "color": gap_contract["color"],
                        },
                        "occurrence": current["occurrence"],
                        "wave_id": current["wave_id"],
                        "derived_gap": gap_contract,
                        "begin_raw": previous["end_raw"],
                        "end_raw": current["begin_raw"],
                        "begin": previous["end"],
                        "end": current["begin"],
                        "duration": gap_duration,
                    }
                )
        rendered_spans[record_key] = row_spans
        for point_id, values in record["timestamps"].items():
            for value in values:
                if value is None or (
                    missing_value is not None and value == missing_value
                ):
                    continue
                plotted.append(_relative(value, raw_reference, bits))

    if not plotted:
        raise ContractError("normalized trace contains no plottable timestamps")
    min_cycle = min(plotted)
    max_cycle = max(plotted)
    if min_cycle == max_cycle:
        max_cycle += 1
    cycle_range = max_cycle - min_cycle

    label_width = 210 if grouped_by_ai_block else 138
    plot_width = 1500
    counter_width = 420
    row_height = 30
    top = 72
    svg_width = label_width + plot_width + counter_width
    svg_height = top + len(records) * row_height + 48

    def x_pos(cycle: int) -> float:
        return label_width + (cycle - min_cycle) * plot_width / cycle_range

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_width}" height="{svg_height}" '
        f'viewBox="0 0 {svg_width} {svg_height}">',
        "<style>",
        "text{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;fill:#0f172a}",
        ".label{font-size:12px;font-weight:650}.axis{font-size:10px}.counter{font-size:10px}",
        ".grid{stroke:#cbd5e1;stroke-width:.7}.group{stroke:#334155;stroke-width:1.4}",
        ".marker{stroke:#111827;stroke-width:.65;opacity:.38}.nonexact{stroke:#475569;stroke-dasharray:3 2}",
        "</style>",
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="8" y="20" font-size="16" font-weight="700">{html.escape(trace_spec["name"])} 关键路径</text>',
        f'<text x="8" y="40" font-size="10">{html.escape(axis_note)}</text>',
    ]

    legend_x = 8
    for span in spec["spans"]:
        label = SPAN_LABEL_ZH.get(span["id"], span["label"])
        opacity = "0.92" if span["quality"] == "exact" else "0.42"
        quality_label = "精确边界" if span["quality"] == "exact" else "近似边界"
        svg.append(
            f'<rect x="{legend_x}" y="49" width="12" height="9" fill="{span["color"]}" opacity="{opacity}"/>'
            f'<text x="{legend_x + 16}" y="57" font-size="9">{html.escape(label)}（{quality_label}）</text>'
        )
        legend_x += 76 + len(label) * 10
    rendered_gap_labels: set[str] = set()
    for gap in derived_gap_by_span.values():
        if gap["label"] in rendered_gap_labels:
            continue
        rendered_gap_labels.add(gap["label"])
        svg.append(
            f'<rect class="nonexact" x="{legend_x}" y="49" width="12" height="9" '
            f'fill="{gap["color"]}" opacity="0.46"/>'
            f'<text x="{legend_x + 16}" y="57" font-size="9">'
            f"{html.escape(gap['label'])}（推断间隙）</text>"
        )
        legend_x += 90 + len(gap["label"]) * 10

    for tick_index in range(11):
        cycle = min_cycle + cycle_range * tick_index // 10
        x = x_pos(cycle)
        svg.append(
            f'<line class="grid" x1="{x:.2f}" y1="{top - 4}" x2="{x:.2f}" y2="{svg_height - 24}"/>'
            f'<text class="axis" x="{x:.2f}" y="{svg_height - 8}" text-anchor="middle">{cycle}</text>'
        )

    previous_group: int | str | None = None
    points_in_spans = {
        point_id for span in spec["spans"] for point_id in (span["begin"], span["end"])
    }
    for row_index, record in enumerate(records):
        y = top + row_index * row_height
        current_group: int | str = (
            record["core"] if grouped_by_ai_block else record["role"]
        )
        if current_group != previous_group:
            svg.append(
                f'<line class="group" x1="0" y1="{y}" x2="{svg_width}" y2="{y}"/>'
            )
        previous_group = current_group
        shade_index = record["core"] if grouped_by_ai_block else row_index
        shade = "#f1f5f9" if shade_index % 2 else "#ffffff"
        svg.append(
            f'<rect x="0" y="{y}" width="{svg_width}" height="{row_height}" fill="{shade}"/>'
        )
        display_label = (
            f"AI Block {record['core']:02d} / {record['role']}"
            if grouped_by_ai_block
            else record["label"]
        )
        svg.append(
            f'<text class="label" x="6" y="{y + 20}">{html.escape(display_label)}</text>'
        )

        for rendered in rendered_spans[(record["role"], record["core"])]:
            span = rendered["span"]
            span_label = SPAN_LABEL_ZH.get(span["id"], span["label"])
            x_begin = x_pos(rendered["begin"])
            width = max(1.0, x_pos(rendered["end"]) - x_begin)
            exact = span["quality"] == "exact"
            css_class = "" if exact else "nonexact"
            opacity = "0.92" if exact else "0.42"
            wave_label = (
                f"专家 {rendered.get('wave_expert')} / M-wave {rendered.get('m_wave')}"
                if rendered.get("wave_expert") is not None
                else (
                    f"专家 {rendered.get('tile_expert')} / 本核 tile {rendered.get('tile_ordinal')}"
                    if rendered.get("tile_expert") is not None
                    else (
                        f"wave {rendered['wave_id']} / 专家 {rendered['occurrence']}"
                        if rendered["wave_id"] is not None
                        else f"记录项 {rendered['occurrence']}"
                    )
                )
            )
            if rendered["derived_gap"] is not None:
                gap = rendered["derived_gap"]
                title = html.escape(
                    f"{span_label}（{wave_label}）：{rendered['begin']} → "
                    f"{rendered['end']} cycle（持续 {rendered['duration']} cycle）；"
                    f"边界性质：推断间隙；注意：{gap['caveat']}"
                )
                opacity = "0.46"
            else:
                begin_text = POINT_TEXT_ZH[span["begin"]]
                end_text = POINT_TEXT_ZH[span["end"]]
                title = html.escape(
                    f"{span_label}（{wave_label}）：{rendered['begin']} → "
                    f"{rendered['end']} cycle（持续 {rendered['duration']} cycle）；"
                    f"开始时机：{begin_text['when']}；结束时机：{end_text['when']}；"
                    f"为什么打点：{begin_text['why']} / {end_text['why']}"
                )
            svg.append(
                f'<rect class="{css_class}" x="{x_begin:.2f}" y="{y + 7}" width="{width:.2f}" '
                f'height="16" rx="2" fill="{span["color"]}" opacity="{opacity}"><title>{title}</title></rect>'
            )

        raw_reference = row_references[(record["role"], record["core"])]
        for point_id, values in record["timestamps"].items():
            if point_id in points_in_spans:
                continue
            point_text = POINT_TEXT_ZH[point_id]
            for occurrence, value in enumerate(values):
                if value is None or (
                    missing_value is not None and value == missing_value
                ):
                    continue
                delta = _relative(value, raw_reference, bits)
                x = x_pos(delta)
                title = html.escape(
                    f"{point_id}[{occurrence}]={delta} cycle；"
                    f"打点时机：{point_text['when']}；为什么打点：{point_text['why']}"
                )
                svg.append(
                    f'<line class="marker" x1="{x:.2f}" y1="{y + 4}" x2="{x:.2f}" y2="{y + 26}"><title>{title}</title></line>'
                )

        fine_counter_parts: list[str] = []
        for gap in derived_gap_by_span.values():
            span = next(
                (item for item in spec["spans"] if item["id"] == gap["span"]), None
            )
            if span is None or record["role"] not in span["roles"]:
                continue
            begins = record["timestamps"].get(span["begin"], [])
            ends = record["timestamps"].get(span["end"], [])
            active_experts = [
                index
                for index, (begin, end) in enumerate(zip(begins, ends))
                if begin not in (None, missing_value)
                and end not in (None, missing_value)
            ]
            waves = [expert_wave_ids[index] for index in active_experts]
            stage_label = SPAN_LABEL_ZH.get(gap["span"], gap["span"])
            fine_counter_parts.append(
                f"{stage_label}: 活跃专家={active_experts}, waves={waves}"
            )
        remaining_counters = [
            f"{key}={value}" for key, value in sorted(record["counters"].items())
        ]
        counter_text = "；".join(fine_counter_parts + remaining_counters) or "无计数器"
        svg.append(
            f'<text class="counter" x="{label_width + plot_width + 8}" y="{y + 19}">{html.escape(counter_text)}</text>'
        )
    svg.append("</svg>")

    metadata_rows = "".join(
        f"<tr><th>{html.escape(str(key))}</th><td><code>{html.escape(_json_pretty(value))}</code></td></tr>"
        for key, value in sorted(trace_data.get("metadata", {}).items())
    )
    protocol = trace_spec["protocol"]
    fine_tail = protocol.get("fine_metadata_tail")
    protocol_layout = (
        f"{protocol['endianness']} uint{protocol['word_bits']}; "
        f"header={protocol['header_words']} words, "
        f"record={_record_words_text(protocol, runtime_header)} words"
    )
    if fine_tail is not None:
        protocol_layout += f", fine-tail={fine_tail['words_per_core']} words / AI Block"
    write_payload = "records"
    if fine_tail is not None:
        write_payload += " 与细粒度元数据尾部"
    overview_rows = "".join(
        f"<tr><th>{html.escape(key)}</th><td><code>{html.escape(str(value))}</code></td></tr>"
        for key, value in (
            ("Trace 名称", trace_spec["name"]),
            ("协议版本", protocol["version"]),
            ("Schema ID", protocol["schema_id"]),
            ("角色", ", ".join(trace_spec["roles"])),
            (
                "线协议布局",
                protocol_layout,
            ),
            ("线协议记录顺序", protocol["record_order"]),
            (
                "时间戳作用域",
                f"{clock['api']} / {clock['bits']}-bit {clock['unit']} / {clock['scope']}",
            ),
            (
                "输出载体",
                "正常 BF16 y 完成后，将 y 的原始字节临时复用为诊断记录载体",
            ),
            ("是否覆盖正常输出", trace_spec["output"]["overwrites_user_output"]),
            (
                "容量检查",
                "设备端和解析端同时核对 y 容量；不足时不得发布有效 magic",
            ),
            (
                "写入阶段",
                f"有效工作和最终 Unpermute 写回完成后：先写 {write_payload}，"
                "再写 header，最后发布 magic",
            ),
            (
                "终结同步",
                f"两次混合 AIC/AIV rendezvous：先保护 y 生命周期，再等待全部 {write_payload}",
            ),
        )
    )

    audit_by_id = (
        {point["id"]: point for point in audit.get("points", [])} if audit else {}
    )
    audit_rows: list[str] = []
    for point in spec["points"]:
        point_text = POINT_TEXT_ZH[point["id"]]
        evidence = audit_by_id.get(point["id"])
        location = (
            f"{evidence['file']}:{evidence['marker_line'] or evidence['target_line']}"
            if evidence
            else f"{point['source']['file']}（未提供 implemented audit，行号未解析）"
        )
        dependency = point["dependency"]
        placement_zh = {"before": "锚点之前", "after": "锚点之后", "at": "锚点同行"}[
            point["source"]["placement"]
        ]
        quality_zh = {
            "causal": "因果起点",
            "actual": "实际完成",
            "issue": "发出边界",
            "inferred": "推断边界",
        }[point["boundary_quality"]]
        audit_rows.append(
            "<tr>"
            f"<td><code>{html.escape(point['id'])}</code></td>"
            f"<td>{html.escape(', '.join(point['roles']))}</td>"
            f"<td><code>{html.escape(location)}</code><br>{placement_zh}，函数/符号 "
            f"<code>{html.escape(point['source']['symbol'])}</code></td>"
            f"<td><code>{quality_zh}</code><br>{html.escape(point_text['when'])}</td>"
            f"<td><strong>桥接：</strong><code>{html.escape(dependency['bridge'])}</code><br>"
            f"<strong>证明：</strong>{html.escape(point_text['proof'])}</td>"
            f"<td>{html.escape(point_text['why'])}</td>"
            f"<td>{html.escape(point_text['perturbation'])}</td>"
            "</tr>"
        )

    if grouped_by_ai_block:
        warning = (
            '<div class="warning"><strong>时间戳边界：</strong>同一 AI Block 内的 '
            "AIC、AIV0、AIV1 使用同一时钟并可直接对齐；不同 AI Block 可能存在固定偏移，"
            "因此每组单独归零，禁止从不同组的横向位置推断绝对先后或 handoff 间隔。</div>"
        )
    elif comparable:
        warning = ""
    else:
        warning = (
            '<div class="warning">不同记录的时钟不可直接比较；每行已经独立归零，'
            "禁止推断跨行 handoff 间隔。</div>"
        )
    if derived_gap_by_span:
        warning += (
            '<div class="warning"><strong>细粒度边界：</strong>GMM1/GMM2/SwiGLU '
            "按本卡 routed expert index 展开，设备记录长度由 header 中的专家数动态决定；"
            "未参与该专家的物理核保留空槽，wave 归属由已校验的运行时分组规则推导。"
            "橙色只连接同一 wave 的相邻专家：它由前一专家 end 与后一专家 causal begin 推断，"
            "包含流水尾部、真实依赖等待和少量 Scalar 调度，不能解释为纯等待指令耗时。"
            "GMM1 专家 end 是低扰动 issue 边界，采用半透明虚线。</div>"
        )
    return (
        '<!doctype html><html><head><meta charset="utf-8">'
        f"<title>{html.escape(trace_spec['name'])} 关键路径</title>"
        "<style>body{margin:0;background:#e2e8f0;color:#0f172a;font-family:system-ui,sans-serif}"
        "main{padding:14px}.panel{background:#fff;border-radius:9px;box-shadow:0 1px 6px #64748b;"
        "padding:12px;margin-bottom:14px;overflow:auto}h1,h2{margin:.2em 0 .6em}"
        "table{border-collapse:collapse;width:100%;font-size:12px}th,td{border:1px solid #cbd5e1;"
        "padding:6px;text-align:left;vertical-align:top}th{background:#f1f5f9}.warning{background:#fef3c7;"
        "border:1px solid #f59e0b;padding:9px;border-radius:6px;margin-bottom:10px}code{white-space:pre-wrap}"
        "</style></head><body><main>"
        f"<h1>{html.escape(trace_spec['name'])}</h1>{warning}"
        f'<section class="panel"><h2>协议与运行信息</h2><table>{overview_rows}{metadata_rows}</table></section>'
        f'<section class="panel">{"".join(svg)}</section>'
        '<section class="panel"><h2>打点位置与语义说明</h2>'
        "<table><thead><tr><th>时间点</th><th>角色</th><th>打点位置</th><th>什么时候打</th>"
        "<th>流水同步与证明</th><th>为什么打点</th><th>潜在扰动</th></tr></thead><tbody>"
        f"{''.join(audit_rows)}</tbody></table></section>"
        "</main></body></html>"
    )


def _combine_protocol_words(words: list[int], word_bits: int, endianness: str) -> int:
    ordered = words if endianness == "little" else list(reversed(words))
    value = 0
    for index, word in enumerate(ordered):
        value |= word << (index * word_bits)
    return value


def _decode_header_field(
    words: list[int], field: dict[str, Any], protocol: dict[str, Any]
) -> int | list[int]:
    start = field["word"]
    width = field["width_words"]
    selected = words[start : start + width]
    if field["name"] == "role_counts":
        return selected
    if width == 1:
        return selected[0]
    return _combine_protocol_words(
        selected, protocol["word_bits"], protocol["endianness"]
    )


def _fine_tail_word_count(protocol: dict[str, Any], role_counts: list[int]) -> int:
    fine_tail = protocol.get("fine_metadata_tail")
    if fine_tail is None:
        return 0
    if not role_counts or any(count != role_counts[0] for count in role_counts):
        raise ContractError(
            "fine metadata tail requires identical paired-core counts for all roles"
        )
    return int(fine_tail["words_per_core"]) * role_counts[0]


def _decode_fine_metadata_tail(
    *,
    words: list[int],
    protocol: dict[str, Any],
    role_counts: list[int],
    record_count: int,
    record_words: int,
    records: list[dict[str, Any]],
) -> int:
    fine_tail = protocol.get("fine_metadata_tail")
    if fine_tail is None:
        return 0
    tail_words = _fine_tail_word_count(protocol, role_counts)
    tail_base = protocol["header_words"] + record_count * record_words
    records_by_key = {(record["role"], record["core"]): record for record in records}
    core_count = role_counts[0]
    stride = int(fine_tail["core_stride_words"])
    for core in range(core_count):
        for field in fine_tail["fields"]:
            role = field["writer_role"]
            record = records_by_key.get((role, core))
            if record is None:
                raise ContractError(
                    f"fine metadata tail lacks paired record {role}/{core}"
                )
            packed_word = words[tail_base + core * stride + int(field["word_in_core"])]
            for packed in field["packing"]:
                bit = int(packed["bit"])
                width = int(packed["width"])
                packed_value = (packed_word >> bit) & ((1 << width) - 1)
                if "count" in packed:
                    count = int(packed["count"])
                    element_bits = int(packed["element_bits"])
                    element_mask = (1 << element_bits) - 1
                    prefix = packed["normalized_prefix"]
                    for index in range(count):
                        record["counters"][f"{prefix}_{index}"] = (
                            packed_value >> (index * element_bits)
                        ) & element_mask
                else:
                    record["counters"][packed["name"]] = packed_value
    return tail_words


def _parse_assignments(values: list[str], option: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in values:
        name, separator, value = item.partition("=")
        name = name.strip()
        value = value.strip()
        if not separator or not name or not value:
            raise ContractError(f"{option} requires NAME=VALUE, got {item!r}")
        if name in parsed:
            raise ContractError(f"{option} repeats {name!r}")
        parsed[name] = value
    return parsed


def _cmd_decode(args: argparse.Namespace) -> int:
    spec_path = args.spec.expanduser().resolve()
    raw_path = args.raw.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    spec = _load_valid_spec(spec_path)
    trace_spec = spec["trace"]
    protocol = trace_spec["protocol"]
    if protocol["record_order"] != "role-major":
        raise ContractError(
            "generic decode supports only record_order=role-major; use a thin target decoder for custom order"
        )
    word_bytes = protocol["word_bits"] // 8
    if args.byte_offset < 0 or args.byte_offset % word_bytes != 0:
        raise ContractError(
            f"--byte-offset must be non-negative and aligned to {word_bytes} bytes"
        )
    try:
        raw = raw_path.read_bytes()
    except FileNotFoundError as exc:
        raise ContractError(f"raw trace file does not exist: {raw_path}") from exc
    if args.byte_offset >= len(raw):
        raise ContractError(
            f"byte offset {args.byte_offset} is outside raw file of {len(raw)} bytes"
        )
    payload = raw[args.byte_offset :]
    header_bytes = protocol["header_words"] * word_bytes
    if len(payload) < header_bytes:
        raise ContractError(
            f"raw trace has {len(payload)} bytes after offset; header requires {header_bytes}"
        )
    words = [
        int.from_bytes(
            payload[offset : offset + word_bytes],
            byteorder=protocol["endianness"],
            signed=False,
        )
        for offset in range(0, len(payload) - len(payload) % word_bytes, word_bytes)
    ]
    header_words = words[: protocol["header_words"]]
    header = {
        field["name"]: _decode_header_field(header_words, field, protocol)
        for field in protocol["header_fields"]
    }
    runtime_record_words = _resolve_record_words(
        protocol, header, "$.trace.protocol.record_words"
    )

    expected_header_values = {
        "magic": _parse_uint(protocol["magic"], "$.trace.protocol.magic"),
        "version": protocol["version"],
        "schema_id": _parse_uint(protocol["schema_id"], "$.trace.protocol.schema_id"),
        "header_words": protocol["header_words"],
        "record_words": runtime_record_words,
        "clock_bits": trace_spec["clock"]["bits"],
    }
    mismatches = [
        f"{name}=0x{actual:08x} (expected 0x{expected:08x})"
        if name in ("magic", "schema_id")
        else f"{name}={actual} (expected {expected})"
        for name, expected in expected_header_values.items()
        for actual in (header[name],)
        if actual != expected
    ]
    if mismatches:
        raise ContractError("runtime trace header mismatch: " + ", ".join(mismatches))

    role_counts = header["role_counts"]
    if not isinstance(role_counts, list) or len(role_counts) != len(
        trace_spec["roles"]
    ):
        raise ContractError("runtime role_counts width does not match trace.roles")
    record_count = header["record_count"]
    if not isinstance(record_count, int) or record_count != sum(role_counts):
        raise ContractError(
            f"runtime record_count={record_count!r} does not equal sum(role_counts)={sum(role_counts)}"
        )
    repetition_count = header["repetition_count"]
    capture_capacity = header["capture_capacity"]
    truncated = header["truncated"]
    if not all(
        isinstance(value, int) and value >= 0
        for value in (repetition_count, capture_capacity, truncated)
    ):
        raise ContractError(
            "runtime repetition/capacity/truncated header fields are invalid"
        )
    if truncated not in (0, 1) or truncated != int(repetition_count > capture_capacity):
        raise ContractError(
            f"runtime truncated={truncated} disagrees with repetition_count={repetition_count} "
            f"and capture_capacity={capture_capacity}"
        )
    if truncated and not args.allow_truncated:
        raise ContractError(
            "runtime trace is truncated; rerun with sufficient capture capacity or pass "
            "--allow-truncated only when partial capture is intentional"
        )

    expected_headers = _parse_assignments(args.expect_header, "--expect-header")
    for name, expected_text in expected_headers.items():
        if name not in header:
            raise ContractError(f"--expect-header names unknown header field {name!r}")
        actual = header[name]
        if not isinstance(actual, int):
            raise ContractError(
                f"--expect-header cannot compare aggregate field {name!r}; use --expect-role-count for role_counts"
            )
        expected = _parse_uint(
            expected_text, f"--expect-header {name}", protocol["word_bits"]
        )
        if actual != expected:
            raise ContractError(
                f"runtime header {name}={actual} does not match expected {expected}"
            )

    expected_role_counts = _parse_assignments(
        args.expect_role_count, "--expect-role-count"
    )
    role_count_by_name = dict(zip(trace_spec["roles"], role_counts))
    for role, expected_text in expected_role_counts.items():
        if role not in role_count_by_name:
            raise ContractError(f"--expect-role-count names unknown role {role!r}")
        expected = _parse_uint(expected_text, f"--expect-role-count {role}")
        actual = role_count_by_name[role]
        if actual != expected:
            raise ContractError(
                f"runtime role count {role}={actual} does not match expected {expected}"
            )

    if (args.require_flags_mask is None) != (args.require_flags_value is None):
        raise ContractError(
            "--require-flags-mask and --require-flags-value must be supplied together"
        )
    if args.require_flags_mask is not None:
        mask = _parse_uint(
            args.require_flags_mask, "--require-flags-mask", protocol["word_bits"]
        )
        expected_flags = _parse_uint(
            args.require_flags_value, "--require-flags-value", protocol["word_bits"]
        )
        if expected_flags & ~mask:
            raise ContractError(
                "--require-flags-value contains bits outside --require-flags-mask"
            )
        actual_flags = header["flags"]
        if actual_flags & mask != expected_flags:
            raise ContractError(
                f"runtime flags 0x{actual_flags:08x} masked by 0x{mask:08x} "
                f"do not equal 0x{expected_flags:08x}"
            )

    fine_tail_words = _fine_tail_word_count(protocol, role_counts)
    required_words = (
        protocol["header_words"] + record_count * runtime_record_words + fine_tail_words
    )
    required_bytes = required_words * word_bytes
    if len(payload) < required_bytes:
        raise ContractError(
            f"raw trace has {len(payload)} bytes after offset; decoded layout requires {required_bytes}"
        )

    records: list[dict[str, Any]] = []
    record_index = 0
    for role, role_count in zip(trace_spec["roles"], role_counts):
        for core in range(role_count):
            base = protocol["header_words"] + record_index * runtime_record_words
            record_words = words[base : base + runtime_record_words]
            timestamps: dict[str, list[int]] = {}
            counters: dict[str, int] = {}
            for field in protocol["record_fields"]:
                if role not in field["roles"]:
                    continue
                values: list[int] = []
                field_word = _resolve_field_word(
                    field, header, f"record field {field['name']} word"
                )
                field_count = _resolve_field_count(
                    field, header, f"record field {field['name']} count"
                )
                for repetition in range(field_count):
                    start = field_word + repetition * field["stride_words"]
                    selected = record_words[start : start + field["width_words"]]
                    values.append(
                        _combine_protocol_words(
                            selected, protocol["word_bits"], protocol["endianness"]
                        )
                    )
                if field["kind"] == "timestamp":
                    timestamps[field["name"]] = values
                else:
                    counters[field["name"]] = values[0]
            records.append(
                {
                    "role": role,
                    "core": core,
                    "label": f"{role}-{core:02d}",
                    "timestamps": timestamps,
                    "counters": counters,
                }
            )
            record_index += 1

    decoded_tail_words = _decode_fine_metadata_tail(
        words=words,
        protocol=protocol,
        role_counts=role_counts,
        record_count=record_count,
        record_words=runtime_record_words,
        records=records,
    )
    if decoded_tail_words != fine_tail_words:
        raise ContractError("decoded fine metadata tail size disagrees with layout")

    normalized = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "trace_name": trace_spec["name"],
        "protocol_version": protocol["version"],
        "schema_id": protocol["schema_id"],
        "metadata": {
            "rank": header["rank"],
            "runtime_config_hash": f"0x{header['runtime_config_hash']:08x}",
            "repetition_count": repetition_count,
            "capture_capacity": capture_capacity,
            "trace_truncated": bool(truncated),
            "flags": f"0x{header['flags']:08x}",
            "raw_artifact": str(raw_path),
            "raw_sha256": _sha256_bytes(raw),
            "byte_offset": args.byte_offset,
            "required_trace_bytes": required_bytes,
            "resolved_record_words": runtime_record_words,
            "fine_metadata_tail_words": fine_tail_words,
            "available_bytes_after_offset": len(payload),
            "header": header,
            "spec_sha256": _spec_sha256(spec),
        },
        "reference": {
            "role": args.reference_role,
            "core": args.reference_core,
            "point": args.reference_point,
            "occurrence": args.reference_occurrence,
        },
        "records": records,
    }
    _validate_normalized_trace(normalized, spec)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(normalized, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"[OK] decoded raw trace: {output_path} records={record_count} "
        f"rank={header['rank']} repetitions={repetition_count}/{capture_capacity} "
        f"truncated={bool(truncated)}"
    )
    return 0


def _cmd_render(args: argparse.Namespace) -> int:
    spec_path = args.spec.expanduser().resolve()
    trace_path = args.trace.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    spec = _load_valid_spec(spec_path)
    trace_data = _load_json(trace_path)
    records, reference = _validate_normalized_trace(trace_data, spec)
    audit = _load_audit(args.audit, spec)
    rendered = _render_html(spec, trace_data, records, reference, audit)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    print(
        f"[OK] rendered timeline: {output_path} records={len(records)} "
        f"audit={'included' if audit else 'not supplied'}"
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create, validate, source-audit, decode, and render operator critical-path trace contracts."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="copy a new trace-spec template")
    init_parser.add_argument("--name", required=True, help="trace contract name")
    init_parser.add_argument(
        "--output", required=True, type=Path, help="new trace-spec JSON path"
    )
    init_parser.set_defaults(handler=_cmd_init)

    validate_parser = subparsers.add_parser(
        "validate", help="validate a completed trace spec"
    )
    validate_parser.add_argument("--spec", required=True, type=Path)
    validate_parser.set_defaults(handler=_cmd_validate)

    audit_parser = subparsers.add_parser(
        "audit",
        help="resolve semantic anchors and optionally implemented TRACE_POINT markers",
    )
    audit_parser.add_argument("--phase", choices=("plan", "implemented"), required=True)
    audit_parser.add_argument("--spec", required=True, type=Path)
    audit_parser.add_argument("--repo-root", required=True, type=Path)
    audit_parser.add_argument("--output-dir", required=True, type=Path)
    audit_parser.set_defaults(handler=_cmd_audit)

    decode_parser = subparsers.add_parser(
        "decode", help="decode a raw binary trace using the spec's checked word layout"
    )
    decode_parser.add_argument("--spec", required=True, type=Path)
    decode_parser.add_argument("--raw", required=True, type=Path)
    decode_parser.add_argument("--output", required=True, type=Path)
    decode_parser.add_argument("--byte-offset", type=int, default=0)
    decode_parser.add_argument("--reference-role", required=True)
    decode_parser.add_argument("--reference-core", required=True, type=int)
    decode_parser.add_argument("--reference-point", required=True)
    decode_parser.add_argument("--reference-occurrence", type=int, default=0)
    decode_parser.add_argument(
        "--expect-header",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="require a scalar runtime header field value; repeat as needed",
    )
    decode_parser.add_argument(
        "--expect-role-count",
        action="append",
        default=[],
        metavar="ROLE=COUNT",
        help="require a runtime physical-core count for a role; repeat as needed",
    )
    decode_parser.add_argument("--require-flags-mask")
    decode_parser.add_argument("--require-flags-value")
    decode_parser.add_argument(
        "--allow-truncated",
        action="store_true",
        help="accept an explicitly truncated capture; default is to reject it",
    )
    decode_parser.set_defaults(handler=_cmd_decode)

    render_parser = subparsers.add_parser(
        "render", help="render normalized JSON as a self-contained HTML timeline"
    )
    render_parser.add_argument("--spec", required=True, type=Path)
    render_parser.add_argument("--trace", required=True, type=Path)
    render_parser.add_argument(
        "--audit", type=Path, help="passing implemented trace_point_audit.json"
    )
    render_parser.add_argument("--output", required=True, type=Path)
    render_parser.set_defaults(handler=_cmd_render)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return int(args.handler(args))
    except ContractError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
