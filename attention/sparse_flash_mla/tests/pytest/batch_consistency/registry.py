#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Canonical names and compatibility aliases for consistency modes."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, Optional, Tuple
import warnings


class ConsistencyMode(str, Enum):
    REORDER = "reorder"
    SPLIT = "split"
    TOKEN_SPLIT = "token-split"
    SHAPE_CHANGE = "shape-change"


@dataclass(frozen=True)
class ModeSpec:
    mode: ConsistencyMode
    numeric_alias: int
    implemented: bool
    min_batch: int
    requires_semantic_guard: bool


class ModeRegistry:
    """Owns mode parsing so shell scripts never need mode semantics."""

    SPECS: Tuple[ModeSpec, ...] = (
        ModeSpec(ConsistencyMode.REORDER, 1, True, 2, False),
        ModeSpec(ConsistencyMode.SPLIT, 2, True, 2, False),
        ModeSpec(ConsistencyMode.TOKEN_SPLIT, 3, True, 1, True),
        ModeSpec(ConsistencyMode.SHAPE_CHANGE, 4, False, 1, True),
    )
    PROFILES: Dict[str, Tuple[ConsistencyMode, ...]] = {
        "blackbox-basic": (ConsistencyMode.REORDER, ConsistencyMode.SPLIT),
        "blackbox-supported": (
            ConsistencyMode.REORDER,
            ConsistencyMode.SPLIT,
            ConsistencyMode.TOKEN_SPLIT,
        ),
        "blackbox-all": (
            ConsistencyMode.REORDER,
            ConsistencyMode.SPLIT,
            ConsistencyMode.TOKEN_SPLIT,
            ConsistencyMode.SHAPE_CHANGE,
        ),
    }

    @classmethod
    def parse(cls, value: str, warn_legacy: bool = True) -> Tuple[ConsistencyMode, ...]:
        tokens = tuple(token.strip().lower() for token in value.split(",") if token.strip())
        if not tokens:
            raise ValueError("At least one consistency mode is required")

        numeric = tuple(token.isdigit() for token in tokens)
        if any(numeric) and not all(numeric):
            raise ValueError("Do not mix numeric and named consistency modes")

        if all(numeric):
            if warn_legacy:
                warnings.warn(
                    "Numeric consistency modes are deprecated; use reorder,split,...",
                    DeprecationWarning,
                    stacklevel=2,
                )
            alias_map = {str(spec.numeric_alias): spec.mode for spec in cls.SPECS}
        else:
            alias_map = {spec.mode.value: spec.mode for spec in cls.SPECS}

        unknown = [token for token in tokens if token not in alias_map]
        if unknown:
            raise ValueError(f"Unknown consistency modes: {unknown}")

        result = []
        for token in tokens:
            mode = alias_map[token]
            if mode not in result:
                result.append(mode)
        return tuple(result)

    @classmethod
    def resolve_profile(cls, name: str) -> Tuple[ConsistencyMode, ...]:
        try:
            return cls.PROFILES[name]
        except KeyError as error:
            raise ValueError(f"Unknown consistency profile: {name}") from error

    @classmethod
    def resolve(
        cls,
        modes: Optional[str] = None,
        profile: Optional[str] = None,
        warn_legacy: bool = True,
    ) -> Tuple[ConsistencyMode, ...]:
        """Resolve one explicit mode list or one named profile."""
        if modes and profile:
            raise ValueError("Specify either consistency modes or a profile, not both")
        if modes:
            return cls.parse(modes, warn_legacy=warn_legacy)
        return cls.resolve_profile(profile or "blackbox-basic")

    @classmethod
    def get_spec(cls, mode: ConsistencyMode) -> ModeSpec:
        return next(spec for spec in cls.SPECS if spec.mode == mode)

    @classmethod
    def unsupported(cls, modes: Iterable[ConsistencyMode]) -> Tuple[ConsistencyMode, ...]:
        return tuple(mode for mode in modes if not cls.get_spec(mode).implemented)
