# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under
# the terms and conditions of CANN Open Software License Agreement Version 2.0.

from pathlib import Path
import re


_OPS_DIR = Path(__file__).resolve().parents[1]


def _without_comments(source: str) -> str:
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    return re.sub(r"//.*?$", "", source, flags=re.MULTILINE)


def test_flash_kda_l0_uses_exactly_one_generated_entry_builder():
    source = _without_comments(
        (_OPS_DIR / "flash_kda_l0.cpp").read_text(encoding="utf-8")
    )

    assert source.count("cannbotdsl::generated::BuildFlashKdaEntry(") == 1
    assert "cannbotdsl::generated::FlashKdaEntryArguments" in source
    for forbidden in (
        "FlashKdaProviderArguments",
        "PrepareFlashKdaProviderArguments",
        "MakeFlashKdaProviderOpArgContext",
        "SetFlashKdaProviderWorkspaceTensors",
        "ProviderAclnnRegistrySpec",
        "ProviderKernelSpec",
        "AddProviderKernelLauncher",
        "kProviderAclnnVariants",
        "kProviderAclnnVariantCount",
        "cannbotdsl_provider_registry.h",
        "opdev/provider_kernel.h",
        "../../op_impl/cannbotdsl",
    ):
        assert forbidden not in source


def test_flash_kda_public_l0_and_l2_abi_remain_provider_agnostic():
    header = _without_comments(
        (_OPS_DIR / "flash_kda_l0.h").read_text(encoding="utf-8")
    )
    l2_source = _without_comments(
        (_OPS_DIR / "custom_flash_kda.cpp").read_text(encoding="utf-8")
    )
    compact = " ".join(header.split())
    assert (
        "aclnnStatus FlashKda( const aclTensor* k, const aclTensor* v, "
        "const aclTensor* q, const aclTensor* beta, const aclTensor* g, "
        "const aclTensor* aLog, const aclTensor* dtBias, "
        "const aclTensor* initialState, float scale, float lowerBound, "
        "const aclTensor* out, const aclTensor* finalState, "
        "aclOpExecutor* executor);"
        in compact
    )
    assert '#include "flash_kda_l0.h"' in l2_source
    for forbidden in (
        "cannbotdsl_provider_adapter.h",
        "cannbotdsl_provider_registry.h",
        "ProviderArguments",
        "ProviderKernelSpec",
        "ProviderAclnnRegistrySpec",
    ):
        assert forbidden not in header
        assert forbidden not in l2_source
