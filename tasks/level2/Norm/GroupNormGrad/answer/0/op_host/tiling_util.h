/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 */

#pragma once

#include "register/op_impl_registry.h"
#include "platform/soc_spec.h"

namespace OpTiling {

bool IsRegbaseSocVersion(const gert::TilingParseContext* context);

bool IsRegbaseSocVersion(const gert::TilingContext* context);

const gert::Shape& EnsureNotScalar(const gert::Shape& inShape);
} // namespace OpTiling
