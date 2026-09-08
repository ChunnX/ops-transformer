# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
_native_python="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/python"
export PYTHONPATH="${_native_python}${PYTHONPATH:+:${PYTHONPATH}}"
unset _native_python
