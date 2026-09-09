# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
_cannbotdsl_python="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/python"
export PYTHONPATH="${_cannbotdsl_python}${PYTHONPATH:+:${PYTHONPATH}}"
unset _cannbotdsl_python
