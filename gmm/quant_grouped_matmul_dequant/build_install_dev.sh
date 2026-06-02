#!/bin/bash
# ============================================================
# Device-side qgmmd build + install wrapper for the aarch64
# 310P3 server (sourced in CLAUDE.md "Device Test Server").
#
# - Activates pyenv .venv3.11
# - Sources CANN set_env.sh
# - Builds the qgmmd op with `gcc`/`g++` (NOT gcc-12 — that's Debian-only)
# - Installs the resulting .run to ~/.local/ascend-toolkit/
#
# Invoke from local box via:
#   ssh p00829650@192.168.13.113 \
#     'bash ~/Operators/ops-transformer/gmm/quant_grouped_matmul_dequant/build_install_dev.sh'
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 1. Activate pyenv venv (no aliases — explicit init).
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)" 2>/dev/null || true
pyenv activate .venv3.11

# 2. CANN env.
source "$HOME/.local/ascend-toolkit/cann-8.5.0/set_env.sh" 2>/dev/null \
  || source "$HOME/.local/ascend-toolkit/set_env.sh"

cd "$REPO_ROOT"

# 3. Build + install — aarch64 → plain gcc/g++.
CC=gcc CXX=g++ bash build.sh \
    --pkg \
    --ops=quant_grouped_matmul_dequant \
    --soc=ascend310p \
    --build-type=Release \
    -p "$HOME/.local/ascend-toolkit/cann-8.5.0"

# 4. Install the freshly-built run-pack.
RUN="$(ls -t build_out/cann-ops-transformer-custom_linux-aarch64.run 2>/dev/null \
       || ls -t build_out/custom_opp_*.run 2>/dev/null \
       || true | head -n 1)"
if [ -z "$RUN" ] || [ ! -f "$RUN" ]; then
    echo "ERROR: no .run package found under build_out/" >&2
    exit 1
fi

echo "Installing $RUN ..."
bash "$RUN" --install-path="$HOME/.local/ascend-toolkit/" --quiet --nox11

echo ""
echo "Done. Don't forget to activate the custom-op vendor env before running:"
echo "  source \$HOME/.local/ascend-toolkit/vendors/custom_transformer/bin/set_env.bash"
