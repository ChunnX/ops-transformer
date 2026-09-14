---
name: "qmla-build-install"
description: "Build and install QMLA custom operator package. Invoke when needing to compile, install, or rebuild quant_flash_mla_with_kvcache operators."
---

# QMLA Build & Install

## Environment Setup

```bash
conda activate wk
source /home/wangke293/Ascend_0827/cann/set_env.sh
```

## Build Custom Package

```bash
cd /home/wangke293/ops-transformer_qfa_mla
bash build.sh --pkg \
  --ops="quant_flash_mla_with_kvcache,quant_flash_mla_with_kvcache_metadata" \
  --soc=ascend950 \
  --vendor_name=custom \
  &>buildLog.log
```

**IMPORTANT**: 
- `--ops` must include BOTH `quant_flash_mla_with_kvcache` AND `quant_flash_mla_with_kvcache_metadata`
- `--vendor_name=custom` is required
- If also building qfa interface, add `quant_flash_attn,quant_flash_attn_metadata` to `--ops`

## Install Custom Package

```bash
cd build_out
./cann-ops-transformer-custom_linux-x86_64.run --quiet --install-path=/home/wangke293/Ascend_custom_qfa/
source /home/wangke293/Ascend_custom_qfa/vendors/custom_transformer/bin/set_env.bash
```

## Build & Install Torch Extension

```bash
cd /home/wangke293/ops-transformer_qfa_mla
python -m build --wheel -n
python -m pip install dist/*.whl --force-reinstall --no-deps
```


## Run Test

```bash
cd attention/quant_flash_mla_with_kvcache/tests/pytest
pytest -m func_rdv -v -s -k "CASE_NAME" &> runLog.log
```

## Key Paths

- Source: `/home/wangke293/ops-transformer_qfa_mla`
- Custom package install: `/home/wangke293/Ascend_custom_qfa/`
- CANN toolkit: `/home/wangke293/Ascend_0827/cann`
- Conda env: `wk`
- Python: `/root/miniconda3/envs/wk`
