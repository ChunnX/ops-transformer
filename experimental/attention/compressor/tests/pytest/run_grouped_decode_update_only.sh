#!/bin/bash
set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR"

echo "===== grouped decode x_cache update-only non-compress cases ====="
python3 -m pytest -rA -s \
    test_compressor_grouped_decode_xcache.py::test_grouped_decode_x_cache_update_only \
    test_compressor_grouped_decode_xcache.py::test_grouped_decode_x_cache_updates_all_valid_mtp_tokens \
    test_compressor_grouped_decode_xcache.py::test_grouped_decode_li_inactive_round_is_read_only \
    test_compressor_grouped_decode_xcache.py::test_grouped_decode_li_mixed_phase_keeps_active_batch \
    -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning
