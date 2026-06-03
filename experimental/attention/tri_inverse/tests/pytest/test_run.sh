#!/bin/bash

TEST_TRI_INVERSE_SINGLE_SCRIPT="test_tri_inverse_single.py"
PROFILE_TRI_INVERSE_SCRIPT="profile_tri_inverse.py"

run_single() {
    echo "===== run tri_inverse single pytest ====="
    python3 -m pytest -rA -s ${TEST_TRI_INVERSE_SINGLE_SCRIPT} -v -m ci \
        -W ignore::UserWarning -W ignore::DeprecationWarning
}

show_help() {
    echo "Usage: $0 [single|profile|help]"
}

run_profile() {
    echo "===== run tri_inverse profiling ====="
    python3 ${PROFILE_TRI_INVERSE_SCRIPT}
}

if [ $# -ne 1 ]; then
    show_help
    exit 1
fi

case "$1" in
    single)
        run_single
        ;;
    profile)
        run_profile
        ;;
    help)
        show_help
        ;;
    *)
        show_help
        exit 1
        ;;
esac
