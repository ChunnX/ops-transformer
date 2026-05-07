#!/bin/bash
set -e

SOC="ascend950"

# grouped_matmul
bash build.sh -u --ophost --ops=grouped_matmul --soc=${SOC}
# bash build.sh -u --opapi --ops=grouped_matmul --soc=${SOC}
# bash build.sh -u --opkernel --ops=grouped_matmul --soc=${SOC}

# grouped_matmul_finalize_routing
# bash build.sh -u --ophost --ops=grouped_matmul_finalize_routing --soc=${SOC}
# bash build.sh -u --opapi --ops=grouped_matmul_finalize_routing --soc=${SOC}
# bash build.sh -u --opkernel --ops=grouped_matmul_finalize_routing --soc=${SOC}
