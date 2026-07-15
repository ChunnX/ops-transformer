#!/bin/bash
cd /mnt/workspace/cannBot/code/FIAAntiquant0615/torch_extension
source /mnt/workspace/cannBot/t00954448/vendors/custom_transformer/bin/set_env.bash
export PATH="/opt/buildtools/Python-3.11.4/bin:$PATH"
python3 test.py
