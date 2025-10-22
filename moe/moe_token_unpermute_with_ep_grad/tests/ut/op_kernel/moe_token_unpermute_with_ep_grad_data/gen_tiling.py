import numpy as np
import sys

# [10, 3, 64] bfloat16_t
case0_params = [10, 3, 64, 30, 10, 38, 1, 0, 3, 0, 256, 1, 64, 3, 3, 16, 196352, 0, 10]
case1_params = [10, 1, 64, 10, 10, 38, 1, 0, 1, 0, 64, 1, 64, 1, 1, 16, 196352, 0, 10]
case2_params = [10, 3, 8192, 30, 10, 38, 1, 0, 3, 0, 7936, 2, 256, 1, 3, 16, 196352, 0, 10]

params_info = {
    "case0": case0_params,
    "case1": case1_params,
    "case2": case2_params
}

def main():
    params_list = params_info[sys.argv[1]]   # python gen_tiling.py case0  sys.argv[1]="case0"
    base_params = np.array(params_list, dtype=np.int64)
    tiling_file = open("tiling.bin", "wb")
    base_params.tofile(tiling_file)


if __name__ == '__main__':
    main()
