import numpy as np
import sys
import openpyxl
import pandas
import os
from tensorflow import bfloat16

cur_path=os.getcwd()

def get_casename(file,start,op):
    data=pandas.read_excel(file,sheet_name=op)
    case_name=data['case_name'][start]
    x3_dtype=data['x3_dtype'][start]
    world_size=data['world_size'][start]
    return case_name,x3_dtype,world_size

def get_data():
    case_name,x3_dtype,world_size=get_casename(file,strat,op)
    print(case_name)
    print(x3_dtype)
    print(world_size)
    if x3_dtype=="fp16":
        np_dtype=np.float16
    elif x3_dtype=="fp32":
        np_dtype=np.float32
    else:
        np_dtype= bfloat16.as_numpy_dtype
    # print("11111111111111:",np_dtype)
    temp_path=os.path.join(os.getcwd(),"bin_path",case_name)
    for i in range(int(world_size)):
        for bin_path in os.listdir(temp_path):
            if bin_path.endswith("cpu_output_"+str(i)+".bin"):
                # print(bin_path)
                cpu_path = os.path.join(temp_path, bin_path)
        npu_path=os.path.join(temp_path,"output",str(i),"output_npu.bin")
        hccl_path = os.path.join(temp_path, "output", str(i), "output_hccl.bin")
        print(npu_path,hccl_path)
        print("-----------------------------------------------------------------------")
        print("rank {} cpu data \n {}".format(i,np.fromfile(cpu_path,np.float32)))
        print("-----------------------------------------------------------------------")
        print("rank {} hccl data \n {}".format(i,np.fromfile(hccl_path,np_dtype)))
        print("-----------------------------------------------------------------------")
        print("rank {} npu data \n {}".format(i,np.fromfile(npu_path,np_dtype)))
        print("-----------------------------------------------------------------------")





if __name__ == '__main__':

    file=sys.argv[1]
    op=sys.argv[2]
    strat=int(sys.argv[3])-1
    get_data()
