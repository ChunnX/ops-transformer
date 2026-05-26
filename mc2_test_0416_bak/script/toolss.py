import numpy as np
import functools
import random
import time
import multiprocessing as mp
import gc
import psutil
import datetime
import copy

"""
@author: s00417895 for binaryDataCompare
@author: l00469792 for integration
"""


def getDataType(type_str):
    type_dict = {
        'fp32': np.float32,
        'fp16': np.float16,
        'int8': np.int8,
        'uint8': np.uint8,
        'int16': np.int16,
        'uint16': np.uint16,
        'int32': np.int32,
        'uint32': np.uint32,
        'int64': np.int64,
        'uint64': np.uint64,
        'bool': np.bool,
        'fp64': np.float64,
        'complex64': np.complex64,
        'complex128': np.complex128,
    }
    return type_dict[type_str]


def getStrDataType(type_id):
    type_dict = {
        0: "fp32",
        1: "fp16",
        2: "int8",
        3: "int32",
        4: "uint8",
        6: "int16",
        7: "uint16",
        8: "uint32",
        9: "int64",
        10: "uint64",
        11: "double",
        12: "bool",
        13: "string",
        14: "sint8",  # DT_DUAL_SUB_INT8 = 14,    /**< dual output int8 type */
        15: "suint8",  # DT_DUAL_SUB_UINT8 = 15,    /**< dual output uint8 type */
        16: "complex64",  # complex64 type
        17: "complex128",  # complex128 type
        18: "qint8",
        19: "qint16",
        20: "qint32",
        21: "quint8",
        22: "quint16",
        23: "resource",  # DT_RESOURCE = 23,          // resource type
        24: "string_ref",  # DT_STRING_REF = 24,        // string ref type
        25: "dual",  # /**< dual output type */
        26: "undefined"

    }
    return type_dict[int(float(type_id))]


def getStrDtype(dtype):
    type_dict = {
        np.float32: "fp32",
        np.float32: "fp16",
        np.int8: "int16",
    }
    return type_dict[dtype]


def gen_data(data_size, min, max, dtype):
    # return np.random.uniform(min, max, size=data_size).astype(getDataType(dtype))
    return np.random.uniform(min, max, size=data_size).astype(dtype)


def getData(data_path, shape, dtype):
    # data_pool = np.fromfile(data_path, dtype=getDataType(dtype))
    data_pool = np.fromfile(data_path, dtype)
    data_len = functools.reduce(lambda x, y: x * y, shape)
    # print('data_len: ', data_len)
    return data_pool[0:data_len].reshape(shape)


def dump_file(data, path, dtype):
    # path = os.path.join(self.data_path, path)
    # dtype = getStrDataType(dtype)
    # print("getStrDataType:",dtype)
    if dtype in ('fp16', 'FP16', 'Fp16'):
        np.array(data).astype(np.float16).tofile(path)
    elif dtype in ('fp32', 'FP32', 'Fp32'):
        np.array(data).astype(np.float32).tofile(path)
    elif dtype in ('int8', 'INT8', 'Int8'):
        np.array(data).astype(np.int8).tofile(path)
    elif dtype in ('int16', 'INT16', 'Int16'):
        np.array(data).astype(np.int16).tofile(path)
    elif dtype in ('int32', 'INT32', 'Int32'):
        np.array(data).astype(np.int32).tofile(path)
    elif dtype in ('int64', 'INT64', 'Int64'):
        np.array(data).astype(np.int64).tofile(path)
    elif dtype in ('uint8', 'UINT8', 'Uint8'):
        np.array(data).astype(np.uint8).tofile(path)
    elif dtype in ('uint16', 'UINT16', 'Uint16'):
        np.array(data).astype(np.uint16).tofile(path)
    elif dtype in ('uint32', 'UINT32', 'Uint32'):
        np.array(data).astype(np.uint32).tofile(path)
    elif dtype in ('uint64', 'UINT64', 'Uint64'):
        np.array(data).astype(np.uint64).tofile(path)
    elif dtype in ('fp64'):
        np.array(data).astype(np.float64).tofile(path)
    elif dtype in ('complex64'):
        np.array(data).astype(np.complex64).tofile(path)
    elif dtype in ('complex128'):
        np.array(data).astype(np.complex128).tofile(path)
    elif dtype in ('bool'):
        np.array(data).astype(np.bool).tofile(path)


def dump_data(inputData, path, dtype):
    # fileName=os.path.join(self.data_path,filename)
    outDataFmt = getStrDataType(dtype)
    fOutput = open(path, "wb")
    if outDataFmt == "fp16":
        for elem in inputData:
            fOutput.write(np.float16(elem).tobytes())
    elif outDataFmt == "fp32":
        for elem in inputData:
            fOutput.write(np.float32(elem).tobytes())
    elif outDataFmt == "int8":
        for elem in inputData:
            fOutput.write(np.int8(elem).tobytes())
    elif outDataFmt == "int32":
        for elem in inputData:
            fOutput.write(np.int32(elem).tobytes())
    elif outDataFmt == "uint8":
        for elem in inputData:
            fOutput.write(np.uint8(elem).tobytes())
    elif outDataFmt == "uint32":
        for elem in inputData:
            fOutput.write(np.uint32(elem).tobytes())
    elif outDataFmt == "uint16":
        for elem in inputData:
            fOutput.write(np.uint16(elem).tobytes())
    fOutput.close()
    print('file:%s dump success.' % path)


def read_file(path, dtype):
    dtype = getDataType(dtype)
    return np.fromfile(path, dtype=dtype)


# def calRelativediff(x,y):
#     return abs(float(x)-float(y))/max((float(max(abs(x),abs(y)))+10e-10),2**(-12))

def calRelativediff(x, y, diffThd):
    if float(max(abs(x), abs(y))) < float((1.0 / (1 << 14)) / diffThd):
        result = abs(float(x) - float(y)) / (float((1.0 / (1 << 14)) / diffThd) + 10e-10)
    else:
        result = abs(float(x) - float(y)) / (float(max(abs(x), abs(y))) + 10e-10)

    return result


def calRelativediffNumpy(dataCheck, dataExpect, diffThd):
    print("check*********************************")
    # a = np.abs(np.subtract(dataCheck, dataExpect))
    # b1 = np.maximum(np.abs(dataCheck), (np.abs(dataExpect)))
    # b2 = float((1.0 / (1 << 14)) / diffThd)
    # b = np.maximum(b1, b2)
    # result = np.divide(a, np.add(b, 10e-10))
    a = np.abs(np.subtract(dataCheck, dataExpect))
    b1 = np.maximum(np.abs(dataCheck), (np.abs(dataExpect)))
    b2 = float((1.0 / (1 << 14)) / diffThd)
    b = np.add(np.maximum(b1, b2), 10e-10)
    result = np.where(a < diffThd, a, a / b)
    return result


def displayOutput(dr, de, start, end, diffThd):
    print('\n--------------------------------------------------------------------')
    print('Loop \t  ExpectOut \t RealOut \t FpDiff \t RateDiff')
    print('--------------------------------------------------------------------')
    dataCount = dr.size
    splitCount = int(end - start)

    if splitCount <= 20:
        for i in range(splitCount + 1):
            j = i + start
            print('%d \t %.7f \t %.7f \t %.7f \t %.7f' % (
                start + i + 1, de[j], dr[j], abs(np.float64(de[j]) - np.float64(dr[j])),
                calRelativediff(de[j], dr[j], diffThd)))
    else:
        for i in range(10):
            j = i + start
            print('%d \t %.7f \t %.7f \t %.7f \t %.7f' % (
                start + i + 1, de[j], dr[j], abs(np.float64(de[j]) - np.float64(dr[j])),
                calRelativediff(de[j], dr[j], diffThd)))
        print('...   \t   ...   \t   ...   \t   ...    \t   ...')
        for i in range(splitCount - 10 + 1, splitCount + 1):
            j = i + start
            print('%d \t %.7f \t %.7f \t %.7f \t %.7f' % (
                start + i + 1, de[j], dr[j], abs(np.float64(de[j]) - np.float64(dr[j])),
                calRelativediff(de[j], dr[j], diffThd)))


def displayOutput_new(dr, de, de32, start, end, diffThd):
    print('\n--------------------------------------------------------------------')
    print('Loop \t  ExpFP32Out \t ExpFP16Out \t  NPUOut \t FpDiff(NPU-ExpFP16) \t RateDiff')
    print('--------------------------------------------------------------------')
    dataCount = dr.size
    splitCount = int(end - start)

    if splitCount <= 20:
        for i in range(splitCount + 1):
            j = i + start
            print('%d \t %.7f \t %.7f \t %.7f \t %.7f \t %.7f' % (
                start + i + 1, de32[j], de[j], dr[j], abs(np.float64(de[j]) - np.float64(dr[j])),
                calRelativediff(de[j], dr[j], diffThd)))
    else:
        for i in range(10):
            j = i + start
            print('%d \t %.7f \t %.7f \t %.7f \t %.7f \t %.7f' % (
                start + i + 1, de32[j], de[j], dr[j], abs(np.float64(de[j]) - np.float64(dr[j])),
                calRelativediff(de[j], dr[j], diffThd)))
        print('...   \t   ...   \t   ...   \t   ...    \t   ...')
        for i in range(splitCount - 10 + 1, splitCount + 1):
            j = i + start
            print('%d \t %.7f \t %.7f \t %.7f \t %.7f \t %.7f' % (
                start + i + 1, de32[j], de[j], dr[j], abs(np.float64(de[j]) - np.float64(dr[j])),
                calRelativediff(de[j], dr[j], diffThd)))


def displayErrorOutput(dr, de, rdiff, start, end, diffThd):
    print('\n----------------------------Error Line------------------------------')
    print('--------------------------------------------------------------------')
    print('Loop \t  ExpectOut \t RealOut \t FpDiff \t RateDiff')
    print('--------------------------------------------------------------------')
    dataCount = dr.size
    splitCount = int(end - start)
    count = 0
    for i in range(len(rdiff)):
        j = i + start
        if rdiff[j] > diffThd:
            count += 1
            print('%d \t %.7f \t %.7f \t %.7f \t %.7f' % (
                start + i + 1, de[j], dr[j], abs(np.float64(de[j]) - np.float64(dr[j])),
                calRelativediff(de[j], dr[j], diffThd)))
        if count == 100:
            break


def displayErrorOutput_new(datareal, datacomp, dataexp, err_idx, rdiff, start, end, diffThd):
    print('\n----------------------------Error Line------------------------------')
    print('--------------------------------------------------------------------')
    print('Loop \t  ExpFp32Out \t ExpFp16Out \t NPUOut \t FpDiff(NPU- ExpFp16) \t Fp16rateDiff')
    print('--------------------------------------------------------------------')
    dataCount = datareal.size
    splitCount = int(end - start)
    count = 0
    for i in err_idx:
        count += 1
        print('%d \t %.8f \t %.8f \t %.8f \t %.8f  \t %.8f ' % (
            i, dataexp[i], datacomp[i], datareal[i], rdiff[count - 1],
            calRelativediff(datacomp[i], datareal[i], diffThd)))
        if count == 100:
            break


# ɾ��ǰ�����
def broadcast1(shape):
    newlist = shape.copy()
    if len(newlist) > 1:
        for i in range(0, random.randint(1, len(newlist) - 1)):
            newlist.pop(0)
    return newlist


# �������ֵ��Ϊ1
def broadcast2(shape):
    newlist = shape.copy()
    indexlist = []
    if len(newlist) > 1:
        for i in range(0, random.randint(0, len(newlist) - 1)):
            axis = random.randint(0, len(newlist) - 1)
            if np.sum(list(map(lambda x: x > 1, [abs(x - axis) for x in indexlist]))) <= 1:
                newlist[axis] = 1
                indexlist.append(axis)
            else:
                pass
    return newlist


#
def broadcast3(shape):
    newlist = shape.copy()
    len_s = len(shape)
    indexlist = []
    if len(shape) < 8:
        len_d = random.randint(len_s + 1, 8)
        for i in range(0, random.randint(0, len(newlist) - 1)):
            axis = random.randint(0, len(newlist) - 1)
            if np.sum(list(map(lambda x: x > 1, [abs(x - axis) for x in indexlist]))) <= 1:
                newlist[axis] = 1
                indexlist.append(axis)
            else:
                pass
        for i in range(len_d - len_s):
            newlist.insert(0, 1)
    return newlist


# ��axis
def getAxis(shape):
    length = len(shape)
    axislist = []
    num = random.randint(1, length)
    for j in range(num):
        axis_loop = 1
        while axis_loop:
            axis = random.randint(-length, length - 1)
            if axis not in axislist and (axis + length) not in axislist and (axis - length) not in axislist:
                axislist.append(axis)
                axis_loop = 0
    axislist = list(axislist)
    # print(type(axislist))
    return axislist


def dataCompare(dataCheck, dataExpect, diffThd=0.01, pctThd=0.05):
    npu_shape = dataCheck.shape
    expect_shape = dataExpect.shape
    dataCheck = dataCheck.flatten()
    dataExpect = dataExpect.flatten()
    start = 0
    end = dataCheck.size - 1
    rdiff1 = calRelativediffNumpy(dataCheck[:len(dataCheck) // 8].astype(np.float32),
                                  dataExpect[:len(dataExpect) // 8].astype(np.float32), diffThd)
    rdiff2 = calRelativediffNumpy(dataCheck[len(dataCheck) // 8:2 * len(dataCheck) // 8].astype(np.float32),
                                  dataExpect[len(dataExpect) // 8:2 * len(dataCheck) // 8].astype(np.float32), diffThd)
    rdiff3 = calRelativediffNumpy(dataCheck[2 * len(dataCheck) // 8:3 * len(dataCheck) // 8].astype(np.float32),
                                  dataExpect[2 * len(dataExpect) // 8:3 * len(dataCheck) // 8].astype(np.float32),
                                  diffThd)
    rdiff4 = calRelativediffNumpy(dataCheck[3 * len(dataCheck) // 8:4 * len(dataCheck) // 8].astype(np.float32),
                                  dataExpect[3 * len(dataExpect) // 8:4 * len(dataCheck) // 8].astype(np.float32),
                                  diffThd)
    rdiff5 = calRelativediffNumpy(dataCheck[4 * len(dataCheck) // 8:5 * len(dataCheck) // 8].astype(np.float32),
                                  dataExpect[4 * len(dataExpect) // 8:5 * len(dataCheck) // 8].astype(np.float32),
                                  diffThd)
    rdiff6 = calRelativediffNumpy(dataCheck[5 * len(dataCheck) // 8:6 * len(dataCheck) // 8].astype(np.float32),
                                  dataExpect[5 * len(dataExpect) // 8:6 * len(dataCheck) // 8].astype(np.float32),
                                  diffThd)
    rdiff7 = calRelativediffNumpy(dataCheck[6 * len(dataCheck) // 8:7 * len(dataCheck) // 8].astype(np.float32),
                                  dataExpect[6 * len(dataExpect) // 8:7 * len(dataCheck) // 8].astype(np.float32),
                                  diffThd)
    rdiff8 = calRelativediffNumpy(dataCheck[7 * len(dataCheck) // 8:].astype(np.float32),
                                  dataExpect[7 * len(dataExpect) // 8:].astype(np.float32), diffThd)
    rdiff = np.concatenate((rdiff1, rdiff2, rdiff3, rdiff4, rdiff5, rdiff6, rdiff7, rdiff8), axis=0)
    print(len(rdiff1), "\n", len(rdiff2), "\n", len(rdiff3), "\n", len(rdiff4), len(rdiff5), "\n", len(rdiff6), "\n",
          len(rdiff7), "\n", len(rdiff8), "\n", len(rdiff))
    split_rdiff = rdiff[start:end + 1]
    splitCount = int(end - start + 1) if end != start else 1
    ltNum = rdiff1[rdiff1 < diffThd].size + rdiff2[rdiff2 < diffThd].size + rdiff3[rdiff3 < diffThd].size + rdiff4[
        rdiff4 < diffThd].size + rdiff5[rdiff5 < diffThd].size + rdiff6[rdiff6 < diffThd].size + rdiff7[
                rdiff7 < diffThd].size + rdiff8[rdiff8 < diffThd].size

    j = 0
    ltNum = ltNum + dataExpect[np.isinf(dataExpect)].size + dataExpect[np.isnan(dataExpect)].size
    ltPct = float(ltNum) / float(splitCount) * 100.0
    displayOutput(dataCheck, dataExpect, start, end, diffThd)
    pctThd = (1 - pctThd) * 100.0
    result = "PASS" if (ltPct >= pctThd) else "FAIL"
    if npu_shape != expect_shape:
        result = "FAIL"

    print('\n--------------------------------------------------------------------')
    print('DiffThd  \t PctThd   \t PctRlt   \t Result')
    print('--------------------------------------------------------------------')
    print('%.3f     \t %.2f%%   \t %.6f%%   \t %s   \n' % (diffThd, pctThd, ltPct, result))

    if npu_shape != expect_shape:
        print('============ out_shape is not equal expect!')
    else:
        print('============ out_shape is equal expect!')

    if result == "FAIL":
        displayErrorOutput(dataCheck, dataExpect, rdiff, start, end, diffThd)
    return result, ltPct


# --------------------------------------------------
# Author:TangJianhua 00505743 @ 2020/01/12
# --------------------------------------------------
def calcdiff(Check, Expect, diffThd):
    # result = list(map(lambda x,y: calRelativediff(x,y,diffThd), Check.astype(np.float32), Expect.astype(np.float32)))
    # result = list(map(lambda x,y: calRelativediff(x,y,diffThd), Check, Expect))
    # return result
    return list(map(lambda x, y: calRelativediff(x, y, diffThd), Check, Expect))


def dataComparev2(dataCheck, dataExpect, start, end, diffThd=0.01, pctThd=0.05):
    print("xxxxxxxxxxxxxx in dataComparev2")
    manager = mp.Manager()
    return_list = manager.list()
    starttime = time.time()
    NUMPROC = min([16, int(mp.cpu_count() / 2)])
    # for i in range(len(dataExpect)):
    #    if str(dataExpect[i]) =='nan' or str(dataExpect[i]) =='inf':
    #        print(i,dataExpect[i])
    splitstep = len(dataExpect) // NUMPROC  # 3273600
    rdiff = []
    if splitstep > 100:
        runtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print('%s I: splitstep is %d,using multi-processing.' % (runtime, splitstep))
        pool = mp.Pool(processes=NUMPROC)
        e = start
        result = []
        while e <= end:
            s = e
            e = s + splitstep
            if e > end + 1:
                e = end + 1
            result.append(pool.apply_async(calcdiff, args=(dataCheck[s:e], dataExpect[s:e], diffThd)))
        pool.close()
        pool.join()
        for p in result:
            rdiff.extend(p.get())
    else:
        runtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print('%s I: splitstep is %d,smaller data-set,using single processing.' % (runtime, splitstep))
        rdiff = list(map(lambda x, y: calRelativediff(x, y, diffThd), dataCheck.astype(np.float32),
                         dataExpect.astype(np.float32)))
    runtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print('%s I: getting rdiff have compeled.Duration:%0.2f second.' % (runtime, time.time() - starttime))
    split_rdiff = rdiff[start:end + 1]
    # print('split_rdiff',split_rdiff)
    splitCount = int(end - start + 1) if end != start else 1
    ltNum = functools.reduce(lambda x, y: x + 1 if y < diffThd else x, split_rdiff, 0)
    # print('=reduce=====Duration:',time.time()-starttime)
    ltPct = float(ltNum) / float(splitCount) * 100.0
    displayOutput(dataCheck, dataExpect, start, end, diffThd)
    pctThd = (1 - pctThd) * 100.0
    result = "Pass" if (ltPct >= pctThd) else "Failed"
    print('\n--------------------------------------------------------------------------------------------')
    print('DiffThd  \t PctThd   \t PctRlt   \t Result')
    print('--------------------------------------------------------------------------------------------')
    print('%.3f     \t %.2f%%   \t %.6f%%   \t %s   \n' % (diffThd, pctThd, ltPct, result))
    if result == "Failed":
        displayErrorOutput(dataCheck, dataExpect, rdiff, start, end, diffThd)
    # print('=display=====Duration:',time.time()-starttime)
    return result, ltPct


def dataCompare_new(NpuOut, ExpFP16Out, ExpFP32Out, diffThd=0.01, pctThd=0.05, MaxdiffThd=0.1):
    print("xxxxxxxxxxxxxx in dataCompare_new")
    npu_shape = NpuOut.shape
    expect_shape = ExpFP16Out.shape
    print('npu_out_shape:', npu_shape)
    print('expect_out_shape:', expect_shape)

    dataCheck = NpuOut.flatten()
    dataCompe = ExpFP16Out.flatten()
    dataExpect = ExpFP32Out.flatten()

    errcnt = 0
    start = 0
    end = dataCheck.size - 1

    # j=0
    # for i in range(len(dataExpect)):
    #     if 'nan' in str(dataExpect[i])  or 'inf' in str(dataExpect[i]) :
    #         if j<5:
    #            print(i,dataExpect[i],dataCheck[i])
    #            j=j+1
    #         else:
    #            break

    # rdiff = list(map(lambda x,y: calRelativediff(x,y,diffThd), dataCheck.astype(np.float32), dataExpect.astype(np.float32)))
    # ����������?
    # diff1 = list(map(lambda x,y: abs(x-y), dataCheck.astype(np.float32), dataExpect.astype(np.float32)))
    # diff2 = list(map(lambda x,y: abs(x-y), dataCompe.astype(np.float32), dataExpect.astype(np.float32)))

    diff1 = np.abs(np.subtract(dataCheck.astype(np.float32), dataExpect.astype(np.float32)))
    diff2 = np.abs(np.subtract(dataCompe.astype(np.float32), dataExpect.astype(np.float32)))

    # split_rdiff = rdiff[start:end+1]
    splitCount = int(end - start + 1) if end != start else 1
    print('splitCount: ', splitCount)
    print('float(splitCount): ', float(splitCount))

    err_idx = []
    # err_diff=[]

    diff = np.subtract(diff1, diff2)
    diff_index = np.where(diff > 0)
    print("test________collect_before：" + (str)(psutil.virtual_memory().percent) + "%")
    gc.collect()
    print("test________collect_after：" + (str)(psutil.virtual_memory().percent) + "%")

    rdiff = calRelativediffNumpy(dataCheck[diff_index], dataCompe[diff_index], diffThd)
    err_diff = rdiff[rdiff > diffThd]
    errcnt = err_diff.size

    # for i in range(splitCount):
    #     if diff1[i] > diff2[i]:
    #         rdiff=calRelativediff(dataCheck[i], dataCompe[i], diffThd)
    #         if rdiff > diffThd:
    #             errcnt=errcnt+1
    #             err_idx.append(i)
    #             err_diff.append(rdiff)
    ltNum = splitCount - errcnt

    # ltNum = functools.reduce(lambda x, y: x+1 if y < diffThd else x, split_rdiff, 0)
    ltPct = float(ltNum) / float(splitCount) * 100.0
    displayOutput_new(dataCheck, dataCompe, dataExpect, start, end, diffThd)
    pctThd = (1 - pctThd) * 100.0
    result = "Pass" if (ltPct >= pctThd) else "Failed"
    if len(err_diff) > 0:
        max_diff = max(err_diff)
        if max(err_diff) >= MaxdiffThd:
            result = "Failed"
    if npu_shape != expect_shape:
        result = "Failed"

    print('\n--------------------------------------------------------------------')
    print('DiffThd  \t PctThd   \t PctRlt   \t Result')
    print('--------------------------------------------------------------------')
    print('%.3f     \t %.2f%%   \t %.6f%%   \t %s   \n' % (diffThd, pctThd, ltPct, result))

    if len(err_diff) > 0:
        print('=============maximum error is:', max_diff)

    if npu_shape != expect_shape:
        print('============ out_shape is not equal expect!')
    else:
        print('============ out_shape is equal expect!')

    if result == "Failed":
        displayErrorOutput_new(dataCheck, dataCompe, dataExpect, err_idx, err_diff, start, end, diffThd)
    return result, ltPct


def data_compare_np(npu_output, cpu_output, diff_thd=0.01, pct_thd=0.05, max_diff_hd=0.1,
                    precision_method=0,
                    rtol=0.005, atol=0.000025, output_dtype=None):
    max_error_idx = 10000000
    real_data = npu_output.flatten()
    data_compe = cpu_output.flatten()
    if real_data.size == 0 and real_data.size == data_compe.size:
        print_log(
            'The npu_output is [],and it is same as bm_output, the result of data_compare is \"PASS\"')
        return "PASS", 100.0, 0
    start = 0
    end = real_data.size - 1
    if end < start:
        end = start
    max_error = 0
    result = "FAIL"
    if real_data.size != data_compe.size:
        print_log(
            'Error,the size of npu output[%s] and benchmark[%s] is not equal.' % (real_data.size, data_compe.size))
        return result, 0.0, max_error

    overflows_count = data_compe[np.isinf(
        data_compe)].size + data_compe[np.isnan(data_compe)].size
    if overflows_count > 0:
        print_log('Overflow,size:%s,benchmark_output:%s, %s' % (
            overflows_count, data_compe[np.isinf(data_compe)][0:10], data_compe[np.isnan(data_compe)][0:10]))

    split_count = int(end - start + 1) if end != start else 1
    print_log('split_count:%s; max_diff_hd:%s;' %
              (float(split_count), max_diff_hd))

    has_nan_inf = False
    if 'nan' in str(real_data) or 'inf' in str(real_data) or 'nan' in str(data_compe) or 'inf' in str(data_compe):
        has_nan_inf = True

    # 默认精度比对方式
    if precision_method == 0:
        try:
            diff_abs = np.abs(np.subtract(real_data.astype(
                np.float32), data_compe.astype(np.float32)))
        except MemoryError:
            return result, 0.0, max_error
        if has_nan_inf:
            err_diff, err_idx = cal_nan_inf_diff(real_data, data_compe, output_dtype, diff_abs, diff_thd)
        else:
            diff_index = np.where(diff_abs > 0)
            rdiff = cal_relative_diff_np(real_data[diff_index].astype(np.float32),
                                         data_compe[diff_index].astype(np.float32),
                                         diff_thd)
            err_diff = rdiff[rdiff > diff_thd]
            diff_idx_list = diff_index[0]
            err_idx = diff_idx_list[np.where(rdiff > diff_thd)]

        fulfill_percent = float(split_count - err_diff.size) / \
                          float(split_count) * 100.0

        display_output(real_data, data_compe, start, end, diff_thd)
        pct_thd = (1 - pct_thd) * 100.0
        result = "PASS" if (fulfill_percent >= pct_thd) else "FAIL"
        if len(err_diff) > 0:
            max_error = max(err_diff[0:max_error_idx])
            if max_error > max_diff_hd:
                result = "FAIL"
                print("max_error > max_diff_hd")
        print_log(
            '---------------------------------------------------------------------------------------')
        print_log('DiffThd  \t PctThd   \t PctRlt   \t Result')
        print_log(
            '---------------------------------------------------------------------------------------')
        print_log('%.4f     \t %.2f%%   \t %.6f%%   \t %s' %
                  (diff_thd, pct_thd, fulfill_percent, result))
        if len(err_diff) > 0:
            print_log('Max-RelativeError is: %s. Threshold is: %s.' %
                      (max_error, max_diff_hd))
        if result == "FAIL":
            display_error_output(real_data, data_compe,
                                 err_idx, err_diff[0:max_error_idx])

    # np.isclose精度比对方式
    if precision_method == 1:
        if str(real_data.dtype) == 'bfloat16':
            diff_result = np.isclose(real_data.astype(np.float32), data_compe.astype(np.float32), rtol=rtol, atol=atol,
                                     equal_nan=True)
        else:
            diff_result = np.isclose(real_data, data_compe, rtol=rtol, atol=atol, equal_nan=True)
        err_idx = np.where(diff_result != np.array((True,)))[0]

        diff_abs = abs(data_compe - real_data)
        eps = 2.220446049250313e-16
        err_diff = diff_abs / (np.abs(data_compe) + eps)
        err_diff = err_diff[err_idx]

        fulfill_percent = float(split_count - err_idx.size) / \
                          float(split_count) * 100.0

        display_output_np_isclose(real_data, data_compe, start, end)
        pct_thd = (1 - pct_thd) * 100.0
        result = "PASS" if (fulfill_percent >= pct_thd) else "FAIL"
        if len(err_diff) > 0:
            max_error = max(err_diff[0:max_error_idx])
            if max_error >= max_diff_hd:
                result = "FAIL"
        print_log(
            '---------------------------------------------------------------------------------------')
        print_log('Rtol   \t Atol   \t PctThd   \t PctRlt   \t Result')
        print_log(
            '---------------------------------------------------------------------------------------')
        print_log('%.4f    \t %.6f  \t %.2f%%   \t %.6f%%   \t %s' %
                  (rtol, atol, pct_thd, fulfill_percent, result))
        if len(err_diff) > 0:
            print_log('Max-RelativeError is: %s. Threshold is: %s.' %
                      (max_error, max_diff_hd))
        if result == "FAIL":
            display_error_output(real_data, data_compe,
                                 err_idx, err_diff[0:max_error_idx])
    # 余弦相似度对比
    if precision_method == 3:
        start_time = time()
        cos = cosine_similarity(npu_output, cpu_output)
        print_log(
            'The data comparison by cosine_similarity is：%s; Duration:%0.2f second.' % (cos, (time() - start_time)))
        result = "PASS" if (cos >= 0.99) else "FAIL"
        fulfill_percent = cos
        max_error = 'NA'

    return result, fulfill_percent


def cal_nan_inf_diff(real_data, expect_data, output_dtype, diff_abs, diff_thd):
    err_diff = []
    err_idx = []

    if output_dtype == 'fp32':
        inf_value = 3.4028e38
    elif output_dtype == 'bf16':
        inf_value = 3.38e38
    else:
        inf_value = 65504

    real_data_copy = copy.deepcopy(real_data)
    expect_data_copy = copy.deepcopy(expect_data)
    inf_idx = np.where(np.isinf(real_data_copy))[0]
    pos_inf_idx = np.where(real_data_copy[np.isinf(real_data_copy)] > 0)[0]
    neg_inf_idx = np.where(real_data_copy[np.isinf(real_data_copy)] < 0)[0]
    real_data_copy[inf_idx[pos_inf_idx]] = inf_value
    real_data_copy[inf_idx[neg_inf_idx]] = -inf_value
    inf_idx = np.where(np.isinf(expect_data_copy))[0]
    pos_inf_idx = np.where(expect_data_copy[np.isinf(expect_data_copy)] > 0)[0]
    neg_inf_idx = np.where(expect_data_copy[np.isinf(expect_data_copy)] < 0)[0]
    expect_data_copy[inf_idx[pos_inf_idx]] = inf_value
    expect_data_copy[inf_idx[neg_inf_idx]] = -inf_value

    num_idx = np.where(~np.isnan(real_data_copy) & ~np.isinf(real_data_copy) & ~np.isnan(expect_data_copy) & ~np.isinf(
        expect_data_copy))
    nan_inf_idx = np.setdiff1d(np.arange(len(real_data_copy)), num_idx)

    rdiff = cal_relative_diff_np(real_data_copy[num_idx].astype(np.float32),
                                 expect_data_copy[num_idx].astype(np.float32),
                                 diff_thd)
    num_err_diff = rdiff[rdiff > diff_thd]
    diff_idx_list = num_idx[0]
    num_err_idx = diff_idx_list[np.where(rdiff > diff_thd)]

    real_data_str = list(map(lambda x: str(x), real_data[nan_inf_idx].tolist()))
    expect_data_str = list(map(lambda x: str(x), expect_data[nan_inf_idx].tolist()))
    temp_err_idx = np.where(np.array(real_data_str) != np.array(expect_data_str))[0]
    nan_inf_err_idx = nan_inf_idx[temp_err_idx]
    nan_inf_err_diff = diff_abs[nan_inf_err_idx]

    err_idx = num_err_idx.tolist() + nan_inf_err_idx.tolist()
    err_diff = num_err_diff.tolist() + nan_inf_err_diff.tolist()

    return np.array(err_diff), np.array(err_idx)


def cal_relative_diff_np(real_data, expect_data, diff_thd):
    a = np.abs(np.subtract(real_data, expect_data))
    b1 = np.maximum(np.abs(real_data), (np.abs(expect_data)))
    b2 = float((1.0 / (1 << 14)) / diff_thd)
    b = np.add(np.maximum(b1, b2), 10e-10)
    result = np.where(a < diff_thd, a, a / b)
    return result


def display_output(real_data, expect_data, start, end, diff_thd, expect_fp32_data=None, if_mix=False):
    def display_inner(idx):
        j = idx + start
        if if_mix:
            diff_rate = cal_relative_diff_mix(
                expect_data[j], real_data[j], diff_thd)
        else:

            diff_rate = cal_relative_diff(
                expect_data[j], real_data[j], diff_thd)

        if "inf" in str(expect_data[j]) or "nan" in str(expect_data[j]):
            diff_abs = "inf" if "inf" in str(expect_data[j]) else "nan"
            if expect_fp32_data is not None:
                print_log('%08d \t %-7s \t %-7s \t %-7s \t %-7s \t %-7s' % (
                    start + idx + 1, expect_fp32_data[j], expect_data[j], real_data[j], diff_abs, diff_rate))
            else:
                print_log('%08d \t %-7s \t %-7s \t %-7s \t %-7s' % (
                    start + idx + 1, expect_data[j], real_data[j], diff_abs, diff_rate))
        else:
            diff_abs = abs(np.float64(
                expect_data[j]) - np.float64(real_data[j]))
            if expect_fp32_data is not None:
                print_log('%08d \t %0.7f \t %0.7f \t %0.7f \t %0.7f \t %0.7f' % (
                    start + idx + 1, expect_fp32_data[j], expect_data[j], real_data[j], diff_abs, diff_rate))
            else:
                print_log('%08d \t %0.7f \t %0.7f \t %0.7f \t %0.7f' % (
                    start + idx + 1, expect_data[j], real_data[j], diff_abs, diff_rate))

    print_log(
        '---------------------------------------------------------------------------------------')
    if expect_fp32_data is not None:
        print_log(
            'Loop \t ExpFP32Out \t ExpFP16Out \t NPUOut \tFpDiff(min) \t RateDiff')
    else:
        print_log('Loop \t ExpectOut \t RealOut \t FpDiff \t RateDiff')
    print_log(
        '---------------------------------------------------------------------------------------')
    split_count = int(end - start)
    if split_count <= 20:
        for i in range(split_count + 1):
            display_inner(i)
    else:
        for i in range(10):
            display_inner(i)
        print_log('...   \t   ...   \t   ...   \t   ...    \t   ...')
        for i in range(split_count - 10 + 1, split_count + 1):
            display_inner(i)

def print_log(data=None, level='INFO'):
    print("[%s] [%s] %s" % (datetime.datetime.now().strftime(
        "%Y/%m/%d %H:%M:%S"), level, data))

def cal_relative_diff(real_data, expect_data, diff_thd, type_str='fp16'):
    if 'nan' in str(expect_data) or 'inf' in str(expect_data):
        if type_str.lower() == 'fp16':
            expect_data = 65504
        else:
            expect_data = 3.4028e38
    diff = abs(float(real_data) - float(expect_data))
    if abs(float(real_data) - float(expect_data)) < diff_thd:
        result = diff
    else:
        result = diff / (float(max(abs(real_data), abs(expect_data))) + 10e-10)
    return result

def display_error_output(real_data, expect_data, err_idx, relative_diff):
    print_log(
        'Error Line-----------------------------------------------------------------------------')
    print_log('Loop \t ExpectOut \t RealOut \t FpDiff \t RateDiff')
    print_log(
        '---------------------------------------------------------------------------------------')
    count = 0
    len_err = len(err_idx)
    for i in err_idx:
        count += 1
        if count < 10 or (90 < count < 100):
            print_log('%08d \t %.7f \t %.7f \t %.7f \t %.7f' % (
                i, expect_data[i], real_data[i], abs(np.float64(
                    expect_data[i]) - np.float64(real_data[i])),
                relative_diff[count - 1]))
        elif count == 10 or (count == 100 and len_err > 100):
            dot_3 = '...'
            print_log('%08s \t %07s \t %07s \t %07s \t %07s' %
                      (dot_3, dot_3, dot_3, dot_3, dot_3))
        elif count > 100:
            break

    print_log(
        'Max-RE line:---------------------------------------------------------------------------')
    max_error = max(relative_diff)
    m_idx_list = err_idx[np.where(relative_diff == max_error)]
    m_count = 0
    for m_idx in m_idx_list:
        m_count += 1
        if m_count < 4:
            print_log('%08d \t %.7f \t %.7f \t %.7f \t %.7f' % (
                m_idx, expect_data[m_idx], real_data[m_idx],
                abs(np.float64(expect_data[m_idx]) -
                    np.float64(real_data[m_idx])),
                max_error))
        else:
            break
    print_log(
        '---------------------------------------------------------------------------------------')
