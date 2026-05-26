import os.path
import os
import shutil

import numpy as np
import time
from typing import List, Union, Dict
from functools import reduce, wraps
import xlsxwriter
import multiprocessing
from collections import defaultdict
import csv
import yaml
import traceback
from ml_dtypes import bfloat16
from absl import app
from absl import flags
import psutil
import string


class TestLog:
    DEBUG_LEVEL = 3
    INFO_LEVEL = 2
    WARN_LEVEL = 1
    ERROR_LEVEL = 0

    def __init__(self, log_level: int):
        self.log_level = log_level
        self.process_name = ""

    def set_process_name(self, process_name: str):
        self.process_name = process_name

    def debug(self, msg: str):
        if self.log_level >= TestLog.DEBUG_LEVEL:
            print("[%s] [%s] [DEBUG] %s" % (
                self.process_name, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), msg), flush=True)
            # print("[" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())) + "] [DEBUG]" + msg, flush=True)

    def info(self, msg: str):
        if self.log_level >= TestLog.INFO_LEVEL:
            print("[%s] [%s] [INFO] %s" % (
                self.process_name, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), msg), flush=True)

    def warn(self, msg: str):
        if self.log_level >= TestLog.WARN_LEVEL:
            print("[%s] [%s] [WARN] %s" % (
                self.process_name, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), msg), flush=True)

    def error(self, msg: str):
        if self.log_level >= TestLog.ERROR_LEVEL:
            print("[%s] [%s] [ERROR] %s" % (
                self.process_name, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), msg), flush=True)


OP_LOGGER = TestLog(TestLog.DEBUG_LEVEL)


def func_log_wrapper():
    def _out_wrapper(func):
        @wraps(func)
        def _in_wrapper(*kargs, **kwargs):
            OP_LOGGER.debug("start to run: %s" % func.__name__)
            x = func(*kargs, **kwargs)
            OP_LOGGER.debug("end to run: %s" % func.__name__)
            return x

        return _in_wrapper

    return _out_wrapper


class REHistogramBins:

    @classmethod
    def get_histogram_bins(cls, dtype: str) -> List[float]:
        if "float16" == dtype:
            return [0, 0.001, 0.002, 0.005, 0.01, np.inf]
        if "float32" == dtype:
            return [0, 0.000001, 0.00001, 0.0001, 0.0005, np.inf]
        return [0, 0.004, 0.008, np.inf]  # for bf16

    @classmethod
    def get_histogram_bin_titles(cls, dtype: str) -> List[str]:
        if np.float16 == dtype or "float16" == dtype:
            return ["千分之\n0~1", "千分之\n1~2", "千分之\n2~5", "千分之\n5~10", "千分之\n>10"]
        if np.float32 == dtype or "float32" == dtype:
            return ["百万分\n0~1", "百万分\n1~10", "十万分\n1~10", "万分\n1~10", "万分\n>5"]
        return ["千分\n0~4", "千分\n4~8", "千分\n>8"]  # for bf16

    @classmethod
    def get_ulp_histogram_bin(cls, ulp_limit: int) -> List[float]:
        if ulp_limit <= 4:
            return [-np.inf, -ulp_limit*10, -ulp_limit*5, -ulp_limit, 0, ulp_limit, ulp_limit*5, ulp_limit*10, np.inf]
        return [-np.inf, -ulp_limit*4, -ulp_limit*2, -ulp_limit, 0, ulp_limit, ulp_limit*2, ulp_limit*4, np.inf]

    @classmethod
    def get_ulp_histogram_bin_titles(cls, ulp_limit: int) -> List[str]:
        if ulp_limit <= 4:
            ulp_range = [-ulp_limit*10, -ulp_limit*5, -ulp_limit, 0, ulp_limit, ulp_limit*5, ulp_limit*10]
        else:
            ulp_range = [-ulp_limit*4, -ulp_limit*2, -ulp_limit, 0, ulp_limit, ulp_limit*2, ulp_limit*4]
        ulp_title = ["小于\n%s" % ulp_range[0]]
        for i in range(len(ulp_range)-1):
            ulp_str = "%s~%s" % (ulp_range[i], ulp_range[i+1])
            ulp_title.append(ulp_str)
        ulp_title.append("大于\n%s" % ulp_range[-1])
        return ulp_title


class REDistributeInfo:
    def __init__(self, histogram_stat: List[int], histogram_bins: List[float], max_re: float, max_re_value_info: List,
                 avg_re: float):
        self.histogram_stat = histogram_stat
        self.histogram_bins = histogram_bins
        self.total_cnt = reduce(lambda x, y: x + y, histogram_stat)
        if self.total_cnt == 0:
            self.histogram_stat_ratio = [0 for _ in histogram_stat]
            self.histogram_stat_ratio_str = ["0" for _ in histogram_stat]
        else:
            self.histogram_stat_ratio = [(x / self.total_cnt) for x in histogram_stat]
            self.histogram_stat_ratio_str = ["%.2f" % (x / self.total_cnt * 100) for x in histogram_stat]
        self.max_re = float(max_re)
        self.avg_re = float(avg_re)
        self.max_re_value_info = max_re_value_info  # (npu_value, golden_value)

    def __str__(self):
        histogram_info_str = ";".join(["%f~%f: %s%%(%d)" % (
            self.histogram_bins[i], self.histogram_bins[i + 1], self.histogram_stat_ratio_str[i],
            self.histogram_stat[i]) for i in range(len(self.histogram_stat))])

        return "histogram info: %s\n, max re: %f(actual:%f, golden:%f), avg re: %f" % (
            histogram_info_str, self.max_re, self.max_re_value_info[0], self.max_re_value_info[1], self.avg_re)


class ULPDistributeInfo:
    def __init__(self, histogram_bin, histogram_stat, total_cnt: Union[int, np.ndarray],
                 abs_ulp_err_cnt: Union[int, np.ndarray], max_ulp: Union[float, np.ndarray],
                 min_ulp: Union[float, np.ndarray], avg_ulp: Union[float, np.ndarray],
                 avg_abs_ulp: Union[float, np.ndarray], max_value_info=None, min_val_info=None):
        self.total_cnt = total_cnt
        self.abs_ulp_err_cnt = abs_ulp_err_cnt
        self.max_ulp = np.floor(max_ulp * 2) / 2
        self.min_ulp = np.floor(min_ulp) / 2
        self.avg_ulp = np.floor(avg_ulp) / 2
        self.avg_abs_ulp = np.floor(avg_abs_ulp) / 2
        self.histogram_bin = histogram_bin
        self.histogram_stat = histogram_stat
        if self.total_cnt == 0:
            self.histogram_stat_ratio = [0 for _ in histogram_stat]
            self.histogram_stat_ratio_str = ["0" for _ in histogram_stat]
        else:
            self.histogram_stat_ratio = [(x / self.total_cnt) for x in histogram_stat]
            self.histogram_stat_ratio_str = ["%.2f" % (x / self.total_cnt * 100) for x in histogram_stat]
        self.max_val_info = max_value_info  # npu, golden
        self.min_val_info = min_val_info  # npu, golden

    def __str__(self):
        histogram_info_str = ";".join(["%f~%f: %s%%(%d)" % (
            self.histogram_bins[i], self.histogram_bins[i + 1], self.histogram_stat_ratio_str[i],
            self.histogram_stat[i]) for i in range(len(self.histogram_stat))])

        return "histogram info: %s\n, max ulp: %f(actual:%f, golden:%f), min ulp: %f(actual:%f, golden:%f), " \
               "avg ulp: %f, avg abs ulp: %f" % (histogram_info_str, self.max_ulp, self.max_value_info[0],
                                                 self.max_value_info[1], self.min_ulp, self.min_value_info[0],
                                                 self.min_value_info[1], self.avg_ulp, self.avg_abs_ulp)


class SmallValueInfo:
    def __init__(self, small_value_thresh: float,
                 small_value_count: Union[int, np.ndarray],
                 small_value_err_count: Union[int, np.ndarray], total_cnt: Union[int, np.ndarray]):
        self.small_value_thresh = small_value_thresh
        self.small_value_count = small_value_count
        self.small_value_err_count = small_value_err_count
        self.total_cnt = total_cnt
        self.small_value_ratio = 0 if total_cnt == 0 else small_value_count / total_cnt
        self.small_value_err_ratio = 0 if small_value_count == 0 else small_value_err_count / small_value_count

    def __str__(self):
        return "small info: small value count: %d, small value err count: %d" % (
            self.small_value_count, self.small_value_err_count)


class EBInfo:
    def __init__(self, equal_cnt: Union[int, np.ndarray], larger_cnt: Union[int, np.ndarray],
                 smaller_cnt: Union[int, np.ndarray], total_cnt: Union[int, np.ndarray]):
        self.equal_cnt = equal_cnt
        self.larger_cnt = larger_cnt
        self.smaller_cnt = smaller_cnt
        self.total_cnt = total_cnt
        self.comparable_cnt = equal_cnt + larger_cnt + smaller_cnt
        self.eb = abs(self.larger_cnt - self.smaller_cnt) / (self.comparable_cnt + 0.000001)

    def __str__(self):
        return "eb info: total count: %d, larger count: %d, smaller count: %d" % (
            self.total_cnt, self.larger_cnt, self.smaller_cnt)


class PrecisionStatInfo:
    def __init__(self, re_distribute_info: REDistributeInfo, small_value_info: SmallValueInfo,
                 eb_info: EBInfo, rmse: float, ulp_distribute_info: ULPDistributeInfo):
        self.re_distribute_info = re_distribute_info
        self.small_value_info = small_value_info
        self.eb_info = eb_info
        self.rmse = rmse
        self.ulp_distribute_info = ulp_distribute_info

    def __str__(self):
        return "Precision Stat Info:\n%s\n%s\n%s\n%s" % (
            self.re_distribute_info, self.small_value_info, self.eb_info, "rmse:%f" % self.rmse)


class TestCaseInfo:
    def __init__(self, op_type: str, case_name: str, case_param_info: str, case_input_value_range_info: str):
        self.op_type = op_type
        self.case_name = case_name
        self.case_param_info = case_param_info
        self.case_input_value_range_info = case_input_value_range_info


class PrecisionCheckStandard:
    NAME = "PrecisionCheckStandard"
    SHEET_NAME = "精度标准"

    def __init__(self, standard_name: str, data_type: str, sheet_name: str):
        self.standard_name = standard_name
        self.data_type = data_type
        self.sheet_name = sheet_name

    def __str__(self):
        return "%s_%s" % (self.standard_name, self.data_type)

    @classmethod
    def build_by_yml_info(cls, data_type: str, **kwargs):
        return PrecisionCheckStandard(PrecisionCheckStandard.NAME, data_type, PrecisionCheckStandard.SHEET_NAME)


class BinaryCompareStandard(PrecisionCheckStandard):
    NAME = "BinaryCompareStandard"
    SHEET_NAME = "二进制一致"

    def __init__(self, data_type: str):
        super().__init__("BinaryCompareStandard", data_type, BinaryCompareStandard.SHEET_NAME)

    def __str__(self):
        return super().__str__()

    @classmethod
    def build_by_yml_info(cls, data_type: str, **kwargs):
        return BinaryCompareStandard(data_type)


class AbsoluteThreshStandard(PrecisionCheckStandard):
    NAME = "AbsoluteThreshStandard"
    SHEET_NAME = "绝对阈值"

    def __init__(self, data_type: str, rtol: float, small_value: float, small_value_atol: float, ulp_limit: int,
                 golden_dtype: Union[str, None] = None):
        """

        :param rtol: relative tolerance
        :param small_value: less than small value not use relative tolerance, use small value absolute tolerance
        :param small_value_atol: small value absolute tolerance
        """
        super().__init__("AbsoluteThreshStandard", data_type, AbsoluteThreshStandard.SHEET_NAME)
        self.rtol = rtol
        self.small_value = small_value
        self.small_value_atol = small_value_atol
        self.ulp_limit = ulp_limit

        if golden_dtype is None:
            self.golden_dtype = self._get_golden_dtype(data_type)
        else:
            self.golden_dtype = golden_dtype

    def _get_golden_dtype(self, data_type):
        if data_type == "float16":
            return "float32"
        if data_type == "float32":
            return "float64"
        if data_type == "bfloat16":
            return "float32"
        if data_type == "complex32":
            return "complex64"
        if data_type == "complex64":
            return "complex128"
        return data_type

    def __str__(self):
        return "%s_%s_%f_%f_%f" % (
            self.standard_name, self.data_type, self.rtol, self.small_value, self.small_value_atol)

    @classmethod
    def build_by_yml_info(cls, data_type: str, **kwargs):
        rtol = kwargs.get("rtol", None)
        if rtol is None:
            rtol = {"float16": 1e-3, "bfloat16": 4e-3, "float32": 1e-6, "float": 1e-6}[data_type]
        small_value = kwargs.get("small_value", None)
        if small_value is None:
            small_value = {"float16": 1e-3, "bfloat16": 1e-3, "float32": 1e-6, "float": 1e-6}[data_type]
        small_value_atol = kwargs.get("small_value_atol", None)
        if small_value_atol is None:
            small_value_atol = {"float16": 1e-5, "bfloat16": 1e-5, "float32": 1e-9, "float": 1e-9}[data_type]
        ulp_limit = kwargs.get("ulp_limit", None)
        if ulp_limit is None:
            ulp_limit = {"float16": 1, "bfloat16": 1, "float32": 32, "float": 32}[data_type]
        golden_dtype = kwargs.get("golden_dtype", None)
        return AbsoluteThreshStandard(data_type, rtol, small_value, small_value_atol, ulp_limit, golden_dtype)


class AbsoluteThreshStandard2(PrecisionCheckStandard):
    NAME = "AbsoluteThreshStandard2"
    SHEET_NAME = "绝对阈值2"

    def __init__(self, data_type: str, rtol: float, small_value: float, small_value_atol: float,  total_atol: float,
                 ulp_limit: int, golden_dtype: Union[str, None] = None):
        """

        :param rtol: relative tolerance
        :param small_value: less than small value not use relative tolerance, use small value absolute tolerance
        :param small_value_atol: small value absolute tolerance
        """
        super().__init__("AbsoluteThreshStandard2", data_type, AbsoluteThreshStandard2.SHEET_NAME)
        self.rtol = rtol
        self.small_value = small_value
        self.small_value_atol = small_value_atol
        self.total_atol = total_atol
        self.ulp_limit = ulp_limit

        if golden_dtype is None:
            self.golden_dtype = self._get_golden_dtype(data_type)
        else:
            self.golden_dtype = golden_dtype

    def _get_golden_dtype(self, data_type):
        if data_type == "float16":
            return "float32"
        if data_type == "float32":
            return "float64"
        if data_type == "bfloat16":
            return "float32"
        if data_type == "complex32":
            return "complex64"
        if data_type == "complex64":
            return "complex128"
        return data_type

    def __str__(self):
        return "%s_%s_%f_%f_%f" % (
            self.standard_name, self.data_type, self.rtol, self.small_value, self.small_value_atol)

    @classmethod
    def build_by_yml_info(cls, data_type: str, **kwargs):
        rtol = kwargs.get("rtol", None)
        if rtol is None:
            rtol = {"float16": 1e-3, "bfloat16": 4e-3, "float32": 1e-6, "float": 1e-6}[data_type]
        small_value = kwargs.get("small_value", None)
        if small_value is None:
            small_value = {"float16": 1e-3, "bfloat16": 1e-3, "float32": 1e-6, "float": 1e-6}[data_type]
        small_value_atol = kwargs.get("small_value_atol", None)
        if small_value_atol is None:
            small_value_atol = {"float16": 1e-5, "bfloat16": 1e-5, "float32": 1e-9, "float": 1e-9}[data_type]
        total_atol = kwargs.get("total_atol", None)
        if total_atol is None:
            total_atol = {"float16": 1e-3, "bfloat16": 1e-3, "float32": 1e-4, "float": 1e-4}[data_type]
        ulp_limit = kwargs.get("ulp_limit", None)
        if ulp_limit is None:
            ulp_limit = {"float16": 1, "bfloat16": 1, "float32": 32, "float": 32}[data_type]
        golden_dtype = kwargs.get("golden_dtype", None)
        return AbsoluteThreshStandard2(data_type, rtol, small_value, small_value_atol, total_atol, ulp_limit, golden_dtype)


class BenchmarkCompareStandard(PrecisionCheckStandard):
    NAME = "BenchmarkCompareStandard"
    SHEET_NAME = "标杆对比"

    def __init__(self, data_type: str,
                 max_re_rtol: float,
                 avg_re_rtol: float,
                 rmse_rtol: float,
                 small_ae_rtol: float,
                 ulp_rtol: float,
                 ulp_atol: float,
                 ulp_limit: int,
                 ae_rtol: float,
                 small_value: float,
                 small_value_atol: Union[float, None] = None,
                 golden_dtype: Union[str, None] = None
                 ):
        super().__init__("BenchmarkCompareStandard", data_type, BenchmarkCompareStandard.SHEET_NAME)
        # self.rtol = rtol
        # self.atol = atol
        self.max_re_rtol = max_re_rtol
        self.avg_re_rtol = avg_re_rtol
        self.rmse_rtol = rmse_rtol
        self.small_ae_rtol = small_ae_rtol
        self.ulp_rtol = ulp_rtol
        self.ulp_atol = ulp_atol
        self.ulp_limit = ulp_limit
        self.ae_rtol = ae_rtol
        self.small_value = small_value
        self.small_value_atol = small_value_atol

        if golden_dtype is None:
            self.golden_dtype = self._get_golden_dtype(data_type)
        else:
            self.golden_dtype = golden_dtype

    def _get_golden_dtype(self, data_type):
        if data_type == "float16":
            return "float32"
        if data_type == "float32":
            return "float64"
        if data_type == "bfloat16":
            return "float32"
        if data_type == "complex32":
            return "complex64"
        if data_type == "complex64":
            return "complex128"
        return data_type

    # def __str__(self):
    #     return "%s_%s_%f_%f" % (
    #         self.standard_name, self.data_type, self.rtol, self.atol)

    @classmethod
    def build_by_yml_info(cls, data_type: str, **kwargs):
        max_re_rtol = kwargs.get("max_re_rtol", 10.0)
        avg_re_rtol = kwargs.get("avg_re_rtol", 2.0)
        rmse_rtol = kwargs.get("rmse_rtol", 2.0)
        small_ae_rtol = kwargs.get("small_ae_rtol", 2.0)
        ulp_rtol = kwargs.get("ulp_rtol", 2.0)
        ulp_atol = kwargs.get("ulp_atol")
        ulp_limit = kwargs.get("ulp_limit")
        ae_rtol = kwargs.get("ae_rtol", 2.0)
        small_value = kwargs.get("small_value", None)
        small_value_atol = kwargs.get("small_value_atol", None)
        if small_value is None:
            small_value = {"float16": 1e-3, "bfloat16": 1e-3, "float32": 1e-6, "float": 1e-6}[data_type]
        if ulp_atol is None:
            ulp_atol = {"float16": 1e-3, "bfloat16": 1e-3, "float32": 5e-2, "float": 5e-2}[data_type]
        if ulp_limit is None:
            ulp_limit = {"float16": 1, "bfloat16": 1, "float32": 32, "float": 32}[data_type]
        if small_value_atol is None:
            small_value_atol = {"float16": 1e-5, "bfloat16": 1e-5, "float32": 1e-9, "float": 1e-9}[data_type]
        return BenchmarkCompareStandard(data_type, max_re_rtol, avg_re_rtol, rmse_rtol, small_ae_rtol, ulp_rtol,
                                        ulp_atol, ulp_limit, ae_rtol, small_value, small_value_atol)


class ULPBenchmarkCompareStandard(PrecisionCheckStandard):
    NAME = "ULPBenchmarkCompareStandard"
    SHEET_NAME = "ULP标杆对比"

    def __init__(self, data_type: str,
                 max_re_rtol: float,
                 avg_re_rtol: float,
                 rmse_rtol: float,
                 small_ae_rtol: float,
                 ulp_rtol: float,
                 ulp_atol: float,
                 ulp_limit: int,
                 ae_rtol: float,
                 small_value: float,
                 small_value_atol: Union[float, None] = None,
                 golden_dtype: Union[str, None] = None
                 ):
        super().__init__("ULPBenchmarkCompareStandard", data_type, ULPBenchmarkCompareStandard.SHEET_NAME)
        # self.rtol = rtol
        # self.atol = atol
        self.max_re_rtol = max_re_rtol
        self.avg_re_rtol = avg_re_rtol
        self.rmse_rtol = rmse_rtol
        self.small_ae_rtol = small_ae_rtol
        self.ulp_rtol = ulp_rtol
        self.ulp_atol = ulp_atol
        self.ulp_limit = ulp_limit
        self.ae_rtol = ae_rtol
        self.small_value = small_value
        self.small_value_atol = small_value_atol

        if golden_dtype is None:
            self.golden_dtype = self._get_golden_dtype(data_type)
        else:
            self.golden_dtype = golden_dtype

    def _get_golden_dtype(self, data_type):
        if data_type == "float16":
            return "float32"
        if data_type == "float32":
            return "float64"
        if data_type == "bfloat16":
            return "float32"
        if data_type == "complex32":
            return "complex64"
        if data_type == "complex64":
            return "complex128"
        return data_type

    @classmethod
    def build_by_yml_info(cls, data_type: str, **kwargs):
        max_re_rtol = kwargs.get("max_re_rtol", 10.0)
        avg_re_rtol = kwargs.get("avg_re_rtol", 2.0)
        rmse_rtol = kwargs.get("rmse_rtol", 2.0)
        small_ae_rtol = kwargs.get("small_ae_rtol", 2.0)
        ulp_rtol = kwargs.get("ulp_rtol", 2.0)
        ulp_atol = kwargs.get("ulp_atol")
        ulp_limit = kwargs.get("ulp_limit")
        ae_rtol = kwargs.get("ae_rtol", 2.0)
        small_value = kwargs.get("small_value", None)
        small_value_atol = kwargs.get("small_value_atol", None)
        if small_value is None:
            small_value = {"float16": 1e-3, "bfloat16": 1e-3, "float32": 1e-6, "float": 1e-6}[data_type]
        if ulp_atol is None:
            ulp_atol = {"float16": 1e-3, "bfloat16": 1e-3, "float32": 5e-2, "float": 5e-2}[data_type]
        if ulp_limit is None:
            ulp_limit = {"float16": 1, "bfloat16": 1, "float32": 32, "float": 32}[data_type]
        if small_value_atol is None:
            small_value_atol = {"float16": 1e-5, "bfloat16": 1e-5, "float32": 1e-9, "float": 1e-9}[data_type]
        return ULPBenchmarkCompareStandard(data_type, max_re_rtol, avg_re_rtol, rmse_rtol, small_ae_rtol, ulp_rtol,
                                           ulp_atol, ulp_limit, ae_rtol, small_value, small_value_atol)


class GPUCompareStandard(PrecisionCheckStandard):
    NAME = "GPUCompareStandard"
    SHEET_NAME = "GPU对比"

    def __init__(self, data_type: str,
                 rtol: float,
                 small_value: float,
                 small_value_atol: float):
        super().__init__("GPUCompareStandard", data_type, GPUCompareStandard.SHEET_NAME)
        self.rtol = rtol
        self.small_value = small_value
        self.small_value_atol = small_value_atol

    def __str__(self):
        return "%s_%s_%f_%f_%f" % (
            self.standard_name, self.data_type, self.rtol, self.small_value, self.small_value_atol)

    @classmethod
    def build_by_yml_info(cls, data_type: str, **kwargs):
        rtol = kwargs.get("rtol", None)
        if rtol is None:
            rtol = 10
        small_value = kwargs.get("small_value", None)
        if small_value is None:
            small_value = {"float32": 0.000001, "float16": 0.001, "bfloat16": 0.004, "float": 0.000001}[data_type]
        small_value_atol = kwargs.get("small_value_atol", None)
        if small_value_atol is None:
            small_value_atol = {"float32": 0.000001, "float16": 0.001, "bfloat16": 0.004, "float": 0.000001}[data_type]
        return GPUCompareStandard(data_type, rtol, small_value, small_value_atol)


class PrecisionCheckResult:
    SUCCESS = 0
    WARNING = 1
    ERROR = 2

    def __init__(self, case_info: TestCaseInfo, precision_check_standard: PrecisionCheckStandard):
        self.case_info: TestCaseInfo = case_info
        self.precision_check_standard = precision_check_standard
        self.check_result = PrecisionCheckResult.SUCCESS
        self.run_success = True
        self.run_err_msg = ""

    def run_err(self, err_msg):
        self.run_success = False
        self.run_err_msg = err_msg

    def get_result_msg(self):
        return ""

    def get_check_result_str(self):
        if self.check_result == PrecisionCheckResult.SUCCESS:
            return "PASS"
        if self.check_result == PrecisionCheckResult.WARNING:
            return "WARNING"
        if self.check_result == PrecisionCheckResult.ERROR:
            return "ERROR"
        return "Unknown"


class BinaryCompareCheckResult(PrecisionCheckResult):
    def __init__(self, case_info: TestCaseInfo, precision_check_standard: BinaryCompareStandard):
        super().__init__(case_info, precision_check_standard)
        self.total_cnt = 0
        self.err_cnt = 0
        self.err_ratio = 0
        self.err_ele_list: List[List] = []  # (idx, value, golden_value)

    def do_summary(self, total_cnt: Union[int, np.ndarray], err_cnt: Union[int, np.ndarray], err_eles: List):
        self.total_cnt = total_cnt
        self.err_cnt = err_cnt
        self.err_ratio = 0 if total_cnt == 0 else self.err_cnt / self.total_cnt
        self.err_ele_list = err_eles
        if err_cnt > 0:
            self.check_result = PrecisionCheckResult.ERROR
        else:
            self.check_result = PrecisionCheckResult.SUCCESS

    def get_result_msg(self):
        if self.check_result == PrecisionCheckResult.SUCCESS:
            return ""
        err_ele_str_list = ["%d: %s    vs    %s" % (x[0], x[1], x[2]) for x in self.err_ele_list]
        err_ele_str = "\n".join(err_ele_str_list)
        return "Error element info(first 10): \n%s" % err_ele_str


class AbsoluteThreshCheckResult(PrecisionCheckResult):
    def __init__(self, case_info: TestCaseInfo, precision_check_standard: AbsoluteThreshStandard):
        super().__init__(case_info, precision_check_standard)
        self.total_cnt: int = 0
        self.total_err_cnt: int = 0
        self.total_pass_ratio: float = 0
        self.total_err_ratio: float = 0

        self.finite_cnt = 0
        self.finite_err_cnt = 0
        self.finite_err_ratio = 0
        self.finite_err_ele_list: List[List] = []

        self.small_value_cnt: int = 0
        self.small_value_err_cnt: int = 0
        self.small_value_err_ratio: float = 0
        self.small_value_err_ele_list: List[List] = []

        self.inf_nan_cnt: int = 0
        self.inf_nan_err_cnt: int = 0
        self.inf_nan_err_ratio: float = 0
        self.inf_nan_err_ele_list: List[List] = []

        self.small_value_check_result = PrecisionCheckResult.SUCCESS
        self.finite_check_result = PrecisionCheckResult.SUCCESS
        self.inf_nan_check_result = PrecisionCheckResult.SUCCESS
        self.golden_avg = 0
        self.stat_info: Union[None, PrecisionStatInfo] = None

    def update_small_value_err_cnt(self, err_cnt: Union[int, np.ndarray]):
        if err_cnt > 0:
            self.small_value_check_result = max(PrecisionCheckResult.ERROR, self.small_value_check_result)
        self.small_value_err_cnt += err_cnt

    def update_inf_nan_err_cnt(self, err_cnt: Union[int, np.ndarray]):
        if err_cnt > 0:
            self.inf_nan_check_result = max(PrecisionCheckResult.ERROR, self.inf_nan_check_result)
        self.inf_nan_err_cnt += err_cnt

    def update_finite_err_cnt(self, err_cnt: Union[int, np.ndarray]):
        if err_cnt > 0:
            self.finite_check_result = max(PrecisionCheckResult.ERROR, self.finite_check_result)
        self.finite_err_cnt += err_cnt

    def add_small_value_err_ele_list(self, err_list: List[List]):
        self.small_value_err_ele_list.extend(err_list)
        self.small_value_err_cnt += len(err_list)
        self.small_value_check_result = max(PrecisionCheckResult.ERROR, self.small_value_check_result)

    def add_finite_err_ele_list(self, err_list: List[List]):
        self.finite_err_ele_list.extend(err_list)
        self.finite_err_cnt += len(err_list)
        self.finite_check_result = max(PrecisionCheckResult.ERROR, self.finite_check_result)

    def add_inf_nan_err_ele_list(self, err_list: List[List]):
        self.inf_nan_err_ele_list.extend(err_list)
        self.inf_nan_err_cnt += len(err_list)
        self.inf_nan_check_result = max(PrecisionCheckResult.ERROR, self.inf_nan_check_result)

    def do_summary(self, total_cnt: Union[int, None] = None):
        if total_cnt is not None:
            self.total_cnt = total_cnt
        self.small_value_err_ratio = 0 if self.small_value_cnt == 0 else self.small_value_err_cnt / self.small_value_cnt
        self.finite_err_ratio = 0 if self.finite_cnt == 0 else self.finite_err_cnt / self.finite_cnt
        self.inf_nan_err_ratio = 0 if self.inf_nan_cnt == 0 else self.inf_nan_err_cnt / self.inf_nan_cnt
        self.total_err_cnt = self.finite_err_cnt + self.inf_nan_err_cnt + self.small_value_err_cnt
        self.total_err_ratio = self.total_err_cnt / self.total_cnt
        self.total_pass_ratio = 1 - self.total_err_ratio
        self.check_result = max(self.finite_check_result, self.inf_nan_check_result, self.small_value_check_result)


class AbsoluteThreshCheckResult2(PrecisionCheckResult):
    def __init__(self, case_info: TestCaseInfo, precision_check_standard: AbsoluteThreshStandard2):
        super().__init__(case_info, precision_check_standard)
        self.total_atol = precision_check_standard.total_atol
        self.total_cnt: int = 0
        self.total_err_cnt: int = 0
        self.total_pass_ratio: float = 0
        self.total_err_ratio: float = 0

        self.finite_cnt = 0
        self.finite_err_cnt = 0
        self.finite_err_ratio = 0
        self.finite_err_ele_list: List[List] = []

        self.small_value_cnt: int = 0
        self.small_value_err_cnt: int = 0
        self.small_value_err_ratio: float = 0
        self.small_value_err_ele_list: List[List] = []

        self.inf_nan_cnt: int = 0
        self.inf_nan_err_cnt: int = 0
        self.inf_nan_err_ratio: float = 0
        self.inf_nan_err_ele_list: List[List] = []

        self.small_value_check_result = PrecisionCheckResult.SUCCESS
        self.finite_check_result = PrecisionCheckResult.SUCCESS
        self.inf_nan_check_result = PrecisionCheckResult.SUCCESS
        self.golden_avg = 0
        self.stat_info: Union[None, PrecisionStatInfo] = None

        self.dtype = precision_check_standard.data_type
        self.small_value_err_str = ""
        self.inf_nan_err_str = ""
        self.finite_err_str = ""

    def update_small_value_err_cnt(self, err_cnt: Union[int, np.ndarray], err_eles: List):
        if err_cnt > 0:
            self.small_value_check_result = max(PrecisionCheckResult.ERROR, self.small_value_check_result)
            self.small_value_err_ = str(err_eles)
        self.small_value_err_cnt += err_cnt

    def update_inf_nan_err_cnt(self, err_cnt: Union[int, np.ndarray], err_eles: List):
        if err_cnt > 0:
            self.inf_nan_check_result = max(PrecisionCheckResult.ERROR, self.inf_nan_check_result)
            self.inf_nan_err_str = str(err_eles)
        self.inf_nan_err_cnt += err_cnt

    def update_finite_err_cnt(self, err_cnt: Union[int, np.ndarray], err_eles: List):
        if err_cnt > 0:
            self.finite_check_result = max(PrecisionCheckResult.ERROR, self.finite_check_result)
            self.finite_err_str = str(err_eles)
        self.finite_err_cnt += err_cnt

    def add_small_value_err_ele_list(self, err_list: List[List]):
        self.small_value_err_ele_list.extend(err_list)
        self.small_value_err_cnt += len(err_list)
        self.small_value_check_result = max(PrecisionCheckResult.ERROR, self.small_value_check_result)

    def add_finite_err_ele_list(self, err_list: List[List]):
        self.finite_err_ele_list.extend(err_list)
        self.finite_err_cnt += len(err_list)
        self.finite_check_result = max(PrecisionCheckResult.ERROR, self.finite_check_result)

    def add_inf_nan_err_ele_list(self, err_list: List[List]):
        self.inf_nan_err_ele_list.extend(err_list)
        self.inf_nan_err_cnt += len(err_list)
        self.inf_nan_check_result = max(PrecisionCheckResult.ERROR, self.inf_nan_check_result)

    def do_summary(self, total_cnt: Union[int, None] = None):
        if total_cnt is not None:
            self.total_cnt = total_cnt
        self.small_value_err_ratio = 0 if self.small_value_cnt == 0 else self.small_value_err_cnt / self.small_value_cnt
        self.finite_err_ratio = 0 if self.finite_cnt == 0 else self.finite_err_cnt / self.finite_cnt
        self.inf_nan_err_ratio = 0 if self.inf_nan_cnt == 0 else self.inf_nan_err_cnt / self.inf_nan_cnt
        self.total_err_cnt = self.finite_err_cnt + self.inf_nan_err_cnt + self.small_value_err_cnt
        self.total_err_ratio = self.total_err_cnt / self.total_cnt
        self.total_pass_ratio = 1 - self.total_err_ratio
        if self.total_atol < self.total_err_ratio:
            self.check_result = PrecisionCheckResult.ERROR
        else:
            self.check_result = PrecisionCheckResult.SUCCESS

    def get_result_msg(self):
        result_str = ""
        if self.check_result == PrecisionCheckResult.SUCCESS:
            return result_str
        if self.total_atol < self.total_err_ratio:
            result_str += "逐值比较ERROR: 标准误差比例：%f < 用例误差比例：%f\n" % (self.total_atol, self.total_err_ratio)
            if self.small_value_err_str:
                result_str += "small_value_err_str， %s\n" % self.small_value_err_str
            if self.inf_nan_err_str:
                result_str += "inf_nan_err_str， %s\n" % self.inf_nan_err_str
            if self.finite_err_str:
                result_str += "finite_err_str， %s\n" % self.finite_err_str
        return result_str

    def printd(self):
        OP_LOGGER.error(f"self.total_cnt is {self.total_cnt}")
        OP_LOGGER.error(f"self.total_err_cnt is {self.total_err_cnt}")
        OP_LOGGER.error(f"self.total_pass_ratio is {self.total_pass_ratio}")
        OP_LOGGER.error(f"self.total_err_ratio is {self.total_err_ratio}")
        OP_LOGGER.error(f"self.finite_cnt is {self.finite_cnt}")
        OP_LOGGER.error(f"self.finite_err_cnt is {self.finite_err_cnt}")
        OP_LOGGER.error(f"self.finite_err_ratio is {self.finite_err_ratio}")
        OP_LOGGER.error(f"self.finite_err_ele_list{self.finite_err_ele_list}")
        OP_LOGGER.error(f"self.small_value_cnt is {self.small_value_cnt}")
        OP_LOGGER.error(f"self.small_value_err_cnt is {self.small_value_err_cnt}")
        OP_LOGGER.error(f"self.small_value_err_ratio is {self.small_value_err_ratio}")
        OP_LOGGER.error(f"self.small_value_err_ele_list is {self.small_value_err_ele_list}")
        OP_LOGGER.error(f"self.inf_nan_cnt is {self.inf_nan_cnt}")
        OP_LOGGER.error(f"self.inf_nan_err_cnt is {self.inf_nan_err_cnt}")
        OP_LOGGER.error(f"self.inf_nan_err_ratio is {self.inf_nan_err_ratio}")
        OP_LOGGER.error(f"self.inf_nan_err_ele_list is {self.inf_nan_err_ele_list}")
        OP_LOGGER.error(f"self.small_value_check_result is {self.small_value_check_result}")
        OP_LOGGER.error(f"self.finite_check_result is {self.finite_check_result}")
        OP_LOGGER.error(f"self.inf_nan_check_result is {self.inf_nan_check_result}")
        OP_LOGGER.error(f"self.golden_avg is {self.golden_avg}")
        OP_LOGGER.error(f"self.stat_info is {self.stat_info}")
        OP_LOGGER.error(f"self.dtype is {self.dtype}")


class BenchmarkCompareCheckResult(PrecisionCheckResult):
    def __init__(self, case_info: TestCaseInfo, precision_check_standard: Union[BenchmarkCompareStandard, ULPBenchmarkCompareStandard]):
        super().__init__(case_info, precision_check_standard)
        self.case_info = case_info
        self.precision_check_standard = precision_check_standard
        self.golden_avg = 0
        self.npu_stat_info: Union[None, PrecisionStatInfo] = None
        self.benchmark_stat_info: Union[None, PrecisionStatInfo] = None
        self.total_cnt = 0
        self.inf_nan_cnt = 0
        self.inf_nan_err_cnt = 0
        self.inf_nan_err_ele = []
        self.inf_nan_warn_cnt = 0
        self.inf_nan_warn_ele = []
        self.max_re_ratio = 1
        self.eb_difference = 0
        self.avg_re_ratio = 1
        self.rmse_ratio = 1
        self.small_ae_ratio = 1
        self.max_ulp_ratio = 1
        self.min_ulp_ratio = 1
        self.avg_ulp_ratio = 1
        self.max_abs_ulp_ratio = 1
        self.avg_abs_ulp_ratio = 1
        self.ulp_err_ratio = 1
        self.ae_err_cnt = 0
        self.ae_warn_cnt = 0
        self.ae_err_ele = []
        self.ae_warn_ele = []
        self.ae_err_ele_str = ""
        self.ae_warn_ele_str = ""
        self.gpu_precision_check = True

    def update_inf_check_result(self, inf_nan_total_cnt: Union[int, np.ndarray],
                                err_cnt: Union[int, np.ndarray], warn_cnt: Union[int, np.ndarray],
                                err_ele_list: List, warn_ele_list: List):
        self.inf_nan_cnt = inf_nan_total_cnt
        self.inf_nan_err_cnt = err_cnt
        self.inf_nan_warn_cnt = warn_cnt
        self.inf_nan_err_ele = err_ele_list[:min(len(err_ele_list), 10)]
        self.inf_nan_warn_ele = warn_ele_list[:min(len(warn_ele_list), 10)]

    def update_check_result(self, err_cnt: Union[int, np.ndarray], warn_cnt: Union[int, np.ndarray],
                            err_ele_list: List, warn_ele_list: List):
        self.ae_err_cnt = self.inf_nan_err_cnt + err_cnt
        self.ae_warn_cnt = self.inf_nan_warn_cnt + warn_cnt
        self.ae_err_ele = self.inf_nan_err_ele + err_ele_list[:min(len(err_ele_list), 10)]
        self.ae_err_ele_str = "(%s: %s    vs    %s     vs      %s)\n" % ("idx", "npu", "golden", "gpu") + \
                              "\n".join(["%d: %s    vs    %s     vs      %s" % (x[0], x[1], x[2], x[3]) for x in self.ae_err_ele])
        self.ae_warn_ele = self.inf_nan_warn_ele + warn_ele_list[:min(len(warn_ele_list), 10)]
        self.ae_warn_ele_str = "(%s: %s    vs    %s     vs      %s)\n" % ("idx", "npu", "golden", "gpu") + \
                               "\n".join(["%d: %s    vs    %s     vs      %s" % (x[0], x[1], x[2], x[3]) for x in self.ae_warn_ele])

    def _calc_ratio(self, x: float, y: float, default_value: float = 1.0):
        if y == 0:
            if x == 0:
                return 1.0
            else:
                return default_value
        else:
            return x / y

    def do_summary(self):
        npu_max_re = self.npu_stat_info.re_distribute_info.max_re
        benchmark_max_re = self.benchmark_stat_info.re_distribute_info.max_re
        self.max_re_ratio = self._calc_ratio(npu_max_re, benchmark_max_re, 10000.0)
        npu_eb = self.npu_stat_info.eb_info.eb
        benchmark_eb = self.benchmark_stat_info.eb_info.eb
        self.eb_difference = npu_eb - benchmark_eb
        npu_avg_re = self.npu_stat_info.re_distribute_info.avg_re
        benchmark_avg_re = self.benchmark_stat_info.re_distribute_info.avg_re
        self.avg_re_ratio = self._calc_ratio(npu_avg_re, benchmark_avg_re, 10000.0)
        self.rmse_ratio = self._calc_ratio(self.npu_stat_info.rmse, self.benchmark_stat_info.rmse, 10000.0)
        npu_ulp = self.npu_stat_info.ulp_distribute_info
        benchmark_ulp = self.benchmark_stat_info.ulp_distribute_info
        self.max_ulp_ratio = self._calc_ratio(npu_ulp.max_ulp, benchmark_ulp.max_ulp, 10000.0)
        self.min_ulp_ratio = self._calc_ratio(npu_ulp.min_ulp, benchmark_ulp.min_ulp, 10000.0)
        self.avg_ulp_ratio = self._calc_ratio(npu_ulp.avg_ulp, benchmark_ulp.avg_ulp, 10000.0)
        self.max_abs_ulp_ratio = self._calc_ratio(max(abs(npu_ulp.max_ulp), abs(npu_ulp.min_ulp)),
                                                  max(abs(benchmark_ulp.max_ulp), abs(benchmark_ulp.min_ulp)), 10000.0)
        self.avg_abs_ulp_ratio = self._calc_ratio(npu_ulp.avg_abs_ulp, benchmark_ulp.avg_abs_ulp, 10000.0)
        self.ae_pass_ratio = self._calc_ratio(self.total_cnt - self.ae_err_cnt, self.total_cnt)
        self.ulp_err_ratio = self._calc_ratio(npu_ulp.abs_ulp_err_cnt, benchmark_ulp.abs_ulp_err_cnt, 10000.0)
        npu_small_value = self.npu_stat_info.small_value_info
        benchmark_small_value = self.benchmark_stat_info.small_value_info
        self.small_ae_ratio = self._calc_ratio(npu_small_value.small_value_err_count,
                                               benchmark_small_value.small_value_err_count, 10000.0)
        if self.benchmark_stat_info.re_distribute_info.histogram_stat_ratio[-1] > 0.9 \
                or (benchmark_small_value.small_value_ratio > 0.7 and benchmark_small_value.small_value_err_ratio > 0.9):
            self.gpu_precision_check = False

        if not self.gpu_precision_check:
            self.check_result = PrecisionCheckResult.ERROR
        elif self.max_re_ratio > self.precision_check_standard.max_re_rtol \
                or self.avg_re_ratio > self.precision_check_standard.avg_re_rtol \
                or self.rmse_ratio > self.precision_check_standard.rmse_rtol\
                or self.small_ae_ratio > self.precision_check_standard.small_ae_rtol:
            self.check_result = PrecisionCheckResult.ERROR
        elif self.max_re_ratio > 1 or self.avg_re_ratio > 1 or self.rmse_ratio > 1 or self.small_ae_ratio > 1:
            self.check_result = PrecisionCheckResult.WARNING
        else:
            self.check_result = PrecisionCheckResult.SUCCESS

    def get_result_msg(self):
        result_str = ""
        if self.check_result == PrecisionCheckResult.SUCCESS:
            return result_str
        if not self.gpu_precision_check:
            result_str = "ERROR: GPU输出数据存在问题，请校验GPU实现"
            return result_str
        if self.max_re_ratio > self.precision_check_standard.max_re_rtol:
            result_str += "ERROR: 最大相对误差差距超过阈值， %s\n" % self.max_re_ratio
        elif self.max_re_ratio > 1.0:
            result_str += "Warning: 最大相对误差差距超过阈值， %s\n" % self.max_re_ratio
        if self.avg_re_ratio > self.precision_check_standard.avg_re_rtol:
            result_str += "ERROR: 相对误差均值差距超过阈值， %s\n" % self.avg_re_ratio
        elif self.avg_re_ratio > 1.0:
            result_str += "Warning: 相对误差均值差距超过阈值， %s\n" % self.avg_re_ratio
        if self.rmse_ratio > self.precision_check_standard.rmse_rtol:
            result_str += "ERROR: 均方根误差差距超过阈值， %s\n" % self.rmse_ratio
        elif self.rmse_ratio > 1.0:
            result_str += "Warning: 均方根误差差距超过阈值， %s\n" % self.rmse_ratio
        if self.small_ae_ratio > self.precision_check_standard.small_ae_rtol:
            result_str += "ERROR: 小值域绝对误差差距超过阈值， %s\n" % self.small_ae_ratio
        elif self.small_ae_ratio > 1.0:
            result_str += "Warning: 小值域绝对误差差距超过阈值， %s\n" % self.small_ae_ratio
        # if self.ae_err_cnt > 0:
        #     result_str += "ERROR: 逐点绝对误差超过阈值， %s\n" % self.ae_err_ele_str
        # elif self.ae_warn_cnt > 0:
        #     result_str += "Warning: 逐点绝对误差超过阈值， %s\n" % self.ae_warn_ele_str
        return result_str


class ULPBenchmarkCompareCheckResult(BenchmarkCompareCheckResult):
    def __init__(self, case_info: TestCaseInfo, precision_check_standard: ULPBenchmarkCompareStandard):
        super().__init__(case_info, precision_check_standard)
        self.npu_ulp_err_ratio = 0

    def do_summary(self):
        npu_max_re = self.npu_stat_info.re_distribute_info.max_re
        benchmark_max_re = self.benchmark_stat_info.re_distribute_info.max_re
        self.max_re_ratio = self._calc_ratio(npu_max_re, benchmark_max_re, 10000.0)
        npu_eb = self.npu_stat_info.eb_info.eb
        benchmark_eb = self.benchmark_stat_info.eb_info.eb
        self.eb_difference = npu_eb - benchmark_eb
        npu_avg_re = self.npu_stat_info.re_distribute_info.avg_re
        benchmark_avg_re = self.benchmark_stat_info.re_distribute_info.avg_re
        self.avg_re_ratio = self._calc_ratio(npu_avg_re, benchmark_avg_re, 10000.0)
        self.rmse_ratio = self._calc_ratio(self.npu_stat_info.rmse, self.benchmark_stat_info.rmse, 10000.0)
        npu_ulp = self.npu_stat_info.ulp_distribute_info
        benchmark_ulp = self.benchmark_stat_info.ulp_distribute_info
        self.max_ulp_ratio = self._calc_ratio(npu_ulp.max_ulp, benchmark_ulp.max_ulp, 10000.0)
        self.min_ulp_ratio = self._calc_ratio(npu_ulp.min_ulp, benchmark_ulp.min_ulp, 10000.0)
        self.avg_ulp_ratio = self._calc_ratio(npu_ulp.avg_ulp, benchmark_ulp.avg_ulp, 10000.0)
        self.max_abs_ulp_ratio = self._calc_ratio(max(abs(npu_ulp.max_ulp), abs(npu_ulp.min_ulp)),
                                                  max(abs(benchmark_ulp.max_ulp), abs(benchmark_ulp.min_ulp)), 10000.0)
        self.avg_abs_ulp_ratio = self._calc_ratio(npu_ulp.avg_abs_ulp, benchmark_ulp.avg_abs_ulp, 10000.0)
        self.ae_pass_ratio = self._calc_ratio(self.total_cnt - self.ae_err_cnt, self.total_cnt)
        self.ulp_err_ratio = self._calc_ratio(npu_ulp.abs_ulp_err_cnt, benchmark_ulp.abs_ulp_err_cnt, 10000.0)
        self.npu_ulp_err_ratio = self._calc_ratio(npu_ulp.abs_ulp_err_cnt, self.total_cnt, 0.0)
        npu_small_value = self.npu_stat_info.small_value_info
        benchmark_small_value = self.benchmark_stat_info.small_value_info
        self.small_ae_ratio = self._calc_ratio(npu_small_value.small_value_err_count,
                                               benchmark_small_value.small_value_err_count, 10000.0)
        if self.benchmark_stat_info.re_distribute_info.histogram_stat_ratio[-1] > 0.9 \
                or (benchmark_small_value.small_value_ratio > 0.7 and benchmark_small_value.small_value_err_ratio > 0.9):
            self.gpu_precision_check = False

        if not self.gpu_precision_check:
            self.check_result = PrecisionCheckResult.ERROR
        elif self.npu_ulp_err_ratio <= self.precision_check_standard.ulp_atol \
                or (self.case_info.op_type == "flot32" and npu_ulp.avg_abs_ulp <= 64):
            self.check_result = PrecisionCheckResult.SUCCESS
        else:
            if self.ulp_err_ratio > self.precision_check_standard.ulp_rtol:
                self.check_result = PrecisionCheckResult.ERROR
            elif self.ulp_err_ratio > 1:
                self.check_result = PrecisionCheckResult.WARNING
            else:
                self.check_result = PrecisionCheckResult.SUCCESS

    def get_result_msg(self):
        result_str = ""
        if self.check_result == PrecisionCheckResult.SUCCESS:
            return result_str
        if not self.gpu_precision_check:
            result_str = "ERROR: GPU输出数据存在问题，请校验GPU实现"
            return result_str
        # if self.npu_ulp_err_ratio > self.precision_check_standard.ulp_atol:
        #     result_str += "ERROR: NPU_ULP绝对值ERROR数据超过阈值， %s\n" % self.npu_ulp_err_ratio
        if self.ulp_err_ratio > self.precision_check_standard.ulp_rtol:
            result_str += "ERROR: ULP绝对值ERROR数据超过阈值， %s\n" % self.ulp_err_ratio
        elif self.ulp_err_ratio > 1.0:
            result_str += "Warning: ULP绝对值ERROR的数据超过阈值， %s\n" % self.ulp_err_ratio
        return result_str


class GPUCompareCheckResult(PrecisionCheckResult):
    def __init__(self, case_info: TestCaseInfo, precision_check_standard: GPUCompareStandard):
        super().__init__(case_info, precision_check_standard)
        self.npu_stat_info: Union[None, PrecisionStatInfo] = None
        self.gpu_stat_info: Union[None, PrecisionStatInfo] = None
        self.golden_avg: float = 0

        self.total_cnt: int = 0
        self.total_pass_cnt: int = 0
        self.total_pass_ratio: float = 0
        self.total_err_cnt: int = 0
        self.total_err_ratio: float = 0
        self.total_warn_cnt: int = 0
        self.total_warn_ratio: float = 0

        self.finite_cnt: int = 0
        self.finite_warn_cnt: int = 0
        self.finite_warn_ratio: float = 0
        self.finite_warn_ele_list: List = []  # (idx, npu_value, golden_value, gpu_value)
        self.finite_err_cnt: int = 0
        self.finite_err_ratio: float = 0
        self.finite_err_ele_list: List = []  # (idx, npu_value, golden_value, gpu_value)

        self.small_value_cnt: int = 0
        self.small_value_warn_cnt: int = 0
        self.small_value_warn_ratio: float = 0
        self.small_value_warn_ele_list: List = []
        self.small_value_err_cnt: int = 0
        self.small_value_err_ratio: float = 0
        self.small_value_err_ele_list: List = []  # (idx, npu_value, golden_value, gpu_value)

        self.inf_nan_cnt: int = 0
        self.inf_nan_err_cnt: int = 0
        self.inf_nan_err_ratio: float = 0
        self.inf_nan_err_ele_list = []  # (idx, npu_value, golden_value, gpu_value)
        self.inf_nan_warn_cnt: int = 0
        self.inf_nan_warn_ratio: float = 0
        self.inf_nan_warn_ele_list = []

        self.finite_check_result = PrecisionCheckResult.SUCCESS
        self.small_value_check_result = PrecisionCheckResult.SUCCESS
        self.inf_nan_check_result = PrecisionCheckResult.SUCCESS
        self.total_result = PrecisionCheckResult.SUCCESS

    def _get_finite_check_msg(self):
        if self.finite_check_result == PrecisionCheckResult.SUCCESS:
            return ""
        res_msg = "Normal value check info:\n"
        if self.finite_err_cnt > 0:
            err_ele_str_list = ["%d: %s    vs    %s    vs    %s" % (x[0], x[1], x[2], x[3]) for x in
                                self.finite_err_ele_list]
            err_ele_str = "\n".join(err_ele_str_list)
            err_str = "Error element info(first 10): \n%s" % err_ele_str
            res_msg += err_str
        if self.finite_warn_cnt > 0:
            warn_ele_str_list = ["%d: %s    vs    %s    vs    %s" % (x[0], x[1], x[2], x[3]) for x in
                                 self.finite_warn_ele_list]
            warn_ele_str = "\n".join(warn_ele_str_list)
            warn_str = "Error element info(first 10): \n%s" % warn_ele_str
            res_msg += warn_str
        return res_msg

    def _get_small_value_check_msg(self):
        if self.small_value_check_result == PrecisionCheckResult.SUCCESS:
            return ""
        res_msg = "Small value check info:\n"
        if self.small_value_err_cnt > 0:
            err_ele_str_list = ["%d: %s    vs    %s    vs    %s" % (x[0], x[1], x[2], x[3]) for x in
                                self.small_value_err_ele_list]
            err_ele_str = "\n".join(err_ele_str_list)
            err_str = "Error element info(first 10): \n%s" % err_ele_str
            res_msg += err_str
        if self.finite_warn_cnt > 0:
            warn_ele_str_list = ["%d: %s    vs    %s    vs    %s" % (x[0], x[1], x[2], x[3]) for x in
                                 self.small_value_warn_ele_list]
            warn_ele_str = "\n".join(warn_ele_str_list)
            warn_str = "Error element info(first 10): \n%s" % warn_ele_str
            res_msg += warn_str
        return res_msg

    def _get_inf_nan_check_msg(self):
        if self.inf_nan_check_result == PrecisionCheckResult.SUCCESS:
            return ""
        res_msg = "Inf nan check info:\n"
        if self.inf_nan_err_cnt > 0:
            err_ele_str_list = ["%d: %s    vs    %s    vs    %s" % (x[0], x[1], x[2], x[3]) for x in
                                self.inf_nan_err_ele_list]
            err_ele_str = "\n".join(err_ele_str_list)
            err_str = "Error element info(first 10): \n%s" % err_ele_str
            res_msg += err_str
        if self.finite_warn_cnt > 0:
            warn_ele_str_list = ["%d: %s    vs    %s    vs    %s" % (x[0], x[1], x[2], x[3]) for x in
                                 self.inf_nan_warn_ele_list]
            warn_ele_str = "\n".join(warn_ele_str_list)
            warn_str = "Error element info(first 10): \n%s" % warn_ele_str
            res_msg += warn_str
        return res_msg

    def get_result_msg(self):
        if self.check_result == PrecisionCheckResult.SUCCESS:
            return ""
        res_msg = ""
        res_msg += self._get_finite_check_msg()
        res_msg += "\n"
        res_msg += self._get_small_value_check_msg()
        res_msg += "\n"
        res_msg += self._get_inf_nan_check_msg()
        return res_msg

    def init_stat_cnt(self, total_cnt, finite_cnt: int, small_value_cnt: int, inf_nan_cnt: int):
        self.total_cnt = total_cnt
        self.small_value_cnt = small_value_cnt
        self.inf_nan_cnt = inf_nan_cnt

    def add_finite_err_ele_list(self, err_list: List[List]):
        self.finite_err_ele_list.extend(err_list)
        self.finite_err_cnt += len(err_list)
        self.finite_check_result = max(PrecisionCheckResult.ERROR, self.finite_check_result)

    def add_finite_warn_ele_list(self, warn_list: List[List]):
        self.finite_warn_ele_list.extend(warn_list)
        self.finite_warn_cnt += len(warn_list)
        self.finite_check_result = max(PrecisionCheckResult.WARNING, self.finite_check_result)

    def add_small_value_err_ele_list(self, err_list: List[List]):
        self.small_value_err_ele_list.extend(err_list)
        self.small_value_err_cnt += len(err_list)
        self.small_value_check_result = max(PrecisionCheckResult.ERROR, self.small_value_check_result)

    def add_small_value_warn_ele_list(self, warn_list: List[List]):
        self.small_value_warn_ele_list.extend(warn_list)
        self.small_value_warn_cnt += len(warn_list)
        self.small_value_check_result = max(PrecisionCheckResult.WARNING, self.small_value_check_result)

    def add_inf_nan_err_ele_list(self, err_list: List[List]):
        self.inf_nan_err_ele_list.extend(err_list)
        self.inf_nan_err_cnt += len(err_list)
        self.inf_nan_check_result = max(PrecisionCheckResult.ERROR, self.inf_nan_check_result)

    def add_inf_nan_err_ele(self, err_ele: List):
        self.inf_nan_err_ele_list.append(err_ele)
        self.inf_nan_err_cnt += 1
        self.inf_nan_check_result = max(PrecisionCheckResult.ERROR, self.inf_nan_check_result)

    def add_inf_nan_warn_ele_list(self, warn_list: List[List]):
        self.inf_nan_warn_ele_list.extend(warn_list)
        self.inf_nan_warn_cnt += len(warn_list)
        self.inf_nan_check_result = max(PrecisionCheckResult.WARNING, self.inf_nan_check_result)

    def add_inf_nan_warn_ele(self, warn_ele: List):
        self.inf_nan_warn_ele_list.append(warn_ele)
        self.inf_nan_warn_cnt += 1
        self.inf_nan_check_result = max(PrecisionCheckResult.WARNING, self.inf_nan_check_result)

    def do_summary_finite(self, finite_cnt: int, finite_pass_cnt: Union[int, np.ndarray],
                          finite_err_cnt: Union[int, np.ndarray], finite_warn_cnt: Union[int, np.ndarray],
                          finite_err_ele_list: List[List] = None, finite_warn_ele_list: List[List] = None):
        self.finite_cnt = finite_cnt
        self.finite_warn_cnt = finite_warn_cnt
        self.finite_err_cnt = finite_err_cnt
        if finite_err_ele_list is not None:
            self.finite_err_ele_list.extend(finite_err_ele_list)
        if finite_warn_ele_list is not None:
            self.finite_warn_ele_list.extend(finite_warn_ele_list)

    def update_finite_summary_cnt(self, warn_cnt: int, err_cnt: int):
        self.finite_warn_cnt += warn_cnt
        self.finite_err_cnt += err_cnt
        if warn_cnt > 0:
            self.finite_check_result = max(PrecisionCheckResult.WARNING, self.finite_check_result)
        if err_cnt > 0:
            self.finite_check_result = max(PrecisionCheckResult.WARNING, self.finite_check_result)

    def do_summary(self):
        self.finite_err_ratio = 0 if self.finite_cnt == 0 else self.finite_err_cnt / self.finite_cnt
        self.finite_warn_ratio = 0 if self.finite_cnt == 0 else self.finite_warn_cnt / self.finite_cnt
        self.small_value_err_ratio = 0 if self.small_value_cnt == 0 else self.small_value_err_cnt / self.small_value_cnt
        self.small_value_warn_ratio = 0 if self.small_value_cnt == 0 else self.small_value_warn_cnt / self.small_value_cnt
        self.inf_nan_err_ratio = 0 if self.inf_nan_err_cnt == 0 else self.inf_nan_err_cnt / self.inf_nan_cnt

        self.total_err_cnt = self.finite_err_cnt + self.small_value_err_cnt + self.inf_nan_err_cnt
        self.total_err_ratio = 0 if self.total_err_cnt == 0 else self.total_err_cnt / self.total_cnt
        self.check_result = max(self.finite_check_result, self.inf_nan_check_result, self.small_value_check_result)

        self.total_warn_cnt = self.finite_warn_cnt + self.small_value_warn_cnt
        self.total_warn_ratio = 0 if self.total_err_cnt == 0 else self.total_warn_cnt / self.total_cnt

        self.total_pass_cnt = self.total_cnt - self.total_warn_cnt - self.total_err_cnt
        self.total_pass_ratio = 1 if self.total_err_cnt == 0 else self.total_pass_cnt / self.total_cnt

        if self.total_warn_cnt > 0:
            self.check_result = max(self.check_result, PrecisionCheckResult.WARNING)

        if self.total_err_cnt > 0:
            self.check_result = max(self.check_result, PrecisionCheckResult.ERROR)


class PrecisionStatistician:
    @classmethod
    @func_log_wrapper()
    def _stat_re_distribute(cls,
                            actual_res: np.ndarray,
                            golden_res: np.ndarray,
                            re_info: np.ndarray,
                            bins: List[float],
                            small_value_mask: np.ndarray,
                            inf_nan_mask: np.ndarray) -> REDistributeInfo:
        mask = np.logical_or(small_value_mask, inf_nan_mask)
        masked_re_info = np.where(mask, -1, re_info)
        masked_bins = [-1.0, ] + bins
        dist_stat, dist_stat_bin = np.histogram(masked_re_info, masked_bins)
        stat_count = np.sum(dist_stat[1:])
        masked_count = dist_stat[:1][0]
        if stat_count == 0:
            max_re_idx = 0
            avg_re = 0
            max_re = 0
        else:
            max_re_idx = np.argmax(masked_re_info)
            max_re = re_info[max_re_idx]
            total_count = masked_count + stat_count
            # masked_re_avg = np.average(masked_re_info)
            # avg_re = (masked_re_avg + masked_count / total_count) * (total_count / stat_count)
            stat_re_info = masked_re_info[np.logical_not(mask)]
            avg_re = np.average(stat_re_info)
        return REDistributeInfo(dist_stat[1:],
                                masked_bins[1:],
                                max_re,
                                [actual_res[max_re_idx], golden_res[max_re_idx]],
                                avg_re)

    @classmethod
    @func_log_wrapper()
    def _stat_small_value_info(cls,
                               abs_ae: np.ndarray,
                               small_value_mask: np.ndarray,
                               small_value_thresh: float,
                               small_value_atol: float,
                               total_cnt: int) -> SmallValueInfo:
        small_case_cnt = np.sum(small_value_mask)
        npu_small_err_mask = np.logical_not(np.less_equal(abs_ae, small_value_atol))  # use less to fix nan
        npu_small_err_mask = np.logical_and(npu_small_err_mask, small_value_mask)
        small_case_err_cnt = np.sum(npu_small_err_mask)
        return SmallValueInfo(small_value_thresh, small_case_cnt, small_case_err_cnt, total_cnt)

    @classmethod
    @func_log_wrapper()
    def _stat_rmse_info(cls, abs_ae: np.ndarray, inf_nan_mask: np.ndarray) -> float:
        masked_ae = np.where(inf_nan_mask, 0, abs_ae)
        rmse = np.mean(np.square(masked_ae))
        inf_nan_cnt = np.sum(inf_nan_mask)
        rmse = rmse * (abs_ae.size / (abs_ae.size - inf_nan_cnt + 0.0001) + 0.0001)
        rmse = np.sqrt(rmse)
        return rmse

    @classmethod
    @func_log_wrapper()
    def _stat_eb_info(cls, actual: np.ndarray, golden: np.ndarray) -> EBInfo:
        golden_round = golden.astype(actual.dtype)
        smaller_cnt = np.sum(np.less(actual, golden_round))
        larger_cnt = np.sum(np.greater(actual, golden_round))
        equal_cnt = np.sum(np.equal(actual, golden_round))

        return EBInfo(equal_cnt, larger_cnt, smaller_cnt, golden.size)

    @classmethod
    def _get_ulp_params(cls, np_data_type):
        if np_data_type == "float32":
            return -126, 23
        if np_data_type == "bfloat16":
            return -126, 7
        return -14, 10

    @classmethod
    @func_log_wrapper()
    def _stat_ulp_info(cls,
                       actual: np.ndarray,
                       ae: np.ndarray,
                       abs_golden: np.ndarray,
                       finite_mask: np.ndarray,
                       golden: np.ndarray,
                       actual_dtype: str,
                       ulp_limit: int):
        eb = np.floor(np.log2(abs_golden, dtype=np.float64))
        eb = np.where(np.equal(abs_golden, 0), 0, eb)
        min_eb, exponent_num = cls._get_ulp_params(actual_dtype)
        eb = np.maximum(eb, min_eb)
        eb = np.where(finite_mask, eb, 0)

        # err_mask_idx = np.nonzero(np.logical_not(np.less_equal(eb, 150)))
        # for idx in err_mask_idx[0][:10]:
        #     print(actual[idx], ae[idx], abs_golden[idx])

        ulp_err = ae.astype(np.float64) * np.exp2(exponent_num - eb, dtype=np.float64)
        # set not finite_mask to nan
        ulp_err_fix_to_nan = np.where(finite_mask, ulp_err, np.nan)
        # print("max ulp: %s, min ulp: %s" % (np.max(ulp_err_fix_to_nan), np.min(ulp_err_fix_to_nan)))
        # err_mask_idx = np.nonzero(np.greater(ulp_err_fix_to_nan, 10))
        #
        # for idx in err_mask_idx[0][:10]:
        #     print(actual[idx], ae[idx], abs_golden[idx])

        bins = REHistogramBins.get_ulp_histogram_bin(ulp_limit)
        histogram_stat, histogram_bins = np.histogram(ulp_err_fix_to_nan, bins)
        total_cnt = np.sum(histogram_stat)
        if total_cnt > 0:
            ulp_err_fix_to_neg_inf = np.where(finite_mask, ulp_err, -np.inf)
            max_ulp = np.max(ulp_err_fix_to_neg_inf)
            max_idx = np.argmax(ulp_err_fix_to_neg_inf)
            ulp_err_fix_to_pos_inf = np.where(finite_mask, ulp_err, np.inf)
            min_ulp = np.min(ulp_err_fix_to_pos_inf)
            min_idx = np.argmin(ulp_err_fix_to_pos_inf)
            ulp_err_fix_to_0 = np.where(finite_mask, ulp_err, 0)
            ulp_err_cnt = np.sum(np.abs(ulp_err_fix_to_0) > ulp_limit)
            avg_ulp = np.average(ulp_err_fix_to_0) * ((actual.size / total_cnt) if total_cnt > 0 else 0)
            avg_abs_ulp = np.average(np.abs(ulp_err_fix_to_0)) * ((actual.size / total_cnt) if total_cnt > 0 else 0)
            return ULPDistributeInfo(histogram_bins, histogram_stat, total_cnt, ulp_err_cnt, max_ulp, min_ulp, avg_ulp,
                                     avg_abs_ulp, max_value_info=[actual[max_idx], golden[max_idx]],
                                     min_val_info=[actual[min_idx], golden[min_idx]])
        else:
            return ULPDistributeInfo(histogram_bins, histogram_stat, 0, 0, 0, 0, 0, 0, [0, 0], [0, 0])

    @classmethod
    @func_log_wrapper()
    def stat_precision_info(cls,
                            golden: np.ndarray,
                            actual: np.ndarray, *,
                            abs_golden: np.ndarray = None,
                            abs_golden_with_eps: np.ndarray = None,
                            small_value_mask: np.ndarray = None,
                            small_value_thresh: Union[float, None] = None,
                            small_value_atol: Union[float, None] = None,
                            inf_nan_mask: np.ndarray = None,
                            abs_ae: np.ndarray = None,
                            re: np.ndarray = None,
                            actual_dtype: str = None,
                            ulp_limit: int = 1) -> PrecisionStatInfo:
        OP_LOGGER.info("[PrecisionStatistician] start stat_precision_info")
        shape_size = golden.size
        # actual_dtype = actual.dtype
        golden = golden.reshape((shape_size,))
        if abs_golden is None:
            abs_golden = np.abs(golden)
        actual = actual.reshape((shape_size,))
        if inf_nan_mask is None:
            golden_inf_mask = np.isinf(golden)
            inf_mask = np.isinf(actual)
            golden_nan_mask = np.isnan(golden)
            nan_mask = np.isnan(actual)
            golden_inf_nan_mask = np.logical_or(golden_inf_mask, golden_nan_mask)
            inf_nan_mask = np.logical_or(inf_mask, nan_mask)
            inf_nan_mask = np.logical_or(golden_inf_nan_mask, inf_nan_mask)
        # if ae is None:
        #     ae = actual - golden
        if abs_ae is None:
            abs_ae = np.abs(golden - actual)
        # if small_value_thresh is None:
        #     small_value_thresh = SmallValueThresh.get_small_value_thresh(actual_dtype)
        if small_value_mask is None:
            small_value_mask = np.less_equal(abs_golden, small_value_thresh)  # nan<little ==> false
        if abs_golden_with_eps is None:
            abs_golden_with_eps = abs_golden + np.finfo(golden.dtype).eps
        if re is None:
            re = abs_ae / abs_golden_with_eps
        re_stat_info = cls._stat_re_distribute(actual, golden, re,
                                               REHistogramBins.get_histogram_bins(actual_dtype),
                                               small_value_mask, inf_nan_mask)
        small_case_info = cls._stat_small_value_info(abs_ae, small_value_mask, small_value_thresh,
                                                     small_value_atol, shape_size)
        rmse = cls._stat_rmse_info(abs_ae, np.logical_or(inf_nan_mask, small_value_mask))
        eb_info = cls._stat_eb_info(actual, golden)
        ae = actual - golden
        ulp_distribute_info = cls._stat_ulp_info(actual, ae, abs_golden, finite_mask=np.logical_not(inf_nan_mask),
                                                 golden=golden, actual_dtype=actual_dtype, ulp_limit=ulp_limit)
        res = PrecisionStatInfo(re_stat_info, small_case_info, eb_info, rmse, ulp_distribute_info)
        OP_LOGGER.info("[PrecisionStatistician] end stat_precision_info")
        return res


class PrecisionChecker:
    def __init__(self):
        self.standard_checker_map = {}

    def registry(self, standard_cls):
        def _out_wrapper(func):
            self.standard_checker_map[standard_cls.NAME] = func

            @wraps(func)
            def _in_wrapper(*kargs, **kwargs):
                return func(*kargs, **kwargs)

            return _in_wrapper

        return _out_wrapper

    def check(self, standard: Union[PrecisionCheckStandard, AbsoluteThreshStandard, AbsoluteThreshStandard2,
                                    BenchmarkCompareStandard, ULPBenchmarkCompareStandard, GPUCompareStandard,
                                    BinaryCompareCheckResult],
              case_info: TestCaseInfo,
              npu: np.ndarray, golden: np.ndarray,
              **kwargs) -> Union[PrecisionCheckResult, AbsoluteThreshCheckResult, AbsoluteThreshCheckResult2,
                                 BenchmarkCompareCheckResult, ULPBenchmarkCompareStandard, BinaryCompareCheckResult,
                                 GPUCompareCheckResult]:
        if standard.standard_name not in self.standard_checker_map.keys():
            raise RuntimeError("This standard has not check function, standard name: %s" % (standard.standard_name,))
        return self.standard_checker_map[standard.standard_name](standard, case_info, npu, golden, **kwargs)


PRECISION_CHECKER = PrecisionChecker()


class PrecisionGpuCompareChecker:

    @classmethod
    def _check_small_value(cls,
                           result: GPUCompareCheckResult,
                           standard: GPUCompareStandard,
                           abs_npu_ae: np.ndarray,
                           abs_gpu_ae: np.ndarray,
                           small_value_mask: np.ndarray
                           ):
        small_case_cnt = np.sum(small_value_mask)
        thresh_check_mask = np.less_equal(abs_npu_ae, standard.small_value_atol)  # abs_npu_ae <= small_value_atol
        gpu_compare_check_mask = np.less_equal(abs_npu_ae, standard.rtol * abs_gpu_ae)  # abs_npu_ae <= rtol*abs_gpu_ae
        pass_mask = np.logical_or(thresh_check_mask, gpu_compare_check_mask)
        small_case_pass_mask = np.logical_and(small_value_mask, pass_mask)

        result.small_value_cnt = small_case_cnt
        result.small_value_pass_cnt = np.sum(small_case_pass_mask)
        result.small_value_err_cnt = result.small_value_cnt - result.small_value_pass_cnt
        result.small_value_warn_cnt = 0

    @classmethod
    def _check_npu_gpu_value(cls, npu_value, gpu_value, golden_value, dtype, rtol=2.0):
        success_flag = 0
        warning_flag = 1
        error_flag = 2

        def _clip_value(tmp_value):
            if np.isfinite(tmp_value):
                return tmp_value
            if np.isposinf(tmp_value):
                return np.finfo(dtype).max if dtype != bfloat16 else 3.3895313892515355e+38
            if np.isneginf(tmp_value):
                return np.finfo(dtype).min if dtype != bfloat16 else -3.3895313892515355e+38
            return 0

        def _check_value(npu_tmp, gpu_tmp, gold_tmp):
            npu_tmp = _clip_value(npu_tmp)
            gpu_tmp = _clip_value(gpu_tmp)
            gold_tmp = _clip_value(gold_tmp)
            npu_tmp = float(npu_tmp)
            gpu_tmp = float(gpu_tmp)
            gold_tmp = float(gold_tmp)
            npu_ae_tmp = abs(gold_tmp - npu_tmp)
            gpu_ae_tmp = rtol * abs(gold_tmp - gpu_tmp)
            if npu_ae_tmp <= gpu_ae_tmp:
                return success_flag
            elif npu_ae_tmp <= rtol * gpu_ae_tmp:
                return warning_flag
            else:
                return error_flag

        npu_pos_inf = np.isposinf(npu_value)
        gpu_pos_inf = np.isposinf(gpu_value)
        gold_pos_inf = np.isposinf(golden_value)

        npu_neg_inf = np.isneginf(npu_value)
        gpu_neg_inf = np.isneginf(gpu_value)
        gold_neg_inf = np.isneginf(golden_value)

        npu_nan = np.isnan(npu_value)
        gpu_nan = np.isnan(gpu_value)
        gold_nan = np.isnan(golden_value)

        npu_finite = np.isfinite(npu_value)
        # gpu_finite = np.isfinite(gpu_value)
        gold_finite = np.isfinite(golden_value)

        if (npu_pos_inf and gpu_pos_inf) or (npu_pos_inf and gold_pos_inf) or \
                (npu_neg_inf and gpu_neg_inf) or (npu_neg_inf and gold_neg_inf) or \
                (npu_nan and gpu_nan) or (npu_nan and gold_nan):
            return 0
        if npu_pos_inf and gold_finite:
            return _check_value(npu_value, gpu_value, golden_value)
        if npu_pos_inf and gold_neg_inf:
            return error_flag
        if npu_pos_inf and gold_nan:
            return error_flag

        if npu_neg_inf and gold_finite:
            return _check_value(npu_value, gpu_value, golden_value)
        if npu_neg_inf and gold_pos_inf:
            return error_flag
        if npu_neg_inf and gold_nan:
            return error_flag

        if npu_finite and gold_finite:
            return _check_value(npu_value, gpu_value, golden_value)

        return error_flag

    @classmethod
    def _check_small_size_inf_nan(cls,
                                  result: GPUCompareCheckResult,
                                  standard: GPUCompareStandard,
                                  golden: np.ndarray,
                                  npu_res: np.ndarray,
                                  gpu_res: np.ndarray,
                                  golden_with_actual_dtype: np.ndarray,
                                  inf_or_nan_mask: np.ndarray):
        actual_dtype = npu_res.dtype
        inf_or_nan_indices = np.nonzero(inf_or_nan_mask)
        for idx in inf_or_nan_indices[0]:
            check_res = cls._check_npu_gpu_value(npu_res[idx], golden_with_actual_dtype[idx], gpu_res[idx],
                                                 actual_dtype, standard.rtol)
            if check_res == PrecisionCheckResult.SUCCESS:
                continue
            if check_res == PrecisionCheckResult.WARNING:
                result.add_inf_nan_warn_ele([idx, npu_res[idx], golden[idx], gpu_res[idx]])
            if check_res == PrecisionCheckResult.ERROR:
                result.add_inf_nan_err_ele([idx, npu_res[idx], golden[idx], gpu_res[idx]])

    @classmethod
    def _check_large_size_inf_nan(cls,
                                  result: GPUCompareCheckResult,
                                  standard: GPUCompareStandard,
                                  golden: np.ndarray,
                                  npu_res: np.ndarray,
                                  gpu_res: np.ndarray,
                                  golden_with_actual_dtype: np.ndarray,
                                  inf_or_nan_mask: np.ndarray):
        actual_dtype = npu_res.dtype
        npu_golden_equal = np.equal(npu_res, golden_with_actual_dtype)
        npu_gpu_equal = np.equal(npu_res, gpu_res)
        npu_pass = np.logical_or(npu_gpu_equal, npu_golden_equal)
        npu_nan_mask = np.isnan(npu_res)
        gpu_nan_mask = np.isnan(gpu_res)
        golden_nan_mask = np.isnan(golden_with_actual_dtype)
        npu_golden_nan = np.logical_and(npu_nan_mask, golden_nan_mask)
        npu_gpu_nan = np.logical_and(npu_nan_mask, gpu_nan_mask)
        npu_pass = np.logical_or(npu_pass, npu_golden_nan)
        npu_pass = np.logical_or(npu_pass, npu_gpu_nan)
        inf_or_nan_warning_mask = np.logical_xor(npu_pass, inf_or_nan_mask)
        inf_or_nan_warning_mask = np.logical_and(inf_or_nan_mask, inf_or_nan_warning_mask)
        inf_or_nan_indices = np.nonzero(inf_or_nan_warning_mask)

        for idx in inf_or_nan_indices[0]:
            check_res = cls._check_npu_gpu_value(npu_res[idx], golden_with_actual_dtype[idx], gpu_res[idx],
                                                 actual_dtype, standard.rtol)
            if check_res == PrecisionCheckResult.SUCCESS:
                continue
            if check_res == PrecisionCheckResult.WARNING:
                result.add_inf_nan_warn_ele([idx, npu_res[idx], golden[idx], gpu_res[idx]])
            if check_res == PrecisionCheckResult.ERROR:
                result.add_inf_nan_warn_ele([idx, npu_res[idx], golden[idx], gpu_res[idx]])

    @classmethod
    def _check_inf_nan_value(cls, result: GPUCompareCheckResult,
                             standard: GPUCompareStandard,
                             golden: np.ndarray, npu_res: np.ndarray, gpu_res: np.ndarray,
                             golden_with_actual_dtype: np.ndarray):
        npu_finite_mask = np.isfinite(npu_res)
        gpu_finite_mask = np.isfinite(gpu_res)
        golden_finite_mask = np.isfinite(golden_with_actual_dtype)

        all_finite_mask = np.logical_and(npu_finite_mask, gpu_finite_mask)
        all_finite_mask = np.logical_and(golden_finite_mask, all_finite_mask)

        inf_or_nan_mask = np.logical_not(all_finite_mask)
        inf_or_nan_cnt = np.sum(inf_or_nan_mask)
        if inf_or_nan_cnt < 100 or inf_or_nan_cnt < golden.size / 10:
            cls._check_small_size_inf_nan(result, standard, golden, npu_res, gpu_res, golden_with_actual_dtype,
                                          inf_or_nan_mask)
        else:
            cls._check_large_size_inf_nan(result, standard, golden, npu_res, gpu_res, golden_with_actual_dtype,
                                          inf_or_nan_mask)

    @classmethod
    def _get_warn_err_info(cls, golden: np.ndarray,
                           npu_res: np.ndarray,
                           gpu_res: np.ndarray,
                           mask: np.ndarray):
        warn_idx = np.nonzero(mask)[0]
        warn_npu_value = list(np.take_along_axis(npu_res, warn_idx, axis=-1))
        warn_gold_value = list(np.take_along_axis(golden, warn_idx, axis=-1))
        warn_gpu_value = list(np.take_along_axis(gpu_res, warn_idx, axis=-1))
        warn_info = [[idx, npu, gold, gpu] for idx, npu, gold, gpu in
                     zip(list(warn_idx), warn_npu_value, warn_gold_value, warn_gpu_value)]
        return warn_info

    @classmethod
    def _check_general_elements(cls,
                                result: GPUCompareCheckResult, standard: GPUCompareStandard,
                                golden: np.ndarray, npu_res: np.ndarray, gpu_res: np.ndarray, *,
                                abs_golden: np.ndarray,
                                abs_golden_with_eps: np.ndarray,
                                abs_npu_ae: np.ndarray,
                                abs_gpu_ae: np.ndarray,
                                finite_mask: np.ndarray,
                                npu_re: np.ndarray,
                                gpu_re: np.ndarray):
        if abs_golden is None:
            abs_golden = np.abs(golden)
        if abs_golden_with_eps is None:
            eps = np.finfo(golden.dtype).eps if golden.dtype != bfloat16 else 2 ** -8
            abs_golden_with_eps = abs_golden + eps
        if npu_re is None:
            npu_re = abs_npu_ae / abs_golden_with_eps
        if gpu_re is None:
            gpu_re = abs_gpu_ae / abs_golden_with_eps
        pass_mask = np.less_equal(npu_re, gpu_re)
        pass_mask = np.logical_and(finite_mask, pass_mask)
        pass_cnt = np.sum(pass_mask)
        two_multi_gpu_re = standard.rtol * gpu_re
        warning_mask = np.less_equal(npu_re, two_multi_gpu_re)
        warning_mask = np.logical_and(finite_mask, warning_mask)
        warning_cnt = np.sum(warning_mask)
        warning_cnt = warning_cnt - pass_cnt
        err_mask = np.greater(npu_re, two_multi_gpu_re)
        err_mask = np.logical_and(finite_mask, err_mask)
        err_cnt = np.sum(err_mask)
        total_finite_cnt = pass_cnt + warning_cnt + err_cnt
        xor_warn_pass = np.logical_xor(warning_mask, pass_mask)
        warning_mask = np.logical_and(xor_warn_pass, warning_mask)
        result.do_summary_finite(total_finite_cnt, pass_cnt, err_cnt, warning_cnt,
                                 finite_warn_ele_list=cls._get_warn_err_info(golden, npu_res, gpu_res, warning_mask),
                                 finite_err_ele_list=cls._get_warn_err_info(golden, npu_res, gpu_res, err_mask))
        result.golden_avg = 0 if total_finite_cnt == 0 else np.average(np.where(finite_mask, golden, 0)) * (
                golden.size / total_finite_cnt)

    @classmethod
    def compare_and_check(cls, golden: np.ndarray, npu_res: np.ndarray, gpu_res: np.ndarray, *,
                          case_info: TestCaseInfo = None,
                          standard: GPUCompareStandard = None,
                          abs_golden: np.ndarray = None,
                          abs_golden_with_eps: np.ndarray = None,
                          golden_with_actual_dtype: np.ndarray = None,
                          finite_mask: np.ndarray = None,
                          small_value_mask: np.ndarray = None,
                          inf_or_nan_mask: np.ndarray = None,
                          abs_npu_ae: np.ndarray = None,
                          abs_gpu_ae: np.ndarray = None,
                          npu_re: np.ndarray = None,
                          gpu_re: np.ndarray = None
                          ) -> GPUCompareCheckResult:
        checked_result = GPUCompareCheckResult(case_info, standard)

        actual_dtype = npu_res.dtype
        if golden_with_actual_dtype is None:
            golden_with_actual_dtype = golden.astype(actual_dtype)

        if inf_or_nan_mask is None:
            npu_finite_mask = np.isfinite(npu_res)
            gpu_finite_mask = np.isfinite(gpu_res)
            golden_finite_mask = np.isfinite(golden_with_actual_dtype)
            all_finite_mask = np.logical_and(npu_finite_mask, gpu_finite_mask)
            all_finite_mask = np.logical_and(golden_finite_mask, all_finite_mask)
            inf_or_nan_mask = np.logical_not(all_finite_mask)

        small_value_thresh = standard.small_value
        if small_value_mask is None:
            small_value_mask = np.less_equal(golden, small_value_thresh)
            small_value_mask = np.logical_and(np.logical_not(inf_or_nan_mask), small_value_mask)

        if finite_mask is None:
            finite_mask = np.logical_not(np.logical_or(inf_or_nan_mask, small_value_mask))

        if abs_npu_ae is None:
            abs_npu_ae = np.abs(golden - npu_res)
        if abs_gpu_ae is None:
            abs_gpu_ae = np.abs(golden - gpu_res)

        checked_result.init_stat_cnt(golden.size,
                                     finite_cnt=int(np.sum(finite_mask)),
                                     small_value_cnt=int(np.sum(small_value_mask)),
                                     inf_nan_cnt=int(np.sum(inf_or_nan_mask)))

        cls._check_inf_nan_value(checked_result, standard, golden, npu_res, gpu_res, golden_with_actual_dtype)
        cls._check_small_value(checked_result, standard, abs_npu_ae, abs_gpu_ae, small_value_mask)
        cls._check_general_elements(checked_result, standard, golden, npu_res, gpu_res,
                                    abs_golden=abs_golden,
                                    abs_golden_with_eps=abs_golden_with_eps,
                                    abs_npu_ae=abs_npu_ae,
                                    abs_gpu_ae=abs_gpu_ae,
                                    finite_mask=finite_mask,
                                    npu_re=npu_re,
                                    gpu_re=gpu_re)
        checked_result.do_summary()
        return checked_result


@PRECISION_CHECKER.registry(GPUCompareStandard)
def check_with_gpu_compare_standard(standard: GPUCompareStandard,
                                    case_info: TestCaseInfo,
                                    npu_res: np.ndarray,
                                    golden: np.ndarray,
                                    gpu_res: np.ndarray) -> GPUCompareCheckResult:
    shape_size: object = golden.size
    actual_dtype = npu_res.dtype
    golden = golden.reshape((shape_size,))
    abs_golden = np.abs(golden)
    npu_res = npu_res.reshape((shape_size,))
    gpu_res = gpu_res.reshape((shape_size,))
    eps = np.finfo(golden.dtype).eps if golden.dtype != bfloat16 else 2 ** -8
    abs_golen_with_eps = abs_golden + eps
    golden_with_actual_dtype = golden.astype(actual_dtype)
    npu_finite_mask = np.isfinite(npu_res)
    gpu_finite_mask = np.isfinite(gpu_res)
    golden_finite_mask = np.isfinite(golden_with_actual_dtype)
    all_finite_mask = np.logical_and(npu_finite_mask, gpu_finite_mask)
    all_finite_mask = np.logical_and(golden_finite_mask, all_finite_mask)
    inf_or_nan_mask = np.logical_not(all_finite_mask)

    small_value_thresh = standard.small_value
    small_value_mask = np.less_equal(golden, small_value_thresh)
    small_value_mask = np.logical_and(all_finite_mask, small_value_mask)

    finite_mask = np.logical_and(all_finite_mask, np.logical_not(small_value_mask))

    npu_ae = npu_res - golden
    abs_npu_ae = np.abs(npu_ae)
    gpu_ae = gpu_res - golden
    abs_gpu_ae = np.abs(gpu_ae)
    npu_re = abs_npu_ae / abs_golen_with_eps
    gpu_re = abs_gpu_ae / abs_golen_with_eps

    npu_stat_info = PrecisionStatistician.stat_precision_info(golden, npu_res,
                                                              abs_golden=abs_golden,
                                                              abs_golden_with_eps=abs_golen_with_eps,
                                                              small_value_mask=small_value_mask,
                                                              inf_nan_mask=inf_or_nan_mask,
                                                              # ae=npu_ae,
                                                              small_value_thresh=standard.small_value,
                                                              small_value_atol=standard.small_value_atol,
                                                              abs_ae=abs_npu_ae,
                                                              re=npu_re,
                                                              actual_dtype=standard.data_type)
    gpu_stat_info = PrecisionStatistician.stat_precision_info(golden, gpu_res,
                                                              abs_golden=abs_golden,
                                                              abs_golden_with_eps=abs_golen_with_eps,
                                                              small_value_mask=small_value_mask,
                                                              inf_nan_mask=inf_or_nan_mask,
                                                              # ae=gpu_ae,
                                                              small_value_thresh=standard.small_value,
                                                              small_value_atol=standard.small_value_atol,
                                                              abs_ae=abs_gpu_ae,
                                                              re=gpu_re,
                                                              actual_dtype=standard.data_type)

    result = PrecisionGpuCompareChecker.compare_and_check(golden, npu_res, gpu_res,
                                                          standard=standard,
                                                          case_info=case_info,
                                                          abs_golden=abs_golden,
                                                          abs_golden_with_eps=abs_golen_with_eps,
                                                          golden_with_actual_dtype=golden_with_actual_dtype,
                                                          finite_mask=finite_mask,
                                                          small_value_mask=small_value_mask,
                                                          inf_or_nan_mask=inf_or_nan_mask,
                                                          abs_npu_ae=abs_npu_ae,
                                                          abs_gpu_ae=abs_gpu_ae,
                                                          npu_re=npu_re,
                                                          gpu_re=gpu_re)
    result.npu_stat_info = npu_stat_info
    result.gpu_stat_info = gpu_stat_info
    return result


class Utils:
    @classmethod
    def np_dtype_to_str(cls, data_type):
        if data_type == np.float32:
            return "float32"
        if data_type == np.float16:
            return "float16"
        if data_type == np.complex64:
            return "complex64"
        if data_type == np.complex128:
            return "complex128"
        return "%s" % data_type


class BenchmarkCompareChecker:
    @classmethod
    def _check_params(cls,
                      npu_res: np.ndarray,
                      golden: np.ndarray,
                      benchmark_res: np.ndarray):
        if npu_res.dtype != benchmark_res.dtype:
            return False, "data type of npu and benchmark is not same. npu: %s, benchmark: %s" % (
                Utils.np_dtype_to_str(npu_res.dtype),
                Utils.np_dtype_to_str(benchmark_res.dtype))
        if npu_res.size != golden.size or npu_res.size != benchmark_res.size:
            return False, "size of npu, golden, benchmark is not same: %d, %d, %d" % (npu_res.size,
                                                                                      golden.size,
                                                                                      benchmark_res.size)
        return True, ""

    @classmethod
    def _check_npu_gpu_value(cls, npu_value, golden_value, gpu_value, dtype, rtol=2.0, atol=1e-6):

        def _clip_value(tmp_value):
            if np.isfinite(tmp_value):
                return tmp_value
            if np.isposinf(tmp_value):
                return np.finfo(dtype).max if dtype != bfloat16 else 3.3895313892515355e+38
            if np.isneginf(tmp_value):
                return np.finfo(dtype).min if dtype != bfloat16 else -3.3895313892515355e+38
            return 0

        def _check_value(npu_tmp, gpu_tmp, gold_tmp):
            npu_tmp = _clip_value(npu_tmp)
            gpu_tmp = _clip_value(gpu_tmp)
            gold_tmp = _clip_value(gold_tmp)
            # npu_tmp = float(npu_tmp)
            # gpu_tmp = float(gpu_tmp)
            # gold_tmp = float(gold_tmp)
            npu_ae_tmp = abs(gold_tmp - npu_tmp)
            gpu_ae_tmp = rtol * abs(gold_tmp - gpu_tmp)
            if npu_ae_tmp <= gpu_ae_tmp or npu_ae_tmp <= atol:
                return PrecisionCheckResult.SUCCESS
            elif npu_ae_tmp <= rtol * gpu_ae_tmp:
                return PrecisionCheckResult.WARNING
            else:
                return PrecisionCheckResult.ERROR

        npu_pos_inf = np.isposinf(npu_value)
        gpu_pos_inf = np.isposinf(gpu_value)
        gold_pos_inf = np.isposinf(golden_value)

        npu_neg_inf = np.isneginf(npu_value)
        gpu_neg_inf = np.isneginf(gpu_value)
        gold_neg_inf = np.isneginf(golden_value)

        npu_nan = np.isnan(npu_value)
        gpu_nan = np.isnan(gpu_value)
        gold_nan = np.isnan(golden_value)

        npu_finite = np.isfinite(npu_value)
        # gpu_finite = np.isfinite(gpu_value)
        gold_finite = np.isfinite(golden_value)
        # NPU和Golden或者gpu的inf，nan一致
        if (npu_pos_inf and gpu_pos_inf) or (npu_pos_inf and gold_pos_inf) or \
                (npu_neg_inf and gpu_neg_inf) or (npu_neg_inf and gold_neg_inf) or \
                (npu_nan and gpu_nan) or (npu_nan and gold_nan):
            return PrecisionCheckResult.SUCCESS
        # NPU是inf，golden是正常值，使用clip的方式，对比误差
        if npu_pos_inf and gold_finite:
            return _check_value(npu_value, gpu_value, golden_value)
        if npu_pos_inf and gold_neg_inf:
            return PrecisionCheckResult.ERROR
        if npu_pos_inf and gold_nan:
            return PrecisionCheckResult.ERROR
        # NPU是-inf，golden是正常值，使用clip的方式，对比误差
        if npu_neg_inf and gold_finite:
            return _check_value(npu_value, gpu_value, golden_value)
        if npu_neg_inf and gold_pos_inf:
            return PrecisionCheckResult.ERROR
        if npu_neg_inf and gold_nan:
            return PrecisionCheckResult.ERROR

        if npu_finite and gold_finite:
            return _check_value(npu_value, gpu_value, golden_value)

        return PrecisionCheckResult.ERROR

    @classmethod
    def _get_warn_err_info(cls, golden: np.ndarray,
                           npu_res: np.ndarray,
                           gpu_res: np.ndarray,
                           mask: np.ndarray):
        warn_idx = np.nonzero(mask)[0]
        warn_npu_value = list(np.take_along_axis(npu_res, warn_idx, axis=-1))
        warn_gold_value = list(np.take_along_axis(golden, warn_idx, axis=-1))
        warn_gpu_value = list(np.take_along_axis(gpu_res, warn_idx, axis=-1))
        warn_info = list(zip(list(warn_idx), warn_npu_value, warn_gold_value, warn_gpu_value))
        return warn_info

    @classmethod
    def _check_all_finite(cls,
                          result: Union[BenchmarkCompareCheckResult, ULPBenchmarkCompareCheckResult],
                          standard: Union[BenchmarkCompareStandard, ULPBenchmarkCompareStandard],
                          abs_npu_ae: np.ndarray,
                          abs_gpu_ae: np.ndarray,
                          golden: np.ndarray,
                          npu_res: np.ndarray,
                          benchmark_res: np.ndarray,
                          all_finite_mask: np.ndarray):
        abs_ae_rtol = np.divide(abs_npu_ae, abs_gpu_ae)
        abs_ae_err = np.where(all_finite_mask, abs_npu_ae > standard.small_value_atol, False)
        warn_mask = np.logical_and(np.logical_and(abs_ae_rtol > 1, abs_ae_rtol <= standard.ae_rtol), abs_ae_err)
        warn_cnt = np.sum(warn_mask)
        warn_ele = cls._get_warn_err_info(golden, npu_res, benchmark_res, warn_mask)
        err_mask = np.logical_and(abs_ae_rtol > standard.ae_rtol, abs_ae_err)
        err_cnt = np.sum(err_mask)
        err_ele = cls._get_warn_err_info(golden, npu_res, benchmark_res, err_mask)
        result.update_check_result(err_cnt, warn_cnt, err_ele, warn_ele)

    @classmethod
    def _check_small_size_inf_nan(cls,
                                  result: Union[BenchmarkCompareCheckResult, ULPBenchmarkCompareCheckResult],
                                  standard: Union[BenchmarkCompareStandard, ULPBenchmarkCompareStandard],
                                  golden: np.ndarray,
                                  npu_res: np.ndarray,
                                  benchmark_res: np.ndarray,
                                  golden_with_actual_dtype: np.ndarray,
                                  inf_or_nan_mask: np.ndarray):
        actual_dtype = npu_res.dtype
        inf_or_nan_indices = np.nonzero(inf_or_nan_mask)
        err_ele = []
        warn_ele = []
        inf_nan_cnt = np.sum(inf_or_nan_mask)
        for idx in inf_or_nan_indices[0]:
            check_res = cls._check_npu_gpu_value(npu_res[idx],
                                                 golden_with_actual_dtype[idx],
                                                 benchmark_res[idx],
                                                 actual_dtype,
                                                 standard.ae_rtol,
                                                 standard.small_value_atol)
            if check_res == PrecisionCheckResult.SUCCESS:
                continue
            if check_res == PrecisionCheckResult.WARNING:
                warn_ele.append((idx, npu_res[idx], golden[idx], benchmark_res[idx]))
            if check_res == PrecisionCheckResult.ERROR:
                err_ele.append((idx, npu_res[idx], golden[idx], benchmark_res[idx]))
        err_cnt = len(err_ele)
        warn_cnt = len(warn_ele)
        result.update_inf_check_result(inf_nan_cnt, err_cnt, warn_cnt, err_ele, warn_ele)

    @classmethod
    def _check_large_size_inf_nan(cls,
                                  result: Union[BenchmarkCompareCheckResult, ULPBenchmarkCompareCheckResult],
                                  standard: Union[BenchmarkCompareStandard, ULPBenchmarkCompareStandard],
                                  golden: np.ndarray,
                                  npu_res: np.ndarray,
                                  benchmark_res: np.ndarray,
                                  golden_with_actual_dtype: np.ndarray,
                                  inf_or_nan_mask: np.ndarray):
        actual_dtype = npu_res.dtype
        npu_golden_equal = np.equal(npu_res, golden_with_actual_dtype)
        npu_gpu_equal = np.equal(npu_res, benchmark_res)
        npu_pass = np.logical_or(npu_gpu_equal, npu_golden_equal)
        npu_nan_mask = np.isnan(npu_res)
        gpu_nan_mask = np.isnan(benchmark_res)
        golden_nan_mask = np.isnan(golden_with_actual_dtype)
        npu_golden_nan = np.logical_and(npu_nan_mask, golden_nan_mask)
        npu_gpu_nan = np.logical_and(npu_nan_mask, gpu_nan_mask)
        npu_pass = np.logical_or(npu_pass, npu_golden_nan)
        npu_pass = np.logical_or(npu_pass, npu_gpu_nan)

        inf_or_nan_need_check_mask = np.logical_xor(npu_pass, inf_or_nan_mask)
        inf_or_nan_need_check_mask = np.logical_and(inf_or_nan_mask, inf_or_nan_need_check_mask)

        inf_or_nan_need_check_indices = np.nonzero(inf_or_nan_need_check_mask)
        err_ele = []
        warn_ele = []
        for idx in inf_or_nan_need_check_indices[0]:
            check_res = cls._check_npu_gpu_value(npu_res[idx], golden_with_actual_dtype[idx], benchmark_res[idx],
                                                 actual_dtype, standard.ae_rtol, standard.small_value_atol)
            if check_res == PrecisionCheckResult.SUCCESS:
                continue
            if check_res == PrecisionCheckResult.WARNING:
                warn_ele.append((idx, npu_res[idx], golden[idx], benchmark_res[idx]))
            if check_res == PrecisionCheckResult.ERROR:
                err_ele.append((idx, npu_res[idx], golden[idx], benchmark_res[idx]))
        err_cnt = len(err_ele)
        warn_cnt = len(warn_ele)
        inf_nan_cnt = np.sum(inf_or_nan_mask)
        result.update_inf_check_result(inf_nan_cnt, err_cnt, warn_cnt, err_ele, warn_ele)

    @classmethod
    @func_log_wrapper()
    def _check_inf_nan_value(cls,
                             result: Union[BenchmarkCompareCheckResult, ULPBenchmarkCompareCheckResult],
                             standard: Union[BenchmarkCompareStandard, ULPBenchmarkCompareStandard],
                             golden: np.ndarray,
                             npu_res: np.ndarray,
                             benchmark_res: np.ndarray,
                             golden_with_actual_dtype: np.ndarray):
        npu_finite_mask = np.isfinite(npu_res)
        benchmark_finite_mask = np.isfinite(benchmark_res)
        golden_finite_mask = np.isfinite(golden_with_actual_dtype)

        all_finite_mask = np.logical_and(npu_finite_mask, benchmark_finite_mask)
        all_finite_mask = np.logical_and(golden_finite_mask, all_finite_mask)

        inf_or_nan_mask = np.logical_not(all_finite_mask)
        inf_or_nan_cnt = np.sum(inf_or_nan_mask)
        if inf_or_nan_cnt < 100 or inf_or_nan_cnt < golden.size / 10:
            cls._check_small_size_inf_nan(result, standard, golden, npu_res, benchmark_res, golden_with_actual_dtype,
                                          inf_or_nan_mask)
        else:
            cls._check_large_size_inf_nan(result, standard, golden, npu_res, benchmark_res, golden_with_actual_dtype,
                                          inf_or_nan_mask)

    @classmethod
    def check(cls,
              case_info: TestCaseInfo,
              standard: Union[BenchmarkCompareStandard, ULPBenchmarkCompareStandard],
              npu_res: np.ndarray,
              golden: np.ndarray,
              benchmark_res: np.ndarray) -> Union[BenchmarkCompareCheckResult, ULPBenchmarkCompareCheckResult]:
        if standard.data_type == "bfloat16":
            npu_res = npu_res.astype(np.uint32) << 16
            npu_res.dtype = np.float32
            benchmark_res = benchmark_res.astype(np.uint32) << 16
            benchmark_res.dtype = np.float32
        result = BenchmarkCompareCheckResult(case_info, standard) if standard.standard_name == "BenchmarkCompareStandard"\
            else ULPBenchmarkCompareCheckResult(case_info, standard)
        check_res, err_msg = cls._check_params(npu_res, golden, benchmark_res)
        if not check_res:
            result.run_err(err_msg)
            return result

        actual_dtype = npu_res.dtype
        abs_golden = np.abs(golden)
        eps = np.finfo(golden.dtype).eps if golden.dtype != bfloat16 else 2 ** -8
        abs_golen_with_eps = abs_golden + eps

        golden_with_actual_dtype = golden.astype(actual_dtype)
        npu_finite_mask = np.isfinite(npu_res)
        gpu_finite_mask = np.isfinite(benchmark_res)
        golden_finite_mask = np.isfinite(golden_with_actual_dtype)
        all_finite_mask = np.logical_and(npu_finite_mask, gpu_finite_mask)
        all_finite_mask = np.logical_and(golden_finite_mask, all_finite_mask)
        inf_or_nan_mask = np.logical_not(all_finite_mask)

        small_value_thresh = standard.small_value
        small_value_mask = np.less_equal(abs_golden, small_value_thresh)
        small_value_mask = np.logical_and(all_finite_mask, small_value_mask)

        # finite_mask = np.logical_and(all_finite_mask, np.logical_not(small_value_mask))
        golden_finite = np.where(all_finite_mask, golden, 0)
        finite_cnt = np.sum(all_finite_mask)
        if finite_cnt > 0:
            golden_avg = np.average(golden_finite) * (golden.size / finite_cnt)
        else:
            golden_avg = 0

        npu_ae = npu_res - golden
        abs_npu_ae = np.abs(npu_ae)
        gpu_ae = benchmark_res - golden
        abs_gpu_ae = np.abs(gpu_ae)
        npu_re = abs_npu_ae / abs_golen_with_eps
        gpu_re = abs_gpu_ae / abs_golen_with_eps
        OP_LOGGER.info("[BenchmarkCompareChecker] start stat npu precision info")
        npu_stat_info = PrecisionStatistician.stat_precision_info(golden, npu_res,
                                                                  abs_golden=abs_golden,
                                                                  abs_golden_with_eps=abs_golen_with_eps,
                                                                  small_value_mask=small_value_mask,
                                                                  inf_nan_mask=inf_or_nan_mask,
                                                                  # ae=npu_ae,
                                                                  small_value_thresh=standard.small_value,
                                                                  small_value_atol=standard.small_value_atol,
                                                                  abs_ae=abs_npu_ae,
                                                                  re=npu_re,
                                                                  actual_dtype=standard.data_type,
                                                                  ulp_limit=standard.ulp_limit)
        OP_LOGGER.info("[BenchmarkCompareChecker] end stat gpu precision info")
        OP_LOGGER.info("[BenchmarkCompareChecker] start stat gpu precision info")
        gpu_stat_info = PrecisionStatistician.stat_precision_info(golden, benchmark_res,
                                                                  abs_golden=abs_golden,
                                                                  abs_golden_with_eps=abs_golen_with_eps,
                                                                  small_value_mask=small_value_mask,
                                                                  inf_nan_mask=inf_or_nan_mask,
                                                                  # ae=gpu_ae,
                                                                  small_value_thresh=standard.small_value,
                                                                  small_value_atol=standard.small_value_atol,
                                                                  abs_ae=abs_gpu_ae,
                                                                  re=gpu_re,
                                                                  actual_dtype=standard.data_type,
                                                                  ulp_limit=standard.ulp_limit)
        OP_LOGGER.info("[BenchmarkCompareChecker] end stat gpu precision info")
        OP_LOGGER.info("[BenchmarkCompareChecker] start _check_inf_nan_value")
        cls._check_inf_nan_value(result, standard, golden, npu_res, benchmark_res, golden_with_actual_dtype)
        OP_LOGGER.info("[BenchmarkCompareChecker] end _check_inf_nan_value")
        OP_LOGGER.info("[BenchmarkCompareChecker] start _check_all_finite")
        cls._check_all_finite(result, standard, abs_npu_ae, abs_gpu_ae, golden, npu_res, benchmark_res, all_finite_mask)
        OP_LOGGER.info("[BenchmarkCompareChecker] end _check_all_finite")
        result.npu_stat_info = npu_stat_info
        result.golden_avg = golden_avg
        result.benchmark_stat_info = gpu_stat_info
        # result.total_cnt = npu_stat_info.re_distribute_info.total_cnt
        result.total_cnt = int(golden.size)
        result.do_summary()
        return result


@PRECISION_CHECKER.registry(BenchmarkCompareStandard)
def check_with_benchmark_compare_standard(standard: BenchmarkCompareStandard,
                                          case_info: TestCaseInfo,
                                          npu_res: np.ndarray,
                                          golden: np.ndarray,
                                          benchmark_res: np.ndarray) -> BenchmarkCompareCheckResult:
    return BenchmarkCompareChecker.check(case_info, standard, npu_res, golden, benchmark_res)


@PRECISION_CHECKER.registry(ULPBenchmarkCompareStandard)
def check_with_ulp_benchmark_compare_standard(standard: ULPBenchmarkCompareStandard,
                                              case_info: TestCaseInfo,
                                              npu_res: np.ndarray,
                                              golden: np.ndarray,
                                              benchmark_res: np.ndarray) -> ULPBenchmarkCompareCheckResult:
    return BenchmarkCompareChecker.check(case_info, standard, npu_res, golden, benchmark_res)


class AbsoluteThreshChecker:
    @classmethod
    def _check_small_value(cls,
                           result: AbsoluteThreshCheckResult,
                           standard: AbsoluteThreshStandard,
                           abs_ae: np.ndarray,
                           small_value_mask: np.ndarray,
                           npu_res: np.ndarray,
                           golden: np.ndarray
                           ):
        greater_mask = np.greater(abs_ae, standard.small_value_atol)
        err_mask = np.logical_and(greater_mask, small_value_mask)
        err_cnt = np.sum(err_mask)
        result.update_small_value_err_cnt(err_cnt)

    @classmethod
    def _check_inf_nan_value(cls,
                             result: AbsoluteThreshCheckResult,
                             standard: AbsoluteThreshStandard,
                             inf_nan_mask: np.ndarray,
                             npu_res: np.ndarray,
                             golden: np.ndarray,
                             abs_gpu_with_eps: np.ndarray):
        golden_same_dtype = golden.astype(npu_res.dtype)
        a_min = np.finfo(npu_res.dtype).min if npu_res.dtype != bfloat16 else -3.3895313892515355e+38
        a_max = np.finfo(npu_res.dtype).max if npu_res.dtype != bfloat16 else 3.3895313892515355e+38
        golden_clip = np.clip(golden_same_dtype, a_min, a_max)
        npu_clip = np.clip(npu_res, a_min, a_max)
        clipped_abs_ae = np.abs(npu_clip - golden_clip)
        clipped_re = clipped_abs_ae / abs_gpu_with_eps
        pass_mask = np.less_equal(clipped_re, standard.rtol)
        both_nan_mask = np.logical_and(np.isnan(npu_res), np.isnan(golden_clip))
        pass_mask = np.logical_or(pass_mask, both_nan_mask)
        not_pass_mask = np.logical_not(pass_mask)
        not_pass_mask = np.logical_and(not_pass_mask, inf_nan_mask)

        inf_nan_err_cnt = np.sum(not_pass_mask)
        result.update_inf_nan_err_cnt(inf_nan_err_cnt)

    @classmethod
    def _check_norm_value(cls, result: AbsoluteThreshCheckResult,
                          standard: AbsoluteThreshStandard,
                          normal_mask: np.ndarray,
                          re: np.ndarray):
        err_mask = np.greater(re, standard.rtol)
        err_mask = np.logical_and(err_mask, normal_mask)
        err_cnt = np.sum(err_mask)
        result.update_finite_err_cnt(err_cnt)

    @classmethod
    def check(cls, standard: AbsoluteThreshStandard,
              case_info: TestCaseInfo,
              npu_res: np.ndarray,
              golden: np.ndarray) -> AbsoluteThreshCheckResult:
        if standard.data_type == "bfloat16":
            npu_res = npu_res.astype(np.uint32) << 16
            npu_res.dtype = np.float32
        abs_ae = np.abs(npu_res - golden)
        abs_golden = np.abs(golden)
        eps = np.finfo(golden.dtype).eps if golden.dtype != bfloat16 else 2 ** -8
        abs_golden_with_eps = abs_golden + eps
        re = abs_ae / abs_golden_with_eps

        npu_finite_mask = np.isfinite(npu_res)
        golden_finite_mask = np.isfinite(golden.astype(npu_res.dtype))
        both_finite_mask = np.logical_and(npu_finite_mask, golden_finite_mask)

        inf_nan_mask = np.logical_not(both_finite_mask)

        small_value_mask = np.less_equal(abs_golden, standard.small_value)
        small_value_mask = np.logical_and(small_value_mask, both_finite_mask)

        normal_value_mask = np.logical_and(both_finite_mask, np.logical_not(small_value_mask))

        result = AbsoluteThreshCheckResult(case_info, standard)
        result.small_value_cnt = np.sum(small_value_mask)
        result.inf_nan_cnt = np.sum(inf_nan_mask)
        result.finite_cnt = np.sum(normal_value_mask)
        cls._check_inf_nan_value(result, standard, inf_nan_mask, npu_res, golden, abs_golden_with_eps)
        cls._check_small_value(result, standard, abs_ae, small_value_mask, npu_res, golden)
        cls._check_norm_value(result, standard, normal_value_mask, re)
        result.do_summary(npu_res.size)
        stat_info = PrecisionStatistician.stat_precision_info(golden, npu_res, abs_golden=abs_golden,
                                                              abs_golden_with_eps=abs_golden_with_eps,
                                                              small_value_mask=small_value_mask,
                                                              small_value_thresh=standard.small_value,
                                                              small_value_atol=standard.small_value_atol,
                                                              inf_nan_mask=inf_nan_mask,
                                                              abs_ae=abs_ae, re=re,
                                                              actual_dtype=standard.data_type)
        result.stat_info = stat_info
        if result.finite_cnt > 0:
            result.golden_avg = np.average(np.where(both_finite_mask, golden, 0)) * (golden.size / result.finite_cnt)
        return result


class AbsoluteThreshChecker2:
    @classmethod
    def _check_small_value(cls,
                           result: AbsoluteThreshCheckResult2,
                           standard: AbsoluteThreshStandard2,
                           abs_ae: np.ndarray,
                           small_value_mask: np.ndarray,
                           npu_res: np.ndarray,
                           golden: np.ndarray
                           ):
        greater_mask = np.greater(abs_ae, standard.small_value_atol)
        err_mask = np.logical_and(greater_mask, small_value_mask)
        err_cnt = np.sum(err_mask)

        err_idx = np.nonzero(err_mask)[0]
        show_size = min(err_idx.size, 10)
        err_idx = err_idx[:show_size]
        err_npu_value = np.take_along_axis(npu_res, err_idx, axis=-1)
        err_gold_value = np.take_along_axis(golden, err_idx, axis=-1)
        err_eles = []
        for i in range(show_size):
            err_eles.append([err_idx[i], err_npu_value[i], err_gold_value[i]])

        result.update_small_value_err_cnt(err_cnt, err_eles)

    @classmethod
    def _check_inf_nan_value(cls,
                             result: AbsoluteThreshCheckResult2,
                             standard: AbsoluteThreshStandard2,
                             inf_nan_mask: np.ndarray,
                             npu_res: np.ndarray,
                             golden: np.ndarray,
                             abs_gpu_with_eps: np.ndarray):
        golden_same_dtype = golden.astype(npu_res.dtype)
        a_min = np.finfo(npu_res.dtype).min if npu_res.dtype != bfloat16 else -3.3895313892515355e+38
        a_max = np.finfo(npu_res.dtype).max if npu_res.dtype != bfloat16 else 3.3895313892515355e+38
        golden_clip = np.clip(golden_same_dtype, a_min, a_max)
        npu_clip = np.clip(npu_res, a_min, a_max)
        clipped_abs_ae = np.abs(npu_clip - golden_clip)
        clipped_re = clipped_abs_ae / abs_gpu_with_eps
        pass_mask = np.less_equal(clipped_re, standard.rtol)
        both_nan_mask = np.logical_and(np.isnan(npu_res), np.isnan(golden_clip))
        pass_mask = np.logical_or(pass_mask, both_nan_mask)
        not_pass_mask = np.logical_not(pass_mask)
        not_pass_mask = np.logical_and(not_pass_mask, inf_nan_mask)

        err_idx = np.nonzero(not_pass_mask)[0]
        show_size = min(err_idx.size, 10)
        err_idx = err_idx[:show_size]
        err_npu_value = np.take_along_axis(npu_res, err_idx, axis=-1)
        err_gold_value = np.take_along_axis(golden, err_idx, axis=-1)
        err_eles = []
        for i in range(show_size):
            err_eles.append([err_idx[i], err_npu_value[i], err_gold_value[i]])
        # OP_LOGGER.error("err_eles:")
        # OP_LOGGER.error(err_eles)

        inf_nan_err_cnt = np.sum(not_pass_mask)
        result.update_inf_nan_err_cnt(inf_nan_err_cnt, err_eles)

    @classmethod
    def _check_norm_value(cls, result: AbsoluteThreshCheckResult2,
                          standard: AbsoluteThreshStandard2,
                          normal_mask: np.ndarray,
                          npu_res: np.ndarray,
                          golden: np.ndarray,
                          re: np.ndarray,
                          abs_ae: np.ndarray):
        err_mask = np.logical_and(np.greater(re, standard.rtol), np.greater(abs_ae, standard.small_value_atol))
        err_mask = np.logical_and(err_mask, normal_mask)
        err_cnt = np.sum(err_mask)

        err_idx = np.nonzero(err_mask)[0]
        show_size = min(err_idx.size, 10)
        err_idx = err_idx[:show_size]
        err_npu_value = np.take_along_axis(npu_res, err_idx, axis=-1)
        err_gold_value = np.take_along_axis(golden, err_idx, axis=-1)
        err_eles = []
        for i in range(show_size):
            err_eles.append([err_idx[i], err_npu_value[i], err_gold_value[i]])

        result.update_finite_err_cnt(err_cnt, err_eles)

    @classmethod
    def check(cls, standard: AbsoluteThreshStandard2,
              case_info: TestCaseInfo,
              npu_res: np.ndarray,
              golden: np.ndarray) -> AbsoluteThreshCheckResult2:
        if standard.data_type == "bfloat16":
            npu_res = npu_res.astype(np.uint32) << 16
            npu_res.dtype = np.float32
        abs_ae = np.abs(npu_res - golden)
        abs_golden = np.abs(golden)
        eps = {"float32": 0.000001, "float16": 0.001, "bfloat16": 0.001, "float": 0.000001}[str(npu_res.dtype)]
        abs_golden_with_eps = abs_golden + eps
        re = abs_ae / abs_golden_with_eps

        npu_finite_mask = np.isfinite(npu_res)
        golden_finite_mask = np.isfinite(golden.astype(npu_res.dtype))
        both_finite_mask = np.logical_and(npu_finite_mask, golden_finite_mask)

        inf_nan_mask = np.logical_not(both_finite_mask)

        small_value_mask = np.less_equal(abs_golden, standard.small_value)
        small_value_mask = np.logical_and(small_value_mask, both_finite_mask)

        normal_value_mask = np.logical_and(both_finite_mask, np.logical_not(small_value_mask))

        result = AbsoluteThreshCheckResult2(case_info, standard)
        result.small_value_cnt = np.sum(small_value_mask)
        result.inf_nan_cnt = np.sum(inf_nan_mask)
        result.finite_cnt = np.sum(normal_value_mask)
        cls._check_inf_nan_value(result, standard, inf_nan_mask, npu_res, golden, abs_golden_with_eps)
        cls._check_small_value(result, standard, abs_ae, small_value_mask, npu_res, golden)
        cls._check_norm_value(result, standard, normal_value_mask, npu_res, golden, re, abs_ae)

        result.do_summary(npu_res.size)
        stat_info = PrecisionStatistician.stat_precision_info(golden, npu_res, abs_golden=abs_golden,
                                                              abs_golden_with_eps=abs_golden_with_eps,
                                                              small_value_mask=small_value_mask,
                                                              small_value_thresh=standard.small_value,
                                                              small_value_atol=standard.small_value_atol,
                                                              inf_nan_mask=inf_nan_mask,
                                                              abs_ae=abs_ae, re=re,
                                                              actual_dtype=standard.data_type)
        result.stat_info = stat_info
        if result.finite_cnt > 0:
            result.golden_avg = np.average(np.where(both_finite_mask, golden, 0)) * (golden.size / result.finite_cnt)
        # result.printd()
        return result


@PRECISION_CHECKER.registry(AbsoluteThreshStandard)
def check_with_absolute_thesh_standard(standard: AbsoluteThreshStandard,
                                       case_info: TestCaseInfo,
                                       npu_res: np.ndarray,
                                       golden: np.ndarray) -> AbsoluteThreshCheckResult:
    return AbsoluteThreshChecker.check(standard, case_info, npu_res, golden)


@PRECISION_CHECKER.registry(AbsoluteThreshStandard2)
def check_with_absolute_thesh_standard(standard: AbsoluteThreshStandard2,
                                       case_info: TestCaseInfo,
                                       npu_res: np.ndarray,
                                       golden: np.ndarray) -> AbsoluteThreshCheckResult2:
    return AbsoluteThreshChecker2.check(standard, case_info, npu_res, golden)


class BinaryCompareChecker:
    @classmethod
    def _get_xor_dtype(cls, actual: np.ndarray):
        if actual.itemsize == 1:
            return np.uint8
        if actual.itemsize == 2:
            return np.uint16
        if actual.itemsize == 4:
            return np.uint32
        if actual.itemsize == 8:
            return np.uint64
        return None

    @classmethod
    def _get_err_elements(cls,
                          npu_res: np.ndarray,
                          golden: np.ndarray,
                          err_mask: np.ndarray):
        err_idx = np.nonzero(err_mask)[0]
        show_size = min(err_idx.size, 10)
        err_idx = err_idx[:show_size]
        err_npu_value = np.take_along_axis(npu_res, err_idx, axis=-1)
        err_gold_value = np.take_along_axis(golden, err_idx, axis=-1)
        err_eles = []
        for i in range(show_size):
            err_eles.append([err_idx[i], err_npu_value[i], err_gold_value[i]])
        return err_eles

    @classmethod
    def check(cls,
              standard: BinaryCompareStandard,
              case_info: TestCaseInfo,
              npu_res: np.ndarray,
              golden: np.ndarray) -> BinaryCompareCheckResult:
        result = BinaryCompareCheckResult(case_info, standard)
        if npu_res.dtype != golden.dtype:
            result.run_err("Actual data type should be same to golden data type when use BinaryCompareStandard")
            return result
        compare_type = cls._get_xor_dtype(npu_res)
        if not compare_type:
            result.run_err("This data type:%s can't use BinaryCompareStandard" % npu_res.dtype)
            return result
        ori_dtype = npu_res.dtype
        npu_res.dtype = compare_type
        golden.dtype = compare_type
        xor_res = np.bitwise_xor(npu_res, golden)
        err_mask = np.not_equal(xor_res, 0)
        npu_res.dtype = ori_dtype
        golden.dtype = ori_dtype
        err_count = np.sum(err_mask)
        if err_count > 0:
            if standard.data_type == "bfloat16":
                npu_res_uint32 = npu_res.astype(np.uint32) << 16
                npu_res_uint32.dtype = np.float32
                golden_uint32 = golden.astype(np.uint32) << 16
                golden_uint32.dtype = np.float32
                err_eles = cls._get_err_elements(npu_res_uint32, golden_uint32, err_mask)
            else:
                err_eles = cls._get_err_elements(npu_res, golden, err_mask)
        else:
            err_eles = []
        result.do_summary(total_cnt=npu_res.size, err_cnt=err_count, err_eles=err_eles)
        return result


@PRECISION_CHECKER.registry(BinaryCompareStandard)
def check_with_binary_compare_standard(standard: BinaryCompareStandard,
                                       case_info: TestCaseInfo,
                                       npu_res: np.ndarray,
                                       golden: np.ndarray) -> BinaryCompareCheckResult:
    return BinaryCompareChecker.check(standard, case_info, npu_res, golden)


class PrecisionCheckStandardFactory:

    @classmethod
    def _get_default_small_value_by_dtype(cls, data_type):
        return {"float32": 0.000001, "float16": 0.001, "bfloat16": 0.004, "float": 0.000001}[data_type]

    @classmethod
    def _get_default_absolute_standard_by_dtype(cls, data_type: str, **kwargs):
        rtol = kwargs.get("rtol", None)
        if rtol is None:
            rtol = {"float32": 0.000001, "float16": 0.001, "bfloat16": 0.004, "float": 0.000001}[data_type]
        small_value = kwargs.get("small_value", None)
        if small_value is None:
            small_value = cls._get_default_small_value_by_dtype(data_type)
        small_value_atol = kwargs.get("small_value_atol", None)
        if small_value_atol is None:
            small_value_atol = small_value
        return AbsoluteThreshStandard(data_type, rtol, small_value, small_value_atol)

    @classmethod
    def build_standard(cls, standard_name: str, data_type: Union[str, None], **kwargs) -> Union[
        BinaryCompareStandard, AbsoluteThreshStandard, GPUCompareStandard]:
        if standard_name == "BinaryCompareStandard":
            return BinaryCompareStandard(data_type)
        if standard_name == "AbsoluteThreshStandard":
            return cls._get_default_absolute_standard_by_dtype(data_type, **kwargs)
        if standard_name == "GPUCompareStandard":
            rtol = kwargs.get("rtol", 2.0)
            small_value = kwargs.get("small_value", None)
            if small_value is None:
                small_value = cls._get_default_small_value_by_dtype(data_type)
            small_value_atol = kwargs.get("small_value_atol", None)
            if small_value_atol is None:
                small_value_atol = small_value
            return GPUCompareStandard(data_type, rtol, small_value, small_value_atol)

    @classmethod
    def build_standard_by_yml_info(cls, data_type: str, **kwargs) -> Union[PrecisionCheckStandard, None]:
        standard_cls_map = {
            BinaryCompareStandard.NAME: BinaryCompareStandard,
            GPUCompareStandard.NAME: GPUCompareStandard,
            AbsoluteThreshStandard.NAME: AbsoluteThreshStandard,
            AbsoluteThreshStandard2.NAME: AbsoluteThreshStandard2,
            BenchmarkCompareStandard.NAME: BenchmarkCompareStandard,
            ULPBenchmarkCompareStandard.NAME: ULPBenchmarkCompareStandard
        }
        standard_name = kwargs.get("standard")
        if standard_name is None or standard_name not in standard_cls_map.keys():
            return None
        standard_cls: PrecisionCheckStandard = standard_cls_map.get(standard_name)
        return standard_cls.build_by_yml_info(data_type, **kwargs)


class StandardReader:
    def __init__(self, yaml_file_path: str):
        self.op_standards = {}
        self._file_path = yaml_file_path
        self.info_obj = None
        self._read_yaml(yaml_file_path)

    def _read_one_op(self, one_op_info):
        if not isinstance(one_op_info, dict):
            raise RuntimeError("Read op standard file failed, file path: %s" % self._file_path)
        res = {}
        for dtype, info in one_op_info.items():
            res[dtype] = PrecisionCheckStandardFactory.build_standard_by_yml_info(dtype, **info)
            # standard_name = info.get("standard")
            # res[dtype] = PrecisionCheckStandardFactory.build_standard(standard_name, dtype, **info)
        return res

    def _read_yaml(self, yaml_file_path: str):
        # print(yaml_file_path,os.getcwd())
        with open(yaml_file_path) as ff:
            result = yaml.safe_load(ff)

        self.info_obj = result

    def read_op_stand(self, op_type, data_type):
        standards = self.op_standards.get(op_type, None)
        if standards is not None:
            return standards.get(data_type, None)
        if op_type not in self.info_obj.keys():
            return None
        self.op_standards[op_type] = self._read_one_op(self.info_obj[op_type])
        return self.op_standards.get(op_type, {}).get(data_type, None)





class OpPrecisionTestSummary:
    def __init__(self, op_type):
        self.op_type = op_type
        # self.grouped_cases_by_standard: Dict[str, List[PrecisionCheckResult]] = {}
        self.dtype_standard_grouped: Dict[str, List[PrecisionCheckResult]] = {}
        self.check_result = PrecisionCheckResult.SUCCESS

    def add_case_result(self, case_result_info: PrecisionCheckResult):
        if not case_result_info:
            return
        # data_type = case_result_info.precision_check_standard.data_type
        standard = case_result_info.precision_check_standard
        case_key = "%s_%s" % (standard.data_type, standard.standard_name)
        case_values = self.dtype_standard_grouped.get(case_key, [])
        self.dtype_standard_grouped[case_key] = case_values
        case_values.append(case_result_info)
        self.check_result = max(self.check_result, case_result_info.check_result)

    def do_summary(self):
        pass


class XlsxSheetWriter:
    @classmethod
    def _column_name(cls, num: int):
        if num < 26:
            return chr(ord('A') + num)
        letters = string.ascii_uppercase
        num_to_letter = {i + 1: letter for i, letter in enumerate(letters)}
        pre = num_to_letter[num // 26]
        return pre + chr(ord('A') + num % 26)

    @classmethod
    def _add_case_info_title(cls, worksheet, title_format, start_column, title_height: int) -> int:
        case_info_title = ["算子\n类型", "用例", "用例参数", "输入值域"]
        end_column = start_column + len(case_info_title) - 1
        worksheet.set_column("%s:%s" % (cls._column_name(start_column), cls._column_name(end_column)), 6)
        if title_height == 1:
            for i, title in enumerate(case_info_title):
                worksheet.write(0, start_column + i, title, title_format)
        else:
            for i, title in enumerate(case_info_title):
                worksheet.merge_range("%s1:%s%d" % (cls._column_name(start_column + i),
                                                    cls._column_name(start_column + i),
                                                    title_height),
                                      title, title_format)
        return end_column + 1

    @classmethod
    def _write_stat_info_title(cls, worksheet, title_format,
                               start_column: int, data_type: str,
                               ulp_limit: int = 1, title_key: str = "NPU") -> int:
        # 相对误差分布情况
        re_dis_histogram_bins = REHistogramBins.get_histogram_bin_titles(data_type)
        re_dis_histogram_bins = ["统计\n个数"] + re_dis_histogram_bins + ["最大值", "均值"]
        # ULP误差分布情况
        ulp_sub_titles = REHistogramBins.get_ulp_histogram_bin_titles(ulp_limit)
        ulp_sub_titles = ulp_sub_titles + ["最大", "最小", "均值", "abs均值"]
        # 小值域分布情况
        small_value_titles = ["统计\n个数", "总个数\n占比", "error\n个数", "error\n占比"]
        # 其他统计项
        other_titles = ["均方根\n误差", "误差\n均衡性"]

        end_column = len(re_dis_histogram_bins) + len(small_value_titles) + len(other_titles) + len(
            ulp_sub_titles) + start_column - 1
        range_name = "%s:%s" % (cls._column_name(start_column), cls._column_name(end_column))
        worksheet.set_column(range_name, 5)
        range_name = "%s1:%s1" % (cls._column_name(start_column), cls._column_name(end_column))

        # 第一层title
        worksheet.merge_range(range_name, title_key + '误差情况', title_format)
        # 相对误差分布
        re_start, re_end = start_column, start_column + len(re_dis_histogram_bins) - 1
        range_name = "%s2:%s2" % (cls._column_name(re_start), cls._column_name(re_end))
        worksheet.merge_range(range_name, '相对误差分布', title_format)
        for i, re_title in enumerate(re_dis_histogram_bins):
            worksheet.write(2, re_start + i, re_title, title_format)
        # ULP误差情况
        next_column = re_end + 1
        ulp_start, ulp_end = next_column, next_column + len(ulp_sub_titles) - 1
        range_name = "%s2:%s2" % (cls._column_name(ulp_start), cls._column_name(ulp_end))
        worksheet.merge_range(range_name, 'ULP误差情况', title_format)
        for i, ulp_title in enumerate(ulp_sub_titles):
            worksheet.write(2, ulp_start + i, ulp_title, title_format)
        # 小值域
        next_column = ulp_end + 1
        small_start, small_end = next_column, next_column + len(small_value_titles) - 1
        range_name = "%s2:%s2" % (cls._column_name(small_start), cls._column_name(small_end))
        worksheet.merge_range(range_name, '小值域', title_format)
        for i, small_t in enumerate(small_value_titles):
            worksheet.write(2, small_start + i, small_t, title_format)
        next_column = small_end + 1
        # 其他统计项
        for i, o_t in enumerate(other_titles):
            range_name = "%s2:%s3" % (cls._column_name(next_column + i), cls._column_name(next_column + i))
            worksheet.merge_range(range_name, o_t, title_format)
        return end_column + 1

    @classmethod
    def _write_stat_info_row(cls, worksheet, row_format,
                             percent_cell_format,
                             row_num, start_column_num,
                             stat_info: PrecisionStatInfo) -> int:
        # 相对误差统计情况
        re_info = stat_info.re_distribute_info
        worksheet.write(row_num, start_column_num, re_info.total_cnt, row_format)  # total cnt
        next_column = start_column_num + 1
        for i, v in enumerate(re_info.histogram_stat_ratio):  # re_dist
            worksheet.write(row_num, next_column + i, v, percent_cell_format)
        next_column += len(stat_info.re_distribute_info.histogram_stat_ratio)
        worksheet.write(row_num, next_column, re_info.max_re, row_format)  # max_re
        worksheet.write(row_num, next_column + 1, re_info.avg_re, row_format)  # avg_re
        next_column += 2
        # ULP 误差情况：["最大", "最小", "均值", "abs均值"]
        ulp_info = stat_info.ulp_distribute_info
        for i, v in enumerate(ulp_info.histogram_stat_ratio):
            worksheet.write(row_num, next_column + i, v, percent_cell_format)
        next_column += len(ulp_info.histogram_stat_ratio)
        worksheet.write(row_num, next_column, "%s" % ulp_info.max_ulp, row_format)
        worksheet.write(row_num, next_column + 1, "%s" % ulp_info.min_ulp, row_format)
        worksheet.write(row_num, next_column + 2, "%s" % ulp_info.avg_ulp, row_format)
        worksheet.write(row_num, next_column + 3, "%s" % ulp_info.avg_abs_ulp, row_format)
        next_column += 4
        # 小值域： ["统计\n个数", "总个数\n占比", "error\n个数", "error\n占比"]
        small_value_info = stat_info.small_value_info
        worksheet.write(row_num, next_column, small_value_info.small_value_count, row_format)  # small_value
        worksheet.write(row_num, next_column + 1, small_value_info.small_value_ratio, percent_cell_format)
        worksheet.write(row_num, next_column + 2, small_value_info.small_value_err_count, row_format)
        worksheet.write(row_num, next_column + 3, small_value_info.small_value_err_ratio, percent_cell_format)
        next_column += 4
        # 其他: ["均方根\n误差", "误差\n均衡性"]
        worksheet.write(row_num, next_column, stat_info.rmse, row_format)
        worksheet.write(row_num, next_column + 1, stat_info.eb_info.eb, row_format)
        return next_column + 2

    @classmethod
    def _write_case_info_row(cls, worksheet, row_format, row_num,
                             start_column_num,
                             case_result: PrecisionCheckResult) -> int:
        # case_info_title = ["算子\n类型", "用例", "用例参数", "输入值域"]
        worksheet.write(row_num, start_column_num, case_result.case_info.op_type, row_format)
        worksheet.write(row_num, start_column_num + 1, case_result.case_info.case_name, row_format)
        worksheet.write(row_num, start_column_num + 2, case_result.case_info.case_param_info, row_format)
        worksheet.write(row_num, start_column_num + 3, case_result.case_info.case_input_value_range_info, row_format)
        return start_column_num + 4


class BinaryCompareXlsxSheetWriter(XlsxSheetWriter):
    @classmethod
    def _write_title(cls, workbook, worksheet):
        title_format = workbook.add_format({
            'bold': True,
            'align': 'center',
            'border': 1,
            'text_wrap': 1,
            'valign': 'vcenter',
            'fg_color': '#00b0f0'
        })
        next_column = super()._add_case_info_title(worksheet, title_format, start_column=0, title_height=1)
        binary_compare_title = ["统计个数", "错误个数", "错误占比", "测试结果", "错误信息"]
        for i, t in enumerate(binary_compare_title):
            worksheet.write(0, next_column + i, t, title_format)

    @classmethod
    def _write_rows(cls, workbook, worksheet, result_list: List[BinaryCompareCheckResult]):
        cell_format = workbook.add_format({
            'align': 'center',
            'border': 1,
            'valign': 'vcenter'
        })
        percent_cell_format = workbook.add_format({
            'align': 'center',
            'border': 1,
            'valign': 'vcenter'
        })
        percent_cell_format.set_num_format("0.00%")
        for idx, result in enumerate(result_list):
            next_column = super()._write_case_info_row(worksheet, cell_format, idx + 1, 0, result)
            if not result.run_success:
                worksheet.write(idx + 1, next_column + 4, result.run_err_msg, cell_format)
            else:
                worksheet.write(idx + 1, next_column, result.total_cnt, cell_format)
                worksheet.write(idx + 1, next_column + 1, result.err_cnt, cell_format)
                worksheet.write(idx + 1, next_column + 2, result.err_ratio, percent_cell_format)
                worksheet.write(idx + 1, next_column + 3, result.get_check_result_str(), cell_format)
                worksheet.write(idx + 1, next_column + 4, result.get_result_msg(), cell_format)

    @classmethod
    def write_op_test_summary(cls, workbook, worksheet, result_list: List[BinaryCompareCheckResult]):
        OP_LOGGER.info("start write_op_test_summary for BinaryCompare, case count: %d" % len(result_list))
        cls._write_title(workbook, worksheet)
        cls._write_rows(workbook, worksheet, result_list)


class GpuCompareXlsxSheetWriter(XlsxSheetWriter):
    @classmethod
    def _write_title(cls, workbook, worksheet, data_type):
        title_format_one = workbook.add_format({
            'bold': True,
            'align': 'center',
            'border': 1,
            'text_wrap': 1,
            'valign': 'vcenter',
            'fg_color': '#00b0f0'
        })
        title_format_two = workbook.add_format({
            'bold': True,
            'align': 'center',
            'border': 1,
            'text_wrap': 1,
            'valign': 'vcenter',
            'fg_color': '#92d050'
        })
        next_column = super()._add_case_info_title(worksheet, title_format_one, start_column=0, title_height=3)
        merge_range_str = "%s1:%s3" % (super()._column_name(next_column), super()._column_name(next_column))
        worksheet.merge_range(merge_range_str, "真值\n均值", title_format_one)
        next_column += 1

        next_column = super()._write_stat_info_title(worksheet, title_format_two, next_column, data_type, title_key="NPU")
        next_column = super()._write_stat_info_title(worksheet, title_format_one, next_column, data_type, title_key="GPU")

        compare_title = ["统计\n数量", "PASS\n占比", "WARN\n占比", "ERROR\n占比", "最大\n相对\n误差\n差值",
                         "误差\n均衡性\n差值",
                         "均方根\n误差\n差值"]
        compare_end_column = next_column + len(compare_title) - 1
        range_name = "%s1:%s1" % (super()._column_name(next_column), super()._column_name(compare_end_column))
        worksheet.merge_range(range_name, "NPU VS GPU", title_format_two)
        for i, o_t in enumerate(compare_title):
            range_name = "%s:%s" % (super()._column_name(next_column + i), super()._column_name(next_column + i))
            worksheet.set_column(range_name, 5)
            range_name = "%s2:%s3" % (super()._column_name(next_column + i), super()._column_name(next_column + i))
            worksheet.merge_range(range_name, o_t, title_format_two)

        next_column = compare_end_column + 1
        range_name = "%s1:%s3" % (super()._column_name(next_column), super()._column_name(next_column))
        worksheet.merge_range(range_name, "测试结果", title_format_one)
        range_name = "%s1:%s3" % (super()._column_name(next_column + 1), super()._column_name(next_column + 1))
        worksheet.merge_range(range_name, "错误信息", title_format_one)

    @classmethod
    def _write_gpu_compare_row(cls, worksheet, row_format, percent_cell_format, row_num, column_num,
                               case_result: GPUCompareCheckResult):
        # ["PASS\n占比", "WARN\n占比", "ERROR\n占比", "最大\n相对\n误差\n差值", "误差\n均衡性\n差值", "均方根\n误差\n差值"]
        # column_num = column_num - 1
        worksheet.write(row_num, column_num, case_result.total_cnt, row_format)
        worksheet.write(row_num, column_num + 1, case_result.total_pass_ratio, percent_cell_format)
        worksheet.write(row_num, column_num + 2, case_result.total_warn_ratio, percent_cell_format)
        worksheet.write(row_num, column_num + 3, case_result.total_err_ratio, percent_cell_format)
        diff = case_result.npu_stat_info.re_distribute_info.max_re - case_result.gpu_stat_info.re_distribute_info.max_re
        worksheet.write(row_num, column_num + 4, diff, row_format)
        diff = case_result.npu_stat_info.eb_info.eb - case_result.gpu_stat_info.eb_info.eb
        worksheet.write(row_num, column_num + 5, diff, row_format)
        diff = case_result.npu_stat_info.rmse - case_result.gpu_stat_info.rmse
        worksheet.write(row_num, column_num + 6, diff, row_format)

        return column_num + 7

    @classmethod
    def _write_rows(cls, workbook, worksheet, result_list: List[GPUCompareCheckResult]):
        cell_format = workbook.add_format({
            'align': 'center',
            'border': 1,
            'valign': 'vcenter'
        })
        percent_cell_format = workbook.add_format({
            'align': 'center',
            'border': 1,
            'valign': 'vcenter'
        })
        percent_cell_format.set_num_format("0.00%")
        for idx, result in enumerate(result_list):
            next_column = super()._write_case_info_row(worksheet, cell_format, idx + 3, 0, result)
            worksheet.write(idx + 3, next_column, result.golden_avg, cell_format)
            next_column += 1
            npu_stat_info = result.npu_stat_info
            next_column = super()._write_stat_info_row(worksheet, cell_format, percent_cell_format, idx + 3,
                                                       next_column, npu_stat_info)
            gpu_stat_info = result.gpu_stat_info
            next_column = super()._write_stat_info_row(worksheet, cell_format, percent_cell_format, idx + 3,
                                                       next_column, gpu_stat_info)

            next_column = cls._write_gpu_compare_row(worksheet, cell_format, percent_cell_format, idx + 3, next_column,
                                                     result)
            worksheet.write(idx + 3, next_column, result.get_check_result_str(), cell_format)
            worksheet.write(idx + 3, next_column + 1, result.get_result_msg(), cell_format)

    @classmethod
    def write_op_test_summary(cls, workbook, worksheet, result_list: List[GPUCompareCheckResult]):
        OP_LOGGER.info("start write_op_test_summary for gpu compare, case count: %d" % len(result_list))
        if len(result_list) > 0:
            data_type = result_list[0].precision_check_standard.data_type
        else:
            data_type = "float32"
        cls._write_title(workbook, worksheet, data_type)
        cls._write_rows(workbook, worksheet, result_list)


class BenchmarkCompareXlsxSheetWriter(XlsxSheetWriter):
    @classmethod
    def _write_title(cls, workbook, worksheet, data_type, ulp_limit):
        title_format_one = workbook.add_format({
            'bold': True,
            'align': 'center',
            'border': 1,
            'text_wrap': 1,
            'valign': 'vcenter',
            'fg_color': '#00b0f0'
        })
        title_format_two = workbook.add_format({
            'bold': True,
            'align': 'center',
            'border': 1,
            'text_wrap': 1,
            'valign': 'vcenter',
            'fg_color': '#92d050'
        })
        next_column = super()._add_case_info_title(worksheet, title_format_one, start_column=0, title_height=3)
        merge_range_str = "%s1:%s3" % (super()._column_name(next_column), super()._column_name(next_column))
        worksheet.merge_range(merge_range_str, "真值\n均值", title_format_one)
        next_column += 1

        next_column = super()._write_stat_info_title(worksheet, title_format_two, next_column, data_type, ulp_limit, "NPU")
        next_column = super()._write_stat_info_title(worksheet, title_format_one, next_column, data_type, ulp_limit, "GPU")
        compare_title = ["统计\n数量", "INF/NAN数量", "INF/NAN错误数量", "逐点绝对误差通过数量", "最大abs(ULP)误差比值",
                         "abs(ULP)平均误差比值", "最大相对误差比值", "平均相对误差比值", "均方根误差比值", "误差均衡性差值",
                         "小值域绝对误差比值", "逐点绝对误差通过率", "abs(ULP)>1误差比值"]
        compare_end_column = next_column + len(compare_title) - 1
        range_name = "%s1:%s1" % (super()._column_name(next_column), super()._column_name(compare_end_column))
        worksheet.merge_range(range_name, "NPU VS GPU", title_format_two)
        for i, o_t in enumerate(compare_title):
            range_name = "%s:%s" % (super()._column_name(next_column + i), super()._column_name(next_column + i))
            worksheet.set_column(range_name, 5)
            range_name = "%s2:%s3" % (super()._column_name(next_column + i), super()._column_name(next_column + i))
            worksheet.merge_range(range_name, o_t, title_format_two)

        next_column = compare_end_column + 1
        range_name = "%s1:%s3" % (super()._column_name(next_column), super()._column_name(next_column))
        worksheet.merge_range(range_name, "测试结果", title_format_one)
        range_name = "%s1:%s3" % (super()._column_name(next_column + 1), super()._column_name(next_column + 1))
        worksheet.merge_range(range_name, "错误信息", title_format_one)

    @classmethod
    def _write_benchmark_compare_row(cls, worksheet, row_format, percent_cell_format, row_num, column_num,
                                     case_result: BenchmarkCompareCheckResult):
        # ["PASS\n占比", "WARN\n占比", "ERROR\n占比", "最大\n相对\n误差\n差值", "误差\n均衡性\n差值", "均方根\n误差\n差值"]
        # column_num = column_num - 1
        worksheet.write(row_num, column_num, case_result.total_cnt, row_format)
        worksheet.write(row_num, column_num + 1, case_result.inf_nan_cnt, row_format)
        worksheet.write(row_num, column_num + 2, case_result.inf_nan_err_cnt, row_format)
        worksheet.write(row_num, column_num + 3, case_result.total_cnt - case_result.ae_err_cnt, row_format)
        worksheet.write(row_num, column_num + 4, case_result.max_abs_ulp_ratio, row_format)
        worksheet.write(row_num, column_num + 5, case_result.avg_abs_ulp_ratio, row_format)
        worksheet.write(row_num, column_num + 6, case_result.max_re_ratio, row_format)
        # diff = case_result.npu_stat_info.re_distribute_info.max_re - case_result.gpu_stat_info.re_distribute_info.max_re
        worksheet.write(row_num, column_num + 7, case_result.avg_re_ratio, row_format)
        # diff = case_result.npu_stat_info.eb_info.eb - case_result.gpu_stat_info.eb_info.eb
        worksheet.write(row_num, column_num + 8, case_result.rmse_ratio, row_format)
        # diff = case_result.npu_stat_info.rmse - case_result.gpu_stat_info.rmse
        worksheet.write(row_num, column_num + 9, case_result.eb_difference, row_format)
        worksheet.write(row_num, column_num + 10, case_result.small_ae_ratio, row_format)
        worksheet.write(row_num, column_num + 11, case_result.ae_pass_ratio, percent_cell_format)
        worksheet.write(row_num, column_num + 12, case_result.ulp_err_ratio, row_format)

        return column_num + 13

    @classmethod
    def _write_rows(cls, workbook, worksheet, result_list: List[BenchmarkCompareCheckResult]):
        cell_format = workbook.add_format({
            'align': 'center',
            'border': 1,
            'valign': 'vcenter'
        })
        percent_cell_format = workbook.add_format({
            'align': 'center',
            'border': 1,
            'valign': 'vcenter'
        })
        percent_cell_format.set_num_format("0.00%")
        for idx, result in enumerate(result_list):
            if not result.run_success:
                next_column = super()._write_case_info_row(worksheet, cell_format, idx + 3, 0, result)
                worksheet.write(idx + 3, next_column + 1, result.run_err_msg, cell_format)
                continue
            next_column = super()._write_case_info_row(worksheet, cell_format, idx + 3, 0, result)
            worksheet.write(idx + 3, next_column, result.golden_avg, cell_format)
            next_column += 1
            npu_stat_info = result.npu_stat_info
            next_column = super()._write_stat_info_row(worksheet, cell_format, percent_cell_format, idx + 3,
                                                       next_column, npu_stat_info)
            benchmark_stat_info = result.benchmark_stat_info
            next_column = super()._write_stat_info_row(worksheet, cell_format, percent_cell_format, idx + 3,
                                                       next_column, benchmark_stat_info)

            next_column = cls._write_benchmark_compare_row(worksheet, cell_format, percent_cell_format, idx + 3,
                                                           next_column, result)
            worksheet.write(idx + 3, next_column, result.get_check_result_str(), cell_format)
            worksheet.write(idx + 3, next_column + 1, result.get_result_msg(), cell_format)

    @classmethod
    def write_op_test_summary(cls, workbook, worksheet, result_list: List[BenchmarkCompareCheckResult]):
        OP_LOGGER.info("start write_op_test_summary for gpu compare, case count: %d" % len(result_list))
        if len(result_list) > 0:
            data_type = result_list[0].precision_check_standard.data_type
        else:
            data_type = "float32"
        ulp_limit = result_list[0].precision_check_standard.ulp_limit
        cls._write_title(workbook, worksheet, data_type, ulp_limit)
        cls._write_rows(workbook, worksheet, result_list)


class AbsoluteThreshXlsxSheetWriter(XlsxSheetWriter):
    @classmethod
    def _write_title(cls, workbook, worksheet, data_type, ulp_limit):
        title_format_one = workbook.add_format({
            'bold': True,
            'align': 'center',
            'border': 1,
            'text_wrap': 1,
            'valign': 'vcenter',
            'fg_color': '#00b0f0'
        })
        title_format_two = workbook.add_format({
            'bold': True,
            'align': 'center',
            'border': 1,
            'text_wrap': 1,
            'valign': 'vcenter',
            'fg_color': '#92d050'
        })
        next_column = super()._add_case_info_title(worksheet, title_format_one, start_column=0, title_height=3)
        merge_range_str = "%s1:%s3" % (super()._column_name(next_column), super()._column_name(next_column))
        worksheet.merge_range(merge_range_str, "真值\n均值", title_format_one)
        next_column += 1

        next_column = super()._write_stat_info_title(worksheet, title_format_two, next_column, data_type, ulp_limit, "NPU")

        compare_title = ["统计\n数量", "PASS\n占比", "ERROR\n占比"]
        worksheet.merge_range(
            "%s1:%s1" % (super()._column_name(next_column), super()._column_name(next_column + len(compare_title) - 1)),
            "精度检查情况",
            title_format_one)
        for i, o_t in enumerate(compare_title):
            worksheet.set_column(
                "%s:%s" % (super()._column_name(next_column + i), super()._column_name(next_column + i)), 5)
            worksheet.merge_range(
                "%s2:%s3" % (super()._column_name(next_column + i), super()._column_name(next_column + i)),
                o_t,
                title_format_one)
        next_column += len(compare_title)
        worksheet.merge_range("%s1:%s3" % (super()._column_name(next_column), super()._column_name(next_column)),
                              "测试结果",
                              title_format_two)
        range_name = "%s1:%s3" % (super()._column_name(next_column + 1), super()._column_name(next_column + 1))
        worksheet.merge_range(range_name, "错误信息", title_format_one)

    @classmethod
    def _write_absolute_check_row(cls, worksheet, row_format, percent_cell_format, row_num, column_num,
                                  case_result: AbsoluteThreshCheckResult):
        # ["统计\n数量", "PASS\n占比", "ERROR\n占比"]
        worksheet.write(row_num, column_num, case_result.total_cnt, row_format)
        worksheet.write(row_num, column_num + 1, case_result.total_pass_ratio, percent_cell_format)
        worksheet.write(row_num, column_num + 2, case_result.total_err_ratio, percent_cell_format)
        return column_num + 3

    @classmethod
    def _write_rows(cls, workbook, worksheet, result_list: List[AbsoluteThreshCheckResult]):
        cell_format = workbook.add_format({
            'align': 'center',
            'border': 1,
            'valign': 'vcenter'
        })
        percent_cell_format = workbook.add_format({
            'align': 'center',
            'border': 1,
            'valign': 'vcenter'
        })
        percent_cell_format.set_num_format("0.00%")
        for idx, result in enumerate(result_list):
            if not result.run_success:
                next_column = super()._write_case_info_row(worksheet, cell_format, idx + 3, 0, result)
                worksheet.write(idx + 3, next_column + 1, result.run_err_msg, cell_format)
                continue
            next_column = super()._write_case_info_row(worksheet, cell_format, idx + 3, 0, result)
            worksheet.write(idx + 3, next_column, result.golden_avg, cell_format)
            next_column += 1
            npu_stat_info = result.stat_info
            next_column = super()._write_stat_info_row(worksheet, cell_format, percent_cell_format, idx + 3,
                                                       next_column, npu_stat_info)
            next_column = cls._write_absolute_check_row(worksheet, cell_format, percent_cell_format, idx + 3,
                                                        next_column,
                                                        result)
            worksheet.write(idx + 3, next_column, result.get_check_result_str(), cell_format)
            worksheet.write(idx + 3, next_column + 1, result.get_result_msg(), cell_format)

    @classmethod
    def write_op_test_summary(cls, workbook, worksheet, result_list: List[AbsoluteThreshCheckResult]):
        OP_LOGGER.info("start write_op_test_summary for AbsoluteThreshStandard, case count: %d" % len(result_list))
        if len(result_list) > 0:
            data_type = result_list[0].precision_check_standard.data_type
        else:
            data_type = "float32"
        ulp_limit = result_list[0].precision_check_standard.ulp_limit
        cls._write_title(workbook, worksheet, data_type, ulp_limit)
        cls._write_rows(workbook, worksheet, result_list)


class AbsoluteThreshXlsxSheetWriter2(XlsxSheetWriter):
    @classmethod
    def _write_title(cls, workbook, worksheet, data_type, ulp_limit):
        title_format_one = workbook.add_format({
            'bold': True,
            'align': 'center',
            'border': 1,
            'text_wrap': 1,
            'valign': 'vcenter',
            'fg_color': '#00b0f0'
        })
        title_format_two = workbook.add_format({
            'bold': True,
            'align': 'center',
            'border': 1,
            'text_wrap': 1,
            'valign': 'vcenter',
            'fg_color': '#92d050'
        })
        next_column = super()._add_case_info_title(worksheet, title_format_one, start_column=0, title_height=3)
        merge_range_str = "%s1:%s3" % (super()._column_name(next_column), super()._column_name(next_column))
        worksheet.merge_range(merge_range_str, "真值\n均值", title_format_one)
        next_column += 1

        next_column = super()._write_stat_info_title(worksheet, title_format_two, next_column, data_type, ulp_limit, "NPU")

        compare_title = ["统计\n数量", "PASS\n占比", "ERROR\n占比"]
        worksheet.merge_range(
            "%s1:%s1" % (super()._column_name(next_column), super()._column_name(next_column + len(compare_title) - 1)),
            "精度检查情况",
            title_format_one)
        for i, o_t in enumerate(compare_title):
            worksheet.set_column(
                "%s:%s" % (super()._column_name(next_column + i), super()._column_name(next_column + i)), 5)
            worksheet.merge_range(
                "%s2:%s3" % (super()._column_name(next_column + i), super()._column_name(next_column + i)),
                o_t,
                title_format_one)
        next_column += len(compare_title)
        worksheet.merge_range("%s1:%s3" % (super()._column_name(next_column), super()._column_name(next_column)),
                              "测试结果",
                              title_format_two)
        range_name = "%s1:%s3" % (super()._column_name(next_column + 1), super()._column_name(next_column + 1))
        worksheet.merge_range(range_name, "错误信息", title_format_one)

    @classmethod
    def _write_absolute_check_row(cls, worksheet, row_format, percent_cell_format, row_num, column_num,
                                  case_result: AbsoluteThreshCheckResult2):
        # ["统计\n数量", "PASS\n占比", "ERROR\n占比"]
        worksheet.write(row_num, column_num, case_result.total_cnt, row_format)
        worksheet.write(row_num, column_num + 1, case_result.total_pass_ratio, percent_cell_format)
        worksheet.write(row_num, column_num + 2, case_result.total_err_ratio, percent_cell_format)
        return column_num + 3

    @classmethod
    def _write_rows(cls, workbook, worksheet, result_list: List[AbsoluteThreshCheckResult2]):
        cell_format = workbook.add_format({
            'align': 'center',
            'border': 1,
            'valign': 'vcenter'
        })
        percent_cell_format = workbook.add_format({
            'align': 'center',
            'border': 1,
            'valign': 'vcenter'
        })
        percent_cell_format.set_num_format("0.00%")
        for idx, result in enumerate(result_list):
            if not result.run_success:
                next_column = super()._write_case_info_row(worksheet, cell_format, idx + 3, 0, result)
                worksheet.write(idx + 3, next_column + 1, result.run_err_msg, cell_format)
                continue
            next_column = super()._write_case_info_row(worksheet, cell_format, idx + 3, 0, result)
            worksheet.write(idx + 3, next_column, result.golden_avg, cell_format)
            next_column += 1
            npu_stat_info = result.stat_info
            next_column = super()._write_stat_info_row(worksheet, cell_format, percent_cell_format, idx + 3,
                                                       next_column, npu_stat_info)
            next_column = cls._write_absolute_check_row(worksheet, cell_format, percent_cell_format, idx + 3,
                                                        next_column,
                                                        result)
            worksheet.write(idx + 3, next_column, result.get_check_result_str(), cell_format)
            worksheet.write(idx + 3, next_column + 1, result.get_result_msg(), cell_format)

    @classmethod
    def write_op_test_summary(cls, workbook, worksheet, result_list: List[AbsoluteThreshCheckResult2]):
        OP_LOGGER.info("start write_op_test_summary for AbsoluteThreshStandard2, %s_%s case count: %d" %
                       (result_list[0].case_info.op_type, worksheet.name, len(result_list)))
        if len(result_list) > 0:
            data_type = result_list[0].precision_check_standard.data_type
        else:
            data_type = "float32"
        ulp_limit = result_list[0].precision_check_standard.ulp_limit
        cls._write_title(workbook, worksheet, data_type, ulp_limit)
        cls._write_rows(workbook, worksheet, result_list)


class PrecisionCheckResultReporter:
    @classmethod
    def save_to_xlsx(cls, op_precision_test_summary: OpPrecisionTestSummary, xlsx_path: str):
        OP_LOGGER.info("Start save_to_xlsx, op type: %s" % op_precision_test_summary.op_type)
        workbook = xlsxwriter.Workbook(xlsx_path, {'nan_inf_to_errors': True})
        sheet_names = []
        OP_LOGGER.info("op type: %s has %d summary." % (
            op_precision_test_summary.op_type, len(op_precision_test_summary.dtype_standard_grouped.keys())))
        for k, v in op_precision_test_summary.dtype_standard_grouped.items():
            standard = v[0].precision_check_standard
            sheet_name = standard.data_type + "_" + standard.sheet_name
            i = 1
            tmp_name = sheet_name
            while tmp_name in sheet_names:
                tmp_name = sheet_name + ("_%d" % (i,))
                i += 1
            sheet_name = tmp_name
            worksheet = workbook.add_worksheet(sheet_name)
            if standard.standard_name == GPUCompareStandard.NAME:
                GpuCompareXlsxSheetWriter.write_op_test_summary(workbook, worksheet, v)
            elif standard.standard_name == AbsoluteThreshStandard.NAME:
                AbsoluteThreshXlsxSheetWriter.write_op_test_summary(workbook, worksheet, v)
            elif standard.standard_name == AbsoluteThreshStandard2.NAME:
                AbsoluteThreshXlsxSheetWriter2.write_op_test_summary(workbook, worksheet, v)
            elif standard.standard_name == BinaryCompareStandard.NAME:
                BinaryCompareXlsxSheetWriter.write_op_test_summary(workbook, worksheet, v)
            elif standard.standard_name == BenchmarkCompareStandard.NAME:
                BenchmarkCompareXlsxSheetWriter.write_op_test_summary(workbook, worksheet, v)
            elif standard.standard_name == ULPBenchmarkCompareStandard.NAME:
                BenchmarkCompareXlsxSheetWriter.write_op_test_summary(workbook, worksheet, v)
            else:
                OP_LOGGER.error("Unknown standard: %s" % standard.standard_name)

        workbook.close()


def test_analyse():
    # data_dir = "/home/allan/op_test_data"
    data_dir = "./"
    case_files = os.listdir(os.path.join(data_dir, "bmm_bin", "npu_output"))
    case_names = []
    for tmp_name in case_files:
        if "_bin_" in tmp_name:
            case_names.append(tmp_name[:len("pangu_bmm_loss_0001_01_")])
    # case_names = ["pangu_bmm_loss_0002_08_", ]
    # case_name = "pangu_bmm_loss_0001_01_"
    case_names = case_names[:3]
    case_info_and_check_results = []
    summary = OpPrecisionTestSummary("BatchMatMul")
    for case_name in case_names:
        print("case name:", case_name)
        npu_file = os.path.join(data_dir, "bmm_bin", "npu_output", "%sbin_output_0.bin" % case_name)
        gpu_file = os.path.join(data_dir, "bmm_bin", "gpu_output", "%sgpu.bin" % case_name)
        golden_file = os.path.join(data_dir, "bmm_bin", "cpu_output", "%sgolden_cpu_0.bin" % case_name)
        start = time.perf_counter()
        npu_res = np.fromfile(npu_file, dtype=np.float16)
        gpu_res = np.fromfile(gpu_file, dtype=np.float16)
        golden = np.fromfile(golden_file, dtype=np.float32)
        npu_res = npu_res[:10000]
        gpu_res = gpu_res[:10000]
        golden = golden[:10000]
        print("test dtype: %s shape size: %d(M), read data cost time(s): %d" % (
            "fp16", npu_res.size / 1024 / 1024, time.perf_counter() - start))
        start = time.perf_counter()

        case_info = TestCaseInfo("BatchMatMul", case_name, "", "")
        standard = PrecisionCheckStandardFactory.build_standard(GPUCompareStandard.NAME, "float16")
        check_result = PRECISION_CHECKER.check(standard, case_info, npu_res, golden, gpu_res=gpu_res)
        summary.add_case_result(check_result)
        print("test dtype: %s shape size: %d(M), precision analyse cost time(s): %d" % (
            "fp16", npu_res.size / 1024 / 1024, time.perf_counter() - start))

    PrecisionCheckResultReporter.save_to_xlsx(summary, "./BatchMatMul.xlsx")


class TaskInfo:
    def __init__(self, npu_file, golden_file):
        self.npu_file = npu_file
        self.golden_file = golden_file
        self.benchmark_file = None


class TaskRunner:
    def __init__(self):
        self.standard_runner = {}

    def registry(self, standard_cls):
        def _out_wrapper(func):
            if isinstance(standard_cls, (tuple, list)):
                for _standard_cls in standard_cls:
                    self.standard_runner[_standard_cls.NAME] = func
            else:
                self.standard_runner[standard_cls.NAME] = func

            @wraps(func)
            def _in_wrapper(*kargs, **kwargs):
                return func(*kargs, **kwargs)

            return _in_wrapper

        return _out_wrapper

    def run(self, standard: PrecisionCheckStandard, case_info: TestCaseInfo,
            task_info: TaskInfo, **kwargs) -> PrecisionCheckResult:
        if standard.standard_name not in self.standard_runner.keys():
            result = PrecisionCheckResult(case_info, precision_check_standard=standard)
            result.run_err("This standard has not runner, standard name: %s" % (standard.standard_name,))
            return result
        return self.standard_runner[standard.standard_name](standard, case_info, task_info, **kwargs)


TASK_RUNNER = TaskRunner()


@TASK_RUNNER.registry(BinaryCompareStandard)
def binary_compare_task(standard: BinaryCompareStandard, case_info: TestCaseInfo,
                        task_config: TaskInfo) -> BinaryCompareCheckResult:
    OP_LOGGER.info("start to run case: %s, standard: BinaryCompareStandard" % case_info.case_name)
    result = BinaryCompareCheckResult(case_info, standard)
    check_success = True
    check_msg = ""
    if task_config.npu_file is None:
        check_success = False
        check_msg += "未找到NPU的数据文件，文件路径为空，请检查运行配置。"
    if not os.path.exists(task_config.npu_file):
        check_success = False
        check_msg += "未找到NPU的数据文件，文件路径不存在，请检查运行配置。路径为：%s" % task_config.npu_file
    if task_config.golden_file is None:
        check_success = False
        check_msg += "未找到Golden的数据文件，文件路径为空，请检查运行配置。"
    if not os.path.exists(task_config.golden_file):
        check_success = False
        check_msg += "未找到Golden的数据文件，文件路径不存在，请检查运行配置。路径为：%s" % task_config.golden_file
    if not check_success:
        result.run_err(check_msg)
        OP_LOGGER.info(
            "finish run case: %s, data file is none or not exist, standard: BinaryCompareStandard" % case_info.case_name)
        return result

    fix_bfloat16 = standard.data_type if standard.data_type != "bfloat16" else "uint16"
    npu_data = np.fromfile(task_config.npu_file, fix_bfloat16)
    golden_data = np.fromfile(task_config.golden_file, fix_bfloat16)

    if npu_data.size > golden_data.size:
        OP_LOGGER.warn("NPU data is larger than golden data, case name: %s, npu: %d, golden: %d" % (
            case_info.case_name, npu_data.size, golden_data.size))
        npu_data = npu_data[:golden_data.size]
    OP_LOGGER.info("finish run case: %s, standard: BinaryCompareStandard" % case_info.case_name)
    return PRECISION_CHECKER.check(standard, case_info, npu_data, golden_data)


@TASK_RUNNER.registry(AbsoluteThreshStandard)
def absolute_thresh_task(standard: AbsoluteThreshStandard, case_info: TestCaseInfo,
                         task_config: TaskInfo) -> AbsoluteThreshCheckResult:
    OP_LOGGER.info("start to run case: %s, standard: AbsoluteThreshStandard" % case_info.case_name)
    result = AbsoluteThreshCheckResult(case_info, standard)
    check_success = True
    check_msg = ""
    if task_config.npu_file is None:
        check_success = False
        check_msg += "未找到NPU的数据文件，文件路径为空，请检查运行配置。"
    if not os.path.exists(task_config.npu_file):
        check_success = False
        check_msg += "未找到NPU的数据文件，文件路径不存在，请检查运行配置。路径为：%s" % task_config.npu_file
    if task_config.golden_file is None:
        check_success = False
        check_msg += "未找到Golden的数据文件，文件路径为空，请检查运行配置。"
    if not os.path.exists(task_config.golden_file):
        check_success = False
        check_msg += "未找到Golden的数据文件，文件路径不存在，请检查运行配置。路径为：%s" % task_config.golden_file
    if not check_success:
        result.run_err(check_msg)
        OP_LOGGER.info(
            "finish run case: %s, data file is none or not exist, standard: AbsoluteThreshStandard" % case_info.case_name)
        return result
    fix_bf16_datype = standard.data_type if standard.data_type != "bfloat16" else "uint16"
    npu_data = np.fromfile(task_config.npu_file, fix_bf16_datype)
    golden_data = np.fromfile(task_config.golden_file, standard.golden_dtype)
    if npu_data.size > golden_data.size:
        OP_LOGGER.warn("NPU data is larger than golden data, case name: %s, npu: %d, golden: %d" % (
            case_info.case_name, npu_data.size, golden_data.size))
        npu_data = npu_data[:golden_data.size]
    OP_LOGGER.info("finish run case: %s, standard: AbsoluteThreshStandard" % case_info.case_name)
    return PRECISION_CHECKER.check(standard, case_info, npu_data, golden_data)


@TASK_RUNNER.registry(AbsoluteThreshStandard2)
def absolute_thresh_task(standard: AbsoluteThreshStandard2, case_info: TestCaseInfo,
                         task_config: TaskInfo) -> AbsoluteThreshCheckResult2:
    OP_LOGGER.info("start to run case: %s, standard: AbsoluteThreshStandard2" % case_info.case_name)
    result = AbsoluteThreshCheckResult2(case_info, standard)
    check_success = True
    check_msg = ""
    if task_config.npu_file is None:
        check_success = False
        check_msg += "未找到NPU的数据文件，文件路径为空，请检查运行配置。"
    if not os.path.exists(task_config.npu_file):
        check_success = False
        check_msg += "未找到NPU的数据文件，文件路径不存在，请检查运行配置。路径为：%s" % task_config.npu_file
    if task_config.golden_file is None:
        check_success = False
        check_msg += "未找到Golden的数据文件，文件路径为空，请检查运行配置。"
    if not os.path.exists(task_config.golden_file):
        check_success = False
        check_msg += "未找到Golden的数据文件，文件路径不存在，请检查运行配置。路径为：%s" % task_config.golden_file
    if not check_success:
        result.run_err(check_msg)
        OP_LOGGER.info(
            "finish run case: %s, data file is none or not exist, standard: AbsoluteThreshStandard2" % case_info.case_name)
        return result
    fix_bf16_datype = standard.data_type if standard.data_type != "bfloat16" else "uint16"
    npu_data = np.fromfile(task_config.npu_file, fix_bf16_datype)
    golden_data = np.fromfile(task_config.golden_file, standard.golden_dtype)
    if npu_data.size > golden_data.size:
        OP_LOGGER.warn("NPU data is larger than golden data, case name: %s, npu: %d, golden: %d" % (
            case_info.case_name, npu_data.size, golden_data.size))
        npu_data = npu_data[:golden_data.size]
    OP_LOGGER.info("finish run case: %s, standard: AbsoluteThreshStandard2" % case_info.case_name)
    return PRECISION_CHECKER.check(standard, case_info, npu_data, golden_data)


@TASK_RUNNER.registry([BenchmarkCompareStandard, ULPBenchmarkCompareStandard])
def benchmark_compare_task(standard: [BenchmarkCompareStandard, ULPBenchmarkCompareStandard], case_info: TestCaseInfo,
                           task_config: TaskInfo) -> Union[BenchmarkCompareCheckResult, ULPBenchmarkCompareCheckResult]:
    OP_LOGGER.info("start to run case: %s, standard: BenchmarkCompareStandard" % case_info.case_name)
    result = BenchmarkCompareCheckResult(case_info, standard)
    check_success = True
    check_msg = ""
    if task_config.npu_file is None:
        check_success = False
        check_msg += "未找到NPU的数据文件，文件路径为空，请检查运行配置。"
    if not os.path.exists(task_config.npu_file):
        check_success = False
        check_msg += "未找到NPU的数据文件，文件路径不存在，请检查运行配置。路径为：%s" % task_config.npu_file
    if task_config.golden_file is None:
        check_success = False
        check_msg += "未找到Golden的数据文件，文件路径为空，请检查运行配置。"
    if not os.path.exists(task_config.golden_file):
        check_success = False
        check_msg += "未找到Golden的数据文件，文件路径不存在，请检查运行配置。路径为：%s" % task_config.golden_file
    if task_config.benchmark_file is None:
        check_success = False
        check_msg += "未找到标杆的数据文件，文件路径为空，请检查运行配置。"
    if not os.path.exists(task_config.benchmark_file):
        check_success = False
        check_msg += "未找到标杆的数据文件，文件路径不存在，请检查运行配置。路径为：%s" % task_config.benchmark_file
    if not check_success:
        result.run_err(check_msg)
        OP_LOGGER.info(
            "finish run case: %s, data file is none or not exist,standard: BenchmarkCompareStandard" % case_info.case_name)
        return result
    npu_data = np.fromfile(task_config.npu_file,
                           standard.data_type if standard.data_type != "bfloat16" else "uint16")
    golden_data = np.fromfile(task_config.golden_file, standard.golden_dtype)
    benchmark_data = np.fromfile(task_config.benchmark_file,
                                 standard.data_type if standard.data_type != "bfloat16" else "uint16")
    OP_LOGGER.info("finish read case data file, case: %s" % case_info.case_name)
    if npu_data.size > golden_data.size:
        OP_LOGGER.warn("NPU data is larger than golden data, case name: %s, npu: %d, golden: %d" % (
            case_info.case_name, npu_data.size, golden_data.size))
        npu_data = npu_data[:golden_data.size]
    if npu_data.size < golden_data.size or benchmark_data.size < golden_data.size:
        OP_LOGGER.error(
            "golden data size is larger than npu or benchmark, case name: %s, npu: %d, golden: %d, benchmark: %d" % (
                case_info.case_name, npu_data.size, golden_data.size, benchmark_data.size))
        check_msg += "数据文件异常， npu file: %s, size: %d; golden file: %s, size: %d; benchmark file: %s, size: %d" % (
            task_config.npu_file, npu_data.size, task_config.golden_file, golden_data.size,
            task_config.benchmark_file, benchmark_data.size
        )
        result.run_err(check_msg)
        OP_LOGGER.info(
            "finish run case: %s, data file size error， standard: BenchmarkCompareStandard" % case_info.case_name)
        return result
    result = PRECISION_CHECKER.check(standard, case_info, npu_data, golden_data, benchmark_res=benchmark_data)
    OP_LOGGER.info("finish run case: %s, standard: BenchmarkCompareStandard" % case_info.case_name)
    return result


class MultiProcessRunConfig:
    def __init__(self):
        self.max_use_memory = 0
        self.process_num = 2
        self.case_root_path = ""
        self.test_result_save_path = os.path.relpath("./")
        self.precision_standard_config_path = os.path.relpath("./op_precision_standard.yml")


class PrecisionCheckTask:
    STATUS_SUCCESS = 0
    STATUS_FAILED = 1
    TASK_TYPE_RUN = 0
    TASK_TYPE_SHUTDOWN = 1

    def __init__(self,
                 task_type: int,
                 case_info: Union[TestCaseInfo, None],
                 standard: Union[PrecisionCheckStandard, None],
                 task_info: Union[TaskInfo, None],
                 use_memory: int):
        self.task_type = task_type
        self.case_info: TestCaseInfo = case_info

        self.case_info = case_info
        self.standard = standard
        self.task_info = task_info
        self.use_memory = use_memory

        self.run_result: Union[PrecisionCheckResult, None] = None
        self.run_status: int = PrecisionCheckTask.STATUS_SUCCESS
        self.run_msg: Union[str, None] = None

    def update_result(self, run_result: Union[PrecisionCheckResult, None],
                      run_status: int = 0, run_msg: Union[str, None] = None):
        self.run_result = run_result
        self.run_status = run_status
        self.run_msg = run_msg


def task_consumer_func(task_queue: multiprocessing.Queue,
                       task_queue_done: multiprocessing.Queue,
                       run_config: MultiProcessRunConfig,
                       process_name: str = None):
    process_name = "Task Consumer" if process_name is None else process_name
    OP_LOGGER.set_process_name(process_name)
    while True:
        try:
            OP_LOGGER.info("%s start get task." % process_name)
            task: PrecisionCheckTask = task_queue.get()
            OP_LOGGER.info("[%s] get a task, task type: %s" % (process_name, task.task_type))
            task_type = task.task_type
            if task_type == PrecisionCheckTask.TASK_TYPE_RUN:
                case_info = task.case_info
                standard = task.standard
                OP_LOGGER.info("[%s] start run task, case_name: %s" % (process_name, case_info.case_name))
                result = TASK_RUNNER.run(standard, case_info, task.task_info)
                task.update_result(result)
                task_queue_done.put_nowait(task)
                continue
            if task_type == PrecisionCheckTask.TASK_TYPE_SHUTDOWN:
                task.update_result(None, PrecisionCheckTask.STATUS_SUCCESS)
                task_queue_done.put_nowait(task)
                break
        except Exception as err:
            err_msg = traceback.format_exc()
            traceback.print_exc()
            task.update_result(None, PrecisionCheckTask.STATUS_FAILED, err_msg)
            task_queue_done.put_nowait(task)


class CaseInfoReaderAndTaskBuilder:
    @classmethod
    def _read_case_infos(cls, root_path: str):
        root_path = os.path.realpath(root_path)
        op_case_path_map: Dict[str, List] = defaultdict(list)
        for root, dirs, files in os.walk(root_path):
            csv_flag = False
            op_dir_root_flag = False
            for file_name in files:
                if file_name.endswith(".csv"):
                    csv_flag = True
                if file_name.endswith("ulp.csv"):
                    op_dir_root_flag = True
                    break
            if csv_flag and not op_dir_root_flag:
                print("[WARNING] find csv but not found gpu_output_gpu_npu.csv")
            if op_dir_root_flag:
                op_type = os.path.basename(root)
                op_case_path_map[op_type].append(root)
        return op_case_path_map

    @classmethod
    def _parse_shape_str(cls, shape_str: str) -> List[List[int]]:
        shape_tuple = eval(shape_str)
        shape_list = [list(shape) if isinstance(shape, (tuple, list)) else shape for shape in shape_tuple]
        return shape_list

    @classmethod
    def _parse_output_dtype_str(cls, dtype_str: str):
        dtype_str = dtype_str.strip()
        dtype_str = dtype_str[1:-1]
        dtype_str_split = dtype_str.split(",")
        dtype_list = []
        for dtype_str in dtype_str_split:
            dtype_str = dtype_str.strip()
            if not dtype_str:
                continue
            dtype_list.append(dtype_str[1:-1])
        return dtype_list

    @classmethod
    def _use_memory_calc(cls, shape: List[int], dtype: str, num_size: int):
        dtype_size = 8
        if dtype in ["bool", "int8", "uint8"]:
            dtype_size = 1
        if dtype in ["float16", "int16", "uint16", "bfloat16"]:
            dtype_size = 2
        if dtype in ["float32", "int32", "uint32", "complex32"]:
            dtype_size = 4
        if dtype in ["float64", "int64", "uint64", "complex64"]:
            dtype_size = 8
        if dtype in ["complex128"]:
            dtype_size = 16

        if len(shape) == 0:
            return dtype_size
        return reduce(lambda x, y: x * y, shape) * dtype_size * num_size  # 用4倍内存

    @classmethod
    def _get_file_paths(cls, op_dir, case_name, total_output_cnt, cur_cnt,bin_name):
        print("1111",op_dir)
        npu_file_name = "output.bin"
        npu_file_path = os.path.join(op_dir, case_name,"output",str(cur_cnt), npu_file_name)
        print("----------------npu_file_path----:",npu_file_path)

        # if not os.path.exists(npu_file_path):
        #     npu_file_name = "%s_cst_output_%d.bin" % (case_name, cur_cnt)
        #     npu_file_path = os.path.join(op_dir, "npu_bin", npu_file_name)
        #     if not os.path.exists(npu_file_path):
        #         npu_file_name = "%s_stc_output_%d.bin" % (case_name, cur_cnt)
        #         npu_file_path = os.path.join(op_dir, "npu_bin", npu_file_name)
        gpu_file_name = "output_hccl.bin"
        gpu_file_path = os.path.join(op_dir, case_name, "output", str(cur_cnt), gpu_file_name)
        print("----------------gpu_file_path----:",gpu_file_path)

        # if total_output_cnt == 1 and not os.path.exists(gpu_file_path):
        #     gpu_file_name = "%s_gpu.bin" % case_name
        #     gpu_file_path = os.path.join(op_dir, "gpu_bin", gpu_file_name)

        golden_file_name = "output_cpu.bin"
        golden_file_path =os.path.join(op_dir, case_name, "output", str(cur_cnt), golden_file_name)
        print(npu_file_path, gpu_file_path, golden_file_path)
        print("----------------golden_file_path----:",golden_file_path)
        # if not os.path.exists(golden_file_path):
        #     golden_file_name = "%s_golden_%d.bin" % (case_name, cur_cnt)
        #     golden_file_path = os.path.join(op_dir, "cpu_bin", golden_file_name)
        print(npu_file_path, golden_file_path, gpu_file_path)
        return npu_file_path, golden_file_path, gpu_file_path

    @classmethod
    def _read_case_csv(cls, op_type, csv_file_dir: str, standard_reader: StandardReader, max_use_memory: int,
                       testcase_name:str) -> List[PrecisionCheckTask]:
        csv_file = os.path.join(csv_file_dir, "%s.csv" % ("MC2",))
        print(csv_file)
        op_type="MC2"
        task_obj_list: List[PrecisionCheckTask] = []
        testcase_name = testcase_name.split(",")
        with open(csv_file,encoding='utf-8-sig') as f:
            row_idx = 0
            for row in csv.reader(f):
                if row_idx == 0:
                    print(row)
                    output_dtype_idx = row.index("output_dtypes")
                    output_shape_idx = row.index("stc_outputs")
                    # input_shape_idx = row.index("stc_inputs")
                    # other_params_idx = row.index("other_runtime_params")
                    data_range_idx = row.index("input_data_ranges")
                    case_name_idx = row.index("testcase_name")
                    bin_name_idx = row.index("bin_name")
                    manual_bins_idx = row.index("manual_bins") if "manual_bins" in row else None
                    row_idx += 1
                    continue
                case_name = row[case_name_idx]
                bin_name=row[bin_name_idx]
                if any(testcase_name):
                    if case_name not in testcase_name:
                        continue
                ori_case_param_info = '%s' % ( row[output_shape_idx])
                case_input_value_range_info = row[data_range_idx]
                output_shape_str = row[output_shape_idx]
                output_dtype_str = row[output_dtype_idx]
                output_shapes = cls._parse_shape_str(output_shape_str)
                output_dtypes = cls._parse_shape_str(output_dtype_str)
                if manual_bins_idx is not None:
                    manual_bins = row[manual_bins_idx]
                    if manual_bins:
                        try:
                            manual_bins = eval(manual_bins)
                        except:
                            OP_LOGGER.info("[manual_bins] should be list/tuple")
                else:
                    manual_bins = None
                if len(output_dtypes) == 1 and len(output_shapes) > 1:
                    output_dtypes = output_dtypes * len(output_shapes)
                total_out_cnt = len(output_dtypes)
                out_idx = 0
                for output_shape, output_dtype in zip(output_shapes, output_dtypes):
                    if output_shape is None:
                        out_idx += 1
                        continue
                    case_param_info = "output%s %s" % (out_idx, ori_case_param_info)
                    case_info = TestCaseInfo(op_type, case_name, case_param_info=case_param_info,
                                             case_input_value_range_info=case_input_value_range_info)
                    standard = standard_reader.read_op_stand(op_type, output_dtype)
                    print(op_type,output_dtype,standard)
                    if standard is None:
                        OP_LOGGER.error("standard is None, op_type: %s, data type: %s" % (op_type, output_dtype))
                    else:
                        if standard.standard_name == "BinaryCompareStandard":
                            num_size = 2
                        else:
                            num_size = 40
                        use_memory = cls._use_memory_calc(output_shape, output_dtype, num_size)
                        if use_memory > max_use_memory:
                            OP_LOGGER.error("ID: %s size out of memory, case Memory: %s, max_use_memory: %s" % (
                                case_name, use_memory, max_use_memory))
                            continue
                        if isinstance(manual_bins, (tuple, list)) and len(manual_bins) > 0:
                            if isinstance(manual_bins[0], str):
                                npu_file, golden_file, gpu_file = manual_bins
                            elif isinstance(manual_bins[0], (tuple, list)):
                                npu_file, golden_file, gpu_file = manual_bins[out_idx]
                            else:
                                OP_LOGGER.info("[manual_bins[%s] should be str/list/tuple" % out_idx)
                        else:
                            npu_file, golden_file, gpu_file = cls._get_file_paths(csv_file_dir, case_name, total_out_cnt,
                                                                                  out_idx,bin_name)
                        task_info = TaskInfo(npu_file, golden_file)
                        task_info.benchmark_file = gpu_file
                        op_task = PrecisionCheckTask(PrecisionCheckTask.TASK_TYPE_RUN, case_info, standard, task_info,
                                                     use_memory)
                        task_obj_list.append(op_task)
                    out_idx += 1
                row_idx += 1
        return task_obj_list

    @classmethod
    def _build_tasks(cls, op_case_path_map: Dict[str, List[str]], standard_reader: StandardReader,
                     max_use_memory: int, testcase_name: str) -> Dict[str, List[PrecisionCheckTask]]:
        op_tasks: Dict[str, List[PrecisionCheckTask]] = defaultdict(list)
        for op_type, op_path_list in op_case_path_map.items():
            op_type="MC2"
            for op_path in op_path_list:
                task_list = cls._read_case_csv(op_type, op_path, standard_reader, max_use_memory, testcase_name)
                total_tasks = op_tasks.get(op_type, [])
                op_tasks[op_type] = total_tasks
                total_tasks.extend(task_list)
        return op_tasks

    @classmethod
    def scan_path_and_build_tasks(cls, case_root_path: str, standard_yml_path: str,
                                  max_use_memory: int, testcase_name: str):
        print("start to scan %s for case info." % case_root_path)
        case_infos = cls._read_case_infos(case_root_path)
        standard_reader = StandardReader(standard_yml_path)
        op_tasks = cls._build_tasks(case_infos, standard_reader, max_use_memory, testcase_name)
        print("end scan %s for case info. total case: %d" % (case_root_path, len(op_tasks)))
        return op_tasks


def task_producer_func(task_queue: multiprocessing.Queue,
                       task_queue_done: multiprocessing.Queue,
                       run_config: MultiProcessRunConfig):
    op_tasks = CaseInfoReaderAndTaskBuilder.scan_path_and_build_tasks(run_config.case_root_path,
                                                                      run_config.precision_standard_config_path,
                                                                      run_config.max_use_memory,
                                                                      run_config.testcase_name)
    max_memory = run_config.max_use_memory
    used_memory = 0
    produce_task_cnt = 0
    total_task_cnt = 0
    op_test_summarys: Dict[str, OpPrecisionTestSummary] = {}
    for op_type, op_tasks in op_tasks.items():
        op_test_summarys[op_type] = OpPrecisionTestSummary(op_type)
        for op_task in op_tasks:
            used_memory += op_task.use_memory
            while used_memory > max_memory:
                OP_LOGGER.info("[Producer] memory poor")
                finish_task: PrecisionCheckTask = task_queue_done.get()
                OP_LOGGER.info("[Producer] finish task, rest task: %d" % produce_task_cnt)
                if finish_task.run_status == PrecisionCheckTask.STATUS_SUCCESS:
                    if finish_task.run_result:
                        OP_LOGGER.info("[Producer] add case result.")
                        op_test_summarys[finish_task.run_result.case_info.op_type].add_case_result(
                            finish_task.run_result)
                    else:
                        OP_LOGGER.info("[Producer] task run success, but run result is None.")
                else:
                    OP_LOGGER.info("[Producer] finish task, but run failed.")
                produce_task_cnt -= 1
                used_memory -= finish_task.use_memory
            OP_LOGGER.info("[Producer] put task to queue: %s, case name: %s" % (op_type, op_task.case_info.case_name))
            task_queue.put_nowait(op_task)
            produce_task_cnt += 1
            total_task_cnt += 1

    OP_LOGGER.info("[Producer] all task produced, task num(%d)" % total_task_cnt)
    while produce_task_cnt > 0:
        try:
            done_task: PrecisionCheckTask = task_queue_done.get()
            if done_task.run_status == PrecisionCheckTask.STATUS_SUCCESS:
                if done_task.run_result:
                    OP_LOGGER.info("[Producer] add case result.")
                    op_test_summarys[done_task.run_result.case_info.op_type].add_case_result(done_task.run_result)
                else:
                    OP_LOGGER.info("[Producer] task run success, but run result is None.")
            else:
                OP_LOGGER.info("[Producer] finish task, but run failed.")
            produce_task_cnt -= 1
            OP_LOGGER.info("[Producer] finish task, rest task: %d" % produce_task_cnt)
        except TimeoutError as err:
            OP_LOGGER.info("[Producer] get finish task info timeout.")
            traceback.print_exc()
            produce_task_cnt = 0
        except Exception as err:
            traceback.print_exc()
            produce_task_cnt = 0

    OP_LOGGER.info("[Producer] complete all task(%d)" % total_task_cnt)

    for op_type, op_summary in op_test_summarys.items():
        op_summary.do_summary()
        result_name = op_type + "_" + run_config.result_label if run_config.result_label else op_type
        save_path = os.path.join(run_config.test_result_save_path, "%s.xlsx" % result_name)
        PrecisionCheckResultReporter.save_to_xlsx(op_summary, save_path)

    OP_LOGGER.info("[Producer] queue size: %d" % (0 if task_queue.empty() else 1,))
    for i in range(run_config.process_num - 1):
        task_obj = PrecisionCheckTask(PrecisionCheckTask.TASK_TYPE_SHUTDOWN, None, None, None, 0)
        task_queue.put_nowait(task_obj)
        done_task = task_queue_done.get()
    OP_LOGGER.info("[Producer] all customer closed")


def get_golden_dtype(data_type):
    if data_type == "float16":
        return "float32"
    if data_type == "float32":
        return "float64"
    if data_type == "bfloat16":
        return "float32"
    return data_type



def test_multi_process_analyse(data_path, yml_file, result_path, pc, label, testcase_name):
    run_config = MultiProcessRunConfig()
    run_config.process_num = pc
    mem = psutil.virtual_memory()
    run_config.max_use_memory = mem.total * 0.9
    run_config.case_root_path = data_path
    run_config.precision_standard_config_path = yml_file
    run_config.result_label = label
    run_config.testcase_name = testcase_name
    run_config.test_result_save_path = os.path.relpath(result_path)
    if not os.path.exists(run_config.test_result_save_path):
        os.mkdir(run_config.test_result_save_path)
    task_queue = multiprocessing.Queue()
    task_queue_done = multiprocessing.Queue()
    task_producer = multiprocessing.Process(target=task_producer_func, args=(task_queue, task_queue_done, run_config))

    task_producer.start()
    task_consumer_list = []
    for idx in range(run_config.process_num - 1):
        task_customer = multiprocessing.Process(target=task_consumer_func,
                                                args=(task_queue, task_queue_done, run_config, "Consumer_%d" % idx))

        task_customer.start()
        task_consumer_list.append(task_customer)
    task_producer.join()
    for task_customer in task_consumer_list:
        if task_customer.is_alive():
            task_customer.terminate()  # 超时后强制终止进程


FLAGS = flags.FLAGS
yml_mc2_path=os.path.join(os.getcwd(),"script","MC2.yml")
flags.DEFINE_string("data_path", None, "OPS_BIN data_path")
flags.DEFINE_string("yml_file", yml_mc2_path, "yml file")
flags.DEFINE_string("result_path", "test_result", "test_result")
flags.DEFINE_string("label", "", "result label")
flags.DEFINE_string("case", "", "testcase name")
flags.DEFINE_integer("pc", 8, "process count")

flags.mark_flag_as_required("data_path")



def main(argv):
    try:
        shutil.copy("MC2.csv", FLAGS.data_path)
    except Exception as e:
        print(e)
    del argv
    test_multi_process_analyse(FLAGS.data_path, FLAGS.yml_file, FLAGS.result_path, FLAGS.pc, FLAGS.label, FLAGS.case)


if __name__ == '__main__':
    app.run(main)
