import torch_npu
from .common import *
import copy
from op_class.aclnnMatmulAllReduce import MM_ALLREDUCE
from torch.distributed import ReduceOp
from mindspeed.ops.npu_mm_all_reduce_add_rms_norm import npu_mm_all_reduce_add_rms_norm
from mindspeed.ops.npu_mm_all_reduce_add_rms_norm_ import npu_mm_all_reduce_add_rms_norm_
from script import toolss
from .registry import class_registry


class MM_ALLREDUCE_ARN_GRAPH_Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2, hcom, reduce_op, bias, antiquant_scale, antiquant_offset, dequant_scale,
                antiquant_group_size, residual, gamma, epsilon, is_inplace, type, residual_hccl):
        if (dequant_scale != None) and (dequant_scale.dtype == torch.float32):
            dequant_scale = torch_npu.npu_trans_quant_param(dequant_scale)
        if x1.shape[-1] != x2.shape[0]:
            x2 = x2.t()
        if type == "mc2":
            if is_inplace:
                y, output = npu_mm_all_reduce_add_rms_norm_(x1=x1,
                                                            x2=x2,
                                                            hcom=hcom,
                                                            reduce_op=reduce_op,
                                                            bias=bias,
                                                            antiquant_scale=antiquant_scale,
                                                            antiquant_offset=antiquant_offset,
                                                            dequant_scale=dequant_scale,
                                                            antiquant_group_size=antiquant_group_size,
                                                            residual=residual,
                                                            gamma=gamma,
                                                            epsilon=epsilon
                                                            )
            else:
                y, output = npu_mm_all_reduce_add_rms_norm(x1=x1,
                                                           x2=x2,
                                                           hcom=hcom,
                                                           reduce_op=reduce_op,
                                                           bias=bias,
                                                           antiquant_scale=antiquant_scale,
                                                           antiquant_offset=antiquant_offset,
                                                           dequant_scale=dequant_scale,
                                                           antiquant_group_size=antiquant_group_size,
                                                           residual=residual,
                                                           gamma=gamma,
                                                           epsilon=epsilon
                                                           )
        elif type == "hccl_mm":
            mc2_output = torch_npu.npu_mm_all_reduce_base(x1=x1,
                                                          x2=x2,
                                                          hcom=hcom,
                                                          reduce_op=reduce_op,
                                                          bias=bias,
                                                          antiquant_scale=antiquant_scale,
                                                          antiquant_offset=antiquant_offset,
                                                          x3=None,
                                                          dequant_scale=dequant_scale,
                                                          antiquant_group_size=antiquant_group_size
                                                          )
            mc2_output = mc2_output.reshape(residual.shape)
            output, rstd, y = torch_npu.npu_add_rms_norm(mc2_output,
                                                         residual_hccl,
                                                         gamma,
                                                         epsilon)
        else:
            print("In allreduce_arn get_graph,wrong type.")
        return output


@class_registry("aclnnMatmulAllReduceAddRmsNorm", "aclnnQuantMatmulAllReduceAddRmsNorm",
                "aclnnWeightQuantMatmulAllReduceAddRmsNorm", "aclnnInplaceMatmulAllReduceAddRmsNorm",
                "aclnnInplaceQuantMatmulAllReduceAddRmsNorm", "aclnnInplaceWeightQuantMatmulAllReduceAddRmsNorm")
class MM_ALLREDUCE_ARN(MM_ALLREDUCE):
    def __init__(self, case, rank, dist=None, hcomm_info=None):
        super().__init__(case, rank, dist, hcomm_info)
        self.is_inplace = case['is_inplace'] if 'is_inplace' in case else None
        self.epsilon = case['epsilon'] if 'epsilon' in case else 0.000001
        self.case_name = case['case_name']

    def get_tmp_tensor(self):
        self.residual_list = []
        self.residual = func.get_tensor_bais(self.case, self.rank, name='residual')
        if "upper_boundary" in self.case_name:
            self.residual_hccl = self.residual
        else:
            self.residual_hccl = copy.deepcopy(self.residual)
            for i in range(10):
                residual_tmp = copy.deepcopy(self.residual)
                self.residual_list.append(residual_tmp)
        self.gamma = func.get_tensor_bais(self.case, self.rank, name='gamma')
        self.bias = func.get_tensor_bais(self.case, self.rank) if self.is_bias else None
        self.x3 = func.get_tensor_bais(self.case, self.rank, name='x3') if self.is_x3 else None
        self.antiquant_scale = func.get_tensor_bais(self.case, self.rank,
                                                    name='antiquant_scale') if self.is_antiquant_scale else None
        self.antiquant_offset = func.get_tensor_bais(self.case, self.rank,
                                                     name='antiquant_offset') if self.is_antiquant_offset else None
        self.antiquant_group_size = self.antiquant_group_size if self.is_group_size else 0
        # a8w8,dequant_scale=fp32要经过trans接口
        if self.is_dequant_scale:
            self.dequant_scale = func.get_tensor_bais(self.case, self.rank, name='dequant_scale')
            self.dequant_scale_graph = self.dequant_scale
            if self.dequant_scale.dtype == torch.float32:
                self.dequant_scale = torch_npu.npu_trans_quant_param(self.dequant_scale)
        else:
            self.dequant_scale = None
            self.dequant_scale_graph = self.dequant_scale
        # 图模式转置应在forward内
        self.weight_graph = self.weight
        if self.input_shape[-1] != self.weight_shape[0]:
            self.weight = self.weight.t()

    def get_hccl_mm(self):
        self.mc2_output = torch_npu.npu_mm_all_reduce_base(x1=self.input,
                                                           x2=self.weight,
                                                           hcom=self.hcomm_info,
                                                           reduce_op="sum",
                                                           bias=self.bias,
                                                           antiquant_scale=self.antiquant_scale,
                                                           antiquant_offset=self.antiquant_offset,
                                                           x3=self.x3,
                                                           dequant_scale=self.dequant_scale,
                                                           antiquant_group_size=self.antiquant_group_size
                                                           )
        self.mc2_output = self.mc2_output.reshape(self.case['residual_shape'])
        self.output_golden, self.rstd_golden, self.y_golden = torch_npu.npu_add_rms_norm(self.mc2_output, self.residual,
                                                                                         self.gamma,
                                                                                         self.epsilon)
        return self.output_golden, self.rstd_golden, self.y_golden

    def get_npu(self, i=0):
        if self.is_inplace:
            self.y_npu, self.output_npu = npu_mm_all_reduce_add_rms_norm_(x1=self.input,
                                                                          x2=self.weight,
                                                                          hcom=self.hcomm_info,
                                                                          reduce_op="sum",
                                                                          bias=self.bias,
                                                                          antiquant_scale=self.antiquant_scale,
                                                                          antiquant_offset=self.antiquant_offset,
                                                                          dequant_scale=self.dequant_scale,
                                                                          antiquant_group_size=self.antiquant_group_size,
                                                                          residual=self.residual_list[i],
                                                                          gamma=self.gamma,
                                                                          epsilon=self.epsilon
                                                                          )
        else:
            self.y_npu, self.output_npu = npu_mm_all_reduce_add_rms_norm(x1=self.input,
                                                                         x2=self.weight,
                                                                         hcom=self.hcomm_info,
                                                                         reduce_op="sum",
                                                                         bias=self.bias,
                                                                         antiquant_scale=self.antiquant_scale,
                                                                         antiquant_offset=self.antiquant_offset,
                                                                         dequant_scale=self.dequant_scale,
                                                                         antiquant_group_size=self.antiquant_group_size,
                                                                         residual=self.residual,
                                                                         gamma=self.gamma,
                                                                         epsilon=self.epsilon
                                                                         )
        return self.output_npu

    def compare(self, diffThd=0.005, pctThd=0.005, check_md5=0):
        np_output_npu = np.array(self.output_npu.to(torch.float32).cpu())
        np_output_golden = np.array(self.output_golden.to(torch.float32).cpu())
        res_output = toolss.data_compare_np(np_output_npu, np_output_golden, diff_thd=diffThd, pct_thd=pctThd)
        print('[INFO] device_{} output测试的精度结果为：{}'.format(self.rank, res_output))
        func.write_csv('mm_allreduce_ARN_result.csv', [self.case_name, self.rank, res_output])

        np_y_npu = np.array(self.y_npu.to(torch.float32).cpu())
        np_y_golden = np.array(self.y_golden.to(torch.float32).cpu())
        res_y = toolss.data_compare_np(np_y_npu, np_y_golden, diff_thd=diffThd, pct_thd=pctThd)
        print('[INFO] device_{} y测试的精度结果为：{}'.format(self.rank, res_y))
        func.write_csv('mm_allreduce_ARN_result.csv', [self.case_name, self.rank, res_y, 'y_out'])

    def get_graph(self, type="mc2"):
        model = MM_ALLREDUCE_ARN_GRAPH_Model()
        model = self.define_model(model, self.graph_type)
        y, output = model(x1=self.input,
                          x2=self.weight_graph,
                          hcom=self.hcomm_info,
                          reduce_op="sum",
                          bias=self.bias,
                          antiquant_scale=self.antiquant_scale,
                          antiquant_offset=self.antiquant_offset,
                          dequant_scale=self.dequant_scale_graph,
                          antiquant_group_size=self.antiquant_group_size,
                          residual=self.residual,
                          gamma=self.gamma,
                          epsilon=self.epsilon,
                          is_inplace=self.is_inplace,
                          type=type,
                          residual_hccl=self.residual_hccl,
                          )
        if type == "mc2":
            self.y_npu = y
            self.output_npu = output
        elif type == "hccl_mm":
            self.y_golden = y
            self.output_golden = output
        else:
            print("In allreduce_arn get_graph,wrong type..")
        return y, output

    def get_performace(self, pef_type='npu', soc='910B'):
        if pef_type == 'npu':
            path_tmp = 'mc2_'
        elif pef_type == 'golden':
            path_tmp = 'hccl_mm_'
        experimental_config, profiling_result = func.get_profiling_config(self.case['case_name'], path_tmp)
        with torch_npu.profiler.profile(
                schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=0),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profiling_result),
                profile_memory=True,
                with_stack=False,
                with_flops=False,
                with_modules=False,
                experimental_config=experimental_config,
        ) as prof:
            for i in range(10):
                if pef_type == 'npu':
                    self.get_npu(i)
                elif pef_type == 'golden':
                    self.get_hccl_mm()
        if self.rank == 0:
            op = self.case['op_name']
            csv_file = f'{path_tmp}{op}_pro_result.csv'
            if soc == '910C':
                *_, memory, performance = func.get_memory_performance(self.case, path_tmp, op, "910C")
            else:
                *_, memory, performance = func.get_memory_performance(self.case, path_tmp)
            func.write_csv(csv_file, [self.case_name, self.rank, memory, performance])
            print(f'[INFO] MC2融合算子 内存为：{memory} M，性能为：{performance} us \n')
