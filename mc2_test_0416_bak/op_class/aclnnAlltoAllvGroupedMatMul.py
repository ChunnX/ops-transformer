from .common import *
import torch
import random
from mindspeed.ops import gmm
import torch_npu
from script import toolss
from ml_dtypes import bfloat16
from .registry import class_registry

ACLNN_INDEX_MAP_ALLTOALL = {0: "out", 1: "mm_out", 2: "permute_out"}

# bin文件名
BIN_FILE_NAME_MAP = {
    "input": {
        "gmm_x": "gmm_x.bin",
        "gmm_weight": "gmm_weight.bin",
        "mm_x": "mm_x.bin",
        "mm_weight": "mm_weight.bin",
        "send_counts": "send_counts.bin",
        "recv_counts": "recv_counts.bin",
    },
    "output_cpu": {
        "gmm_y": "cpu_out.bin",
        "mm_y": "cpu_mm_out.bin",
        "permute_out": "cpu_permute_out.bin",
    },
    "output_hccl": {
        "gmm_y": "output_hccl.bin",
        "mm_y": "mm_out_hccl.bin",
        "permute_out": "permute_out_hccl.bin",
    },
    "output_npu": {
        "gmm_y": "npu_gmm_y.bin",
        "mm_y": "npu_mm_y.bin",
        "permute_out": "npu_permute_out.bin",
    },
}


def str_to_bool(s):
    if isinstance(s, bool):
        return s
    if isinstance(s, str):
        return s.strip().lower() in ("true", "1", "yes", "y")
    return False


class AlltoAllvGroupedMatmulGraphModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, gmm_x, gmm_weight, mm_x, mm_weight, hcom, ep_size, send_counts, recv_counts, send_counts_tensor,
        all_to_allv_out, recv_counts_tensor, trans_gmm_weight, trans_mm_weight, exp_token_nums, group_list,
        e, permute_out_flag, out_splits, input_split_sizes, rank_id, gmm_weight_after_trans, mm_weight_after_trans,
        type,
    ):
        gmm_y, mm_y, permute_out = None, None, None
        if type == "mc2":
            gmm_y, mm_y, permute_out = torch_npu.npu_alltoallv_gmm(gmm_x=gmm_x, gmm_weight=gmm_weight,
                hcom=hcom, ep_world_size=ep_size, send_counts=list(send_counts), recv_counts=list(recv_counts),
                send_counts_tensor=None, recv_counts_tensor=None, mm_x=mm_x, mm_weight=mm_weight,
                trans_gmm_weight=trans_gmm_weight, trans_mm_weight=trans_mm_weight, permute_out_flag=permute_out_flag,
            )
        return gmm_y, mm_y, permute_out


@class_registry("aclnnAlltoAllvGroupedMatMul")
class AlltoAllvGroupedMatMul(MC2_OP):

    def __init__(self, case, rank, dist=None, hcomm_info=None):
        super().__init__(case, rank, dist, hcomm_info)

        # 读取表格
        # input_tensor_shape
        (self.gmm_x_shape, self.gmm_weight_shape, self.send_counts_shape, self.recv_counts_shape, self.mm_x_shape, self.mm_weight_shape) = case["input_tensor_shape"]
        # input_tensor_range
        (self.gmm_x_range, self.gmm_weight_range, self.send_counts_range, self.recv_counts_range, self.mm_x_range, self.mm_weight_range) = case["input_tensor_range"]
        # input_tensor_dtype
        (self.gmm_x_dtype, self.gmm_weight_dtype, self.send_counts_dtype, self.recv_counts_dtype, self.mm_x_dtype, self.mm_weight_dtype) = (case["input_tensor_dtype"] if "input_tensor_dtype" in case else None)
        # input_tensor_format
        self.input_tensor_format = case["input_tensor_format"] if "input_tensor_format" in case else None
        # output_tensor_dtype
        self.gmm_y_dtype, self.mm_y_dtype, self.permute_out_dtype = case["output_tensor_dtype"] if "output_tensor_dtype" in case else None
        # output_tensor_format
        self.output_tensor_format = case["output_tensor_format"] if "output_tensor_format" in case else None
        # output_tensor_shape
        self.gmm_y_shape, self.mm_y_shape, self.permute_out_shape = case["output_tensor_shape"]
        # world_size common.py 中读取
        # trans_gmm_weight_flag
        self.trans_gmm_weight_flag = str_to_bool(case["trans_gmm_weight_flag"]) if "trans_gmm_weight_flag" in case else False
        # trans_mm_weight_flag
        self.trans_mm_weight_flag = str_to_bool(case["trans_mm_weight_flag"]) if "trans_mm_weight_flag" in case else False
        # permute_out_flag
        self.permute_out_flag = str_to_bool(case["permute_out_flag"]) if "permute_out_flag" in case else False
        # mm_out_flag
        self.mm_out_flag = str_to_bool(case["mm_out_flag"]) if "mm_out_flag" in case else False
        # seed
        self.seed = case["seed"] if "seed" in case else 42
        # graph_type
        self.graph_type = case["graph_type"] if "graph_type" in case else 0
        # recv_count_zero_rank_list
        recv_count_val = case.get("recv_count_zero_rank_list")
        if recv_count_val is not None and recv_count_val != "":
            self.recv_count_zero_rank_list = ast.literal_eval(recv_count_val) if isinstance(recv_count_val, str) else recv_count_val
        else:
            self.recv_count_zero_rank_list = []

        # shape值读取
        # gmm_x
        self.bsk, self.h1 = self.gmm_x_shape
        self.n1 = self.gmm_weight_shape[1] if self.trans_gmm_weight_flag else self.gmm_weight_shape[2]
        # gmm_wegiht
        self.e = self.gmm_weight_shape[0]
        # mm_x
        self.bs, self.h2 = self.mm_x_shape
        self.n2 = self.mm_weight_shape[0] if self.trans_mm_weight_flag else self.mm_weight_shape[1]
        self.exe_bin_name = "aclnnAlltoAllvGroupedMatMul"

        # 生成tokenNum
        self.expTokenNums, self.exp_torken_num_tensor = func.generate_expTokenNums(self.e * self.world_size, self.world_size, self.bsk, self.seed, self.recv_count_zero_rank_list)
        # 调整 gmm_output_shape（根据 recv_count_zero_rank_list）
        self.gmm_y_shape_list = []
        self.permute_out_shape_list = []
        for rank_id in range(self.world_size):
            gmm_y_shape_tmp = self.gmm_y_shape.copy()
            permute_out_shape_tmp = self.permute_out_shape.copy()
            exp_per_card = self.e
            start = rank_id * exp_per_card
            end = (rank_id + 1) * exp_per_card
            if isinstance(self.expTokenNums, torch.Tensor):
                total_recv = self.expTokenNums[:, start:end].sum().item()
            else:
                total_recv = sum(sum(src_list[start:end]) for src_list in self.expTokenNums)
            gmm_y_shape_tmp[0] = total_recv
            permute_out_shape_tmp[0] = total_recv
            self.gmm_y_shape_list.append(gmm_y_shape_tmp)
            self.permute_out_shape_list.append(permute_out_shape_tmp)

    def get_input_weight(self, weight_same=0):
        self.gmm_x = self.tensor(self.create_array("gmm_x", self.gmm_x_shape, self.gmm_x_range, self.gmm_x_dtype, self.input_path, self.seed))
        self.gmm_weight = self.tensor(self.create_array("gmm_weight", self.gmm_weight_shape, self.gmm_weight_range, self.gmm_weight_dtype, self.input_path, self.seed))
        self.gmm_weight_before_trans = self.gmm_weight
        if self.trans_gmm_weight_flag:
            self.gmm_weight = torch.transpose(self.gmm_weight, 1, 2)
        self.mm_x = (self.tensor(self.create_array("mm_x", self.mm_x_shape, self.mm_x_range, self.mm_x_dtype, self.input_path, self.seed)) if self.mm_out_flag else None)
        self.mm_weight = (self.tensor(self.create_array("mm_weight", self.mm_weight_shape, self.mm_weight_range, self.mm_weight_dtype, self.input_path, self.seed)) if self.mm_out_flag else None)
        self.mm_weight_before_trans = self.mm_weight if self.mm_out_flag else None
        if self.trans_mm_weight_flag and self.mm_out_flag:
            self.mm_weight = torch.transpose(self.mm_weight, 0, 1)
        self.input_split_sizes = self.gen_input_split_sizes()
        self.output_split_sizes = self.gen_output_split_sizes()
        self.group_list = self.get_group_list()
        # 保存输入bin
        if self.save_flag:
            self.save(self.gmm_x, self.input_path, "gmm_x.bin")
            self.save(self.gmm_weight, self.input_path, "gmm_weight.bin")
            if self.mm_out_flag:
                self.save(self.mm_x, self.input_path, "mm_x.bin")
                self.save(self.mm_weight, self.input_path, "mm_weight.bin")
            self.save(torch.tensor(self.get_send_counts()).to(TORCH_DTYPE_MAP[self.send_counts_dtype]), self.input_path, "send_counts.bin")
            self.save(torch.tensor(self.get_recv_counts()).to(TORCH_DTYPE_MAP[self.recv_counts_dtype]), self.input_path, "recv_counts.bin")

    def get_tmp_tensor(self):
        self.alltoallv_output = torch.zeros(
            self.permute_out_shape_list[self.rank],
            dtype=TORCH_DTYPE_MAP[self.gmm_x_dtype],
        ).to(self.device)

    def get_tmp_tensor_memory(self):
        return 0

    def permute(self, tokens):
        indices = torch.zeros(self.e, self.world_size).long()
        for j in range(self.e):
            for i in range(self.world_size):
                indices[j][i] = self.expTokenNums[i][j + (self.e * self.rank)]
        trans = indices.permute(1, 0)
        flaten = trans.reshape(-1).to(self.device)
        sum_list = torch.cumsum(flaten, dim=0)
        tmp = []
        for i in range(len(sum_list)):
            if i == 0:
                tmp.append(range(0, sum_list[i]))
            else:
                tmp.append(range(sum_list[i - 1], sum_list[i]))
        out = []
        for e in range(self.e):
            exp_token = []
            for r in range(self.world_size):
                exp_token += list(tmp[e + r * self.e])
            combined = torch.tensor(exp_token, dtype=torch.int64).to(self.device)
            original_dtype = tokens.dtype
            tokens_device = tokens.to(self.device)
            if tokens_device.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                tokens_device = tokens_device.to(torch.float32)
            result = tokens_device.index_select(0, combined)
            if original_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                result = result.to(original_dtype)
            out.append(result)
        if self.device.type == "npu":
            return torch.cat(out, dim=0).to(self.device)
        else:
            return out

    def get_cpu(self):
        # 升精度到float32
        self.gmm_x = self.gmm_x.to(torch.float32)
        self.gmm_weight = self.gmm_weight.to(torch.float32)
        self.alltoallv_output = self.alltoallv_output.to(torch.float32)
        if self.mm_out_flag:
            self.mm_x = self.mm_x.to(torch.float32)
            self.mm_weight = self.mm_weight.to(torch.float32)
        # alltoallv
        dist.all_to_all_single(
            self.alltoallv_output,
            input=self.gmm_x,
            output_split_sizes=self.output_split_sizes,
            input_split_sizes=self.input_split_sizes,
        )
        torch.cpu.synchronize()
        # permute
        self.cpu_permute_out = self.permute(self.alltoallv_output)
        # 路由专家计算
        cpu_gmm_y_tmp = []
        for e, expert_tokens in enumerate(self.cpu_permute_out):
            cpu_gmm_y_tmp.append(torch.matmul(expert_tokens, self.gmm_weight[e]))
        self.cpu_gmm_y = torch.cat(cpu_gmm_y_tmp, dim=0)
        self.cpu_permute_out = torch.cat(self.cpu_permute_out, dim=0)
        # 共享专家计算
        if self.mm_out_flag:
            self.cpu_mm_y = torch.mm(self.mm_x, self.mm_weight)
        else:
            self.cpu_mm_y = None
        # 保存输出bin
        self.save(self.cpu_gmm_y, self.output_path, "cpu_out.bin")
        if self.mm_out_flag:
            self.save(self.cpu_mm_y, self.output_path, "cpu_mm_out.bin")
        if self.permute_out_flag:
            self.save(self.cpu_permute_out, self.output_path, "cpu_permute_out.bin")

    def gen_input_split_sizes(self):
        input_split_sizes = []
        expTokenNum = self.expTokenNums[self.rank]
        for i in range(self.world_size):
            input_split_sizes.append(np.sum(expTokenNum[i * self.e:(i + 1) * self.e]))
        return input_split_sizes

    def gen_output_split_sizes(self):
        recv_counts = []
        for i in range(self.world_size):
            recv_counts.append(np.sum(self.expTokenNums[i][self.rank * self.e:(self.rank + 1) * self.e]))
        self.output_split_sizes = recv_counts
        return self.output_split_sizes

    def get_send_counts(self):
        expTokenNum = self.expTokenNums[self.rank]
        return expTokenNum

    def get_recv_counts(self):
        recv_counts = []
        for i in range(self.world_size):
            tmp = self.expTokenNums[i][self.rank * self.e:(self.rank + 1) * self.e]
            for j in range(len(tmp)):
                recv_counts.append(tmp[j])
        return recv_counts

    def get_group_list(self):
        recv = torch.zeros(self.world_size, self.e, dtype=torch.int64, device=self.device)
        for i in range(self.world_size):
            tmp = self.expTokenNums[i][self.rank * self.e:(self.rank + 1) * self.e]
            recv[i] = torch.tensor(tmp, device=self.device)
        return torch.cumsum(torch.sum(recv, dim=0).to(self.device), dim=0)

    def get_hccl_mm(self):
        # alltoallv
        dist.all_to_all_single(self.alltoallv_output, input=self.gmm_x, output_split_sizes=self.output_split_sizes, input_split_sizes=self.input_split_sizes)
        # permute
        self.hccl_permute_out = self.permute(self.alltoallv_output)
        # 路由专家计算
        self.hccl_gmm_y = gmm.npu_gmm(
            self.hccl_permute_out,
            self.gmm_weight,
            bias=None,
            group_list=self.group_list,
            group_type=0,
        ).to(TORCH_DTYPE_MAP[self.gmm_y_dtype])
        # 共享专家计算
        if self.mm_out_flag:
            self.hccl_mm_y = torch.mm(self.mm_x, self.mm_weight).to(TORCH_DTYPE_MAP[self.mm_y_dtype])
        else:
            self.hccl_mm_y = None
        # 保存输出bin
        if self.save_flag:
            self.save(self.hccl_gmm_y, self.output_path, "npu_gmm_y.bin")
            if self.permute_out_flag:
                self.save(self.hccl_permute_out, self.output_path, "npu_permute_out.bin")
            if self.mm_out_flag:
                self.save(self.hccl_mm_y, self.output_path, "npu_mm_y.bin")
        return (self.hccl_gmm_y, None, None, self.hccl_mm_y, self.hccl_permute_out if self.permute_out_flag else None)

    def get_npu(self):
        gmm_y, mm_y, permute_out = torch_npu.npu_alltoallv_gmm(
            gmm_x=self.gmm_x,
            gmm_weight=self.gmm_weight_before_trans,
            hcom=self.hcomm_info,
            ep_world_size=self.world_size,
            send_counts=self.get_send_counts(),
            recv_counts=self.get_recv_counts(),
            send_counts_tensor=None,
            recv_counts_tensor=None,
            mm_x=self.mm_x if self.mm_out_flag else None,
            mm_weight=self.mm_weight_before_trans if self.mm_out_flag else None,
            trans_gmm_weight=self.trans_gmm_weight_flag,
            trans_mm_weight=self.trans_mm_weight_flag,
            permute_out_flag=self.permute_out_flag,
        )
        self.npu_gmm_y = gmm_y
        self.npu_mm_y = mm_y if self.mm_out_flag else None
        self.npu_permute_out = permute_out if self.permute_out_flag else None
        return self.npu_gmm_y, self.npu_mm_y, self.npu_permute_out

    def compare(self, rank=0, check_md5=0):
        if len(self.recv_count_zero_rank_list) > 0:
            torch.distributed.barrier()
        # cpu所有场景都从文件读取
        cpu_gmm_y_path = os.path.join(self.output_path, BIN_FILE_NAME_MAP["output_cpu"]["gmm_y"])
        cpu_mm_y_path = os.path.join(self.output_path, BIN_FILE_NAME_MAP["output_cpu"]["mm_y"])
        cpu_permute_out_path = os.path.join(self.output_path, BIN_FILE_NAME_MAP["output_cpu"]["permute_out"])
        self.cpu_gmm_y, self.cpu_mm_y, self.cpu_permute_out = None, None, None
        self.cpu_gmm_y = torch.tensor(np.fromfile(cpu_gmm_y_path, np.float32).astype(np.float32))
        if self.mm_out_flag:
            self.cpu_mm_y = torch.tensor(np.fromfile(cpu_mm_y_path, np.float32).astype(np.float32))
        if self.permute_out_flag:
            self.cpu_permute_out = torch.tensor(np.fromfile(cpu_permute_out_path, np.float32).astype(np.float32))
        # 小算子级联、npu输出在aclnn场景从文件读取
        if self.scene in ["aclnn"]:
            hccl_gmm_y_path = os.path.join(self.output_path, BIN_FILE_NAME_MAP["output_hccl"]["gmm_y"])
            hccl_mm_y_path = os.path.join(self.output_path, BIN_FILE_NAME_MAP["output_hccl"]["mm_y"])
            hccl_permute_out_path = os.path.join(self.output_path, BIN_FILE_NAME_MAP["output_hccl"]["permute_out"])

            self.hccl_gmm_y = torch.tensor(np.fromfile(hccl_gmm_y_path, NUMPY_DTYPE_MAP[self.gmm_y_dtype]).astype(np.float32))
            if self.mm_out_flag:
                self.hccl_mm_y = torch.tensor(np.fromfile(hccl_mm_y_path, NUMPY_DTYPE_MAP[self.mm_y_dtype]).astype(np.float32))
            else:
                self.hccl_mm_y = None
            if self.permute_out_flag:
                if self.gmm_x_dtype == 'fp4_e2m1' or self.gmm_x_dtype == 'fp4_e1m2':
                    hccl_permute_out_temp = self.trans_to_float32(
                        self.unpack_int4(np.fromfile(hccl_permute_out_path, np.uint8)), 
                        self.gmm_x_dtype
                    )
                    self.hccl_permute_out = torch.tensor(hccl_permute_out_temp)
                else:
                    self.hccl_permute_out = torch.tensor(np.fromfile(hccl_permute_out_path, NUMPY_DTYPE_MAP[self.gmm_x_dtype]).astype(np.float32))
            else:
                self.hccl_permute_out = None

            npu_gmm_y_path = os.path.join(self.output_path, BIN_FILE_NAME_MAP["output_npu"]["gmm_y"])
            npu_mm_y_path = os.path.join(self.output_path, BIN_FILE_NAME_MAP["output_npu"]["mm_y"])
            npu_permute_out_path = os.path.join(self.output_path, BIN_FILE_NAME_MAP["output_npu"]["permute_out"])

            self.npu_gmm_y = torch.tensor(np.fromfile(npu_gmm_y_path, NUMPY_DTYPE_MAP[self.gmm_y_dtype]).astype(np.float32))
            if self.mm_out_flag:
                self.npu_mm_y = torch.tensor(np.fromfile(npu_mm_y_path, NUMPY_DTYPE_MAP[self.mm_y_dtype]).astype(np.float32))
            else:
                self.npu_mm_y = None
            if self.permute_out_flag:
                if self.gmm_x_dtype == 'fp4_e2m1' or self.gmm_x_dtype == 'fp4_e1m2':
                    npu_permute_out_temp = self.trans_to_float32(
                        self.unpack_int4(np.fromfile(npu_permute_out_path, np.uint8)), 
                        self.gmm_x_dtype
                    )
                    self.npu_permute_out = torch.tensor(npu_permute_out_temp)
                else:
                    self.npu_permute_out = torch.tensor(np.fromfile(npu_permute_out_path, NUMPY_DTYPE_MAP[self.gmm_x_dtype]).astype(np.float32))
            else:
                self.npu_permute_out = None

        output_cpu = (self.cpu_gmm_y, self.cpu_mm_y, self.cpu_permute_out)
        output_npu = (self.npu_gmm_y, self.npu_mm_y, self.npu_permute_out)
        output_npu_hccl = (self.hccl_gmm_y, self.hccl_mm_y, self.hccl_permute_out)
        for index, value in enumerate(output_npu):
            if value is not None:
                compare_rank = rank if "aclnn" in self.case["scene"] else self.rank
                output_cpu_single = output_cpu[index].view(-1).to(torch.float32)
                output_npu_singe = output_npu[index].cpu().view(-1).to(torch.float32)
                output_npu_hccl_single = output_npu_hccl[index].cpu().contiguous().view(-1).to(torch.float32)
                res = "FAIL"
                rst_1 = checkResult(output_npu_singe, output_cpu_single, "{}_npu_cpu".format(ACLNN_INDEX_MAP_ALLTOALL[index]))
                rst_1.print_result()
                rst_2 = checkResult(output_npu_hccl_single, output_cpu_single, "{}_hccl_npu".format(ACLNN_INDEX_MAP_ALLTOALL[index]))
                rst_2.print_result()
                str1, str2 = rst_1.check_result(rst_2)
                if "error" in str1:
                    res = "FAIL"
                else:
                    res = "PASS"
                print("[INFO] device_{} OUT {} 精度结果为：{}".format(compare_rank, ACLNN_INDEX_MAP_ALLTOALL[index], res))
                print("[INFO] device_{} OUT {} 精度计算结束：{} ".format(
                    compare_rank,
                    ACLNN_INDEX_MAP_ALLTOALL[index],
                    time.strftime("%H:%M:%S", time.localtime()),
                ))
                with open("alltoallv_gmm_result.csv", "a") as f:
                    f.write(self.case_name + "," + str(compare_rank) + "," + str(res) + "," + str(ACLNN_INDEX_MAP_ALLTOALL[index]) + "\n")
        return res

    def get_graph(self, type="mc2"):
        model = AlltoAllvGroupedMatmulGraphModel()
        model = self.define_model(model, self.graph_type)
        __send_counts = [int(x) for x in self.get_send_counts()]
        __recv_counts = [int(x) for x in self.get_recv_counts()]
        out_graph = model(
            self.gmm_x,
            self.gmm_weight_before_trans,
            self.mm_x if self.mm_out_flag else None,
            self.mm_weight_before_trans if self.mm_out_flag else None,
            self.hcomm_info,
            self.world_size,
            __send_counts,
            __recv_counts,
            None,  # sendTensor
            self.alltoallv_output,
            None,  # recvTensor
            self.trans_gmm_weight_flag,
            self.trans_mm_weight_flag,
            self.expTokenNums,
            self.group_list,
            self.e,
            self.permute_out_flag,
            self.output_split_sizes,
            self.input_split_sizes,
            self.rank,
            self.gmm_weight,
            self.mm_weight,
            type=type,
        )
        if type == "mc2":
            self.npu_gmm_y = out_graph[0]
            self.npu_mm_y = None
            self.npu_permute_out = None
            if self.mm_out_flag:
                self.npu_mm_y = out_graph[1]
            if self.permute_out_flag:
                self.npu_permute_out = out_graph[2]
            return self.npu_gmm_y, self.npu_mm_y, self.npu_permute_out
        elif type == "hccl_mm":
            out_graph = self.get_hccl_mm()
            self.hccl_gmm_y = out_graph[0]
            self.hccl_mm_y = out_graph[3] if self.mm_out_flag and len(out_graph) > 1 else None
            self.hccl_permute_out = out_graph[4] if self.permute_out_flag and len(out_graph) > 2 else None
            return self.hccl_gmm_y, self.hccl_mm_y, self.hccl_permute_out
        else:
            print("In get_graph, wrong type.")
            return None, None, None

    def process_aclnn(self):
        self.root_path = os.path.abspath(os.getcwd())
        self.bin_path = self.root_path + "/bin_path/" + self.case_name
        self.loop = 1

        def fmt_dtype(dtype_list):
            if isinstance(dtype_list, (list, tuple)):
                return ",".join(str(d).replace("'", "").replace(" ", "") for d in dtype_list)
            return str(dtype_list).replace("'", "").replace(" ", "")

        self.input_tensor_dtype = fmt_dtype(self.case["input_tensor_dtype"])
        self.output_tensor_dtype = fmt_dtype(self.case["output_tensor_dtype"])

        result = subprocess.check_output(["uname", "-a"]).decode("utf-8")
        arch = "aarch" if "aarch64" in result.lower() else "x86"
        exe_binary = "cpp/bin/" + self.exe_bin_name + "_" + arch

        if self.case.get("soc") == "910D":
            tmp_binary = self.exe_bin_name + "_aarch"  # 根据实际命名调整
            command = [f"nohup /mnt/mc2_test/bin_path/{tmp_binary}"]
            command.append(f"--bin_path /mnt/mc2_test/bin_path/")
        else:
            command = [f"{self.root_path}/{exe_binary}"]
            command.append(f"--bin_path {self.bin_path}")

        command.append(f"--gmm_x_shape {','.join(map(str, self.gmm_x_shape))}")
        command.append(f"--gmm_weight_shape {','.join(map(str, self.gmm_weight_shape))}")
        send_len = len(self.get_send_counts())
        recv_len = len(self.get_recv_counts())
        command.append(f"--send_counts_shape {send_len}")
        command.append(f"--recv_counts_shape {recv_len}")
        command.append(f"--mm_x_shape {','.join(map(str, self.mm_x_shape))}")
        command.append(f"--mm_weight_shape {','.join(map(str, self.mm_weight_shape))}")
        command.append(f"--input_tensor_dtype {self.input_tensor_dtype}")
        command.append(f"--input_tensor_format ND,ND,ND,ND,ND,ND")
        command.append(f"--output_tensor_dtype {self.output_tensor_dtype}")
        command.append(f"--output_tensor_format ND,ND,ND")
        command.append(f"--mm_y_shape {','.join(map(str, self.mm_y_shape))}")
        command.append(f"--ep_world_size {self.world_size}")
        command.append(f"--trans_gmm_weight_flag {str(self.trans_gmm_weight_flag).lower()}")
        command.append(f"--trans_mm_weight_flag {str(self.trans_mm_weight_flag).lower()}")
        command.append(f"--n_dev {self.world_size}")
        command.append(f"--loop_cnt {self.loop}")
        command.append(f"--permute_out_shape {','.join(map(str, self.permute_out_shape))}")
        command.append(f"--mm_out_flag {str(self.mm_out_flag).lower()}")
        command.append(f"--permute_out_flag {str(self.permute_out_flag).lower()}")

        if self.case.get("soc") == "910D" and self.case.get("scene") == "aclnn":
            shell_path = f"{self.bin_path}/{self.case_name}_mc2.sh"
            with open(shell_path, "w") as f:
                f.write(f"cp -f /mnt/mc2_test/bin_path/{self.world_size}P1K/*.json ./\n")
                f.write(f"export HCCL_TOPO_FILE_PATH=/mnt/mc2_test/bin_path/{self.case_name}/topo.json\n")
                for i in range(self.world_size):
                    tmp_cmd = command.copy()
                    tmp_cmd.append(f"--gmm_y_shape {','.join(map(str, self.gmm_y_shape_list[i]))}")
                    tmp_cmd.append(f"--permute_out_shape {','.join(map(str, self.permute_out_shape_list[i]))}")
                    tmp_cmd.append(f"--rankId {i} &")
                    f.write(" ".join(tmp_cmd) + "\n")
            subprocess.run(f"bash {shell_path}", shell=True, check=True)
        else:
            processes = []
            for i in range(self.world_size):
                tmp_cmd = command.copy()
                tmp_cmd.append(f"--gmm_y_shape {','.join(map(str, self.gmm_y_shape_list[i]))}")
                tmp_cmd.append(f"--permute_out_shape {','.join(map(str, self.permute_out_shape_list[i]))}")
                tmp_cmd.append(f"--rankId {i}")
                aclnn_cmd = " ".join(tmp_cmd)
                proc = subprocess.Popen(aclnn_cmd, shell=True)
                processes.append(proc)
            for i, proc in enumerate(processes):
                proc.wait()