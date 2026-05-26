from .common import *
import torch
import random
from mindspeed.ops import gmm
import torch_npu
from script import toolss
from ml_dtypes import bfloat16
from .registry import class_registry

ACLNN_INDEX_MAP_ALLTOALL = {0: "out", 1: "mm_out"}

# bin文件名
BIN_FILE_NAME_MAP = {
    "input": {
        "gmm_x": "gmm_x.bin",
        "gmm_weight": "gmm_weight.bin",
        "mm_x": "mm_x.bin",
        "mm_weight": "mm_weight.bin",
    },
    "output_cpu": {
        "gmm_y": "cpu_out.bin",
        "mm_y": "cpu_mm_out.bin",
    },
    "output_hccl": {
        "gmm_y": "output_hccl.bin",
        "mm_y": "mm_out_hccl.bin",
    },
    "output_npu": {
        "gmm_y": "npu_out.bin",
        "mm_y": "npu_mm_y.bin",
    },
}


def str_to_bool(s):
    if isinstance(s, bool):
        return s
    if isinstance(s, str):
        return s.strip().lower() in ("true", "1", "yes", "y")
    return False


def unpermute_fun(tokens,exp_per_card,ep_world_size,rankId,expTokenNums,device="cpu"):
    emptyArr = np.zeros((ep_world_size, exp_per_card),dtype=np.int64)
    for i in range(ep_world_size):
        for j in range(exp_per_card):
            emptyArr[i][j] = int(expTokenNums[i][rankId * exp_per_card + j])
    tmp1 = emptyArr.transpose()
    sum_list1 = np.sum(tmp1, axis=1)
    sum_list2= np.cumsum(sum_list1,axis=0)
    offsets = [0]
    for s in sum_list2[0:-1]:
        offsets.append(int(s))
    sum_list = np.cumsum(tmp1, axis=1)
    indices_list = []
    for i in range(exp_per_card):
        tmp = []
        for j in range(ep_world_size):
            if j == 0:
                tmp.append(list(map(lambda x: x + offsets[i],  list(range(0, sum_list[i][j])))))
            else:
                tmp.append(list(map(lambda x: x + offsets[i], list(range(sum_list[i][j - 1], sum_list[i][j])))))
        indices_list.append(tmp)
    selected = []
    for i in range(ep_world_size):
        for j in range(exp_per_card):
            indices = torch.tensor(indices_list[j][i], dtype=torch.long).to(device)
            selected_rows = tokens.to(device).index_select(dim=0, index=indices)
            selected.append(selected_rows)
    return torch.cat(selected, dim=0).to(tokens.dtype)


class GMM_ALLTOALLV_GRAPH_Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, _gmm_x, _gmm_weight, _ep_world_size, _hcom, _send_counts, _recv_counts, _all_to_allv_out, _mm_x, _mm_weight, _trans_gmm_weight, _trans_mm_weight,_expTokenNums,
                _group_list, _e, _out_splits,_input_splits, _rankId,_gmm_weight_after_trans, _mm_weight_after_trans, type):
        output=None
        if type == "mc2":
            output=torch_npu.npu_gmm_alltoallv(gmm_x = _gmm_x,
                                    gmm_weight = _gmm_weight,
                                    ep_world_size = _ep_world_size,
                                    hcom= _hcom,
                                    send_counts = _send_counts,
                                    recv_counts = _recv_counts,
                                    send_counts_tensor = None,#self.send_tensor,
                                    recv_counts_tensor = None,#self.recv_tensor,
                                    mm_x = _mm_x,
                                    mm_weight = _mm_weight,
                                    trans_gmm_weight = _trans_gmm_weight,
                                    trans_mm_weight = _trans_mm_weight
                                    )
        if type == "hccl_mm":
            output = []
            # _gmm_out = gmm.npu_gmm(_gmm_x, _gmm_weight_after_trans, bias=None, group_list=_group_list, group_type=0)
            # unpermuted_out  = unpermute_fun(_gmm_out, _e, _ep_world_size, _rankId, _expTokenNums, "npu")
            # _output = _all_to_allv_out.to(_gmm_x.dtype)
            # dist.all_to_all_single(_output, input=unpermuted_out, output_split_sizes=_out_splits, input_split_sizes=_input_splits)
            # output.append(_output)
            # mm_out = None
            # if _mm_x != None:
            #     mm_out = torch.mm(_mm_x,_mm_weight_after_trans)
            #     output.append(mm_out)
        return output

@class_registry("aclnnGroupedMatMulAlltoAllv")
class GMM_ALLTOALLV(MC2_OP):

    def __init__(self, case, rank, dist=None, hcomm_info=None):
        super().__init__(case, rank, dist, hcomm_info)

        self.gmm_x_shape = case['input_tensor_shape'][0]
        self.gmm_weight_shape = case['input_tensor_shape'][1]
        self.send_counts_shape = case['input_tensor_shape'][2]
        self.recv_counts_shape = case['input_tensor_shape'][3]
        self.mm_x_shape = case['input_tensor_shape'][4]
        self.mm_weight_shape = case['input_tensor_shape'][5]
        (self.gmm_x_range, self.gmm_weight_range, self.send_counts_range, self.recv_counts_range, self.mm_x_range, self.mm_weight_range) = case["input_tensor_range"]
        (self.gmm_x_dtype, self.gmm_weight_dtype, self.send_counts_dtype, self.recv_counts_dtype, self.mm_x_dtype, self.mm_weight_dtype) = (case["input_tensor_dtype"] if "input_tensor_dtype" in case else None)
        self.gmm_y_dtype, self.mm_y_dtype = case["output_tensor_dtype"] if "output_tensor_dtype" in case else None
        self.input_tensor_dtype   = case['input_tensor_dtype'] if 'input_tensor_dtype' in case else None
        self.input_tensor_format  = case['input_tensor_format'] if 'input_tensor_format' in case else None
        self.output_tensor_dtype  = case['output_tensor_dtype'] if 'output_tensor_dtype' in case else None
        self.output_tensor_format = case['output_tensor_format'] if 'output_tensor_format' in case else None
        self.mm_output_shape = case['output_tensor_shape'][1]
        self.tp_size = 1
        self.trans_gmm_weight_flag = case['trans_gmm_weight_flag'] if 'trans_gmm_weight_flag' in case else None
        self.trans_mm_weight_flag = case['trans_mm_weight_flag'] if 'trans_mm_weight_flag' in case else None
        self.mm_out_flag  = case['mm_out_flag'] if 'mm_out_flag' in case else False
        self.seed = case["seed"] if "seed" in case else 42
        self.exp_num =  self.gmm_weight_shape[0] * self.world_size
        self.exp_per_card =self.gmm_weight_shape[0]
        self.A = self.gmm_x_shape[0]
        self.k = self.gmm_x_shape[1]
        self.n = self.gmm_weight_shape[1] if self.trans_gmm_weight_flag else self.gmm_weight_shape[2]
        self.bs = self.mm_x_shape[0]
        self.h = self.mm_x_shape[1]
        self.n2 = self.mm_weight_shape[0] if self.trans_mm_weight_flag else self.mm_weight_shape[1]
        self.ranks = self.world_size
        recv_count_val = case.get("recv_count_zero_rank_list") if 'recv_count_zero_rank_list' in case else None
        if recv_count_val is not None and recv_count_val != "":
            self.recv_count_zero_rank_list = ast.literal_eval(recv_count_val) if isinstance(recv_count_val, str) else recv_count_val
        else:
            self.recv_count_zero_rank_list = []
        self.gmm_x_A_array = func.generate_positive_integers(self.ranks,self.A,case["seed"])
        self.expTokenNums,self.exp_torken_num_tensor = func.generate_gmm_alltoallv_matrix(self.gmm_x_A_array, self.exp_per_card, self.recv_count_zero_rank_list, case["seed"])
        self.bsk = self.A//self.ranks
        self.output_dtype =  str(self.output_tensor_dtype).replace("[","").replace("]","").replace("'","").replace(" ","").split(",")[0]
        self.mm_output_dtype =  str(self.output_tensor_dtype).replace("[","").replace("]","").replace("'","").replace(" ","").split(",")[1]
        self.input_dtype =  str(self.input_tensor_dtype).replace("[","").replace("]","").replace("'","").replace(" ","").split(",")[0]
        self.exe_bin_name = "aclnnGroupedMatMulAlltoAllv"

    def get_recv_counts(self):
        return self.expTokenNums[self.rank]

    def get_send_counts(self):
        recv_counts = []
        for i in range(self.ranks):
            tmp = self.expTokenNums[i][ self.rank*self.exp_per_card:(self.rank+1)*self.exp_per_card]
            for j in range(len(tmp)):
                recv_counts.append(tmp[j])
        return recv_counts

    def gen_input_splits(self):
        input_splits = []
        expTokenNum = self.expTokenNums[self.rank]
        for i in range(self.ranks):
            input_splits.append(sum(expTokenNum[i*self.exp_per_card:(i+1)*self.exp_per_card]))
        return input_splits

    def gen_output_splits(self):
        recv_counts = []
        for i in range(self.ranks):
            recv_counts.append(sum(self.expTokenNums[i][ self.rank*self.exp_per_card:(self.rank+1)*self.exp_per_card]))
        return recv_counts

    def unpermute(self,tokens,device="cpu"):
        emptyArr = np.zeros((self.ranks, self.exp_per_card),dtype=np.int64)
        for i in range(self.world_size):
            tmp = self.expTokenNums[i][self.rank * self.exp_per_card:(self.rank + 1) * self.exp_per_card]
            emptyArr[i:] = tmp
        tmp1 = emptyArr.transpose()
        sum_list1 = np.sum(tmp1, axis=1)
        sum_list2= np.cumsum(sum_list1,axis=0)
        offsets = sum_list2[0:-1].tolist()
        offsets = [0] + offsets
        sum_list = np.cumsum(tmp1, axis=1)
        indices_list = []
        for i in range(self.exp_per_card):
            tmp = []
            for j in range(self.ranks):
                if j == 0:
                    tmp.append(list(map(lambda x: x + offsets[i],  list(range(0, sum_list[i][j])))))
                else:
                    tmp.append(list(map(lambda x: x + offsets[i], list(range(sum_list[i][j - 1], sum_list[i][j])))))
            indices_list.append(tmp)
        selected = []
        for i in range(self.ranks):
            for j in range(self.exp_per_card):
                indices = torch.tensor(indices_list[j][i], dtype=torch.long).to(device)
                selected_rows = tokens.to(device).index_select(dim=0, index=indices)
                selected.append(selected_rows)
        return torch.cat(selected, dim=0).to(tokens.dtype)

    def unpermute_npu(self,tokens):
        return self.unpermute(tokens,"npu")

    def get_input_weight(self, weight_same=0):
        input_splits = self.gen_output_splits()
        self.input_splits = input_splits
        output_splits = self.gen_input_splits()
        self.output_splits = output_splits
        self.expert_token_num = torch.tensor(self.expTokenNums[self.rank]).to(torch.int64)
        tmp_dtype = torch.float16 if self.input_dtype == "fp16" else torch.bfloat16
        tmp_seed = int(self.case['seed']) if 'seed' in self.case else 1
        tmp_seed = tmp_seed + self.rank
        np.random.seed(tmp_seed)
        tmp = []
        tmp.append(self.gmm_x_A_array[self.rank])
        tmp.append(self.case["input_tensor_shape"][0][1])
        self.gmm_x_shape = tmp

        self.gmm_x = self.tensor(self.create_array("gmm_x", self.gmm_x_shape, self.gmm_x_range, self.gmm_x_dtype, self.input_path, self.seed))
        self.gmm_weight = self.tensor(self.create_array("gmm_weight", self.gmm_weight_shape, self.gmm_weight_range, self.gmm_weight_dtype, self.input_path, self.seed))
        self.mm_x = (self.tensor(self.create_array("mm_x", self.mm_x_shape, self.mm_x_range, self.mm_x_dtype, self.input_path, self.seed)) if self.mm_out_flag else None)
        self.mm_weight = (self.tensor(self.create_array("mm_weight", self.mm_weight_shape, self.mm_weight_range, self.mm_weight_dtype, self.input_path, self.seed)) if self.mm_out_flag else None)
        self.output = torch.empty((sum(output_splits), self.n),dtype=tmp_dtype)
        send_dtype = torch.int64 if self.input_tensor_dtype[2] == "int64" else torch.int32
        recv_dtype = torch.int64 if self.input_tensor_dtype[3] == "int64" else torch.int32
        self.send_tensor = torch.tensor(self.get_send_counts()).to(send_dtype)
        self.recv_tensor = torch.tensor(self.get_recv_counts()).to(recv_dtype)
        self.group_list = self.get_group_list()
        self.gmm_weight_before_trans = self.gmm_weight
        self.mm_weight_before_trans = self.mm_weight
        if self.trans_gmm_weight_flag:
           self.gmm_weight = torch.transpose(self.gmm_weight,1,2)
        if self.trans_mm_weight_flag and self.mm_out_flag:
           self.mm_weight = torch.transpose(self.mm_weight,0,1)
        
        # 保存输入bin
        if self.save_flag:
            self.save(self.gmm_x, self.input_path, "gmm_x.bin")
            self.save(self.gmm_weight, self.input_path, "gmm_weight.bin")
            func.save_file(self.input_path, self.send_tensor, self.send_tensor.dtype, "send_counts", "None", self.rank)
            func.save_file(self.input_path, self.recv_tensor, self.recv_tensor.dtype, "recv_counts", "None", self.rank)
            if self.mm_out_flag:
                self.save(self.mm_x, self.input_path, "mm_x.bin")
                self.save(self.mm_weight, self.input_path, "mm_weight.bin")

    def get_tmp_tensor(self):
        if self.mm_out_flag ==False:
            self.mm_weight =None
            self.mm_x = None
        if "cpu" not in self.case['scene']:
            from itertools import accumulate
            self.gmm_x = self.gmm_x.npu()
            self.gmm_weight = self.gmm_weight.npu()
            self.mm_x = self.mm_x.npu() if self.mm_out_flag else None
            self.mm_weight = self.mm_weight.npu() if self.mm_out_flag else None
            self.expert_token_num = self.expert_token_num.npu()
            self.group_list = torch.tensor(list(accumulate(self.group_list))).npu()
            self.output =self.output.npu()
            self.send_tensor = self.send_tensor.npu()
            self.recv_tensor = self.recv_tensor.npu()
            self.gmm_weight_before_trans = self.gmm_weight_before_trans.npu()
            if self.mm_out_flag:
                self.mm_weight_before_trans = self.mm_weight_before_trans.npu()

    def get_tmp_tensor_memory(self):
        return 0

    def get_group_list(self):
        recv = torch.zeros(self.ranks, self.exp_per_card).to(torch.int64)
        for i in range(self.world_size):
            tmp = self.expTokenNums[i][self.rank * self.exp_per_card:(self.rank + 1) * self.exp_per_card]
            recv[i:] = torch.tensor(tmp)
        return torch.sum(recv, dim=0)

    def get_hccl_mm(self):
        group_type = 0
        group_list = self.group_list
        gmm_out = gmm.npu_gmm(self.gmm_x, self.gmm_weight, bias=None, group_list=group_list, group_type=group_type)
        unpermuted_out  = self.unpermute_npu(gmm_out)
        output = self.output.to(self.gmm_x.dtype)
        dist.all_to_all_single(output, input=unpermuted_out, output_split_sizes=self.output_splits, input_split_sizes=self.input_splits)
        mm_out = None
        if self.mm_out_flag:
            mm_out = torch.mm(self.mm_x,self.mm_weight)
        self.npu_hccl_out = output
        self.npu_hccl_mm_out = mm_out

        # 保存输出bin
        if self.save_flag:
            self.save(self.npu_hccl_out, self.output_path, "npu_out.bin")
            if self.mm_out_flag:
                self.save(self.npu_hccl_mm_out, self.output_path, "npu_mm_y.bin")

        return output,None,None, mm_out, None

    def get_npu(self):
        #MC2接口使用转置前的tensor。因为可以通过标志区分
        tmp_mm_weight = self.mm_weight_before_trans if self.mm_out_flag else None
        npu_out, mm_out = torch_npu.npu_gmm_alltoallv(gmm_x=self.gmm_x,
                                       gmm_weight=self.gmm_weight_before_trans,
                                       ep_world_size=self.world_size,
                                       hcom=self.hcomm_info,
                                       send_counts=self.get_send_counts(),
                                       recv_counts=self.get_recv_counts(),
                                       send_counts_tensor=None,#self.send_tensor,
                                       recv_counts_tensor=None,#self.recv_tensor,
                                       mm_x=self.mm_x,
                                       mm_weight=tmp_mm_weight,
                                       trans_gmm_weight=self.trans_gmm_weight_flag,
                                       trans_mm_weight=self.trans_mm_weight_flag
                                       )
        self.npu_mm_out = mm_out if self.mm_out_flag else None
        self.npu_out = npu_out
        return self.npu_out

    def grouped_matmul_cpu(self,gmm_x, gmm_weight, group_list):
        # 按行分割左矩阵
        B_list = list(torch.unbind(gmm_weight, dim=0))
        A_groups = torch.split(gmm_x, group_list, dim=0)
        # 逐组相乘并拼接结果
        results = []
        for i in range(len(group_list)):
            a = A_groups[i].numpy()
            b = B_list[i].numpy()
            results.append(torch.from_numpy(np.matmul(a,b)))
        #因为torch2.1有bug 会报 could not create a primitive descriptor for a matmul primitive所以转成np处理
        #results = [torch.mm(a,b) for a, b in zip(A_groups, B_list)]
        return torch.cat(results, dim=0)

    def get_cpu(self):
        cpu_gmm_out = self.grouped_matmul_cpu(self.gmm_x.to(torch.float32),self.gmm_weight.to(torch.float32),self.group_list.tolist())
        unpermute_out =  self.unpermute(cpu_gmm_out)
        unpermute_out_float = unpermute_out.to(torch.float32)
        output = self.output.to(torch.float32)
        dist.all_to_all_single(output, input=unpermute_out_float, output_split_sizes=self.output_splits,input_split_sizes=self.input_splits)
        torch.cpu.synchronize()
        self.cpu_alltoallv_out = output
        mm_out=None
        if self.mm_out_flag:
            if self.input_dtype == "fp16":
                mm_out = torch.mm(self.mm_x.to(torch.float32),self.mm_weight.to(torch.float32)).to(torch.float32)
            else:
                mm_out = torch.mm(self.mm_x.to(torch.float32),self.mm_weight.to(torch.float32)).to(torch.float32)

        input_path = f"{self.save_path}/input/{self.rank}/"
        output_path = f"{self.save_path}/output/{self.rank}/"
        #转置存储转置前的tensor aclnn接口创建方法不区分转置不转值按照(n,k)创建通过标志区分是否转置
        func.save_file(output_path, self.cpu_alltoallv_out, self.cpu_alltoallv_out.dtype, "cpu_out", "None", self.rank)
        if self.mm_out_flag:
            func.save_file(output_path,mm_out,mm_out.dtype,"cpu_mm_out","None", self.rank)
        self.cpu_golden_out = self.cpu_alltoallv_out
        self.cpu_golden_mm_out = mm_out
        print("get_cpu success")

    def get_graph(self, type="mc2"):
        model = GMM_ALLTOALLV_GRAPH_Model()
        alltoallv_output= self.output
        _send_counts = [int(x) for x in self.get_send_counts()]
        _recv_counts = [int(y) for y in self.get_recv_counts()]
        model = self.define_model(model, self.graph_type)
        tmp_mm_weight = self.mm_weight_before_trans if self.mm_out_flag else None
        out_graph = model(self.gmm_x, self.gmm_weight_before_trans, self.world_size, self.hcomm_info,_send_counts,
                _recv_counts,alltoallv_output,self.mm_x, tmp_mm_weight,self.trans_gmm_weight_flag,self.trans_mm_weight_flag,
                self.expTokenNums,self.group_list,self.exp_per_card,self.output_splits,self.input_splits,self.rank,self.gmm_weight, self.mm_weight, type=type)
        if type == "mc2":
            self.npu_out = out_graph[0]
            self.npu_mm_out = None
            if self.mm_out_flag:
                self.npu_mm_out = out_graph[1]
            return self.npu_out, self.npu_mm_out
        elif type == "hccl_mm":
            out_graph = self.get_hccl_mm()
            self.npu_hccl_out = out_graph[0]
            self.npu_hccl_mm_out = out_graph[3]
            return self.npu_hccl_out, self.npu_hccl_mm_out
        else:
            print("In allgather get_graph,wrong type..")

    def compare(self, rank=0, check_md5=0):
        rank = rank if "aclnn" in self.case['scene'] else self.rank
        cpu_golden_out_path = f"{self.save_path}/output/{rank}/cpu_out.bin"
        cpu_golden_mm_out_path = f"{self.save_path}/output/{rank}/cpu_mm_out.bin"

        np_dtype = NUMPY_DTYPE_MAP[self.output_dtype]
        mm_np_dtype = NUMPY_DTYPE_MAP[self.mm_output_dtype]

        self.cpu_golden_out, self.cpu_golden_mm_out = None,None
        self.cpu_golden_out = torch.tensor(np.fromfile(cpu_golden_out_path, np.float32).astype(np.float32))
        if self.mm_out_flag:
            self.cpu_golden_mm_out = torch.tensor(np.fromfile(cpu_golden_mm_out_path, np.float32).astype(np.float32))
        if self.case["scene"] in ["aclnn", "compare"]:# 使用三方比对
            npu_out_path = f"{self.save_path}/output/{rank}/npu_out.bin"
            npu_hccl_output_path = f"{self.save_path}/output/{rank}/output_hccl.bin"
            if self.mm_out_flag:#使用三方比对
                npu_mm_out_path = f"{self.save_path}/output/{rank}/npu_mm_out.bin"
                npu_hccl_mm_out_path = f"{self.save_path}/output/{rank}/mm_out_hccl.bin"

            self.npu_out, self.npu_mm_out = None,None
            self.npu_hccl_out, self.npu_hccl_mm_out = None,None,

            self.npu_out = torch.tensor(np.fromfile(npu_out_path, np_dtype).astype(np.float32))
            self.npu_hccl_out = torch.tensor(np.fromfile(npu_hccl_output_path, np_dtype).astype(np.float32))
            if self.mm_out_flag:
                self.npu_mm_out = torch.tensor(np.fromfile(npu_mm_out_path, mm_np_dtype).astype(np.float32))
                self.npu_hccl_mm_out = torch.tensor(np.fromfile(npu_hccl_mm_out_path, mm_np_dtype).astype(np.float32))

        output_npu = (self.npu_out, self.npu_mm_out)
        output_cpu = (self.cpu_golden_out,self.cpu_golden_mm_out)
        output_npu_hccl = (self.npu_hccl_out,self.npu_hccl_mm_out)

        for index, value in enumerate(output_cpu):
            if value is not None:
                compare_rank = rank if "aclnn" in self.case['scene'] else self.rank
                cpu_input_path = os.path.join('bin_path',self.case_name,'output/{}/cpu_{}.bin'.format(compare_rank,ACLNN_INDEX_MAP_ALLTOALL[index]))
                print("cpu bin path",cpu_input_path)
                output_cpu_single = output_cpu[index].cpu().view(-1).to(torch.float32)
                output_npu_singe = output_npu[index].cpu().view(-1).to(torch.float32)
                output_npu_hccl_single = output_npu_hccl[index].cpu().contiguous().view(-1).to(torch.float32)
                res ="FAIL"
                rst_1 = checkResult(output_cpu_single, output_npu_singe, "{}_dq_npu".format(ACLNN_INDEX_MAP_ALLTOALL[index]))
                rst_1.print_result()
                rst_2 = checkResult(output_cpu_single,output_npu_hccl_single , "{}_dq_npu".format(ACLNN_INDEX_MAP_ALLTOALL[index]))
                rst_2.print_result()
                str1, str2 = rst_1.check_result(rst_2)
                if 'error' in str1:
                    res = 'FAIL'
                else:
                    res = 'PASS'
                print('[INFO] device_{} OUT {} 精度结果为：{}'.format(compare_rank, ACLNN_INDEX_MAP_ALLTOALL[index], res))
                print('[INFO] device_{} OUT {} 精度计算结束：{} '.format(compare_rank, ACLNN_INDEX_MAP_ALLTOALL[index],
                                                                      time.strftime('%H:%M:%S', time.localtime())))
                with open('gmm_alltoallv_result.csv', 'a') as f:
                    f.write(self.case_name + "," + str(compare_rank) + ',' + str(res) + ',' + str(ACLNN_INDEX_MAP_ALLTOALL[index]) + '\n')
                print("cpu bin path",cpu_input_path)
        return res

    def process_aclnn(self):
        self.root_path = os.path.abspath(os.getcwd())
        self.case_format = ['ND', 'ND', 'ND']
        self.output_dtype = self.case['input_tensor_dtype'][0]
        self.output_format = self.case_format[0]
        self.bin_path = self.root_path + "/bin_path/" + self.case_name
        self.input_tensor_dtype = str(self.input_tensor_dtype).replace("[","").replace("]","").replace("'","").replace(" ","")
        self.input_tensor_format = str(self.input_tensor_format).replace("[","").replace("]","").replace("'","").replace(" ","")
        self.output_tensor_dtype = str(self.output_tensor_dtype).replace("[","").replace("]","").replace("'","").replace(" ","")
        self.output_tensor_format = str(self.output_tensor_format).replace("[","").replace("]","").replace("'","").replace(" ","")
        print(f"bin_path: {self.bin_path}")
        self.loop = 1
        # 根据架构aarch/x86获取待执行二进制名称
        result = subprocess.check_output(["uname", "-a"]).decode("utf-8")
        arch = "aarch" if "aarch64" in result.lower() else "x86"
        exe_binary = "cpp/bin/" + self.exe_bin_name + "_" + arch
        # 拼二进制执行命令
        if self.case['soc'] == "910D":
            tmp_binary = self.exe_bin_name + "aarch"
            commond =  [f"nohup /mnt/mc2_test/bin_path/{tmp_binary}"]
            command.append(f"--bin_path {'/mnt/mc2_test/bin_path/'}")
        else:
            command = [f"{self.root_path}/{exe_binary}"]
            command.append(f"--bin_path {self.bin_path}")

        tmp = self.gmm_x_A_array.tolist()
        tmp.append(self.k)
        alltoallv_shape = [self.bsk,self.n]
        mm_out_shape = [self.bs,self.n2]

        command.append(f"--gmm_x_shape {','.join(map(str, tmp))}")
        command.append(f"--gmm_weight_shape {','.join(map(str, self.gmm_weight_shape))}")
        command.append(f"--send_counts_shape {','.join(map(str, self.send_counts_shape))}")
        command.append(f"--recv_counts_shape {','.join(map(str, self.recv_counts_shape))}")
        command.append(f"--mm_x_shape {','.join(map(str, self.mm_x_shape))}")
        command.append(f"--mm_weight_shape {','.join(map(str, self.mm_weight_shape))}")
        command.append(f"--input_tensor_dtype { self.input_tensor_dtype}")
        command.append(f"--input_tensor_format {self.input_tensor_format}")
        command.append(f"--output_tensor_dtype {self.output_tensor_dtype}")
        command.append(f"--output_tensor_format {self.output_tensor_format}")
        command.append(f"--mm_output_shape {','.join(map(str, mm_out_shape))}")
        command.append(f"--ep_world_size {self.world_size}")
        command.append(f"--trans_gmm_weight_flag {str(self.trans_gmm_weight_flag).lower()}")
        command.append(f"--trans_mm_weight_flag {str(self.trans_mm_weight_flag).lower()}")
        command.append(f"--n_dev {self.ranks}")
        command.append(f"--loop_cnt {self.loop}")
        command.append(f"--mm_out_flag {str(self.mm_out_flag).lower()}")
        command.append(f"--alltoallv_out_shape {','.join(map(str, alltoallv_shape))}")

        if self.case['soc'] == "910D":
            if self.case['scene'] == 'aclnn':
                ranktable_cmd = f"cp -f /mnt/mc2_test/bin_path/{self.world_size}P1K/*.json ./"
                env_cmd = f"export HCCL_TOPO_FILE_PATH=/mnt/mc2_test/bin_path/{self.case_name}/topo.json"
                mc2_sh = self.case_name + '_mc2'
                # 移除已有shell文件
                mv_command = f"rm -rf bin_path/{self.case_name}/{self.case_name}*.sh"
                ret = subprocess.run(mv_command, shell=True)
                #mc2结果生成命令
                ret = subprocess.check_output(f"echo {ranktable_cmd} >> {self.bin_path}/{mc2_sh}.sh", shell=True)
                ret = subprocess.check_output(f"echo {env_cmd} >> {self.bin_path}/{mc2_sh}.sh", shell=True)
                #拼二进制执行命令
                for i in range(self.ranks):
                    tmp_command = command.copy()
                    tmp_command.append(f"--rankId {i} \&")
                    aclnn_command = " ".join(tmp_command)
                    print(f"[INFO] rank:{i}, aclnn command:{aclnn_command}")
                    # 命令写入shell
                    ret = subprocess.check_output(f"echo {aclnn_command} >> {self.bin_path}/{mc2_sh}.sh", shell=True)
                if ret:
                    print(f"Generate shell command excute result: {ret}")
                sh_command = f"bash bin_path/{self.case_name}/{self.case_name}.sh"
                print(f"[INFO] sh command:{sh_command}")
        else:
            aclnn_command = " ".join(command)
            print(f"[INFO] aclnn command:{aclnn_command}")
            # 命令写入shell
            ret = subprocess.check_output(f"echo {aclnn_command} > {self.bin_path}/{self.case_name}.sh", shell=True)
            if ret:
                print(f"Generate shell command excute result: {ret}")
            sh_command = f"bash bin_path/{self.case_name}/{self.case_name}.sh"
            print(f"[INFO] sh command:{sh_command}")
            try:
                ret = subprocess.run(sh_command, shell=True)
            except subprocess.CalledProcessError:
                print(f"Aclnn command excute result: {ret}")