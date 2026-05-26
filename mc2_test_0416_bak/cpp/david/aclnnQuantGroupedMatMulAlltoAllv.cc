#include "common.h"
#include <cstring>
#include "aclnnop/aclnn_quant_grouped_mat_mul_allto_allv.h"
 
using namespace std;
// 定义全局变量
std::string g_platform = "1971";
std::string g_bin_path = "./";
// 输入
std::string g_input_tensor_dtype[128] = {};
std::string g_input_tensor_format[128] = {};

std::string g_input_scales_dtype[128] = {};
std::string g_input_scales_format[128] = {};

std::vector<int64_t> g_gmm_x_shape = {};
std::vector<int64_t> g_gmm_weight_shape = {};
std::vector<int64_t> g_gmm_x_scale_shape = {};
std::vector<int64_t> g_gmm_weight_scale_shape = {};
std::vector<int64_t> g_send_counts_shape = {};
std::vector<int64_t> g_recv_counts_shape = {};
std::vector<int64_t> g_mm_x_shape = {};
std::vector<int64_t> g_mm_weight_shape = {};
std::vector<int64_t> g_mm_x_scale_shape = {};
std::vector<int64_t> g_mm_weight_scale_shape = {};
 
// 输出
//按照接口定义的顺序排列
std::string g_output_tensor_dtype[128] = {};
std::string g_output_tensor_format[128] = {};
std::vector<int64_t> g_alltoallv_out_shape = {};
std::vector<int64_t> g_mm_out_shape = {};
 
// device数量
int g_ndev = 2 ;
int g_loop_cnt = 10;
 
// EP域和TP域大小
int ep_size = 8;
int tp_size = 1;
 
// 其余attr
std::string g_group_ep = "";
int64_t g_ep_world_size = 0;
 
bool g_trans_gmm_weight_flag = false;
bool g_trans_mm_weight_flag  = false; 
bool g_mm_out_flag  = false;
int rankId = 0;
int64_t g_hcclBufferSize = 200;

// 量化相关接口
int64_t g_gmm_x_quant_mode = 0;
int64_t g_gmm_weight_quant_mode = 0;
int64_t g_mm_x_quant_mode = 0;
int64_t g_mm_weight_quant_mode = 0;
int64_t g_comm_quant_mode = 0;
int64_t g_comm_quant_dtype = -1;
int64_t g_groupSize = 0;

void PrintMem(void* ptr, size_t len) {
  unsigned char* p = (unsigned char*)ptr;
  for (size_t i = 0; i < len; ++i) printf("%02x ", p[i]);
  printf("\n");
}

void GetOption(int argc, char **argv) {
    while (1) {
        int optionIndex = 0;
        struct option longOptions[] =
                {
                    {"gmm_x_shape",                    1, 0, 'a'},
                    {"gmm_weight_shape",               1, 0, 'b'},
                    {"send_counts_shape",              1, 0, 'c'},
                    {"recv_counts_shape",              1, 0, 'd'},
                    {"mm_x_shape",                     1, 0, 'e'},
                    {"mm_weight_shape",                1, 0, 'f'},
                    {"input_tensor_dtype",             1, 0, 'g'},
                    {"input_tensor_format",            1, 0, 'h'},
                    {"output_tensor_dtype",            1, 0, 'i'},
                    {"output_tensor_format",           1, 0, 'j'},
                    {"mm_output_shape",                1, 0, 'k'},
                    {"group_ep",                       1, 0, 'l'},
                    {"ep_world_size",                  1, 0, 'm'},
                    {"trans_gmm_weight_flag",          1, 0, 'n'},
                    {"trans_mm_weight_flag",           1, 0, 'o'},
                    {"n_dev",                          1, 0, 'p'},
                    {"platform",                       1, 0, 'q'},
                    {"bin_path",                       1, 0, 'r'},
                    {"loop_cnt",                       1, 0, 's'},
                    {"mm_out_flag",                    1, 0, 't'},
                    {"alltoallv_out_shape",            1, 0, 'u'},
                    {"rankId",                         1, 0, 'v'},
                    {"gmm_x_quant_mode",               1, 0, 'A'},  // 新增
                    {"gmm_weight_quant_mode",          1, 0, 'B'},  // 新增
                    {"mm_x_quant_mode",                1, 0, 'C'},  // 新增
                    {"mm_weight_quant_mode",           1, 0, 'D'},  // 新增
                    {"comm_quant_mode",                1, 0, 'E'},  // 新增
                    {"input_scales_dtype",             1, 0, 'F'},  // 新增
                    {"groupSize",                      1, 0, 'G'},  // 新增
                    {"gmm_x_scale_shape",              1, 0, 'H'},  // 新增
                    {"gmm_weight_scale_shape",         1, 0, 'I'},  // 新增
                    {"mm_x_scale_shape",               1, 0, 'J'},  // 新增
                    {"mm_weight_scale_shape",          1, 0, 'K'},  // 新增
                    {0,                                0, 0, 0  }
                };
        int c = getopt_long(argc, argv, "v:", longOptions, &optionIndex);
        if (c == -1) {
            break;
        }
        switch (c) {
            case 'a':
            {
                std::vector<std::string> gmm_x_shape;
                split(optarg, gmm_x_shape, ',');
                for (uint32_t i = 0; i < gmm_x_shape.size(); ++i) {
                    printf("gmm_x_shape[%d]=%s", i, gmm_x_shape[i].c_str() );
                    g_gmm_x_shape.push_back(atoi(gmm_x_shape[i].c_str()));
                }
            }
                break;
            case 'b': {
                std::vector<std::string> gmm_weight_shape;
                split(optarg, gmm_weight_shape, ',');
                for (uint32_t i = 0; i < gmm_weight_shape.size(); ++i) {
                    g_gmm_weight_shape.push_back(atoi(gmm_weight_shape[i].c_str()));
                }
            }
                break;
             case 'c': {
                std::vector<std::string> shape;
                split(optarg, shape, ',');
                for (uint32_t i = 0; i < shape.size(); ++i) {
                    g_send_counts_shape.push_back(atoi(shape[i].c_str()));
                }
            }
            break;
            case 'd': {
                std::vector<std::string> shape;
                split(optarg, shape, ',');
                for (uint32_t i = 0; i < shape.size(); ++i) {
                    g_recv_counts_shape.push_back(atoi(shape[i].c_str()));
                }
               
            }
                break;
            case 'e': {
                std::vector<std::string> mm_x_shape;
                split(optarg, mm_x_shape, ',');
                for (uint32_t i = 0; i < mm_x_shape.size(); ++i) {
                    g_mm_x_shape.push_back(atoi(mm_x_shape[i].c_str()));
                }
            }
                break;
            case 'f': {
                std::vector<std::string> mm_weight_shape;
                split(optarg, mm_weight_shape, ',');
                for (uint32_t i = 0; i < mm_weight_shape.size(); ++i) {
                    g_mm_weight_shape.push_back(atoi(mm_weight_shape[i].c_str()));
                }
            }
                break;
            case 'g': 
            {
                std::vector<std::string> dtypes;
                split(optarg, dtypes, ',');
                for (uint32_t i = 0; i < dtypes.size(); ++i) {
                    g_input_tensor_dtype[i] = dtypes[i];
                }
            }
                break;
            case 'h': 
            {
                 std::vector<std::string> formats;
                split(optarg, formats, ',');
                for (uint32_t i = 0; i < formats.size(); ++i) {
                    g_input_tensor_format[i] = formats[i];
                }
            }
                break;
            case 'i': 
            {
                std::vector<std::string> dtypes;
                split(optarg, dtypes, ',');
                for (uint32_t i = 0; i < dtypes.size(); ++i) {
                    g_output_tensor_dtype[i] = dtypes[i];
                }
            }
                break;
            case 'j': 
            {
                std::vector<std::string> formats;
                split(optarg, formats, ',');
                for (uint32_t i = 0; i < formats.size(); ++i) {
                    g_output_tensor_format[i] = formats[i];
                } 
            }
                break;
            case 'k': {
                 std::vector<std::string> mm_out_tensor_shape;
                split(optarg, mm_out_tensor_shape, ',');
                for (uint32_t i = 0; i < mm_out_tensor_shape.size(); ++i) {
                    g_mm_out_shape.push_back(atoi(mm_out_tensor_shape[i].c_str()));
                }
            }
                break;
            case 'l': 
                g_group_ep = string(optarg);
                break;
            case 'm':
                g_ep_world_size = atoi(optarg);
                break;
            case 'n':
                g_trans_gmm_weight_flag = stringToBool(optarg);
                break;
            case 'o':
                g_trans_mm_weight_flag = stringToBool(optarg);
                break;
            case 'p':
                g_ndev = atoi(optarg);
                break;
            case 'q':
                g_platform = string(optarg);
                break;
            case 'r':
                g_bin_path = string(optarg);   
                break;
            case 's':
                g_loop_cnt = atoi(optarg);
                break;
            case 't':
                g_mm_out_flag = stringToBool(optarg);
                break; 
            case 'u':{            
                std::vector<std::string> shape;
               split(optarg, shape, ',');
               for (uint32_t i = 0; i < shape.size(); ++i) {
                g_alltoallv_out_shape.push_back(atoi(shape[i].c_str()));
               }
           }
                break;
            case 'v':
                rankId = atoi(optarg);
                break;
            case 'A':
                g_gmm_x_quant_mode = atoi(optarg);
                break;
            case 'B':
                g_gmm_weight_quant_mode = atoi(optarg);
                break;
            case 'C':
                g_mm_x_quant_mode = atoi(optarg);
                break;
            case 'D':
                g_mm_weight_quant_mode = atoi(optarg);
                break;
            case 'E':
                g_comm_quant_mode = atoi(optarg);
                break;
            case 'F':
            {
                std::vector<std::string> dtypes;
                split(optarg, dtypes, ',');
                for (uint32_t i = 0; i < dtypes.size(); ++i) {
                    g_input_scales_dtype[i] = dtypes[i];
                }
            }
                break;
            case 'G':
                g_groupSize = atoi(optarg);
                break;
            case 'H':{
                std::vector<std::string> gmm_x_scale_shape;
                split(optarg, gmm_x_scale_shape, ',');
                for (uint32_t i = 0; i < gmm_x_scale_shape.size(); ++i) {
                    g_gmm_x_scale_shape.push_back(atoi(gmm_x_scale_shape[i].c_str()));
                }
            }
                break;
            case 'I':{
                std::vector<std::string> gmm_weight_scale_shape;
                split(optarg, gmm_weight_scale_shape, ',');
                for (uint32_t i = 0; i < gmm_weight_scale_shape.size(); ++i) {
                    g_gmm_weight_scale_shape.push_back(atoi(gmm_weight_scale_shape[i].c_str()));
                }
            }
                break;
            case 'J':{
                std::vector<std::string> mm_x_scale_shape;
                split(optarg, mm_x_scale_shape, ',');
                for (uint32_t i = 0; i < mm_x_scale_shape.size(); ++i) {
                    g_mm_x_scale_shape.push_back(atoi(mm_x_scale_shape[i].c_str()));
                }
            }
                break;
            case 'K':{
                std::vector<std::string> mm_weight_scale_shape;
                split(optarg, mm_weight_scale_shape, ',');
                for (uint32_t i = 0; i < mm_weight_scale_shape.size(); ++i) {
                    g_mm_weight_scale_shape.push_back(atoi(mm_weight_scale_shape[i].c_str()));
                }
            }
                break;
            default:
                break;
        }
    }
}

int launchOneThread(Args &args) {
    int ret;
    ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
    char hcomEpName[128] = {0};
    ret = HcclGetCommName(args.hcclComm, hcomEpName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetEpCommName failed. ret: %d\n", ret); return -1);
    LOG_PRINT("[INFO] rank = %d, hcomEpName = %s, stream = %p, context = %p\n", args.rankId, hcomEpName, args.stream, args.context);
    timespec tp1, tp2;
    long cost;
    
    std::vector<int64_t> gmm_x_shape       = g_gmm_x_shape;
    std::vector<int64_t> gmm_weight_shape  = g_gmm_weight_shape;
    std::vector<int64_t> gmm_x_scale_shape = g_gmm_x_scale_shape;
    std::vector<int64_t> gmm_weight_scale_shape = g_gmm_weight_scale_shape;
    std::vector<int64_t> mm_x_shape        = g_mm_x_shape;
    std::vector<int64_t> mm_weight_shape   = g_mm_weight_shape;
    std::vector<int64_t> mm_x_scale_shape = g_mm_x_scale_shape;
    std::vector<int64_t> mm_weight_scale_shape = g_mm_weight_scale_shape;
    std::vector<int64_t> send_counts_shape = g_send_counts_shape;
    std::vector<int64_t> recv_counts_shape = g_recv_counts_shape;
    std::vector<int64_t> mm_out_shape      = g_mm_out_shape;
    std::vector<int64_t> alltoallv_out_shape = g_alltoallv_out_shape;   
 
    g_group_ep = hcomEpName;
    void *gmm_x_device_addr       = nullptr;
    void *gmm_weight_device_addr  = nullptr;
    void *send_counts_device_addr = nullptr;
    void *recv_counts_device_addr = nullptr;
    void *trans_gmm_weight_device_addr = nullptr;
    void *trans_mm_weight_device_addr  = nullptr;
    void *mm_x_device_addr        = nullptr;
    void *mm_weight_device_addr   = nullptr;
    void *alltoallv_out_device_addr    = nullptr;
    void *mm_out_device_addr      = nullptr;
    void *workspaceAddr           = nullptr;
 
    std::string gmm_x_dtype              = g_input_tensor_dtype[0];
    std::string gmm_weight_dtype         = g_input_tensor_dtype[1];
    std::string send_counts_dtype        = g_input_tensor_dtype[2];
    std::string recv_counts_dtype        = g_input_tensor_dtype[3];
    std::string mm_x_dtype               = g_input_tensor_dtype[4];
    std::string mm_weight_dtype          = g_input_tensor_dtype[5];
    std::string gmm_x_scale_dtype        = g_input_scales_dtype[0];
    std::string gmm_weight_scale_dtype   = g_input_scales_dtype[1];
    std::string mm_x_scale_dtype         = g_input_scales_dtype[2];
    std::string mm_weight_scale_dtype    = g_input_scales_dtype[3];
    std::string comm_quant_scale_dtype   = g_input_scales_dtype[4];

    std::string gmm_x_format       = g_input_tensor_format[0];
    std::string gmm_weight_format  = g_input_tensor_format[1];
    std::string send_counts_format = g_input_tensor_format[2];
    std::string recv_counts_format = g_input_tensor_format[3];
    std::string mm_x_format        = g_input_tensor_format[4];
    std::string mm_weight_format   = g_input_tensor_format[5];
 
    std::string alltoallv_out_dtype = g_output_tensor_dtype[0];
    std::string mm_out_dtype = g_output_tensor_dtype[1];
 
    std::string alltoallv_out_format = g_output_tensor_format[0];
    std::string mm_out_format = g_output_tensor_format[1];
    int64_t send_counts_len = g_send_counts_shape[0];
    int64_t recv_counts_len = g_recv_counts_shape[0];
    
    // 量化参数（仅在量化模式下使用）
    void *gmm_x_scale_device_addr   = nullptr;
    void *gmm_weight_scale_device_addr = nullptr;
    void *mm_x_scale_device_addr    = nullptr;
    void *mm_weight_scale_device_addr = nullptr;
    void *comm_quant_scale_device_addr = nullptr;
 
    aclTensor *gmm_x       = nullptr;
    aclTensor *gmm_weight  = nullptr;
    aclTensor *mm_x        = nullptr;
    aclTensor *mm_weight   = nullptr;
    aclTensor *send_counts_tensor = nullptr;
    aclTensor *recv_counts_tensor = nullptr;
    aclTensor *trans_gmm_weight = nullptr;
    aclTensor *trans_mm_weight  = nullptr;
    aclTensor *alltoallv_out      = nullptr;
    aclTensor *mm_out      = nullptr;
    aclTensor *all_toallv_out   = nullptr;
 
    aclTensor *gmm_x_scale   = nullptr;
    aclTensor *gmm_weight_scale = nullptr;
    aclTensor *mm_x_scale    = nullptr;
    aclTensor *mm_weight_scale = nullptr;
    aclTensor *comm_quant_scale  = nullptr;
 
    aclIntArray *send_counts_arr = nullptr;
    aclIntArray *recv_counts_arr = nullptr;
 
    uint64_t workspaceSize = 0;
    vector<int64_t> send_counts_vec(send_counts_shape[0], 0);
    vector<int64_t> recv_counts_vec(recv_counts_shape[0], 0);

    printf("rank :%d gmm_x_shape [0]:%ld\n",args.rankId,gmm_x_shape[0]);
    printf("rank :%d gmm_x_shape [1]:%ld\n",args.rankId,gmm_x_shape[1]);
 
    printf("rank :%d alltoallv_out_shape [0]:%ld\n",args.rankId,alltoallv_out_shape[0]);
    printf("rank :%d alltoallv_out_shape [1]:%ld\n",args.rankId,alltoallv_out_shape[1]);
 
    aclOpExecutor *executor = nullptr;
    // 创建 tensor
    ACL_CHECK(CreateTensor(gmm_x_shape, aclFormat::ACL_FORMAT_ND, &gmm_x_device_addr, &gmm_x, gmm_x_dtype, g_bin_path, args.rankId, "gmm_x"));
    ACL_CHECK(CreateTensor(gmm_weight_shape, aclFormat::ACL_FORMAT_ND, &gmm_weight_device_addr, &gmm_weight, gmm_weight_dtype, g_bin_path, args.rankId, "gmm_weight"));
    ACL_CHECK(CreateTensor(gmm_x_scale_shape, aclFormat::ACL_FORMAT_ND, &gmm_x_scale_device_addr, &gmm_x_scale, gmm_x_scale_dtype, g_bin_path, args.rankId, "gmm_x_scale"));
    ACL_CHECK(CreateTensor(gmm_weight_scale_shape, aclFormat::ACL_FORMAT_ND, &gmm_weight_scale_device_addr, &gmm_weight_scale, gmm_weight_scale_dtype, g_bin_path, args.rankId, "gmm_weight_scale"));
    ACL_CHECK(CreateTensor(send_counts_shape, aclFormat::ACL_FORMAT_ND, &send_counts_device_addr, &send_counts_tensor, send_counts_dtype, g_bin_path, args.rankId, "send_counts"));
    ACL_CHECK(CreateTensor(recv_counts_shape, aclFormat::ACL_FORMAT_ND, &recv_counts_device_addr, &recv_counts_tensor, recv_counts_dtype, g_bin_path, args.rankId, "recv_counts"));

    if(g_mm_out_flag){
        ACL_CHECK(CreateTensor(mm_x_shape, aclFormat::ACL_FORMAT_ND, &mm_x_device_addr, &mm_x, mm_x_dtype, g_bin_path, args.rankId, "mm_x"));
        ACL_CHECK(CreateTensor(mm_weight_shape, aclFormat::ACL_FORMAT_ND, &mm_weight_device_addr, &mm_weight, mm_weight_dtype, g_bin_path, args.rankId, "mm_weight"));
        ACL_CHECK(CreateTensor(mm_x_scale_shape, aclFormat::ACL_FORMAT_ND, &mm_x_scale_device_addr, &mm_x_scale, mm_x_scale_dtype, g_bin_path, args.rankId, "mm_x_scale"));
        ACL_CHECK(CreateTensor(mm_weight_scale_shape, aclFormat::ACL_FORMAT_ND, &mm_weight_scale_device_addr, &mm_weight_scale, mm_weight_scale_dtype, g_bin_path, args.rankId, "mm_weight_scale"));

    }

    ACL_CHECK(CreateIntArrayFromFile(send_counts_vec,&send_counts_arr, g_bin_path,"send_counts",args.rankId,send_counts_dtype));

    ACL_CHECK(CreateIntArrayFromFile(recv_counts_vec,&recv_counts_arr, g_bin_path,"recv_counts",args.rankId,recv_counts_dtype));

   
    auto alltoallv_out_size = GetShapeSize(alltoallv_out_shape) * dtype_to_sizeof(alltoallv_out_dtype);
    auto mm_out_size  = GetShapeSize(mm_out_shape) * dtype_to_sizeof(mm_out_dtype);
    
    ACL_CHECK(CreateTensorWithoutHostData(alltoallv_out_shape, aclFormat::ACL_FORMAT_ND, &alltoallv_out_device_addr, &alltoallv_out, alltoallv_out_dtype));
    if (g_mm_out_flag) { ACL_CHECK(CreateTensorWithoutHostData(mm_out_shape, aclFormat::ACL_FORMAT_ND, &mm_out_device_addr, &mm_out, mm_out_dtype));}
    
    // 内存校验
    aclrtMemAttr attr = ACL_HBM_MEM;
    size_t mem_free;
    size_t mem_total;
    auto ret_before = aclrtGetMemInfo(attr, &mem_free, &mem_total);
    size_t mem_used_before = (mem_total - mem_free) / 1024 / 1024;
    printf(" before RunGraph : free :%ld M,  total:%ld M, used :%ld M, ret :%d \n", mem_free / 1024 / 1024,
           mem_total / 1024 / 1024, mem_used_before, ret_before);
    printf(" g_loop_cnt is %d \n", g_loop_cnt);
    for (int i = 0; i < g_loop_cnt; i++) {
        printf(" clock_gettime init \n");
        clock_gettime(CLOCK_MONOTONIC, &tp1);
        printf(" clock_gettime end \n");
        // 全量化模式：必须传入所有 scale 和 quant_mode 参数
        printf(" aclnnQuantGroupedMatMulAlltoAllvGetWorkspaceSize begin.. \n");
        ret = aclnnQuantGroupedMatMulAlltoAllvGetWorkspaceSize(
            gmm_x, gmm_weight,
            gmm_x_scale, gmm_weight_scale,
            nullptr, nullptr,
            mm_x, mm_weight,
            mm_x_scale, mm_weight_scale,
            nullptr, 
            g_gmm_x_quant_mode, g_gmm_weight_quant_mode, g_mm_x_quant_mode, g_mm_weight_quant_mode, 0, -1, g_groupSize,
            g_group_ep.c_str(), g_ep_world_size,
            send_counts_arr, recv_counts_arr,
            g_trans_gmm_weight_flag, g_trans_mm_weight_flag,
            alltoallv_out, mm_out, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS, printf("[ERROR] aclnnQuantGroupedMatMulAlltoAllvGetWorkspaceSize failed. ret = %d \n", ret); return ret);
        printf(" once stage sucess \n");
        std::cout << "once stage sucess" << std::endl;
        printf(" workspaceSize init \n");
        std::cout << "workspaceSize init" << std::endl;
        if (workspaceSize > 0 &&i == 0) {
            ACL_CHECK(aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
        }
        printf(" second stage start \n");
        std::cout << "second stage start" << std::endl;
        // 调用第二段接口
        printf(" second stage start 2\n");
        ret = aclnnQuantGroupedMatMulAlltoAllv(workspaceAddr, workspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] check second stage failed. ret = %d \n", ret);return ret);
        printf(" second stage success \n");
        // 同步等待任务执行结束
        ACL_CHECK(aclrtSynchronizeStreamWithTimeout(args.stream, 20000));
        clock_gettime(CLOCK_MONOTONIC, &tp2);
        if (tp2.tv_sec != tp1.tv_sec) {
            cost = tp2.tv_nsec - tp1.tv_nsec + 1000000000;
        }
        else {
            cost = tp2.tv_nsec - tp1.tv_nsec;
        }
        printf("[INFO] device:%d step:%d cost_time: %lu, \n", args.rankId, i, cost);
    }
    printf("[INFO] workspaceSize = %lu\n", workspaceSize);
    // 内存校验
    auto ret_after = aclrtGetMemInfo(attr, &mem_free, &mem_total);
    size_t mem_used_after = (mem_total - mem_free) / 1024 / 1024;
    printf("last run : free :%lu M,  total:%lu M, used :%lu M, ret :%d \n",
           mem_free / 1024 / 1024, mem_total / 1024 / 1024, mem_used_after, ret_after);
    printf("rank %d op mc2 cost memory :%lu M \n", args.rankId, mem_used_after - mem_used_before);
 
    //保存结果
    saveOutPut(g_bin_path, alltoallv_out_size, args.rankId, alltoallv_out_device_addr, "npu_out.bin");
    if (g_mm_out_flag)
    {
        saveOutPut(g_bin_path, mm_out_size, args.rankId, mm_out_device_addr, "npu_mm_out.bin");
    }
    
    // 释放资源
    if (gmm_x != nullptr){aclDestroyTensor(gmm_x);}
    if (gmm_weight != nullptr) {aclDestroyTensor(gmm_weight);}
    if (trans_gmm_weight != nullptr) {aclDestroyTensor(trans_gmm_weight);}
    if (send_counts_tensor != nullptr) {aclDestroyTensor(send_counts_tensor);}
    if (recv_counts_tensor != nullptr) {aclDestroyTensor(recv_counts_tensor);}
    if (mm_x != nullptr) {aclDestroyTensor(mm_x);}
    if (mm_weight != nullptr){aclDestroyTensor(mm_weight);}
    if (trans_mm_weight != nullptr) {aclDestroyTensor(trans_mm_weight);}
    if (mm_out != nullptr) {aclDestroyTensor(mm_out);}
    if (all_toallv_out != nullptr) {aclDestroyTensor(all_toallv_out);}
 
    if (gmm_x_scale != nullptr) {aclDestroyTensor(gmm_x_scale);}
    if (gmm_weight_scale != nullptr) {aclDestroyTensor(gmm_weight_scale);}
    if (mm_x_scale != nullptr) {aclDestroyTensor(mm_x_scale);}
    if (mm_weight_scale != nullptr) {aclDestroyTensor(mm_weight_scale);}
    if (comm_quant_scale != nullptr) {aclDestroyTensor(comm_quant_scale);}
 
    if (gmm_x_device_addr != nullptr) {aclrtFree(gmm_x_device_addr);}
    if (gmm_weight_device_addr != nullptr) {aclrtFree(gmm_weight_device_addr); }
    if (send_counts_device_addr != nullptr) {aclrtFree(send_counts_device_addr);}
    if (recv_counts_device_addr != nullptr) {aclrtFree(recv_counts_device_addr);}
    if (mm_x_device_addr != nullptr) {aclrtFree(mm_x_device_addr);}
    if (mm_weight_device_addr != nullptr) {aclrtFree(mm_weight_device_addr);}
    if (mm_out_device_addr != nullptr) {aclrtFree(mm_out_device_addr);}
    if (alltoallv_out_device_addr != nullptr) { aclrtFree(alltoallv_out_device_addr);}
 
    if (gmm_x_scale_device_addr != nullptr) {aclrtFree(gmm_x_scale_device_addr);}
    if (gmm_weight_scale_device_addr != nullptr) {aclrtFree(gmm_weight_scale_device_addr);}
    if (mm_x_scale_device_addr != nullptr) {aclrtFree(mm_x_scale_device_addr);}
    if (mm_weight_scale_device_addr != nullptr) {aclrtFree(mm_weight_scale_device_addr);}
    if (trans_gmm_weight_device_addr != nullptr) {aclrtFree(trans_gmm_weight_device_addr);}
    if (trans_mm_weight_device_addr != nullptr) {aclrtFree(trans_mm_weight_device_addr);}
    if (comm_quant_scale_device_addr != nullptr) {aclrtFree(comm_quant_scale_device_addr);}
 
    if (workspaceSize > 0) {aclrtFree(workspaceAddr);}
    if(send_counts_arr != NULL); {aclDestroyIntArray(send_counts_arr);}
    if(recv_counts_arr != NULL ) {aclDestroyIntArray(recv_counts_arr);}
 
    ACL_CHECK(aclrtDestroyStreamForce(args.stream));
    HcclCommDestroy(args.hcclComm);
    ACL_CHECK(aclrtDestroyContext(args.context));
    ACL_CHECK(aclrtResetDevice(args.rankId));
    return 0;
}
 
int main(int argc, char *argv[]) {
    GetOption(argc, argv);
    // for mx quant
    if (g_gmm_x_quant_mode == 6 && g_gmm_weight_quant_mode == 6) {
        g_groupSize = 4295032864LL;  
        printf("[INFO] Detected gmm_x_quant_mode == 6 and gmm_weight_quant_mode == 6, forcing g_groupSize = %ld\n", g_groupSize);
    }
    int ret;
    int32_t devices[g_ndev];
    for (int i = 0; i < g_ndev; i++) {
        devices[i] = i;
    }
    ACL_CHECK(aclInit(nullptr));
    // 启动多线程
    Args args;
    aclrtStream stream;
    aclrtContext context;
    HcclComm comms;
    printf("----start aclrtSetDevice rankId = %d\n", rankId);
    ret = aclrtSetDevice(rankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    printf("----start aclrtCreateContext\n");
    ret = aclrtCreateContext(&context, rankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
    printf("----start aclrtCreateStream\n");
    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    HcclCommConfig config;
    HcclCommConfigInit(&config);
    config.hcclDeterministic = 1;
    config.hcclBufferSize = g_hcclBufferSize;
    printf("----start strncpy hcclCommName \n");
    strncpy(config.hcclCommName, "hccl_comm_test", COMM_NAME_MAX_LENGTH - 1);
    printf("----start getenv from RANK_TABLE_FILE \n");
    string rankTableFile = getenv("RANK_TABLE_FILE");
    std::cout << "rankTableFilePath is :" << rankTableFile << std::endl;
    // int rank_id = rankId + FIRST_RANK_ID;
    printf("----start HcclCommInitClusterInfoConfig \n");
    ret = HcclCommInitClusterInfoConfig(rankTableFile.c_str(), rankId, &config, &comms);
    CHECK_RET(
        ret == ACL_SUCCESS,
        LOG_PRINT("[ERROR] HcclCommInitClusterInfoConfig ep world %d failed. ret = %d\n", rankId, ret); return ret
    );
    args.rankId = rankId;
    args.hcclComm = comms;
    args.stream = stream;
    args.context = context;
    LOG_PRINT("before launchOneThread");
    int re = launchOneThread(args);
    LOG_PRINT("end launchOneThread re = %d", re);
    ACL_CHECK(aclFinalize());
    std::cout << "rankTableFilePath is :" << rankTableFile << std::endl;
    return 0;
}