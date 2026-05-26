#include "common.h"
#include <cstring>
#include "aclnnop/aclnn_allto_allv_quant_grouped_mat_mul.h"  // 新增：全量化头文件
using namespace std;

// 定义全局变量
std::string g_platform = "1971";
std::string g_bin_path = "./";

// 输入
std::string g_input_tensor_dtype[128] = {};
std::string g_input_scales_dtype[128] = {};
std::string g_input_tensor_format[128] = {};
std::string g_input_scales_format[128] = {};

// 按照接口定义的顺序排列
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
std::string g_output_tensor_dtype[128] = {};
std::string g_output_tensor_format[128] = {};
std::vector<int64_t> g_gmm_out_shape = {};
std::vector<int64_t> g_mm_out_shape = {};
std::vector<int64_t> g_permute_out_shape = {};

// device数量
int g_ndev = 2;
int g_loop_cnt = 10;

// EP域和TP域大小
int ep_size = 8;
int tp_size = 1;

// 其余attr
std::string g_group_ep = "group";
int64_t g_ep_world_size = 0;
bool g_trans_gmm_weight_flag = false;
bool g_trans_mm_weight_flag = false;
bool g_permute_out_flag = false;
bool g_mm_out_flag = false;

int rankId = 0;
int64_t g_hcclBufferSize = 200;
// 新增：量化模式参数（0=非量化，1=PerTensor）
int64_t g_gmm_x_quant_mode = 0;
int64_t g_gmm_weight_quant_mode = 0;
int64_t g_mm_x_quant_mode = 0;
int64_t g_mm_weight_quant_mode = 0;

void GetOption(int argc, char **argv)
{
    while (1) {
        int optionIndex = 0;
        struct option longOptions[] = {
            {"bin_path", 1, 0, 'a'},
            {"gmm_x_shape", 1, 0, 'b'},
            {"gmm_weight_shape", 1, 0, 'c'},
            {"gmm_x_scale_shape", 1, 0, 'd'},
            {"gmm_weight_scale_shape", 1, 0, 'e'},
            {"send_counts_shape", 1, 0, 'f'},
            {"recv_counts_shape", 1, 0, 'g'},
            {"mm_x_shape", 1, 0, 'h'},
            {"mm_weight_shape", 1, 0, 'i'},
            {"mm_x_scale_shape", 1, 0, 'j'},
            {"mm_weight_scale_shape", 1, 0, 'k'},
            {"input_tensor_dtype", 1, 0, 'l'},
            {"input_tensor_format", 1, 0, 'm'},
            {"input_scales_dtype", 1, 0, 'n'},
            {"input_scales_format", 1, 0, 'o'},
            {"output_tensor_dtype", 1, 0, 'p'},
            {"output_tensor_format", 1, 0, 'q'},
            {"ep_world_size", 1, 0, 'r'},
            {"trans_gmm_weight_flag", 1, 0, 's'},
            {"trans_mm_weight_flag", 1, 0, 't'},
            {"n_dev", 1, 0, 'u'},
            {"loop_cnt", 1, 0, 'v'},
            {"mm_out_flag", 1, 0, 'w'},
            {"permute_out_flag", 1, 0, 'x'},
            {"gmm_x_quant_mode", 1, 0, 'y'},       
            {"gmm_weight_quant_mode", 1, 0, 'z'},  
            {"mm_x_quant_mode", 1, 0, 'A'},        
            {"mm_weight_quant_mode", 1, 0, 'B'},
            {"mm_y_shape", 1, 0, 'C'},
            {"gmm_y_shape", 1, 0, 'D'},
            {"permute_out_shape", 1, 0, 'E'},
            {"rankId", 1, 0, 'F'},
            {0, 0, 0, 0}};
        int c = getopt_long(argc, argv, "v:", longOptions, &optionIndex);
        if (c == -1) {
            break;
        }
        switch (c) {
            case 'a':
                g_bin_path = string(optarg);
                break;
            case 'b': {
                std::vector<std::string> gmm_x_shape;
                split(optarg, gmm_x_shape, ',');
                for (uint32_t i = 0; i < gmm_x_shape.size(); ++i) {
                    g_gmm_x_shape.push_back(atoi(gmm_x_shape[i].c_str()));
                }
            } break;
            case 'c': {
                std::vector<std::string> gmm_weight_shape;
                split(optarg, gmm_weight_shape, ',');
                for (uint32_t i = 0; i < gmm_weight_shape.size(); ++i) {
                    g_gmm_weight_shape.push_back(atoi(gmm_weight_shape[i].c_str()));
                }
            } break;
            case 'd': {
                std::vector<std::string> gmm_x_scale_shape;
                split(optarg, gmm_x_scale_shape, ',');
                for (uint32_t i = 0; i < gmm_x_scale_shape.size(); ++i) {
                    g_gmm_x_scale_shape.push_back(atoi(gmm_x_scale_shape[i].c_str()));
                }
            } break;
            case 'e': {
                std::vector<std::string> gmm_weight_scale_shape;
                split(optarg, gmm_weight_scale_shape, ',');
                for (uint32_t i = 0; i < gmm_weight_scale_shape.size(); ++i) {
                    g_gmm_weight_scale_shape.push_back(atoi(gmm_weight_scale_shape[i].c_str()));
                }
            } break;
            case 'f': {
                std::vector<std::string> send_shape;
                split(optarg, send_shape, ',');
                for (uint32_t i = 0; i < send_shape.size(); ++i) {
                    g_send_counts_shape.push_back(atoi(send_shape[i].c_str()));
                }
            } break;
            case 'g': {
                std::vector<std::string> recv_shape;
                split(optarg, recv_shape, ',');
                for (uint32_t i = 0; i < recv_shape.size(); ++i) {
                    g_recv_counts_shape.push_back(atoi(recv_shape[i].c_str()));
                }
            } break;
            case 'h': {
                std::vector<std::string> mm_x_shape;
                split(optarg, mm_x_shape, ',');
                for (uint32_t i = 0; i < mm_x_shape.size(); ++i) {
                    g_mm_x_shape.push_back(atoi(mm_x_shape[i].c_str()));
                }
            } break;
            case 'i': {
                std::vector<std::string> mm_weight_shape;
                split(optarg, mm_weight_shape, ',');
                for (uint32_t i = 0; i < mm_weight_shape.size(); ++i) {
                    g_mm_weight_shape.push_back(atoi(mm_weight_shape[i].c_str()));
                }
            } break;
            case 'j': {
                std::vector<std::string> mm_x_scale_shape;
                split(optarg, mm_x_scale_shape, ',');
                for (uint32_t i = 0; i < mm_x_scale_shape.size(); ++i) {
                    g_mm_x_scale_shape.push_back(atoi(mm_x_scale_shape[i].c_str()));
                }
            } break;
            case 'k': {
                std::vector<std::string> mm_weight_scale_shape;
                split(optarg, mm_weight_scale_shape, ',');
                for (uint32_t i = 0; i < mm_weight_scale_shape.size(); ++i) {
                    g_mm_weight_scale_shape.push_back(atoi(mm_weight_scale_shape[i].c_str()));
                }
            } break;
            case 'l': {
                std::vector<std::string> dtypes;
                split(optarg, dtypes, ',');
                for (uint32_t i = 0; i < dtypes.size(); ++i) {
                    g_input_tensor_dtype[i] = dtypes[i];
                }
            } break;
            case 'm': {
                std::vector<std::string> formats;
                split(optarg, formats, ',');
                for (uint32_t i = 0; i < formats.size(); ++i) {
                    g_input_tensor_format[i] = formats[i];
                }
            } break;
            case 'n': {
                std::vector<std::string> dtypes;
                split(optarg, dtypes, ',');
                for (uint32_t i = 0; i < dtypes.size(); ++i) {
                    g_input_scales_dtype[i] = dtypes[i];
                }
            } break;
            case 'o': {
                std::vector<std::string> formats;
                split(optarg, formats, ',');
                for (uint32_t i = 0; i < formats.size(); ++i) {
                    g_input_scales_format[i] = formats[i];
                }
            } break;
            case 'p': {
                std::vector<std::string> dtypes;
                split(optarg, dtypes, ',');
                for (uint32_t i = 0; i < dtypes.size(); ++i) {
                    g_output_tensor_dtype[i] = dtypes[i];
                }
            } break;
            case 'q': {
                std::vector<std::string> formats;
                split(optarg, formats, ',');
                for (uint32_t i = 0; i < formats.size(); ++i) {
                    g_output_tensor_format[i] = formats[i];
                }
            } break;
            case 'r':
                g_ep_world_size = atoi(optarg);
                break;
            case 's':
                g_trans_gmm_weight_flag = stringToBool(optarg);
                break;
            case 't':
                g_trans_mm_weight_flag = stringToBool(optarg);
                break;
            case 'u':
                g_ndev = atoi(optarg);
                break;
            case 'v':
                g_loop_cnt = atoi(optarg);
                break;
            case 'w':
                g_mm_out_flag = stringToBool(optarg);
                break;
            case 'x':
                g_permute_out_flag = stringToBool(optarg);
                break;
            case 'y':
                g_gmm_x_quant_mode = atoi(optarg);
                break;
            case 'z':
                g_gmm_weight_quant_mode = atoi(optarg);
                break;
            case 'A':
                g_mm_x_quant_mode = atoi(optarg);
                break;
            case 'B':
                g_mm_weight_quant_mode = atoi(optarg);
                break;
            case 'C': {
                std::vector<std::string> mm_out_tensor_shape;
                split(optarg, mm_out_tensor_shape, ',');
                for (uint32_t i = 0; i < mm_out_tensor_shape.size(); ++i) {
                    g_mm_out_shape.push_back(atoi(mm_out_tensor_shape[i].c_str()));
                }
            } break;
            case 'D': {
                std::vector<std::string> shape;
                split(optarg, shape, ',');
                for (uint32_t i = 0; i < shape.size(); ++i) {
                    g_gmm_out_shape.push_back(atoi(shape[i].c_str()));
                }
            } break;
            case 'E': {
                std::vector<std::string> out_tensor_shape;
                split(optarg, out_tensor_shape, ',');
                for (uint32_t i = 0; i < out_tensor_shape.size(); ++i) {
                    g_permute_out_shape.push_back(atoi(out_tensor_shape[i].c_str()));
                }
            } break;
            case 'F':
                rankId = atoi(optarg);
                break;
            default:
                break;
        }
    }
}

int launchOneThread(Args &args)
{
    int ret;
    ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);

    char hcomEpName[128] = {0};
    ret = HcclGetCommName(args.hcclComm, hcomEpName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetEpCommName failed. ret: %d\n", ret); return -1);
    LOG_PRINT("[INFO] rank = %d, hcomEpName = %s, stream = %p, context = %p\n",
        args.rankId,
        hcomEpName,
        args.stream,
        args.context);

    timespec tp1, tp2;
    long cost;

    // 输入 tensor shape
    std::vector<int64_t> gmm_x_shape = g_gmm_x_shape;
    std::vector<int64_t> gmm_weight_shape = g_gmm_weight_shape;
    std::vector<int64_t> gmm_x_scale_shape = g_gmm_x_scale_shape;
    std::vector<int64_t> gmm_weight_scale_shape = g_gmm_weight_scale_shape;
    std::vector<int64_t> mm_x_shape = g_mm_x_shape;
    std::vector<int64_t> mm_weight_shape = g_mm_weight_shape;
    std::vector<int64_t> mm_x_scale_shape = g_mm_x_scale_shape;
    std::vector<int64_t> mm_weight_scale_shape = g_mm_weight_scale_shape;
    std::vector<int64_t> send_counts_shape = g_send_counts_shape;
    std::vector<int64_t> recv_counts_shape = g_recv_counts_shape;

    // 输出 tensor shape
    std::vector<int64_t> gmm_out_shape = g_gmm_out_shape;
    std::vector<int64_t> mm_out_shape = g_mm_out_shape;
    std::vector<int64_t> permute_out_shape = g_permute_out_shape;

    g_group_ep = hcomEpName;

    // 定义所有 tensor 指针
    void *gmm_x_device_addr = nullptr;
    void *gmm_weight_device_addr = nullptr;
    void *gmm_x_scale_device_addr = nullptr;
    void *gmm_weight_scale_device_addr = nullptr;
    void *send_counts_device_addr = nullptr;
    void *recv_counts_device_addr = nullptr;
    void *mm_x_device_addr = nullptr;
    void *mm_weight_device_addr = nullptr;
    void *mm_x_scale_device_addr = nullptr;
    void *mm_weight_scale_device_addr = nullptr;
    void *gmm_out_device_addr = nullptr;
    void *mm_out_device_addr = nullptr;
    void *workspaceAddr = nullptr;
    void *permute_out_device_addr = nullptr;

    aclTensor *gmm_x = nullptr;
    aclTensor *gmm_weight = nullptr;
    aclTensor *gmm_x_scale = nullptr;
    aclTensor *gmm_weight_scale = nullptr;
    aclTensor *mm_x = nullptr;
    aclTensor *mm_weight = nullptr;
    aclTensor *mm_x_scale = nullptr;
    aclTensor *mm_weight_scale = nullptr;
    aclTensor *gmm_out = nullptr;
    aclTensor *mm_out = nullptr;
    aclTensor *permute_out = nullptr;
    aclTensor *send_counts_tensor = nullptr;
    aclTensor *recv_counts_tensor = nullptr;
    aclIntArray *send_counts_arr = nullptr;
    aclIntArray *recv_counts_arr = nullptr;

    uint64_t workspaceSize = 0;
    vector<int64_t> send_counts_vec(send_counts_shape[0], 0);
    vector<int64_t> recv_counts_vec(recv_counts_shape[0], 0);

    // 数据类型和格式
    std::string gmm_x_dtype = g_input_tensor_dtype[0];
    std::string gmm_weight_dtype = g_input_tensor_dtype[1];
    std::string send_counts_dtype = g_input_tensor_dtype[2];
    std::string recv_counts_dtype = g_input_tensor_dtype[3];
    std::string mm_x_dtype = g_input_tensor_dtype[4];
    std::string mm_weight_dtype = g_input_tensor_dtype[5];
    std::string gmm_x_scale_dtype = g_input_scales_dtype[0];
    std::string gmm_weight_scale_dtype = g_input_scales_dtype[1];
    std::string mm_x_scale_dtype = g_input_scales_dtype[2];
    std::string mm_weight_scale_dtype = g_input_scales_dtype[3];

    std::string gmm_x_format = g_input_tensor_format[0];
    std::string gmm_weight_format = g_input_tensor_format[1];
    std::string send_counts_format = g_input_tensor_format[2];
    std::string recv_counts_format = g_input_tensor_format[3];
    std::string mm_x_format = g_input_tensor_format[4];
    std::string mm_weight_format = g_input_tensor_format[5];

    std::string gmm_out_dtype = g_output_tensor_dtype[0];
    std::string mm_out_dtype = g_output_tensor_dtype[1];
    std::string permute_out_dtype = g_output_tensor_dtype[2];

    std::string gmm_out_format = g_output_tensor_format[0];
    std::string permute_out_format = g_output_tensor_format[1];
    std::string mm_out_format = g_output_tensor_format[2];

    int64_t send_counts_len = g_send_counts_shape[0];
    int64_t recv_counts_len = g_recv_counts_shape[0];

    aclOpExecutor *executor = nullptr;
    // 创建 tensor
    ACL_CHECK(CreateTensor(gmm_x_shape,
        aclFormat::ACL_FORMAT_ND,
        &gmm_x_device_addr,
        &gmm_x,
        gmm_x_dtype,
        g_bin_path,
        args.rankId,
        "gmm_x"));
    ACL_CHECK(CreateTensor(gmm_weight_shape,
        aclFormat::ACL_FORMAT_ND,
        &gmm_weight_device_addr,
        &gmm_weight,
        gmm_weight_dtype,
        g_bin_path,
        args.rankId,
        "gmm_weight"));
    ACL_CHECK(CreateTensor(gmm_x_scale_shape,
        aclFormat::ACL_FORMAT_ND,
        &gmm_x_scale_device_addr,
        &gmm_x_scale,
        gmm_x_scale_dtype,
        g_bin_path,
        args.rankId,
        "gmm_x_scale"));
    ACL_CHECK(CreateTensor(gmm_weight_scale_shape,
        aclFormat::ACL_FORMAT_ND,
        &gmm_weight_scale_device_addr,
        &gmm_weight_scale,
        gmm_weight_scale_dtype,
        g_bin_path,
        args.rankId,
        "gmm_weight_scale"));

    if (g_mm_out_flag) {
        ACL_CHECK(CreateTensor(mm_x_shape,
            aclFormat::ACL_FORMAT_ND,
            &mm_x_device_addr,
            &mm_x,
            mm_x_dtype,
            g_bin_path,
            args.rankId,
            "mm_x"));
        ACL_CHECK(CreateTensor(mm_weight_shape,
            aclFormat::ACL_FORMAT_ND,
            &mm_weight_device_addr,
            &mm_weight,
            mm_weight_dtype,
            g_bin_path,
            args.rankId,
            "mm_weight"));
        ACL_CHECK(CreateTensor(mm_x_scale_shape,
            aclFormat::ACL_FORMAT_ND,
            &mm_x_scale_device_addr,
            &mm_x_scale,
            mm_x_scale_dtype,
            g_bin_path,
            args.rankId,
            "mm_x_scale"));
        ACL_CHECK(CreateTensor(mm_weight_scale_shape,
            aclFormat::ACL_FORMAT_ND,
            &mm_weight_scale_device_addr,
            &mm_weight_scale,
            mm_weight_scale_dtype,
            g_bin_path,
            args.rankId,
            "mm_weight_scale"));
    }
    
    ACL_CHECK(CreateIntArrayFromFile(
        send_counts_vec, &send_counts_arr, g_bin_path, "send_counts", args.rankId, send_counts_dtype));
    ACL_CHECK(CreateIntArrayFromFile(
        recv_counts_vec, &recv_counts_arr, g_bin_path, "recv_counts", args.rankId, recv_counts_dtype));

    // 创建输出 tensor
    ACL_CHECK(CreateTensorWithoutHostData(
        gmm_out_shape, aclFormat::ACL_FORMAT_ND, &gmm_out_device_addr, &gmm_out, gmm_out_dtype));
    if (g_mm_out_flag) {
        ACL_CHECK(CreateTensorWithoutHostData(
            mm_out_shape, aclFormat::ACL_FORMAT_ND, &mm_out_device_addr, &mm_out, mm_out_dtype));
    }
    if (g_permute_out_flag) {
        ACL_CHECK(CreateTensorWithoutHostData(
            permute_out_shape, aclFormat::ACL_FORMAT_ND, &permute_out_device_addr, &permute_out, permute_out_dtype));
    }

    // 内存校验
    aclrtMemAttr attr = ACL_HBM_MEM;
    size_t mem_free, mem_total;
    auto ret_before = aclrtGetMemInfo(attr, &mem_free, &mem_total);
    size_t mem_used_before = (mem_total - mem_free) / 1024 / 1024;
    printf(" before RunGraph : free :%ld M,  total:%ld M, used :%ld M, ret :%d \n",
        mem_free / 1024 / 1024,
        mem_total / 1024 / 1024,
        mem_used_before,
        ret_before);

    for (int i = 0; i < g_loop_cnt; i++) {
        clock_gettime(CLOCK_MONOTONIC, &tp1);

        ret = aclnnAlltoAllvQuantGroupedMatMulGetWorkspaceSize(
            gmm_x,
            gmm_weight,
            gmm_x_scale,
            gmm_weight_scale,
            nullptr, // sendCountsTensorOptional
            nullptr, // recvCountsTensorOptional
            mm_x,
            mm_weight,
            mm_x_scale,
            mm_weight_scale,
            g_gmm_x_quant_mode,
            g_gmm_weight_quant_mode,
            g_mm_x_quant_mode,
            g_mm_weight_quant_mode,
            g_group_ep.c_str(),
            g_ep_world_size,
            send_counts_arr,
            recv_counts_arr,
            g_trans_gmm_weight_flag,
            g_trans_mm_weight_flag,
            0, // groupSize
            g_permute_out_flag,
            gmm_out,
            mm_out,
            permute_out,
            &workspaceSize,
            &executor);
        CHECK_RET(ret == ACL_SUCCESS,
                    printf("[ERROR] aclnnAlltoAllvQuantGroupedMatMulGetWorkspaceSize failed. ret = %d \n", ret);
                    return ret);

        if (workspaceSize > 0 && i == 0) {
            ACL_CHECK(aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
        }

        ret = aclnnAlltoAllvQuantGroupedMatMul(workspaceAddr, workspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] second stage failed. ret = %d \n", ret); return ret);

        ACL_CHECK(aclrtSynchronizeStreamWithTimeout(args.stream, 20000));
        clock_gettime(CLOCK_MONOTONIC, &tp2);
        if (tp2.tv_sec != tp1.tv_sec) {
            cost = tp2.tv_nsec - tp1.tv_nsec + 1000000000;
        } else {
            cost = tp2.tv_nsec - tp1.tv_nsec;
        }
        printf("[INFO] device:%d step:%d cost_time: %lu, \n", args.rankId, i, cost);
    }

    // 内存校验
    auto ret_after = aclrtGetMemInfo(attr, &mem_free, &mem_total);
    size_t mem_used_after = (mem_total - mem_free) / 1024 / 1024;
    printf("last run : free :%lu M,  total:%lu M, used :%lu M, ret :%d \n",
        mem_free / 1024 / 1024,
        mem_total / 1024 / 1024,
        mem_used_after,
        ret_after);
    printf("rank %d op mc2 cost memory :%lu M \n", args.rankId, mem_used_after - mem_used_before);

    // 保存输出
    uint64_t gmm_out_size = GetShapeSize(gmm_out_shape) * dtype_to_sizeof(gmm_out_dtype);
    uint64_t mm_out_size = GetShapeSize(mm_out_shape) * dtype_to_sizeof(mm_out_dtype);
    uint64_t permute_out_size = 0;

    saveOutPut(g_bin_path, gmm_out_size, args.rankId, gmm_out_device_addr, "npu_gmm_y.bin");
    if (g_mm_out_flag) {
        saveOutPut(g_bin_path, mm_out_size, args.rankId, mm_out_device_addr, "npu_mm_y.bin");
    }
    if (g_permute_out_flag) {
        if ((permute_out_dtype == "fp4_e2m1" || (permute_out_dtype == "fp4_e1m2"))) {
            permute_out_size = (GetShapeSize(permute_out_shape) * dtype_to_sizeof(permute_out_dtype) + 1) / 2;
        }else {
            permute_out_size = GetShapeSize(permute_out_shape) * dtype_to_sizeof(permute_out_dtype);
        }
        saveOutPut(g_bin_path, permute_out_size, args.rankId, permute_out_device_addr, "npu_permute_out.bin");
    }

    // 释放资源
    // ...（释放所有 tensor 和内存，与原代码一致）
    if (gmm_x != nullptr)
        aclDestroyTensor(gmm_x);
    if (gmm_weight != nullptr)
        aclDestroyTensor(gmm_weight);
    if (gmm_x_scale != nullptr)
        aclDestroyTensor(gmm_x_scale);
    if (gmm_weight_scale != nullptr)
        aclDestroyTensor(gmm_weight_scale);
    if (send_counts_tensor != nullptr)
        aclDestroyTensor(send_counts_tensor);
    if (recv_counts_tensor != nullptr)
        aclDestroyTensor(recv_counts_tensor);
    if (mm_x != nullptr)
        aclDestroyTensor(mm_x);
    if (mm_weight != nullptr)
        aclDestroyTensor(mm_weight);
    if (mm_x_scale != nullptr)
        aclDestroyTensor(mm_x_scale);
    if (mm_weight_scale != nullptr)
        aclDestroyTensor(mm_weight_scale);
    if (gmm_out != nullptr)
        aclDestroyTensor(gmm_out);
    if (mm_out != nullptr)
        aclDestroyTensor(mm_out);
    if (permute_out != nullptr)
        aclDestroyTensor(permute_out);

    if (gmm_x_device_addr != nullptr)
        aclrtFree(gmm_x_device_addr);
    if (gmm_weight_device_addr != nullptr)
        aclrtFree(gmm_weight_device_addr);
    if (gmm_x_scale_device_addr != nullptr)
        aclrtFree(gmm_x_scale_device_addr);
    if (gmm_weight_scale_device_addr != nullptr)
        aclrtFree(gmm_weight_scale_device_addr);
    if (send_counts_device_addr != nullptr)
        aclrtFree(send_counts_device_addr);
    if (recv_counts_device_addr != nullptr)
        aclrtFree(recv_counts_device_addr);
    if (mm_x_device_addr != nullptr)
        aclrtFree(mm_x_device_addr);
    if (mm_weight_device_addr != nullptr)
        aclrtFree(mm_weight_device_addr);
    if (mm_x_scale_device_addr != nullptr)
        aclrtFree(mm_x_scale_device_addr);
    if (mm_weight_scale_device_addr != nullptr)
        aclrtFree(mm_weight_scale_device_addr);
    if (gmm_out_device_addr != nullptr)
        aclrtFree(gmm_out_device_addr);
    if (mm_out_device_addr != nullptr)
        aclrtFree(mm_out_device_addr);
    if (permute_out_device_addr != nullptr)
        aclrtFree(permute_out_device_addr);

    if (workspaceSize > 0)
        aclrtFree(workspaceAddr);
    if (send_counts_arr != nullptr)
        aclDestroyIntArray(send_counts_arr);
    if (recv_counts_arr != nullptr)
        aclDestroyIntArray(recv_counts_arr);

    ACL_CHECK(aclrtDestroyStreamForce(args.stream));
    HcclCommDestroy(args.hcclComm);
    ACL_CHECK(aclrtDestroyContext(args.context));
    ACL_CHECK(aclrtResetDevice(args.rankId));

    return 0;
}

int main(int argc, char *argv[]) {
    GetOption(argc, argv);
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
    ret = aclrtSetDevice(rankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateContext(&context, rankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    HcclCommConfig config;
    HcclCommConfigInit(&config);
    config.hcclDeterministic = 1;
    config.hcclBufferSize = g_hcclBufferSize;
    strncpy(config.hcclCommName, "hccl_comm_test", COMM_NAME_MAX_LENGTH - 1);
    string rankTableFile = getenv("RANK_TABLE_FILE");
    std::cout << "rankTableFilePath is :" << rankTableFile << std::endl;
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
    return 0;
}

// int main(int argc, char *argv[])
// {
//     GetOption(argc, argv);
//     int ret = aclInit(nullptr);
//     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed. ret = %d \n", ret); return ret);
//     aclrtStream stream[g_ep_world_size];
//     aclrtContext context[g_ep_world_size];
//     for (uint32_t rankId = 0; rankId < g_ep_world_size; rankId++) {
//         ret = aclrtSetDevice(rankId);
//         CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d \n", ret); return ret);
//         ret = aclrtCreateContext(&context[rankId], rankId);
//         CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed. ret = %d \n", ret); return ret);
//         ret = aclrtCreateStream(&stream[rankId]);
//         CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d \n", ret); return ret);
//     }

//     int32_t devices[g_ep_world_size];
//     for (int i = 0; i < g_ep_world_size; i++) {
//         devices[i] = i;
//     }
//     // 初始化集合通信域
//     HcclComm comms[g_ep_world_size];
//     ret = HcclCommInitAll(g_ep_world_size, devices, comms);
//     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitAll failed. ret = %d \n", ret); return ret);

//     Args args[g_ep_world_size];
//     // 启动多线程
//     std::vector<std::unique_ptr<std::thread>> threads(g_ep_world_size);
//     for (uint32_t rankId = 0; rankId < g_ep_world_size; rankId++) {
//         args[rankId].rankId = rankId;
//         args[rankId].hcclComm = comms[rankId];
//         args[rankId].stream = stream[rankId];
//         args[rankId].context = context[rankId];
//         threads[rankId].reset(new std::thread(&launchOneThread, std::ref(args[rankId])));
//     }
//     for (uint32_t rankId = 0; rankId < g_ep_world_size; rankId++) {
//         threads[rankId]->join();
//     }
//     aclFinalize();
// //     return 0;
// // }