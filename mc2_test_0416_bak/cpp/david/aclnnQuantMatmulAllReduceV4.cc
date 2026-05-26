#include "aclnnop/aclnn_quant_matmul_all_reduce_v4.h"
#include "aclnnop/aclnn_trans_quant_param_v2.h"
#include "aclnnop/aclnn_quant_matmul_v5.h"
#include "aclnnop/aclnn_add.h"
#include "common.h"

 std::string g_platform ;
 std::string g_bin_path ;
// 输入
 std::string g_input_tensor_dtype[128] = {};
 std::string g_input_tensor_format[128]= {};
 std::vector<int64_t> g_x1_shape= {};
 std::vector<int64_t> g_x2_shape= {};
 std::vector<int64_t> g_bias_shape= {};
 std::vector<int64_t> g_dequant_scale_shape= {};
 std::vector<int64_t> g_x3_shape= {};
 std::vector<int64_t> g_pertoken_shape = {};
 std::vector<int64_t> g_mm_dequantscale_shape= {};
 std::vector<int64_t> g_mm_pertoken_shape= {};
// 输出
 std::string g_output_tensor_dtype;
 std::vector<int64_t> g_output_tensor_shape= {};
 std::string g_output_tensor_format ;
// 可选tensor开关
 bool g_bias_flag = false ;
 bool g_x3_flag= false ;
 bool g_pertoken_flag= false ;
 int64_t g_block_size = 128;
 int64_t g_block_mode =0 ;
 bool  g_per_block_flag= false ; 
// 低精度场景开关
 int g_low_pre_flag= false ;
// 低精度场景执行小算子级联开关
 int g_is_hccl_mm = false ;
// 转置开关
 int g_transpose_b_flag = false ;
// device数量
 int g_ndev =2;
 int g_loop_cnt = 1;
 int rankId =0;

 void GetOption(int argc, char **argv) {
    while (1) {
        int optionIndex = 0;
        struct option longOptions[] =
                {
                    {"x1_shape",               1, 0, 'a'},
                    {"x2_shape",               1, 0, 'b'},
                    {"bias_shape",             1, 0, 'c'},
                    {"dequant_scale_shape",    1, 0, 'd'},
                    {"x3_shape",               1, 0, 'e'},
                    {"input_tensor_dtype",     1, 0, 'f'},
                    {"input_tensor_format",    1, 0, 'g'},
                    {"output_shape",           1, 0, 'h'},
                    {"output_dtype",           1, 0, 'i'},
                    {"output_format",          1, 0, 'j'},
                    {"n_dev",                  1, 0, 'k'},
                    {"platform",               1, 0, 'l'},
                    {"bias_flag",              1, 0, 'm'},
                    {"x3_flag",                1, 0, 'n'},
                    {"transpose_b_flag",       1, 0, 'o'},
                    {"bin_path",               1, 0, 'p'},
                    {"loop_cnt",               1, 0, 'q'},
                    {"pertoken_flag",          1, 0, 'r'},
                    {"pertoken_shape",         1, 0, 's'},
                    {"low_pre_flag",           1, 0, 't'},
                    {"is_hccl_mm",             1, 0, 'u'},
                    {"rankId",                 1, 0, 'v'},
                    {"mm_dequant_scale_shape", 1, 0, 'w'},
                    {"block_size",             1, 0, 'x'},
                    {"block_mode",             1, 0, 'y'},
                    {"per_block_flag",         1, 0, 'z'},
                    {"mm_pertoken_scale_shape",1, 0, 'A'},
                    {0,                        0, 0, 0}
                };

        int c = getopt_long(argc, argv, "y:", longOptions, &optionIndex);
        if (c == -1) {
            break;
        }

        switch (c) {
            case 'a': {
                std::vector<std::string> x1_shape;
                split(optarg, x1_shape, ',');
                for (uint32_t i = 0; i < x1_shape.size(); ++i) {
                    g_x1_shape.push_back(stoll(x1_shape[i].c_str()));
                }
            }
                break;
            case 'b': {
                std::vector<std::string> x2_shape;
                split(optarg, x2_shape, ',');
                for (uint32_t i = 0; i < x2_shape.size(); ++i) {
                    g_x2_shape.push_back(stoll(x2_shape[i].c_str()));
                }
            }
                break;
            case 'c': {
                std::vector<std::string> bias_shape;
                split(optarg, bias_shape, ',');
                for (uint32_t i = 0; i < bias_shape.size(); ++i) {
                    g_bias_shape.push_back(stoll(bias_shape[i].c_str()));
                }
            }
                break;
            case 'd': {
                std::vector<std::string> dequant_scale_shape;
                split(optarg, dequant_scale_shape, ',');
                for (uint32_t i = 0; i < dequant_scale_shape.size(); ++i) {
                    g_dequant_scale_shape.push_back(stoll(dequant_scale_shape[i].c_str()));
                }
            }
                break;
            case 'e': {
                std::vector<std::string> x3_shape;
                split(optarg, x3_shape, ',');
                for (uint32_t i = 0; i < x3_shape.size(); ++i) {
                    g_x3_shape.push_back(stoll(x3_shape[i].c_str()));
                }
            }
                break;
            case 'f': {
                std::vector<std::string> dtypes;
                split(optarg, dtypes, ',');
                for (uint32_t i = 0; i < dtypes.size(); ++i) {
                    g_input_tensor_dtype[i] = dtypes[i];
                }
            }
                break;
            case 'g': {
                std::vector<std::string> formats;
                split(optarg, formats, ',');
                for (uint32_t i = 0; i < formats.size(); ++i) {
                    g_input_tensor_format[i] = formats[i];
                }
            }
                break;
            case 'h': {
                std::vector<std::string> output_shape;
                split(optarg, output_shape, ',');
                for (uint32_t i = 0; i < output_shape.size(); ++i) {
                    g_output_tensor_shape.push_back(stoll(output_shape[i].c_str()));
                }
            }
                break;
            case 'i':
                g_output_tensor_dtype = string(optarg);
                break;
            case 'j':
                g_output_tensor_format = string(optarg);
                break;
            case 'k':
                g_ndev = atoi(optarg);
                break;
            case 'l':
                g_platform = string(optarg);
                break;
            case 'm':
                g_bias_flag = atoi(optarg);
                break;
            case 'n':
                g_x3_flag = atoi(optarg);
                break;
            case 'o':
                g_transpose_b_flag = atoi(optarg);
                break;
            case 'p':
                g_bin_path = string(optarg);
                break;
            case 'q':
                g_loop_cnt = atoi(optarg);
                break;
            case 'r':
                g_pertoken_flag = atoi(optarg);
                break;
            case 's':{
                std::vector<std::string> pertoken_shape;
                split(optarg, pertoken_shape, ',');
                for (uint32_t i = 0; i < pertoken_shape.size(); ++i) {
                    g_pertoken_shape.push_back(stoll(pertoken_shape[i].c_str()));
                }
            }
                break;   
            case 't':
                g_low_pre_flag = atoi(optarg);
                break; 
            case 'u':
                g_is_hccl_mm = atoi(optarg);
                break;
            case 'v':
                rankId = atoi(optarg);
                break;
            case 'w': {
                std::vector<std::string> golden_dequant_scale_shape;
                split(optarg, golden_dequant_scale_shape, ',');
                for (uint32_t i = 0; i < golden_dequant_scale_shape.size(); ++i) {
                    g_mm_dequantscale_shape.push_back(stoll(golden_dequant_scale_shape[i].c_str()));
                }
            }
                break;
            case 'x':
                g_block_size = atoi(optarg);
                break;
            case 'y':
                g_block_mode = atoi(optarg);
                break;
            case 'z':
                g_per_block_flag = stringToBool(optarg);
                break;
            case 'A': {
                std::vector<std::string> mm_pertoken_scale_shape;
                split(optarg, mm_pertoken_scale_shape, ',');
                for (uint32_t i = 0; i < mm_pertoken_scale_shape.size(); ++i) {
                    g_mm_pertoken_shape.push_back(stoll(mm_pertoken_scale_shape[i].c_str()));
                }
            }
                break;
            default:
                break;
        }
    }
}
 
static int launchOneThread(Args &args) 
{
    int ret;
    printf("----Use aclnnQuantMatmulAllReduceV4\n");
    ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
    char hcomName[128];
    ret = HcclGetCommName(args.hcclComm, hcomName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret = %d \n", ret); return ret);
    LOG_PRINT("[INFO] rank %d hcom: %s stream: %p, context : %p\n", args.rankId, hcomName, args.stream, args.context);

    vector<int64_t> x1_shape = g_x1_shape;
    vector<int64_t> x2_shape = g_x2_shape;
    vector<int64_t> x3_shape = g_x3_shape;
    vector<int64_t> bias_shape = g_bias_shape;
    std::vector<int64_t> dequant_scale_shape = g_dequant_scale_shape;
    std::vector<int64_t> reduce_out_shape    = g_output_tensor_shape; 
    std::vector<int64_t> mm_out_shape        = g_output_tensor_shape;  
    std::vector<int64_t> mm_dequant_scale_shape = g_mm_dequantscale_shape;
    std::vector<int64_t> mm_pertoken_scale_shape = g_mm_pertoken_shape;
    std::vector<int64_t> transpose_x2_shape;
    transpose_x2_shape.push_back(x2_shape[1]);
    transpose_x2_shape.push_back(x2_shape[0]);

    void *x1_device_addr   = nullptr;
    void *x2_device_addr   = nullptr;
    void *x3_device_addr   = nullptr;
    void *bias_device_addr = nullptr;
    void *workspaceAddr    = nullptr;
    void *out_device_addr  = nullptr;
    void *dequant_scale_device_addr = nullptr;
    void *transpose_x2_device_addr  = nullptr;
    void *pertoken_device_addr      = nullptr;    
    void *u64_dequant_scale_device_addr = nullptr; 
    void *u64_trans_WorkspaceAddr       = nullptr;
    void *mm_out_device_addr            = nullptr;
    void *mm_u64_dequant_scale_device_addr = nullptr; 
    void *mm_pertoken_scale_device_addr    = nullptr;
    void *mm_dequant_scale_device_addr     = nullptr;
    void *mm_workspaceAddr  = nullptr;
    void *add_workspaceAddr = nullptr;
    void *golden_out_device_addr = nullptr;
    void *tanspose_dequant_scale_device_addr = nullptr;
    void *add_out_device_addr = nullptr;
    
    std::string x1_dtype   = g_input_tensor_dtype[0];
    std::string x2_dtype   = g_input_tensor_dtype[1];
    std::string bias_dtype = g_input_tensor_dtype[2];
    std::string dequant_scale_dtype = g_input_tensor_dtype[3];
    std::string  x3_dtype;

    std::string x1_format   = g_input_tensor_format[0];
    std::string x2_format   = g_input_tensor_format[1];
    std::string bias_format = g_input_tensor_format[2];
    std::string dequant_scale_format = g_input_tensor_format[3];
    std::string  x3_format;

    aclOpExecutor *executor;
    int64_t commTurn   = 0;
    int64_t streamMode = 1;
    uint64_t workspaceSize = 0;
    uint64_t u64_trans_workspaceSize = 0;
    uint64_t mm_workspaceSize = 0;
    uint64_t add_workspaceSize = 0;

    aclTensor *re_out = nullptr;
    aclTensor *x1     = nullptr;
    aclTensor *x2     = nullptr;
    aclTensor *transpose_x2 = nullptr;
    aclTensor *x3     = nullptr;
    aclTensor *bias   = nullptr;
    aclTensor *mm_out = nullptr;
    aclTensor *dequant_scale = nullptr;
    aclTensor *pertoken      = nullptr;
    aclTensor * u64_dequant_scale      = nullptr;
    aclTensor *transpose_dequant_scale = nullptr;
    aclTensor *mm_dequant_scale        = nullptr;
    aclTensor *mm_pertoken_scale       = nullptr;
    aclTensor *mm_u64_dequant_scale    = nullptr;
    aclTensor *add_out                 = nullptr;
    aclTensor *golden_out              = nullptr;
    
    dequant_scale_shape = g_dequant_scale_shape;
    dequant_scale_dtype = g_input_tensor_dtype[3];
    dequant_scale_format = g_input_tensor_format[3];

    int x3_num = 4; 
    if(g_pertoken_flag)
    {
        x3_num = 5; 
    }

    vector<int64_t>pertoken_shape  = g_pertoken_shape;
    string pertoken_dtype  = g_input_tensor_dtype[x3_num-1];
    string pertoken_format = g_input_tensor_format[x3_num-1];

    if(g_x3_flag){
        x3_dtype  =  g_input_tensor_dtype[x3_num];
        x3_format =  g_input_tensor_format[x3_num];
    }

    bool mx_flag= false;
    if (pertoken_dtype == "fp8_e8m0"){mx_flag = true;}
    //matmul 的规定
    int64_t mm_group_size = 0;
    if(mx_flag){mm_group_size = 32;}
    if(g_per_block_flag){mm_group_size= 549764202624;}
    string per_block_scale_dtype = "fp32";

    aclScalar* alpha = nullptr;
    float alphaValue = 1.0f;
    alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);
    CHECK_RET(alpha != nullptr, return ret);

    CreateTensor(x1_shape, aclFormat::ACL_FORMAT_ND, &x1_device_addr, &x1, x1_dtype, g_bin_path, args.rankId, "x1");
    CreateTensor(x2_shape, aclFormat::ACL_FORMAT_ND, &x2_device_addr, &x2, x2_dtype, g_bin_path, args.rankId, "x2",g_transpose_b_flag);
    //x2转置时 
    if(g_transpose_b_flag)
    {
       CreateTensor(transpose_x2_shape, aclFormat::ACL_FORMAT_ND, &transpose_x2_device_addr, &transpose_x2, x2_dtype, g_bin_path, args.rankId, "x2");
    }

    if (g_x3_flag){
        CreateTensor(x3_shape, aclFormat::ACL_FORMAT_ND, &x3_device_addr, &x3, x3_dtype, g_bin_path, args.rankId, "x3");
        CreateTensorWithoutHostData(x3_shape, aclFormat::ACL_FORMAT_ND, &add_out_device_addr, &add_out, g_output_tensor_dtype);
    }

    if (g_bias_flag){
        CreateTensor(bias_shape, aclFormat::ACL_FORMAT_ND, &bias_device_addr, &bias, bias_dtype, g_bin_path, args.rankId, "bias");
    }
    
    CreateTensor(dequant_scale_shape, aclFormat::ACL_FORMAT_ND, &dequant_scale_device_addr,&dequant_scale, dequant_scale_dtype, g_bin_path, args.rankId, "dequant_scale");
    CreateTensorWithoutHostData(reduce_out_shape, aclFormat::ACL_FORMAT_ND, &golden_out_device_addr, &golden_out, g_output_tensor_dtype);

    //perblock 和 mxfp4/FP8场景，且x2转置时，需要将dequantscale进行转置
    if ((g_per_block_flag || mx_flag) && g_transpose_b_flag && dequant_scale_shape.size() == 2) 
    {
        vector<int64_t>transpose_deqaunt_scale_shape; transpose_deqaunt_scale_shape.push_back(dequant_scale_shape[1]);transpose_deqaunt_scale_shape.push_back(dequant_scale_shape[0]);
        CreateTensor(transpose_deqaunt_scale_shape, aclFormat::ACL_FORMAT_ND, &tanspose_dequant_scale_device_addr ,&transpose_dequant_scale, per_block_scale_dtype, g_bin_path, args.rankId, "dequant_scale",g_transpose_b_flag);
    }
    //非pertoken场景输入为hi8，fp8 dequantscale 需要转换成uint64类型
    if (!g_pertoken_flag && !mx_flag)
    {   
        create_u64_dequant_scale(dequant_scale_shape,dequant_scale,&u64_dequant_scale_device_addr,&u64_dequant_scale,args.stream);
        if(g_is_hccl_mm)
        {
            create_u64_dequant_scale(mm_dequant_scale_shape,dequant_scale,&mm_u64_dequant_scale_device_addr,&mm_u64_dequant_scale,args.stream);
        } 
    }
    // 非pertoken场景，不走进来， pertoken 为 null 
    if(g_pertoken_flag) 
    {
        CreateTensor(pertoken_shape, aclFormat::ACL_FORMAT_ND, &pertoken_device_addr, &pertoken, pertoken_dtype, g_bin_path, args.rankId, "pertoken_scale");
    }
    //如果是perblock场景 且小算子拼接，需要为mm创建专用的 mmpertokenScale 因为mm需要和x1维度保持一致；mc2的仅支持二维的不能复用 
    if(g_per_block_flag && g_is_hccl_mm)  
    {
        CreateTensor(mm_pertoken_scale_shape, aclFormat::ACL_FORMAT_ND, &mm_pertoken_scale_device_addr,&mm_pertoken_scale, pertoken_dtype, g_bin_path, args.rankId, "pertoken_scale");
    }

    if(g_is_hccl_mm)
    {
        CreateTensor(mm_dequant_scale_shape, aclFormat::ACL_FORMAT_ND, &mm_dequant_scale_device_addr,&mm_dequant_scale, dequant_scale_dtype, g_bin_path, args.rankId, "dequant_scale");
        CreateTensorWithoutHostData(mm_out_shape, aclFormat::ACL_FORMAT_ND, &mm_out_device_addr, &mm_out, g_output_tensor_dtype);
    }
    
    auto reduce_out_size = GetShapeSize(reduce_out_shape) * dtype_to_sizeof(g_output_tensor_dtype);
    CreateTensorWithoutHostData(reduce_out_shape, aclFormat::ACL_FORMAT_ND, &out_device_addr, &re_out, g_output_tensor_dtype);

    //小算子级联
    if (g_is_hccl_mm)
    {    
        aclTensor* tmp_x2 = nullptr;
        g_transpose_b_flag ? tmp_x2 = transpose_x2 : tmp_x2 =x2;
        if(!g_pertoken_flag){
            //非pertoken 场景需要使用uint64的dequant_scale
            ret = aclnnQuantMatmulV5GetWorkspaceSize(x1, tmp_x2, nullptr, mm_u64_dequant_scale, nullptr, nullptr, nullptr, nullptr, bias, 
                                                    false, g_transpose_b_flag, mm_group_size, mm_out, &mm_workspaceSize, &executor);
        }else
        {
            aclTensor *tmp_pertoken = nullptr;
            // perblock场景，matmul pertokenscale需要传3维，非perblock场景，可以与mc2复用同一个
            g_per_block_flag?tmp_pertoken= mm_pertoken_scale : tmp_pertoken=pertoken;
            if(g_per_block_flag || mx_flag){
                // 转置场景，x2scale需要转置
                aclTensor *tmp_perblock_mx_dequant = nullptr;
                g_transpose_b_flag?tmp_perblock_mx_dequant=transpose_dequant_scale:tmp_perblock_mx_dequant=dequant_scale;
                ret = aclnnQuantMatmulV5GetWorkspaceSize(x1, tmp_x2, tmp_pertoken, tmp_perblock_mx_dequant, nullptr, nullptr, nullptr, nullptr, bias, 
                                                    false, g_transpose_b_flag, mm_group_size, mm_out, &mm_workspaceSize, &executor);
            }
            else{
                ret = aclnnQuantMatmulV5GetWorkspaceSize(x1, tmp_x2, tmp_pertoken, mm_dequant_scale, nullptr, nullptr, nullptr, nullptr, bias, 
                                                    false, g_transpose_b_flag, mm_group_size, mm_out, &mm_workspaceSize, &executor);
            }
        }
        
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV5GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);        
        // 根据第一段接口计算出的workspaceSize申请device内存
        if (mm_workspaceSize > 0) 
        {
            ret = aclrtMalloc(&mm_workspaceAddr, mm_workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        }
        // 调用aclnnQuantMatmulV5第二段接口
        ret = aclnnQuantMatmulV5(mm_workspaceAddr, mm_workspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV5 failed. ERROR: %d\n", ret); return ret);
        // 小算子级联 add
        // 调用aclnnInplaceAdd第一段接口 
        if (g_x3_flag){
  
            ret = aclnnAddGetWorkspaceSize(mm_out, x3, alpha,add_out, &add_workspaceSize, &executor);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceAddGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
            // 根据第一段接口计算出的workspaceSize申请device内存
            if (add_workspaceSize > 0 ) {
                ret = aclrtMalloc(&add_workspaceAddr, add_workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
            }
            // 调用aclnnInplaceAdd第二段接口
            ret = aclnnAdd(add_workspaceAddr, add_workspaceSize, executor, args.stream);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceAdd failed. ERROR: %d\n", ret); return ret);
        }
        // 小算子级联 allreduce
        auto out_size = GetShapeSize(reduce_out_shape);
        void* tmp_device_addr = nullptr;
        g_x3_flag ? tmp_device_addr= add_out_device_addr:tmp_device_addr=mm_out_device_addr;
        HcclAllReduce((void *) mm_out_device_addr, golden_out_device_addr, out_size, dtype_to_HcclDataType[g_output_tensor_dtype],
                    HcclReduceOp::HCCL_REDUCE_SUM, args.hcclComm, args.stream);
        // 4. （固定写法）同步等待任务执行结束
        ret = aclrtSynchronizeStream(args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);   
    }
    
    for(int i = 0;i<g_loop_cnt;i++)
    {   
        if(!g_pertoken_flag) 
        {   //per-tensor,per-channel
            ret = aclnnQuantMatmulAllReduceV4GetWorkspaceSize(x1, x2, bias, x3, pertoken, u64_dequant_scale, 
                         nullptr,nullptr,hcomName,"sum",commTurn, streamMode,g_block_mode,g_block_size, re_out, &workspaceSize, &executor);
        }
        else
        {   
            //per-token,per-block
            if(g_per_block_flag||mx_flag)
            {
                aclTensor * tmp_dequant_scale = nullptr;
                g_transpose_b_flag? tmp_dequant_scale = transpose_dequant_scale : tmp_dequant_scale = dequant_scale;
                ret = aclnnQuantMatmulAllReduceV4GetWorkspaceSize(x1, x2, bias, x3, pertoken,tmp_dequant_scale ,
                         nullptr,nullptr,hcomName,"sum",commTurn, streamMode,g_block_mode,g_block_size, re_out, &workspaceSize, &executor);
            }
            else
            {
                ret = aclnnQuantMatmulAllReduceV4GetWorkspaceSize(x1, x2, bias, x3, pertoken, dequant_scale,
                         nullptr,nullptr,hcomName,"sum",commTurn, streamMode,g_block_mode,g_block_size, re_out, &workspaceSize, &executor);
            }
        }
         CHECK_RET(ret == ACL_SUCCESS, printf("[ERROR] aclnnQuantMatmulAllReduceV4GetWorkspaceSize failed. ret = %d \n", ret); return ret);
        if (workspaceSize > 0 && i==0) {
            ACL_CHECK(aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
        }
        //调用第二段接口
        ret = aclnnQuantMatmulAllReduceV4(workspaceAddr, workspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, printf("[ERROR] aclnnQuantMatmulAllReduce failed. ret = %d \n", ret); return ret);
        ACL_CHECK(aclrtSynchronizeStreamWithTimeout(args.stream, 2000000000));
    }
    
    saveOutPut(g_bin_path,reduce_out_size,args.rankId,out_device_addr,"output_npu.bin");
    if (g_is_hccl_mm){
        saveOutPut(g_bin_path,reduce_out_size,args.rankId,golden_out_device_addr,"output_hccl.bin");
    }

    if (x1 != nullptr) {aclDestroyTensor(x1);}
    if (x2 != nullptr) {aclDestroyTensor(x2);}
    if (x3 != nullptr) {aclDestroyTensor(x3);}
    if (bias != nullptr) {aclDestroyTensor(bias);}
    if (dequant_scale != nullptr){aclDestroyTensor(dequant_scale);}
    if (pertoken != nullptr){aclDestroyTensor(pertoken);}
    if (re_out != nullptr) {aclDestroyTensor(re_out);}
    if (u64_dequant_scale != nullptr) {aclDestroyTensor(u64_dequant_scale);}
    if (transpose_x2 != nullptr) {aclDestroyTensor(transpose_x2);}
    if (mm_out != nullptr) {aclDestroyTensor(mm_out);}
    if (transpose_dequant_scale != nullptr) {aclDestroyTensor(transpose_dequant_scale);}
    if (add_out != nullptr) {aclDestroyTensor(add_out);}
    if (mm_dequant_scale != nullptr) {aclDestroyTensor(mm_dequant_scale);}
    if (mm_u64_dequant_scale != nullptr) {aclDestroyTensor(mm_u64_dequant_scale);}
    if (mm_pertoken_scale != nullptr) {aclDestroyTensor(mm_pertoken_scale);}
    if (golden_out != nullptr) {aclDestroyTensor(golden_out);}
    
    if (x1_device_addr != nullptr) {aclrtFree(x1_device_addr);}
    if (x2_device_addr != nullptr) {aclrtFree(x2_device_addr);}
    if (x3_device_addr != nullptr) {aclrtFree(x3_device_addr);}
    if (bias_device_addr != nullptr) {aclrtFree(bias_device_addr);}
    if (out_device_addr != nullptr) {aclrtFree(out_device_addr);}
    if (dequant_scale_device_addr != nullptr) {aclrtFree(dequant_scale_device_addr);}
    if (pertoken_device_addr != nullptr) {aclrtFree(pertoken_device_addr);}
    if (u64_dequant_scale_device_addr != nullptr) {aclrtFree(u64_dequant_scale_device_addr);}
    if (tanspose_dequant_scale_device_addr != nullptr) {aclrtFree(tanspose_dequant_scale_device_addr);}
    if (transpose_x2_device_addr != nullptr) {aclrtFree(transpose_x2_device_addr);}
    if (mm_out_device_addr != nullptr) {aclrtFree(mm_out_device_addr);}
    if (mm_u64_dequant_scale_device_addr != nullptr) {aclrtFree(mm_u64_dequant_scale_device_addr);}
    if (mm_pertoken_scale_device_addr != nullptr) {aclrtFree(mm_pertoken_scale_device_addr);}
    if (mm_dequant_scale_device_addr != nullptr) {aclrtFree(mm_dequant_scale_device_addr);}
    if (golden_out_device_addr != nullptr) {aclrtFree(golden_out_device_addr);}
    if (add_out_device_addr != nullptr) {aclrtFree(add_out_device_addr);}
    
    if (workspaceSize > 0) {aclrtFree(workspaceAddr);}
    if (u64_trans_workspaceSize>0) {aclrtFree(u64_trans_WorkspaceAddr);}
    if (mm_workspaceSize>0) {aclrtFree(mm_workspaceAddr);}
    if (add_workspaceSize>0) {aclrtFree(add_workspaceAddr);}

    DestoryRes(args);
    return 0;
}

int main(int argc,char* argv[])
{
    int ret;
    Args args;
    GetOption(argc,argv);
    ret = init(args,g_ndev,rankId);
    CHECK_RET(ret == 0, LOG_PRINT("aclnnQuantMatmulAllReduceV4 init failed. ERROR: %d\n", ret); return -1);
    ret = launchOneThread(args);
    CHECK_RET(ret == 0, LOG_PRINT("aclnnQuantMatmulAllReduceV4Func launchOneThread  failed. ERROR: %d\n", ret); return -1);
    LOG_PRINT("end launchOneThread ret = %d", ret);
    ACL_CHECK(aclFinalize());
    return 0;
}
