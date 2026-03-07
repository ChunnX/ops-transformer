namespace MoeDispatchLog_NEW {

template<typename T>
constexpr bool is_decimal_integer_v =
    std::is_same_v<T, int8_t> ||
    std::is_same_v<T, int16_t> ||
    std::is_same_v<T, int32_t> ||
    std::is_same_v<T, int64_t>;


template<typename T>
constexpr bool is_unsigned_integer_v =
    std::is_same_v<T, bool> ||
    std::is_same_v<T, uint8_t> ||
    std::is_same_v<T, uint16_t> ||
    std::is_same_v<T, uint32_t> ||
    std::is_same_v<T, uint64_t>;


template<typename T>
constexpr bool is_fp_point_v =
    std::is_same_v<T, float> ||
    std::is_same_v<T, half> ||
    std::is_same_v<T, bfloat16_t>;

using namespace AscendC;
class Log {
public:

    __aicore__ inline void Init(TPipe * tpipe_,int32_t rankId);

    template<typename T>
    __aicore__ inline void print_value(const T& val);
    __aicore__ inline void LogInfo(const uint32_t line, const __gm__ char *msg, const GM_ADDR addr);
    __aicore__ inline void LogInfo(const uint32_t line, const __gm__ char *msg);
    template<typename T>
    __aicore__ inline void LogInfo(const uint32_t line, const __gm__ char *msg,   const T num);

    template<typename T>
    __aicore__ inline uint32_t GetElementWidth(T val);

    template<typename T, size_t N>
    __aicore__  inline void PrintLevel(const LocalTensor<T>& tensor, size_t actual_size,
                       const uint32_t (&shape)[N], const int (&strides)[N],
                       uint32_t max_element_width);

    template<typename T, size_t N>
    __aicore__ inline  void PrintValue(const LocalTensor<T>& tensor, const uint32_t (&shape)[N]);

    template<typename T, size_t N>
    __aicore__  inline void PrintLevel(const GlobalTensor<T>& tensor, size_t actual_size,
                        const uint32_t (&shape)[N], const int (&strides)[N],
                        uint32_t max_element_width);
    

    template<typename T, size_t N>
    __aicore__ inline void PrintValue(const GlobalTensor<T>& tensor, const uint32_t (&shape)[N]);
    
    template <typename T,size_t N>
    __aicore__ inline void LogInfo(uint32_t line, const __gm__ char *msg, LocalTensor<T>& tensor,const uint32_t (&shape)[N]); //按指定shap打

    template <typename T,size_t N>
    __aicore__ inline void LogInfo(uint32_t line, const __gm__ char *msg, GlobalTensor<T>& tensor,const uint32_t (&shape)[N]); //按指定shap打

private:
    int32_t rankId_ = -1;
    uint32_t aivId_ = GetBlockIdx();
    TPipe * tpipe_ = nullptr;
};

template<typename T>
__aicore__ inline void Log::print_value(const T& val) {
    if constexpr (is_decimal_integer_v<T>) {
        printf("%d", val);
    }
    else if constexpr (is_unsigned_integer_v<T>) {
        printf("%u",val);
    }
    else if constexpr (is_fp_point_v<T>) {
        printf("%f", val);
    }
}

__aicore__ inline void Log::Init(TPipe * tpipe,int32_t rankId) {
    tpipe_ = tpipe;
    rankId_ =rankId;
}

__aicore__ inline void Log::LogInfo(const uint32_t line, const __gm__ char *msg, const GM_ADDR addr){
    const __gm__ void * addrtemp = static_cast<const __gm__ void*>(addr);
    printf("[rankId: %d][aivId: %d][line: %d]", rankId_, aivId_, line);
    printf("%s: %p\n", msg, addrtemp);
}
__aicore__ inline void Log::LogInfo(const uint32_t line, const __gm__ char *msg){
    printf("[rankId: %d][aivId: %d][line: %d]%s\n", rankId_, aivId_, line, msg);
}

template<typename T>
__aicore__ inline void Log::LogInfo(const uint32_t line, const __gm__ char *msg,   const T num){
    printf("[rankId: %d][aivId: %d][line: %d]%s: ", rankId_, aivId_, line, msg);
    print_value(num);
    printf("\n");
}

template<typename T>
__aicore__ inline uint32_t Log::GetElementWidth(T val) {
    float32_t d_val{0};
    if constexpr ( is_fp_point_v<T> && std::is_same_v<T, bfloat16_t>){
        d_val = AscendC::ToFloat(val);
    }
    else{
       d_val = static_cast<float32_t>(val);
    }
     
    uint32_t length = 0;
    if(d_val < 0) {
        length++;
        d_val = -d_val;
    }
    long long int_part = static_cast<long long>(d_val);
    if(int_part == 0){
        length++;
    } else {
        uint64_t temp = int_part;
        while(temp > 0){
            temp = temp / 10;
            length++;
        }
    }

    if(std::is_floating_point<T>::value) {
        length +=7;
    }

    return length;
}

template<typename T, size_t N>
__aicore__  inline void Log::PrintLevel(const LocalTensor<T>& tensor, size_t actual_size,
                       const uint32_t (&shape)[N], const int (&strides)[N],
                       uint32_t max_element_width) {
    struct State {
        int dim;
        int pos;
        size_t offset;
    };

    State stack[8];
    int top = 0;
    stack[0] = {0, 0, 0};
    printf("[");

    while (top >= 0) {
        State& s = stack[top];

        if (s.dim == N - 1) {
            size_t elem_offset = s.offset + s.pos * strides[s.dim];
            bool is_last = (s.pos == shape[s.dim] - 1);

            int current_printed = 0;
            if (elem_offset < actual_size) {
                // 打印数值
                T val = tensor(elem_offset);
                current_printed = GetElementWidth(val);
                print_value(val);
            } else {
                // 打印占位符，并记录其宽度为 1
                printf("-");
                current_printed = 1;
            }

            // 打印逗号
            if (!is_last) {
                printf(",");
                current_printed += 1;
            }

            // 计算补白：确保 "-" 和数字占据相同的总宽度
            // 补白 = (最大宽度 + 逗号位) - 当前已打印长度 + 额外间距
            int padding = (max_element_width + (is_last ? 0 : 1)) - current_printed + 1;
            for (int i = 0; i < padding; ++i){ // 增加 +1，保持间隔
                printf(" ");
            }

            if (is_last) {
                printf("]");
                top--;
                if (top >= 0) {
                    stack[top].pos++;
                    if (stack[top].pos < shape[stack[top].dim]) {
                        printf("\n");
                        for (int i = 0; i <= top; ++i) printf(" ");
                    }
                }
            } else {
                s.pos++;
            }
        } else {
            if (s.pos < shape[s.dim]) {
                size_t next_offset = s.offset + s.pos * strides[s.dim];
                top++;
                stack[top] = {(int)s.dim + 1, 0, next_offset};
                printf("[");
            } else {
                printf("]");
                top--;
                if (top >= 0) {
                    stack[top].pos++;
                    if (stack[top].pos < shape[stack[top].dim]) {
                        printf(",\n");
                        for (int i = 0; i <= top; ++i){
                            printf(" ");
                        }
                    }
                }
            }
        }
    }
    printf("\n");
}


template<typename T, size_t N>
__aicore__ inline void Log::PrintValue(const LocalTensor<T>& tensor, const uint32_t (&shape)[N]) {
    PipeBarrier<PIPE_ALL>();
    size_t actual_size = tensor.GetSize();

    uint32_t max_w = 0;
    for (uint32_t index = 0; index < actual_size; index++) {
        T val = tensor(index);
        uint32_t len = GetElementWidth(val);
        max_w = max_w > len? max_w : len;
    }

    int strides[N];
    strides[N-1] = 1;
    for (int i = (int)N - 2; i >= 0; --i) {
        strides[i] = strides[i+1] * shape[i+1];
    } 

    PrintLevel(tensor, actual_size, shape, strides, max_w);
    printf("\n");
}


template <typename T,size_t N>
__aicore__ inline void Log::LogInfo(const uint32_t line, const __gm__ char *msg, LocalTensor<T>& tensor,const uint32_t (&shape)[N]) {//按指定shap打
    printf("[rankId: %d][aivId: %d][line: %d]", rankId_, aivId_, line);
    printf("%s\n", msg);
    if(tensor.GetSize() == 0){
        printf("tensor is empty\n");
        return;
    }
    bool cast = !is_decimal_integer_v<T> && !is_unsigned_integer_v<T> && !is_fp_point_v<T>;
    if (cast) {
        printf("[ERROR][printf only support float, int, uint, bool .. plese read document and cast data type]\n");
        return ;
    }
    PrintValue(tensor, shape);
}

template<typename T, size_t N>
__aicore__  inline void Log::PrintLevel(const GlobalTensor<T>& tensor, size_t actual_size,
                       const uint32_t (&shape)[N], const int (&strides)[N],
                       uint32_t max_element_width) {
    struct State {
        int dim;
        int pos;
        size_t offset;
    };

    State stack[8];
    int top = 0;
    stack[0] = {0, 0, 0};
    printf("[");

    while (top >= 0) {
        State& s = stack[top];

        if (s.dim == N - 1) {
            size_t elem_offset = s.offset + s.pos * strides[s.dim];
            bool is_last = (s.pos == shape[s.dim] - 1);

            int current_printed = 0;
            if (elem_offset < actual_size) {
                // 打印数值
                T val = tensor(elem_offset);
                current_printed = GetElementWidth(val);
                print_value(val);
            } else {
                // 打印占位符，并记录其宽度为 1
                printf("-");
                current_printed = 1;
            }

            // 打印逗号
            if (!is_last) {
                printf(",");
                current_printed += 1;
            }

            // 计算补白：确保 "-" 和数字占据相同的总宽度
            // 补白 = (最大宽度 + 逗号位) - 当前已打印长度 + 额外间距
            int padding = (max_element_width + (is_last ? 0 : 1)) - current_printed + 1;
            for (int i = 0; i < padding; ++i){ // 保留 +1，间距隔开
                printf(" ");
            }

            if (is_last) {
                printf("]");
                top--;
                if (top >= 0) {
                    stack[top].pos++;
                    if (stack[top].pos < shape[stack[top].dim]) {
                        printf("\n");
                        for (int i = 0; i <= top; ++i) printf(" ");
                    }
                }
            } else {
                s.pos++;
            }
        } else {
            if (s.pos < shape[s.dim]) {
                size_t next_offset = s.offset + s.pos * strides[s.dim];
                top++;
                stack[top] = {(int)s.dim + 1, 0, next_offset};
                printf("[");
            } else {
                printf("]");
                top--;
                if (top >= 0) {
                    stack[top].pos++;
                    if (stack[top].pos < shape[stack[top].dim]) {
                        printf(",\n");
                        for (int i = 0; i <= top; ++i){
                            printf(" ");
                        }
                    }
                }
            }
        }
    }
    printf("\n");
}

template<typename T, size_t N>
__aicore__  inline void Log::PrintValue(const GlobalTensor<T>& tensor, const uint32_t (&shape)[N]) {
    PipeBarrier<PIPE_ALL>();
    size_t actual_size = tensor.GetSize();

    // 预扫描获取最大元素宽度
    uint32_t max_w = 0;
    for (uint32_t index = 0; index < actual_size; index++) {
        T val = tensor(index);
        uint32_t len = GetElementWidth(val);
        max_w = max_w > len? max_w : len;
    }

    int strides[N];
    strides[N-1] = 1;
    for (int i = (int)N - 2; i >= 0; --i) {
        strides[i] = strides[i+1] * shape[i+1];
    } 

    PrintLevel(tensor, actual_size, shape, strides, max_w);
    printf("\n");
}

template <typename T,size_t N>
__aicore__ inline void Log::LogInfo(const uint32_t line, const __gm__ char *msg, GlobalTensor<T>& tensor,const uint32_t (&shape)[N]) {//按指定shap打
    printf("[rankId: %d][aivId: %d][line: %d]", rankId_, aivId_, line);
    printf("%s\n", msg);
    bool cast = !is_decimal_integer_v<T> && !is_unsigned_integer_v<T> && !is_fp_point_v<T>;
    if (cast) {
        printf("[ERROR][printf only support float, int, uint, bool.. plese read document and cast data type]\n");
        return ; 
    }
    uint64_t elemNum = 1;
    for (uint32_t i = 0; i < N; i++) {
        elemNum = elemNum * shape[i];
    }
    if(elemNum == 0) {
        printf("tensor is empty\n");
        return;
    }
    GlobalTensor<T> tempTensor;
    tempTensor.SetGlobalBuffer(const_cast<__gm__ T*>(tensor.GetPhyAddr()),elemNum);
    PrintValue(tempTensor, shape);
}
//用法，在类中定义logger，然后在需要打印的地方调用LOG_INFO(...)
//在最开始定义LOG_INIT(...)，
//#define DEBUG 1
//#if DEBUG
//#define LOG_INFO(...) this->logger.LogInfo(__LINE__,__VA_ARGS__)
//#define LOG_INIT(...) this->logger.Init(__VA_ARGS__)
//#else
//#define LOG_INFO(...) 
//#define LOG_INIT(...) 
//#endif

}




// //需要加kernel_operator头文件


// template <TemplateDispatchKFCTypeClass>
// __aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::CalServerSendCnt()
// {
//     TBuf<> tempBuf;
//     TBuf<> sharedTmpBuf;
//     tpipe_->InitBuffer(tempBuf, expertIdsCnt_ * sizeof(int16_t));
//     tpipe_->InitBuffer(sharedTmpBuf, 128 * activeMaskBsCnt_);
//     uint32_t calCnt = isTokenMaskFlag_? sendToMoeExpTokenCnt_ : expertIdsCnt_; 
//     uint32_t activeBsCnt = static_cast<uint32_t>(activeMaskBsCnt_);

//     LocalTensor<float32_t> dstServerIdFp32 = dstExpBuf_.Get<float32_t>();
//     LocalTensor<float32_t> tempTensorFp32 = subExpBuf_.Get<float32_t>();
//     LocalTensor<float32_t> tempServerIdFp32 = gatherMaskTBuf_.Get<float32_t>();
//     LocalTensor<float32_t> sharedTensorFp32 = sharedTmpBuf.Get<float32_t>();
//     LocalTensor<int32_t> dstServerIdInt32 = dstExpBuf_.Get<int32_t>();
//     LocalTensor<int32_t> tempTensorInt32 = subExpBuf_.Get<int32_t>();
//     LocalTensor<int32_t> tempServerIdInt32 = gatherMaskTBuf_.Get<int32_t>();
//     LocalTensor<int16_t> tempServerIdInt16 = gatherMaskTBuf_.Get<int16_t>();
//     LocalTensor<int16_t> tempTensorInt16 = subExpBuf_.Get<int16_t>(); 
//     LocalTensor<int16_t> compareInt16 = tempBuf.Get<int16_t>();
//     LocalTensor<uint8_t> sharedTensorInt8 = sharedTmpBuf.Get<uint8_t>();
    
//     Cast(dstServerIdFp32, expertIdsTensor_, RoundMode::CAST_NONE, calCnt);
//     Duplicate<float32_t>(tempTensorFp32, static_cast<float32_t>(moeExpertNumPerRank_), calCnt);
//     PipeBarrier<PIPE_V>();
//     Div(dstServerIdFp32, dstServerIdFp32, tempTensorFp32, calCnt);
//     PipeBarrier<PIPE_V>();
//     Adds(dstServerIdFp32, dstServerIdFp32, static_cast<float32_t>(sharedExpertRankNum_), calCnt);
//     PipeBarrier<PIPE_V>();
//     Duplicate<float32_t>(tempTensorFp32, static_cast<float32_t>(serverRankSize_), calCnt);
//     PipeBarrier<PIPE_V>();
//     Div(dstServerIdFp32, dstServerIdFp32, tempTensorFp32, calCnt); //计算每个专家所在的serverID
//     PipeBarrier<PIPE_V>();
//     Cast(dstServerIdInt32, dstServerIdFp32, RoundMode::CAST_TRUNC, calCnt); 
//     PipeBarrier<PIPE_V>();

//     for(uint32_t serverIndex = 0; serverIndex < serverNum_; serverIndex++) {
//         uint32_t dstServerId = serverIndex;
//         Duplicate<int32_t>(tempTensorInt32, dstServerId, calCnt);
//         PipeBarrier<PIPE_V>();
//         Sub(tempServerIdInt32, dstServerIdInt32, tempTensorInt32, calCnt);
//         PipeBarrier<PIPE_V>();
//         Abs(tempServerIdFp32, tempServerIdFp32, calCnt);
//         PipeBarrier<PIPE_V>();
//         Mins(tempServerIdInt32, tempServerIdInt32, 1, calCnt);
//         PipeBarrier<PIPE_V>();
//         Cast(tempServerIdInt16, tempServerIdInt32, RoundMode::CAST_NONE, calCnt);
//         Duplicate<int16_t>(tempTensorInt16, 1, calCnt);
//         PipeBarrier<PIPE_V>();
//         Xor(compareInt16, tempServerIdInt16, tempTensorInt16, sharedTensorInt8, calCnt);
//         PipeBarrier<PIPE_V>();
//         Cast(tempServerIdFp32, compareInt16, RoundMode::CAST_NONE, calCnt);//这里是 0 1 结果
//         PipeBarrier<PIPE_V>();
//         if(isExpertMaskFlag_) {
//             LocalTensor<int8_t> maskInputInt8Tensor = expertMaskInputTensor_.ReinterpretCast<int8_t>();
//             Cast(tempTensorFp32, maskInputInt8Tensor, RoundMode::CAST_NONE, calCnt);
//             Mul(tempServerIdFp32, tempServerIdFp32, tempTensorFp32, calCnt); //需要与mask相乘。
//             PipeBarrier<PIPE_V>(); 
//         }
//         uint32_t srcShape[2] = {activeBsCnt, axisK_};
//         ReduceSum<float32_t, Pattern::Reduce::AR, true>(tempTensorFp32, tempServerIdFp32, sharedTensorInt8, srcShape, false);
//         PipeBarrier<PIPE_V>();
//         Mins(tempServerIdFp32, tempTensorFp32, float32_t(1.0), activeBsCnt); // 得到需要发送到这个server的TokenID
//         PipeBarrier<PIPE_V>();
//         uint32_t inner = Ceil(activeBsCnt * sizeof(float32_t), UB_ALIGN) * UB_ALIGN / sizeof(float32_t);
//         const AscendC::CumSumInfo CumSumInfo{1, inner};
//         CumSum(tempTensorFp32, sharedTensorFp32, tempServerIdFp32, sharedTensorInt8, CumSumInfo); // 这里不输出最后一行，因此复用buf
//         PipeBarrier<PIPE_V>();
//         Cast(tempServerIdInt32, tempServerIdFp32, RoundMode::CAST_TRUNC, activeBsCnt); // 转为int类型
//         Cast(tempTensorInt32, tempTensorFp32, RoundMode::CAST_TRUNC, activeBsCnt);
//         PipeBarrier<PIPE_V>();
//         uint32_t serverTotalCnt = tempTensorInt32.GetValue(activeBsCnt - 1);
//         serverCountTensor_.SetValue(dstServerId, serverTotalCnt);
//         Mul(tokenSendMap_[serverIndex * activeBsCnt], tempTensorInt32, tempServerIdInt32, activeBsCnt); //TODO：需要32字节对齐
//         PipeBarrier<PIPE_V>();
//     } 
// }

// template <TemplateDispatchKFCTypeClass>
// __aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::CalTokenSendExpertCnt(
//     uint32_t dstExpertId, int32_t calCnt, int32_t &curExpertCnt)
// {
//     LocalTensor<int32_t> dstExpIdTensor = dstExpBuf_.Get<int32_t>();
//     LocalTensor<int32_t> subExpIdTensor = subExpBuf_.Get<int32_t>();
//     Duplicate<int32_t>(dstExpIdTensor, dstExpertId, calCnt);
//     PipeBarrier<PIPE_V>();
//     Sub(subExpIdTensor, expertIdsTensor_, dstExpIdTensor, calCnt);
//     PipeBarrier<PIPE_V>();
//     LocalTensor<float> tmpFp32 = subExpIdTensor.ReinterpretCast<float>();
//     LocalTensor<float> tmpoutFp32 = dstExpIdTensor.ReinterpretCast<float>();
//     Abs(tmpoutFp32, tmpFp32, calCnt);
//     PipeBarrier<PIPE_V>();
//     Mins(subExpIdTensor, dstExpIdTensor, 1, calCnt);
//     PipeBarrier<PIPE_V>();
//     ReduceSum<float>(tmpoutFp32, tmpFp32, workLocalTensor_, calCnt);
//     SyncFunc<AscendC::HardEvent::V_S>();
//     int32_t curOtherExpertCnt = dstExpIdTensor(0);
//     curExpertCnt = calCnt - curOtherExpertCnt;
// }


// template <TemplateDispatchKFCTypeClass>
// __aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::CalTokenSendExpertOffset()//pase
// {
//     TBuf<> tempBuf;
//     TBuf<> sharedTmpBuf;
//     uint32_t startTokenId, endTokenId, sendTokenNum;
//     SplitToCore(activeMaskBsCnt_, aivUsedDispatch_, startTokenId, endTokenId, sendTokenNum);
//     if(startTokenId >= endTokenId || sendTokenNum == 0) {
//         return ;
//     }

//     uint32_t calCnt = endTokenId * axisK_;
//     uint32_t inner = Ceil(endTokenId * sizeof(float32_t), UB_ALIGN) * UB_ALIGN / sizeof(float32_t);

//     tpipe_->InitBuffer(tempBuf, expertIdsCnt_ * sizeof(int16_t));
//     tpipe_->InitBuffer(sharedTmpBuf, 128 * 512);



//     LocalTensor<float32_t> dstExpIdTensorFP32 = dstExpBuf_.Get<float32_t>();
//     LocalTensor<float32_t> tempTensorFp32 = subExpBuf_.Get<float32_t>();
//     LocalTensor<float32_t> sharedTensorFp32 = sharedTmpBuf.Get<float32_t>();
//     LocalTensor<int32_t> tempTensorInt32 = subExpBuf_.Get<int32_t>();
//     LocalTensor<int32_t> dstExpIdTensorInt32 = dstExpBuf_.Get<int32_t>();
//     LocalTensor<int16_t> dstExpIdTensorInt16 = dstExpBuf_.Get<int16_t>();
//     LocalTensor<int16_t> tempTensorInt16 = subExpBuf_.Get<int16_t>();
//     LocalTensor<int16_t> compareInt16 = tempBuf.Get<int16_t>();
//     LocalTensor<uint8_t> sharedTensorInt8 = sharedTmpBuf.Get<uint8_t>();
//     Duplicate<int32_t>(expertOffsetCntTensor_, 0, expertIdsCnt_); //TOken * moeId


//     for(uint32_t Index = 0; Index < sendTokenNum * axisK_; Index++) { //TODO:需要考虑二维mask影响
//         uint32_t expertIndex = aivId_ * sendTokenNum + Index;
//         uint32_t dstExpertId = expertIdsTensor_(expertIndex);
//         Duplicate<int32_t>(tempTensorInt32, dstExpertId, calCnt);
//         PipeBarrier<PIPE_V>();
//         Sub(dstExpIdTensorInt32, expertIdsTensor_, tempTensorInt32, calCnt);
//         PipeBarrier<PIPE_V>();
//         Abs(dstExpIdTensorFP32, dstExpIdTensorFP32, calCnt);
//         PipeBarrier<PIPE_V>();
//         Mins(dstExpIdTensorInt32, dstExpIdTensorInt32, 1, calCnt);
//         PipeBarrier<PIPE_V>();
//         Cast(dstExpIdTensorInt16, dstExpIdTensorInt32, RoundMode::CAST_NONE, calCnt);
//         Duplicate<int16_t>(tempTensorInt16, 1, calCnt); 
//         PipeBarrier<PIPE_V>();
//         Xor(compareInt16, dstExpIdTensorInt16, tempTensorInt16, sharedTensorInt8, calCnt);
//         PipeBarrier<PIPE_V>();
//         Cast(dstExpIdTensorFP32, compareInt16, RoundMode::CAST_NONE, calCnt);//这里是 0 1 结果
//         PipeBarrier<PIPE_V>();
//         if(isExpertMaskFlag_) {
//             LocalTensor<int8_t> maskInputInt8Tensor = expertMaskInputTensor_.ReinterpretCast<int8_t>();
//             Cast(tempTensorFp32, maskInputInt8Tensor, RoundMode::CAST_NONE, calCnt);
//             Mul(dstExpIdTensorFP32, dstExpIdTensorFP32, tempTensorFp32, calCnt); //这里需要于mask相乘，且需要保留
//             PipeBarrier<PIPE_V>(); //应该使用Reduce
//         }
//         uint32_t srcShape[2] = {endTokenId, axisK_};
//         ReduceSum<float32_t, Pattern::Reduce::AR, true>(tempTensorFp32, dstExpIdTensorFP32, sharedTensorInt8, srcShape, false);
//         PipeBarrier<PIPE_V>();
//         const AscendC::CumSumInfo CumSumInfo{1, inner};
//         CumSum(dstExpIdTensorFP32, sharedTensorFp32, tempTensorFp32, sharedTensorInt8, CumSumInfo); //这里不输出最后一行，因此复用buf
//         PipeBarrier<PIPE_V>();
//         Cast(dstExpIdTensorInt32, dstExpIdTensorFP32, RoundMode::CAST_TRUNC, calCnt);//这里是最终结果
//         Cast(tempTensorInt32, tempTensorFp32, RoundMode::CAST_TRUNC, calCnt);//这里是累加和
//         PipeBarrier<PIPE_V>();
//         Mul(dstExpIdTensorInt32, tempTensorInt32, dstExpIdTensorInt32, calCnt);
//         PipeBarrier<PIPE_V>();
//         // for(uint32_t i = 0; i < )
//         //Add(expertOffsetCntTensor_, expertOffsetCntTensor_, dstExpIdTensorInt32, calCnt); // TODO： 需要32字节对齐。【1 2 3 4 5 6】
//         PipeBarrier<PIPE_V>(); 
//     }
// }




// template <TemplateDispatchKFCTypeClass>
// __aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::CalExpertTotalSendCnt() 
// {
//     LocalTensor<int32_t> tokenNumToExpertTensor_; //todo:放在全局
//     uint32_t aivNumUse_ = 1; //todo:放在全局

//     uint64_t maskCnt = 0;
//     uint32_t compareCount = 0;
//     uint32_t maskSizePerExpert = 0;
//     uint32_t calCnt = moeExpertNum_;
//     uint32_t startExpertId, endExpertId, sendExpertNum;
//     uint32_t mask = isTokenMaskFlag_ ? (activeMaskBsCnt_ * axisK_) : expertIdsCnt_;
//     LocalTensor<uint8_t> expertMaskTensorU8 = subExpBuf_.Get<uint8_t>();
//     LocalTensor<uint32_t> expertMaskTensorU32 = subExpBuf_.Get<uint32_t>();
//     LocalTensor<int32_t> gatherTempTensor = gatherMaskTBuf_.Get<int32_t>();;
//     //SplitToCore(calCnt, aivNumUse_, startExpertId, endExpertId, sendExpertNum, true);
//     if(startExpertId >= endExpertId || sendExpertNum == 0){
//         return;
//     }
//     compareCount = Ceil(mask * sizeof(int32_t), ALIGNED_LEN_256) * ALIGNED_LEN_256 / sizeof(int32_t);
//     maskSizePerExpert = Ceil(compareCount / BITS_PER_BYTE, UB_ALIGN) * UB_ALIGN; // 为了匹配CompareScalar接口1bit一共8位
    
//     // if(isExpertMaskFlag_) { //TODO:需要去放在某个地方
//     //     uint64_t rsvdCnt = 0;
//     //     uint32_t mask = expertIdsCnt_;
//     //     LocalTensor<int32_t> tmpExpertIdsTensor = subExpBuf_.Get<int32_t>();
//     //     DataCopyPad(tmpExpertIdsTensor, expertIdsGMTensor_, expertIdsCntParams, expertIdsCntCopyPadParams);
//     //     SyncFunc<AscendC::HardEvent::MTE2_V>();
//     //     GatherMask(expertIdsTensor_, tmpExpertIdsTensor, gatherMaskTensor_, true, mask, {1, 1, 1, 0}, rsvdCnt);
//     // }

//     for (int32_t expertIndex = 0; expertIndex < sendExpertNum; expertIndex++) {
//         int32_t dstExpertId = expertIndex + startExpertId;
//         CompareScalar(expertMaskTensorU8[maskSizePerExpert * expertIndex], expertIdsTensor_, dstExpertId, CMPMODE::EQ, compareCount);
//         PipeBarrier<PIPE_V>();
//         GatherMask(gatherTempTensor, expertIdsTensor_, expertMaskTensorU32[maskSizePerExpert * expertIndex / sizeof(uint32_t)],
//             true, mask, {1, 1, 0, 0}, maskCnt); // 是否可以简化计算
//         SyncFunc<AscendC::HardEvent::V_S>();
//         tokenNumToExpertTensor_.SetValue(expertIndex, static_cast<uint32_t>(maskCnt));
//     }
// }

// //计算出tensor之后直接相乘 这样可以得到想要的，然后累加。