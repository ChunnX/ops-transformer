
#include "mi_chunk_local_cumsum_v4_tiling.h"
#include "log/log.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto aivNum = ascendcPlatform.GetCoreNumAiv();
    // int32_t asccoreNum = ascendcPlatform.GetCoreNum();

    const int32_t chunkSize = *(context->GetAttrs()->GetAttrPointer<int32_t>(0));
    const int64_t *cuSeqlenDictAttr =
        const_cast<int64_t *>(context->GetAttrs()->GetListInt(1)->GetData());

    auto shape_input = context->GetInputTensor(0)->GetOriginShape();
    auto shape_cu_seqlens = context->GetInputTensor(1)->GetOriginShape();

    // int32_t batchSize = useCuSeqlen ? 1 : shape_q.GetDim(0);
    int32_t headNum = shape_input.GetDim(2);
    int32_t tokenNum = shape_input.GetDim(1);
    int32_t seqNum = shape_cu_seqlens.GetDim(0) - 1;

    constexpr int32_t MAX_SEQ_NUM = 128;  // curChunkNumDict 固定 128 个槽，对应最大 batch 数
    if (seqNum > MAX_SEQ_NUM) {
        OP_LOGE("MIChunkLocalCumsumV4", "cu_seqlens length is %ld, seqNum %d exceeds max supported batch %d.",
                static_cast<int64_t>(shape_cu_seqlens.GetDim(0)), seqNum, MAX_SEQ_NUM);
        return ge::GRAPH_FAILED;
    }

    constexpr int32_t MAX_HEAD_NUM = 32;    // kernel UB 槽按 64*32*4B=8KB 分配，headNum>32 时单组超容量
    constexpr int32_t MAX_CHUNK_SIZE = 64;  // kernel 侧 UB 槽与扫描逻辑均按 64 设计
    if (headNum > MAX_HEAD_NUM) {
        OP_LOGE("MIChunkLocalCumsumV4", "headNum %d exceeds max supported %d.", headNum, MAX_HEAD_NUM);
        return ge::GRAPH_FAILED;
    }
    if (chunkSize > MAX_CHUNK_SIZE) {
        OP_LOGE("MIChunkLocalCumsumV4", "chunkSize %d exceeds max supported %d.", chunkSize, MAX_CHUNK_SIZE);
        return ge::GRAPH_FAILED;
    }

    if (headNum > 16) {
        context->SetTilingKey(1);
    } else if (headNum > 8) {
        context->SetTilingKey(2);
    }else if (headNum == 8){
        context->SetTilingKey(3);
    }else{
        context->SetTilingKey(4);
    }

    // printf("seqNum:%i\n",seqNum);
    // printf("tokenNum:%i\n:",tokenNum);
    // printf("headNum :%i\n",headNum);
    // printf("voHeadNum :%i\n",voHeadNum);
    // printf("headDim:%i\n",headDim);
    // printf("chunkNum:%i\n",chunkNum);
    // printf("attentionScale:%f\n",attentionScale);
    // printf("chunkSize:%i\n",chunkSize);
    // printf("aicNum:%i\n",aicNum);

    int32_t totalChunkNum = 0;
    int32_t curChunkNumArray[128] = {0};  // 初始化为0

    for (int32_t seqIdx = 0; seqIdx < seqNum; seqIdx++) {
        // printf("curQBlockNum : %i\n", seqlenKvDictAttr[0]);
        int32_t curSeqlen = cuSeqlenDictAttr[seqIdx+1] - cuSeqlenDictAttr[seqIdx];
        int32_t curChunkNum = ((curSeqlen + chunkSize - 1) / chunkSize);

        totalChunkNum += curChunkNum;
        curChunkNumArray[seqIdx] = totalChunkNum;
    }


    MIChunkLocalCumsumV4TilingData tiling;
    int32_t usedCore = totalChunkNum  > aivNum ? aivNum : totalChunkNum;

    // printf("usedCore:%i\n",usedCore);

    context->SetBlockDim(usedCore);  // 直接拉起所有的core
    tiling.set_coreNum(aivNum);
    tiling.set_tokenNum(tokenNum);
    tiling.set_headNum(headNum);
    tiling.set_chunkNum(totalChunkNum);
    tiling.set_chunkSize(chunkSize);
    tiling.set_curChunkNumDict(curChunkNumArray);
 

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t userWorkspaceSize = 0;
    size_t systemWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = userWorkspaceSize + systemWorkspaceSize;

    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ops {
class MIChunkLocalCumsumV4 : public OpDef {
   public:
    explicit MIChunkLocalCumsumV4(const char *name) : OpDef(name) {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("cu_seqlens")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("chunkSize").Int();
        this->Attr("cuSeqlenDict").ListInt();
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(MIChunkLocalCumsumV4);
}  // namespace ops
