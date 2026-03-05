#include "max_pool3d_with_argmax_v2_tiling_base.h"

constexpr uint64_t TILING_KEY_NOSPLIT_FLOAT = 100000;
constexpr uint64_t TILING_KEY_NOSPLIT_HALF = 100001;
constexpr uint64_t TILING_KEY_NOSPLIT_BF16 = 100002;


namespace optiling {

bool MaxPool3DWithArgmaxV2NoSplitTiling::IsCapable()
{
    array<uint64_t, DHW_DIMS> tmpOutShape{
        inputData.outShape[D_DIM], inputData.outShape[H_DIM], inputData.outShape[W_DIM]};
    auto summaryMemory = CalcBufferSizes(
        padInputData.padInputShape, tmpOutShape, padInputData.padInputShape[W_DIM] * padInputData.padInputShape[H_DIM],
        bufSizes);
    return (summaryMemory < ubSizeNew);
}

uint64_t MaxPool3DWithArgmaxV2NoSplitTiling::GetTilingKey() const
{
    if (dtype == ge::DataType::DT_FLOAT) {
        return TILING_KEY_NOSPLIT_FLOAT;
    } else if (dtype == ge::DataType::DT_FLOAT16) {
        return TILING_KEY_NOSPLIT_HALF;
    } else {
        return TILING_KEY_NOSPLIT_BF16;
    }
}

void MaxPool3DWithArgmaxV2NoSplitTiling::DoUBTiling()
{
    uint64_t batchesBlock = inputData.batches / blockLength;
    uint64_t batchesRem = inputData.batches % blockLength;
    const uint64_t batchesBlockPerCore = batchesBlock / coreNum;
    const uint64_t leftOverBatchesBlock = batchesBlock % coreNum;
    batchesPerCore = batchesBlockPerCore * blockLength;
    leftOverBatches = leftOverBatchesBlock * blockLength + batchesRem;
}

void MaxPool3DWithArgmaxV2NoSplitTiling::SetTilingData()
{
    tiling.set_inputShapes(&(inputData.inputShape[0]));
    tiling.set_outShapes(&(inputData.outShape[0]));
    tiling.set_kD(inputData.kernelSize[D_DIM]);
    tiling.set_kH(inputData.kernelSize[H_DIM]);
    tiling.set_kW(inputData.kernelSize[W_DIM]);
    tiling.set_sD(inputData.stride[D_DIM]);
    tiling.set_sH(inputData.stride[H_DIM]);
    tiling.set_sW(inputData.stride[W_DIM]);
    tiling.set_pD(inputData.pad[D_DIM]);
    tiling.set_pH(inputData.pad[H_DIM]);
    tiling.set_pW(inputData.pad[W_DIM]);
    tiling.set_dD(inputData.dilation[D_DIM]);
    tiling.set_dH(inputData.dilation[H_DIM]);
    tiling.set_dW(inputData.dilation[W_DIM]);
    tiling.set_batchesPerCore(batchesPerCore);
    tiling.set_leftOverBatches(leftOverBatches);
    tiling.set_partD(padInputData.padInputShape[D_DIM]);
    tiling.set_partH(padInputData.padInputShape[H_DIM]);
    tiling.set_partW(padInputData.padInputShape[W_DIM]);
    tiling.set_minFloat(-std::numeric_limits<float>::infinity());
    tiling.set_ceilD(padInputData.ceil[D_DIM]);
    tiling.set_ceilH(padInputData.ceil[H_DIM]);
    tiling.set_ceilW(padInputData.ceil[W_DIM]);
    tiling.set_sizeUb1(bufSizes.sizeUb1);
    tiling.set_sizeUb2(bufSizes.sizeUb2);
    tiling.set_sizeValues(bufSizes.valSize);
}

ge::graphStatus MaxPool3DWithArgmaxV2NoSplitTiling::DoOpTiling()
{
    DoUBTiling();
    SetTilingData();

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPool3DWithArgmaxV2NoSplitTiling::PostTiling()
{
    context_->SetBlockDim(coreNum);
    tiling.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    std::cout << "*******************START*******************" << std::endl;
    std::cout << "coreNum = " << coreNum << std::endl;
    std::cout << "batchesPerCore = " << tiling.get_batchesPerCore() << std::endl;
    std::cout << "leftOverBatches = " << tiling.get_leftOverBatches() << std::endl;
    std::cout << "inputD = " << tiling.get_inputShapes()[0] << std::endl;
    std::cout << "inputH = " << tiling.get_inputShapes()[1] << std::endl;
    std::cout << "inputW = " << tiling.get_inputShapes()[2] << std::endl;
    std::cout << "outD = " << tiling.get_outShapes()[0] << std::endl;
    std::cout << "outH = " << tiling.get_outShapes()[1] << std::endl;
    std::cout << "outW = " << tiling.get_outShapes()[2] << std::endl;
    std::cout << "kD = " << tiling.get_kD() << std::endl;
    std::cout << "kH = " << tiling.get_kH() << std::endl;
    std::cout << "kW = " << tiling.get_kW() << std::endl;
    std::cout << "sD = " << tiling.get_sD() << std::endl;
    std::cout << "sH = " << tiling.get_sH() << std::endl;
    std::cout << "sW = " << tiling.get_sW() << std::endl;
    std::cout << "pD = " << tiling.get_pD() << std::endl;
    std::cout << "pH = " << tiling.get_pH() << std::endl;
    std::cout << "pW = " << tiling.get_pW() << std::endl;
    std::cout << "dD = " << tiling.get_dD() << std::endl;
    std::cout << "dH = " << tiling.get_dH() << std::endl;
    std::cout << "dW = " << tiling.get_dW() << std::endl;
    std::cout << "partD = " << tiling.get_partD() << std::endl;
    std::cout << "partH = " << tiling.get_partH() << std::endl;
    std::cout << "partW = " << tiling.get_partW() << std::endl;
    std::cout << "ceilD = " << tiling.get_ceilD() << std::endl;
    std::cout << "ceilH = " << tiling.get_ceilH() << std::endl;
    std::cout << "ceilW = " << tiling.get_ceilW() << std::endl;
    std::cout << "sizeUb1 = " << tiling.get_sizeUb1() << std::endl;
    std::cout << "sizeUb2 = " << tiling.get_sizeUb2() << std::endl;
    std::cout << "sizeValues = " << tiling.get_sizeValues() << std::endl;
    std::cout << "*******************END*******************" << std::endl;

    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("MaxPool3DWithArgmaxV2", MaxPool3DWithArgmaxV2NoSplitTiling, 2);

} // namespace optiling