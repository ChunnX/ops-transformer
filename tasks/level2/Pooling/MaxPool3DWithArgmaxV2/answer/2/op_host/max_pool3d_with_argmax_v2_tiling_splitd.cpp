#include "max_pool3d_with_argmax_v2_tiling_base.h"

constexpr uint64_t TILING_KEY_SPLITD_FLOAT = 110000;
constexpr uint64_t TILING_KEY_SPLITD_HALF = 110001;
constexpr uint64_t TILING_KEY_SPLITD_BF16 = 110002;


namespace optiling {

bool MaxPool3DWithArgmaxV2SplitDTiling::IsCapable()
{
    array<uint64_t, DHW_DIMS> parts{
        padInputData.padInputShape[D_DIM], padInputData.padInputShape[H_DIM], padInputData.padInputShape[W_DIM]};
    array<uint64_t, DHW_DIMS> partOuts{inputData.outShape[D_DIM], inputData.outShape[H_DIM], inputData.outShape[W_DIM]};

    FindSplitParts(parts, partOuts, bufSizes, D_DIM);

    splitData.partD = parts[D_DIM];
    splitData.partOutD = partOuts[D_DIM];
    return ((splitData.partD != 0UL) && (splitData.partOutD != 0UL));
}

uint64_t MaxPool3DWithArgmaxV2SplitDTiling::GetTilingKey() const
{
    if (dtype == ge::DataType::DT_FLOAT) {
        return TILING_KEY_SPLITD_FLOAT;
    } else if (dtype == ge::DataType::DT_FLOAT16) {
        return TILING_KEY_SPLITD_HALF;
    } else {
        return TILING_KEY_SPLITD_BF16;
    }
}

ge::graphStatus MaxPool3DWithArgmaxV2SplitDTiling::SmallBatchesTiling(void)
{
    if (splitData.partOutD == 0UL) {
        return ge::GRAPH_FAILED;
    }
    uint64_t batchesBlock = splitData.leftOverBatches / blockLength;
    uint64_t batchesRem = splitData.leftOverBatches % blockLength;
    uint64_t wholeCount = RoundDownBlock(inputData.outShape[D_DIM], splitData.partOutD) / splitData.partOutD +
                          ((inputData.outShape[D_DIM] % splitData.partOutD) != 0 ? 1 : 0);

    uint64_t additionBatch = (batchesRem != 0UL) ? 1UL : 0UL;
    splitData.partsPerCore = (wholeCount * (batchesBlock + additionBatch)) / coreNum;
    splitData.leftOverParts = (wholeCount * (batchesBlock + additionBatch)) % coreNum;
    uint64_t count = 0;
    uint64_t curPos = 0;
    uint64_t d = 0;
    uint64_t dOut = 0;
    uint64_t offD =
        splitData.partD - (inputData.kernelSize[D_DIM] - 1) * inputData.dilation[D_DIM] + inputData.stride[D_DIM] - 1;
    for (uint64_t b = 0UL; b < (batchesBlock + (batchesRem != 0UL)); b++) {
        if ((count != 0UL) && (curPos > 0UL)) {
            if (b < batchesBlock) {
                splitData.sizeIn[curPos - 1UL] += blockLength;
            } else {
                splitData.sizeIn[curPos - 1UL] += batchesRem;
            }
        }
        for (d = 0, dOut = 0; dOut < inputData.outShape[D_DIM]; d += offD, dOut += splitData.partOutD) {
            count++;
            if (count == 1) {
                splitData.splitPointDIn[curPos] = d;
                splitData.splitPointDOut[curPos] = dOut;
                splitData.sizeIn[curPos] = b < batchesBlock ? blockLength : batchesRem;
                splitData.batchStart[curPos] = b * blockLength;
                curPos++;
            }
            if (count > (splitData.partsPerCore + (curPos <= splitData.leftOverParts ? 1 : 0) - 1)) {
                count = 0;
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

void MaxPool3DWithArgmaxV2SplitDTiling::DoUBTiling()
{
    uint64_t batchesBlock = inputData.batches / blockLength;
    uint64_t batchesRem = inputData.batches % blockLength;
    const uint64_t batchesBlockPerCore = batchesBlock / coreNum;
    const uint64_t leftOverBatchesBlock = batchesBlock % coreNum;
    splitData.batchesPerCore = blockLength * batchesBlockPerCore;
    splitData.leftOverBatches = leftOverBatchesBlock * blockLength + batchesRem;

    if (splitData.batchesPerCore == 0UL) {
        SmallBatchesTiling();
    }
}

void MaxPool3DWithArgmaxV2SplitDTiling::SetTilingData()
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
    tiling.set_batchesPerCore(splitData.batchesPerCore);
    tiling.set_leftOverBatches(splitData.leftOverBatches);
    tiling.set_partD(splitData.partD);
    tiling.set_partH(padInputData.padInputShape[H_DIM]);
    tiling.set_partW(padInputData.padInputShape[W_DIM]);
    tiling.set_minFloat(-std::numeric_limits<float>::infinity());
    tiling.set_ceilD(padInputData.ceil[D_DIM]);
    tiling.set_ceilH(padInputData.ceil[H_DIM]);
    tiling.set_ceilW(padInputData.ceil[W_DIM]);
    tiling.set_sizeUb1(bufSizes.sizeUb1);
    tiling.set_sizeUb2(bufSizes.sizeUb2);
    tiling.set_sizeValues(bufSizes.valSize);

    tiling.set_partsPerCore(splitData.partsPerCore);
    tiling.set_leftOverParts(splitData.leftOverParts);
    tiling.set_splitPointDIn(splitData.splitPointDIn);
    tiling.set_splitPointDOut(splitData.splitPointDOut);
    tiling.set_sizeIn(splitData.sizeIn);
    tiling.set_batchStart(splitData.batchStart);
}

ge::graphStatus MaxPool3DWithArgmaxV2SplitDTiling::DoOpTiling()
{
    DoUBTiling();
    SetTilingData();

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPool3DWithArgmaxV2SplitDTiling::PostTiling()
{
    context_->SetBlockDim(coreNum);
    tiling.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    std::cout << "*******************START*******************" << std::endl;
    std::cout << "coreNum = " << coreNum << std::endl;
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
    std::cout << "batchesPerCore = " << tiling.get_batchesPerCore() << std::endl;
    std::cout << "leftOverBatches = " << tiling.get_leftOverBatches() << std::endl;
    std::cout << "partD = " << tiling.get_partD() << std::endl;
    std::cout << "partH = " << tiling.get_partH() << std::endl;
    std::cout << "partW = " << tiling.get_partW() << std::endl;
    std::cout << "ceilD = " << tiling.get_ceilD() << std::endl;
    std::cout << "ceilH = " << tiling.get_ceilH() << std::endl;
    std::cout << "ceilW = " << tiling.get_ceilW() << std::endl;
    std::cout << "sizeUb1 = " << tiling.get_sizeUb1() << std::endl;
    std::cout << "sizeUb2 = " << tiling.get_sizeUb2() << std::endl;
    std::cout << "sizeValues = " << tiling.get_sizeValues() << std::endl;
    std::cout << "partsPerCore = " << tiling.get_partsPerCore() << std::endl;
    std::cout << "leftOverParts = " << tiling.get_leftOverParts() << std::endl;
    std::cout << "*******************END*******************" << std::endl;

    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("MaxPool3DWithArgmaxV2", MaxPool3DWithArgmaxV2SplitDTiling, 3);

} // namespace optiling