#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

#include "acl/acl.h"
#include "aclnn_abs.h"
#include "aclnn_add.h"
#include "aclnn_amax.h"
#include "aclnn_amin.h"
#include "aclnn_cast.h"
#include "aclnn_cat.h"
#include "aclnn_ceil.h"
#include "aclnn_channel_shuffle.h"
#include "aclnn_chunk_cat.h"
#include "aclnn_clamp.h"
#include "aclnn_cos.h"
#include "aclnn_cumsum.h"
#include "aclnn_div.h"
#include "aclnn_exp.h"
#include "aclnn_floor.h"
#include "aclnn_ger.h"
#include "aclnn_global_average_pool.h"
#include "aclnn_global_max_pool.h"
#include "aclnn_hardtanh.h"
#include "aclnn_is_inf.h"
#include "aclnn_isfinite.h"
#include "aclnn_log.h"
#include "aclnn_masked_select.h"
#include "aclnn_max.h"
#include "aclnn_maximum.h"
#include "aclnn_min.h"
#include "aclnn_minimum.h"
#include "aclnn_mul.h"
#include "aclnn_neg.h"
#include "aclnn_permute.h"
#include "aclnn_prod.h"
#include "aclnn_reciprocal.h"
#include "aclnn_reduce_sum.h"
#include "aclnn_round.h"
#include "aclnn_rsqrt.h"
#include "aclnn_sign.h"
#include "aclnn_sin.h"
#include "aclnn_slice.h"
#include "aclnn_sqrt.h"
#include "aclnn_split_with_size.h"
#include "aclnn_std.h"
#include "aclnn_stack.h"
#include "aclnn_strided_slice.h"
#include "aclnn_sub.h"
#include "aclnn_tanh.h"
#include "aclnn_tril.h"
#include "aclnn_trunc.h"
#include "aclnn_triu.h"
#include "aclnn_unfold_grad.h"
#include "aclnn_var.h"
#include "aclnn_var_mean.h"

namespace {

enum class HalfKind {
    Fp16,
    Bf16,
};

template <HalfKind KindValue>
struct HalfStorage {
    static constexpr HalfKind kind = KindValue;
    uint16_t bits = 0;
};

template <typename T>
struct IsHalfStorage : std::false_type {};

template <HalfKind KindValue>
struct IsHalfStorage<HalfStorage<KindValue>> : std::true_type {};

struct Args {
    std::string api;
    std::string dtype;
    std::string dtypeB;
    std::string dtypeC;
    std::vector<int64_t> shapeA;
    std::vector<int64_t> shapeB;
    std::vector<int64_t> shapeC;
    std::vector<int64_t> maskShape;
    std::vector<std::vector<int64_t>> tensorShapes;
    std::vector<std::pair<double, double>> tensorRanges;
    double rangeALow = -1.0;
    double rangeAHigh = 1.0;
    double rangeBLow = -1.0;
    double rangeBHigh = 1.0;
    double rangeCLow = -1.0;
    double rangeCHigh = 1.0;
    double alpha = 1.0;
    double other = 1.0;
    double clipMin = 0.0;
    double clipMax = 0.0;
    int64_t dim = 0;
    int64_t numChunks = 1;
    std::vector<int64_t> splitSizes;
    std::vector<int64_t> dims;
    int64_t groups = 1;
    bool keepDims = false;
    bool noopWithEmptyAxes = false;
    bool exclusive = false;
    bool reverse = false;
    int64_t diagonal = 0;
    int64_t correction = 0;
    int64_t size = 0;
    int64_t start = 0;
    int64_t sliceEnd = 0;
    int64_t step = 1;
    std::vector<int64_t> begin;
    std::vector<int64_t> ends;
    std::vector<int64_t> strides;
    std::vector<int64_t> inputSizes;
    int64_t beginMask = 0;
    int64_t endMask = 0;
    int64_t ellipsisMask = 0;
    int64_t newAxisMask = 0;
    int64_t shrinkAxisMask = 0;
    std::string dtypeAttr;
    std::string outDtype;
    std::vector<int64_t> queryShape;
    std::vector<int64_t> keyShape;
    std::vector<int64_t> weightsShape;
    std::vector<int64_t> valueShape;
    std::vector<int64_t> betaShape;
    std::vector<int64_t> stateShape;
    std::vector<int64_t> gShape;
    std::vector<int64_t> gkShape;
    std::vector<int64_t> blockTableShape;
    std::vector<int32_t> actualSeqLengthsQuery;
    std::vector<int32_t> actualSeqLengthsKey;
    std::vector<int32_t> blockTableValues;
    std::vector<int32_t> actualSeqLengths;
    std::vector<int32_t> ssmStateIndices;
    std::vector<int32_t> numAcceptedTokens;
    double queryRangeLow = -1.0;
    double queryRangeHigh = 1.0;
    double keyRangeLow = -1.0;
    double keyRangeHigh = 1.0;
    double weightsRangeLow = -1.0;
    double weightsRangeHigh = 1.0;
    double valueRangeLow = -1.0;
    double valueRangeHigh = 1.0;
    double betaRangeLow = 0.0;
    double betaRangeHigh = 1.0;
    double stateRangeLow = -1.0;
    double stateRangeHigh = 1.0;
    double gRangeLow = 0.0;
    double gRangeHigh = 0.0;
    double gkRangeLow = 0.0;
    double gkRangeHigh = 0.0;
    std::string weightsDtype;
    std::string layoutQuery = "BSND";
    std::string layoutKey = "BSND";
    int64_t sparseCount = 0;
    int64_t sparseMode = 0;
    int64_t preTokens = std::numeric_limits<int64_t>::max();
    int64_t nextTokens = std::numeric_limits<int64_t>::max();
    bool returnValues = false;
    bool gPresent = false;
    bool gkPresent = false;
    float scaleValue = 1.0F;
    int32_t device = 0;
    std::string opapiLib;
    std::string batchFile;
};

struct NativeResult {
    std::string status = "passed";
    std::string message;
    double maxAbsError = 0.0;
    int retCode = 0;
};

struct NativeStageTimings {
    double parseMs = 0.0;
    double inputGenerateH2dMs = 0.0;
    double goldenMs = 0.0;
    double workspaceMs = 0.0;
    double launchSyncMs = 0.0;
    double d2hMs = 0.0;
    double compareMs = 0.0;
};

using SteadyClock = std::chrono::steady_clock;

double ElapsedMs(SteadyClock::time_point start)
{
    return std::chrono::duration<double, std::milli>(SteadyClock::now() - start).count();
}

double *StageTimingTarget(NativeStageTimings *timings, double NativeStageTimings::*field)
{
    if (timings == nullptr) {
        return nullptr;
    }
    return &(timings->*field);
}

class ScopedStageTimer {
public:
    explicit ScopedStageTimer(double *target) : target_(target), start_(SteadyClock::now()) {}

    ~ScopedStageTimer()
    {
        if (target_ != nullptr) {
            *target_ += ElapsedMs(start_);
        }
    }

private:
    double *target_ = nullptr;
    SteadyClock::time_point start_;
};

struct TensorHolder {
    void *deviceAddr = nullptr;
    aclTensor *tensor = nullptr;

    void Cleanup()
    {
        if (tensor != nullptr) {
            aclDestroyTensor(tensor);
            tensor = nullptr;
        }
        if (deviceAddr != nullptr) {
            aclrtFree(deviceAddr);
            deviceAddr = nullptr;
        }
    }
};

std::vector<std::string> Split(const std::string &value, char delim)
{
    std::vector<std::string> result;
    std::stringstream ss(value);
    std::string item;
    while (std::getline(ss, item, delim)) {
        if (!item.empty()) {
            result.push_back(item);
        }
    }
    return result;
}

std::vector<int64_t> ParseShape(const std::string &value)
{
    std::vector<int64_t> shape;
    for (const auto &item : Split(value, ',')) {
        shape.push_back(std::stoll(item));
    }
    return shape;
}

std::vector<int32_t> ParseInt32List(const std::string &value)
{
    std::vector<int32_t> values;
    for (const auto &item : Split(value, ',')) {
        values.push_back(static_cast<int32_t>(std::stoll(item)));
    }
    return values;
}

std::vector<std::vector<int64_t>> ParseShapeList(const std::string &value)
{
    std::vector<std::vector<int64_t>> shapes;
    for (const auto &item : Split(value, ';')) {
        shapes.push_back(ParseShape(item));
    }
    return shapes;
}

std::pair<double, double> ParseRange(const std::string &value)
{
    auto parts = Split(value, ',');
    if (parts.size() < 2) {
        return {-1.0, 1.0};
    }
    return {std::stod(parts[0]), std::stod(parts[1])};
}

std::vector<std::pair<double, double>> ParseRangeList(const std::string &value)
{
    std::vector<std::pair<double, double>> ranges;
    for (const auto &item : Split(value, ';')) {
        ranges.push_back(ParseRange(item));
    }
    return ranges;
}

bool ParseBool(const std::string &value)
{
    return value == "1" || value == "true" || value == "True" || value == "TRUE";
}

Args ParseArgs(int argc, char **argv)
{
    std::map<std::string, std::string> values;
    for (int i = 1; i + 1 < argc; i += 2) {
        values[argv[i]] = argv[i + 1];
    }

    Args args;
    args.api = values["--api"];
    args.dtype = values["--dtype"];
    if (values.count("--dtype-b")) {
        args.dtypeB = values["--dtype-b"];
    }
    if (values.count("--dtype-c")) {
        args.dtypeC = values["--dtype-c"];
    }
    if (values.count("--shape-a")) {
        args.shapeA = ParseShape(values["--shape-a"]);
    }
    if (values.count("--shape-b")) {
        args.shapeB = ParseShape(values["--shape-b"]);
    }
    if (values.count("--shape-c")) {
        args.shapeC = ParseShape(values["--shape-c"]);
    }
    if (values.count("--mask-shape")) {
        args.maskShape = ParseShape(values["--mask-shape"]);
    }
    if (values.count("--shapes")) {
        args.tensorShapes = ParseShapeList(values["--shapes"]);
    }
    if (values.count("--ranges")) {
        args.tensorRanges = ParseRangeList(values["--ranges"]);
    }
    if (values.count("--range-a")) {
        auto parsed = ParseRange(values["--range-a"]);
        args.rangeALow = parsed.first;
        args.rangeAHigh = parsed.second;
    }
    if (values.count("--range-b")) {
        auto parsed = ParseRange(values["--range-b"]);
        args.rangeBLow = parsed.first;
        args.rangeBHigh = parsed.second;
    }
    if (values.count("--range-c")) {
        auto parsed = ParseRange(values["--range-c"]);
        args.rangeCLow = parsed.first;
        args.rangeCHigh = parsed.second;
    }
    if (values.count("--alpha")) {
        args.alpha = std::stod(values["--alpha"]);
    }
    if (values.count("--other")) {
        args.other = std::stod(values["--other"]);
    }
    if (values.count("--min")) {
        args.clipMin = std::stod(values["--min"]);
    }
    if (values.count("--max")) {
        args.clipMax = std::stod(values["--max"]);
    }
    if (values.count("--dim")) {
        args.dim = std::stoll(values["--dim"]);
    }
    if (values.count("--num-chunks")) {
        args.numChunks = std::stoll(values["--num-chunks"]);
    }
    if (values.count("--split-sizes")) {
        args.splitSizes = ParseShape(values["--split-sizes"]);
    }
    if (values.count("--dims")) {
        args.dims = ParseShape(values["--dims"]);
    }
    if (values.count("--groups")) {
        args.groups = std::stoll(values["--groups"]);
    }
    if (values.count("--keep-dims")) {
        args.keepDims = ParseBool(values["--keep-dims"]);
    }
    if (values.count("--noop-with-empty-axes")) {
        args.noopWithEmptyAxes = ParseBool(values["--noop-with-empty-axes"]);
    }
    if (values.count("--exclusive")) {
        args.exclusive = ParseBool(values["--exclusive"]);
    }
    if (values.count("--reverse")) {
        args.reverse = ParseBool(values["--reverse"]);
    }
    if (values.count("--diagonal")) {
        args.diagonal = std::stoll(values["--diagonal"]);
    }
    if (values.count("--correction")) {
        args.correction = std::stoll(values["--correction"]);
    }
    if (values.count("--size")) {
        args.size = std::stoll(values["--size"]);
    }
    if (values.count("--start")) {
        args.start = std::stoll(values["--start"]);
    }
    if (values.count("--end")) {
        if (args.api == "StridedSlice") {
            args.ends = ParseShape(values["--end"]);
        } else {
            args.sliceEnd = std::stoll(values["--end"]);
        }
    }
    if (values.count("--step")) {
        args.step = std::stoll(values["--step"]);
    }
    if (values.count("--begin")) {
        args.begin = ParseShape(values["--begin"]);
    }
    if (values.count("--strides")) {
        args.strides = ParseShape(values["--strides"]);
    }
    if (values.count("--input-sizes")) {
        args.inputSizes = ParseShape(values["--input-sizes"]);
    }
    if (values.count("--begin-mask")) {
        args.beginMask = std::stoll(values["--begin-mask"]);
    }
    if (values.count("--end-mask")) {
        args.endMask = std::stoll(values["--end-mask"]);
    }
    if (values.count("--ellipsis-mask")) {
        args.ellipsisMask = std::stoll(values["--ellipsis-mask"]);
    }
    if (values.count("--new-axis-mask")) {
        args.newAxisMask = std::stoll(values["--new-axis-mask"]);
    }
    if (values.count("--shrink-axis-mask")) {
        args.shrinkAxisMask = std::stoll(values["--shrink-axis-mask"]);
    }
    if (values.count("--dtype-attr")) {
        args.dtypeAttr = values["--dtype-attr"];
    }
    if (values.count("--out-dtype")) {
        args.outDtype = values["--out-dtype"];
    }
    if (values.count("--device")) {
        args.device = static_cast<int32_t>(std::stoi(values["--device"]));
    }
    if (values.count("--opapi-lib")) {
        args.opapiLib = values["--opapi-lib"];
    }
    if (values.count("--batch-file")) {
        args.batchFile = values["--batch-file"];
    }
    return args;
}

Args ParseBatchLine(const std::string &line, int32_t device, const std::string &opapiLib, int *caseIndex)
{
    std::map<std::string, std::string> values;
    for (const auto &item : Split(line, '\t')) {
        const auto pos = item.find('=');
        if (pos == std::string::npos) {
            continue;
        }
        values[item.substr(0, pos)] = item.substr(pos + 1);
    }

    Args args;
    args.api = values["api"];
    args.dtype = values["dtype"];
    if (values.count("dtype_b")) {
        args.dtypeB = values["dtype_b"];
    }
    if (values.count("dtype_c")) {
        args.dtypeC = values["dtype_c"];
    }
    if (values.count("shape_a")) {
        args.shapeA = ParseShape(values["shape_a"]);
    }
    if (values.count("shape_b")) {
        args.shapeB = ParseShape(values["shape_b"]);
    }
    if (values.count("shape_c")) {
        args.shapeC = ParseShape(values["shape_c"]);
    }
    if (values.count("mask_shape")) {
        args.maskShape = ParseShape(values["mask_shape"]);
    }
    if (values.count("shapes")) {
        args.tensorShapes = ParseShapeList(values["shapes"]);
    }
    if (values.count("ranges")) {
        args.tensorRanges = ParseRangeList(values["ranges"]);
    }
    if (values.count("range_a")) {
        auto parsed = ParseRange(values["range_a"]);
        args.rangeALow = parsed.first;
        args.rangeAHigh = parsed.second;
    }
    if (values.count("range_b")) {
        auto parsed = ParseRange(values["range_b"]);
        args.rangeBLow = parsed.first;
        args.rangeBHigh = parsed.second;
    }
    if (values.count("range_c")) {
        auto parsed = ParseRange(values["range_c"]);
        args.rangeCLow = parsed.first;
        args.rangeCHigh = parsed.second;
    }
    if (values.count("alpha")) {
        args.alpha = std::stod(values["alpha"]);
    }
    if (values.count("other")) {
        args.other = std::stod(values["other"]);
    }
    if (values.count("min")) {
        args.clipMin = std::stod(values["min"]);
    }
    if (values.count("max")) {
        args.clipMax = std::stod(values["max"]);
    }
    if (values.count("dim")) {
        args.dim = std::stoll(values["dim"]);
    }
    if (values.count("num_chunks")) {
        args.numChunks = std::stoll(values["num_chunks"]);
    }
    if (values.count("split_sizes")) {
        args.splitSizes = ParseShape(values["split_sizes"]);
    }
    if (values.count("dims")) {
        args.dims = ParseShape(values["dims"]);
    }
    if (values.count("groups")) {
        args.groups = std::stoll(values["groups"]);
    }
    if (values.count("keep_dims")) {
        args.keepDims = ParseBool(values["keep_dims"]);
    }
    if (values.count("noop_with_empty_axes")) {
        args.noopWithEmptyAxes = ParseBool(values["noop_with_empty_axes"]);
    }
    if (values.count("exclusive")) {
        args.exclusive = ParseBool(values["exclusive"]);
    }
    if (values.count("reverse")) {
        args.reverse = ParseBool(values["reverse"]);
    }
    if (values.count("diagonal")) {
        args.diagonal = std::stoll(values["diagonal"]);
    }
    if (values.count("correction")) {
        args.correction = std::stoll(values["correction"]);
    }
    if (values.count("size")) {
        args.size = std::stoll(values["size"]);
    }
    if (values.count("start")) {
        args.start = std::stoll(values["start"]);
    }
    if (values.count("end")) {
        if (args.api == "StridedSlice") {
            args.ends = ParseShape(values["end"]);
        } else {
            args.sliceEnd = std::stoll(values["end"]);
        }
    }
    if (values.count("step")) {
        args.step = std::stoll(values["step"]);
    }
    if (values.count("begin")) {
        args.begin = ParseShape(values["begin"]);
    }
    if (values.count("strides")) {
        args.strides = ParseShape(values["strides"]);
    }
    if (values.count("input_sizes")) {
        args.inputSizes = ParseShape(values["input_sizes"]);
    }
    if (values.count("begin_mask")) {
        args.beginMask = std::stoll(values["begin_mask"]);
    }
    if (values.count("end_mask")) {
        args.endMask = std::stoll(values["end_mask"]);
    }
    if (values.count("ellipsis_mask")) {
        args.ellipsisMask = std::stoll(values["ellipsis_mask"]);
    }
    if (values.count("new_axis_mask")) {
        args.newAxisMask = std::stoll(values["new_axis_mask"]);
    }
    if (values.count("shrink_axis_mask")) {
        args.shrinkAxisMask = std::stoll(values["shrink_axis_mask"]);
    }
    if (values.count("dtype_attr")) {
        args.dtypeAttr = values["dtype_attr"];
    }
    if (values.count("out_dtype")) {
        args.outDtype = values["out_dtype"];
    }
    if (values.count("query_shape")) {
        args.queryShape = ParseShape(values["query_shape"]);
    }
    if (values.count("key_shape")) {
        args.keyShape = ParseShape(values["key_shape"]);
    }
    if (values.count("weights_shape")) {
        args.weightsShape = ParseShape(values["weights_shape"]);
    }
    if (values.count("value_shape")) {
        args.valueShape = ParseShape(values["value_shape"]);
    }
    if (values.count("beta_shape")) {
        args.betaShape = ParseShape(values["beta_shape"]);
    }
    if (values.count("state_shape")) {
        args.stateShape = ParseShape(values["state_shape"]);
    }
    if (values.count("g_shape")) {
        args.gShape = ParseShape(values["g_shape"]);
    }
    if (values.count("gk_shape")) {
        args.gkShape = ParseShape(values["gk_shape"]);
    }
    if (values.count("block_table_shape")) {
        args.blockTableShape = ParseShape(values["block_table_shape"]);
    }
    if (values.count("query_range")) {
        auto parsed = ParseRange(values["query_range"]);
        args.queryRangeLow = parsed.first;
        args.queryRangeHigh = parsed.second;
    }
    if (values.count("key_range")) {
        auto parsed = ParseRange(values["key_range"]);
        args.keyRangeLow = parsed.first;
        args.keyRangeHigh = parsed.second;
    }
    if (values.count("weights_range")) {
        auto parsed = ParseRange(values["weights_range"]);
        args.weightsRangeLow = parsed.first;
        args.weightsRangeHigh = parsed.second;
    }
    if (values.count("value_range")) {
        auto parsed = ParseRange(values["value_range"]);
        args.valueRangeLow = parsed.first;
        args.valueRangeHigh = parsed.second;
    }
    if (values.count("beta_range")) {
        auto parsed = ParseRange(values["beta_range"]);
        args.betaRangeLow = parsed.first;
        args.betaRangeHigh = parsed.second;
    }
    if (values.count("state_range")) {
        auto parsed = ParseRange(values["state_range"]);
        args.stateRangeLow = parsed.first;
        args.stateRangeHigh = parsed.second;
    }
    if (values.count("g_range")) {
        auto parsed = ParseRange(values["g_range"]);
        args.gRangeLow = parsed.first;
        args.gRangeHigh = parsed.second;
    }
    if (values.count("gk_range")) {
        auto parsed = ParseRange(values["gk_range"]);
        args.gkRangeLow = parsed.first;
        args.gkRangeHigh = parsed.second;
    }
    if (values.count("weights_dtype")) {
        args.weightsDtype = values["weights_dtype"];
    }
    if (values.count("layout_query")) {
        args.layoutQuery = values["layout_query"];
    }
    if (values.count("layout_key")) {
        args.layoutKey = values["layout_key"];
    }
    if (values.count("sparse_count")) {
        args.sparseCount = std::stoll(values["sparse_count"]);
    }
    if (values.count("sparse_mode")) {
        args.sparseMode = std::stoll(values["sparse_mode"]);
    }
    if (values.count("pre_tokens")) {
        args.preTokens = std::stoll(values["pre_tokens"]);
    }
    if (values.count("next_tokens")) {
        args.nextTokens = std::stoll(values["next_tokens"]);
    }
    if (values.count("return_values")) {
        args.returnValues = ParseBool(values["return_values"]);
    }
    if (values.count("actual_seq_lengths_query")) {
        args.actualSeqLengthsQuery = ParseInt32List(values["actual_seq_lengths_query"]);
    }
    if (values.count("actual_seq_lengths_key")) {
        args.actualSeqLengthsKey = ParseInt32List(values["actual_seq_lengths_key"]);
    }
    if (values.count("block_table_values")) {
        args.blockTableValues = ParseInt32List(values["block_table_values"]);
    }
    if (values.count("actual_seq_lengths")) {
        args.actualSeqLengths = ParseInt32List(values["actual_seq_lengths"]);
    }
    if (values.count("ssm_state_indices")) {
        args.ssmStateIndices = ParseInt32List(values["ssm_state_indices"]);
    }
    if (values.count("num_accepted_tokens")) {
        args.numAcceptedTokens = ParseInt32List(values["num_accepted_tokens"]);
    }
    if (values.count("g_present")) {
        args.gPresent = ParseBool(values["g_present"]);
    }
    if (values.count("gk_present")) {
        args.gkPresent = ParseBool(values["gk_present"]);
    }
    if (values.count("scale_value")) {
        args.scaleValue = static_cast<float>(std::stod(values["scale_value"]));
    }
    args.device = device;
    args.opapiLib = opapiLib;
    *caseIndex = values.count("case_index") ? std::stoi(values["case_index"]) : -1;
    return args;
}

void *g_opapiHandle = nullptr;

using AbsGetWorkspaceSizeFn = aclnnStatus (*)(const aclTensor *, aclTensor *, uint64_t *, aclOpExecutor **);
using AbsLaunchFn = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
using UnaryGetWorkspaceSizeFn = aclnnStatus (*)(const aclTensor *, aclTensor *, uint64_t *, aclOpExecutor **);
using UnaryLaunchFn = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
using AddGetWorkspaceSizeFn =
    aclnnStatus (*)(const aclTensor *, const aclTensor *, const aclScalar *, aclTensor *, uint64_t *, aclOpExecutor **);
using AddLaunchFn = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
using AddsGetWorkspaceSizeFn =
    aclnnStatus (*)(const aclTensor *, const aclScalar *, const aclScalar *, aclTensor *, uint64_t *, aclOpExecutor **);
using AddsLaunchFn = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
using AmaxGetWorkspaceSizeFn =
    aclnnStatus (*)(const aclTensor *, const aclIntArray *, bool, aclTensor *, uint64_t *, aclOpExecutor **);
using AmaxLaunchFn = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
using AminGetWorkspaceSizeFn =
    aclnnStatus (*)(const aclTensor *, const aclIntArray *, bool, aclTensor *, uint64_t *, aclOpExecutor **);
using AminLaunchFn = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
using CosGetWorkspaceSizeFn = aclnnStatus (*)(const aclTensor *, aclTensor *, uint64_t *, aclOpExecutor **);
using CosLaunchFn = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
using CumsumGetWorkspaceSizeFn =
    aclnnStatus (*)(const aclTensor *, int64_t, aclDataType, aclTensor *, uint64_t *, aclOpExecutor **);
using CumsumLaunchFn = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
using CumsumV2GetWorkspaceSizeFn =
    aclnnStatus (*)(const aclTensor *, int64_t, bool, bool, aclTensor *, uint64_t *, aclOpExecutor **);
using CumsumV2LaunchFn = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
using ScalarClipGetWorkspaceSizeFn =
    aclnnStatus (*)(const aclTensor *, const aclScalar *, const aclScalar *, aclTensor *, uint64_t *, aclOpExecutor **);
using ScalarClipLaunchFn = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
using ScalarClipOneBoundGetWorkspaceSizeFn =
    aclnnStatus (*)(const aclTensor *, const aclScalar *, aclTensor *, uint64_t *, aclOpExecutor **);
using ScalarClipOneBoundLaunchFn = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
using TensorClipGetWorkspaceSizeFn =
    aclnnStatus (*)(const aclTensor *, const aclTensor *, const aclTensor *, aclTensor *, uint64_t *, aclOpExecutor **);
using TensorClipOneBoundGetWorkspaceSizeFn =
    aclnnStatus (*)(const aclTensor *, const aclTensor *, aclTensor *, uint64_t *, aclOpExecutor **);
using TensorClipLaunchFn = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
using CastGetWorkspaceSizeFn =
    aclnnStatus (*)(const aclTensor *, aclDataType, aclTensor *, uint64_t *, aclOpExecutor **);
using CastLaunchFn = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
using TensorListGetWorkspaceSizeFn =
    aclnnStatus (*)(const aclTensorList *, int64_t, aclTensor *, uint64_t *, aclOpExecutor **);
using TensorListLaunchFn = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
using ChunkCatGetWorkspaceSizeFn =
    aclnnStatus (*)(const aclTensorList *, int64_t, int64_t, aclTensor *, uint64_t *, aclOpExecutor **);
using SplitWithSizeGetWorkspaceSizeFn =
    aclnnStatus (*)(const aclTensor *, const aclIntArray *, int64_t, aclTensorList *, uint64_t *, aclOpExecutor **);
using SplitWithSizeLaunchFn = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
using DivGetWorkspaceSizeFn =
    aclnnStatus (*)(const aclTensor *, const aclTensor *, aclTensor *, uint64_t *, aclOpExecutor **);
using DivLaunchFn = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
using DivsGetWorkspaceSizeFn =
    aclnnStatus (*)(const aclTensor *, const aclScalar *, aclTensor *, uint64_t *, aclOpExecutor **);
using DivsLaunchFn = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
using GerGetWorkspaceSizeFn = aclnnStatus (*)(const aclTensor *, const aclTensor *, aclTensor *, uint64_t *, aclOpExecutor **);
using GerLaunchFn = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
using MaximumGetWorkspaceSizeFn =
    aclnnStatus (*)(const aclTensor *, const aclTensor *, aclTensor *, uint64_t *, aclOpExecutor **);
using MaximumLaunchFn = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
using MinimumGetWorkspaceSizeFn =
    aclnnStatus (*)(const aclTensor *, const aclTensor *, aclTensor *, uint64_t *, aclOpExecutor **);
using MinimumLaunchFn = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
using MulGetWorkspaceSizeFn = aclnnStatus (*)(const aclTensor *, const aclTensor *, aclTensor *, uint64_t *, aclOpExecutor **);
using MulLaunchFn = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
using MulsGetWorkspaceSizeFn =
    aclnnStatus (*)(const aclTensor *, const aclScalar *, aclTensor *, uint64_t *, aclOpExecutor **);
using MulsLaunchFn = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
using MaskedSelectGetWorkspaceSizeFn =
    aclnnStatus (*)(const aclTensor *, const aclTensor *, aclTensor *, uint64_t *, aclOpExecutor **);
using MaskedSelectLaunchFn = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
using MaxGetWorkspaceSizeFn = aclnnStatus (*)(const aclTensor *, aclTensor *, uint64_t *, aclOpExecutor **);
using MaxLaunchFn = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
using MinGetWorkspaceSizeFn = aclnnStatus (*)(const aclTensor *, aclTensor *, uint64_t *, aclOpExecutor **);
using MinLaunchFn = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
using PermuteGetWorkspaceSizeFn =
    aclnnStatus (*)(const aclTensor *, const aclIntArray *, aclTensor *, uint64_t *, aclOpExecutor **);
using PermuteLaunchFn = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
using ChannelShuffleGetWorkspaceSizeFn =
    aclnnStatus (*)(const aclTensor *, int64_t, aclTensor *, uint64_t *, aclOpExecutor **);
using ChannelShuffleLaunchFn = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
using ProdGetWorkspaceSizeFn =
    aclnnStatus (*)(const aclTensor *, aclDataType, aclTensor *, uint64_t *, aclOpExecutor **);
using ProdLaunchFn = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
using ProdDimGetWorkspaceSizeFn =
    aclnnStatus (*)(const aclTensor *, int64_t, bool, aclDataType, aclTensor *, uint64_t *, aclOpExecutor **);
using ProdDimLaunchFn = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
using ReduceSumGetWorkspaceSizeFn = aclnnStatus (*)(
    const aclTensor *, const aclIntArray *, bool, aclDataType, aclTensor *, uint64_t *, aclOpExecutor **);
using ReduceSumLaunchFn = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
using SubGetWorkspaceSizeFn =
    aclnnStatus (*)(const aclTensor *, const aclTensor *, const aclScalar *, aclTensor *, uint64_t *, aclOpExecutor **);
using SubLaunchFn = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
using SubsGetWorkspaceSizeFn =
    aclnnStatus (*)(const aclTensor *, const aclScalar *, const aclScalar *, aclTensor *, uint64_t *, aclOpExecutor **);
using SubsLaunchFn = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
using TrilGetWorkspaceSizeFn =
    aclnnStatus (*)(const aclTensor *, int64_t, aclTensor *, uint64_t *, aclOpExecutor **);
using TrilLaunchFn = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
using TriuGetWorkspaceSizeFn =
    aclnnStatus (*)(const aclTensor *, int64_t, aclTensor *, uint64_t *, aclOpExecutor **);
using TriuLaunchFn = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
using StdGetWorkspaceSizeFn =
    aclnnStatus (*)(const aclTensor *, const aclIntArray *, int64_t, bool, aclTensor *, uint64_t *, aclOpExecutor **);
using StdLaunchFn = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
using VarGetWorkspaceSizeFn =
    aclnnStatus (*)(const aclTensor *, const aclIntArray *, bool, bool, aclTensor *, uint64_t *, aclOpExecutor **);
using VarLaunchFn = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
using VarCorrectionGetWorkspaceSizeFn =
    aclnnStatus (*)(const aclTensor *, const aclIntArray *, int64_t, bool, aclTensor *, uint64_t *, aclOpExecutor **);
using VarCorrectionLaunchFn = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
using VarMeanGetWorkspaceSizeFn = aclnnStatus (*)(
    const aclTensor *, const aclIntArray *, int64_t, bool, aclTensor *, aclTensor *, uint64_t *, aclOpExecutor **);
using VarMeanLaunchFn = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
using GlobalPoolGetWorkspaceSizeFn = aclnnStatus (*)(const aclTensor *, aclTensor *, uint64_t *, aclOpExecutor **);
using GlobalPoolLaunchFn = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
using BoolUnaryGetWorkspaceSizeFn = aclnnStatus (*)(const aclTensor *, aclTensor *, uint64_t *, aclOpExecutor **);
using BoolUnaryLaunchFn = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
using SliceGetWorkspaceSizeFn =
    aclnnStatus (*)(const aclTensor *, int64_t, int64_t, int64_t, int64_t, aclTensor *, uint64_t *, aclOpExecutor **);
using SliceLaunchFn = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
using StridedSliceGetWorkspaceSizeFn = aclnnStatus (*)(
    const aclTensor *, const aclIntArray *, const aclIntArray *, const aclIntArray *, int64_t, int64_t, int64_t,
    int64_t, int64_t, aclTensor *, uint64_t *, aclOpExecutor **);
using StridedSliceLaunchFn = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
using UnfoldGradGetWorkspaceSizeFn = aclnnStatus (*)(
    const aclTensor *, const aclIntArray *, int64_t, int64_t, int64_t, const aclTensor *, uint64_t *,
    aclOpExecutor **);
using UnfoldGradLaunchFn = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
using LightningIndexerGetWorkspaceSizeFn = aclnnStatus (*)(
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    char *, char *, int64_t, int64_t, int64_t, int64_t, bool, const aclTensor *, const aclTensor *, uint64_t *,
    aclOpExecutor **);
using LightningIndexerLaunchFn = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
using RecurrentGatedDeltaRuleGetWorkspaceSizeFn = aclnnStatus (*)(
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, aclTensor *, const aclTensor *,
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, float, aclTensor *, uint64_t *,
    aclOpExecutor **);
using RecurrentGatedDeltaRuleLaunchFn = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);

void LoadOpApi(const std::string &path)
{
    if (path.empty()) {
        throw std::runtime_error("--opapi-lib is required");
    }
    g_opapiHandle = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (g_opapiHandle == nullptr) {
        throw std::runtime_error(std::string("failed to load op_api library: ") + dlerror());
    }
}

template <typename T>
T ResolveSymbol(const char *name)
{
    dlerror();
    void *symbol = dlsym(g_opapiHandle, name);
    const char *error = dlerror();
    if (error != nullptr || symbol == nullptr) {
        throw std::runtime_error(std::string("missing op_api symbol: ") + name);
    }
    return reinterpret_cast<T>(symbol);
}

int64_t Numel(const std::vector<int64_t> &shape)
{
    if (shape.empty()) {
        return 1;
    }
    return std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>());
}

std::vector<int64_t> Strides(const std::vector<int64_t> &shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = strides[static_cast<size_t>(i + 1)] * shape[static_cast<size_t>(i + 1)];
    }
    return strides;
}

aclDataType DtypeFromName(const std::string &dtype)
{
    if (dtype == "fp32") {
        return aclDataType::ACL_FLOAT;
    }
    if (dtype == "fp64") {
        return aclDataType::ACL_DOUBLE;
    }
    if (dtype == "fp16") {
        return aclDataType::ACL_FLOAT16;
    }
    if (dtype == "bf16") {
        return aclDataType::ACL_BF16;
    }
    if (dtype == "int32") {
        return aclDataType::ACL_INT32;
    }
    if (dtype == "int64") {
        return aclDataType::ACL_INT64;
    }
    if (dtype == "int16") {
        return aclDataType::ACL_INT16;
    }
    if (dtype == "int8") {
        return aclDataType::ACL_INT8;
    }
    if (dtype == "uint8") {
        return aclDataType::ACL_UINT8;
    }
    if (dtype == "bool") {
        return aclDataType::ACL_BOOL;
    }
    throw std::runtime_error("unsupported dtype: " + dtype);
}

std::vector<int64_t> BroadcastShape(const std::vector<int64_t> &a, const std::vector<int64_t> &b)
{
    const size_t rank = std::max(a.size(), b.size());
    std::vector<int64_t> out(rank, 1);
    for (size_t i = 0; i < rank; ++i) {
        const int64_t ad = (i < rank - a.size()) ? 1 : a[i - (rank - a.size())];
        const int64_t bd = (i < rank - b.size()) ? 1 : b[i - (rank - b.size())];
        if (ad != bd && ad != 1 && bd != 1) {
            throw std::runtime_error("input shapes are not broadcastable");
        }
        out[i] = std::max(ad, bd);
    }
    return out;
}

int64_t NormalizeCatDim(const std::vector<std::vector<int64_t>> &shapes, int64_t dim)
{
    if (shapes.empty()) {
        throw std::runtime_error("Cat expects a non-empty TensorList input");
    }
    const int64_t rank = static_cast<int64_t>(shapes[0].size());
    if (rank == 0) {
        throw std::runtime_error("Cat scalar TensorList is not supported by native runner");
    }
    if (dim < -rank || dim >= rank) {
        throw std::runtime_error("Cat dim is out of range");
    }
    return dim < 0 ? dim + rank : dim;
}

int64_t NormalizeStackDim(const std::vector<std::vector<int64_t>> &shapes, int64_t dim)
{
    if (shapes.empty()) {
        throw std::runtime_error("Stack expects a non-empty TensorList input");
    }
    const int64_t rank = static_cast<int64_t>(shapes[0].size());
    if (dim < -(rank + 1) || dim > rank) {
        throw std::runtime_error("Stack dim is out of range");
    }
    return dim < 0 ? dim + rank + 1 : dim;
}

std::vector<int64_t> CatOutputShape(const std::vector<std::vector<int64_t>> &shapes, int64_t axis)
{
    const size_t rank = shapes[0].size();
    std::vector<int64_t> outShape = shapes[0];
    outShape[static_cast<size_t>(axis)] = 0;
    for (const auto &shape : shapes) {
        if (shape.size() != rank) {
            throw std::runtime_error("Cat TensorList ranks must match");
        }
        for (size_t index = 0; index < rank; ++index) {
            if (static_cast<int64_t>(index) != axis && shape[index] != shapes[0][index]) {
                throw std::runtime_error("Cat TensorList shapes must match except concat dim");
            }
        }
        outShape[static_cast<size_t>(axis)] += shape[static_cast<size_t>(axis)];
    }
    return outShape;
}

std::vector<int64_t> StackOutputShape(const std::vector<std::vector<int64_t>> &shapes, int64_t axis)
{
    for (const auto &shape : shapes) {
        if (shape != shapes[0]) {
            throw std::runtime_error("Stack TensorList shapes must match");
        }
    }
    std::vector<int64_t> outShape = shapes[0];
    outShape.insert(outShape.begin() + axis, static_cast<int64_t>(shapes.size()));
    return outShape;
}

std::vector<int64_t> ChunkCatOutputShape(const std::vector<std::vector<int64_t>> &shapes, int64_t dim, int64_t numChunks)
{
    if (shapes.empty()) {
        throw std::runtime_error("ChunkCat expects a non-empty TensorList input");
    }
    if (dim != 0) {
        throw std::runtime_error("ChunkCat dim only supports 0 on ascend910b native runner");
    }
    if (numChunks <= 0) {
        throw std::runtime_error("ChunkCat num_chunks must be positive");
    }

    int64_t dim1 = 0;
    for (const auto &shape : shapes) {
        if (shape.empty() || shape.size() > 8) {
            throw std::runtime_error("ChunkCat TensorList rank must be between 1 and 8");
        }
        for (const auto dimValue : shape) {
            if (dimValue <= 0) {
                throw std::runtime_error("zero-sized tensors are not supported by native runner");
            }
        }
        const int64_t chunkRows = (shape[0] + numChunks - 1) / numChunks;
        std::vector<int64_t> trailingShape(shape.begin() + 1, shape.end());
        dim1 += chunkRows * Numel(trailingShape);
    }
    return {numChunks, dim1};
}

int64_t NormalizeSplitWithSizeDim(const std::vector<int64_t> &shape, int64_t dim)
{
    const int64_t rank = static_cast<int64_t>(shape.size());
    if (rank == 0) {
        throw std::runtime_error("SplitWithSize rank must be at least 1");
    }
    if (dim < -rank || dim >= rank) {
        throw std::runtime_error("SplitWithSize dim is out of range");
    }
    return dim < 0 ? dim + rank : dim;
}

std::vector<std::vector<int64_t>> SplitWithSizeOutputShapes(
    const std::vector<int64_t> &shape, const std::vector<int64_t> &splitSizes, int64_t axis)
{
    if (splitSizes.empty()) {
        throw std::runtime_error("SplitWithSize split_sizes must be non-empty");
    }
    int64_t total = 0;
    std::vector<std::vector<int64_t>> outputShapes;
    outputShapes.reserve(splitSizes.size());
    for (const auto splitSize : splitSizes) {
        if (splitSize < 0) {
            throw std::runtime_error("SplitWithSize split_sizes must be non-negative");
        }
        total += splitSize;
        auto outShape = shape;
        outShape[static_cast<size_t>(axis)] = splitSize;
        outputShapes.push_back(outShape);
    }
    if (total != shape[static_cast<size_t>(axis)]) {
        throw std::runtime_error("SplitWithSize split_sizes sum must match split dimension");
    }
    return outputShapes;
}

bool ContainsAxis(const std::vector<int64_t> &axes, int64_t axis)
{
    return std::find(axes.begin(), axes.end(), axis) != axes.end();
}

std::vector<int64_t> NormalizeReduceDims(
    const std::string &api, const std::vector<int64_t> &shape, const std::vector<int64_t> &dims, bool noopWithEmptyAxes)
{
    const int64_t rank = static_cast<int64_t>(shape.size());
    std::vector<int64_t> rawDims = dims;
    if (rawDims.empty() && !noopWithEmptyAxes) {
        for (int64_t axis = 0; axis < rank; ++axis) {
            rawDims.push_back(axis);
        }
    }

    std::vector<int64_t> normalized;
    for (const auto dim : rawDims) {
        if (rank == 0) {
            throw std::runtime_error(api + " dims are invalid for scalar input");
        }
        if (dim < -rank || dim >= rank) {
            throw std::runtime_error(api + " dims are out of range");
        }
        const int64_t axis = dim < 0 ? dim + rank : dim;
        if (ContainsAxis(normalized, axis)) {
            throw std::runtime_error(api + " duplicate dims are not supported");
        }
        normalized.push_back(axis);
    }
    std::sort(normalized.begin(), normalized.end());
    return normalized;
}

std::vector<int64_t> NormalizeReduceDimsUnique(
    const std::string &api, const std::vector<int64_t> &shape, const std::vector<int64_t> &dims, bool noopWithEmptyAxes)
{
    const int64_t rank = static_cast<int64_t>(shape.size());
    std::vector<int64_t> rawDims = dims;
    if (rawDims.empty() && !noopWithEmptyAxes) {
        for (int64_t axis = 0; axis < rank; ++axis) {
            rawDims.push_back(axis);
        }
    }

    std::vector<int64_t> normalized;
    for (const auto dim : rawDims) {
        if (rank == 0) {
            throw std::runtime_error(api + " dims are invalid for scalar input");
        }
        if (dim < -rank || dim >= rank) {
            throw std::runtime_error(api + " dims are out of range");
        }
        normalized.push_back(dim < 0 ? dim + rank : dim);
    }
    std::sort(normalized.begin(), normalized.end());
    normalized.erase(std::unique(normalized.begin(), normalized.end()), normalized.end());
    return normalized;
}

int64_t NormalizeSingleDim(const std::string &api, const std::vector<int64_t> &shape, const std::vector<int64_t> &dims)
{
    if (dims.size() != 1) {
        throw std::runtime_error(api + " expects exactly 1 dim");
    }
    const int64_t rank = static_cast<int64_t>(shape.size());
    if (rank == 0) {
        throw std::runtime_error(api + " dim is invalid for scalar input");
    }
    const int64_t dim = dims[0];
    if (dim < -rank || dim >= rank) {
        throw std::runtime_error(api + " dim is out of range");
    }
    return dim < 0 ? dim + rank : dim;
}

std::vector<int64_t> ReduceOutputShape(
    const std::vector<int64_t> &shape, const std::vector<int64_t> &axes, bool keepDims)
{
    if (axes.empty()) {
        return shape;
    }
    std::vector<int64_t> outShape;
    for (int64_t axis = 0; axis < static_cast<int64_t>(shape.size()); ++axis) {
        if (ContainsAxis(axes, axis)) {
            if (keepDims) {
                outShape.push_back(1);
            }
            continue;
        }
        outShape.push_back(shape[static_cast<size_t>(axis)]);
    }
    return outShape;
}

std::vector<int64_t> GlobalPoolOutputShape(const std::vector<int64_t> &shape)
{
    if (shape.size() < 4 || shape.size() > 8) {
        throw std::runtime_error("GlobalPool rank must be in [4, 8]");
    }
    std::vector<int64_t> outShape = shape;
    for (size_t axis = 2; axis < outShape.size(); ++axis) {
        outShape[axis] = 1;
    }
    return outShape;
}

bool MaskBit(int64_t mask, int64_t bit)
{
    if (bit < 0 || bit >= 63) {
        return false;
    }
    return (mask & (int64_t{1} << bit)) != 0;
}

std::vector<int64_t> SliceOutputShape(
    const std::vector<int64_t> &shape, int64_t dim, int64_t start, int64_t end, int64_t step)
{
    const int64_t rank = static_cast<int64_t>(shape.size());
    if (rank == 0) {
        throw std::runtime_error("Slice dim is out of range");
    }
    if (step <= 0) {
        throw std::runtime_error("Slice step attr is invalid");
    }
    if (dim < -rank || dim >= rank) {
        throw std::runtime_error("Slice dim is out of range");
    }
    const int64_t axis = dim < 0 ? dim + rank : dim;
    const int64_t axisSize = shape[static_cast<size_t>(axis)];
    start = std::max<int64_t>(0, std::min(start, axisSize));
    end = std::max<int64_t>(0, std::min(end, axisSize));
    const int64_t length = end <= start ? 0 : (end - start + step - 1) / step;
    std::vector<int64_t> outShape = shape;
    outShape[static_cast<size_t>(axis)] = length;
    return outShape;
}

std::vector<int64_t> BuildSliceAxisIndices(int64_t axisSize, int64_t start, int64_t end, int64_t step)
{
    start = std::max<int64_t>(0, std::min(start, axisSize));
    end = std::max<int64_t>(0, std::min(end, axisSize));
    std::vector<int64_t> indices;
    for (int64_t index = start; index < end; index += step) {
        indices.push_back(index);
    }
    return indices;
}

struct StridedSlicePlan {
    std::vector<int64_t> outShape;
    std::vector<std::vector<int64_t>> axisIndices;
    std::vector<int64_t> inputAxisToOutputAxis;
};

int64_t NormalizeStridedSliceIndex(int64_t value, int64_t dim, int64_t stride)
{
    if (stride > 0) {
        if (value < 0) {
            value += dim;
        }
        return std::max<int64_t>(0, std::min(value, dim));
    }

    if (value < 0) {
        value += dim;
    }
    return std::max<int64_t>(-1, std::min(value, dim - 1));
}

std::vector<int64_t> BuildStridedAxisIndices(
    int64_t dim, int64_t rawBegin, int64_t rawEnd, int64_t stride, bool beginMask, bool endMask, bool shrinkAxis)
{
    if (stride == 0) {
        throw std::runtime_error("StridedSlice strides attr is invalid");
    }
    if (shrinkAxis) {
        if (dim <= 0) {
            throw std::runtime_error("StridedSlice shrink axis is invalid for empty dimension");
        }
        int64_t index = rawBegin;
        if (index < 0) {
            index += dim;
        }
        index = std::max<int64_t>(0, std::min(index, dim - 1));
        return {index};
    }

    int64_t begin = rawBegin;
    int64_t end = rawEnd;
    if (beginMask) {
        begin = stride > 0 ? 0 : dim - 1;
    } else {
        begin = NormalizeStridedSliceIndex(begin, dim, stride);
    }
    if (endMask) {
        end = stride > 0 ? dim : -1;
    } else {
        end = NormalizeStridedSliceIndex(end, dim, stride);
    }

    std::vector<int64_t> indices;
    if (stride > 0) {
        for (int64_t index = begin; index < end; index += stride) {
            if (index >= 0 && index < dim) {
                indices.push_back(index);
            }
        }
    } else {
        for (int64_t index = begin; index > end; index += stride) {
            if (index >= 0 && index < dim) {
                indices.push_back(index);
            }
        }
    }
    return indices;
}

int64_t CountSpecifiedInputAxesAfter(
    int64_t specPos, int64_t specLen, int64_t newAxisMask, int64_t ellipsisMask)
{
    int64_t count = 0;
    for (int64_t pos = specPos + 1; pos < specLen; ++pos) {
        if (MaskBit(newAxisMask, pos) || MaskBit(ellipsisMask, pos)) {
            continue;
        }
        ++count;
    }
    return count;
}

StridedSlicePlan BuildStridedSlicePlan(
    const std::vector<int64_t> &shape, const std::vector<int64_t> &begin, const std::vector<int64_t> &ends,
    const std::vector<int64_t> &strides, int64_t beginMask, int64_t endMask, int64_t ellipsisMask,
    int64_t newAxisMask, int64_t shrinkAxisMask)
{
    if (begin.size() != ends.size() || begin.size() != strides.size()) {
        throw std::runtime_error("StridedSlice begin/end/strides attrs are invalid");
    }

    StridedSlicePlan plan;
    plan.axisIndices.resize(shape.size());
    plan.inputAxisToOutputAxis.assign(shape.size(), -1);

    const int64_t rank = static_cast<int64_t>(shape.size());
    const int64_t specLen = static_cast<int64_t>(begin.size());
    int64_t inputAxis = 0;
    bool ellipsisSeen = false;

    for (int64_t pos = 0; inputAxis < rank || pos < specLen; ++pos) {
        if (MaskBit(newAxisMask, pos)) {
            plan.outShape.push_back(1);
            continue;
        }

        if (MaskBit(ellipsisMask, pos) && !ellipsisSeen) {
            ellipsisSeen = true;
            const int64_t specifiedAfter = CountSpecifiedInputAxesAfter(pos, specLen, newAxisMask, ellipsisMask);
            const int64_t ellipsisAxes = std::max<int64_t>(0, rank - inputAxis - specifiedAfter);
            for (int64_t i = 0; i < ellipsisAxes && inputAxis < rank; ++i, ++inputAxis) {
                const int64_t dim = shape[static_cast<size_t>(inputAxis)];
                auto indices = BuildStridedAxisIndices(dim, 0, dim, 1, true, true, false);
                plan.axisIndices[static_cast<size_t>(inputAxis)] = std::move(indices);
                plan.inputAxisToOutputAxis[static_cast<size_t>(inputAxis)] =
                    static_cast<int64_t>(plan.outShape.size());
                plan.outShape.push_back(dim);
            }
            continue;
        }

        if (inputAxis >= rank) {
            if (pos >= specLen) {
                break;
            }
            continue;
        }

        const bool hasSpec = pos < specLen;
        const int64_t stride = hasSpec ? strides[static_cast<size_t>(pos)] : 1;
        const int64_t rawBegin = hasSpec ? begin[static_cast<size_t>(pos)] : 0;
        const int64_t rawEnd = hasSpec ? ends[static_cast<size_t>(pos)] : shape[static_cast<size_t>(inputAxis)];
        const bool shrinkAxis = MaskBit(shrinkAxisMask, pos);
        auto indices = BuildStridedAxisIndices(
            shape[static_cast<size_t>(inputAxis)], rawBegin, rawEnd, stride, MaskBit(beginMask, pos),
            MaskBit(endMask, pos), shrinkAxis);
        plan.axisIndices[static_cast<size_t>(inputAxis)] = std::move(indices);
        if (!shrinkAxis) {
            plan.inputAxisToOutputAxis[static_cast<size_t>(inputAxis)] =
                static_cast<int64_t>(plan.outShape.size());
            plan.outShape.push_back(static_cast<int64_t>(plan.axisIndices[static_cast<size_t>(inputAxis)].size()));
        }
        ++inputAxis;
    }
    return plan;
}

size_t LinearOffset(const std::vector<int64_t> &shape, const std::vector<int64_t> &coords)
{
    if (shape.empty()) {
        return 0;
    }
    const auto strides = Strides(shape);
    size_t offset = 0;
    for (size_t axis = 0; axis < shape.size(); ++axis) {
        offset += static_cast<size_t>(coords[axis] * strides[axis]);
    }
    return offset;
}

std::vector<int64_t> CoordinatesFromFlat(const std::vector<int64_t> &shape, size_t flat)
{
    std::vector<int64_t> coords(shape.size(), 0);
    for (size_t axisRev = 0; axisRev < shape.size(); ++axisRev) {
        const size_t axis = shape.size() - 1 - axisRev;
        if (shape[axis] <= 0) {
            coords[axis] = 0;
            continue;
        }
        coords[axis] = static_cast<int64_t>(flat % static_cast<size_t>(shape[axis]));
        flat /= static_cast<size_t>(shape[axis]);
    }
    return coords;
}

std::vector<int64_t> NormalizePermuteDims(const std::vector<int64_t> &shape, const std::vector<int64_t> &dims)
{
    const int64_t rank = static_cast<int64_t>(shape.size());
    if (static_cast<int64_t>(dims.size()) != rank) {
        throw std::runtime_error("Permute dims must be a rank-length permutation");
    }
    std::vector<int64_t> normalized;
    std::vector<int64_t> seen(static_cast<size_t>(rank), 0);
    normalized.reserve(dims.size());
    for (const auto dim : dims) {
        if (dim < -rank || dim >= rank) {
            throw std::runtime_error("Permute dims must be a rank-length permutation");
        }
        const int64_t axis = dim < 0 ? dim + rank : dim;
        if (seen[static_cast<size_t>(axis)] != 0) {
            throw std::runtime_error("Permute dims must be a rank-length permutation");
        }
        seen[static_cast<size_t>(axis)] = 1;
        normalized.push_back(axis);
    }
    return normalized;
}

std::vector<int64_t> PermuteOutputShape(const std::vector<int64_t> &shape, const std::vector<int64_t> &dims)
{
    std::vector<int64_t> outShape;
    outShape.reserve(dims.size());
    for (const auto axis : dims) {
        outShape.push_back(shape[static_cast<size_t>(axis)]);
    }
    return outShape;
}

size_t BroadcastOffset(const std::vector<int64_t> &shape, const std::vector<int64_t> &outShape, size_t flat)
{
    if (shape.empty()) {
        return 0;
    }
    const auto strides = Strides(shape);
    const size_t rank = outShape.size();
    const size_t inRank = shape.size();
    size_t offset = 0;
    for (size_t axisRev = 0; axisRev < rank; ++axisRev) {
        const size_t outAxis = rank - 1 - axisRev;
        const int64_t coord = flat % static_cast<size_t>(outShape[outAxis]);
        flat /= static_cast<size_t>(outShape[outAxis]);
        if (axisRev >= inRank) {
            continue;
        }
        const size_t inAxis = inRank - 1 - axisRev;
        const int64_t inCoord = (shape[inAxis] == 1) ? 0 : coord;
        offset += static_cast<size_t>(inCoord * strides[inAxis]);
    }
    return offset;
}

uint32_t FloatBits(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

float BitsFloat(uint32_t bits)
{
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

uint16_t FloatToFp16Bits(float value)
{
    const uint32_t bits = FloatBits(value);
    const uint32_t sign = (bits >> 16) & 0x8000U;
    const uint32_t floatExp = (bits >> 23) & 0xffU;
    int32_t exp = static_cast<int32_t>((bits >> 23) & 0xffU) - 127 + 15;
    uint32_t mant = bits & 0x7fffffU;

    if (exp <= 0) {
        if (exp < -10) {
            return static_cast<uint16_t>(sign);
        }
        mant |= 0x800000U;
        const uint32_t shift = static_cast<uint32_t>(14 - exp);
        uint32_t rounded = mant >> shift;
        const uint32_t roundBit = (mant >> (shift - 1)) & 1U;
        rounded += roundBit;
        return static_cast<uint16_t>(sign | rounded);
    }
    if (exp >= 31) {
        if (floatExp == 0xffU && mant != 0) {
            return static_cast<uint16_t>(sign | 0x7e00U);
        }
        return static_cast<uint16_t>(sign | 0x7c00U);
    }

    mant += 0x1000U;
    if (mant & 0x800000U) {
        mant = 0;
        ++exp;
        if (exp >= 31) {
            return static_cast<uint16_t>(sign | 0x7c00U);
        }
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
}

float Fp16BitsToFloat(uint16_t value)
{
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000U) << 16;
    uint32_t exp = (value >> 10) & 0x1fU;
    uint32_t mant = value & 0x03ffU;

    if (exp == 0) {
        if (mant == 0) {
            return BitsFloat(sign);
        }
        exp = 1;
        while ((mant & 0x0400U) == 0) {
            mant <<= 1;
            --exp;
        }
        mant &= 0x03ffU;
    } else if (exp == 31) {
        return BitsFloat(sign | 0x7f800000U | (mant << 13));
    }

    const uint32_t exp32 = exp + (127 - 15);
    return BitsFloat(sign | (exp32 << 23) | (mant << 13));
}

uint16_t FloatToBf16Bits(float value)
{
    const uint32_t bits = FloatBits(value);
    const uint32_t lsb = (bits >> 16) & 1U;
    return static_cast<uint16_t>((bits + 0x7fffU + lsb) >> 16);
}

float Bf16BitsToFloat(uint16_t value)
{
    return BitsFloat(static_cast<uint32_t>(value) << 16);
}

template <typename T>
double HalfTolerance()
{
    if constexpr (IsHalfStorage<T>::value) {
        if constexpr (T::kind == HalfKind::Bf16) {
            return 5e-2;
        }
        return 1e-2;
    }
    return 1e-4;
}

template <typename T>
T CastValue(double value, bool boolType)
{
    if (boolType) {
        if constexpr (IsHalfStorage<T>::value) {
            return CastValue<T>(std::fabs(value) > 0.5 ? 1.0 : 0.0, false);
        } else {
            return static_cast<T>(std::fabs(value) > 0.5);
        }
    }
    if constexpr (IsHalfStorage<T>::value) {
        const float typed = static_cast<float>(value);
        T out;
        if constexpr (T::kind == HalfKind::Bf16) {
            out.bits = FloatToBf16Bits(typed);
        } else {
            out.bits = FloatToFp16Bits(typed);
        }
        return out;
    } else if constexpr (std::is_integral<T>::value) {
        if constexpr (std::is_unsigned<T>::value) {
            value = std::max(0.0, value);
        }
        return static_cast<T>(std::llround(value));
    } else {
        return static_cast<T>(value);
    }
}

template <typename T>
T CastExpectedValue(double value, bool boolType)
{
    if (boolType) {
        return CastValue<T>(value, true);
    }
    if constexpr (std::is_integral<T>::value) {
        return static_cast<T>(std::llround(value));
    } else {
        return CastValue<T>(value, false);
    }
}

template <typename T>
std::vector<T> MakeData(const std::vector<int64_t> &shape, double low, double high, bool boolType)
{
    const int64_t count = Numel(shape);
    std::vector<T> data(static_cast<size_t>(count));
    if (count <= 0) {
        return data;
    }
    if constexpr (std::is_unsigned<T>::value) {
        low = std::max(0.0, low);
        high = std::max(low, high);
    }
    for (int64_t i = 0; i < count; ++i) {
        const double ratio = static_cast<double>(i % 17) / 16.0;
        data[static_cast<size_t>(i)] = CastValue<T>(low + (high - low) * ratio, boolType);
    }
    return data;
}

template <typename T>
double ToDouble(T value, bool boolType)
{
    if (boolType) {
        if constexpr (IsHalfStorage<T>::value) {
            return ToDouble(CastValue<T>(std::fabs(ToDouble(value, false)) > 0.5 ? 1.0 : 0.0, false), false);
        } else {
            return value == 0 ? 0.0 : 1.0;
        }
    }
    if constexpr (IsHalfStorage<T>::value) {
        if constexpr (T::kind == HalfKind::Bf16) {
            return static_cast<double>(Bf16BitsToFloat(value.bits));
        }
        return static_cast<double>(Fp16BitsToFloat(value.bits));
    } else {
        return static_cast<double>(value);
    }
}

bool SameSpecialValue(double actual, double expected)
{
    if (std::isnan(actual) || std::isnan(expected)) {
        return std::isnan(actual) && std::isnan(expected);
    }
    if (std::isinf(actual) || std::isinf(expected)) {
        return std::isinf(actual) && std::isinf(expected) && std::signbit(actual) == std::signbit(expected);
    }
    return false;
}

template <typename T>
bool NearlyEqual(T actual, double expected, bool boolType)
{
    if (boolType) {
        if constexpr (IsHalfStorage<T>::value) {
            return (std::fabs(ToDouble(actual, false)) > 0.5) == (std::fabs(expected) > 0.5);
        } else {
            return (actual != 0) == (std::fabs(expected) > 0.5);
        }
    }
    if constexpr (IsHalfStorage<T>::value) {
        const double roundedExpected = ToDouble(CastValue<T>(expected, false), false);
        const double actualValue = ToDouble(actual, false);
        if (!std::isfinite(actualValue) || !std::isfinite(roundedExpected)) {
            return SameSpecialValue(actualValue, roundedExpected);
        }
        return std::fabs(actualValue - roundedExpected) <=
               HalfTolerance<T>() * std::max(1.0, std::fabs(roundedExpected));
    } else if constexpr (std::is_floating_point<T>::value) {
        const double roundedExpected = ToDouble(CastExpectedValue<T>(expected, false), false);
        const double actualValue = static_cast<double>(actual);
        if (!std::isfinite(actualValue) || !std::isfinite(roundedExpected)) {
            return SameSpecialValue(actualValue, roundedExpected);
        }
        return std::fabs(actualValue - roundedExpected) <= 1e-4 * std::max(1.0, std::fabs(roundedExpected));
    } else {
        return actual == CastExpectedValue<T>(expected, false);
    }
}

int InitAcl(int32_t device, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        std::cerr << "aclInit failed: " << ret << "\n";
        return ret;
    }
    ret = aclrtSetDevice(device);
    if (ret != ACL_SUCCESS) {
        std::cerr << "aclrtSetDevice failed: " << ret << "\n";
        return ret;
    }
    ret = aclrtCreateStream(stream);
    if (ret != ACL_SUCCESS) {
        std::cerr << "aclrtCreateStream failed: " << ret << "\n";
        return ret;
    }
    return ACL_SUCCESS;
}

template <typename T>
int CreateTensor(const std::vector<T> &host, const std::vector<int64_t> &shape, aclDataType dtype, TensorHolder *holder)
{
    const size_t bytes = std::max<size_t>(host.size() * sizeof(T), sizeof(T));
    auto ret = aclrtMalloc(&holder->deviceAddr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        std::cerr << "aclrtMalloc failed: " << ret << "\n";
        return ret;
    }
    if (!host.empty()) {
        ret = aclrtMemcpy(holder->deviceAddr, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS) {
            std::cerr << "aclrtMemcpy host to device failed: " << ret << "\n";
            return ret;
        }
    }
    auto strides = Strides(shape);
    holder->tensor = aclCreateTensor(
        shape.data(), shape.size(), dtype, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        holder->deviceAddr);
    if (holder->tensor == nullptr) {
        std::cerr << "aclCreateTensor failed\n";
        return 1;
    }
    return ACL_SUCCESS;
}

template <typename T>
int CreateGeneratedTensorTyped(
    const std::vector<int64_t> &shape, double low, double high, aclDataType dtype, bool boolType,
    TensorHolder *holder, std::vector<double> *values)
{
    auto host = MakeData<T>(shape, low, high, boolType);
    values->clear();
    values->reserve(host.size());
    for (const auto &value : host) {
        values->push_back(ToDouble(value, boolType));
    }
    return CreateTensor(host, shape, dtype, holder);
}

int CreateGeneratedTensorByDtype(
    const std::string &dtypeName, const std::vector<int64_t> &shape, double low, double high,
    TensorHolder *holder, std::vector<double> *values)
{
    const auto dtype = DtypeFromName(dtypeName);
    if (dtypeName == "fp32") {
        return CreateGeneratedTensorTyped<float>(shape, low, high, dtype, false, holder, values);
    }
    if (dtypeName == "fp64") {
        return CreateGeneratedTensorTyped<double>(shape, low, high, dtype, false, holder, values);
    }
    if (dtypeName == "fp16") {
        return CreateGeneratedTensorTyped<HalfStorage<HalfKind::Fp16>>(shape, low, high, dtype, false, holder, values);
    }
    if (dtypeName == "bf16") {
        return CreateGeneratedTensorTyped<HalfStorage<HalfKind::Bf16>>(shape, low, high, dtype, false, holder, values);
    }
    if (dtypeName == "int32") {
        return CreateGeneratedTensorTyped<int32_t>(shape, low, high, dtype, false, holder, values);
    }
    if (dtypeName == "int64") {
        return CreateGeneratedTensorTyped<int64_t>(shape, low, high, dtype, false, holder, values);
    }
    if (dtypeName == "int16") {
        return CreateGeneratedTensorTyped<int16_t>(shape, low, high, dtype, false, holder, values);
    }
    if (dtypeName == "int8") {
        return CreateGeneratedTensorTyped<int8_t>(shape, low, high, dtype, false, holder, values);
    }
    if (dtypeName == "uint8") {
        return CreateGeneratedTensorTyped<uint8_t>(shape, low, high, dtype, false, holder, values);
    }
    if (dtypeName == "bool") {
        return CreateGeneratedTensorTyped<uint8_t>(shape, low, high, dtype, true, holder, values);
    }
    throw std::runtime_error("unsupported generated tensor dtype: " + dtypeName);
}

template <typename T>
aclScalar *CreateScalar(double value, aclDataType dtype, bool boolType)
{
    if (boolType) {
        bool typed = std::fabs(value) > 0.5;
        return aclCreateScalar(&typed, dtype);
    }
    T typed = CastValue<T>(value, false);
    if constexpr (IsHalfStorage<T>::value) {
        return aclCreateScalar(&typed.bits, dtype);
    }
    return aclCreateScalar(&typed, dtype);
}

template <typename T>
double ScalarGoldenValue(double value, bool boolType)
{
    return ToDouble(CastValue<T>(value, boolType), boolType);
}

template <typename T, typename Fn>
std::vector<double> GoldenUnary(const std::vector<T> &a, bool boolType, Fn fn)
{
    std::vector<double> golden(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        golden[i] = fn(ToDouble(a[i], boolType));
    }
    return golden;
}

template <typename T>
std::vector<double> GoldenAbs(const std::vector<T> &a, bool boolType)
{
    std::vector<double> golden(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        golden[i] = std::fabs(ToDouble(a[i], boolType));
    }
    return golden;
}

template <typename T>
std::vector<double> GoldenAdd(
    const std::vector<T> &a, const std::vector<int64_t> &shapeA, const std::vector<T> &b,
    const std::vector<int64_t> &shapeB, const std::vector<int64_t> &outShape, double alpha, bool boolType)
{
    std::vector<double> golden(static_cast<size_t>(Numel(outShape)));
    for (size_t i = 0; i < golden.size(); ++i) {
        golden[i] =
            ToDouble(a[BroadcastOffset(shapeA, outShape, i)], boolType) +
            alpha * ToDouble(b[BroadcastOffset(shapeB, outShape, i)], boolType);
    }
    return golden;
}

template <typename T>
std::vector<double> GoldenMul(
    const std::vector<T> &a, const std::vector<int64_t> &shapeA, const std::vector<T> &b,
    const std::vector<int64_t> &shapeB, const std::vector<int64_t> &outShape, bool boolType)
{
    std::vector<double> golden(static_cast<size_t>(Numel(outShape)));
    for (size_t i = 0; i < golden.size(); ++i) {
        golden[i] =
            ToDouble(a[BroadcastOffset(shapeA, outShape, i)], boolType) *
            ToDouble(b[BroadcastOffset(shapeB, outShape, i)], boolType);
    }
    return golden;
}

template <typename T>
std::vector<double> GoldenMaximum(
    const std::vector<T> &a, const std::vector<int64_t> &shapeA, const std::vector<T> &b,
    const std::vector<int64_t> &shapeB, const std::vector<int64_t> &outShape, bool boolType)
{
    std::vector<double> golden(static_cast<size_t>(Numel(outShape)));
    for (size_t i = 0; i < golden.size(); ++i) {
        const double lhs = ToDouble(a[BroadcastOffset(shapeA, outShape, i)], boolType);
        const double rhs = ToDouble(b[BroadcastOffset(shapeB, outShape, i)], boolType);
        golden[i] = std::max(lhs, rhs);
    }
    return golden;
}

template <typename T>
std::vector<double> GoldenMinimum(
    const std::vector<T> &a, const std::vector<int64_t> &shapeA, const std::vector<T> &b,
    const std::vector<int64_t> &shapeB, const std::vector<int64_t> &outShape, bool boolType)
{
    std::vector<double> golden(static_cast<size_t>(Numel(outShape)));
    for (size_t i = 0; i < golden.size(); ++i) {
        const double lhs = ToDouble(a[BroadcastOffset(shapeA, outShape, i)], boolType);
        const double rhs = ToDouble(b[BroadcastOffset(shapeB, outShape, i)], boolType);
        golden[i] = std::min(lhs, rhs);
    }
    return golden;
}

template <typename T>
std::vector<double> GoldenGer(const std::vector<T> &a, const std::vector<T> &b, bool boolType)
{
    std::vector<double> golden(a.size() * b.size());
    for (size_t row = 0; row < a.size(); ++row) {
        for (size_t col = 0; col < b.size(); ++col) {
            golden[row * b.size() + col] = ToDouble(a[row], boolType) * ToDouble(b[col], boolType);
        }
    }
    return golden;
}

template <typename T>
std::vector<double> GoldenDiv(
    const std::vector<T> &a, const std::vector<int64_t> &shapeA, const std::vector<T> &b,
    const std::vector<int64_t> &shapeB, const std::vector<int64_t> &outShape, bool boolType)
{
    std::vector<double> golden(static_cast<size_t>(Numel(outShape)));
    for (size_t i = 0; i < golden.size(); ++i) {
        golden[i] =
            ToDouble(a[BroadcastOffset(shapeA, outShape, i)], boolType) /
            ToDouble(b[BroadcastOffset(shapeB, outShape, i)], boolType);
    }
    return golden;
}

template <typename T>
std::vector<double> GoldenSub(
    const std::vector<T> &a, const std::vector<int64_t> &shapeA, const std::vector<T> &b,
    const std::vector<int64_t> &shapeB, const std::vector<int64_t> &outShape, double alpha, bool boolType)
{
    std::vector<double> golden(static_cast<size_t>(Numel(outShape)));
    for (size_t i = 0; i < golden.size(); ++i) {
        golden[i] =
            ToDouble(a[BroadcastOffset(shapeA, outShape, i)], boolType) -
            alpha * ToDouble(b[BroadcastOffset(shapeB, outShape, i)], boolType);
    }
    return golden;
}

template <typename T>
std::vector<double> GoldenMuls(const std::vector<T> &a, double other, bool boolType)
{
    std::vector<double> golden(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        golden[i] = ToDouble(a[i], boolType) * other;
    }
    return golden;
}

template <typename T>
std::vector<double> GoldenDivs(const std::vector<T> &a, double other, bool boolType)
{
    std::vector<double> golden(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        golden[i] = ToDouble(a[i], boolType) / other;
    }
    return golden;
}

template <typename T>
std::vector<double> GoldenClip(const std::vector<T> &a, double clipMin, double clipMax, bool hasMin, bool hasMax, bool boolType)
{
    std::vector<double> golden(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        double value = ToDouble(a[i], boolType);
        if (hasMin) {
            value = std::max(value, clipMin);
        }
        if (hasMax) {
            value = std::min(value, clipMax);
        }
        golden[i] = value;
    }
    return golden;
}

template <typename T>
std::vector<double> GoldenTensorClip(
    const std::vector<T> &a, const std::vector<int64_t> &shapeA, const std::vector<double> &clipMin,
    const std::vector<int64_t> &shapeMin, const std::vector<double> &clipMax, const std::vector<int64_t> &shapeMax,
    const std::vector<int64_t> &outShape, bool hasMin, bool hasMax, bool boolType)
{
    std::vector<double> golden(static_cast<size_t>(Numel(outShape)));
    for (size_t i = 0; i < golden.size(); ++i) {
        double value = ToDouble(a[BroadcastOffset(shapeA, outShape, i)], boolType);
        if (hasMin) {
            value = std::max(value, clipMin[BroadcastOffset(shapeMin, outShape, i)]);
        }
        if (hasMax) {
            value = std::min(value, clipMax[BroadcastOffset(shapeMax, outShape, i)]);
        }
        golden[i] = value;
    }
    return golden;
}

template <typename TOut>
double CastGoldenValue(double value, bool boolType)
{
    if (boolType) {
        return value != 0.0 ? 1.0 : 0.0;
    }
    if constexpr (IsHalfStorage<TOut>::value) {
        return ToDouble(CastValue<TOut>(value, false), false);
    } else if constexpr (std::is_integral<TOut>::value) {
        const double truncated = std::trunc(value);
        const double minValue = static_cast<double>(std::numeric_limits<TOut>::min());
        const double maxValue = static_cast<double>(std::numeric_limits<TOut>::max());
        const double bounded = std::max(minValue, std::min(maxValue, truncated));
        return ToDouble(static_cast<TOut>(bounded), false);
    } else {
        return static_cast<double>(static_cast<TOut>(value));
    }
}

template <typename TIn, typename TOut>
std::vector<double> GoldenCast(const std::vector<TIn> &a, bool inputBoolType, bool outputBoolType)
{
    std::vector<double> golden(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        golden[i] = CastGoldenValue<TOut>(ToDouble(a[i], inputBoolType), outputBoolType);
    }
    return golden;
}

template <typename T>
std::vector<double> GoldenSubs(const std::vector<T> &a, double other, double alpha, bool boolType)
{
    std::vector<double> golden(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        golden[i] = ToDouble(a[i], boolType) - alpha * other;
    }
    return golden;
}

template <typename T>
std::vector<double> GoldenPermute(
    const std::vector<T> &a, const std::vector<int64_t> &shape, const std::vector<int64_t> &dims,
    const std::vector<int64_t> &outShape, bool boolType)
{
    std::vector<double> golden(static_cast<size_t>(Numel(outShape)));
    for (size_t flat = 0; flat < golden.size(); ++flat) {
        const auto outCoords = CoordinatesFromFlat(outShape, flat);
        std::vector<int64_t> inCoords(shape.size(), 0);
        for (size_t outAxis = 0; outAxis < dims.size(); ++outAxis) {
            inCoords[static_cast<size_t>(dims[outAxis])] = outCoords[outAxis];
        }
        golden[flat] = ToDouble(a[LinearOffset(shape, inCoords)], boolType);
    }
    return golden;
}

template <typename T>
std::vector<double> GoldenCat(
    const std::vector<std::vector<T>> &inputs, const std::vector<std::vector<int64_t>> &shapes,
    const std::vector<int64_t> &outShape, int64_t axis, bool boolType)
{
    std::vector<double> golden(static_cast<size_t>(Numel(outShape)));
    std::vector<int64_t> offsets;
    offsets.reserve(shapes.size() + 1);
    offsets.push_back(0);
    for (const auto &shape : shapes) {
        offsets.push_back(offsets.back() + shape[static_cast<size_t>(axis)]);
    }

    for (size_t flat = 0; flat < golden.size(); ++flat) {
        auto coords = CoordinatesFromFlat(outShape, flat);
        const int64_t axisCoord = coords[static_cast<size_t>(axis)];
        size_t inputIndex = 0;
        while (inputIndex + 1 < offsets.size() && axisCoord >= offsets[inputIndex + 1]) {
            ++inputIndex;
        }
        coords[static_cast<size_t>(axis)] = axisCoord - offsets[inputIndex];
        golden[flat] =
            ToDouble(inputs[inputIndex][LinearOffset(shapes[inputIndex], coords)], boolType);
    }
    return golden;
}

template <typename T>
std::vector<double> GoldenStack(
    const std::vector<std::vector<T>> &inputs, const std::vector<std::vector<int64_t>> &shapes,
    const std::vector<int64_t> &outShape, int64_t axis, bool boolType)
{
    std::vector<double> golden(static_cast<size_t>(Numel(outShape)));
    for (size_t flat = 0; flat < golden.size(); ++flat) {
        auto outCoords = CoordinatesFromFlat(outShape, flat);
        const size_t inputIndex = static_cast<size_t>(outCoords[static_cast<size_t>(axis)]);
        std::vector<int64_t> inCoords = outCoords;
        inCoords.erase(inCoords.begin() + axis);
        golden[flat] =
            ToDouble(inputs[inputIndex][LinearOffset(shapes[inputIndex], inCoords)], boolType);
    }
    return golden;
}

template <typename T>
std::vector<double> GoldenSplitWithSize(
    const std::vector<T> &input, const std::vector<int64_t> &shape,
    const std::vector<std::vector<int64_t>> &outputShapes, int64_t axis, bool boolType)
{
    std::vector<double> golden;
    golden.reserve(input.size());
    int64_t axisOffset = 0;
    for (const auto &outShape : outputShapes) {
        const int64_t outNumel = Numel(outShape);
        for (int64_t flat = 0; flat < outNumel; ++flat) {
            auto coords = CoordinatesFromFlat(outShape, static_cast<size_t>(flat));
            coords[static_cast<size_t>(axis)] += axisOffset;
            golden.push_back(ToDouble(input[LinearOffset(shape, coords)], boolType));
        }
        axisOffset += outShape[static_cast<size_t>(axis)];
    }
    return golden;
}

template <typename T>
std::vector<double> GoldenChunkCat(
    const std::vector<std::vector<T>> &inputs, const std::vector<std::vector<int64_t>> &shapes,
    const std::vector<int64_t> &outShape, int64_t numChunks, bool boolType)
{
    std::vector<double> golden(static_cast<size_t>(Numel(outShape)), 0.0);
    if (outShape.size() != 2) {
        throw std::runtime_error("ChunkCat output shape must be rank 2");
    }

    std::vector<int64_t> chunkRows;
    std::vector<int64_t> cols;
    chunkRows.reserve(shapes.size());
    cols.reserve(shapes.size());
    for (const auto &shape : shapes) {
        const int64_t rows = shape[0];
        chunkRows.push_back((rows + numChunks - 1) / numChunks);
        std::vector<int64_t> trailingShape(shape.begin() + 1, shape.end());
        cols.push_back(Numel(trailingShape));
    }

    const int64_t outWidth = outShape[1];
    for (int64_t chunk = 0; chunk < numChunks; ++chunk) {
        int64_t segmentOffset = 0;
        for (size_t inputIndex = 0; inputIndex < shapes.size(); ++inputIndex) {
            const int64_t rows = shapes[inputIndex][0];
            const int64_t chunkRowCount = chunkRows[inputIndex];
            const int64_t colCount = cols[inputIndex];
            for (int64_t rowInChunk = 0; rowInChunk < chunkRowCount; ++rowInChunk) {
                const int64_t srcRow = chunk * chunkRowCount + rowInChunk;
                for (int64_t col = 0; col < colCount; ++col) {
                    const int64_t outIndex = chunk * outWidth + segmentOffset + rowInChunk * colCount + col;
                    if (srcRow < rows) {
                        const int64_t srcIndex = srcRow * colCount + col;
                        golden[static_cast<size_t>(outIndex)] =
                            ToDouble(inputs[inputIndex][static_cast<size_t>(srcIndex)], boolType);
                    }
                }
            }
            segmentOffset += chunkRowCount * colCount;
        }
    }
    return golden;
}

std::vector<uint8_t> MakeMaskData(const std::vector<int64_t> &shape)
{
    const int64_t count = Numel(shape);
    std::vector<uint8_t> mask(static_cast<size_t>(count), 1);
    for (int64_t index = 0; index < count; ++index) {
        mask[static_cast<size_t>(index)] = (count == 1 || index % 3 != 1) ? 1 : 0;
    }
    return mask;
}

template <typename T>
std::vector<double> GoldenMaskedSelect(
    const std::vector<T> &input, const std::vector<int64_t> &inputShape, const std::vector<uint8_t> &mask,
    const std::vector<int64_t> &maskShape, const std::vector<int64_t> &outShape, bool boolType)
{
    std::vector<double> golden;
    const int64_t outputNumel = Numel(outShape);
    golden.reserve(static_cast<size_t>(outputNumel));
    for (int64_t flat = 0; flat < outputNumel; ++flat) {
        const size_t maskOffset = BroadcastOffset(maskShape, outShape, static_cast<size_t>(flat));
        if (mask[maskOffset] == 0) {
            continue;
        }
        const size_t inputOffset = BroadcastOffset(inputShape, outShape, static_cast<size_t>(flat));
        golden.push_back(ToDouble(input[inputOffset], boolType));
    }
    return golden;
}

template <typename T>
std::vector<double> GoldenSlice(
    const std::vector<T> &a, const std::vector<int64_t> &shape, int64_t dim, int64_t start, int64_t end,
    int64_t step, const std::vector<int64_t> &outShape, bool boolType)
{
    const int64_t rank = static_cast<int64_t>(shape.size());
    const int64_t axis = dim < 0 ? dim + rank : dim;
    const auto axisIndices = BuildSliceAxisIndices(shape[static_cast<size_t>(axis)], start, end, step);
    std::vector<double> golden(static_cast<size_t>(Numel(outShape)));
    for (size_t flat = 0; flat < golden.size(); ++flat) {
        auto inCoords = CoordinatesFromFlat(outShape, flat);
        inCoords[static_cast<size_t>(axis)] = axisIndices[static_cast<size_t>(inCoords[static_cast<size_t>(axis)])];
        golden[flat] = ToDouble(a[LinearOffset(shape, inCoords)], boolType);
    }
    return golden;
}

template <typename T>
std::vector<double> GoldenStridedSlice(
    const std::vector<T> &a, const std::vector<int64_t> &shape, const StridedSlicePlan &plan, bool boolType)
{
    std::vector<double> golden(static_cast<size_t>(Numel(plan.outShape)));
    for (size_t flat = 0; flat < golden.size(); ++flat) {
        const auto outCoords = CoordinatesFromFlat(plan.outShape, flat);
        std::vector<int64_t> inCoords(shape.size(), 0);
        for (size_t axis = 0; axis < shape.size(); ++axis) {
            const int64_t outAxis = plan.inputAxisToOutputAxis[axis];
            if (outAxis < 0) {
                inCoords[axis] = plan.axisIndices[axis].empty() ? 0 : plan.axisIndices[axis][0];
                continue;
            }
            const int64_t outCoord = outCoords[static_cast<size_t>(outAxis)];
            inCoords[axis] = plan.axisIndices[axis][static_cast<size_t>(outCoord)];
        }
        golden[flat] = ToDouble(a[LinearOffset(shape, inCoords)], boolType);
    }
    return golden;
}

std::vector<int64_t> UnfoldGradExpectedShape(
    const std::vector<int64_t> &inputSizes, int64_t dim, int64_t size, int64_t step)
{
    const int64_t rank = static_cast<int64_t>(inputSizes.size());
    if (rank == 0) {
        throw std::runtime_error("UnfoldGrad input_sizes attrs are invalid");
    }
    if (dim < -rank || dim >= rank) {
        throw std::runtime_error("UnfoldGrad dim is out of range");
    }
    const int64_t axis = dim < 0 ? dim + rank : dim;
    if (size <= 0 || size > inputSizes[static_cast<size_t>(axis)]) {
        throw std::runtime_error("UnfoldGrad size attr is invalid");
    }
    if (step <= 0) {
        throw std::runtime_error("UnfoldGrad step attr is invalid");
    }
    std::vector<int64_t> expected = inputSizes;
    expected[static_cast<size_t>(axis)] = (inputSizes[static_cast<size_t>(axis)] - size) / step + 1;
    expected.push_back(size);
    return expected;
}

int64_t NormalizeUnfoldGradDim(const std::vector<int64_t> &inputSizes, int64_t dim)
{
    const int64_t rank = static_cast<int64_t>(inputSizes.size());
    if (rank == 0 || dim < -rank || dim >= rank) {
        throw std::runtime_error("UnfoldGrad dim is out of range");
    }
    return dim < 0 ? dim + rank : dim;
}

template <typename T>
std::vector<double> GoldenUnfoldGrad(
    const std::vector<T> &gradOut, const std::vector<int64_t> &gradOutShape,
    const std::vector<int64_t> &inputSizes, int64_t dim, int64_t size, int64_t step, bool boolType)
{
    const int64_t axis = NormalizeUnfoldGradDim(inputSizes, dim);
    const auto expectedGradOutShape = UnfoldGradExpectedShape(inputSizes, axis, size, step);
    if (gradOutShape != expectedGradOutShape) {
        throw std::runtime_error("UnfoldGrad grad_out shape does not match input_sizes/dim/size/step");
    }

    std::vector<double> golden(static_cast<size_t>(Numel(inputSizes)), 0.0);
    const auto gradInStrides = Strides(inputSizes);
    for (size_t flat = 0; flat < gradOut.size(); ++flat) {
        const auto outCoords = CoordinatesFromFlat(gradOutShape, flat);
        std::vector<int64_t> inCoords(outCoords.begin(), outCoords.begin() + static_cast<int64_t>(inputSizes.size()));
        inCoords[static_cast<size_t>(axis)] =
            outCoords[static_cast<size_t>(axis)] * step + outCoords[static_cast<size_t>(inputSizes.size())];

        size_t inOffset = 0;
        for (size_t coordIndex = 0; coordIndex < inCoords.size(); ++coordIndex) {
            inOffset += static_cast<size_t>(inCoords[coordIndex] * gradInStrides[coordIndex]);
        }
        golden[inOffset] += ToDouble(gradOut[flat], boolType);
    }
    return golden;
}

template <typename T>
std::vector<double> GoldenChannelShuffle(const std::vector<T> &a, const std::vector<int64_t> &shape, int64_t groups, bool boolType)
{
    if (shape.size() < 3 || shape.size() > 7) {
        throw std::runtime_error("aclnnChannelShuffle rank must be in [3, 7]");
    }
    const int64_t channels = shape[1];
    if (groups <= 0 || channels % groups != 0) {
        throw std::runtime_error("aclnnChannelShuffle channels must be divisible by groups");
    }
    const int64_t channelsPerGroup = channels / groups;
    std::vector<double> golden(static_cast<size_t>(Numel(shape)));
    for (size_t flat = 0; flat < golden.size(); ++flat) {
        auto inCoords = CoordinatesFromFlat(shape, flat);
        const int64_t outChannel = inCoords[1];
        inCoords[1] = (outChannel % groups) * channelsPerGroup + outChannel / groups;
        golden[flat] = ToDouble(a[LinearOffset(shape, inCoords)], boolType);
    }
    return golden;
}

template <typename T>
std::vector<double> GoldenCos(const std::vector<T> &a, bool boolType)
{
    std::vector<double> golden(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        golden[i] = std::cos(ToDouble(a[i], boolType));
    }
    return golden;
}

template <typename T, typename Fn>
void RunUnaryApi(
    const char *getWorkspaceSizeName, const char *launchName, const std::vector<T> &a,
    const std::vector<int64_t> &shape, aclDataType dtype, bool boolType, aclrtStream stream, TensorHolder &aTensor,
    TensorHolder &outTensor, void *&workspace, std::vector<T> &out, std::vector<double> &golden, Fn fn)
{
    auto getWorkspaceSize = ResolveSymbol<UnaryGetWorkspaceSizeFn>(getWorkspaceSizeName);
    auto launch = ResolveSymbol<UnaryLaunchFn>(launchName);
    golden = GoldenUnary(a, boolType, fn);
    out.assign(golden.size(), T{});
    if (CreateTensor(out, shape, dtype, &outTensor) != ACL_SUCCESS) {
        throw std::runtime_error(std::string("failed to create ") + launchName + " output tensor");
    }
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto ret = getWorkspaceSize(aTensor.tensor, outTensor.tensor, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        throw std::runtime_error(std::string(getWorkspaceSizeName) + " failed: " + std::to_string(ret));
    }
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error("workspace allocation failed");
        }
    }
    ret = launch(workspace, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        throw std::runtime_error(std::string(launchName) + " failed: " + std::to_string(ret));
    }
}

enum class ReduceKind {
    Sum,
    Prod,
    Min,
    Max,
};

double ReduceInitialValue(ReduceKind kind)
{
    if (kind == ReduceKind::Min) {
        return std::numeric_limits<double>::infinity();
    }
    if (kind == ReduceKind::Max) {
        return -std::numeric_limits<double>::infinity();
    }
    if (kind == ReduceKind::Prod) {
        return 1.0;
    }
    return 0.0;
}

void ApplyReduceValue(double *current, double value, ReduceKind kind)
{
    if (kind == ReduceKind::Min) {
        *current = std::min(*current, value);
    } else if (kind == ReduceKind::Max) {
        *current = std::max(*current, value);
    } else if (kind == ReduceKind::Prod) {
        *current *= value;
    } else {
        *current += value;
    }
}

template <typename T>
uint64_t IntegralMask()
{
    if constexpr (sizeof(T) >= sizeof(uint64_t)) {
        return std::numeric_limits<uint64_t>::max();
    } else {
        return (uint64_t{1} << (sizeof(T) * 8)) - 1;
    }
}

template <typename T>
uint64_t IntegralBits(T value)
{
    using U = typename std::make_unsigned<T>::type;
    return static_cast<uint64_t>(static_cast<U>(value)) & IntegralMask<T>();
}

template <typename T>
uint64_t WrappedProduct(uint64_t current, T value)
{
    return (current * IntegralBits(value)) & IntegralMask<T>();
}

template <typename T>
uint64_t WrappedSum(uint64_t current, T value)
{
    return (current + IntegralBits(value)) & IntegralMask<T>();
}

template <typename T>
double IntegralBitsToDouble(uint64_t bits)
{
    constexpr int bitWidth = sizeof(T) * 8;
    bits &= IntegralMask<T>();
    if constexpr (std::is_signed<T>::value) {
        const uint64_t signBit = uint64_t{1} << (bitWidth - 1);
        if ((bits & signBit) != 0) {
            const uint64_t magnitude = ((~bits) + 1U) & IntegralMask<T>();
            return -static_cast<double>(magnitude);
        }
    }
    return static_cast<double>(bits);
}

template <typename T>
std::vector<double> GoldenWrappedIntegralProductReduce(
    const std::vector<T> &a, const std::vector<int64_t> &shape, const std::vector<int64_t> &axes,
    const std::vector<int64_t> &outShape, bool keepDims)
{
    std::vector<uint64_t> accum(static_cast<size_t>(Numel(outShape)), IntegralBits(static_cast<T>(1)));
    if (a.empty()) {
        std::vector<double> golden(accum.size(), IntegralBitsToDouble<T>(IntegralBits(static_cast<T>(1))));
        return golden;
    }
    const auto inputStrides = Strides(shape);
    for (size_t flat = 0; flat < a.size(); ++flat) {
        std::vector<int64_t> outCoords;
        outCoords.reserve(outShape.size());
        for (int64_t axis = 0; axis < static_cast<int64_t>(shape.size()); ++axis) {
            const size_t axisIndex = static_cast<size_t>(axis);
            const int64_t coord =
                shape[axisIndex] == 0 ? 0 : static_cast<int64_t>((flat / static_cast<size_t>(inputStrides[axisIndex])) %
                                                                  static_cast<size_t>(shape[axisIndex]));
            if (ContainsAxis(axes, axis)) {
                if (keepDims) {
                    outCoords.push_back(0);
                }
                continue;
            }
            outCoords.push_back(coord);
        }
        if (outCoords.empty() && !outShape.empty()) {
            outCoords.assign(outShape.size(), 0);
        }
        const auto outOffset = LinearOffset(outShape, outCoords);
        accum[outOffset] = WrappedProduct(accum[outOffset], a[flat]);
    }

    std::vector<double> golden(accum.size());
    for (size_t i = 0; i < accum.size(); ++i) {
        golden[i] = IntegralBitsToDouble<T>(accum[i]);
    }
    return golden;
}

template <typename T>
std::vector<double> GoldenAllReduce(const std::vector<T> &a, bool boolType, ReduceKind kind)
{
    if constexpr (std::is_integral<T>::value) {
        if (kind == ReduceKind::Prod && !boolType) {
            uint64_t accum = IntegralBits(static_cast<T>(1));
            for (const auto &value : a) {
                accum = WrappedProduct(accum, value);
            }
            return {IntegralBitsToDouble<T>(accum)};
        }
    }
    std::vector<double> golden(1, ReduceInitialValue(kind));
    for (const auto &value : a) {
        ApplyReduceValue(&golden[0], ToDouble(value, boolType), kind);
    }
    return golden;
}

template <typename T>
std::vector<double> GoldenReduce(
    const std::vector<T> &a, const std::vector<int64_t> &shape, const std::vector<int64_t> &axes,
    const std::vector<int64_t> &outShape, bool keepDims, bool boolType, ReduceKind kind)
{
    if constexpr (std::is_integral<T>::value) {
        if (kind == ReduceKind::Prod && !boolType) {
            return GoldenWrappedIntegralProductReduce(a, shape, axes, outShape, keepDims);
        }
    }
    std::vector<double> golden(static_cast<size_t>(Numel(outShape)), ReduceInitialValue(kind));
    if (a.empty()) {
        return golden;
    }
    const auto inputStrides = Strides(shape);
    for (size_t flat = 0; flat < a.size(); ++flat) {
        std::vector<int64_t> outCoords;
        outCoords.reserve(outShape.size());
        for (int64_t axis = 0; axis < static_cast<int64_t>(shape.size()); ++axis) {
            const size_t axisIndex = static_cast<size_t>(axis);
            const int64_t coord =
                shape[axisIndex] == 0 ? 0 : static_cast<int64_t>((flat / static_cast<size_t>(inputStrides[axisIndex])) %
                                                                  static_cast<size_t>(shape[axisIndex]));
            if (ContainsAxis(axes, axis)) {
                if (keepDims) {
                    outCoords.push_back(0);
                }
                continue;
            }
            outCoords.push_back(coord);
        }
        if (outCoords.empty() && !outShape.empty()) {
            outCoords.assign(outShape.size(), 0);
        }
        ApplyReduceValue(&golden[LinearOffset(outShape, outCoords)], ToDouble(a[flat], boolType), kind);
    }
    return golden;
}

template <typename T>
std::vector<double> GoldenReduceSum(
    const std::vector<T> &a, const std::vector<int64_t> &shape, const std::vector<int64_t> &axes,
    const std::vector<int64_t> &outShape, bool keepDims, bool boolType)
{
    return GoldenReduce(a, shape, axes, outShape, keepDims, boolType, ReduceKind::Sum);
}

template <typename TIn, typename TOut>
std::vector<TOut> CastVectorForProd(const std::vector<TIn> &input, bool inputBoolType, bool outputBoolType)
{
    std::vector<TOut> casted(input.size());
    for (size_t index = 0; index < input.size(); ++index) {
        const double value = ToDouble(input[index], inputBoolType);
        if constexpr (std::is_integral<TOut>::value) {
            if (outputBoolType) {
                casted[index] = CastValue<TOut>(value, true);
            } else {
                const auto truncated = static_cast<int64_t>(value);
                casted[index] = static_cast<TOut>(truncated);
            }
        } else {
            casted[index] = CastValue<TOut>(value, outputBoolType);
        }
    }
    return casted;
}

template <typename TIn, typename TOut>
std::vector<double> GoldenProdWithOutputDtype(
    const std::vector<TIn> &input, bool inputBoolType, bool outputBoolType)
{
    const auto casted = CastVectorForProd<TIn, TOut>(input, inputBoolType, outputBoolType);
    return GoldenAllReduce(casted, outputBoolType, ReduceKind::Prod);
}

template <typename TIn, typename TOut>
std::vector<double> GoldenProdDimWithOutputDtype(
    const std::vector<TIn> &input, const std::vector<int64_t> &shape, const std::vector<int64_t> &axes,
    const std::vector<int64_t> &outShape, bool keepDims, bool inputBoolType, bool outputBoolType)
{
    const auto casted = CastVectorForProd<TIn, TOut>(input, inputBoolType, outputBoolType);
    return GoldenReduce(casted, shape, axes, outShape, keepDims, outputBoolType, ReduceKind::Prod);
}

template <typename T>
std::vector<double> GoldenStd(
    const std::vector<T> &a, const std::vector<int64_t> &shape, const std::vector<int64_t> &axes,
    const std::vector<int64_t> &outShape, bool keepDims, int64_t correction, bool boolType)
{
    const size_t outSize = static_cast<size_t>(Numel(outShape));
    std::vector<double> sums(outSize, 0.0);
    std::vector<double> squaredDiffSums(outSize, 0.0);
    std::vector<int64_t> counts(outSize, 0);
    if (a.empty()) {
        return std::vector<double>(outSize, std::numeric_limits<double>::quiet_NaN());
    }

    const auto inputStrides = Strides(shape);
    for (size_t flat = 0; flat < a.size(); ++flat) {
        std::vector<int64_t> outCoords;
        outCoords.reserve(outShape.size());
        for (int64_t axis = 0; axis < static_cast<int64_t>(shape.size()); ++axis) {
            const size_t axisIndex = static_cast<size_t>(axis);
            const int64_t coord =
                shape[axisIndex] == 0 ? 0 : static_cast<int64_t>((flat / static_cast<size_t>(inputStrides[axisIndex])) %
                                                                  static_cast<size_t>(shape[axisIndex]));
            if (ContainsAxis(axes, axis)) {
                if (keepDims) {
                    outCoords.push_back(0);
                }
                continue;
            }
            outCoords.push_back(coord);
        }
        if (outCoords.empty() && !outShape.empty()) {
            outCoords.assign(outShape.size(), 0);
        }
        const auto outOffset = LinearOffset(outShape, outCoords);
        sums[outOffset] += ToDouble(a[flat], boolType);
        counts[outOffset] += 1;
    }

    std::vector<double> means(outSize, 0.0);
    for (size_t i = 0; i < outSize; ++i) {
        if (counts[i] > 0) {
            means[i] = sums[i] / static_cast<double>(counts[i]);
        }
    }

    for (size_t flat = 0; flat < a.size(); ++flat) {
        std::vector<int64_t> outCoords;
        outCoords.reserve(outShape.size());
        for (int64_t axis = 0; axis < static_cast<int64_t>(shape.size()); ++axis) {
            const size_t axisIndex = static_cast<size_t>(axis);
            const int64_t coord =
                shape[axisIndex] == 0 ? 0 : static_cast<int64_t>((flat / static_cast<size_t>(inputStrides[axisIndex])) %
                                                                  static_cast<size_t>(shape[axisIndex]));
            if (ContainsAxis(axes, axis)) {
                if (keepDims) {
                    outCoords.push_back(0);
                }
                continue;
            }
            outCoords.push_back(coord);
        }
        if (outCoords.empty() && !outShape.empty()) {
            outCoords.assign(outShape.size(), 0);
        }
        const auto outOffset = LinearOffset(outShape, outCoords);
        const double diff = ToDouble(a[flat], boolType) - means[outOffset];
        squaredDiffSums[outOffset] += diff * diff;
    }

    std::vector<double> golden(outSize, 0.0);
    for (size_t i = 0; i < outSize; ++i) {
        const int64_t denominator = counts[i] - correction;
        if (denominator <= 0) {
            golden[i] = std::numeric_limits<double>::quiet_NaN();
        } else {
            golden[i] = std::sqrt(squaredDiffSums[i] / static_cast<double>(denominator));
        }
    }
    return golden;
}

template <typename T>
std::pair<std::vector<double>, std::vector<double>> GoldenVarMean(
    const std::vector<T> &a, const std::vector<int64_t> &shape, const std::vector<int64_t> &axes,
    const std::vector<int64_t> &outShape, bool keepDims, int64_t correction, bool boolType)
{
    const size_t outSize = static_cast<size_t>(Numel(outShape));
    std::vector<double> sums(outSize, 0.0);
    std::vector<double> squaredDiffSums(outSize, 0.0);
    std::vector<int64_t> counts(outSize, 0);
    if (a.empty()) {
        return {
            std::vector<double>(outSize, std::numeric_limits<double>::quiet_NaN()),
            std::vector<double>(outSize, std::numeric_limits<double>::quiet_NaN())};
    }

    const auto inputStrides = Strides(shape);
    for (size_t flat = 0; flat < a.size(); ++flat) {
        std::vector<int64_t> outCoords;
        outCoords.reserve(outShape.size());
        for (int64_t axis = 0; axis < static_cast<int64_t>(shape.size()); ++axis) {
            const size_t axisIndex = static_cast<size_t>(axis);
            const int64_t coord =
                shape[axisIndex] == 0 ? 0 : static_cast<int64_t>((flat / static_cast<size_t>(inputStrides[axisIndex])) %
                                                                  static_cast<size_t>(shape[axisIndex]));
            if (ContainsAxis(axes, axis)) {
                if (keepDims) {
                    outCoords.push_back(0);
                }
                continue;
            }
            outCoords.push_back(coord);
        }
        if (outCoords.empty() && !outShape.empty()) {
            outCoords.assign(outShape.size(), 0);
        }
        const auto outOffset = LinearOffset(outShape, outCoords);
        sums[outOffset] += ToDouble(a[flat], boolType);
        counts[outOffset] += 1;
    }

    std::vector<double> means(outSize, std::numeric_limits<double>::quiet_NaN());
    for (size_t i = 0; i < outSize; ++i) {
        if (counts[i] > 0) {
            means[i] = sums[i] / static_cast<double>(counts[i]);
        }
    }

    for (size_t flat = 0; flat < a.size(); ++flat) {
        std::vector<int64_t> outCoords;
        outCoords.reserve(outShape.size());
        for (int64_t axis = 0; axis < static_cast<int64_t>(shape.size()); ++axis) {
            const size_t axisIndex = static_cast<size_t>(axis);
            const int64_t coord =
                shape[axisIndex] == 0 ? 0 : static_cast<int64_t>((flat / static_cast<size_t>(inputStrides[axisIndex])) %
                                                                  static_cast<size_t>(shape[axisIndex]));
            if (ContainsAxis(axes, axis)) {
                if (keepDims) {
                    outCoords.push_back(0);
                }
                continue;
            }
            outCoords.push_back(coord);
        }
        if (outCoords.empty() && !outShape.empty()) {
            outCoords.assign(outShape.size(), 0);
        }
        const auto outOffset = LinearOffset(outShape, outCoords);
        const double diff = ToDouble(a[flat], boolType) - means[outOffset];
        squaredDiffSums[outOffset] += diff * diff;
    }

    std::vector<double> vars(outSize, 0.0);
    for (size_t i = 0; i < outSize; ++i) {
        const int64_t denominator = counts[i] - correction;
        if (counts[i] <= 0 || denominator <= 0) {
            vars[i] = counts[i] == 1 && counts[i] <= correction
                          ? std::numeric_limits<double>::quiet_NaN()
                          : std::numeric_limits<double>::infinity();
        } else {
            vars[i] = squaredDiffSums[i] / static_cast<double>(denominator);
        }
    }
    return {vars, means};
}

template <typename T>
std::vector<double> GoldenGlobalPool(
    const std::vector<T> &a, const std::vector<int64_t> &shape, const std::vector<int64_t> &outShape, bool average,
    bool boolType)
{
    const size_t outSize = static_cast<size_t>(Numel(outShape));
    std::vector<double> golden(
        outSize, average ? 0.0 : -std::numeric_limits<double>::infinity());
    std::vector<int64_t> counts(outSize, 0);
    if (a.empty()) {
        return golden;
    }

    for (size_t flat = 0; flat < a.size(); ++flat) {
        auto outCoords = CoordinatesFromFlat(shape, flat);
        for (size_t axis = 2; axis < outCoords.size(); ++axis) {
            outCoords[axis] = 0;
        }
        const auto outOffset = LinearOffset(outShape, outCoords);
        const double value = ToDouble(a[flat], boolType);
        if (average) {
            golden[outOffset] += value;
            counts[outOffset] += 1;
        } else {
            golden[outOffset] = std::max(golden[outOffset], value);
        }
    }

    if (average) {
        for (size_t index = 0; index < outSize; ++index) {
            if (counts[index] > 0) {
                golden[index] /= static_cast<double>(counts[index]);
            }
        }
    }
    return golden;
}

template <typename T>
std::vector<double> GoldenCumsum(
    const std::vector<T> &a, const std::vector<int64_t> &shape, int64_t axis, bool exclusive, bool reverse,
    bool boolType)
{
    std::vector<double> golden(a.size(), 0.0);
    if (a.empty()) {
        return golden;
    }

    int64_t inner = 1;
    for (size_t i = static_cast<size_t>(axis) + 1; i < shape.size(); ++i) {
        inner *= shape[i];
    }
    int64_t outer = 1;
    for (int64_t i = 0; i < axis; ++i) {
        outer *= shape[static_cast<size_t>(i)];
    }
    const int64_t axisLen = shape[static_cast<size_t>(axis)];

    for (int64_t outerIndex = 0; outerIndex < outer; ++outerIndex) {
        for (int64_t innerIndex = 0; innerIndex < inner; ++innerIndex) {
            if constexpr (std::is_integral<T>::value) {
                if (!boolType) {
                    uint64_t accum = IntegralBits(static_cast<T>(0));
                    if (reverse) {
                        for (int64_t k = axisLen; k-- > 0;) {
                            const size_t offset =
                                static_cast<size_t>((outerIndex * axisLen + k) * inner + innerIndex);
                            if (exclusive) {
                                golden[offset] = IntegralBitsToDouble<T>(accum);
                                accum = WrappedSum(accum, a[offset]);
                            } else {
                                accum = WrappedSum(accum, a[offset]);
                                golden[offset] = IntegralBitsToDouble<T>(accum);
                            }
                        }
                    } else {
                        for (int64_t k = 0; k < axisLen; ++k) {
                            const size_t offset =
                                static_cast<size_t>((outerIndex * axisLen + k) * inner + innerIndex);
                            if (exclusive) {
                                golden[offset] = IntegralBitsToDouble<T>(accum);
                                accum = WrappedSum(accum, a[offset]);
                            } else {
                                accum = WrappedSum(accum, a[offset]);
                                golden[offset] = IntegralBitsToDouble<T>(accum);
                            }
                        }
                    }
                    continue;
                }
            }

            double accum = 0.0;
            if (reverse) {
                for (int64_t k = axisLen; k-- > 0;) {
                    const size_t offset = static_cast<size_t>((outerIndex * axisLen + k) * inner + innerIndex);
                    if (exclusive) {
                        golden[offset] = accum;
                        accum += ToDouble(a[offset], boolType);
                    } else {
                        accum += ToDouble(a[offset], boolType);
                        golden[offset] = accum;
                    }
                }
            } else {
                for (int64_t k = 0; k < axisLen; ++k) {
                    const size_t offset = static_cast<size_t>((outerIndex * axisLen + k) * inner + innerIndex);
                    if (exclusive) {
                        golden[offset] = accum;
                        accum += ToDouble(a[offset], boolType);
                    } else {
                        accum += ToDouble(a[offset], boolType);
                        golden[offset] = accum;
                    }
                }
            }
        }
    }
    return golden;
}

template <typename T>
std::vector<double> GoldenTriangular(
    const std::vector<T> &a, const std::vector<int64_t> &shape, int64_t diagonal, bool upper, bool boolType)
{
    if (shape.size() < 2) {
        throw std::runtime_error("Triangular rank must be at least 2");
    }
    std::vector<double> golden(a.size(), 0.0);
    if (a.empty()) {
        return golden;
    }

    const int64_t rows = shape[shape.size() - 2];
    const int64_t cols = shape[shape.size() - 1];
    const int64_t matrixSize = rows * cols;
    if (matrixSize <= 0) {
        return golden;
    }

    for (size_t index = 0; index < a.size(); ++index) {
        const int64_t matrixOffset = static_cast<int64_t>(index) % matrixSize;
        const int64_t row = matrixOffset / cols;
        const int64_t col = matrixOffset % cols;
        const bool keep = upper ? (col >= row + diagonal) : (col <= row + diagonal);
        golden[index] = keep ? ToDouble(a[index], boolType) : 0.0;
    }
    return golden;
}

template <typename T>
NativeResult CompareResult(const std::vector<T> &actual, const std::vector<double> &golden, bool boolType)
{
    NativeResult result;
    double maxAbsError = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        double expected = golden[i];
        if constexpr (IsHalfStorage<T>::value || std::is_integral<T>::value || std::is_floating_point<T>::value) {
            expected = ToDouble(CastExpectedValue<T>(golden[i], boolType), boolType);
        }
        const double actualValue = ToDouble(actual[i], boolType);
        if (std::isfinite(actualValue) && std::isfinite(expected)) {
            maxAbsError = std::max(maxAbsError, std::fabs(actualValue - expected));
        } else if (!SameSpecialValue(actualValue, expected)) {
            maxAbsError = std::numeric_limits<double>::infinity();
        }
        if (!NearlyEqual(actual[i], golden[i], boolType)) {
            result.status = "failed";
            result.message = "mismatch at index " + std::to_string(i) + ": actual=" + std::to_string(actualValue) +
                             ", expected=" + std::to_string(expected);
            result.maxAbsError = maxAbsError;
            result.retCode = 1;
            return result;
        }
    }
    result.maxAbsError = maxAbsError;
    return result;
}

std::string JsonEscape(const std::string &value)
{
    std::ostringstream escaped;
    for (char ch : value) {
        switch (ch) {
            case '\\':
                escaped << "\\\\";
                break;
            case '"':
                escaped << "\\\"";
                break;
            case '\n':
                escaped << "\\n";
                break;
            case '\r':
                escaped << "\\r";
                break;
            case '\t':
                escaped << "\\t";
                break;
            default:
                escaped << ch;
                break;
        }
    }
    return escaped.str();
}

void PrintStageTimingsJson(const NativeStageTimings &timings)
{
    std::cout << "\"timings\":{";
    std::cout << "\"parse_ms\":" << timings.parseMs << ",";
    std::cout << "\"input_generate_h2d_ms\":" << timings.inputGenerateH2dMs << ",";
    std::cout << "\"golden_ms\":" << timings.goldenMs << ",";
    std::cout << "\"workspace_ms\":" << timings.workspaceMs << ",";
    std::cout << "\"launch_sync_ms\":" << timings.launchSyncMs << ",";
    std::cout << "\"d2h_ms\":" << timings.d2hMs << ",";
    std::cout << "\"compare_ms\":" << timings.compareMs;
    std::cout << "}";
}

void PrintResultJson(
    const NativeResult &result, int caseIndex, double durationMs, const NativeStageTimings *timings = nullptr)
{
    std::cout << "{";
    if (caseIndex >= 0) {
        std::cout << "\"case_index\":" << caseIndex << ",";
    }
    std::cout << "\"status\":\"" << JsonEscape(result.status) << "\",";
    std::cout << "\"message\":\"" << JsonEscape(result.message) << "\",";
    std::cout << "\"max_abs_error\":";
    if (std::isfinite(result.maxAbsError)) {
        std::cout << result.maxAbsError;
    } else {
        std::cout << "null";
    }
    if (durationMs >= 0.0) {
        std::cout << ",\"duration_ms\":" << durationMs;
    }
    if (timings != nullptr) {
        std::cout << ",";
        PrintStageTimingsJson(*timings);
    }
    std::cout << "}\n";
    std::cout.flush();
}

template <typename T>
NativeResult RunTypedWithStream(const Args &args, aclDataType dtype, bool boolType, aclrtStream stream)
{
    TensorHolder aTensor;
    TensorHolder bTensor;
    TensorHolder cTensor;
    TensorHolder outTensor;
    void *workspace = nullptr;
    aclScalar *alpha = nullptr;
    aclScalar *other = nullptr;
    aclIntArray *dims = nullptr;
    aclIntArray *beginArray = nullptr;
    aclIntArray *endArray = nullptr;
    aclIntArray *stridesArray = nullptr;
    aclIntArray *inputSizesArray = nullptr;
    NativeResult result;

    try {
        auto a = MakeData<T>(args.shapeA, args.rangeALow, args.rangeAHigh, boolType);
        std::vector<int64_t> outShape = args.shapeA;
        std::vector<double> golden;
        std::vector<T> out;
        bool hasSecondOutput = false;
        size_t firstOutputSize = 0;

        if (CreateTensor(a, args.shapeA, dtype, &aTensor) != ACL_SUCCESS) {
            throw std::runtime_error("failed to create input tensor");
        }

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;

        if (args.api == "Abs") {
            auto getWorkspaceSize = ResolveSymbol<AbsGetWorkspaceSizeFn>("aclnnAbsGetWorkspaceSize");
            auto launch = ResolveSymbol<AbsLaunchFn>("aclnnAbs");
            golden = GoldenAbs(a, boolType);
            out.assign(golden.size(), T{});
            if (CreateTensor(out, outShape, dtype, &outTensor) != ACL_SUCCESS) {
                throw std::runtime_error("failed to create output tensor");
            }
            auto ret = getWorkspaceSize(aTensor.tensor, outTensor.tensor, &workspaceSize, &executor);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclnnAbsGetWorkspaceSize failed");
            }
            if (workspaceSize > 0) {
                ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                if (ret != ACL_SUCCESS) {
                    throw std::runtime_error("workspace allocation failed");
                }
            }
            ret = launch(workspace, workspaceSize, executor, stream);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclnnAbs failed");
            }
        } else if (args.api == "Add") {
            auto getWorkspaceSize = ResolveSymbol<AddGetWorkspaceSizeFn>("aclnnAddGetWorkspaceSize");
            auto launch = ResolveSymbol<AddLaunchFn>("aclnnAdd");
            auto b = MakeData<T>(args.shapeB, args.rangeBLow, args.rangeBHigh, boolType);
            outShape = BroadcastShape(args.shapeA, args.shapeB);
            const double alphaValue = ScalarGoldenValue<T>(args.alpha, boolType);
            golden = GoldenAdd(a, args.shapeA, b, args.shapeB, outShape, alphaValue, boolType);
            out.assign(golden.size(), T{});
            if (CreateTensor(b, args.shapeB, dtype, &bTensor) != ACL_SUCCESS ||
                CreateTensor(out, outShape, dtype, &outTensor) != ACL_SUCCESS) {
                throw std::runtime_error("failed to create Add tensors");
            }
            alpha = CreateScalar<T>(args.alpha, dtype, boolType);
            auto ret = getWorkspaceSize(aTensor.tensor, bTensor.tensor, alpha, outTensor.tensor, &workspaceSize, &executor);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclnnAddGetWorkspaceSize failed");
            }
            if (workspaceSize > 0) {
                ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                if (ret != ACL_SUCCESS) {
                    throw std::runtime_error("workspace allocation failed");
                }
            }
            ret = launch(workspace, workspaceSize, executor, stream);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclnnAdd failed");
            }
        } else if (args.api == "Adds") {
            auto getWorkspaceSize = ResolveSymbol<AddsGetWorkspaceSizeFn>("aclnnAddsGetWorkspaceSize");
            auto launch = ResolveSymbol<AddsLaunchFn>("aclnnAdds");
            const double otherValue = ScalarGoldenValue<T>(args.other, boolType);
            const double alphaValue = ScalarGoldenValue<T>(args.alpha, boolType);
            golden.resize(a.size());
            for (size_t i = 0; i < a.size(); ++i) {
                golden[i] = ToDouble(a[i], boolType) + alphaValue * otherValue;
            }
            out.assign(golden.size(), T{});
            if (CreateTensor(out, outShape, dtype, &outTensor) != ACL_SUCCESS) {
                throw std::runtime_error("failed to create Adds output tensor");
            }
            other = CreateScalar<T>(args.other, dtype, boolType);
            alpha = CreateScalar<T>(args.alpha, dtype, boolType);
            auto ret = getWorkspaceSize(aTensor.tensor, other, alpha, outTensor.tensor, &workspaceSize, &executor);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclnnAddsGetWorkspaceSize failed");
            }
            if (workspaceSize > 0) {
                ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                if (ret != ACL_SUCCESS) {
                    throw std::runtime_error("workspace allocation failed");
                }
            }
            ret = launch(workspace, workspaceSize, executor, stream);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclnnAdds failed");
            }
        } else if (args.api == "Cos") {
            auto getWorkspaceSize = ResolveSymbol<CosGetWorkspaceSizeFn>("aclnnCosGetWorkspaceSize");
            auto launch = ResolveSymbol<CosLaunchFn>("aclnnCos");
            golden = GoldenCos(a, boolType);
            out.assign(golden.size(), T{});
            if (CreateTensor(out, outShape, dtype, &outTensor) != ACL_SUCCESS) {
                throw std::runtime_error("failed to create Cos output tensor");
            }
            auto ret = getWorkspaceSize(aTensor.tensor, outTensor.tensor, &workspaceSize, &executor);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclnnCosGetWorkspaceSize failed");
            }
            if (workspaceSize > 0) {
                ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                if (ret != ACL_SUCCESS) {
                    throw std::runtime_error("workspace allocation failed");
                }
            }
            ret = launch(workspace, workspaceSize, executor, stream);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclnnCos failed");
            }
        } else if (args.api == "Ceil") {
            RunUnaryApi(
                "aclnnCeilGetWorkspaceSize", "aclnnCeil", a, args.shapeA, dtype, boolType, stream, aTensor, outTensor,
                workspace, out, golden, [](double value) { return std::ceil(value); });
        } else if (args.api == "Exp") {
            RunUnaryApi(
                "aclnnExpGetWorkspaceSize", "aclnnExp", a, args.shapeA, dtype, boolType, stream, aTensor, outTensor,
                workspace, out, golden, [](double value) { return std::exp(value); });
        } else if (args.api == "Floor") {
            RunUnaryApi(
                "aclnnFloorGetWorkspaceSize", "aclnnFloor", a, args.shapeA, dtype, boolType, stream, aTensor,
                outTensor, workspace, out, golden, [](double value) { return std::floor(value); });
        } else if (args.api == "Log") {
            RunUnaryApi(
                "aclnnLogGetWorkspaceSize", "aclnnLog", a, args.shapeA, dtype, boolType, stream, aTensor, outTensor,
                workspace, out, golden, [](double value) { return std::log(value); });
        } else if (args.api == "Cumsum") {
            if (!args.dtypeAttr.empty() && args.dtypeAttr != args.dtype) {
                throw std::runtime_error("Cumsum dtype_attr different from tensor dtype is not supported");
            }
            auto getWorkspaceSize = ResolveSymbol<CumsumGetWorkspaceSizeFn>("aclnnCumsumGetWorkspaceSize");
            auto launch = ResolveSymbol<CumsumLaunchFn>("aclnnCumsum");
            const int64_t axis = NormalizeSingleDim(args.api, args.shapeA, args.dims);
            golden = GoldenCumsum(a, args.shapeA, axis, false, false, boolType);
            out.assign(golden.size(), T{});
            if (CreateTensor(out, outShape, dtype, &outTensor) != ACL_SUCCESS) {
                throw std::runtime_error("failed to create Cumsum output tensor");
            }
            auto ret = getWorkspaceSize(aTensor.tensor, axis, dtype, outTensor.tensor, &workspaceSize, &executor);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclnnCumsumGetWorkspaceSize failed: " + std::to_string(ret));
            }
            if (workspaceSize > 0) {
                ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                if (ret != ACL_SUCCESS) {
                    throw std::runtime_error("workspace allocation failed");
                }
            }
            ret = launch(workspace, workspaceSize, executor, stream);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclnnCumsum failed: " + std::to_string(ret));
            }
        } else if (args.api == "CumsumV2") {
            auto getWorkspaceSize = ResolveSymbol<CumsumV2GetWorkspaceSizeFn>("aclnnCumsumV2GetWorkspaceSize");
            auto launch = ResolveSymbol<CumsumV2LaunchFn>("aclnnCumsumV2");
            const int64_t axis = NormalizeSingleDim(args.api, args.shapeA, args.dims);
            golden = GoldenCumsum(a, args.shapeA, axis, args.exclusive, args.reverse, boolType);
            out.assign(golden.size(), T{});
            if (CreateTensor(out, outShape, dtype, &outTensor) != ACL_SUCCESS) {
                throw std::runtime_error("failed to create CumsumV2 output tensor");
            }
            auto ret =
                getWorkspaceSize(aTensor.tensor, axis, args.exclusive, args.reverse, outTensor.tensor, &workspaceSize, &executor);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclnnCumsumV2GetWorkspaceSize failed: " + std::to_string(ret));
            }
            if (workspaceSize > 0) {
                ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                if (ret != ACL_SUCCESS) {
                    throw std::runtime_error("workspace allocation failed");
                }
            }
            ret = launch(workspace, workspaceSize, executor, stream);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclnnCumsumV2 failed: " + std::to_string(ret));
            }
        } else if (args.api == "Tril") {
            auto getWorkspaceSize = ResolveSymbol<TrilGetWorkspaceSizeFn>("aclnnTrilGetWorkspaceSize");
            auto launch = ResolveSymbol<TrilLaunchFn>("aclnnTril");
            golden = GoldenTriangular(a, args.shapeA, args.diagonal, false, boolType);
            out.assign(golden.size(), T{});
            if (CreateTensor(out, outShape, dtype, &outTensor) != ACL_SUCCESS) {
                throw std::runtime_error("failed to create Tril output tensor");
            }
            auto ret = getWorkspaceSize(aTensor.tensor, args.diagonal, outTensor.tensor, &workspaceSize, &executor);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclnnTrilGetWorkspaceSize failed: " + std::to_string(ret));
            }
            if (workspaceSize > 0) {
                ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                if (ret != ACL_SUCCESS) {
                    throw std::runtime_error("workspace allocation failed");
                }
            }
            ret = launch(workspace, workspaceSize, executor, stream);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclnnTril failed: " + std::to_string(ret));
            }
        } else if (args.api == "Triu") {
            auto getWorkspaceSize = ResolveSymbol<TriuGetWorkspaceSizeFn>("aclnnTriuGetWorkspaceSize");
            auto launch = ResolveSymbol<TriuLaunchFn>("aclnnTriu");
            golden = GoldenTriangular(a, args.shapeA, args.diagonal, true, boolType);
            out.assign(golden.size(), T{});
            if (CreateTensor(out, outShape, dtype, &outTensor) != ACL_SUCCESS) {
                throw std::runtime_error("failed to create Triu output tensor");
            }
            auto ret = getWorkspaceSize(aTensor.tensor, args.diagonal, outTensor.tensor, &workspaceSize, &executor);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclnnTriuGetWorkspaceSize failed: " + std::to_string(ret));
            }
            if (workspaceSize > 0) {
                ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                if (ret != ACL_SUCCESS) {
                    throw std::runtime_error("workspace allocation failed");
                }
            }
            ret = launch(workspace, workspaceSize, executor, stream);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclnnTriu failed: " + std::to_string(ret));
            }
        } else if (args.api == "Std") {
            auto getWorkspaceSize = ResolveSymbol<StdGetWorkspaceSizeFn>("aclnnStdGetWorkspaceSize");
            auto launch = ResolveSymbol<StdLaunchFn>("aclnnStd");
            const auto reduceDims = NormalizeReduceDimsUnique(args.api, args.shapeA, args.dims, false);
            outShape = ReduceOutputShape(args.shapeA, reduceDims, args.keepDims);
            golden = GoldenStd(a, args.shapeA, reduceDims, outShape, args.keepDims, args.correction, boolType);
            out.assign(golden.size(), T{});
            if (CreateTensor(out, outShape, dtype, &outTensor) != ACL_SUCCESS) {
                throw std::runtime_error("failed to create Std output tensor");
            }
            dims = aclCreateIntArray(reduceDims.empty() ? nullptr : reduceDims.data(), reduceDims.size());
            if (dims == nullptr) {
                throw std::runtime_error("failed to create Std dims");
            }
            auto ret =
                getWorkspaceSize(aTensor.tensor, dims, args.correction, args.keepDims, outTensor.tensor, &workspaceSize, &executor);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclnnStdGetWorkspaceSize failed: " + std::to_string(ret));
            }
            if (workspaceSize > 0) {
                ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                if (ret != ACL_SUCCESS) {
                    throw std::runtime_error("workspace allocation failed");
                }
            }
            ret = launch(workspace, workspaceSize, executor, stream);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclnnStd failed: " + std::to_string(ret));
            }
        } else if (args.api == "Var" || args.api == "VarCorrection" || args.api == "VarMean") {
            const auto reduceDims = NormalizeReduceDims(args.api, args.shapeA, args.dims, false);
            outShape = ReduceOutputShape(args.shapeA, reduceDims, args.keepDims);
            const auto moments =
                GoldenVarMean(a, args.shapeA, reduceDims, outShape, args.keepDims, args.correction, boolType);
            golden = moments.first;
            out.assign(golden.size(), T{});
            if (CreateTensor(out, outShape, dtype, &outTensor) != ACL_SUCCESS) {
                throw std::runtime_error("failed to create " + args.api + " output tensor");
            }
            dims = aclCreateIntArray(reduceDims.empty() ? nullptr : reduceDims.data(), reduceDims.size());
            if (dims == nullptr) {
                throw std::runtime_error("failed to create " + args.api + " dims");
            }

            aclnnStatus ret;
            if (args.api == "Var") {
                auto getWorkspaceSize = ResolveSymbol<VarGetWorkspaceSizeFn>("aclnnVarGetWorkspaceSize");
                auto launch = ResolveSymbol<VarLaunchFn>("aclnnVar");
                ret = getWorkspaceSize(
                    aTensor.tensor, dims, args.correction != 0, args.keepDims, outTensor.tensor, &workspaceSize,
                    &executor);
                if (ret != ACL_SUCCESS) {
                    throw std::runtime_error("aclnnVarGetWorkspaceSize failed: " + std::to_string(ret));
                }
                if (workspaceSize > 0) {
                    ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                    if (ret != ACL_SUCCESS) {
                        throw std::runtime_error("workspace allocation failed");
                    }
                }
                ret = launch(workspace, workspaceSize, executor, stream);
                if (ret != ACL_SUCCESS) {
                    throw std::runtime_error("aclnnVar failed: " + std::to_string(ret));
                }
            } else if (args.api == "VarCorrection") {
                auto getWorkspaceSize =
                    ResolveSymbol<VarCorrectionGetWorkspaceSizeFn>("aclnnVarCorrectionGetWorkspaceSize");
                auto launch = ResolveSymbol<VarCorrectionLaunchFn>("aclnnVarCorrection");
                ret = getWorkspaceSize(
                    aTensor.tensor, dims, args.correction, args.keepDims, outTensor.tensor, &workspaceSize, &executor);
                if (ret != ACL_SUCCESS) {
                    throw std::runtime_error("aclnnVarCorrectionGetWorkspaceSize failed: " + std::to_string(ret));
                }
                if (workspaceSize > 0) {
                    ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                    if (ret != ACL_SUCCESS) {
                        throw std::runtime_error("workspace allocation failed");
                    }
                }
                ret = launch(workspace, workspaceSize, executor, stream);
                if (ret != ACL_SUCCESS) {
                    throw std::runtime_error("aclnnVarCorrection failed: " + std::to_string(ret));
                }
            } else {
                auto getWorkspaceSize = ResolveSymbol<VarMeanGetWorkspaceSizeFn>("aclnnVarMeanGetWorkspaceSize");
                auto launch = ResolveSymbol<VarMeanLaunchFn>("aclnnVarMean");
                std::vector<T> meanOut(moments.second.size(), T{});
                if (CreateTensor(meanOut, outShape, dtype, &cTensor) != ACL_SUCCESS) {
                    throw std::runtime_error("failed to create VarMean mean output tensor");
                }
                firstOutputSize = golden.size();
                golden.insert(golden.end(), moments.second.begin(), moments.second.end());
                hasSecondOutput = true;
                ret = getWorkspaceSize(
                    aTensor.tensor, dims, args.correction, args.keepDims, outTensor.tensor, cTensor.tensor,
                    &workspaceSize, &executor);
                if (ret != ACL_SUCCESS) {
                    throw std::runtime_error("aclnnVarMeanGetWorkspaceSize failed: " + std::to_string(ret));
                }
                if (workspaceSize > 0) {
                    ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                    if (ret != ACL_SUCCESS) {
                        throw std::runtime_error("workspace allocation failed");
                    }
                }
                ret = launch(workspace, workspaceSize, executor, stream);
                if (ret != ACL_SUCCESS) {
                    throw std::runtime_error("aclnnVarMean failed: " + std::to_string(ret));
                }
            }
        } else if (args.api == "GlobalAveragePool" || args.api == "GlobalMaxPool") {
            const bool average = args.api == "GlobalAveragePool";
            auto getWorkspaceSize = average
                                        ? ResolveSymbol<GlobalPoolGetWorkspaceSizeFn>(
                                              "aclnnGlobalAveragePoolGetWorkspaceSize")
                                        : ResolveSymbol<GlobalPoolGetWorkspaceSizeFn>(
                                              "aclnnGlobalMaxPoolGetWorkspaceSize");
            auto launch = average ? ResolveSymbol<GlobalPoolLaunchFn>("aclnnGlobalAveragePool")
                                  : ResolveSymbol<GlobalPoolLaunchFn>("aclnnGlobalMaxPool");
            outShape = GlobalPoolOutputShape(args.shapeA);
            golden = GoldenGlobalPool(a, args.shapeA, outShape, average, boolType);
            out.assign(golden.size(), T{});
            if (CreateTensor(out, outShape, dtype, &outTensor) != ACL_SUCCESS) {
                throw std::runtime_error("failed to create " + args.api + " output tensor");
            }
            auto ret = getWorkspaceSize(aTensor.tensor, outTensor.tensor, &workspaceSize, &executor);
            if (ret != ACL_SUCCESS) {
                const auto name = average ? "aclnnGlobalAveragePoolGetWorkspaceSize"
                                          : "aclnnGlobalMaxPoolGetWorkspaceSize";
                throw std::runtime_error(std::string(name) + " failed: " + std::to_string(ret));
            }
            if (workspaceSize > 0) {
                ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                if (ret != ACL_SUCCESS) {
                    throw std::runtime_error("workspace allocation failed");
                }
            }
            ret = launch(workspace, workspaceSize, executor, stream);
            if (ret != ACL_SUCCESS) {
                const auto name = average ? "aclnnGlobalAveragePool" : "aclnnGlobalMaxPool";
                throw std::runtime_error(std::string(name) + " failed: " + std::to_string(ret));
            }
        } else if (args.api == "Slice") {
            if (args.dims.size() != 1) {
                throw std::runtime_error("Slice expects exactly 1 dim");
            }
            auto getWorkspaceSize = ResolveSymbol<SliceGetWorkspaceSizeFn>("aclnnSliceGetWorkspaceSize");
            auto launch = ResolveSymbol<SliceLaunchFn>("aclnnSlice");
            const int64_t dim = args.dims[0];
            outShape = SliceOutputShape(args.shapeA, dim, args.start, args.sliceEnd, args.step);
            golden = GoldenSlice(a, args.shapeA, dim, args.start, args.sliceEnd, args.step, outShape, boolType);
            out.assign(golden.size(), T{});
            if (CreateTensor(out, outShape, dtype, &outTensor) != ACL_SUCCESS) {
                throw std::runtime_error("failed to create Slice output tensor");
            }
            auto ret = getWorkspaceSize(
                aTensor.tensor, dim, args.start, args.sliceEnd, args.step, outTensor.tensor, &workspaceSize,
                &executor);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclnnSliceGetWorkspaceSize failed: " + std::to_string(ret));
            }
            if (workspaceSize > 0) {
                ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                if (ret != ACL_SUCCESS) {
                    throw std::runtime_error("workspace allocation failed");
                }
            }
            ret = launch(workspace, workspaceSize, executor, stream);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclnnSlice failed: " + std::to_string(ret));
            }
        } else if (args.api == "StridedSlice") {
            auto getWorkspaceSize = ResolveSymbol<StridedSliceGetWorkspaceSizeFn>("aclnnStridedSliceGetWorkspaceSize");
            auto launch = ResolveSymbol<StridedSliceLaunchFn>("aclnnStridedSlice");
            const auto plan = BuildStridedSlicePlan(
                args.shapeA, args.begin, args.ends, args.strides, args.beginMask, args.endMask, args.ellipsisMask,
                args.newAxisMask, args.shrinkAxisMask);
            outShape = plan.outShape;
            golden = GoldenStridedSlice(a, args.shapeA, plan, boolType);
            out.assign(golden.size(), T{});
            if (CreateTensor(out, outShape, dtype, &outTensor) != ACL_SUCCESS) {
                throw std::runtime_error("failed to create StridedSlice output tensor");
            }
            beginArray = aclCreateIntArray(args.begin.empty() ? nullptr : args.begin.data(), args.begin.size());
            endArray = aclCreateIntArray(args.ends.empty() ? nullptr : args.ends.data(), args.ends.size());
            stridesArray = aclCreateIntArray(args.strides.empty() ? nullptr : args.strides.data(), args.strides.size());
            if (beginArray == nullptr || endArray == nullptr || stridesArray == nullptr) {
                throw std::runtime_error("failed to create StridedSlice attrs");
            }
            auto ret = getWorkspaceSize(
                aTensor.tensor, beginArray, endArray, stridesArray, args.beginMask, args.endMask, args.ellipsisMask,
                args.newAxisMask, args.shrinkAxisMask, outTensor.tensor, &workspaceSize, &executor);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclnnStridedSliceGetWorkspaceSize failed: " + std::to_string(ret));
            }
            if (workspaceSize > 0) {
                ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                if (ret != ACL_SUCCESS) {
                    throw std::runtime_error("workspace allocation failed");
                }
            }
            ret = launch(workspace, workspaceSize, executor, stream);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclnnStridedSlice failed: " + std::to_string(ret));
            }
        } else if (args.api == "Div") {
            auto getWorkspaceSize = ResolveSymbol<DivGetWorkspaceSizeFn>("aclnnDivGetWorkspaceSize");
            auto launch = ResolveSymbol<DivLaunchFn>("aclnnDiv");
            auto b = MakeData<T>(args.shapeB, args.rangeBLow, args.rangeBHigh, boolType);
            outShape = BroadcastShape(args.shapeA, args.shapeB);
            golden = GoldenDiv(a, args.shapeA, b, args.shapeB, outShape, boolType);
            out.assign(golden.size(), T{});
            if (CreateTensor(b, args.shapeB, dtype, &bTensor) != ACL_SUCCESS ||
                CreateTensor(out, outShape, dtype, &outTensor) != ACL_SUCCESS) {
                throw std::runtime_error("failed to create Div tensors");
            }
            auto ret = getWorkspaceSize(aTensor.tensor, bTensor.tensor, outTensor.tensor, &workspaceSize, &executor);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclnnDivGetWorkspaceSize failed: " + std::to_string(ret));
            }
            if (workspaceSize > 0) {
                ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                if (ret != ACL_SUCCESS) {
                    throw std::runtime_error("workspace allocation failed");
                }
            }
            ret = launch(workspace, workspaceSize, executor, stream);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclnnDiv failed: " + std::to_string(ret));
            }
        } else if (args.api == "Mul") {
            auto getWorkspaceSize = ResolveSymbol<MulGetWorkspaceSizeFn>("aclnnMulGetWorkspaceSize");
            auto launch = ResolveSymbol<MulLaunchFn>("aclnnMul");
            auto b = MakeData<T>(args.shapeB, args.rangeBLow, args.rangeBHigh, boolType);
            outShape = BroadcastShape(args.shapeA, args.shapeB);
            golden = GoldenMul(a, args.shapeA, b, args.shapeB, outShape, boolType);
            out.assign(golden.size(), T{});
            if (CreateTensor(b, args.shapeB, dtype, &bTensor) != ACL_SUCCESS ||
                CreateTensor(out, outShape, dtype, &outTensor) != ACL_SUCCESS) {
                throw std::runtime_error("failed to create Mul tensors");
            }
            auto ret = getWorkspaceSize(aTensor.tensor, bTensor.tensor, outTensor.tensor, &workspaceSize, &executor);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclnnMulGetWorkspaceSize failed");
            }
            if (workspaceSize > 0) {
                ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                if (ret != ACL_SUCCESS) {
                    throw std::runtime_error("workspace allocation failed");
                }
            }
            ret = launch(workspace, workspaceSize, executor, stream);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclnnMul failed");
            }
        } else if (args.api == "Maximum") {
            auto getWorkspaceSize = ResolveSymbol<MaximumGetWorkspaceSizeFn>("aclnnMaximumGetWorkspaceSize");
            auto launch = ResolveSymbol<MaximumLaunchFn>("aclnnMaximum");
            auto b = MakeData<T>(args.shapeB, args.rangeBLow, args.rangeBHigh, boolType);
            outShape = BroadcastShape(args.shapeA, args.shapeB);
            golden = GoldenMaximum(a, args.shapeA, b, args.shapeB, outShape, boolType);
            out.assign(golden.size(), T{});
            if (CreateTensor(b, args.shapeB, dtype, &bTensor) != ACL_SUCCESS ||
                CreateTensor(out, outShape, dtype, &outTensor) != ACL_SUCCESS) {
                throw std::runtime_error("failed to create Maximum tensors");
            }
            auto ret = getWorkspaceSize(aTensor.tensor, bTensor.tensor, outTensor.tensor, &workspaceSize, &executor);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclnnMaximumGetWorkspaceSize failed: " + std::to_string(ret));
            }
            if (workspaceSize > 0) {
                ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                if (ret != ACL_SUCCESS) {
                    throw std::runtime_error("workspace allocation failed");
                }
            }
            ret = launch(workspace, workspaceSize, executor, stream);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclnnMaximum failed: " + std::to_string(ret));
            }
        } else if (args.api == "Minimum") {
            auto getWorkspaceSize = ResolveSymbol<MinimumGetWorkspaceSizeFn>("aclnnMinimumGetWorkspaceSize");
            auto launch = ResolveSymbol<MinimumLaunchFn>("aclnnMinimum");
            auto b = MakeData<T>(args.shapeB, args.rangeBLow, args.rangeBHigh, boolType);
            outShape = BroadcastShape(args.shapeA, args.shapeB);
            golden = GoldenMinimum(a, args.shapeA, b, args.shapeB, outShape, boolType);
            out.assign(golden.size(), T{});
            if (CreateTensor(b, args.shapeB, dtype, &bTensor) != ACL_SUCCESS ||
                CreateTensor(out, outShape, dtype, &outTensor) != ACL_SUCCESS) {
                throw std::runtime_error("failed to create Minimum tensors");
            }
            auto ret = getWorkspaceSize(aTensor.tensor, bTensor.tensor, outTensor.tensor, &workspaceSize, &executor);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclnnMinimumGetWorkspaceSize failed: " + std::to_string(ret));
            }
            if (workspaceSize > 0) {
                ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                if (ret != ACL_SUCCESS) {
                    throw std::runtime_error("workspace allocation failed");
                }
            }
            ret = launch(workspace, workspaceSize, executor, stream);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclnnMinimum failed: " + std::to_string(ret));
            }
        } else if (args.api == "Ger") {
            auto getWorkspaceSize = ResolveSymbol<GerGetWorkspaceSizeFn>("aclnnGerGetWorkspaceSize");
            auto launch = ResolveSymbol<GerLaunchFn>("aclnnGer");
            auto b = MakeData<T>(args.shapeB, args.rangeBLow, args.rangeBHigh, boolType);
            if (args.shapeA.size() != 1 || args.shapeB.size() != 1) {
                throw std::runtime_error("Ger expects 1D tensor inputs");
            }
            outShape = {args.shapeA[0], args.shapeB[0]};
            golden = GoldenGer(a, b, boolType);
            out.assign(golden.size(), T{});
            if (CreateTensor(b, args.shapeB, dtype, &bTensor) != ACL_SUCCESS ||
                CreateTensor(out, outShape, dtype, &outTensor) != ACL_SUCCESS) {
                throw std::runtime_error("failed to create Ger tensors");
            }
            auto ret = getWorkspaceSize(aTensor.tensor, bTensor.tensor, outTensor.tensor, &workspaceSize, &executor);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclnnGerGetWorkspaceSize failed: " + std::to_string(ret));
            }
            if (workspaceSize > 0) {
                ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                if (ret != ACL_SUCCESS) {
                    throw std::runtime_error("workspace allocation failed");
                }
            }
            ret = launch(workspace, workspaceSize, executor, stream);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclnnGer failed: " + std::to_string(ret));
            }
        } else if (args.api == "Muls") {
            auto getWorkspaceSize = ResolveSymbol<MulsGetWorkspaceSizeFn>("aclnnMulsGetWorkspaceSize");
            auto launch = ResolveSymbol<MulsLaunchFn>("aclnnMuls");
            const double otherValue = ScalarGoldenValue<T>(args.other, boolType);
            golden = GoldenMuls(a, otherValue, boolType);
            out.assign(golden.size(), T{});
            if (CreateTensor(out, outShape, dtype, &outTensor) != ACL_SUCCESS) {
                throw std::runtime_error("failed to create Muls output tensor");
            }
            other = CreateScalar<T>(args.other, dtype, boolType);
            auto ret = getWorkspaceSize(aTensor.tensor, other, outTensor.tensor, &workspaceSize, &executor);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclnnMulsGetWorkspaceSize failed");
            }
            if (workspaceSize > 0) {
                ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                if (ret != ACL_SUCCESS) {
                    throw std::runtime_error("workspace allocation failed");
                }
            }
            ret = launch(workspace, workspaceSize, executor, stream);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclnnMuls failed");
            }
        } else if (args.api == "Divs") {
            auto getWorkspaceSize = ResolveSymbol<DivsGetWorkspaceSizeFn>("aclnnDivsGetWorkspaceSize");
            auto launch = ResolveSymbol<DivsLaunchFn>("aclnnDivs");
            const double otherValue = ScalarGoldenValue<T>(args.other, boolType);
            golden = GoldenDivs(a, otherValue, boolType);
            out.assign(golden.size(), T{});
            if (CreateTensor(out, outShape, dtype, &outTensor) != ACL_SUCCESS) {
                throw std::runtime_error("failed to create Divs output tensor");
            }
            other = CreateScalar<T>(args.other, dtype, boolType);
            auto ret = getWorkspaceSize(aTensor.tensor, other, outTensor.tensor, &workspaceSize, &executor);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclnnDivsGetWorkspaceSize failed: " + std::to_string(ret));
            }
            if (workspaceSize > 0) {
                ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                if (ret != ACL_SUCCESS) {
                    throw std::runtime_error("workspace allocation failed");
                }
            }
            ret = launch(workspace, workspaceSize, executor, stream);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclnnDivs failed: " + std::to_string(ret));
            }
        } else if (args.api == "Hardtanh" || args.api == "Clamp") {
            const std::string getWorkspaceSizeName = args.api == "Hardtanh" ? "aclnnHardtanhGetWorkspaceSize" : "aclnnClampGetWorkspaceSize";
            const std::string launchName = args.api == "Hardtanh" ? "aclnnHardtanh" : "aclnnClamp";
            auto getWorkspaceSize = ResolveSymbol<ScalarClipGetWorkspaceSizeFn>(getWorkspaceSizeName.c_str());
            auto launch = ResolveSymbol<ScalarClipLaunchFn>(launchName.c_str());
            const double clipMinValue = ScalarGoldenValue<T>(args.clipMin, boolType);
            const double clipMaxValue = ScalarGoldenValue<T>(args.clipMax, boolType);
            golden = GoldenClip(a, clipMinValue, clipMaxValue, true, true, boolType);
            out.assign(golden.size(), T{});
            if (CreateTensor(out, outShape, dtype, &outTensor) != ACL_SUCCESS) {
                throw std::runtime_error("failed to create " + args.api + " output tensor");
            }
            alpha = CreateScalar<T>(args.clipMin, dtype, boolType);
            other = CreateScalar<T>(args.clipMax, dtype, boolType);
            auto ret = getWorkspaceSize(aTensor.tensor, alpha, other, outTensor.tensor, &workspaceSize, &executor);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error(getWorkspaceSizeName + " failed: " + std::to_string(ret));
            }
            if (workspaceSize > 0) {
                ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                if (ret != ACL_SUCCESS) {
                    throw std::runtime_error("workspace allocation failed");
                }
            }
            ret = launch(workspace, workspaceSize, executor, stream);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error(launchName + " failed: " + std::to_string(ret));
            }
        } else if (args.api == "ClampMin" || args.api == "ClampMax") {
            const bool hasMin = args.api == "ClampMin";
            const std::string getWorkspaceSizeName = hasMin ? "aclnnClampMinGetWorkspaceSize" : "aclnnClampMaxGetWorkspaceSize";
            const std::string launchName = hasMin ? "aclnnClampMin" : "aclnnClampMax";
            auto getWorkspaceSize = ResolveSymbol<ScalarClipOneBoundGetWorkspaceSizeFn>(getWorkspaceSizeName.c_str());
            auto launch = ResolveSymbol<ScalarClipOneBoundLaunchFn>(launchName.c_str());
            const double boundValue = ScalarGoldenValue<T>(hasMin ? args.clipMin : args.clipMax, boolType);
            golden = GoldenClip(a, boundValue, boundValue, hasMin, !hasMin, boolType);
            out.assign(golden.size(), T{});
            if (CreateTensor(out, outShape, dtype, &outTensor) != ACL_SUCCESS) {
                throw std::runtime_error("failed to create " + args.api + " output tensor");
            }
            alpha = CreateScalar<T>(hasMin ? args.clipMin : args.clipMax, dtype, boolType);
            auto ret = getWorkspaceSize(aTensor.tensor, alpha, outTensor.tensor, &workspaceSize, &executor);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error(getWorkspaceSizeName + " failed: " + std::to_string(ret));
            }
            if (workspaceSize > 0) {
                ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                if (ret != ACL_SUCCESS) {
                    throw std::runtime_error("workspace allocation failed");
                }
            }
            ret = launch(workspace, workspaceSize, executor, stream);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error(launchName + " failed: " + std::to_string(ret));
            }
        } else if (args.api == "ClampTensor") {
            auto getWorkspaceSize = ResolveSymbol<TensorClipGetWorkspaceSizeFn>("aclnnClampTensorGetWorkspaceSize");
            auto launch = ResolveSymbol<TensorClipLaunchFn>("aclnnClampTensor");
            const std::string dtypeBName = args.dtypeB.empty() ? args.dtype : args.dtypeB;
            const std::string dtypeCName = args.dtypeC.empty() ? args.dtype : args.dtypeC;
            std::vector<double> minValues;
            std::vector<double> maxValues;
            outShape = BroadcastShape(BroadcastShape(args.shapeA, args.shapeB), args.shapeC);
            if (CreateGeneratedTensorByDtype(
                    dtypeBName, args.shapeB, args.rangeBLow, args.rangeBHigh, &bTensor, &minValues) !=
                    ACL_SUCCESS ||
                CreateGeneratedTensorByDtype(
                    dtypeCName, args.shapeC, args.rangeCLow, args.rangeCHigh, &cTensor, &maxValues) !=
                    ACL_SUCCESS) {
                throw std::runtime_error("failed to create ClampTensor bound tensors");
            }
            golden = GoldenTensorClip(
                a, args.shapeA, minValues, args.shapeB, maxValues, args.shapeC, outShape, true, true, boolType);
            out.assign(golden.size(), T{});
            if (CreateTensor(out, outShape, dtype, &outTensor) != ACL_SUCCESS) {
                throw std::runtime_error("failed to create ClampTensor output tensor");
            }
            auto ret =
                getWorkspaceSize(aTensor.tensor, bTensor.tensor, cTensor.tensor, outTensor.tensor, &workspaceSize, &executor);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclnnClampTensorGetWorkspaceSize failed: " + std::to_string(ret));
            }
            if (workspaceSize > 0) {
                ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                if (ret != ACL_SUCCESS) {
                    throw std::runtime_error("workspace allocation failed");
                }
            }
            ret = launch(workspace, workspaceSize, executor, stream);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclnnClampTensor failed: " + std::to_string(ret));
            }
        } else if (args.api == "ClampMinTensor" || args.api == "ClampMaxTensor") {
            const bool hasMin = args.api == "ClampMinTensor";
            const std::string getWorkspaceSizeName =
                hasMin ? "aclnnClampMinTensorGetWorkspaceSize" : "aclnnClampMaxTensorGetWorkspaceSize";
            const std::string launchName = hasMin ? "aclnnClampMinTensor" : "aclnnClampMaxTensor";
            auto getWorkspaceSize = ResolveSymbol<TensorClipOneBoundGetWorkspaceSizeFn>(getWorkspaceSizeName.c_str());
            auto launch = ResolveSymbol<TensorClipLaunchFn>(launchName.c_str());
            const std::string dtypeBName = args.dtypeB.empty() ? args.dtype : args.dtypeB;
            std::vector<double> boundValues;
            outShape = BroadcastShape(args.shapeA, args.shapeB);
            if (CreateGeneratedTensorByDtype(
                    dtypeBName, args.shapeB, args.rangeBLow, args.rangeBHigh, &bTensor, &boundValues) != ACL_SUCCESS) {
                throw std::runtime_error("failed to create " + args.api + " bound tensor");
            }
            const std::vector<double> emptyValues;
            const std::vector<int64_t> emptyShape;
            golden = hasMin
                         ? GoldenTensorClip(
                               a, args.shapeA, boundValues, args.shapeB, emptyValues, emptyShape, outShape, true, false,
                               boolType)
                         : GoldenTensorClip(
                               a, args.shapeA, emptyValues, emptyShape, boundValues, args.shapeB, outShape, false, true,
                               boolType);
            out.assign(golden.size(), T{});
            if (CreateTensor(out, outShape, dtype, &outTensor) != ACL_SUCCESS) {
                throw std::runtime_error("failed to create " + args.api + " output tensor");
            }
            auto ret = getWorkspaceSize(aTensor.tensor, bTensor.tensor, outTensor.tensor, &workspaceSize, &executor);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error(getWorkspaceSizeName + " failed: " + std::to_string(ret));
            }
            if (workspaceSize > 0) {
                ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                if (ret != ACL_SUCCESS) {
                    throw std::runtime_error("workspace allocation failed");
                }
            }
            ret = launch(workspace, workspaceSize, executor, stream);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error(launchName + " failed: " + std::to_string(ret));
            }
        } else if (args.api == "Permute") {
            auto getWorkspaceSize = ResolveSymbol<PermuteGetWorkspaceSizeFn>("aclnnPermuteGetWorkspaceSize");
            auto launch = ResolveSymbol<PermuteLaunchFn>("aclnnPermute");
            const auto permuteDims = NormalizePermuteDims(args.shapeA, args.dims);
            outShape = PermuteOutputShape(args.shapeA, permuteDims);
            golden = GoldenPermute(a, args.shapeA, permuteDims, outShape, boolType);
            out.assign(golden.size(), T{});
            if (CreateTensor(out, outShape, dtype, &outTensor) != ACL_SUCCESS) {
                throw std::runtime_error("failed to create Permute output tensor");
            }
            dims = aclCreateIntArray(permuteDims.empty() ? nullptr : permuteDims.data(), permuteDims.size());
            if (dims == nullptr) {
                throw std::runtime_error("failed to create Permute dims");
            }
            auto ret = getWorkspaceSize(aTensor.tensor, dims, outTensor.tensor, &workspaceSize, &executor);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclnnPermuteGetWorkspaceSize failed: " + std::to_string(ret));
            }
            if (workspaceSize > 0) {
                ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                if (ret != ACL_SUCCESS) {
                    throw std::runtime_error("workspace allocation failed");
                }
            }
            ret = launch(workspace, workspaceSize, executor, stream);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclnnPermute failed: " + std::to_string(ret));
            }
        } else if (args.api == "aclnnChannelShuffle") {
            auto getWorkspaceSize =
                ResolveSymbol<ChannelShuffleGetWorkspaceSizeFn>("aclnnChannelShuffleGetWorkspaceSize");
            auto launch = ResolveSymbol<ChannelShuffleLaunchFn>("aclnnChannelShuffle");
            golden = GoldenChannelShuffle(a, args.shapeA, args.groups, boolType);
            out.assign(golden.size(), T{});
            if (CreateTensor(out, outShape, dtype, &outTensor) != ACL_SUCCESS) {
                throw std::runtime_error("failed to create aclnnChannelShuffle output tensor");
            }
            auto ret = getWorkspaceSize(aTensor.tensor, args.groups, outTensor.tensor, &workspaceSize, &executor);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclnnChannelShuffleGetWorkspaceSize failed: " + std::to_string(ret));
            }
            if (workspaceSize > 0) {
                ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                if (ret != ACL_SUCCESS) {
                    throw std::runtime_error("workspace allocation failed");
                }
            }
            ret = launch(workspace, workspaceSize, executor, stream);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclnnChannelShuffle failed: " + std::to_string(ret));
            }
        } else if (args.api == "Neg") {
            RunUnaryApi(
                "aclnnNegGetWorkspaceSize", "aclnnNeg", a, args.shapeA, dtype, boolType, stream, aTensor, outTensor,
                workspace, out, golden, [](double value) { return -value; });
        } else if (args.api == "Amax" || args.api == "Amin") {
            const auto reduceDims = NormalizeReduceDims(args.api, args.shapeA, args.dims, false);
            outShape = ReduceOutputShape(args.shapeA, reduceDims, args.keepDims);
            const auto kind = args.api == "Amin" ? ReduceKind::Min : ReduceKind::Max;
            golden = GoldenReduce(a, args.shapeA, reduceDims, outShape, args.keepDims, boolType, kind);
            out.assign(golden.size(), T{});
            if (CreateTensor(out, outShape, dtype, &outTensor) != ACL_SUCCESS) {
                throw std::runtime_error("failed to create " + args.api + " output tensor");
            }
            dims = aclCreateIntArray(reduceDims.empty() ? nullptr : reduceDims.data(), reduceDims.size());
            if (dims == nullptr) {
                throw std::runtime_error("failed to create " + args.api + " dims");
            }
            if (args.api == "Amin") {
                auto getWorkspaceSize = ResolveSymbol<AminGetWorkspaceSizeFn>("aclnnAminGetWorkspaceSize");
                auto launch = ResolveSymbol<AminLaunchFn>("aclnnAmin");
                auto ret = getWorkspaceSize(aTensor.tensor, dims, args.keepDims, outTensor.tensor, &workspaceSize, &executor);
                if (ret != ACL_SUCCESS) {
                    throw std::runtime_error("aclnnAminGetWorkspaceSize failed: " + std::to_string(ret));
                }
                if (workspaceSize > 0) {
                    ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                    if (ret != ACL_SUCCESS) {
                        throw std::runtime_error("workspace allocation failed");
                    }
                }
                ret = launch(workspace, workspaceSize, executor, stream);
                if (ret != ACL_SUCCESS) {
                    throw std::runtime_error("aclnnAmin failed: " + std::to_string(ret));
                }
            } else {
                auto getWorkspaceSize = ResolveSymbol<AmaxGetWorkspaceSizeFn>("aclnnAmaxGetWorkspaceSize");
                auto launch = ResolveSymbol<AmaxLaunchFn>("aclnnAmax");
                auto ret = getWorkspaceSize(aTensor.tensor, dims, args.keepDims, outTensor.tensor, &workspaceSize, &executor);
                if (ret != ACL_SUCCESS) {
                    throw std::runtime_error("aclnnAmaxGetWorkspaceSize failed: " + std::to_string(ret));
                }
                if (workspaceSize > 0) {
                    ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                    if (ret != ACL_SUCCESS) {
                        throw std::runtime_error("workspace allocation failed");
                    }
                }
                ret = launch(workspace, workspaceSize, executor, stream);
                if (ret != ACL_SUCCESS) {
                    throw std::runtime_error("aclnnAmax failed: " + std::to_string(ret));
                }
            }
        } else if (args.api == "Max" || args.api == "Min") {
            const auto kind = args.api == "Min" ? ReduceKind::Min : ReduceKind::Max;
            outShape = {1};
            golden = GoldenAllReduce(a, boolType, kind);
            out.assign(golden.size(), T{});
            if (CreateTensor(out, outShape, dtype, &outTensor) != ACL_SUCCESS) {
                throw std::runtime_error("failed to create " + args.api + " output tensor");
            }
            if (args.api == "Min") {
                auto getWorkspaceSize = ResolveSymbol<MinGetWorkspaceSizeFn>("aclnnMinGetWorkspaceSize");
                auto launch = ResolveSymbol<MinLaunchFn>("aclnnMin");
                auto ret = getWorkspaceSize(aTensor.tensor, outTensor.tensor, &workspaceSize, &executor);
                if (ret != ACL_SUCCESS) {
                    throw std::runtime_error("aclnnMinGetWorkspaceSize failed: " + std::to_string(ret));
                }
                if (workspaceSize > 0) {
                    ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                    if (ret != ACL_SUCCESS) {
                        throw std::runtime_error("workspace allocation failed");
                    }
                }
                ret = launch(workspace, workspaceSize, executor, stream);
                if (ret != ACL_SUCCESS) {
                    throw std::runtime_error("aclnnMin failed: " + std::to_string(ret));
                }
            } else {
                auto getWorkspaceSize = ResolveSymbol<MaxGetWorkspaceSizeFn>("aclnnMaxGetWorkspaceSize");
                auto launch = ResolveSymbol<MaxLaunchFn>("aclnnMax");
                auto ret = getWorkspaceSize(aTensor.tensor, outTensor.tensor, &workspaceSize, &executor);
                if (ret != ACL_SUCCESS) {
                    throw std::runtime_error("aclnnMaxGetWorkspaceSize failed: " + std::to_string(ret));
                }
                if (workspaceSize > 0) {
                    ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                    if (ret != ACL_SUCCESS) {
                        throw std::runtime_error("workspace allocation failed");
                    }
                }
                ret = launch(workspace, workspaceSize, executor, stream);
                if (ret != ACL_SUCCESS) {
                    throw std::runtime_error("aclnnMax failed: " + std::to_string(ret));
                }
            }
        } else if (args.api == "Prod") {
            if (!args.dtypeAttr.empty() && args.dtypeAttr != args.dtype) {
                throw std::runtime_error("Prod dtype_attr different from tensor dtype is not supported");
            }
            auto getWorkspaceSize = ResolveSymbol<ProdGetWorkspaceSizeFn>("aclnnProdGetWorkspaceSize");
            auto launch = ResolveSymbol<ProdLaunchFn>("aclnnProd");
            outShape = {1};
            golden = GoldenAllReduce(a, boolType, ReduceKind::Prod);
            out.assign(golden.size(), T{});
            if (CreateTensor(out, outShape, dtype, &outTensor) != ACL_SUCCESS) {
                throw std::runtime_error("failed to create Prod output tensor");
            }
            auto ret = getWorkspaceSize(aTensor.tensor, dtype, outTensor.tensor, &workspaceSize, &executor);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclnnProdGetWorkspaceSize failed: " + std::to_string(ret));
            }
            if (workspaceSize > 0) {
                ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                if (ret != ACL_SUCCESS) {
                    throw std::runtime_error("workspace allocation failed");
                }
            }
            ret = launch(workspace, workspaceSize, executor, stream);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclnnProd failed: " + std::to_string(ret));
            }
        } else if (args.api == "ProdDim") {
            if (!args.dtypeAttr.empty() && args.dtypeAttr != args.dtype) {
                throw std::runtime_error("ProdDim dtype_attr different from tensor dtype is not supported");
            }
            if (args.dims.size() != 1) {
                throw std::runtime_error("ProdDim expects exactly 1 dim");
            }
            auto getWorkspaceSize = ResolveSymbol<ProdDimGetWorkspaceSizeFn>("aclnnProdDimGetWorkspaceSize");
            auto launch = ResolveSymbol<ProdDimLaunchFn>("aclnnProdDim");
            const auto reduceDims = NormalizeReduceDims(args.api, args.shapeA, args.dims, false);
            if (reduceDims.size() != 1) {
                throw std::runtime_error("ProdDim expects exactly 1 dim");
            }
            outShape = ReduceOutputShape(args.shapeA, reduceDims, args.keepDims);
            golden = GoldenReduce(a, args.shapeA, reduceDims, outShape, args.keepDims, boolType, ReduceKind::Prod);
            out.assign(golden.size(), T{});
            if (CreateTensor(out, outShape, dtype, &outTensor) != ACL_SUCCESS) {
                throw std::runtime_error("failed to create ProdDim output tensor");
            }
            auto ret = getWorkspaceSize(
                aTensor.tensor, reduceDims[0], args.keepDims, dtype, outTensor.tensor, &workspaceSize, &executor);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclnnProdDimGetWorkspaceSize failed: " + std::to_string(ret));
            }
            if (workspaceSize > 0) {
                ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                if (ret != ACL_SUCCESS) {
                    throw std::runtime_error("workspace allocation failed");
                }
            }
            ret = launch(workspace, workspaceSize, executor, stream);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclnnProdDim failed: " + std::to_string(ret));
            }
        } else if (args.api == "ReduceSum") {
            if (!args.dtypeAttr.empty() && args.dtypeAttr != args.dtype) {
                throw std::runtime_error("ReduceSum dtype_attr different from tensor dtype is not supported");
            }
            auto getWorkspaceSize = ResolveSymbol<ReduceSumGetWorkspaceSizeFn>("aclnnReduceSumGetWorkspaceSize");
            auto launch = ResolveSymbol<ReduceSumLaunchFn>("aclnnReduceSum");
            const auto reduceDims = NormalizeReduceDims(args.api, args.shapeA, args.dims, args.noopWithEmptyAxes);
            outShape = ReduceOutputShape(args.shapeA, reduceDims, args.keepDims);
            golden = GoldenReduceSum(a, args.shapeA, reduceDims, outShape, args.keepDims, boolType);
            out.assign(golden.size(), T{});
            if (CreateTensor(out, outShape, dtype, &outTensor) != ACL_SUCCESS) {
                throw std::runtime_error("failed to create ReduceSum output tensor");
            }
            dims = aclCreateIntArray(reduceDims.empty() ? nullptr : reduceDims.data(), reduceDims.size());
            if (dims == nullptr) {
                throw std::runtime_error("failed to create ReduceSum dims");
            }
            auto ret =
                getWorkspaceSize(aTensor.tensor, dims, args.keepDims, dtype, outTensor.tensor, &workspaceSize, &executor);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclnnReduceSumGetWorkspaceSize failed");
            }
            if (workspaceSize > 0) {
                ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                if (ret != ACL_SUCCESS) {
                    throw std::runtime_error("workspace allocation failed");
                }
            }
            ret = launch(workspace, workspaceSize, executor, stream);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclnnReduceSum failed");
            }
        } else if (args.api == "UnfoldGrad") {
            auto getWorkspaceSize = ResolveSymbol<UnfoldGradGetWorkspaceSizeFn>("aclnnUnfoldGradGetWorkspaceSize");
            auto launch = ResolveSymbol<UnfoldGradLaunchFn>("aclnnUnfoldGrad");
            const int64_t axis = NormalizeUnfoldGradDim(args.inputSizes, args.dim);
            outShape = args.inputSizes;
            golden = GoldenUnfoldGrad(a, args.shapeA, args.inputSizes, axis, args.size, args.step, boolType);
            out.assign(golden.size(), T{});
            if (CreateTensor(out, outShape, dtype, &outTensor) != ACL_SUCCESS) {
                throw std::runtime_error("failed to create UnfoldGrad output tensor");
            }
            inputSizesArray =
                aclCreateIntArray(args.inputSizes.empty() ? nullptr : args.inputSizes.data(), args.inputSizes.size());
            if (inputSizesArray == nullptr) {
                throw std::runtime_error("failed to create UnfoldGrad input_sizes");
            }
            auto ret = getWorkspaceSize(
                aTensor.tensor, inputSizesArray, axis, args.size, args.step, outTensor.tensor, &workspaceSize, &executor);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclnnUnfoldGradGetWorkspaceSize failed: " + std::to_string(ret));
            }
            if (workspaceSize > 0) {
                ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                if (ret != ACL_SUCCESS) {
                    throw std::runtime_error("workspace allocation failed");
                }
            }
            ret = launch(workspace, workspaceSize, executor, stream);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclnnUnfoldGrad failed: " + std::to_string(ret));
            }
        } else if (args.api == "Reciprocal") {
            RunUnaryApi(
                "aclnnReciprocalGetWorkspaceSize", "aclnnReciprocal", a, args.shapeA, dtype, boolType, stream,
                aTensor, outTensor, workspace, out, golden, [](double value) { return 1.0 / value; });
        } else if (args.api == "Round") {
            RunUnaryApi(
                "aclnnRoundGetWorkspaceSize", "aclnnRound", a, args.shapeA, dtype, boolType, stream, aTensor,
                outTensor, workspace, out, golden, [](double value) { return std::nearbyint(value); });
        } else if (args.api == "Rsqrt") {
            RunUnaryApi(
                "aclnnRsqrtGetWorkspaceSize", "aclnnRsqrt", a, args.shapeA, dtype, boolType, stream, aTensor,
                outTensor, workspace, out, golden, [](double value) { return 1.0 / std::sqrt(value); });
        } else if (args.api == "Sin") {
            RunUnaryApi(
                "aclnnSinGetWorkspaceSize", "aclnnSin", a, args.shapeA, dtype, boolType, stream, aTensor, outTensor,
                workspace, out, golden, [](double value) { return std::sin(value); });
        } else if (args.api == "Sign") {
            RunUnaryApi(
                "aclnnSignGetWorkspaceSize", "aclnnSign", a, args.shapeA, dtype, boolType, stream, aTensor,
                outTensor, workspace, out, golden, [](double value) { return (value > 0.0) - (value < 0.0); });
        } else if (args.api == "Sqrt") {
            RunUnaryApi(
                "aclnnSqrtGetWorkspaceSize", "aclnnSqrt", a, args.shapeA, dtype, boolType, stream, aTensor, outTensor,
                workspace, out, golden, [](double value) { return std::sqrt(value); });
        } else if (args.api == "Sub") {
            auto getWorkspaceSize = ResolveSymbol<SubGetWorkspaceSizeFn>("aclnnSubGetWorkspaceSize");
            auto launch = ResolveSymbol<SubLaunchFn>("aclnnSub");
            auto b = MakeData<T>(args.shapeB, args.rangeBLow, args.rangeBHigh, boolType);
            outShape = BroadcastShape(args.shapeA, args.shapeB);
            const double alphaValue = ScalarGoldenValue<T>(args.alpha, boolType);
            golden = GoldenSub(a, args.shapeA, b, args.shapeB, outShape, alphaValue, boolType);
            out.assign(golden.size(), T{});
            if (CreateTensor(b, args.shapeB, dtype, &bTensor) != ACL_SUCCESS ||
                CreateTensor(out, outShape, dtype, &outTensor) != ACL_SUCCESS) {
                throw std::runtime_error("failed to create Sub tensors");
            }
            alpha = CreateScalar<T>(args.alpha, dtype, boolType);
            auto ret = getWorkspaceSize(aTensor.tensor, bTensor.tensor, alpha, outTensor.tensor, &workspaceSize, &executor);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclnnSubGetWorkspaceSize failed");
            }
            if (workspaceSize > 0) {
                ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                if (ret != ACL_SUCCESS) {
                    throw std::runtime_error("workspace allocation failed");
                }
            }
            ret = launch(workspace, workspaceSize, executor, stream);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclnnSub failed");
            }
        } else if (args.api == "Subs") {
            auto getWorkspaceSize = ResolveSymbol<SubsGetWorkspaceSizeFn>("aclnnSubsGetWorkspaceSize");
            auto launch = ResolveSymbol<SubsLaunchFn>("aclnnSubs");
            const double otherValue = ScalarGoldenValue<T>(args.other, boolType);
            const double alphaValue = ScalarGoldenValue<T>(args.alpha, boolType);
            golden = GoldenSubs(a, otherValue, alphaValue, boolType);
            out.assign(golden.size(), T{});
            if (CreateTensor(out, outShape, dtype, &outTensor) != ACL_SUCCESS) {
                throw std::runtime_error("failed to create Subs output tensor");
            }
            other = CreateScalar<T>(args.other, dtype, boolType);
            alpha = CreateScalar<T>(args.alpha, dtype, boolType);
            auto ret = getWorkspaceSize(aTensor.tensor, other, alpha, outTensor.tensor, &workspaceSize, &executor);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclnnSubsGetWorkspaceSize failed");
            }
            if (workspaceSize > 0) {
                ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                if (ret != ACL_SUCCESS) {
                    throw std::runtime_error("workspace allocation failed");
                }
            }
            ret = launch(workspace, workspaceSize, executor, stream);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclnnSubs failed");
            }
        } else if (args.api == "Tanh") {
            RunUnaryApi(
                "aclnnTanhGetWorkspaceSize", "aclnnTanh", a, args.shapeA, dtype, boolType, stream, aTensor, outTensor,
                workspace, out, golden, [](double value) { return std::tanh(value); });
        } else if (args.api == "Trunc") {
            RunUnaryApi(
                "aclnnTruncGetWorkspaceSize", "aclnnTrunc", a, args.shapeA, dtype, boolType, stream, aTensor,
                outTensor, workspace, out, golden, [](double value) { return std::trunc(value); });
        } else {
            throw std::runtime_error("unsupported api");
        }

        auto ret = aclrtSynchronizeStream(stream);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error("aclrtSynchronizeStream failed");
        }

        if (hasSecondOutput) {
            std::vector<T> actual(firstOutputSize);
            const size_t firstBytes = std::max<size_t>(actual.size() * sizeof(T), sizeof(T));
            if (!actual.empty()) {
                ret = aclrtMemcpy(actual.data(), firstBytes, outTensor.deviceAddr, firstBytes, ACL_MEMCPY_DEVICE_TO_HOST);
                if (ret != ACL_SUCCESS) {
                    throw std::runtime_error("aclrtMemcpy VarMean var output failed");
                }
            }
            std::vector<T> actualSecond(golden.size() - firstOutputSize);
            const size_t secondBytes = std::max<size_t>(actualSecond.size() * sizeof(T), sizeof(T));
            if (!actualSecond.empty()) {
                ret = aclrtMemcpy(
                    actualSecond.data(), secondBytes, cTensor.deviceAddr, secondBytes, ACL_MEMCPY_DEVICE_TO_HOST);
                if (ret != ACL_SUCCESS) {
                    throw std::runtime_error("aclrtMemcpy VarMean mean output failed");
                }
            }
            actual.insert(actual.end(), actualSecond.begin(), actualSecond.end());
            result = CompareResult(actual, golden, boolType);
        } else {
            std::vector<T> actual(golden.size());
            const size_t bytes = std::max<size_t>(actual.size() * sizeof(T), sizeof(T));
            if (!actual.empty()) {
                ret = aclrtMemcpy(actual.data(), bytes, outTensor.deviceAddr, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
                if (ret != ACL_SUCCESS) {
                    throw std::runtime_error("aclrtMemcpy device to host failed");
                }
            }
            result = CompareResult(actual, golden, boolType);
        }
    } catch (const std::exception &exc) {
        result.status = "failed";
        result.message = exc.what();
        result.retCode = 1;
    }

    if (alpha != nullptr) {
        aclDestroyScalar(alpha);
    }
    if (other != nullptr) {
        aclDestroyScalar(other);
    }
    if (dims != nullptr) {
        aclDestroyIntArray(dims);
    }
    if (beginArray != nullptr) {
        aclDestroyIntArray(beginArray);
    }
    if (endArray != nullptr) {
        aclDestroyIntArray(endArray);
    }
    if (stridesArray != nullptr) {
        aclDestroyIntArray(stridesArray);
    }
    if (inputSizesArray != nullptr) {
        aclDestroyIntArray(inputSizesArray);
    }
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    outTensor.Cleanup();
    cTensor.Cleanup();
    bTensor.Cleanup();
    aTensor.Cleanup();
    return result;
}

template <typename T>
NativeResult RunTensorListTypedWithStream(const Args &args, aclDataType dtype, bool boolType, aclrtStream stream)
{
    std::vector<TensorHolder> inputTensors(args.tensorShapes.size());
    TensorHolder outTensor;
    aclTensorList *tensorList = nullptr;
    void *workspace = nullptr;
    NativeResult result;

    try {
        if (args.tensorShapes.empty()) {
            throw std::runtime_error(args.api + " expects a non-empty TensorList input");
        }
        if (!args.tensorRanges.empty() && args.tensorRanges.size() != args.tensorShapes.size()) {
            throw std::runtime_error(args.api + " TensorList ranges are invalid");
        }

        int64_t axis = 0;
        std::vector<int64_t> outShape;
        if (args.api == "Cat") {
            axis = NormalizeCatDim(args.tensorShapes, args.dim);
            outShape = CatOutputShape(args.tensorShapes, axis);
        } else if (args.api == "Stack") {
            axis = NormalizeStackDim(args.tensorShapes, args.dim);
            outShape = StackOutputShape(args.tensorShapes, axis);
        } else {
            throw std::runtime_error("unsupported TensorList api");
        }

        std::vector<std::vector<T>> inputs;
        std::vector<const aclTensor *> tensorPtrs;
        inputs.reserve(args.tensorShapes.size());
        tensorPtrs.reserve(args.tensorShapes.size());
        for (size_t index = 0; index < args.tensorShapes.size(); ++index) {
            const auto range = args.tensorRanges.empty() ? std::make_pair(-1.0, 1.0) : args.tensorRanges[index];
            inputs.push_back(MakeData<T>(args.tensorShapes[index], range.first, range.second, boolType));
            if (CreateTensor(inputs.back(), args.tensorShapes[index], dtype, &inputTensors[index]) != ACL_SUCCESS) {
                throw std::runtime_error("failed to create " + args.api + " input tensor");
            }
            tensorPtrs.push_back(inputTensors[index].tensor);
        }

        std::vector<double> golden;
        if (args.api == "Cat") {
            golden = GoldenCat(inputs, args.tensorShapes, outShape, axis, boolType);
        } else {
            golden = GoldenStack(inputs, args.tensorShapes, outShape, axis, boolType);
        }

        std::vector<T> out(golden.size(), T{});
        if (CreateTensor(out, outShape, dtype, &outTensor) != ACL_SUCCESS) {
            throw std::runtime_error("failed to create " + args.api + " output tensor");
        }

        tensorList = aclCreateTensorList(tensorPtrs.data(), tensorPtrs.size());
        if (tensorList == nullptr) {
            throw std::runtime_error("failed to create " + args.api + " TensorList");
        }

        auto getWorkspaceSize = args.api == "Cat"
                                    ? ResolveSymbol<TensorListGetWorkspaceSizeFn>("aclnnCatGetWorkspaceSize")
                                    : ResolveSymbol<TensorListGetWorkspaceSizeFn>("aclnnStackGetWorkspaceSize");
        auto launch = args.api == "Cat" ? ResolveSymbol<TensorListLaunchFn>("aclnnCat")
                                        : ResolveSymbol<TensorListLaunchFn>("aclnnStack");
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        auto ret = getWorkspaceSize(tensorList, args.dim, outTensor.tensor, &workspaceSize, &executor);
        if (ret != ACL_SUCCESS) {
            const auto name = args.api == "Cat" ? "aclnnCatGetWorkspaceSize" : "aclnnStackGetWorkspaceSize";
            throw std::runtime_error(std::string(name) + " failed: " + std::to_string(ret));
        }
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("workspace allocation failed");
            }
        }
        ret = launch(workspace, workspaceSize, executor, stream);
        if (ret != ACL_SUCCESS) {
            const auto name = args.api == "Cat" ? "aclnnCat" : "aclnnStack";
            throw std::runtime_error(std::string(name) + " failed: " + std::to_string(ret));
        }
        ret = aclrtSynchronizeStream(stream);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error("aclrtSynchronizeStream failed");
        }

        std::vector<T> actual(golden.size());
        const size_t bytes = std::max<size_t>(actual.size() * sizeof(T), sizeof(T));
        if (!actual.empty()) {
            ret = aclrtMemcpy(actual.data(), bytes, outTensor.deviceAddr, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclrtMemcpy device to host failed");
            }
        }
        result = CompareResult(actual, golden, boolType);
    } catch (const std::exception &exc) {
        result.status = "failed";
        result.message = exc.what();
        result.retCode = 1;
    }

    if (tensorList != nullptr) {
        aclDestroyTensorList(tensorList);
    }
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    outTensor.Cleanup();
    for (auto &holder : inputTensors) {
        holder.Cleanup();
    }
    return result;
}

template <typename T>
NativeResult RunChunkCatTypedWithStream(const Args &args, aclDataType dtype, bool boolType, aclrtStream stream)
{
    std::vector<TensorHolder> inputTensors(args.tensorShapes.size());
    TensorHolder outTensor;
    aclTensorList *tensorList = nullptr;
    void *workspace = nullptr;
    NativeResult result;

    try {
        if (args.tensorShapes.empty()) {
            throw std::runtime_error("ChunkCat expects a non-empty TensorList input");
        }
        if (!args.tensorRanges.empty() && args.tensorRanges.size() != args.tensorShapes.size()) {
            throw std::runtime_error("ChunkCat TensorList ranges are invalid");
        }

        const auto outShape = ChunkCatOutputShape(args.tensorShapes, args.dim, args.numChunks);

        std::vector<std::vector<T>> inputs;
        std::vector<const aclTensor *> tensorPtrs;
        inputs.reserve(args.tensorShapes.size());
        tensorPtrs.reserve(args.tensorShapes.size());
        for (size_t index = 0; index < args.tensorShapes.size(); ++index) {
            const auto range = args.tensorRanges.empty() ? std::make_pair(-1.0, 1.0) : args.tensorRanges[index];
            inputs.push_back(MakeData<T>(args.tensorShapes[index], range.first, range.second, boolType));
            if (CreateTensor(inputs.back(), args.tensorShapes[index], dtype, &inputTensors[index]) != ACL_SUCCESS) {
                throw std::runtime_error("failed to create ChunkCat input tensor");
            }
            tensorPtrs.push_back(inputTensors[index].tensor);
        }

        const auto golden = GoldenChunkCat(inputs, args.tensorShapes, outShape, args.numChunks, boolType);
        std::vector<T> out(golden.size(), T{});
        if (CreateTensor(out, outShape, dtype, &outTensor) != ACL_SUCCESS) {
            throw std::runtime_error("failed to create ChunkCat output tensor");
        }

        tensorList = aclCreateTensorList(tensorPtrs.data(), tensorPtrs.size());
        if (tensorList == nullptr) {
            throw std::runtime_error("failed to create ChunkCat TensorList");
        }

        auto getWorkspaceSize = ResolveSymbol<ChunkCatGetWorkspaceSizeFn>("aclnnChunkCatGetWorkspaceSize");
        auto launch = ResolveSymbol<TensorListLaunchFn>("aclnnChunkCat");
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        auto ret = getWorkspaceSize(tensorList, args.dim, args.numChunks, outTensor.tensor, &workspaceSize, &executor);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error("aclnnChunkCatGetWorkspaceSize failed: " + std::to_string(ret));
        }
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("workspace allocation failed");
            }
        }
        ret = launch(workspace, workspaceSize, executor, stream);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error("aclnnChunkCat failed: " + std::to_string(ret));
        }
        ret = aclrtSynchronizeStream(stream);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error("aclrtSynchronizeStream failed");
        }

        std::vector<T> actual(golden.size());
        const size_t bytes = std::max<size_t>(actual.size() * sizeof(T), sizeof(T));
        if (!actual.empty()) {
            ret = aclrtMemcpy(actual.data(), bytes, outTensor.deviceAddr, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclrtMemcpy device to host failed");
            }
        }
        result = CompareResult(actual, golden, boolType);
    } catch (const std::exception &exc) {
        result.status = "failed";
        result.message = exc.what();
        result.retCode = 1;
    }

    if (tensorList != nullptr) {
        aclDestroyTensorList(tensorList);
    }
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    outTensor.Cleanup();
    for (auto &holder : inputTensors) {
        holder.Cleanup();
    }
    return result;
}

template <typename T>
NativeResult RunSplitWithSizeTypedWithStream(const Args &args, aclDataType dtype, bool boolType, aclrtStream stream)
{
    TensorHolder inputTensor;
    std::vector<TensorHolder> outputTensors;
    aclIntArray *splitSizeArray = nullptr;
    aclTensorList *outTensorList = nullptr;
    void *workspace = nullptr;
    NativeResult result;

    try {
        const int64_t axis = NormalizeSplitWithSizeDim(args.shapeA, args.dim);
        const auto outputShapes = SplitWithSizeOutputShapes(args.shapeA, args.splitSizes, axis);
        auto input = MakeData<T>(args.shapeA, args.rangeALow, args.rangeAHigh, boolType);
        if (CreateTensor(input, args.shapeA, dtype, &inputTensor) != ACL_SUCCESS) {
            throw std::runtime_error("failed to create SplitWithSize input tensor");
        }

        const auto golden = GoldenSplitWithSize(input, args.shapeA, outputShapes, axis, boolType);
        outputTensors.resize(outputShapes.size());
        std::vector<std::vector<T>> outputHosts;
        std::vector<aclTensor *> outputPtrs;
        outputHosts.reserve(outputShapes.size());
        outputPtrs.reserve(outputShapes.size());
        for (size_t index = 0; index < outputShapes.size(); ++index) {
            outputHosts.emplace_back(static_cast<size_t>(Numel(outputShapes[index])), T{});
            if (CreateTensor(outputHosts.back(), outputShapes[index], dtype, &outputTensors[index]) != ACL_SUCCESS) {
                throw std::runtime_error("failed to create SplitWithSize output tensor");
            }
            outputPtrs.push_back(outputTensors[index].tensor);
        }

        splitSizeArray = aclCreateIntArray(args.splitSizes.data(), args.splitSizes.size());
        if (splitSizeArray == nullptr) {
            throw std::runtime_error("failed to create SplitWithSize splitSize array");
        }
        outTensorList = aclCreateTensorList(outputPtrs.data(), outputPtrs.size());
        if (outTensorList == nullptr) {
            throw std::runtime_error("failed to create SplitWithSize output TensorList");
        }

        auto getWorkspaceSize =
            ResolveSymbol<SplitWithSizeGetWorkspaceSizeFn>("aclnnSplitWithSizeGetWorkspaceSize");
        auto launch = ResolveSymbol<SplitWithSizeLaunchFn>("aclnnSplitWithSize");
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        auto ret = getWorkspaceSize(inputTensor.tensor, splitSizeArray, args.dim, outTensorList, &workspaceSize, &executor);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error("aclnnSplitWithSizeGetWorkspaceSize failed: " + std::to_string(ret));
        }
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("workspace allocation failed");
            }
        }
        ret = launch(workspace, workspaceSize, executor, stream);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error("aclnnSplitWithSize failed: " + std::to_string(ret));
        }
        ret = aclrtSynchronizeStream(stream);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error("aclrtSynchronizeStream failed");
        }

        std::vector<T> actual;
        actual.reserve(golden.size());
        for (size_t index = 0; index < outputShapes.size(); ++index) {
            std::vector<T> out(static_cast<size_t>(Numel(outputShapes[index])));
            if (!out.empty()) {
                const size_t bytes = out.size() * sizeof(T);
                ret = aclrtMemcpy(out.data(), bytes, outputTensors[index].deviceAddr, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
                if (ret != ACL_SUCCESS) {
                    throw std::runtime_error("aclrtMemcpy SplitWithSize output failed");
                }
            }
            actual.insert(actual.end(), out.begin(), out.end());
        }
        result = CompareResult(actual, golden, boolType);
    } catch (const std::exception &exc) {
        result.status = "failed";
        result.message = exc.what();
        result.retCode = 1;
    }

    if (outTensorList != nullptr) {
        aclDestroyTensorList(outTensorList);
    }
    if (splitSizeArray != nullptr) {
        aclDestroyIntArray(splitSizeArray);
    }
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    for (auto &holder : outputTensors) {
        holder.Cleanup();
    }
    inputTensor.Cleanup();
    return result;
}

template <typename T>
NativeResult RunMaskedSelectTypedWithStream(const Args &args, aclDataType dtype, bool boolType, aclrtStream stream)
{
    TensorHolder inputTensor;
    TensorHolder maskTensor;
    TensorHolder outTensor;
    void *workspace = nullptr;
    NativeResult result;

    try {
        const auto outShape = BroadcastShape(args.shapeA, args.maskShape);
        const int64_t outputNumel = Numel(outShape);
        auto input = MakeData<T>(args.shapeA, args.rangeALow, args.rangeAHigh, boolType);
        auto mask = MakeMaskData(args.maskShape);
        auto golden = GoldenMaskedSelect(input, args.shapeA, mask, args.maskShape, outShape, boolType);
        std::vector<T> out(static_cast<size_t>(outputNumel), T{});

        if (CreateTensor(input, args.shapeA, dtype, &inputTensor) != ACL_SUCCESS ||
            CreateTensor(mask, args.maskShape, aclDataType::ACL_BOOL, &maskTensor) != ACL_SUCCESS ||
            CreateTensor(out, {outputNumel}, dtype, &outTensor) != ACL_SUCCESS) {
            throw std::runtime_error("failed to create MaskedSelect tensors");
        }

        auto getWorkspaceSize = ResolveSymbol<MaskedSelectGetWorkspaceSizeFn>("aclnnMaskedSelectGetWorkspaceSize");
        auto launch = ResolveSymbol<MaskedSelectLaunchFn>("aclnnMaskedSelect");
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        auto ret =
            getWorkspaceSize(inputTensor.tensor, maskTensor.tensor, outTensor.tensor, &workspaceSize, &executor);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error("aclnnMaskedSelectGetWorkspaceSize failed: " + std::to_string(ret));
        }
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("workspace allocation failed");
            }
        }
        ret = launch(workspace, workspaceSize, executor, stream);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error("aclnnMaskedSelect failed: " + std::to_string(ret));
        }
        ret = aclrtSynchronizeStream(stream);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error("aclrtSynchronizeStream failed");
        }

        std::vector<T> actual(golden.size());
        const size_t bytes = std::max<size_t>(actual.size() * sizeof(T), sizeof(T));
        if (!actual.empty()) {
            ret = aclrtMemcpy(actual.data(), bytes, outTensor.deviceAddr, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclrtMemcpy device to host failed");
            }
        }
        result = CompareResult(actual, golden, boolType);
    } catch (const std::exception &exc) {
        result.status = "failed";
        result.message = exc.what();
        result.retCode = 1;
    }

    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    outTensor.Cleanup();
    maskTensor.Cleanup();
    inputTensor.Cleanup();
    return result;
}

template <typename T>
NativeResult RunBoolUnaryTypedWithStream(const Args &args, aclDataType dtype, bool boolType, aclrtStream stream)
{
    TensorHolder inputTensor;
    TensorHolder outTensor;
    void *workspace = nullptr;
    NativeResult result;

    try {
        auto input = MakeData<T>(args.shapeA, args.rangeALow, args.rangeAHigh, boolType);
        std::vector<double> golden(input.size(), 0.0);
        for (size_t index = 0; index < input.size(); ++index) {
            const double value = ToDouble(input[index], boolType);
            const bool expected = args.api == "IsFinite" ? std::isfinite(value) : std::isinf(value);
            golden[index] = expected ? 1.0 : 0.0;
        }
        std::vector<uint8_t> out(golden.size(), 0);

        if (CreateTensor(input, args.shapeA, dtype, &inputTensor) != ACL_SUCCESS ||
            CreateTensor(out, args.shapeA, aclDataType::ACL_BOOL, &outTensor) != ACL_SUCCESS) {
            throw std::runtime_error("failed to create " + args.api + " tensors");
        }

        const char *getWorkspaceSymbol =
            args.api == "IsFinite" ? "aclnnIsFiniteGetWorkspaceSize" : "aclnnIsInfGetWorkspaceSize";
        const char *launchSymbol = args.api == "IsFinite" ? "aclnnIsFinite" : "aclnnIsInf";
        auto getWorkspaceSize = ResolveSymbol<BoolUnaryGetWorkspaceSizeFn>(getWorkspaceSymbol);
        auto launch = ResolveSymbol<BoolUnaryLaunchFn>(launchSymbol);
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        auto ret = getWorkspaceSize(inputTensor.tensor, outTensor.tensor, &workspaceSize, &executor);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error(std::string(getWorkspaceSymbol) + " failed: " + std::to_string(ret));
        }
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("workspace allocation failed");
            }
        }
        ret = launch(workspace, workspaceSize, executor, stream);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error(std::string(launchSymbol) + " failed: " + std::to_string(ret));
        }
        ret = aclrtSynchronizeStream(stream);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error("aclrtSynchronizeStream failed");
        }

        std::vector<uint8_t> actual(golden.size(), 0);
        const size_t bytes = std::max<size_t>(actual.size() * sizeof(uint8_t), sizeof(uint8_t));
        if (!actual.empty()) {
            ret = aclrtMemcpy(actual.data(), bytes, outTensor.deviceAddr, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclrtMemcpy device to host failed");
            }
        }
        result = CompareResult(actual, golden, true);
    } catch (const std::exception &exc) {
        result.status = "failed";
        result.message = exc.what();
        result.retCode = 1;
    }

    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    outTensor.Cleanup();
    inputTensor.Cleanup();
    return result;
}

NativeResult DispatchBoolUnaryWithStream(const Args &args, aclrtStream stream)
{
    if (args.dtype == "fp32") {
        return RunBoolUnaryTypedWithStream<float>(args, aclDataType::ACL_FLOAT, false, stream);
    }
    if (args.dtype == "fp16") {
        return RunBoolUnaryTypedWithStream<HalfStorage<HalfKind::Fp16>>(args, aclDataType::ACL_FLOAT16, false, stream);
    }
    if (args.dtype == "bf16") {
        return RunBoolUnaryTypedWithStream<HalfStorage<HalfKind::Bf16>>(args, aclDataType::ACL_BF16, false, stream);
    }
    NativeResult result;
    result.status = "failed";
    result.message = "unsupported dtype: " + args.dtype;
    result.retCode = 1;
    return result;
}

template <typename T>
int CopyDeviceToHost(const TensorHolder &holder, const std::vector<int64_t> &shape, std::vector<T> *out)
{
    out->assign(static_cast<size_t>(Numel(shape)), T{});
    if (out->empty()) {
        return ACL_SUCCESS;
    }
    const size_t bytes = out->size() * sizeof(T);
    return aclrtMemcpy(out->data(), bytes, holder.deviceAddr, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
}

void CleanupHolders(std::initializer_list<TensorHolder *> holders)
{
    for (auto *holder : holders) {
        if (holder != nullptr) {
            holder->Cleanup();
        }
    }
}

#include "attention_runner.inc"

template <typename TIn, typename TOut>
NativeResult RunCastTypedWithStream(
    const Args &args, aclDataType inputDtype, bool inputBoolType, aclDataType outputDtype, bool outputBoolType,
    aclrtStream stream)
{
    TensorHolder aTensor;
    TensorHolder outTensor;
    void *workspace = nullptr;
    NativeResult result;

    try {
        auto a = MakeData<TIn>(args.shapeA, args.rangeALow, args.rangeAHigh, inputBoolType);
        auto golden = GoldenCast<TIn, TOut>(a, inputBoolType, outputBoolType);
        std::vector<TOut> out(golden.size(), TOut{});

        if (CreateTensor(a, args.shapeA, inputDtype, &aTensor) != ACL_SUCCESS ||
            CreateTensor(out, args.shapeA, outputDtype, &outTensor) != ACL_SUCCESS) {
            throw std::runtime_error("failed to create Cast tensors");
        }

        auto getWorkspaceSize = ResolveSymbol<CastGetWorkspaceSizeFn>("aclnnCastGetWorkspaceSize");
        auto launch = ResolveSymbol<CastLaunchFn>("aclnnCast");
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        auto ret = getWorkspaceSize(aTensor.tensor, outputDtype, outTensor.tensor, &workspaceSize, &executor);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error("aclnnCastGetWorkspaceSize failed: " + std::to_string(ret));
        }
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("workspace allocation failed");
            }
        }
        ret = launch(workspace, workspaceSize, executor, stream);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error("aclnnCast failed: " + std::to_string(ret));
        }
        ret = aclrtSynchronizeStream(stream);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error("aclrtSynchronizeStream failed");
        }

        std::vector<TOut> actual(golden.size());
        const size_t bytes = std::max<size_t>(actual.size() * sizeof(TOut), sizeof(TOut));
        if (!actual.empty()) {
            ret = aclrtMemcpy(actual.data(), bytes, outTensor.deviceAddr, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclrtMemcpy device to host failed");
            }
        }
        result = CompareResult(actual, golden, outputBoolType);
    } catch (const std::exception &exc) {
        result.status = "failed";
        result.message = exc.what();
        result.retCode = 1;
    }

    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    outTensor.Cleanup();
    aTensor.Cleanup();
    return result;
}

template <typename TIn>
NativeResult DispatchCastOutput(
    const Args &args, aclDataType inputDtype, bool inputBoolType, aclrtStream stream)
{
    if (args.outDtype == "fp32") {
        return RunCastTypedWithStream<TIn, float>(
            args, inputDtype, inputBoolType, DtypeFromName(args.outDtype), false, stream);
    }
    if (args.outDtype == "fp64") {
        return RunCastTypedWithStream<TIn, double>(
            args, inputDtype, inputBoolType, DtypeFromName(args.outDtype), false, stream);
    }
    if (args.outDtype == "fp16") {
        return RunCastTypedWithStream<TIn, HalfStorage<HalfKind::Fp16>>(
            args, inputDtype, inputBoolType, DtypeFromName(args.outDtype), false, stream);
    }
    if (args.outDtype == "bf16") {
        return RunCastTypedWithStream<TIn, HalfStorage<HalfKind::Bf16>>(
            args, inputDtype, inputBoolType, DtypeFromName(args.outDtype), false, stream);
    }
    if (args.outDtype == "int32") {
        return RunCastTypedWithStream<TIn, int32_t>(
            args, inputDtype, inputBoolType, DtypeFromName(args.outDtype), false, stream);
    }
    if (args.outDtype == "int64") {
        return RunCastTypedWithStream<TIn, int64_t>(
            args, inputDtype, inputBoolType, DtypeFromName(args.outDtype), false, stream);
    }
    if (args.outDtype == "int16") {
        return RunCastTypedWithStream<TIn, int16_t>(
            args, inputDtype, inputBoolType, DtypeFromName(args.outDtype), false, stream);
    }
    if (args.outDtype == "int8") {
        return RunCastTypedWithStream<TIn, int8_t>(
            args, inputDtype, inputBoolType, DtypeFromName(args.outDtype), false, stream);
    }
    if (args.outDtype == "uint8") {
        return RunCastTypedWithStream<TIn, uint8_t>(
            args, inputDtype, inputBoolType, DtypeFromName(args.outDtype), false, stream);
    }
    if (args.outDtype == "bool") {
        return RunCastTypedWithStream<TIn, uint8_t>(
            args, inputDtype, inputBoolType, DtypeFromName(args.outDtype), true, stream);
    }
    NativeResult result;
    result.status = "failed";
    result.message = "unsupported Cast output dtype: " + args.outDtype;
    result.retCode = 1;
    return result;
}

NativeResult DispatchCastWithStream(const Args &args, aclrtStream stream)
{
    if (args.outDtype.empty()) {
        NativeResult result;
        result.status = "failed";
        result.message = "Cast out_dtype is required";
        result.retCode = 1;
        return result;
    }
    if (args.dtype == "fp32") {
        return DispatchCastOutput<float>(args, DtypeFromName(args.dtype), false, stream);
    }
    if (args.dtype == "fp64") {
        return DispatchCastOutput<double>(args, DtypeFromName(args.dtype), false, stream);
    }
    if (args.dtype == "fp16") {
        return DispatchCastOutput<HalfStorage<HalfKind::Fp16>>(args, DtypeFromName(args.dtype), false, stream);
    }
    if (args.dtype == "bf16") {
        return DispatchCastOutput<HalfStorage<HalfKind::Bf16>>(args, DtypeFromName(args.dtype), false, stream);
    }
    if (args.dtype == "int32") {
        return DispatchCastOutput<int32_t>(args, DtypeFromName(args.dtype), false, stream);
    }
    if (args.dtype == "int64") {
        return DispatchCastOutput<int64_t>(args, DtypeFromName(args.dtype), false, stream);
    }
    if (args.dtype == "int16") {
        return DispatchCastOutput<int16_t>(args, DtypeFromName(args.dtype), false, stream);
    }
    if (args.dtype == "int8") {
        return DispatchCastOutput<int8_t>(args, DtypeFromName(args.dtype), false, stream);
    }
    if (args.dtype == "uint8") {
        return DispatchCastOutput<uint8_t>(args, DtypeFromName(args.dtype), false, stream);
    }
    if (args.dtype == "bool") {
        return DispatchCastOutput<uint8_t>(args, DtypeFromName(args.dtype), true, stream);
    }
    NativeResult result;
    result.status = "failed";
    result.message = "unsupported dtype: " + args.dtype;
    result.retCode = 1;
    return result;
}

template <typename TIn, typename TOut>
NativeResult RunProdTypedWithStream(
    const Args &args, aclDataType inputDtype, bool inputBoolType, aclDataType outputDtype, bool outputBoolType,
    aclrtStream stream)
{
    TensorHolder aTensor;
    TensorHolder outTensor;
    void *workspace = nullptr;
    NativeResult result;

    try {
        auto a = MakeData<TIn>(args.shapeA, args.rangeALow, args.rangeAHigh, inputBoolType);
        std::vector<int64_t> outShape = {1};
        std::vector<double> golden;
        if (args.api == "ProdDim") {
            const auto reduceDims = NormalizeReduceDims(args.api, args.shapeA, args.dims, false);
            if (reduceDims.size() != 1) {
                throw std::runtime_error("ProdDim expects exactly 1 dim");
            }
            outShape = ReduceOutputShape(args.shapeA, reduceDims, args.keepDims);
            golden = GoldenProdDimWithOutputDtype<TIn, TOut>(
                a, args.shapeA, reduceDims, outShape, args.keepDims, inputBoolType, outputBoolType);
        } else {
            golden = GoldenProdWithOutputDtype<TIn, TOut>(a, inputBoolType, outputBoolType);
        }
        std::vector<TOut> out(golden.size(), TOut{});

        if (CreateTensor(a, args.shapeA, inputDtype, &aTensor) != ACL_SUCCESS ||
            CreateTensor(out, outShape, outputDtype, &outTensor) != ACL_SUCCESS) {
            throw std::runtime_error("failed to create " + args.api + " tensors");
        }

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        aclnnStatus ret;
        if (args.api == "ProdDim") {
            const auto reduceDims = NormalizeReduceDims(args.api, args.shapeA, args.dims, false);
            auto getWorkspaceSize = ResolveSymbol<ProdDimGetWorkspaceSizeFn>("aclnnProdDimGetWorkspaceSize");
            ret = getWorkspaceSize(
                aTensor.tensor, reduceDims[0], args.keepDims, outputDtype, outTensor.tensor, &workspaceSize,
                &executor);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclnnProdDimGetWorkspaceSize failed: " + std::to_string(ret));
            }
        } else {
            auto getWorkspaceSize = ResolveSymbol<ProdGetWorkspaceSizeFn>("aclnnProdGetWorkspaceSize");
            ret = getWorkspaceSize(aTensor.tensor, outputDtype, outTensor.tensor, &workspaceSize, &executor);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclnnProdGetWorkspaceSize failed: " + std::to_string(ret));
            }
        }
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("workspace allocation failed");
            }
        }
        if (args.api == "ProdDim") {
            auto launch = ResolveSymbol<ProdDimLaunchFn>("aclnnProdDim");
            ret = launch(workspace, workspaceSize, executor, stream);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclnnProdDim failed: " + std::to_string(ret));
            }
        } else {
            auto launch = ResolveSymbol<ProdLaunchFn>("aclnnProd");
            ret = launch(workspace, workspaceSize, executor, stream);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclnnProd failed: " + std::to_string(ret));
            }
        }
        ret = aclrtSynchronizeStream(stream);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error("aclrtSynchronizeStream failed");
        }

        std::vector<TOut> actual(golden.size());
        const size_t bytes = std::max<size_t>(actual.size() * sizeof(TOut), sizeof(TOut));
        if (!actual.empty()) {
            ret = aclrtMemcpy(actual.data(), bytes, outTensor.deviceAddr, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclrtMemcpy device to host failed");
            }
        }
        result = CompareResult(actual, golden, outputBoolType);
    } catch (const std::exception &exc) {
        result.status = "failed";
        result.message = exc.what();
        result.retCode = 1;
    }

    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    outTensor.Cleanup();
    aTensor.Cleanup();
    return result;
}

template <typename TIn>
NativeResult DispatchProdOutput(
    const Args &args, aclDataType inputDtype, bool inputBoolType, aclrtStream stream)
{
    const std::string outputDtypeName = args.dtypeAttr.empty() ? args.dtype : args.dtypeAttr;
    if (outputDtypeName == "fp32") {
        return RunProdTypedWithStream<TIn, float>(
            args, inputDtype, inputBoolType, DtypeFromName(outputDtypeName), false, stream);
    }
    if (outputDtypeName == "fp64") {
        return RunProdTypedWithStream<TIn, double>(
            args, inputDtype, inputBoolType, DtypeFromName(outputDtypeName), false, stream);
    }
    if (outputDtypeName == "fp16") {
        return RunProdTypedWithStream<TIn, HalfStorage<HalfKind::Fp16>>(
            args, inputDtype, inputBoolType, DtypeFromName(outputDtypeName), false, stream);
    }
    if (outputDtypeName == "bf16") {
        return RunProdTypedWithStream<TIn, HalfStorage<HalfKind::Bf16>>(
            args, inputDtype, inputBoolType, DtypeFromName(outputDtypeName), false, stream);
    }
    if (outputDtypeName == "int32") {
        return RunProdTypedWithStream<TIn, int32_t>(
            args, inputDtype, inputBoolType, DtypeFromName(outputDtypeName), false, stream);
    }
    if (outputDtypeName == "int64") {
        return RunProdTypedWithStream<TIn, int64_t>(
            args, inputDtype, inputBoolType, DtypeFromName(outputDtypeName), false, stream);
    }
    if (outputDtypeName == "int16") {
        return RunProdTypedWithStream<TIn, int16_t>(
            args, inputDtype, inputBoolType, DtypeFromName(outputDtypeName), false, stream);
    }
    if (outputDtypeName == "int8") {
        return RunProdTypedWithStream<TIn, int8_t>(
            args, inputDtype, inputBoolType, DtypeFromName(outputDtypeName), false, stream);
    }
    if (outputDtypeName == "uint8") {
        return RunProdTypedWithStream<TIn, uint8_t>(
            args, inputDtype, inputBoolType, DtypeFromName(outputDtypeName), false, stream);
    }
    if (outputDtypeName == "bool") {
        return RunProdTypedWithStream<TIn, uint8_t>(
            args, inputDtype, inputBoolType, DtypeFromName(outputDtypeName), true, stream);
    }
    NativeResult result;
    result.status = "failed";
    result.message = "unsupported " + args.api + " output dtype: " + outputDtypeName;
    result.retCode = 1;
    return result;
}

NativeResult DispatchProdWithStream(const Args &args, aclrtStream stream)
{
    if (args.dtype == "fp32") {
        return DispatchProdOutput<float>(args, DtypeFromName(args.dtype), false, stream);
    }
    if (args.dtype == "fp64") {
        return DispatchProdOutput<double>(args, DtypeFromName(args.dtype), false, stream);
    }
    if (args.dtype == "fp16") {
        return DispatchProdOutput<HalfStorage<HalfKind::Fp16>>(args, DtypeFromName(args.dtype), false, stream);
    }
    if (args.dtype == "bf16") {
        return DispatchProdOutput<HalfStorage<HalfKind::Bf16>>(args, DtypeFromName(args.dtype), false, stream);
    }
    if (args.dtype == "int32") {
        return DispatchProdOutput<int32_t>(args, DtypeFromName(args.dtype), false, stream);
    }
    if (args.dtype == "int64") {
        return DispatchProdOutput<int64_t>(args, DtypeFromName(args.dtype), false, stream);
    }
    if (args.dtype == "int16") {
        return DispatchProdOutput<int16_t>(args, DtypeFromName(args.dtype), false, stream);
    }
    if (args.dtype == "int8") {
        return DispatchProdOutput<int8_t>(args, DtypeFromName(args.dtype), false, stream);
    }
    if (args.dtype == "uint8") {
        return DispatchProdOutput<uint8_t>(args, DtypeFromName(args.dtype), false, stream);
    }
    if (args.dtype == "bool") {
        return DispatchProdOutput<uint8_t>(args, DtypeFromName(args.dtype), true, stream);
    }
    NativeResult result;
    result.status = "failed";
    result.message = "unsupported dtype: " + args.dtype;
    result.retCode = 1;
    return result;
}

NativeResult DispatchTensorListWithStream(const Args &args, aclrtStream stream)
{
    if (args.dtype == "fp32") {
        return RunTensorListTypedWithStream<float>(args, aclDataType::ACL_FLOAT, false, stream);
    }
    if (args.dtype == "fp64") {
        return RunTensorListTypedWithStream<double>(args, aclDataType::ACL_DOUBLE, false, stream);
    }
    if (args.dtype == "fp16") {
        return RunTensorListTypedWithStream<HalfStorage<HalfKind::Fp16>>(args, aclDataType::ACL_FLOAT16, false, stream);
    }
    if (args.dtype == "bf16") {
        return RunTensorListTypedWithStream<HalfStorage<HalfKind::Bf16>>(args, aclDataType::ACL_BF16, false, stream);
    }
    if (args.dtype == "int32") {
        return RunTensorListTypedWithStream<int32_t>(args, aclDataType::ACL_INT32, false, stream);
    }
    if (args.dtype == "int64") {
        return RunTensorListTypedWithStream<int64_t>(args, aclDataType::ACL_INT64, false, stream);
    }
    if (args.dtype == "int16") {
        return RunTensorListTypedWithStream<int16_t>(args, aclDataType::ACL_INT16, false, stream);
    }
    if (args.dtype == "int8") {
        return RunTensorListTypedWithStream<int8_t>(args, aclDataType::ACL_INT8, false, stream);
    }
    if (args.dtype == "uint8") {
        return RunTensorListTypedWithStream<uint8_t>(args, aclDataType::ACL_UINT8, false, stream);
    }
    if (args.dtype == "bool") {
        return RunTensorListTypedWithStream<uint8_t>(args, aclDataType::ACL_BOOL, true, stream);
    }
    NativeResult result;
    result.status = "failed";
    result.message = "unsupported dtype: " + args.dtype;
    result.retCode = 1;
    return result;
}

NativeResult DispatchChunkCatWithStream(const Args &args, aclrtStream stream)
{
    if (args.dtype == "fp32") {
        return RunChunkCatTypedWithStream<float>(args, aclDataType::ACL_FLOAT, false, stream);
    }
    if (args.dtype == "fp16") {
        return RunChunkCatTypedWithStream<HalfStorage<HalfKind::Fp16>>(args, aclDataType::ACL_FLOAT16, false, stream);
    }
    if (args.dtype == "bf16") {
        return RunChunkCatTypedWithStream<HalfStorage<HalfKind::Bf16>>(args, aclDataType::ACL_BF16, false, stream);
    }
    NativeResult result;
    result.status = "failed";
    result.message = "unsupported dtype: " + args.dtype;
    result.retCode = 1;
    return result;
}

NativeResult DispatchSplitWithSizeWithStream(const Args &args, aclrtStream stream)
{
    if (args.dtype == "fp32") {
        return RunSplitWithSizeTypedWithStream<float>(args, aclDataType::ACL_FLOAT, false, stream);
    }
    if (args.dtype == "fp16") {
        return RunSplitWithSizeTypedWithStream<HalfStorage<HalfKind::Fp16>>(args, aclDataType::ACL_FLOAT16, false, stream);
    }
    if (args.dtype == "int32") {
        return RunSplitWithSizeTypedWithStream<int32_t>(args, aclDataType::ACL_INT32, false, stream);
    }
    if (args.dtype == "int64") {
        return RunSplitWithSizeTypedWithStream<int64_t>(args, aclDataType::ACL_INT64, false, stream);
    }
    if (args.dtype == "int16") {
        return RunSplitWithSizeTypedWithStream<int16_t>(args, aclDataType::ACL_INT16, false, stream);
    }
    if (args.dtype == "int8") {
        return RunSplitWithSizeTypedWithStream<int8_t>(args, aclDataType::ACL_INT8, false, stream);
    }
    if (args.dtype == "bool") {
        return RunSplitWithSizeTypedWithStream<uint8_t>(args, aclDataType::ACL_BOOL, true, stream);
    }
    NativeResult result;
    result.status = "failed";
    result.message = "unsupported dtype: " + args.dtype;
    result.retCode = 1;
    return result;
}

NativeResult DispatchMaskedSelectWithStream(const Args &args, aclrtStream stream)
{
    if (args.dtype == "fp32") {
        return RunMaskedSelectTypedWithStream<float>(args, aclDataType::ACL_FLOAT, false, stream);
    }
    if (args.dtype == "fp64") {
        return RunMaskedSelectTypedWithStream<double>(args, aclDataType::ACL_DOUBLE, false, stream);
    }
    if (args.dtype == "fp16") {
        return RunMaskedSelectTypedWithStream<HalfStorage<HalfKind::Fp16>>(args, aclDataType::ACL_FLOAT16, false, stream);
    }
    if (args.dtype == "bf16") {
        return RunMaskedSelectTypedWithStream<HalfStorage<HalfKind::Bf16>>(args, aclDataType::ACL_BF16, false, stream);
    }
    if (args.dtype == "int32") {
        return RunMaskedSelectTypedWithStream<int32_t>(args, aclDataType::ACL_INT32, false, stream);
    }
    if (args.dtype == "int64") {
        return RunMaskedSelectTypedWithStream<int64_t>(args, aclDataType::ACL_INT64, false, stream);
    }
    if (args.dtype == "int16") {
        return RunMaskedSelectTypedWithStream<int16_t>(args, aclDataType::ACL_INT16, false, stream);
    }
    if (args.dtype == "int8") {
        return RunMaskedSelectTypedWithStream<int8_t>(args, aclDataType::ACL_INT8, false, stream);
    }
    if (args.dtype == "uint8") {
        return RunMaskedSelectTypedWithStream<uint8_t>(args, aclDataType::ACL_UINT8, false, stream);
    }
    if (args.dtype == "bool") {
        return RunMaskedSelectTypedWithStream<uint8_t>(args, aclDataType::ACL_BOOL, true, stream);
    }
    NativeResult result;
    result.status = "failed";
    result.message = "unsupported dtype: " + args.dtype;
    result.retCode = 1;
    return result;
}

NativeResult DispatchWithStream(const Args &args, aclrtStream stream, NativeStageTimings *timings = nullptr)
{
    if (args.api == "LightningIndexer") {
        return RunLightningIndexerWithStream(args, stream, timings);
    }
    if (args.api == "RecurrentGatedDeltaRule") {
        return RunRecurrentGatedDeltaRuleWithStream(args, stream, timings);
    }
    if (args.api == "Cast") {
        return DispatchCastWithStream(args, stream);
    }
    if (args.api == "Cat" || args.api == "Stack") {
        return DispatchTensorListWithStream(args, stream);
    }
    if (args.api == "ChunkCat") {
        return DispatchChunkCatWithStream(args, stream);
    }
    if (args.api == "SplitWithSize") {
        return DispatchSplitWithSizeWithStream(args, stream);
    }
    if (args.api == "MaskedSelect") {
        return DispatchMaskedSelectWithStream(args, stream);
    }
    if ((args.api == "Prod" || args.api == "ProdDim") && !args.dtypeAttr.empty() && args.dtypeAttr != args.dtype) {
        return DispatchProdWithStream(args, stream);
    }
    if (args.api == "IsFinite" || args.api == "IsInf") {
        return DispatchBoolUnaryWithStream(args, stream);
    }
    if (args.dtype == "fp32") {
        return RunTypedWithStream<float>(args, aclDataType::ACL_FLOAT, false, stream);
    }
    if (args.dtype == "fp64") {
        return RunTypedWithStream<double>(args, aclDataType::ACL_DOUBLE, false, stream);
    }
    if (args.dtype == "fp16") {
        return RunTypedWithStream<HalfStorage<HalfKind::Fp16>>(args, aclDataType::ACL_FLOAT16, false, stream);
    }
    if (args.dtype == "bf16") {
        return RunTypedWithStream<HalfStorage<HalfKind::Bf16>>(args, aclDataType::ACL_BF16, false, stream);
    }
    if (args.dtype == "int32") {
        return RunTypedWithStream<int32_t>(args, aclDataType::ACL_INT32, false, stream);
    }
    if (args.dtype == "int64") {
        return RunTypedWithStream<int64_t>(args, aclDataType::ACL_INT64, false, stream);
    }
    if (args.dtype == "int16") {
        return RunTypedWithStream<int16_t>(args, aclDataType::ACL_INT16, false, stream);
    }
    if (args.dtype == "int8") {
        return RunTypedWithStream<int8_t>(args, aclDataType::ACL_INT8, false, stream);
    }
    if (args.dtype == "uint8") {
        return RunTypedWithStream<uint8_t>(args, aclDataType::ACL_UINT8, false, stream);
    }
    if (args.dtype == "bool") {
        return RunTypedWithStream<uint8_t>(args, aclDataType::ACL_BOOL, true, stream);
    }
    NativeResult result;
    result.status = "failed";
    result.message = "unsupported dtype: " + args.dtype;
    result.retCode = 1;
    return result;
}

bool NeedsFreshAclPerCase(const std::string &api)
{
    return api == "Cat" || api == "Stack" || api == "ChunkCat" || api == "SplitWithSize";
}

bool BatchLineNeedsFreshAcl(const std::string &line)
{
    return line.find("api=Cat") != std::string::npos || line.find("api=Stack") != std::string::npos ||
           line.find("api=ChunkCat") != std::string::npos || line.find("api=SplitWithSize") != std::string::npos;
}

bool EmitsStageTimings(const std::string &api)
{
    return api == "LightningIndexer" || api == "RecurrentGatedDeltaRule";
}

NativeResult DispatchWithFreshAcl(const Args &args)
{
    NativeResult result;
    aclrtStream stream = nullptr;
    if (InitAcl(args.device, &stream) != ACL_SUCCESS) {
        result.status = "failed";
        result.message = "ACL device initialization failed";
        result.retCode = 1;
        return result;
    }

    result = DispatchWithStream(args, stream);
    if (stream != nullptr) {
        aclrtDestroyStream(stream);
    }
    aclrtResetDevice(args.device);
    aclFinalize();
    return result;
}

int DispatchFreshAclCaseInChild(const Args &args, int caseIndex)
{
    const pid_t pid = fork();
    if (pid < 0) {
        NativeResult result;
        auto start = std::chrono::steady_clock::now();
        try {
            result = DispatchWithFreshAcl(args);
        } catch (const std::exception &exc) {
            result.status = "failed";
            result.message = exc.what();
            result.retCode = 1;
        }
        auto end = std::chrono::steady_clock::now();
        const double durationMs = std::chrono::duration<double, std::milli>(end - start).count();
        PrintResultJson(result, caseIndex, durationMs);
        return result.retCode;
    }

    if (pid == 0) {
        NativeResult result;
        auto start = std::chrono::steady_clock::now();
        try {
            result = DispatchWithFreshAcl(args);
        } catch (const std::exception &exc) {
            result.status = "failed";
            result.message = exc.what();
            result.retCode = 1;
        }
        auto end = std::chrono::steady_clock::now();
        const double durationMs = std::chrono::duration<double, std::milli>(end - start).count();
        PrintResultJson(result, caseIndex, durationMs);
        _exit(result.retCode == 0 ? 0 : 1);
    }

    int status = 0;
    if (waitpid(pid, &status, 0) < 0) {
        std::cerr << "waitpid failed\n";
        return 1;
    }
    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }
    return 1;
}

int RunSingle(const Args &args)
{
    aclrtStream stream = nullptr;
    if (InitAcl(args.device, &stream) != ACL_SUCCESS) {
        return 1;
    }
    NativeStageTimings timings;
    const bool emitTimings = EmitsStageTimings(args.api);
    auto start = std::chrono::steady_clock::now();
    NativeResult result = DispatchWithStream(args, stream, emitTimings ? &timings : nullptr);
    auto end = std::chrono::steady_clock::now();
    const double durationMs = std::chrono::duration<double, std::milli>(end - start).count();
    PrintResultJson(result, -1, durationMs, emitTimings ? &timings : nullptr);
    if (stream != nullptr) {
        aclrtDestroyStream(stream);
    }
    aclrtResetDevice(args.device);
    aclFinalize();
    return result.retCode;
}

int RunBatch(const Args &args)
{
    std::ifstream handle(args.batchFile);
    if (!handle.is_open()) {
        std::cerr << "failed to open batch file: " << args.batchFile << "\n";
        return 1;
    }

    bool freshAclPerCase = false;
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(handle, line)) {
        if (line.empty()) {
            continue;
        }
        freshAclPerCase = freshAclPerCase || BatchLineNeedsFreshAcl(line);
        lines.push_back(line);
    }

    aclrtStream stream = nullptr;
    if (!freshAclPerCase && InitAcl(args.device, &stream) != ACL_SUCCESS) {
        return 1;
    }

    int retCode = 0;
    for (const auto &batchLine : lines) {
        int caseIndex = -1;
        NativeResult result;
        NativeStageTimings timings;
        bool emitTimings = false;
        auto start = std::chrono::steady_clock::now();
        try {
            Args caseArgs;
            {
                ScopedStageTimer parseStage(StageTimingTarget(&timings, &NativeStageTimings::parseMs));
                caseArgs = ParseBatchLine(batchLine, args.device, args.opapiLib, &caseIndex);
            }
            emitTimings = EmitsStageTimings(caseArgs.api);
            if (freshAclPerCase || NeedsFreshAclPerCase(caseArgs.api)) {
                const int childRet = DispatchFreshAclCaseInChild(caseArgs, caseIndex);
                if (childRet != 0) {
                    retCode = 1;
                }
                continue;
            } else {
                result = DispatchWithStream(caseArgs, stream, emitTimings ? &timings : nullptr);
            }
        } catch (const std::exception &exc) {
            result.status = "failed";
            result.message = exc.what();
            result.retCode = 1;
        }
        auto end = std::chrono::steady_clock::now();
        const double durationMs = std::chrono::duration<double, std::milli>(end - start).count();
        PrintResultJson(result, caseIndex, durationMs, emitTimings ? &timings : nullptr);
        if (result.retCode != 0) {
            retCode = 1;
        }
    }

    if (stream != nullptr) {
        aclrtDestroyStream(stream);
        aclrtResetDevice(args.device);
        aclFinalize();
    }
    return retCode;
}

}  // namespace

int main(int argc, char **argv)
{
    try {
        Args args = ParseArgs(argc, argv);
        LoadOpApi(args.opapiLib);
        int ret = args.batchFile.empty() ? RunSingle(args) : RunBatch(args);
        if (g_opapiHandle != nullptr) {
            dlclose(g_opapiHandle);
        }
        return ret;
    } catch (const std::exception &exc) {
        std::cerr << exc.what() << "\n";
        if (g_opapiHandle != nullptr) {
            dlclose(g_opapiHandle);
        }
        return 1;
    }
}
