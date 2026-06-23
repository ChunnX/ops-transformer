/**
 * QLI V2 Metadata CPU Test Runner
 *
 * Usage:
 *   ./qli_v2_metadata_test <config_file>
 *   ./qli_v2_metadata_test                    # use built-in defaults
 *
 * Config file format (key = value, # for comments):
 *   # Attributes
 *   batch_size = 2
 *   max_seqlen_q = 4
 *   seqused_q = [4, 8]
 *   ...
 *
 * See example_config.txt for a complete template.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>
#include <string>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <map>

// Stub headers (must be included before the kernel header)
#include "cpu_kernel.h"
#include "cpu_context.h"
#include "cpu_tensor.h"
#include "cpu_attr_value.h"
#include "status.h"
#include "log.h"

// The kernel under test
#include "kernel/quant_lightning_indexer_v2_metadata_aicpu.h"

using namespace aicpu;

// ============================================================================
// Helper: create a Tensor with data and shape
// ============================================================================
static std::shared_ptr<Tensor> MakeTensor(void *data, const std::vector<int64_t> &dims,
                                           DataType dtype = DT_INT32)
{
    auto tensor = std::make_shared<Tensor>();
    tensor->SetData(data);
    tensor->SetDataType(dtype);
    auto shape = std::make_shared<TensorShape>(dims);
    tensor->SetTensorShape(shape.get());
    uint64_t elemCount = 1;
    for (auto d : dims) { elemCount *= static_cast<uint64_t>(d); }
    int elemSize = 4;
    if (dtype == DT_INT64) elemSize = 8;
    tensor->SetDataSize(elemCount * elemSize);
    return tensor;
}

static std::shared_ptr<AttrValue> MakeIntAttr(int64_t val)
{
    auto attr = std::make_shared<AttrValue>();
    attr->SetInt(val);
    return attr;
}

static std::shared_ptr<AttrValue> MakeStringAttr(const std::string &val)
{
    auto attr = std::make_shared<AttrValue>();
    attr->SetString(val);
    return attr;
}

// ============================================================================
// Config file parser
// ============================================================================
static std::string Trim(const std::string &s)
{
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

static std::vector<int32_t> ParseIntArray(const std::string &val)
{
    std::vector<int32_t> result;
    // Strip brackets
    std::string s = val;
    s.erase(std::remove(s.begin(), s.end(), '['), s.end());
    s.erase(std::remove(s.begin(), s.end(), ']'), s.end());
    std::istringstream iss(s);
    std::string token;
    while (std::getline(iss, token, ',')) {
        std::string t = Trim(token);
        if (!t.empty()) {
            result.push_back(atoi(t.c_str()));
        }
    }
    return result;
}

struct Config {
    // Attributes
    int32_t batchSize   = 2;
    int32_t maxSeqlenQ  = 4;
    int32_t maxSeqlenK  = 128;
    int32_t numHeadsQ   = 8;
    int32_t numHeadsK   = 1;
    int32_t headDim     = 128;
    int32_t topk        = 8;
    int32_t quantMode   = 0;
    int32_t maskMode    = 3;
    int32_t cmpRatio    = 1;
    uint32_t aicCoreNum = optiling::AIC_CORE_MAX_NUM;
    uint32_t aivCoreNum = optiling::AIV_CORE_MAX_NUM;
    std::string socVersion = "Ascend950";
    std::string layoutQ = "BSND";
    std::string layoutK = "BSND";

    // Tensors
    std::vector<int32_t> sequsedQ;
    std::vector<int32_t> sequsedK;
    std::vector<int32_t> cuSeqlensQ;
    std::vector<int32_t> cuSeqlensK;
    std::vector<int32_t> cmpResidualK;

    // Flags: whether tensor was explicitly provided in config
    bool hasSequsedQ     = false;
    bool hasSequsedK     = false;
    bool hasCuSeqlensQ   = false;
    bool hasCuSeqlensK   = false;
    bool hasCmpResidualK = false;
};

static bool ParseConfigFile(const std::string &path, Config &cfg)
{
    std::ifstream file(path);
    if (!file.is_open()) {
        printf("[ERROR] Cannot open config file: %s\n", path.c_str());
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        line = Trim(line);
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') continue;

        // Find '='
        size_t eq = line.find('=');
        if (eq == std::string::npos) continue;

        std::string key = Trim(line.substr(0, eq));
        std::string val = Trim(line.substr(eq + 1));

        // --- Attributes (scalars) ---
        if      (key == "batch_size")    cfg.batchSize   = atoi(val.c_str());
        else if (key == "max_seqlen_q")  cfg.maxSeqlenQ  = atoi(val.c_str());
        else if (key == "max_seqlen_k")  cfg.maxSeqlenK  = atoi(val.c_str());
        else if (key == "num_heads_q")   cfg.numHeadsQ   = atoi(val.c_str());
        else if (key == "num_heads_k")   cfg.numHeadsK   = atoi(val.c_str());
        else if (key == "head_dim")      cfg.headDim     = atoi(val.c_str());
        else if (key == "topk")          cfg.topk        = atoi(val.c_str());
        else if (key == "quant_mode")    cfg.quantMode   = atoi(val.c_str());
        else if (key == "mask_mode")     cfg.maskMode    = atoi(val.c_str());
        else if (key == "cmp_ratio")     cfg.cmpRatio    = atoi(val.c_str());
        else if (key == "aic_core_num")  cfg.aicCoreNum  = static_cast<uint32_t>(atoi(val.c_str()));
        else if (key == "aiv_core_num")  cfg.aivCoreNum  = static_cast<uint32_t>(atoi(val.c_str()));
        else if (key == "soc_version")   cfg.socVersion  = val;
        else if (key == "layout_q")      cfg.layoutQ     = val;
        else if (key == "layout_k")      cfg.layoutK     = val;
        // --- Tensors (arrays) ---
        else if (key == "seqused_q")     { cfg.sequsedQ     = ParseIntArray(val); cfg.hasSequsedQ     = true; }
        else if (key == "seqused_k")     { cfg.sequsedK     = ParseIntArray(val); cfg.hasSequsedK     = true; }
        else if (key == "cu_seqlens_q")  { cfg.cuSeqlensQ   = ParseIntArray(val); cfg.hasCuSeqlensQ   = true; }
        else if (key == "cu_seqlens_k")  { cfg.cuSeqlensK   = ParseIntArray(val); cfg.hasCuSeqlensK   = true; }
        else if (key == "cmp_residual_k"){ cfg.cmpResidualK = ParseIntArray(val); cfg.hasCmpResidualK = true; }
        else {
            printf("[WARN] Unknown config key: %s\n", key.c_str());
        }
    }
    return true;
}

// ============================================================================
// Fill in missing tensors from available data
// ============================================================================
static void FillTensors(Config &cfg)
{
    int B = cfg.batchSize;

    // seqused_q: if not provided, fill with maxSeqlenQ * (i+1)
    if (!cfg.hasSequsedQ) {
        cfg.sequsedQ.resize(B);
        for (int i = 0; i < B; i++) {
            cfg.sequsedQ[i] = cfg.maxSeqlenQ * (i + 1);
        }
    }

    // seqused_k: if not provided, fill with maxSeqlenK * (i+1)
    if (!cfg.hasSequsedK) {
        cfg.sequsedK.resize(B);
        for (int i = 0; i < B; i++) {
            cfg.sequsedK[i] = cfg.maxSeqlenK * (i + 1);
        }
    }

    // cu_seqlens_q: if not provided, compute prefix sum from seqused_q
    if (!cfg.hasCuSeqlensQ) {
        cfg.cuSeqlensQ.resize(B + 1, 0);
        for (int i = 0; i < B; i++) {
            cfg.cuSeqlensQ[i + 1] = cfg.cuSeqlensQ[i] + cfg.sequsedQ[i];
        }
    }

    // cu_seqlens_k: if not provided, compute prefix sum from seqused_k
    if (!cfg.hasCuSeqlensK) {
        cfg.cuSeqlensK.resize(B + 1, 0);
        for (int i = 0; i < B; i++) {
            cfg.cuSeqlensK[i + 1] = cfg.cuSeqlensK[i] + cfg.sequsedK[i];
        }
    }

    // cmp_residual_k: if not provided, fill with zeros
    if (!cfg.hasCmpResidualK) {
        cfg.cmpResidualK.resize(B, 0);
    }
}

// ============================================================================
// Print helpers
// ============================================================================
static void PrintVec(const char *name, const std::vector<int32_t> &v, const char *src = "")
{
    printf("  %-16s: [", name);
    for (size_t i = 0; i < v.size(); i++) {
        printf("%d%s", v[i], (i < v.size() - 1) ? ", " : "");
    }
    printf("]");
    if (src[0]) printf("  (%s)", src);
    printf("\n");
}

static void PrintParams(const Config &cfg)
{
    printf("\n");
    printf("============================================================\n");
    printf("  QLI V2 Metadata - Input Parameters\n");
    printf("============================================================\n");
    printf("  --- Attributes ---\n");
    printf("  soc_version     : %s\n", cfg.socVersion.c_str());
    printf("  aic_core_num    : %u\n", cfg.aicCoreNum);
    printf("  aiv_core_num    : %u\n", cfg.aivCoreNum);
    printf("  batch_size      : %d\n", cfg.batchSize);
    printf("  max_seqlen_q    : %d\n", cfg.maxSeqlenQ);
    printf("  max_seqlen_k    : %d\n", cfg.maxSeqlenK);
    printf("  num_heads_q     : %d\n", cfg.numHeadsQ);
    printf("  num_heads_k     : %d\n", cfg.numHeadsK);
    printf("  head_dim        : %d\n", cfg.headDim);
    printf("  topk            : %d\n", cfg.topk);
    printf("  quant_mode      : %d\n", cfg.quantMode);
    printf("  layout_q        : %s\n", cfg.layoutQ.c_str());
    printf("  layout_k        : %s\n", cfg.layoutK.c_str());
    printf("  mask_mode       : %d", cfg.maskMode);
    switch (cfg.maskMode) {
        case 0: printf(" (DEFAULT_MASK)"); break;
        case 1: printf(" (ALL_MASK)"); break;
        case 2: printf(" (LEFT_UP_CAUSAL)"); break;
        case 3: printf(" (RIGHT_DOWN_CAUSAL)"); break;
        case 4: printf(" (BAND)"); break;
        default: printf(" (UNKNOWN)"); break;
    }
    printf("\n");
    printf("  cmp_ratio       : %d\n", cfg.cmpRatio);
    printf("  --- Tensors ---\n");
    PrintVec("cu_seqlens_q",  cfg.cuSeqlensQ,   cfg.hasCuSeqlensQ   ? "from config" : "auto: prefix_sum(seqused_q)");
    PrintVec("cu_seqlens_k",  cfg.cuSeqlensK,   cfg.hasCuSeqlensK   ? "from config" : "auto: prefix_sum(seqused_k)");
    PrintVec("seqused_q",     cfg.sequsedQ,     cfg.hasSequsedQ     ? "from config" : "auto: max_seqlen_q*(i+1)");
    PrintVec("seqused_k",     cfg.sequsedK,     cfg.hasSequsedK     ? "from config" : "auto: max_seqlen_k*(i+1)");
    PrintVec("cmp_residual_k",cfg.cmpResidualK, cfg.hasCmpResidualK ? "from config" : "auto: zeros");
    printf("============================================================\n");
}

static void PrintMetadata(const optiling::detail::QliV2Metadata &meta,
                           uint32_t aicCoreNum, uint32_t aivCoreNum)
{
    printf("\n");
    printf("============================================================\n");
    printf("  QLI V2 Metadata Output (AIC = LI, AIV = LD)\n");
    printf("============================================================\n");

    printf("\n--- AIC (LI) Metadata [%u cores] ---\n", aicCoreNum);
    printf("%-6s %-8s %-10s %-10s %-10s %-10s %-10s %-10s %-10s\n",
           "Core", "Enable", "BN2Start", "MStart", "S2Start",
           "BN2End", "MEnd", "S2End", "FdWsIdx");
    printf("------ -------- ---------- ---------- ---------- ---------- ---------- ---------- ----------\n");

    uint32_t usedAicCores = 0;
    for (uint32_t i = 0; i < aicCoreNum; i++) {
        uint32_t enable = meta.qliV2Metadata[i][optiling::QLI_V2_CORE_ENABLE_INDEX];
        if (!enable) continue;
        usedAicCores++;
        printf("%-6u %-8u %-10u %-10u %-10u %-10u %-10u %-10u %-10u\n",
               i, enable,
               meta.qliV2Metadata[i][optiling::QLI_V2_BN2_START_INDEX],
               meta.qliV2Metadata[i][optiling::QLI_V2_M_START_INDEX],
               meta.qliV2Metadata[i][optiling::QLI_V2_S2_START_INDEX],
               meta.qliV2Metadata[i][optiling::QLI_V2_BN2_END_INDEX],
               meta.qliV2Metadata[i][optiling::QLI_V2_M_END_INDEX],
               meta.qliV2Metadata[i][optiling::QLI_V2_S2_END_INDEX],
               meta.qliV2Metadata[i][optiling::QLI_V2_FIRST_QLD_V2_DATA_WORKSPACE_IDX_INDEX]);
    }
    printf("  (Total enabled AIC cores: %u)\n", usedAicCores);

    printf("\n--- AIV (LD) Metadata [%u cores] ---\n", aivCoreNum);
    printf("%-6s %-8s %-10s %-10s %-10s %-10s %-10s %-10s\n",
           "Core", "Enable", "BN2Idx", "MIdx", "WsIdx", "WsNum", "MStart", "MNum");
    printf("------ -------- ---------- ---------- ---------- ---------- ---------- ----------\n");

    uint32_t usedAivCores = 0;
    for (uint32_t i = 0; i < aivCoreNum; i++) {
        uint32_t enable = meta.qldV2Metadata[i][optiling::QLD_V2_CORE_ENABLE_INDEX];
        if (!enable) continue;
        usedAivCores++;
        printf("%-6u %-8u %-10u %-10u %-10u %-10u %-10u %-10u\n",
               i, enable,
               meta.qldV2Metadata[i][optiling::QLD_V2_BN2_IDX_INDEX],
               meta.qldV2Metadata[i][optiling::QLD_V2_M_IDX_INDEX],
               meta.qldV2Metadata[i][optiling::QLD_V2_WORKSPACE_IDX_INDEX],
               meta.qldV2Metadata[i][optiling::QLD_V2_WORKSPACE_NUM_INDEX],
               meta.qldV2Metadata[i][optiling::QLD_V2_M_START_INDEX],
               meta.qldV2Metadata[i][optiling::QLD_V2_M_NUM_INDEX]);
    }
    printf("  (Total enabled AIV cores: %u)\n", usedAivCores);
    printf("============================================================\n\n");
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char *argv[])
{
    Config cfg;

    // ---- Load config ----
    if (argc >= 2) {
        printf("Loading config from: %s\n", argv[1]);
        if (!ParseConfigFile(argv[1], cfg)) {
            return 1;
        }
    } else {
        printf("No config file specified, using built-in defaults.\n");
        printf("Usage: %s <config_file>\n\n", argv[0]);
    }

    // ---- Fill missing tensors ----
    FillTensors(cfg);

    // ---- Print input parameters ----
    PrintParams(cfg);

    // ==================================================================
    // Allocate output metadata buffer
    // ==================================================================
    std::vector<int32_t> metadataBuf(optiling::QLI_V2_METADATA_TOTAL_SIZE, 0);

    // ==================================================================
    // Build CpuKernelContext
    // ==================================================================
    CpuKernelContext ctx(HOST);

    // Input 0: cu_seqlens_q  shape [batchSize+1]
    auto tensorCuSeqQ = MakeTensor(cfg.cuSeqlensQ.data(),
                                    {static_cast<int64_t>(cfg.batchSize + 1)});
    ctx.AddInput(tensorCuSeqQ);

    // Input 1: cu_seqlens_k  shape [batchSize+1]
    auto tensorCuSeqK = MakeTensor(cfg.cuSeqlensK.data(),
                                    {static_cast<int64_t>(cfg.batchSize + 1)});
    ctx.AddInput(tensorCuSeqK);

    // Input 2: seqused_q  shape [batchSize]
    auto tensorSeqQ = MakeTensor(cfg.sequsedQ.data(),
                                  {static_cast<int64_t>(cfg.batchSize)});
    ctx.AddInput(tensorSeqQ);

    // Input 3: seqused_k  shape [batchSize]
    auto tensorSeqK = MakeTensor(cfg.sequsedK.data(),
                                  {static_cast<int64_t>(cfg.batchSize)});
    ctx.AddInput(tensorSeqK);

    // Input 4: cmp_residual_k  shape [batchSize]
    auto tensorCmpRes = MakeTensor(cfg.cmpResidualK.data(),
                                    {static_cast<int64_t>(cfg.batchSize)});
    ctx.AddInput(tensorCmpRes);

    // Output 0: metadata  shape [1024]
    auto tensorMeta = MakeTensor(metadataBuf.data(),
                                  {static_cast<int64_t>(optiling::QLI_V2_METADATA_TOTAL_SIZE)});
    ctx.AddOutput(tensorMeta);

    // ==================================================================
    // Set all attributes
    // ==================================================================
    ctx.SetAttr("aic_core_num",  MakeIntAttr(static_cast<int64_t>(cfg.aicCoreNum)));
    ctx.SetAttr("aiv_core_num",  MakeIntAttr(static_cast<int64_t>(cfg.aivCoreNum)));
    ctx.SetAttr("soc_version",   MakeStringAttr(cfg.socVersion));
    ctx.SetAttr("num_heads_q",   MakeIntAttr(cfg.numHeadsQ));
    ctx.SetAttr("num_heads_k",   MakeIntAttr(cfg.numHeadsK));
    ctx.SetAttr("head_dim",      MakeIntAttr(cfg.headDim));
    ctx.SetAttr("topk",          MakeIntAttr(cfg.topk));
    ctx.SetAttr("batch_size",    MakeIntAttr(cfg.batchSize));
    ctx.SetAttr("max_seqlen_q",  MakeIntAttr(cfg.maxSeqlenQ));
    ctx.SetAttr("max_seqlen_k",  MakeIntAttr(cfg.maxSeqlenK));
    ctx.SetAttr("layout_q",      MakeStringAttr(cfg.layoutQ));
    ctx.SetAttr("layout_k",      MakeStringAttr(cfg.layoutK));
    ctx.SetAttr("mask_mode",     MakeIntAttr(cfg.maskMode));
    ctx.SetAttr("cmp_ratio",     MakeIntAttr(cfg.cmpRatio));
    ctx.SetAttr("quant_mode",    MakeIntAttr(cfg.quantMode));

    // ==================================================================
    // Run the kernel
    // ==================================================================
    QuantLightningIndexerV2MetadataCpuKernel kernel;
    uint32_t ret = kernel.Compute(ctx);

    if (ret != KERNEL_STATUS_OK) {
        printf("\n[FAILED] Kernel Compute returned error code: %u\n", ret);
        return 1;
    }

    printf("\n[OK] Kernel Compute succeeded (return code: %u)\n", ret);

    // ==================================================================
    // Print output metadata
    // ==================================================================
    const auto &meta = *static_cast<const optiling::detail::QliV2Metadata *>(
        static_cast<const void *>(metadataBuf.data()));
    PrintMetadata(meta, cfg.aicCoreNum, cfg.aivCoreNum);

    // ==================================================================
    // Dump raw metadata to binary file (for Python integration)
    // Env var METADATA_OUTPUT specifies the output file path.
    // Format: 1024 x int32_t, little-endian, 4096 bytes total.
    // ==================================================================
    const char *metaOutPath = std::getenv("METADATA_OUTPUT");
    if (metaOutPath != nullptr && metaOutPath[0] != '\0') {
        FILE *fp = fopen(metaOutPath, "wb");
        if (fp != nullptr) {
            fwrite(metadataBuf.data(), sizeof(int32_t),
                   optiling::QLI_V2_METADATA_TOTAL_SIZE, fp);
            fclose(fp);
            printf("[DUMP] Raw metadata written to: %s (%zu bytes)\n",
                   metaOutPath,
                   optiling::QLI_V2_METADATA_TOTAL_SIZE * sizeof(int32_t));
        } else {
            printf("[ERROR] Failed to open metadata output file: %s\n", metaOutPath);
        }
    }

    return 0;
}
