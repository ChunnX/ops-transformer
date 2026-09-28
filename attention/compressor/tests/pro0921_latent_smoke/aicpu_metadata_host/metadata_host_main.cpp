/**
 * Host driver for the real SparseFlashMlaMetadata AICPU kernel.
 *
 * Why this exists: `attention/sparse_flash_mla_metadata` is an AICPU-only
 * operator (engine DNN_VM_AICPU, no `*_def.cpp`), so TTK/CAMODEL cannot select
 * it from the TBE op-info registry and cannot run its metadata pre-op.  But the
 * kernel body (`Prepare` / `BalanceSchedule` / `GenMetadata`) is ordinary CPU
 * C++ over the AICPU `CpuKernelContext` API.  This driver builds the same
 * NodeDef the runtime would build (tensors + attrs) through the toolkit's own
 * `NodeDefBuilder`, then calls the *unmodified*
 * `aicpu::SparseFlashMlaMetadataCpuKernel::Compute`.  It therefore executes the
 * real AICPU implementation on this host; it does not re-implement or fake the
 * scheduling algorithm, and it is explicitly NOT a board AICPU runtime.
 *
 * Link contract (see build_metadata_host.sh):
 *   * compile with -D_GLIBCXX_USE_CXX11_ABI=0 so the driver matches the
 *     toolkit's x86-64 host context libraries (libaicpu_context_host.a,
 *     libaicpu_nodedef_host.a);
 *   * link those two plus libhost_ascend_protobuf.a (the `ascend_private`
 *     namespaced protobuf/abseil the context layer uses);
 *   * -DRUN_TEST makes KERNEL_LOG_* print to stdout (log.h) so a failed
 *     ParamsCheck is visible.
 *
 * I/O contract (all paths relative to argv[1] = work dir):
 *   attrs.txt   one attr per line:  "int <name> <value>" | "str <name> <value>"
 *   inputs.txt  one input per line, in op input-index order:
 *               "<name> <dtype> <d0,d1,...> <raw-file>"
 *   output.txt  one line:           "<dtype> <d0,d1,...> <raw-file>"
 * All tensors are raw little-endian bytes; int32 is the only dtype the
 * metadata op reads/writes.  The driver writes the output tensor back to
 * <raw-file> and prints "METADATA_OK ..." or "METADATA_FAIL ...".
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "cpu_context.h"
#include "cpu_kernel.h"
#include "cpu_kernel_utils.h"
#include "cpu_tensor.h"
#include "node_def_builder.h"
#include "status.h"

// Real, unmodified kernel class + implementation.
#include "sparse_flash_mla_metadata_aicpu.h"

namespace {

struct InputSpec {
    std::string name;
    aicpu::DataType dtype = aicpu::DT_INT32;
    std::vector<int64_t> dims;
    std::string file;
};

struct OutputSpec {
    aicpu::DataType dtype = aicpu::DT_INT32;
    std::vector<int64_t> dims;
    std::string file;
};

struct AttrSpec {
    std::string name;
    bool is_str = false;
    long long ival = 0;
    std::string sval;
};

int DtypeWidth(aicpu::DataType dtype)
{
    switch (dtype) {
        case aicpu::DT_INT32:
        case aicpu::DT_FLOAT:
            return 4;
        case aicpu::DT_INT64:
        case aicpu::DT_DOUBLE:
            return 8;
        case aicpu::DT_INT16:
        case aicpu::DT_UINT16:
        case aicpu::DT_FLOAT16:
        case aicpu::DT_BFLOAT16:
            return 2;
        case aicpu::DT_INT8:
        case aicpu::DT_UINT8:
        case aicpu::DT_BOOL:
        case aicpu::DT_HIFLOAT8:
            return 1;
        default:
            return -1;
    }
}

bool ParseDtype(const std::string &text, aicpu::DataType &out)
{
    if (text == "int32") {
        out = aicpu::DT_INT32;
    } else if (text == "int64") {
        out = aicpu::DT_INT64;
    } else if (text == "float32") {
        out = aicpu::DT_FLOAT;
    } else if (text == "float16") {
        out = aicpu::DT_FLOAT16;
    } else if (text == "bfloat16") {
        out = aicpu::DT_BFLOAT16;
    } else if (text == "uint8") {
        out = aicpu::DT_UINT8;
    } else {
        return false;
    }
    return true;
}

std::vector<int64_t> ParseDims(const std::string &text)
{
    std::vector<int64_t> dims;
    std::stringstream stream(text);
    std::string token;
    while (std::getline(stream, token, ',')) {
        if (token.empty()) {
            continue;
        }
        dims.push_back(std::strtoll(token.c_str(), nullptr, 10));
    }
    return dims;
}

int64_t ElementCount(const std::vector<int64_t> &dims)
{
    int64_t total = 1;
    for (int64_t d : dims) {
        total *= d;
    }
    return total;
}

bool ReadAll(const std::string &path, std::vector<uint8_t> &out)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        return false;
    }
    stream.seekg(0, std::ios::end);
    const std::streamoff size = stream.tellg();
    if (size < 0) {
        return false;
    }
    out.assign(static_cast<size_t>(size), 0);
    stream.seekg(0, std::ios::beg);
    if (size > 0) {
        stream.read(reinterpret_cast<char *>(out.data()), size);
    }
    return true;
}

bool WriteAll(const std::string &path, const uint8_t *data, size_t size)
{
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) {
        return false;
    }
    if (size > 0) {
        stream.write(reinterpret_cast<const char *>(data), static_cast<std::streamsize>(size));
    }
    return static_cast<bool>(stream);
}

bool LoadAttrs(const std::string &path, std::vector<AttrSpec> &attrs)
{
    std::ifstream stream(path);
    if (!stream) {
        return false;
    }
    std::string line;
    while (std::getline(stream, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::stringstream ls(line);
        std::string kind;
        AttrSpec attr;
        if (!(ls >> kind >> attr.name)) {
            return false;
        }
        if (kind == "int") {
            if (!(ls >> attr.ival)) {
                return false;
            }
            attr.is_str = false;
        } else if (kind == "str") {
            std::string rest;
            std::getline(ls, rest);
            const size_t start = rest.find_first_not_of(" \t");
            attr.sval = (start == std::string::npos) ? "" : rest.substr(start);
            attr.is_str = true;
        } else {
            return false;
        }
        attrs.push_back(attr);
    }
    return true;
}

bool LoadInputs(const std::string &path, std::vector<InputSpec> &inputs)
{
    std::ifstream stream(path);
    if (!stream) {
        return false;
    }
    std::string line;
    while (std::getline(stream, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::stringstream ls(line);
        InputSpec spec;
        std::string dtype;
        std::string dims;
        if (!(ls >> spec.name >> dtype >> dims >> spec.file)) {
            return false;
        }
        if (!ParseDtype(dtype, spec.dtype)) {
            std::fprintf(stderr, "unknown input dtype: %s\n", dtype.c_str());
            return false;
        }
        spec.dims = ParseDims(dims);
        inputs.push_back(spec);
    }
    return true;
}

bool LoadOutput(const std::string &path, OutputSpec &out)
{
    std::ifstream stream(path);
    if (!stream) {
        return false;
    }
    std::string line;
    while (std::getline(stream, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::stringstream ls(line);
        std::string dtype;
        std::string dims;
        if (!(ls >> dtype >> dims >> out.file)) {
            return false;
        }
        if (!ParseDtype(dtype, out.dtype)) {
            std::fprintf(stderr, "unknown output dtype: %s\n", dtype.c_str());
            return false;
        }
        out.dims = ParseDims(dims);
        return true;
    }
    return false;
}

}  // namespace

int main(int argc, char **argv)
{
    if (argc < 2) {
        std::fprintf(stderr, "usage: metadata_host <workdir>\n");
        return 2;
    }
    const std::string workdir = argv[1];

    std::vector<AttrSpec> attrs;
    std::vector<InputSpec> inputs;
    OutputSpec output;
    if (!LoadAttrs(workdir + "/attrs.txt", attrs) ||
        !LoadInputs(workdir + "/inputs.txt", inputs) ||
        !LoadOutput(workdir + "/output.txt", output)) {
        std::fprintf(stderr, "FAIL: cannot read attrs.txt/inputs.txt/output.txt under %s\n", workdir.c_str());
        return 2;
    }

    // The tensor data pointers must stay valid until Compute() returns; a
    // vector of unique_ptr keeps their addresses stable.
    std::vector<std::unique_ptr<std::vector<uint8_t>>> input_storage;
    input_storage.reserve(inputs.size() + 1);
    for (const InputSpec &spec : inputs) {
        auto buffer = std::make_unique<std::vector<uint8_t>>();
        if (!ReadAll(workdir + "/" + spec.file, *buffer)) {
            std::fprintf(stderr, "FAIL: cannot read input tensor %s (%s)\n", spec.name.c_str(),
                         spec.file.c_str());
            return 2;
        }
        const int width = DtypeWidth(spec.dtype);
        const int64_t expected = ElementCount(spec.dims) * width;
        if (width < 0 || static_cast<int64_t>(buffer->size()) != expected) {
            std::fprintf(stderr, "FAIL: input %s byte size %zu does not match dims x width (%lld)\n",
                         spec.name.c_str(), buffer->size(), static_cast<long long>(expected));
            return 2;
        }
        input_storage.push_back(std::move(buffer));
    }

    const int out_width = DtypeWidth(output.dtype);
    const int64_t out_elems = ElementCount(output.dims);
    if (out_width < 0 || out_elems <= 0) {
        std::fprintf(stderr, "FAIL: bad output spec\n");
        return 2;
    }
    const size_t out_bytes = static_cast<size_t>(out_elems) * static_cast<size_t>(out_width);
    auto out_storage = std::make_unique<std::vector<uint8_t>>(out_bytes, 0);
    input_storage.push_back(std::move(out_storage));
    uint8_t *out_data = input_storage.back()->data();

    auto node_def = aicpu::CpuKernelUtils::CreateNodeDef();
    aicpu::NodeDefBuilder builder(node_def.get(), "SparseFlashMlaMetadata", "SparseFlashMlaMetadata");
    for (size_t i = 0; i < inputs.size(); ++i) {
        builder.Input({inputs[i].name, inputs[i].dtype, inputs[i].dims,
                       input_storage[i]->data(), aicpu::FORMAT_ND});
    }
    builder.Output({"metadata", output.dtype, output.dims, out_data, aicpu::FORMAT_ND});
    for (const AttrSpec &attr : attrs) {
        if (attr.is_str) {
            builder.Attr(attr.name, attr.sval);
        } else {
            builder.Attr(attr.name, static_cast<int>(attr.ival));
        }
    }

    aicpu::CpuKernelContext ctx(aicpu::HOST);
    const uint32_t init_status = ctx.Init(node_def.get());
    if (init_status != aicpu::KERNEL_STATUS_OK) {
        std::fprintf(stderr, "FAIL: CpuKernelContext::Init returned %u\n", init_status);
        return 1;
    }

    aicpu::SparseFlashMlaMetadataCpuKernel kernel;
    const uint32_t rc = kernel.Compute(ctx);
    std::printf("metadata_host: Compute returned %u (inputs=%zu attrs=%zu out_elems=%lld)\n", rc,
                inputs.size(), attrs.size(), static_cast<long long>(out_elems));

    if (!WriteAll(workdir + "/" + output.file, out_data, out_bytes)) {
        std::fprintf(stderr, "FAIL: cannot write output %s\n", output.file.c_str());
        return 2;
    }

    if (rc != aicpu::KERNEL_STATUS_OK) {
        std::printf("METADATA_FAIL rc=%u\n", rc);
        return 1;
    }

    // Summary straight from the produced metadata (optiling::detail::SmlaMetadata
    // layout is fixed by sparse_flash_mla_kernel_metadata.h).
    const int32_t *meta = reinterpret_cast<const int32_t *>(out_data);
    long fa_enabled = 0;
    long fd_enabled = 0;
    for (int i = 0; i < static_cast<int>(optiling::AIC_CORE_MAX_NUM); ++i) {
        if (meta[i * static_cast<int>(optiling::FA_METADATA_SIZE)] != 0) {
            ++fa_enabled;
        }
    }
    const long fd_base = static_cast<long>(optiling::AIC_CORE_MAX_NUM) * optiling::FA_METADATA_SIZE;
    for (int i = 0; i < static_cast<int>(optiling::AIV_CORE_MAX_NUM); ++i) {
        if (meta[fd_base + i * static_cast<int>(optiling::FD_METADATA_SIZE)] != 0) {
            ++fd_enabled;
        }
    }
    std::printf("METADATA_OK fa_enabled_cores=%ld fd_enabled_cores=%ld "
                "fa0[enable,bn2s,ms,s2s,bn2e,me,s2e,fdws,s2max]=%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
                fa_enabled, fd_enabled, meta[0], meta[1], meta[2], meta[3], meta[4], meta[5],
                meta[6], meta[7], meta[8]);
    return 0;
}
