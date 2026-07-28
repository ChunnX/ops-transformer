/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#include <cmath>
#include <iostream>
#include <iomanip>
#include <limits.h>
#include <unistd.h>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include <sys/wait.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include "kernel_basic_intf.h"
#include "acl/acl.h"
#include "hccl/hccl.h"
#include "hccl/hccl_comm.h"
#include "hccl/hccl_res.h"
#include "hccl/hccl_types.h"
#include "hccl/hccl_rank_graph.h"
#include <cstdlib>
#include "utils.h"
#include "apace/tiling/quant_matmul_mx_tiling_swat.h"
#include "apace/kernel/quant_matmul_all_to_all/mx_matmul_all_to_all_hcomm_impl.h"
#include "apace/block/aiv_comm/collective_comm_context.h"
#include "../../utils/comm_channel_builder.h"
#include "../../utils/root_info_exchanger.h"

static constexpr int32_t BENCHMARK_ITERATIONS = 10;
static constexpr uint32_t PING_PONG_BUF_NUM = 4;
static constexpr uint32_t CHANNEL_NOTIFY_NUM = 3U;

#define HCCL_CHECK(status) do { \
  HcclResult ret = (status); \
  if (ret != HCCL_SUCCESS) { \
    std::cerr << __FILE__ << ":" << __LINE__ << " HcclResult:" << ret << std::endl; \
  } \
} while (0)

void PrintAllToAllUsage(const std::string &programName)
{
  std::cerr << "Usage: " << programName << " m k n rankNum [tileCnt] [isLocalDelayed] [isPingPong]" << std::endl;
  std::cerr << "Args: " << std::endl;
  std::cerr << "  m: row of matrix A" << std::endl;
  std::cerr << "  k: col of matrix A (= row of matrix B, full K, not split)" << std::endl;
  std::cerr << "  n: col of matrix B (total N, split to N/rankNum per rank)" << std::endl;
  std::cerr << "  rankNum: number of ranks" << std::endl;
  std::cerr << "  tileCnt: number of regular tiles\n  tileSize: M rows per tile\n  tailSize: M rows per tail tile, 0=no tail" << std::endl;
  std::cerr << "  isLocalDelayed: defer self batch to end (0/1, default: 1)" << std::endl;
  std::cerr << "  isPingPong: use ping-pong shmem (0/1, default: 1, implies isLocalDelayed)" << std::endl;
  std::cerr << "Example: " << programName << " 2048 3584 4096 4 4 0 352 1 0" << std::endl;
}

void ParseAllToAllArgs(int argc, char *argv[], int *m, int *k, int *n, int *rankNum,
             int *tileSize, int *tileCnt, int *tailSize, int *isLocalDelayed, int *isPingPong)
{
  if (argc >= 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
    PrintAllToAllUsage(argv[0]);
    exit(1);
  }
  if (argc < 5) {
    throw std::invalid_argument("ERROR: Lacks Arguments");
  }
  try {
    *m = std::stoi(argv[1]);
    *k = std::stoi(argv[2]);
    *n = std::stoi(argv[3]);
    *rankNum = std::stoi(argv[4]);
    *tileCnt = (argc >= 6) ? std::stoi(argv[5]) : 1;
    *tileSize = (argc >= 7) ? std::stoi(argv[6]) : static_cast<int>(*m / *tileCnt);
    *tailSize = (argc >= 8) ? std::stoi(argv[7]) : 0;
    *isLocalDelayed = (argc >= 9) ? std::stoi(argv[8]) : 1;
    printf("argc >= 9 is %d, Enter isLocalDelayed=%d\n", argc >= 9, *isLocalDelayed);
    *isPingPong = (argc >= 10) ? std::stoi(argv[9]) : 0;
  } catch (const std::invalid_argument&) {
    throw std::invalid_argument("ERROR: m k n rankNum tileCnt isLocalDelayed isPingPong must be Integer");
  }

  if (*m <= 0 || *k <= 0 || *n <= 0 || *rankNum <= 0) {
    throw std::invalid_argument("ERROR: m k n rankNum tileCnt must be positive");
  }

  if (*k % 32 != 0) {
    throw std::invalid_argument("ERROR: K must be divisible by 32 for MXFP8 quantization");
  }

  if (CeilDiv(*k, ::MXFP_DIVISOR_SIZE) % 2 != 0) {
    throw std::invalid_argument("ERROR: CeilDiv(K, 64) must be an even number");
  }

  if (*n % *rankNum != 0) {
    throw std::invalid_argument("ERROR: n must be divisible by rankNum for AllToAll");
  }
}

template <bool LocalDelay>
int LaunchKernel(uint32_t usedCoreNum, aclrtStream stream,
         CommContext *hcommCtx,
         GM_ADDR deviceA, GM_ADDR deviceScaleA,
         GM_ADDR deviceB, GM_ADDR deviceScaleB,
         GM_ADDR deviceOutput, MatmulAllToAllTilingData &tilingData)
{
  MatmulAllToAllMxImpl::MatmulAllToAllMxKernelE4M3E4M3<LocalDelay><<<usedCoreNum, nullptr, stream>>>(
    hcommCtx, deviceA, deviceScaleA, deviceB, deviceScaleB, deviceOutput, tilingData);
  return 0;
}

int RunMatmulAllToAll(int rankNum, int rankId, int m, int k, int n,
             int tileSize, int tileCnt, int tailSize, int isLocalDelayed, int isPingPong)
{
  const char *ipport = "tcp://127.0.0.1:8998";
  INFO_LOG("rankNum=%d, rankId=%d, ipport=%s, localDelayed=%d, pingPong=%d",
       rankNum, rankId, ipport, isLocalDelayed, isPingPong);

  uint32_t np = n / rankNum;

  MatmulAllToAllTilingData tilingData;

  if (tileSize <= 0) { tileSize = m / tileCnt; }
  uint32_t rem = static_cast<uint32_t>(m) - static_cast<uint32_t>(tileSize) * static_cast<uint32_t>(tileCnt);
  uint32_t tailCnt = 0;
  if (tailSize > 0) {
    if (rem % static_cast<uint32_t>(tailSize) != 0) {
      throw std::invalid_argument("ERROR: remainder not divisible by tailSize");
    }
    tailCnt = rem / static_cast<uint32_t>(tailSize);
  } else if (rem != 0) {
    throw std::invalid_argument("ERROR: m must be divisible by tileSize * tileCnt when tailSize=0 or set tailSize>0");
  }
  auto &commTd = tilingData.alltoallTilingData.commTilingData;
  commTd.splitAxisTileSize = static_cast<uint64_t>(tileSize);
  commTd.splitAxisTileCnt  = static_cast<uint64_t>(tileCnt);
  commTd.splitAxisTailSize = static_cast<uint64_t>(tailSize);
  commTd.splitAxisTailCnt  = static_cast<uint64_t>(tailCnt);
  commTd.nonSplitAxisSize  = static_cast<uint64_t>(np);
  uint64_t maxTileM = static_cast<uint64_t>(tileSize);
  if (static_cast<uint64_t>(tailSize) > maxTileM) { maxTileM = static_cast<uint64_t>(tailSize); }
  tilingData.alltoallTilingData.bufferSize = isPingPong ? PING_PONG_BUF_NUM : static_cast<uint32_t>(tileCnt + tailCnt);
  uint64_t shmemBufBytes = static_cast<uint64_t>(tilingData.alltoallTilingData.bufferSize) * maxTileM * n * sizeof(uint16_t);
  // 2MB large page alignment
  shmemBufBytes = (shmemBufBytes + 2UL * 1024 * 1024 - 1) / (2UL * 1024 * 1024) * (2UL * 1024 * 1024);

  printf("CommTurnSet: tileCnt=%u tileM=%u tailM=%u tailCnt=%u maxTileM=%lu shmemBuf=%lu\n",
       static_cast<uint32_t>(commTd.splitAxisTileCnt), static_cast<uint32_t>(commTd.splitAxisTileSize),
       static_cast<uint32_t>(commTd.splitAxisTailSize), static_cast<uint32_t>(commTd.splitAxisTailCnt),
       maxTileM, shmemBufBytes);

  ACL_CHECK(aclInit(nullptr));
  int32_t deviceId = rankId;
  ACL_CHECK(aclrtSetDevice(deviceId));
  aclrtStream stream = nullptr;
  ACL_CHECK(aclrtCreateStream(&stream));

  RootInfoExchanger exchanger(static_cast<uint32_t>(rankId), static_cast<uint32_t>(rankNum), ipport);
  HcclRootInfo rootInfo;
  if (!exchanger.Exchange(rootInfo)) {
    ERROR_LOG("rank %d exchange rootInfo failed", rankId);
    return -1;
  }

  HcclCommConfig config;
  HcclCommConfigInit(&config);
  config.hcclWorldRankID = static_cast<uint32_t>(rankId);
  HcclComm comm = nullptr;
  HCCL_CHECK(HcclCommInitRootInfoConfig(static_cast<uint32_t>(rankNum), &rootInfo,
                                        static_cast<uint32_t>(rankId), &config, &comm));
  if (comm == nullptr) {
    ERROR_LOG("rank %d HcclCommInitRootInfoConfig failed", rankId);
    return -1;
  }

  CommChannelBuilder builder(comm);
  CommContext hostCtx = {};

  const char *ctxTag = "quant_matmul_all_to_all";

  CommContext *devContext = reinterpret_cast<CommContext *>(
      builder.CreateDeviceContext(&hostCtx, sizeof(CommContext), ctxTag,
                                  &hostCtx.udmaCtx, &hostCtx.ubmemCtx));
  if (devContext == nullptr) {
    ERROR_LOG("rank %d CreateDeviceContext failed", rankId);
    return -1;
  }

  exchanger.Barrier();
  exchanger.Close();

  QuantMatmulTilingSwat<mm::DataType::DT_FLOAT8_E4M3FN, mm::DataType::DT_FLOAT8_E4M3FN> tilingEngine;

  uint32_t remoteBatch = isLocalDelayed ? rankNum - 1 : rankNum;
  uint32_t mmTileM = static_cast<uint32_t>(commTd.splitAxisTileSize);
  printf("isLocalDelayed = %d\n", isLocalDelayed);
  tilingEngine.GetTilingData(mmTileM, np, k, tilingData.tileQbmmTilingData);
  if (tailSize > 0) {
    tilingEngine.GetTilingData(static_cast<uint32_t>(tailSize), np, k, tilingData.tailQbmmTilingData);
  }
  if (isLocalDelayed) {
    tilingEngine.GetTilingData(static_cast<uint32_t>(m), np, k, tilingData.localQbmmTilingData);
  }

  std::vector<uint8_t> hostA(m * k, 0);
  std::vector<uint8_t> hostB(k * n, 0);
  std::vector<uint8_t> hostScaleA(m * CeilDiv(k, ::MXFP_DIVISOR_SIZE) * ::MXFP_MULTI_BASE_SIZE, 0);
  std::vector<uint8_t> hostScaleB(
    CeilDiv(k, ::MXFP_DIVISOR_SIZE) * ::MXFP_MULTI_BASE_SIZE * n, 0);
  std::vector<uint16_t> hostOutput(rankNum * m * np, 0);

  auto sizeA = static_cast<size_t>(1) * hostA.size() * sizeof(uint8_t);
  auto sizeB = static_cast<size_t>(1) * hostB.size() * sizeof(uint8_t);
  auto sizeScaleA = static_cast<size_t>(1) * hostScaleA.size() * sizeof(uint8_t);
  auto sizeScaleB = static_cast<size_t>(1) * hostScaleB.size() * sizeof(uint8_t);
  auto sizeOutput = static_cast<size_t>(1) * hostOutput.size() * sizeof(uint16_t);

  char exePath[PATH_MAX];
  ssize_t len = readlink("/proc/self/exe", exePath, sizeof(exePath) - 1);
  std::string baseDir = ".";
  if (len > 0) {
    exePath[len] = '\0';
    baseDir = exePath;
    size_t lastSlash = baseDir.find_last_of('/');
    if (lastSlash != std::string::npos) {
      baseDir.resize(lastSlash);
    }
  }
  std::string inputDir = baseDir + "/input/" + std::to_string(rankId);
  std::string outputDir = baseDir + "/output/" + std::to_string(rankId);
  ReadFile(inputDir + "/input_a.bin", hostA.data(), sizeA);
  ReadFile(inputDir + "/input_b.bin", hostB.data(), sizeB);
  ReadFile(inputDir + "/input_scaleA.bin", hostScaleA.data(), sizeScaleA);
  ReadFile(inputDir + "/input_scaleB.bin", hostScaleB.data(), sizeScaleB);

  GM_ADDR deviceA = nullptr;
  GM_ADDR deviceB = nullptr;
  GM_ADDR deviceScaleA = nullptr;
  GM_ADDR deviceScaleB = nullptr;
  GM_ADDR deviceOutput = nullptr;

  ACL_CHECK(aclrtMalloc((void**)&deviceA, sizeA, ACL_MEM_MALLOC_HUGE_ONLY));
  ACL_CHECK(aclrtMalloc((void**)&deviceB, sizeB, ACL_MEM_MALLOC_HUGE_ONLY));
  ACL_CHECK(aclrtMalloc((void**)&deviceScaleA, sizeScaleA, ACL_MEM_MALLOC_HUGE_ONLY));
  ACL_CHECK(aclrtMalloc((void**)&deviceScaleB, sizeScaleB, ACL_MEM_MALLOC_HUGE_ONLY));
  ACL_CHECK(aclrtMalloc((void**)&deviceOutput, sizeOutput, ACL_MEM_MALLOC_HUGE_ONLY));

  ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(deviceScaleA, sizeScaleA, hostScaleA.data(), sizeScaleA, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(deviceScaleB, sizeScaleB, hostScaleB.data(), sizeScaleB, ACL_MEMCPY_HOST_TO_DEVICE));

  aclrtEvent kernelStartEvent = nullptr;
  aclrtEvent kernelEndEvent = nullptr;
  ACL_CHECK(aclrtCreateEvent(&kernelStartEvent));
  ACL_CHECK(aclrtCreateEvent(&kernelEndEvent));
  ACL_CHECK(aclrtRecordEvent(kernelStartEvent, stream));

  for (int i = 0; i < BENCHMARK_ITERATIONS; ++i) {
    if (isLocalDelayed) {
      LaunchKernel<true>(tilingData.tileQbmmTilingData.usedCoreNum, stream, devContext,
                 deviceA, deviceScaleA, deviceB, deviceScaleB, deviceOutput, tilingData);
    } else {
      LaunchKernel<false>(tilingData.tileQbmmTilingData.usedCoreNum, stream, devContext,
                 deviceA, deviceScaleA, deviceB, deviceScaleB, deviceOutput, tilingData);
    }
  }
  std::cout << "LaunchKernel finished" << std::endl;
  ACL_CHECK(aclrtRecordEvent(kernelEndEvent, stream));
  ACL_CHECK(aclrtSynchronizeStream(stream));

  float kernelElapsedMs = 0.0F;
  ACL_CHECK(aclrtEventElapsedTime(&kernelElapsedMs, kernelStartEvent, kernelEndEvent));
  double kernelElapsedUs = static_cast<double>(kernelElapsedMs) * 1000.0;

  size_t shmemInputSize = static_cast<size_t>(rankNum) * m * np * sizeof(uint16_t);

  ACL_CHECK(aclrtMemcpy(hostOutput.data(), shmemInputSize, deviceOutput,
              shmemInputSize, ACL_MEMCPY_DEVICE_TO_HOST));

  WriteFile(outputDir + "/npu_out.bin", hostOutput.data(), sizeOutput);

  std::cout << std::fixed << std::setprecision(3) << "Kernel elapsed time: " << kernelElapsedUs << " us" << std::endl;

  if (kernelEndEvent != nullptr) {
    aclrtDestroyEvent(kernelEndEvent);
  }
  if (kernelStartEvent != nullptr) {
    aclrtDestroyEvent(kernelStartEvent);
  }

  aclrtFree(deviceA);
  aclrtFree(deviceScaleA);
  aclrtFree(deviceB);
  aclrtFree(deviceScaleB);
  aclrtFree(deviceOutput);

  HcclCommDestroy(comm);

  ACL_CHECK(aclrtDestroyStream(stream));
  ACL_CHECK(aclrtResetDevice(deviceId));
  ACL_CHECK(aclFinalize());

  return 0;
}

int main(int argc, char *argv[])
{
  int m = 0;
  int k = 0;
  int n = 0;
  int rankNum = 0;
  int tileSize = 0;
  int tileCnt = 0;
  int tailSize = 0;
  int isLocalDelayed = 0;
  int isPingPong = 0;
  try {
    ParseAllToAllArgs(argc, argv, &m, &k, &n, &rankNum, &tileSize, &tileCnt, &tailSize, &isLocalDelayed, &isPingPong);
  } catch (const std::invalid_argument& e) {
    std::cerr << e.what() << std::endl;
    PrintAllToAllUsage(argv[0]);
    return -1;
  }

  INFO_LOG("Master (PID=%d) will fork %d processes", getpid(), rankNum);

  std::vector<pid_t> pids(rankNum);
  for (int rankId = 0; rankId < rankNum; ++rankId) {
    pid_t pid = fork();

    if (pid < 0) {
      ERROR_LOG("Fork failed for rank %d", rankId);
      exit(-1);
    } else if (pid == 0) {
      int ret = RunMatmulAllToAll(rankNum, rankId, m, k, n, tileSize, tileCnt, tailSize,
                    isLocalDelayed, isPingPong);
      exit(ret);
    } else {
      pids[rankId] = pid;
      INFO_LOG("Forked Rank %d -> PID %d", rankId, pid);
    }
  }

  int status;
  bool allSuccess = true;
  for (int rankId = 0; rankId < rankNum; ++rankId) {
    pid_t pid = pids[rankId];
    waitpid(pid, &status, 0);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
      allSuccess = false;
      ERROR_LOG("Worker PID %d failed", pid);
    }
  }

  std::cout << "All workers finished. Status: " << (allSuccess ? "SUCCESS" : "FAILURE") << std::endl;
  return allSuccess ? 0 : -1;
}