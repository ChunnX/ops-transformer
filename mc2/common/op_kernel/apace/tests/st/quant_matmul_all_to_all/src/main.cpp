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

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "acl/acl.h"
#include "hccl/hccl.h"
#include "hccl/hccl_comm.h"
#include "hccl/hccl_res.h"
#include "hccl/hccl_types.h"
#include "hccl/hccl_rank_graph.h"
#include <cstdlib>
#include "utils.h"
#include "apace/tiling/quant_matmul_mx_tiling_swat.h"
#include "apace/kernel/quant_matmul_all_to_all/matmul_all_to_all_tiling_data.h"
#include "apace/kernel/quant_matmul_all_to_all/mx_matmul_all_to_all_hcomm_impl.h"
#include "apace/block/aiv_comm/collective_comm_context.h"
#include "comm_channel_builder.h"

using CommUdmaContext = Apace::AivComm::CommUdmaContext;
using CommUbmemContext = Apace::AivComm::CommUbmemContext;

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

static bool ParseEndpoint(const std::string &endpoint, std::string &ip, uint16_t &port)
{
  const std::string prefix = "tcp://";
  if (endpoint.find(prefix) != 0) { return false; }
  std::string addr = endpoint.substr(prefix.size());
  auto colonPos = addr.rfind(':');
  if (colonPos == std::string::npos) { return false; }
  ip = addr.substr(0, colonPos);
  port = static_cast<uint16_t>(atoi(addr.substr(colonPos + 1).c_str()));
  return !ip.empty() && port > 0;
}

static int32_t SendAll(int32_t sock, const void *buf, size_t len)
{
  size_t sent = 0;
  while (sent < len) {
    ssize_t n = send(sock, static_cast<const char *>(buf) + sent, len - sent, 0);
    if (n <= 0) { return -1; }
    sent += static_cast<size_t>(n);
  }
  return 0;
}

static int32_t RecvAll(int32_t sock, void *buf, size_t len)
{
  size_t received = 0;
  while (received < len) {
    ssize_t n = recv(sock, static_cast<char *>(buf) + received, len - received, 0);
    if (n <= 0) { return -1; }
    received += static_cast<size_t>(n);
  }
  return 0;
}

static int32_t ExchangeRootInfo(uint32_t rank, uint32_t rankNum,
                                const std::vector<int32_t> &socks, HcclRootInfo *rootInfo)
{
  if (rank == 0) {
    HCCL_CHECK(HcclGetRootInfo(rootInfo));
    for (uint32_t i = 1; i < rankNum; i++) {
      if (SendAll(socks[i - 1], rootInfo, sizeof(HcclRootInfo)) != 0) { return -1; }
    }
  } else {
    if (RecvAll(socks[0], rootInfo, sizeof(HcclRootInfo)) != 0) { return -1; }
  }
  return 0;
}

static int32_t TcpBarrier(uint32_t rank, uint32_t rankNum, const std::vector<int32_t> &socks)
{
  int32_t flag = 1;
  if (rank == 0) {
    for (uint32_t i = 1; i < rankNum; i++) {
      int32_t recvFlag = 0;
      if (RecvAll(socks[i - 1], &recvFlag, sizeof(recvFlag)) != 0) { return -1; }
    }
    for (uint32_t i = 1; i < rankNum; i++) {
      if (SendAll(socks[i - 1], &flag, sizeof(flag)) != 0) { return -1; }
    }
  } else {
    if (SendAll(socks[0], &flag, sizeof(flag)) != 0) { return -1; }
    int32_t recvFlag = 0;
    if (RecvAll(socks[0], &recvFlag, sizeof(recvFlag)) != 0) { return -1; }
  }
  return 0;
}

static int32_t ConnectPeers(uint32_t rank, uint32_t rankNum,
                             const std::string &ip, uint16_t port,
                             std::vector<int32_t> &socks)
{
  if (rank == 0) {
    int32_t listenSock = socket(AF_INET, SOCK_STREAM, 0);
    if (listenSock < 0) { return -1; }
    int32_t opt = 1;
    setsockopt(listenSock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    struct sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    inet_pton(AF_INET, ip.c_str(), &addr.sin_addr);
    if (bind(listenSock, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr)) < 0 ||
        listen(listenSock, static_cast<int>(rankNum)) < 0) {
      close(listenSock);
      return -1;
    }
    socks.resize(rankNum - 1, -1);
    for (uint32_t i = 0; i < rankNum - 1; i++) {
      socks[i] = accept(listenSock, nullptr, nullptr);
      if (socks[i] < 0) {
        close(listenSock);
        return -1;
      }
    }
    close(listenSock);
  } else {
    socks.resize(1, -1);
    int32_t sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) { return -1; }
    struct sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    inet_pton(AF_INET, ip.c_str(), &addr.sin_addr);
    for (int32_t retry = 0; retry < 100; retry++) {
      if (connect(sock, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr)) == 0) {
        socks[0] = sock;
        return 0;
      }
      usleep(100000);
    }
    close(sock);
    return -1;
  }
  return 0;
}

static void CloseSocks(std::vector<int32_t> &socks)
{
  for (int32_t sock : socks) {
    if (sock >= 0) { close(sock); }
  }
  socks.clear();
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
  commTd.nonSplitAxisSize  = static_cast<uint64_t>(k);
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

  std::string ip;
  uint16_t port = 0;
  if (!ParseEndpoint(ipport, ip, port)) {
    ERROR_LOG("invalid endpoint: %s", ipport);
    return -1;
  }
  std::vector<int32_t> socks;
  if (ConnectPeers(static_cast<uint32_t>(rankId), static_cast<uint32_t>(rankNum), ip, port, socks) != 0) {
    ERROR_LOG("rank %d connect peers failed", rankId);
    return -1;
  }

  HcclRootInfo rootInfo;
  if (ExchangeRootInfo(static_cast<uint32_t>(rankId), static_cast<uint32_t>(rankNum), socks, &rootInfo) != 0) {
    ERROR_LOG("rank %d exchange rootInfo failed", rankId);
    CloseSocks(socks);
    return -1;
  }
  TcpBarrier(static_cast<uint32_t>(rankId), static_cast<uint32_t>(rankNum), socks);

  HcclCommConfig config;
  HcclCommConfigInit(&config);
  config.hcclWorldRankID = static_cast<uint32_t>(rankId);
  HcclComm comm = nullptr;
  HCCL_CHECK(HcclCommInitRootInfoConfig(static_cast<uint32_t>(rankNum), &rootInfo,
                                        static_cast<uint32_t>(rankId), &config, &comm));
  if (comm == nullptr) {
    ERROR_LOG("rank %d HcclCommInitRootInfoConfig failed", rankId);
    CloseSocks(socks);
    return -1;
  }

  constexpr uint32_t MAX_AIV_NUM = Apace::AivComm::COMM_MAX_RANK_NUM;
  constexpr uint64_t SYNC_BUF_SIZE = sizeof(uint64_t) * (2 + MAX_AIV_NUM * 6 + Apace::AivComm::COMM_MAX_RANK_NUM);

  CommChannelBuilder builder(comm);
  CommContext hostCtx = {};
  CommUdmaContext* udmaCtx = &hostCtx.udmaCtx;
  CommUbmemContext* ubmemCtx = &hostCtx.ubmemCtx;
  udmaCtx->rankId = builder.GetRankId();
  udmaCtx->rankSize = builder.GetRankSize();
  ubmemCtx->rankId = builder.GetRankId();
  ubmemCtx->rankSize = builder.GetRankSize();

  if (!builder.AllocRegAndBuildChannels(
      ChannelMode::URMA, false, shmemBufBytes, "shareBuf",
      udmaCtx->channelHandles, udmaCtx->commBufferAddrs)) {
    ERROR_LOG("rank %d AllocRegAndBuildChannels shareBuf failed", rankId);
    CloseSocks(socks);
    return -1;
  }

  if (!builder.AllocRegAndBuildChannels(
      ChannelMode::UBMEM, false, SYNC_BUF_SIZE, "syncBuf",
      nullptr, ubmemCtx->commBufferAddrs)) {
    ERROR_LOG("rank %d AllocRegAndBuildChannels syncBuf failed", rankId);
    CloseSocks(socks);
    return -1;
  }

  TcpBarrier(static_cast<uint32_t>(rankId), static_cast<uint32_t>(rankNum), socks);
  CloseSocks(socks);

  CommContext *devContext = nullptr;
  ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&devContext), sizeof(CommContext),
                        ACL_MEM_MALLOC_HUGE_ONLY));
  ACL_CHECK(aclrtMemcpy(devContext, sizeof(CommContext), &hostCtx,
                        sizeof(CommContext), ACL_MEMCPY_HOST_TO_DEVICE));

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

  aclrtFree(devContext);
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