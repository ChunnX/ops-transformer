
struct TestCase {
    std::string scenario;
    uint32_t MmType;
    uint32_t blockNum;
    uint32_t burstNum;
    uint32_t burstLen;
    uint32_t outerLoopSize;
};

std::vector<TestCase> ParseTestCaseCSV(const std::string& csvPath) {
    std::vector<TestCase> cases;
    std::ifstream file(csvPath);
    if (!file.is_open()) {
        throw std::runtime_error("无法打开CSV文件: " + csvPath);
    }

    std::string line;
    // 跳过表头
    std::getline(file, line);

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string scenario, MmType, blockNum, burstNum, burstLen, outerLoopSize;

        std::getline(ss, scenario, ',');
        std::getline(ss, MmType, ',');
        std::getline(ss, blockNum, ',');
        std::getline(ss, burstNum, ',');
        std::getline(ss, burstLen, ',');
        std::getline(ss, outerLoopSize, ',');

        TestCase tc{};
        tc.scenario = scenario;
        tc.MmType = static_cast<uint32_t>(std::stoi(MmType));
        tc.blockNum = static_cast<uint32_t>(std::stoi(blockNum));
        tc.burstNum = static_cast<uint32_t>(std::stoi(burstNum));
        tc.burstLen = static_cast<uint32_t>(std::stoi(burstLen));
        tc.outerLoopSize = static_cast<uint32_t>(std::stoi(outerLoopSize));
        cases.push_back(tc);
    }
    file.close();
    return cases;
}

#define CHECK_ACL(x)                                                                        \
    do {                                                                                    \
        aclError __ret = x;                                                                 \
        if (__ret != ACL_ERROR_NONE) {                                                      \
            std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << __ret << std::endl; \
        }                                                                                   \
    } while (0);

int32_t main(int32_t argc, char *argv[])
{
    // 命令行参数：仅接收CSV文件路径
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <test_case_csv_path>" << std::endl;
        return 1;
    }
    std::string csvPath = argv[1];

    CHECK_ACL(aclInit(nullptr));
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    std::vector<TestCase> testCases = ParseTestCaseCSV(csvPath);
    if (testCases.empty()) {
        throw std::runtime_error("CSV文件中无有效测试用例");
    }

    // ==============================================
    // 循环执行所有用例
    // ==============================================
    for (size_t i = 0; i < testCases.size(); ++i) {
        const auto& tc = testCases[i];
        std::cout << "\n----------------------------------------" << std::endl;
        std::cout << "执行第 " << i+1 << "/" << testCases.size() << " 个用例" << std::endl;
        std::cout << "参数: Scenario=" << tc.scenario << ", MmType=" << tc.MmType
                    << ", blockNum=" << tc.blockNum << ", burstNum=" << tc.burstNum << ", burstLen=" << tc.burstLen << std::endl;

        uint32_t tileSize = tc.burstNum * tc.burstLen;

        size_t allSize = tileSize * sizeof(int32_t) * tc.outerLoopSize * tc.blockNum;
        size_t timeSize = tc.outerLoopSize * sizeof(uint64_t) * tc.blockNum;
        // uint32_t numBlocks = 1;

        uint8_t *aDevice = nullptr;
        uint64_t *timeDevice = nullptr;
        uint64_t *timeHost = nullptr;
        CHECK_ACL(aclrtMalloc((void **)&aDevice, allSize, ACL_MEM_MALLOC_HUGE_FIRST));
        aclrtMalloc((void **)&timeDevice, timeSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMallocHost((void **)(&timeHost), timeSize);

        // 执行核心测试函数
        TestFuncPtr KernelFunc = GetTestFunction(tc.scenario);
        KernelFunc(tc.blockNum, stream, aDevice, timeDevice, tc.burstNum, tc.burstLen, tc.MmType, tc.outerLoopSize);
        CHECK_ACL(aclrtSynchronizeStream(stream));

        aclrtFree(aDevice);
        aclrtFree(timeDevice);
        aclrtFreeHost(timeHost);
        std::cout << "✅ 用例执行完成" << std::endl;
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}