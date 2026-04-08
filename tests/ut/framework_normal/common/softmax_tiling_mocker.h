#pragma once
#include <cstdint>
#include <string>

// Mocker 单例，UT 通过 SetSocVersion 指定 SoC，
// 使 GetSoftMaxMinTmpSize / GetSoftMaxFlashV2MinTmpSize 调用
// PlatformAscendCManager::GetInstance(socVersion) 而非无参版本。
class SoftmaxTilingMocker {
public:
    static SoftmaxTilingMocker& GetInstance() {
        static SoftmaxTilingMocker instance;
        return instance;
    }

    void SetSocVersion(const std::string& socVersion) { socVersion_ = socVersion; }
    const char* GetSocVersion() const { return socVersion_.empty() ? nullptr : socVersion_.c_str(); }
    void Reset() { socVersion_.clear(); }
private:
    SoftmaxTilingMocker() = default;
    std::string socVersion_;
};
