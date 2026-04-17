#pragma once
#include <cstdint>
#include <string>

class SoftmaxTilingMocker {
public:
    static SoftmaxTilingMocker& GetInstance();
    void SetSocVersion(const std::string& socVersion);
    const char* GetSocVersion() const;
    void Reset();

private:
    SoftmaxTilingMocker() = default;
    std::string socVersion_;
};
