#pragma once

#include <cstdio>
#include <string>
#include <utility>

#if defined(__has_include)
#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define DISTQR_HAS_NVTX 1
#elif __has_include(<nvToolsExt.h>)
#include <nvToolsExt.h>
#define DISTQR_HAS_NVTX 1
#else
#define DISTQR_HAS_NVTX 0
#endif
#else
#define DISTQR_HAS_NVTX 0
#endif

namespace distqr::nvtx {

inline constexpr bool kEnabled = DISTQR_HAS_NVTX != 0;

class ScopedRange {
public:
    ScopedRange() = default;

    explicit ScopedRange(const char* label) {
        Start(label);
    }

    explicit ScopedRange(std::string label) {
        Start(std::move(label));
    }

    ~ScopedRange() {
        Finish();
    }

    ScopedRange(const ScopedRange&) = delete;
    ScopedRange& operator=(const ScopedRange&) = delete;

    ScopedRange(ScopedRange&& other) noexcept
        : label_(std::move(other.label_)), active_(other.active_) {
        other.active_ = false;
    }

    ScopedRange& operator=(ScopedRange&& other) noexcept {
        if (this != &other) {
            Finish();
            label_ = std::move(other.label_);
            active_ = other.active_;
            other.active_ = false;
        }
        return *this;
    }

private:
    void Start(const char* label) {
#if DISTQR_HAS_NVTX
        nvtxRangePushA(label ? label : "");
        active_ = true;
#else
        (void)label;
#endif
    }

    void Start(std::string label) {
#if DISTQR_HAS_NVTX
        label_ = std::move(label);
        nvtxRangePushA(label_.c_str());
        active_ = true;
#else
        (void)label;
#endif
    }

    void Finish() {
#if DISTQR_HAS_NVTX
        if (active_) {
            nvtxRangePop();
        }
#endif
        active_ = false;
    }

    std::string label_;
    bool active_ = false;
};

template <typename... Args>
inline std::string Format(const char* fmt, Args... args) {
    const int size = std::snprintf(nullptr, 0, fmt, args...);
    if (size <= 0) {
        return std::string(fmt ? fmt : "");
    }

    std::string buffer(static_cast<size_t>(size) + 1, '\0');
    std::snprintf(buffer.data(), buffer.size(), fmt, args...);
    buffer.resize(static_cast<size_t>(size));
    return buffer;
}

inline ScopedRange MakeScopedRange(const char* label) {
    return ScopedRange(label);
}

template <typename... Args>
inline ScopedRange MakeScopedRangef(const char* fmt, Args... args) {
#if DISTQR_HAS_NVTX
    return ScopedRange(Format(fmt, args...));
#else
    (void)fmt;
    return ScopedRange();
#endif
}

}  // namespace distqr::nvtx
