
#ifndef IPT_MAX_UTILITY_H
#define IPT_MAX_UTILITY_H

#include <vector>

namespace util {

static inline std::size_t argmax(const std::vector<float>& v) {
    float max = v[0];
    std::size_t index = 0;
    for (std::size_t i = 1; i < v.size(); i++) {
        if (v[i] > max) {
            max = v[i];
            index = i;
        }
    }
    return index;
}


// ==============================================================================================

static inline std::size_t mstosamples(double ms, double sr) {
    return static_cast<std::size_t>(std::ceil(sr * ms / 1000.0));
}


static inline std::size_t mstosamples(int ms, int sr) {
    return mstosamples(static_cast<double>(ms), static_cast<double>(sr));
}


// ==============================================================================================

template<typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
static inline T clamp(T x, T low, T high) {
    return std::max(low, std::min(x, high));
}


// ==============================================================================================

static inline std::vector<float> to_floats(const std::vector<double>& v) {
    std::vector<float> f;
    f.reserve(v.size());
    for (auto i : v) {
        f.push_back(static_cast<float>(i));
    }
    return f;
}


} // namespace util

#endif //IPT_MAX_UTILITY_H
