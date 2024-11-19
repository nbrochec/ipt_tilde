
#ifndef IPT_MAX_ENERGY_THRESHOLD_H
#define IPT_MAX_ENERGY_THRESHOLD_H

#include "circular_buffer.h"

class EnergyThreshold {
public:
    static constexpr double MINIMUM_THRESHOLD = -80.0;


    explicit EnergyThreshold(double threshold_db) : m_threshold_db(threshold_db) {}


    /** Compute the energy over a single vector, without modifying the internal buffer */
    bool is_above_threshold(const std::vector<double>& v) const {
        if (m_threshold_db <= MINIMUM_THRESHOLD) {
            return true;
        }

        return atodb(rms(v)) >= m_threshold_db;
    }


    void set_threshold_db(double threshold_db) {
        m_threshold_db = threshold_db;
    }


    static double atodb(double a) {
        if (a <= 0) return -120.0;
        return 20.0 * std::log10(a);
    }


    static double dbtoa(double a) {
        if (a <= -120.0) return 0.0;
        return std::pow(10.0, a / 20.0);
    }

    static double rms(const std::vector<double>& v) {
        double sum = 0.0;
        for (auto& x: v) sum += x * x;
        return std::sqrt(sum / static_cast<double>(v.size()));
    }


private:
    double m_threshold_db;
};


#endif //IPT_MAX_ENERGY_THRESHOLD_H
