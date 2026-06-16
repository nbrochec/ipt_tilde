
#ifndef IPT_MAX_LEAKY_INTEGRATOR_H
#define IPT_MAX_LEAKY_INTEGRATOR_H

#include <chrono>
#include <vector>
#include <optional>

class LeakyIntegrator {
public:
    std::vector<float> process(const std::vector<float>& input) {
        return process(input, std::chrono::steady_clock::now());
    }

    // timestamp of when the input was produced, so batched results keep their real spacing
    std::vector<float> process(const std::vector<float>& input,
                               std::chrono::time_point<std::chrono::steady_clock> current_time) {
        if (!m_last_callback || m_tau < 1e-6 || m_previous_value.size() != input.size()) {
            m_last_callback = current_time;
            m_previous_value = input;
            return input;
        }

        auto elapsed_time = std::chrono::duration<double, std::milli>(
                current_time - *m_last_callback).count();

        elapsed_time = std::min(m_tau, std::max(0.0, elapsed_time));

        auto output = integrate(input, elapsed_time);
        m_previous_value = output;
        m_last_callback = current_time;

        return output;
    }

    void set_tau(double tau) {
        m_tau = tau;
    }


private:
    std::vector<float> integrate(const std::vector<float>& current_value, double elapsed_time) {
        std::vector<float> result;
        result.reserve(current_value.size());

        auto dt = elapsed_time / m_tau;

        for (std::size_t i = 0; i < current_value.size(); i++) {
            result.emplace_back((1 - dt) * m_previous_value.at(i) + dt * current_value.at(i));
        }
        return result;
    }


    std::optional<std::chrono::time_point<std::chrono::steady_clock>> m_last_callback;
    std::vector<float> m_previous_value;

    double m_tau = 0.0;


};


#endif //IPT_MAX_LEAKY_INTEGRATOR_H
