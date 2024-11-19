
#ifndef IPT_MAX_IPT_CLASSIFIER_H
#define IPT_MAX_IPT_CLASSIFIER_H

#include <torch/script.h>
#include <torch/torch.h>
#include <chrono>
#include "circular_buffer.h"
#include "utility.h"
#include "model.h"
#include "energy_threshold.h"


class IptClassifier {
public:
    static const inline std::string CLASSIFY_METHOD = "forward";
    static const int DEFAULT_THRESHOLD_WINDOW_MS = 20;

    explicit IptClassifier(std::string path
                           , torch::DeviceType device
                           , double energy_threshold_db = EnergyThreshold::MINIMUM_THRESHOLD
                           , int threshold_window_ms = DEFAULT_THRESHOLD_WINDOW_MS)
            : m_model_path(std::move(path))
            , m_device(device)
            , m_energy_threshold(energy_threshold_db)
            , m_threshold_window_ms(threshold_window_ms) {}

    /**
     * @note Make sure to call this on the thread that will call `process()`
     * @throws c10::Error if model cannot be loaded
     * */
    void initialize_model() {
        std::lock_guard lock{m_mutex};
        m_model = std::make_unique<Model>(m_model_path, m_device);

        m_initialized = is_initialized();
    }


    /** @note: should typically be called when dsp is started / restarted */
    void initialize_buffers(int sr, int input_vector_length) {
        assert(m_model);

        std::lock_guard lock{m_mutex};
        m_input_sr = sr;
        m_threshold_buffer = std::make_unique<CircularBuffer<double>>(m_threshold_window_ms, sr);

        m_classification_buffer = std::make_unique<ResamplingBuffer>(m_model->get_segment_length()
                                                                     , input_vector_length
                                                                     , sr
                                                                     , m_model->get_sample_rate());

        m_initialized = is_initialized();
    }


    /** @throws c10::Error if classification fails */
    std::optional<ClassificationResult> process(std::vector<double>&& input) {
        // Note: using a mutex here is completely safe, as this is never called from the audio thread
        std::lock_guard lock{m_mutex};

        if (!m_initialized) {
            return std::nullopt;
        }

        m_threshold_buffer->add_samples(input);
        m_classification_buffer->add_samples(std::move(input));

        if (!m_classification_buffer->is_fully_allocated()) {
            return std::nullopt;
        }


        if (m_active) {
            auto samples = m_classification_buffer->get_samples();

            if (m_energy_threshold.is_above_threshold(samples)) {
                std::cout << "active classification\n";
                return m_model->classify(util::to_floats(samples));
            } else {
                m_active = false;
            }
        } else {
            if (m_energy_threshold.is_above_threshold(m_threshold_buffer->samples_unordered())) {
                m_active = true;

                std::cout << "first classification\n";
                auto samples = m_classification_buffer->get_samples();
                return m_model->classify(util::to_floats(samples));
            }
        }

        /* Note: The conditions for activation and deactivation are different:
         *   - activation: one energy threshold window is above silence,
         *   - deactivation: one entire inference window is below threshold
         *   we might therefore have some edge cases where an entire window is below threshold but still classified.
         *   This is for the moment intentional by design, but might after experimenting need a rework at a later stage
         */

        return std::nullopt;
    }


    void set_energy_threshold(double threshold_db) {
        std::lock_guard lock{m_mutex};
        m_energy_threshold.set_threshold_db(threshold_db);
    }

    void set_threshold_window(int duration_ms) {
        std::lock_guard lock{m_mutex};

        duration_ms = std::max(0, duration_ms);
        m_threshold_window_ms = duration_ms;

        if (m_initialized) {
            m_threshold_buffer->resize(duration_ms, *m_input_sr);
        }
    }

    std::optional<std::vector<std::string>> get_class_names() const {
        if (m_model) {
            return m_model->get_class_names();
        }
        return std::nullopt;
    }


private:

    /** @note: Defines invariant for class */
    bool is_initialized() const {
        return m_model && m_classification_buffer && m_threshold_buffer && m_input_sr;
    }

    // Initialization parameters
    std::string m_model_path;
    torch::DeviceType m_device;
    int m_threshold_window_ms;
    std::optional<int> m_sr;

    EnergyThreshold m_energy_threshold;

    bool m_initialized = false;

    std::unique_ptr<Model> m_model;
    std::unique_ptr<ResamplingBuffer> m_classification_buffer;
    std::unique_ptr<CircularBuffer<double>> m_threshold_buffer;

    std::optional<int> m_input_sr;

    bool m_active = false;

    std::mutex m_mutex;
};

#endif //IPT_MAX_IPT_CLASSIFIER_H
