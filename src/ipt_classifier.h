
#ifndef IPT_MAX_IPT_CLASSIFIER_H
#define IPT_MAX_IPT_CLASSIFIER_H

#include <torch/script.h>
#include <torch/torch.h>
#include <chrono>
#include "circular_buffer.h"
#include "utility.h"
#include "classifier.h"
#include "energy_threshold.h"


class IptClassifier {
public:
    static const inline std::string CLASSIFY_METHOD = "forward";

    explicit IptClassifier(std::string path
                           , torch::DeviceType device
                           , double energy_threshold_db
                           , int threshold_window_ms)
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
        m_classifier = std::make_unique<Classifier>(m_model_path, m_device);

        m_initialized = is_initialized();
    }


    /** @note: should typically be called when dsp is started / restarted */
    void initialize_buffers(int sr, int input_vector_length) {
        assert(m_classifier);

        std::lock_guard lock{m_mutex};
        m_input_sr = sr;
        m_threshold_buffer = std::make_unique<CircularBuffer<double>>(m_threshold_window_ms, sr);

        auto& model = m_classifier->get_model();
        m_classification_buffer = std::make_unique<ResamplingBuffer>(model.get_segment_length()
                                                                     , input_vector_length
                                                                     , sr
                                                                     , model.get_sample_rate());

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
                return m_classifier->classify(util::to_floats(samples));
            } else {
                m_active = false;
            }
        } else {
            if (m_energy_threshold.is_above_threshold(m_threshold_buffer->samples_unordered())) {
                m_active = true;

                auto samples = m_classification_buffer->get_samples();
                return m_classifier->classify(util::to_floats(samples));
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


    void set_energy_threshold(float threshold_db) {
        std::lock_guard lock{m_mutex};
        m_energy_threshold.set_threshold_db(threshold_db);
    }

    void set_threshold_window(int duration_ms) {
        std::lock_guard lock{m_mutex};

        m_threshold_window_ms = duration_ms;

        if (m_initialized) {
            m_threshold_buffer->resize(duration_ms, *m_input_sr);
        }
    }


private:

    /** @note: Defines invariant for class */
    bool is_initialized() const {
        return m_classifier && m_classification_buffer && m_threshold_buffer && m_input_sr;
    }

    // Initialization parameters
    std::string m_model_path;
    torch::DeviceType m_device;
    int m_threshold_window_ms;
    std::optional<int> m_sr;

    EnergyThreshold m_energy_threshold;

    bool m_initialized = false;

    std::unique_ptr<Classifier> m_classifier;
    std::unique_ptr<ResamplingBuffer> m_classification_buffer;
    std::unique_ptr<CircularBuffer<double>> m_threshold_buffer;

    std::optional<int> m_input_sr;

    bool m_active = false;

    std::mutex m_mutex;
};

#endif //IPT_MAX_IPT_CLASSIFIER_H
