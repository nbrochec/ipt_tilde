
#ifndef IPT_MAX_CIRCULAR_BUFFER_H
#define IPT_MAX_CIRCULAR_BUFFER_H

#include <vector>
#include <CDSPResampler.h>
#include "utility.h"


template<typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
class CircularBuffer {
public:
    explicit CircularBuffer(std::size_t size) : m_buffer(size, 0.0f) {}

    CircularBuffer(int ms, int sr) : CircularBuffer(util::mstosamples(ms, sr)) {}


    void add_samples(std::vector<double>&& new_samples) {
        add_samples(std::move(new_samples.data()), new_samples.size());
    }


    void add_samples(const std::vector<T>& new_samples) {
        add_samples(new_samples.data(), new_samples.size());
    }


    void add_samples(const T* samples, std::size_t num_samples) {
        if (!m_fully_allocated && m_write_index + num_samples > m_buffer.size()) {
            m_fully_allocated = true;
        }

        for (int i = 0; i < num_samples; ++i) {
            m_buffer[m_write_index] = samples[i];
            m_write_index = (m_write_index + 1) % m_buffer.size();
        }
    }


    std::vector<T> get_samples() const {
        std::size_t num_samples = size();
        std::vector<T> reordered_samples;
        reordered_samples.reserve(num_samples);

        size_t start_index = m_write_index;

        for (size_t i = 0; i < num_samples; ++i) {
            reordered_samples.push_back(m_buffer[(start_index + i) % num_samples]);
        }

        return reordered_samples;
    }


    const std::vector<T>& samples_unordered() const {
        return m_buffer;
    }


    void resize(std::size_t new_size) {
        if (new_size > m_buffer.size()) {
            m_fully_allocated = false;
        }

        if (m_write_index >= new_size) {
            m_write_index = 0;
        }

        m_buffer.resize(new_size);
    }


    void resize(int ms, int sr) {
        m_buffer.resize(util::mstosamples(ms, sr));
    }


    std::size_t size() const {
        return m_buffer.size();
    }


    /** Returns true if the buffer is fully allocated, i.e. every sample has been written at least once. */
    bool is_fully_allocated() const {
        return m_fully_allocated;
    }


private:
    std::vector<T> m_buffer;
    std::size_t m_write_index = 0;
    bool m_fully_allocated = false;
};


// ==============================================================================================

class ResamplingBuffer {
public:
    ResamplingBuffer(std::size_t buffer_size, std::size_t input_vector_size, int input_sr, int output_sr)
            : m_resampler(static_cast<double>(input_sr)
                          , static_cast<double>(output_sr)
                          , static_cast<int>(input_vector_size))
              , m_buffer(buffer_size) {}


    void add_samples(std::vector<double>&& new_samples) {
        double* output_pointer; // Allocated and freed by the resampler
        int num_samples = m_resampler.process(new_samples.data(), static_cast<int>(new_samples.size()), output_pointer);
        m_buffer.add_samples(output_pointer, static_cast<std::size_t>(num_samples));
    }


    std::vector<double> get_samples() const {
        return m_buffer.get_samples();
    }


    bool is_fully_allocated() const {
        return m_buffer.is_fully_allocated();
    }


private:
    CircularBuffer<double> m_buffer;
    r8b::CDSPResampler m_resampler;


};


#endif //IPT_MAX_CIRCULAR_BUFFER_H
