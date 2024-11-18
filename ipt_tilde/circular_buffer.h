
#ifndef IPT_MAX_CIRCULAR_BUFFER_H
#define IPT_MAX_CIRCULAR_BUFFER_H

#include <vector>


class CustomWindowedBuffer {
public:
    CustomWindowedBuffer() : m_large_buffer(m_segment_length, 0.0f) {}

    void add_samples(const std::vector<float>& new_samples) {
        for (auto sample : new_samples) {
            m_large_buffer[m_write_index] = sample;
            m_write_index = (m_write_index + 1) % m_segment_length;
        }
    }

    std::vector<float> get_windowed_buffer() {
        std::vector<float> windowed_buffer(m_segment_length);
        size_t start_index = m_write_index;

        for (size_t i = 0; i < m_segment_length; ++i) {
            windowed_buffer[i] = m_large_buffer[(start_index + i) % m_segment_length];
        }

        return windowed_buffer;
    }

    std::size_t size() const {
        return m_segment_length;
    }

    void clear() {
        throw std::runtime_error("not implemented") // TODO
    }

private:
    const std::size_t m_segment_length = 7168; // TO DO: to be quired from the model
    std::vector<float> m_large_buffer;
    std::size_t m_write_index = 0;
};



#endif //IPT_MAX_CIRCULAR_BUFFER_H
