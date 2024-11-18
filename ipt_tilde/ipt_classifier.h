
#ifndef IPT_MAX_IPT_CLASSIFIER_H
#define IPT_MAX_IPT_CLASSIFIER_H

#include <torch/script.h>
#include <torch/torch.h>
#include <CDSPResampler.h>
#include <chrono>
#include "circular_buffer.h"

struct ClassificationResult {
    std::vector<float> distribution;
    std::size_t argmax;
    double inference_latency_ms;
};

class IptClassifier {
public:
    explicit IptClassifier(const std::string& path, torch::DeviceType device)
    : m_device(device), m_buffer() {
        at::init_num_threads();
    }

    void read_model(const std::string& path) {
        std::lock_guard lock{m_mutex};
        m_model = torch::jit::load(path);
        m_model.eval();
        m_model.to(m_device);
    }



    /*
    std::optional<ClassificationResult> process(std::vector<double>&& input) {
        std::optional<ClassificationResult> result = std::nullopt;
        for (auto& sample: input) {
            m_buffer[m_write_index] = static_cast<float>(sample);
            m_write_index = (m_write_index + 1) % m_size;

            if (m_write_index == 0) {
                result = analyze_buffer();
            }
        }
        return result;
    }
    */

    std::optional<ClassificationResult> process(std::vector<double>&& input) {
        // Note: using a mutex here is completely safe, as this is never called from the audio thread
        std::lock_guard lock{m_mutex};

        std::optional<ClassificationResult> result = std::nullopt;

        m_buffer.add_samples(std::vector<float>(input.begin(), input.end()));

        if (m_buffer.size() >= m_segment_length) {
            auto windowed_buffer = m_buffer.get_windowed_buffer();
            result = analyze_buffer(windowed_buffer);
        }

        return result;
    }


/*
private:
    template<typename T>
    static std::vector<T> tensor2vector(const at::Tensor& tensor) {
        std::vector<float> v;
        v.reserve(tensor.numel());
        auto out_ptr = tensor.contiguous().data_ptr<float>();

        std::copy(out_ptr, out_ptr + tensor.numel(), std::back_inserter(v));

        return v;
    }

    static at::Tensor vector2tensor(std::vector<float>& v) {
        return torch::from_blob(v.data(), {1, 1, static_cast<long long>(v.size())}, torch::kFloat32);
    }

    torch::DeviceType m_device;

    std::size_t m_size = 7168;

    std::vector<float> m_buffer;
    std::size_t m_write_index = 0;

    torch::jit::Module m_model;

};
*/

    void set_energy_threshold(float threshold) { m_energy_threshold = threshold; }

    void reinitialize(int sr) {
        std::lock_guard lock{m_mutex};
        m_input_sr = sr;
        m_buffer.clear();
    }


private:
    ClassificationResult analyze_buffer(const std::vector<float>& windowed_buffer) {
        auto tensor_in = vector2tensor(windowed_buffer);
        tensor_in  = tensor_in.to(m_device); // TODO: Might need a critical section here
        std::vector<torch::jit::IValue> inputs = {tensor_in};
        auto t1 = std::chrono::high_resolution_clock::now();
        auto tensor_out = m_model.get_method("forward")(inputs).toTensor();
        auto t2 = std::chrono::high_resolution_clock::now();
        tensor_out = tensor_out.to(torch::kCPU);

        auto v = tensor2vector<float>(tensor_out);
        auto amax = argmax(v);
        auto latency_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();

        return ClassificationResult{v, amax, static_cast<double>(latency_ns) / 1e6};
    }

    static std::size_t argmax(const std::vector<float>& v) {
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


    template<typename T>
    static std::vector<T> tensor2vector(const at::Tensor& tensor) {
        std::vector<float> v(tensor.numel());
        auto out_ptr = tensor.contiguous().data_ptr<float>();
        std::copy(out_ptr, out_ptr + tensor.numel(), v.begin());
        return v;
    }

    static at::Tensor vector2tensor(const std::vector<float>& v) {
        return torch::from_blob(const_cast<float*>(v.data()), {1, 1, static_cast<long long>(v.size())}, torch::kFloat32);
    }


    static int parse_sample_rate(torch::jit::Module& model) {
        throw std::runtime_error("Not implemented");  // TODO
    }

    static int parse_segment_length(torch::jit::Module& model) {
        throw std::runtime_error("Not implemented"); // TODO
    }

    torch::DeviceType m_device;
    torch::jit::Module m_model;
    CustomWindowedBuffer m_buffer;
    std::size_t m_segment_length = 7168;

    std::atomic<float> m_energy_threshold;

    std::optional<int> m_input_sr;
    int m_model_sr;

    std::mutex m_mutex;
};

#endif //IPT_MAX_IPT_CLASSIFIER_H
